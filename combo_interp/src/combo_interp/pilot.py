"""Agent-free pilot: cheap per-unit statistics straight from the cache, aggregated by condition.

Metrics (per unit, primary pole):
  kurtosis, frac_gt3         activation-shape statistics (confounds: combos of sparse neurons get Gaussian-ish)
  tok_entropy                normalised entropy of the firing-token identity over the 100 strongest firings
                             (low = fires on a few specific tokens)
  top_tok_share              share of the most common firing token
  rank100_ratio              value of the 100th (deduped) strongest firing / strongest  (detector vs dial)
  emb_coherence (optional)   mean pairwise cosine of sentence embeddings of the top contexts minus the same for
                             random contexts (needs sentence-transformers + tokenizer)
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .corpus import CorpusStore
from .export import Decoder, dedupe_hits, moments_stats
from .units import load_unit_set


def _norm_entropy(counter: Counter):
    n = sum(counter.values())
    if n < 5:
        return float("nan"), float("nan")
    p = np.array(list(counter.values()), dtype=float) / n
    H = float(-(p * np.log(p)).sum() / np.log(n))
    return H, float(p.max())


def pilot_metrics(cfg, paths, tok=None, embed=False, log=print) -> pd.DataFrame:
    store = CorpusStore(paths.corpus("train"))
    Lq = store.seq_len
    emb_model = None
    if embed:
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        dec = Decoder(tok)
    rng = np.random.default_rng(cfg.seed + 11)
    rows = []
    for L in [int(l) for l in cfg.model.layers]:
        us = load_unit_set(paths, L)
        d = paths.cache("train", L)
        big, mom, uni = np.load(d / "topk_big.npz"), np.load(d / "moments.npz"), np.load(d / "uniform.npz")
        big = {k: big[k] for k in big.files}
        mom = {k: mom[k] for k in mom.files}
        ugid = uni["gid"]
        mean_u, std_u = us["mean_u"], np.maximum(us["std_u"], 1e-12)
        for col, m in enumerate(us["meta"]):
            zpos = (big["pos_vals"][col, 0] - mean_u[col]) / std_u[col]
            zneg = (mean_u[col] - big["neg_vals"][col, 0]) / std_u[col]
            pole, sgn = ("pos", 1.0) if zpos >= zneg else ("neg", -1.0)
            sv = sgn * big[f"{pole}_vals"][col]
            if not np.isfinite(sv[0]) or sv[0] <= 0:
                continue
            hits = dedupe_hits(sv, big[f"{pole}_gid"][col], Lq, 1, limit=100)
            toks = Counter()
            for _, _, g in hits:
                t = store.get_tokens(g // Lq, g % Lq, g % Lq + 1)
                if len(t):
                    toks[int(t[0])] += 1
            H, top1 = _norm_entropy(toks)
            ms = moments_stats(mom, col)
            row = {"unit_id": m["unit_id"], "layer": L, "cond": m["cond"], "kind": m["kind"], "k": m["k"],
                   "kurtosis": ms["kurtosis"], "frac_gt3": ms["frac_gt3"], "tok_entropy": H, "top_tok_share": top1,
                   "rank100_ratio": float(hits[-1][1] / hits[0][1]) if len(hits) >= 20 else float("nan"),
                   "n_hits": len(hits)}
            if emb_model is not None and len(hits) >= 20:
                def texts(gs):
                    out = []
                    for g in gs:
                        seq, pos = divmod(int(g), Lq)
                        ids = store.get_tokens(seq, pos - 24, pos + 9)
                        out.append("".join(dec.many(ids)))
                    return out
                top = texts([g for _, _, g in hits[:50]])
                rnd = texts(rng.choice(ugid, size=min(50, len(ugid)), replace=False))
                def mean_cos(tx):
                    e = emb_model.encode(tx, normalize_embeddings=True)
                    c = e @ e.T
                    return float((c.sum() - len(tx)) / (len(tx) * (len(tx) - 1)))
                row["emb_coherence"] = mean_cos(top) - mean_cos(rnd)
            rows.append(row)
        log(f"[pilot] layer {L}: {len(rows)} units")
    return pd.DataFrame(rows)


def summarize_pilot(df: pd.DataFrame, n_boot=1000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = [c for c in ("kurtosis", "frac_gt3", "tok_entropy", "top_tok_share", "rank100_ratio", "emb_coherence")
               if c in df.columns]
    out = []
    for (cond, layer), g in df.groupby(["cond", "layer"], sort=False):
        for mname in metrics:
            x = g[mname].dropna().to_numpy()
            if len(x) < 3:
                continue
            b = np.array([np.median(rng.choice(x, len(x))) for _ in range(n_boot)])
            out.append({"cond": cond, "layer": layer, "metric": mname, "n": len(x), "median": float(np.median(x)),
                        "lo": float(np.percentile(b, 5)), "hi": float(np.percentile(b, 95))})
    return pd.DataFrame(out)


def plot_pilot(df: pd.DataFrame, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    metrics = [c for c in ("tok_entropy", "top_tok_share", "rank100_ratio", "kurtosis", "emb_coherence") if c in df.columns]
    conds = list(dict.fromkeys(df["cond"]))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4), squeeze=False)
    for ax, mname in zip(axes[0], metrics):
        data = [df[df.cond == c][mname].dropna().to_numpy() for c in conds]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(conds) + 1), conds)       # boxplot(labels=) was removed in matplotlib 3.11
        ax.set_title(mname); ax.tick_params(axis="x", rotation=70)
        if mname == "kurtosis":
            ax.set_yscale("symlog")
    fig.tight_layout()
    fig.savefig(outdir / "pilot_metrics.png", dpi=130)
    plt.close(fig)
