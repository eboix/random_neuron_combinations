"""Probe-job construction (CPU) and per-unit score computation (CPU).

Metrics per unit (primary pole; r = signed activation along the pole / the unit's corpus maximum on that pole):
  rate_pred / rate_near / rate_distant   fraction of the agent's own probe texts with r >= strong_frac
  rate_null                              same, for OTHER units' predicted-fire texts (chance baseline)
  rate_rand                              same, for random document prefixes (base rate on ordinary text)
  margin                                 rate_pred - rate_near   (does the explanation discriminate?)
  blind_max_ratio / _logratio            best r among 3 texts written by an agent that saw ONLY the explanation
  sim_spearman                           rank correlation between a blind simulator's 0-10 predictions and the true
                                         activation on held-out contexts (explanation-only)
  n_parts, confidence, expl_words        compactness / self-reported confidence
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .io_utils import read_json, read_jsonl, write_jsonl


# ------------------------------------------------------------------ probe jobs
def build_probe_job(cfg, paths, round_id: int, tok=None, log=print):
    key = read_json(paths.blind_key)
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), 777, int(round_id)]))
    ref = read_json(paths.unit_ref)
    rows = []
    if round_id == 1:
        from .corpus import CorpusStore
        store = CorpusStore(paths.corpus("train"))
        expl = {p.stem: read_json(p) for p in sorted(paths.agent_results("explain").glob("u*.json"))}
        by_layer = defaultdict(list)
        for b, e in expl.items():
            by_layer[ref[b]["layer"]].append(b)
        for b, e in expl.items():
            uid = key[b]
            for kind, field in (("pred", "predicted_fire"), ("near", "near_miss"), ("distant", "distant")):
                for i, t in enumerate(e[field]):
                    rows.append({"probe_id": f"{b}|{kind}|{i}", "unit_id": uid, "kind": kind, "text": t})
            pool = [o for o in by_layer[ref[b]["layer"]] if o != b]
            if pool:
                for i in range(int(cfg.scoring.n_null)):
                    o = pool[rng.integers(len(pool))]
                    t = expl[o]["predicted_fire"][rng.integers(5)]
                    rows.append({"probe_id": f"{b}|null|{i}", "unit_id": uid, "kind": "null", "text": t})
            for i in range(int(cfg.scoring.n_rand)):
                for _ in range(20):
                    seq = int(rng.integers(store.n_seqs))
                    if store.seq_length(seq) >= int(cfg.scoring.rand_prefix_tokens):
                        break
                ids = store.get_tokens(seq, 0, int(cfg.scoring.rand_prefix_tokens))
                rows.append({"probe_id": f"{b}|rand|{i}", "unit_id": uid, "kind": "rand", "text": tok.decode(ids.tolist())})
    elif round_id == 2:
        for p in sorted(paths.agent_results("blindmax").glob("u*.json")):
            b = p.stem
            for i, t in enumerate(read_json(p)["texts"]):
                rows.append({"probe_id": f"{b}|blindmax|{i}", "unit_id": key[b], "kind": "blindmax", "text": t})
    else:
        raise ValueError(round_id)
    out = paths.data / "probe" / f"round{round_id}_in.jsonl"
    write_jsonl(out, rows)
    log(f"[probe-job] round {round_id}: {len(rows)} texts for {len({r['unit_id'] for r in rows})} units -> {out}")
    return out


# ------------------------------------------------------------------ scores
def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def threshold_label(cfg) -> str:
    """Subfolder name for scores/results, so several threshold definitions can be compared side by side."""
    S = cfg.scoring
    mode = S.get("threshold", "corpus_frac")
    if mode == "corpus_frac":
        return f"corpus_frac_{float(S.strong_frac):g}"
    if mode == "random_quantile":
        return f"random_q{float(S.random_quantile):g}"
    raise ValueError(f"unknown scoring.threshold {mode!r}")


def _hit_fn(cfg, paths, ref):
    """-> hit(blind_id, signed_act, n_tokens) -> bool | nan: does a probe text count as 'fires strongly'?"""
    S = cfg.scoring
    if S.get("threshold", "corpus_frac") == "corpus_frac":
        strong = float(S.strong_frac)
        return lambda b, a, n: a / ref[b]["ref"] >= strong
    from .calibrate import load_calibration
    from .units import load_unit_set
    cal = {int(L): load_calibration(paths, int(L)) for L in cfg.model.layers}
    col = {m["unit_id"]: m["col"] for L in cfg.model.layers for m in load_unit_set(paths, int(L))["meta"]}
    q = float(S.random_quantile)
    qi = {L: int(np.argmin(np.abs(c["quantiles"] - q))) for L, c in cal.items()}
    for L, c in cal.items():
        if abs(c["quantiles"][qi[L]] - q) > 1e-9:
            raise ValueError(f"scoring.random_quantile={q} not calibrated (have {list(c['quantiles'])}); rerun 07b_calibrate.py")

    def hit(b, a, n):
        R = ref[b]
        L = int(R["layer"])
        thr = cal[L]["thr_pos" if R["primary"] > 0 else "thr_neg"]
        t = thr[qi[L], min(int(n), thr.shape[1]) - 1, col[R["unit_id"]]]
        return a >= t if np.isfinite(t) else np.nan
    return hit


def compute_scores(cfg, paths, probe_files, log=print) -> pd.DataFrame:
    ref = read_json(paths.unit_ref)
    truth = read_json(paths.holdout_truth)
    hit = _hit_fn(cfg, paths, ref)
    by_unit = defaultdict(lambda: defaultdict(list))
    hits = defaultdict(lambda: defaultdict(list))
    for f in probe_files:
        for r in read_jsonl(f):
            b, kind, _ = r["probe_id"].split("|")
            R = ref[b]
            a = r["max_act"] if R["primary"] > 0 else (None if r["min_act"] is None else -r["min_act"])
            by_unit[b][kind].append(np.nan if a is None else a / R["ref"])
            hits[b][kind].append(np.nan if a is None else float(hit(b, a, r["n_tokens"])))
    rows = []
    for p in sorted(paths.agent_results("explain").glob("u*.json")):
        b = p.stem
        ex = read_json(p)
        R = ref[b]
        row = {"blind_id": b, **{k: R[k] for k in ("unit_id", "layer", "cond", "kind", "k", "kurtosis", "frac_gt3")},
               "primary": R["primary"], "n_parts": len(ex["parts"]), "confidence": ex["confidence"],
               "expl_words": len(ex["explanation"].split())}
        for kind in ("pred", "near", "distant", "null", "rand"):
            v = np.array(by_unit[b].get(kind, []), float)
            v = v[~np.isnan(v)]
            h = np.array(hits[b].get(kind, []), float)
            h = h[~np.isnan(h)]
            row[f"rate_{kind}"] = float(h.mean()) if len(h) else np.nan
            row[f"meanr_{kind}"] = float(v.mean()) if len(v) else np.nan
        row["margin"] = row["rate_pred"] - row["rate_near"]
        bm = np.array(by_unit[b].get("blindmax", []), float)
        bm = bm[~np.isnan(bm)]
        row["blind_max_ratio"] = float(bm.max()) if len(bm) else np.nan
        row["blind_max_logratio"] = float(np.log(max(bm.max(), 1e-3))) if len(bm) else np.nan
        sp = paths.agent_results("simulate") / f"{b}.json"
        if sp.exists() and b in truth:
            sc = read_json(sp)["scores"]
            row["sim_spearman"] = _spearman([sc[str(i)] for i in range(len(truth[b]))], truth[b])
        else:
            row["sim_spearman"] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df["threshold"] = threshold_label(cfg)
    out = paths.scores / threshold_label(cfg)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "unit_scores.csv", index=False)
    log(f"[scores] {len(df)} units -> {out / 'unit_scores.csv'}")
    return df
