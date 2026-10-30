"""Probe-job construction (CPU) and per-unit score computation (CPU).

Metrics per unit (primary pole; r = signed activation along the pole / the unit's corpus maximum on that pole):
  rate_pred / rate_near / rate_distant   fraction of the writer's texts that "fire strongly" (scoring.threshold);
                                         the writer saw ONLY the explanation
  rate_null                              same, for OTHER units' predicted-fire texts (chance baseline)
  rate_rand                              same, for random document prefixes (base rate on ordinary text)
  margin                                 rate_pred - rate_near   (does the explanation discriminate?)
  blind_max_ratio / _logratio            best r among the writer's 3 max_texts
  sim_spearman                           rank correlation between a blind simulator's 0-10 predictions and the true
                                         activation on held-out contexts (explanation-only); with item groups also
                                         sim_spearman_top / _random, sim_auc, sim_rating_top / _random
  n_queries                              model queries the explainer used (agents.query)
  n_parts, confidence, expl_words        compactness / self-reported confidence
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .io_utils import read_json, read_jsonl, write_jsonl


# ------------------------------------------------------------------ probe job
def build_probe_job(cfg, paths, tok, log=print):
    """One GPU round: the writer's texts (max / predicted / near-miss / distant) for every unit with a write result,
    plus baselines: other units' predicted-fire texts (null) and random corpus-document prefixes (rand)."""
    from .corpus import CorpusStore
    key = read_json(paths.blind_key)
    ref = read_json(paths.unit_ref)
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), 777]))
    store = CorpusStore(paths.corpus("train"))
    wr = {p.stem: read_json(p) for p in sorted(paths.agent_results("write").glob("u*.json"))}
    by_layer = defaultdict(list)
    for b in wr:
        by_layer[ref[b]["layer"]].append(b)
    rows = []
    for b, w in wr.items():
        uid = key[b]
        for kind, field in (("blindmax", "max_texts"), ("pred", "predicted_fire"), ("near", "near_miss"),
                            ("distant", "distant")):
            for i, t in enumerate(w[field]):
                rows.append({"probe_id": f"{b}|{kind}|{i}", "unit_id": uid, "kind": kind, "text": t})
        pool = [o for o in by_layer[ref[b]["layer"]] if o != b]
        if pool:
            for i in range(int(cfg.scoring.n_null)):
                o = pool[rng.integers(len(pool))]
                t = wr[o]["predicted_fire"][rng.integers(5)]
                rows.append({"probe_id": f"{b}|null|{i}", "unit_id": uid, "kind": "null", "text": t})
        for i in range(int(cfg.scoring.n_rand)):
            for _ in range(20):
                seq = int(rng.integers(store.n_seqs))
                if store.seq_length(seq) >= int(cfg.scoring.rand_prefix_tokens):
                    break
            ids = store.get_tokens(seq, 0, int(cfg.scoring.rand_prefix_tokens))
            rows.append({"probe_id": f"{b}|rand|{i}", "unit_id": uid, "kind": "rand", "text": tok.decode(ids.tolist())})
    out = paths.probe / "probe_in.jsonl"
    write_jsonl(out, rows)
    log(f"[probe-job] {len(rows)} texts for {len(wr)} units -> {out}")
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


def simulation_metrics(pred, true, groups=None) -> dict:
    """Simulator ratings vs true activations. With item groups ('top' = the unit's strongest held-out firings,
    'random' = typical tokens) also: rank correlation within each group, how well top items are rated above random
    ones (AUC, ties count 1/2), and the mean 0-10 rating per group (random = false alarms)."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    out = {"sim_spearman": _spearman(pred, true)}
    if groups is None or len(groups) != len(pred):
        return out
    g = np.asarray(groups)
    top, rnd = g == "top", g == "random"
    out["sim_spearman_top"] = _spearman(pred[top], true[top])
    out["sim_spearman_random"] = _spearman(pred[rnd], true[rnd])
    if top.any() and rnd.any():
        d = pred[top][:, None] - pred[rnd][None, :]
        out["sim_auc"] = float((d > 0).mean() + 0.5 * (d == 0).mean())
    out["sim_rating_top"] = float(pred[top].mean()) if top.any() else np.nan
    out["sim_rating_random"] = float(pred[rnd].mean()) if rnd.any() else np.nan
    return out


def _query_counts(paths) -> dict:
    """{blind_id: number of texts the explainer ran through the query service}."""
    if not paths.query_log.exists():
        return {}
    return {f.stem: sum(1 for _ in open(f)) for f in paths.query_log.glob("u*.jsonl")}


def compute_scores(cfg, paths, probe_files, log=print) -> pd.DataFrame:
    ref = read_json(paths.unit_ref)
    truth = read_json(paths.holdout_truth)
    hit = _hit_fn(cfg, paths, ref)
    by_unit = defaultdict(lambda: defaultdict(list))
    hits = defaultdict(lambda: defaultdict(list))
    groups = read_json(paths.holdout_groups) if paths.holdout_groups.exists() else {}
    queried = _query_counts(paths)
    for f in probe_files:
        for r in read_jsonl(f):
            b, kind, _ = r["probe_id"].split("|")
            R = ref[b]
            a = r["max_act"] if R["primary"] > 0 else (None if r["min_act"] is None else -r["min_act"])
            by_unit[b][kind].append(np.nan if a is None else a / R["ref"])
            hits[b][kind].append(np.nan if a is None else float(hit(b, a, r["n_tokens"])))
    rows, sim_items = [], []
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
        row["n_queries"] = queried.get(b, 0)
        sp = paths.agent_results("simulate") / f"{b}.json"
        if sp.exists() and b in truth:
            sc = read_json(sp)["scores"]
            row.update(simulation_metrics([sc[str(i)] for i in range(len(truth[b]))], truth[b], groups.get(b)))
            g = groups.get(b) or [None] * len(truth[b])
            for i, t in enumerate(truth[b]):
                sim_items.append({"blind_id": b, "unit_id": R["unit_id"], "layer": R["layer"], "cond": R["cond"],
                                  "item": i, "group": g[i], "true_frac": t, "rating": sc[str(i)]})
        else:
            row["sim_spearman"] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df["threshold"] = threshold_label(cfg)
    out = paths.scores / threshold_label(cfg)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "unit_scores.csv", index=False)
    # one row per simulator item: group = 'top' (max-activating) or 'random' (typical), never shown to the simulator
    pd.DataFrame(sim_items).to_csv(out / "sim_items.csv", index=False)
    log(f"[scores] {len(df)} units -> {out / 'unit_scores.csv'} (+ sim_items.csv)")
    return df
