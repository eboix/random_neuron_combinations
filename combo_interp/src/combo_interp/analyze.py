"""Statistics: is a combination condition equivalent to the neuron baseline?  (TOST via bootstrap CI)

Pre-register `analysis.equiv_margin` in the config BEFORE looking at agent results. For metric m and condition c:
difference d = mean_c(m) - mean_neuron(m) with a layer-stratified bootstrap CI at level `analysis.ci` (90% <=> two
one-sided tests at alpha = .05). Verdicts: equivalent (CI inside +-delta), combo_worse / combo_better (CI entirely
beyond a margin), otherwise inconclusive. Compactness (n_parts) is reported separately because "equally
interpretable" should mean equal predictive power at comparable explanation complexity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["margin", "rate_pred", "sim_spearman", "blind_max_logratio"]
DELTA_KEY = {"margin": "margin", "rate_pred": "margin", "sim_spearman": "sim_spearman",
             "blind_max_logratio": "blind_max_logratio"}


def _strata(df, col):
    return [g[col].dropna().to_numpy() for _, g in df.groupby("layer") if g[col].notna().any()]


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, col: str, n_boot: int, ci: float, rng):
    sa, sb = _strata(a, col), _strata(b, col)
    if not sa or not sb:
        return np.nan, np.nan, np.nan
    est = np.concatenate(sa).mean() - np.concatenate(sb).mean()
    d = np.empty(n_boot)
    for i in range(n_boot):
        ma = np.concatenate([rng.choice(x, len(x)) for x in sa]).mean()
        mb = np.concatenate([rng.choice(x, len(x)) for x in sb]).mean()
        d[i] = ma - mb
    lo, hi = np.percentile(d, [100 * (1 - ci) / 2, 100 * (1 - (1 - ci) / 2)])
    return float(est), float(lo), float(hi)


def verdict(lo, hi, delta):
    if np.isnan(lo):
        return "n/a"
    if lo > -delta and hi < delta:
        return "equivalent"
    if hi < -delta:
        return "combo_worse"
    if lo > delta:
        return "combo_better"
    return "inconclusive"


def equivalence_table(df, cfg, base="neuron"):
    rng = np.random.default_rng(cfg.seed + 5)
    A = cfg.analysis
    ref = df[df.cond == base]
    rows = []
    for cond in [c for c in dict.fromkeys(df.cond) if c != base]:
        sub = df[df.cond == cond]
        for m in METRICS:
            delta = float(A.equiv_margin[DELTA_KEY[m]])
            est, lo, hi = boot_diff(sub, ref, m, int(A.n_boot), float(A.ci), rng)
            rows.append({"cond": cond, "metric": m, "n_cond": int(sub[m].notna().sum()), "n_base": int(ref[m].notna().sum()),
                         "diff": est, "ci_lo": lo, "ci_hi": hi, "delta": delta, "verdict": verdict(lo, hi, delta)})
    return pd.DataFrame(rows)


def by_cond_table(df):
    agg = {m: ["mean", "std", "count"] for m in METRICS + ["n_parts", "confidence", "rate_null", "rate_rand", "rate_distant"]
           if m in df.columns}
    t = df.groupby("cond", sort=False).agg(agg)
    t.columns = ["_".join(c) for c in t.columns]
    return t.reset_index()


def ols_effects(df, col, cfg, base="neuron"):
    """cond effects vs baseline adjusting for layer and log-kurtosis (activation-shape confound)."""
    rng = np.random.default_rng(cfg.seed + 6)
    d = df.dropna(subset=[col, "kurtosis"]).reset_index(drop=True)
    conds = [c for c in dict.fromkeys(d.cond) if c != base]
    layers = sorted(d.layer.unique())[1:]
    def design(x):
        lk = np.log1p(np.clip(x["kurtosis"].to_numpy(float), 0, None))
        lk = (lk - lk.mean()) / (lk.std() + 1e-9)
        cols = [np.ones(len(x))] + [(x["cond"] == c).to_numpy(float) for c in conds] + \
               [(x["layer"] == l).to_numpy(float) for l in layers] + [lk]
        return np.stack(cols, 1)
    X, y = design(d), d[col].to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    boots = []
    for _ in range(int(cfg.analysis.n_boot)):
        idx = rng.integers(0, len(d), len(d))
        try:
            boots.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0])
        except np.linalg.LinAlgError:
            continue
    boots = np.array(boots)
    q = np.percentile(boots[:, 1:1 + len(conds)], [5, 95], axis=0)
    names = conds
    return pd.DataFrame({"cond": names, "metric": col, "effect_vs_neuron": beta[1:1 + len(conds)],
                         "ci_lo": q[0], "ci_hi": q[1], "log_kurtosis_coef": beta[-1]})


def plot_dose_response(df, outdir, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed + 8)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.4 * len(METRICS), 3.8), squeeze=False)
    base = df[df.cond == "neuron"]
    for ax, m in zip(axes[0], METRICS):
        def mean_ci(x):
            x = x.dropna().to_numpy()
            if len(x) < 2:
                return np.nan, np.nan, np.nan
            b = [rng.choice(x, len(x)).mean() for _ in range(300)]
            return x.mean(), np.percentile(b, 5), np.percentile(b, 95)
        mu, lo, hi = mean_ci(base[m])
        if not np.isnan(mu):
            ax.axhline(mu, color="k", lw=1); ax.axhspan(lo, hi, color="k", alpha=0.12, label="neuron")
        for kind, color in (("ksparse", "C0"), ("related", "C1")):
            sub = df[df.kind == kind]
            ks = sorted(sub.k.unique())
            pts = [mean_ci(sub[sub.k == k][m]) for k in ks]
            if ks:
                ax.errorbar(ks, [p[0] for p in pts], yerr=[[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]],
                            marker="o", color=color, capsize=2, label=kind)
        for kind, mk in (("dense_raw", "s"), ("dense_white", "^")):
            sub = df[df.kind == kind]
            if len(sub):
                mu2, l2, h2 = mean_ci(sub[m])
                xk = (max(df.k.max(), 100)) * 2.5
                ax.errorbar([xk], [mu2], yerr=[[mu2 - l2], [h2 - mu2]], marker=mk, color="C3" if kind == "dense_raw" else "C4",
                            capsize=2, label=kind)
        ax.set_xscale("log"); ax.set_xlabel("k (neurons per unit); right-most = dense"); ax.set_title(m)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(outdir / "dose_response.png", dpi=130); plt.close(fig)


def run_analysis(cfg, paths, log=print):
    df = pd.read_csv(paths.scores / "unit_scores.csv")
    out = paths.results
    out.mkdir(parents=True, exist_ok=True)
    eq = equivalence_table(df, cfg)
    eq.to_csv(out / "equivalence.csv", index=False)
    by_cond_table(df).to_csv(out / "by_cond.csv", index=False)
    ols = pd.concat([ols_effects(df, m, cfg) for m in METRICS if df[m].notna().sum() > 10], ignore_index=True) \
        if len(df) > 20 else pd.DataFrame()
    ols.to_csv(out / "ols_effects.csv", index=False)
    parts = df.groupby(["cond", "n_parts"]).agg(n=("margin", "size"), margin=("margin", "mean"),
                                                sim_spearman=("sim_spearman", "mean")).reset_index()
    parts.to_csv(out / "n_parts_breakdown.csv", index=False)
    plot_dose_response(df, out, cfg)
    lines = ["# Equivalence vs neuron baseline", "",
             f"CI level {cfg.analysis.ci}, margins {dict(cfg.analysis.equiv_margin)}", ""]
    for _, r in eq.iterrows():
        lines.append(f"- {r.cond:14s} {r.metric:20s} diff={r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  -> {r.verdict}")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    return eq
