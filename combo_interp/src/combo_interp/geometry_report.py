"""Summary table and plots for scripts/09_centroids.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ["obj", "dist", "ve", "ve_raw", "nmi_doc", "nmi_token", "nmi_next", "eff_k"]
# reference palette, categorical slots 1-7 in fixed order (validated adjacent-pair order for line charts)
PLOT_SETS = [("gate", "#2a78d6"), ("up", "#eb6834"), ("rand_cov", "#1baf7a"), ("rand_rot", "#eda100"),
             ("combo_5", "#e87ba4"), ("data_pts", "#008300"), ("kmeans", "#6250d6")]
KEYS = ["assign", "init", "layer", "M"]


def _fill(df: pd.DataFrame) -> pd.DataFrame:
    """Older results (and scripts/vision_mech.py) have cosine assignment only and no ve_raw / dist."""
    df = df.copy()
    df["assign"] = df["assign"].fillna("cos") if "assign" in df else "cos"
    for m in METRICS:
        if m not in df:
            df[m] = np.nan
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Median over repetitions per (assign, init, layer, M, set), plus `obj_norm` = position of the set between the
    covariance-matched random null (0) and spherical k-means (1); cos rows only (euclid has no `obj`)."""
    s = _fill(df).groupby(KEYS + ["set"])[METRICS].median().reset_index()
    ref = s.set_index(KEYS + ["set"])["obj"]
    lo = ref.xs("rand_cov", level="set")
    hi = ref.xs("kmeans", level="set")
    key = s.set_index(KEYS).index
    s["obj_norm"] = (s["obj"].values - lo.reindex(key).values) / (hi.reindex(key).values - lo.reindex(key).values)
    return s


def gate_vs_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Per (assign, init, layer, M, metric): gate median minus each null's median, and whether the gate median
    beats the null in every repetition. Metric = obj (cos only), ve, ve_raw."""
    out = []
    for (rule, init, L, M), d in _fill(df).groupby(KEYS):
        for metric in ["obj", "ve", "ve_raw"]:
            gate = d[d.set == "gate"][metric].median()
            if np.isnan(gate):
                continue
            for null in ["rand_cov", "rand_rot", "data_pts", "up", "combo_2", "combo_5", "combo_20"]:
                v = d[d.set == null][metric]
                if len(v):
                    out.append({"assign": rule, "init": init, "layer": L, "M": M, "metric": metric, "null": null,
                                "gate_minus_null": gate - v.median(), "gate_beats_all_reps": bool((gate > v).all())})
    return pd.DataFrame(out)


def plot(s: pd.DataFrame, metric: str, path, rule="cos"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = s[s["assign"] == rule]
    inits = [i for i in ["trained", "random"] if i in set(s.init)]
    layers = sorted(set(s.layer))
    fig, axes = plt.subplots(len(inits), len(layers), figsize=(3.6 * len(layers), 3.0 * len(inits)),
                             squeeze=False, sharex=True)
    for r, init in enumerate(inits):
        for c, L in enumerate(layers):
            ax = axes[r, c]
            d = s[(s.init == init) & (s.layer == L)]
            for name, col in PLOT_SETS:
                e = d[d.set == name].sort_values("M")
                if len(e):
                    ax.plot(e.M, e[metric], color=col, lw=2, marker="o", ms=4, label=name)
            ax.set_xscale("log")
            ax.set_title(f"{init} · layer {L}", fontsize=9, color="#0b0b0b")
            ax.grid(color="#e6e5e0", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=8, colors="#52514e")
            if r == len(inits) - 1:
                ax.set_xlabel("centroids M", fontsize=8, color="#52514e")
            if c == 0:
                ax.set_ylabel(f"{metric} ({rule})", fontsize=8, color="#52514e")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(df: pd.DataFrame, out_root, tag: str):
    s = summarize(df)
    s.to_csv(out_root / f"summary_{tag}.csv", index=False)
    gv = gate_vs_nulls(df)
    gv.to_csv(out_root / f"gate_vs_nulls_{tag}.csv", index=False)
    rules = [r for r in ["cos", "euclid"] if r in set(s["assign"])]
    for rule in rules:
        sfx = "" if rule == "cos" else "_euclid"
        for m in (["obj", "obj_norm", "nmi_doc", "nmi_next", "ve", "ve_raw"] if rule == "cos"
                  else ["dist", "nmi_doc", "nmi_next", "ve", "ve_raw"]):
            if s[s["assign"] == rule][m].notna().any():
                plot(s, m, out_root / f"{m}{sfx}_{tag}.png", rule)
    lines = [f"# Centroid test ({tag} MLP inputs)", "",
             "Assignment `cos` = argmax cosine with unit centroids; `euclid` = nearest raw-scale centroid in raw "
             "(centered) x. `obj` = mean max-cosine of held-out tokens to the set (cos only); `obj_norm` = 0 at the "
             "covariance-matched random null, 1 at spherical k-means; `dist` = mean squared distance to the nearest "
             "centroid / total variance (euclid only); `ve` / `ve_raw` = variance of the unit-normalized / raw x "
             "explained by the partition (1 - within-cluster SS / total SS); `nmi_*` = NMI of the partition with doc "
             "label / token / next token, minus the NMI of a size-matched random partition.", ""]
    order = [p for p, _ in PLOT_SETS] + ["rand_iso", "combo_2", "combo_20"]
    for rule in rules:
        vals = (["obj_norm", "ve", "ve_raw", "nmi_doc", "nmi_next"] if rule == "cos"
                else ["dist", "ve", "ve_raw", "nmi_doc", "nmi_next"])
        for init in [i for i in ["trained", "random"] if i in set(s.init)]:
            for M in sorted(set(s.M)):
                t = s[(s["assign"] == rule) & (s.init == init) & (s.M == M)].pivot_table(index="set", columns="layer",
                                                                                      values=vals)
                t = t[[v for v in vals if v in t.columns.get_level_values(0)]]
                t = t.reindex([x for x in order if x in t.index])
                lines += [f"## {rule}, {init}, M = {M}", "", "```", t.round(3).to_string(), "```", ""]
    (out_root / f"summary_{tag}.md").write_text("\n".join(lines))
    return s
