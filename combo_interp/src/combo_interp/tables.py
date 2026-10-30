"""LaTeX tables of per-unit scores by condition and layer (the format of the slides' rerun_tables.tex)."""
import numpy as np
import pandas as pd

# display order; blank lines (\addlinespace) between the groups
COND_GROUPS = [["neuron"], ["related_5", "related_20"], ["ksparse_2", "ksparse_5", "ksparse_20", "ksparse_100"],
               ["dense_raw", "dense_white"]]

def sim_table_specs(cfg):
    E = cfg.export
    top, pool, rnd = int(E.holdout_top_items), int(E.holdout_top_pool), int(E.holdout_random_items)
    return [
        ("sim_spearman", f"Simulator correlation (all {top + rnd} held-out items)",
         "Spearman correlation between the simulator's 0--10 ratings and true activations."),
        ("sim_spearman_top", "Simulator correlation: top-activation items",
         f"Within the {top} items spread over the unit's {pool} strongest held-out firings."),
        ("sim_spearman_random", "Simulator correlation: ordinary items",
         f"Within the {rnd} uniformly random tokens; units whose ratings are all equal are left out."),
    ]


def _cell(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "--"
    if len(v) == 1:
        return f"${v[0]:.2f}$"
    return f"${v.mean():.2f} \\pm {v.std(ddof=1) / np.sqrt(len(v)):.2f}$"


def cond_layer_table(df, metric, title, note, frame=True):
    """Mean +- standard error over units, per condition (rows) and layer (columns), plus All layers."""
    layers = sorted(int(L) for L in df["layer"].unique())
    conds = [c for g in COND_GROUPS for c in g if c in set(df["cond"])]
    extra = sorted(set(df["cond"]) - set(conds))
    groups = [[c for c in g if c in conds] for g in COND_GROUPS] + ([extra] if extra else [])
    groups = [g for g in groups if g]
    n = df.groupby(["cond", "layer"]).size()
    per_layer = sorted(set(n.values))
    lines = []
    if frame:
        lines += [f"\\begin{{frame}}{{{title}}}", "\\centering\\small"]
    lines += ["\\begin{tabular}{l" + "c" * (len(layers) + 1) + "}", "\\toprule",
              "Condition & " + " & ".join(f"Layer {L}" for L in layers) + " & All \\\\", "\\midrule"]
    for gi, g in enumerate(groups):
        if gi:
            lines.append("\\addlinespace")
        for c in g:
            d = df[df["cond"] == c]
            cells = [_cell(d.loc[d["layer"] == L, metric]) for L in layers] + [_cell(d[metric])]
            lines.append(f"\\texttt{{{c.replace('_', chr(92) + '_')}}} & " + " & ".join(cells) + " \\\\")
    count = f"{per_layer[0]} per layer, {per_layer[0] * len(layers)} in All" if len(per_layer) == 1 else "units per cell vary"
    lines += ["\\bottomrule", "\\end{tabular}"]
    if frame:
        lines += ["", "\\vspace{0.5em}", f"{{\\scriptsize Mean $\\pm$ s.e.\\ over units ({count}). {note}}}",
                  "\\end{frame}"]
    return "\n".join(lines)


def sim_tables(df, cfg):
    return "\n".join(cond_layer_table(df, m, t, n) for m, t, n in sim_table_specs(cfg) if m in df.columns) + "\n"
