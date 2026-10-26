#!/usr/bin/env python3
"""CPU/GPU: Experiment B — per-unit locality of the top firings in MLP-input space, and mediation of the agent
metrics (see combo_interp.locality).

  python scripts/10_locality.py --config ...            # capture (one pass over train) + metrics + analysis
  python scripts/10_locality.py --config ... --analyze  # only redo the analysis from unit_locality.csv
"""
from _common import setup


def extra(ap):
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--recapture", action="store_true")


cfg, paths, args = setup(__doc__, extra)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from combo_interp.locality import _unit, design, dominance, mediation, ols, primary_tops, set_metrics, within_cond_spearman  # noqa: E402

lc = cfg.locality
layers = [int(l) for l in cfg.model.layers]
out = paths.results / "locality"
out.mkdir(parents=True, exist_ok=True)
cap_dir = paths.data / "locality"
cap_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(int(cfg.seed) + 2024)


def compute():
    import torch

    from combo_interp.corpus import CorpusStore
    from combo_interp.geometry import mlp_weights
    from combo_interp.locality import capture_pass
    from combo_interp.model_utils import load_model
    from combo_interp.units import load_unit_set

    store = CorpusStore(paths.corpus("train"))
    Lq = store.seq_len
    units, want, R = {}, {}, {}
    uni = np.load(paths.cache("train", layers[0]) / "uniform.npz")["gid"]
    uni = np.sort(rng.choice(uni, min(int(lc.uniform_tokens), len(uni)), replace=False))
    for L in layers:
        us = load_unit_set(paths, L)
        big = dict(np.load(paths.cache("train", L) / "topk_big.npz"))
        std = np.maximum(us["std_u"], 1e-12)
        units[L] = [(m, *primary_tops(big, us["mean_u"], std, col, Lq, int(lc.n_top))) for col, m in enumerate(us["meta"])]
        units[L] = [(m, s, g, us["A"][:, m["col"]]) for m, s, g in units[L]]
        want[L] = np.unique(np.concatenate([uni] + [g for _, _, g, _ in units[L]]))
    model, _ = load_model(cfg, "cuda" if torch.cuda.is_available() else "cpu")
    d = model.config.hidden_size
    for L in layers:
        R[L] = np.random.default_rng(int(cfg.seed) + 7 * L).standard_normal((d, int(lc.n_xdir))).astype(np.float32)
    f = cap_dir / "capture.npz"
    if args.recapture or not f.exists():
        X, ctrl = capture_pass(model, store, want, R, int(lc.n_top), int(cfg.cache.skip_first_tokens),
                               int(cfg.cache.max_batch_tokens), log=lambda m: print(m, flush=True))
        np.savez(f, **{f"X_L{L}": X[L] for L in layers}, **{f"want_L{L}": want[L] for L in layers},
                 **{f"ctrl_{k}_L{L}": ctrl[L][k] for L in layers for k in ctrl[L]})
    z = np.load(f)

    rows, base_rows = [], []
    for L in layers:
        X = z[f"X_L{L}"].astype(np.float32)
        w = z[f"want_L{L}"]
        idx = lambda g: np.searchsorted(w, g)
        mu = X[idx(uni)].mean(0)
        Xu = _unit(X[idx(uni)] - mu)
        Wg, Wu = mlp_weights(model, L)
        Gn = _unit(Wg)
        n = int(lc.n_top)
        # random-token baseline
        base = [set_metrics(Xu[rng.choice(len(Xu), n, replace=False)], rng) for _ in range(int(lc.n_random_sets))]
        bdf = pd.DataFrame(base)
        base_coh = float(bdf.coh.median())
        for r in base:
            base_rows.append({"layer": L, **r})

        def record(meta, sign, Xraw, a=None):
            Xn = _unit(Xraw - mu)
            r = {"unit_id": meta["unit_id"], "layer": L, "cond": meta["cond"], "k": meta.get("k", 0), "n_top": len(Xn),
                 "sign": sign, **set_metrics(Xn, rng)}
            m = _unit(Xn.mean(0, keepdims=True))[0]
            if meta["cond"] == "neuron":
                j = meta["neurons"][0]
                ref = np.median(np.abs(Gn[rng.choice(len(Gn), 200, replace=False)] @ m))
                r["cos_gate"] = float(abs(Gn[j] @ m) / ref)
            if meta["cond"].startswith(("ksparse", "related")):
                r["dom_share"], r["dom_eff"] = dominance(Xraw, Wg, Wu, list(meta["neurons"]), a[list(meta["neurons"])], sign)
            return r

        for meta, sign, gids, a in units[L]:
            if len(gids) < 20:
                continue
            rows.append(record(meta, sign, X[idx(gids)], a))
        cx = z[f"ctrl_x_L{L}"].astype(np.float32)               # (n_top, n_xdir, d)
        for j in range(cx.shape[1]):
            rows.append(record({"unit_id": f"L{L}_xdir_{j:04d}", "cond": "xdir"}, 1.0, cx[:, j]))
        print(f"[locality] layer {L}: {sum(r['layer'] == L for r in rows)} units, random-set coh {base_coh:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "unit_locality.csv", index=False)
    pd.DataFrame(base_rows).to_csv(out / "random_sets.csv", index=False)
    return df


df = pd.read_csv(out / "unit_locality.csv") if args.analyze else compute()

# ----------------------------------------------------------------------------- analysis
COND_ORDER = ["neuron", "related_5", "related_20", "ksparse_2", "ksparse_5", "ksparse_20", "ksparse_100",
              "dense_raw", "dense_white", "xdir"]
base = pd.read_csv(out / "random_sets.csv")
# random token sets have coh ~ 0 after centering, so a ratio to them is ill-defined: use the excess over their median
df["coh_excess"] = df.coh - df.layer.map(base.groupby("layer").coh.median())
summ = df.groupby(["cond", "layer"])[["coh_excess", "pr", "k_star", "sil", "share4", "dom_share", "dom_eff", "cos_gate"]] \
    .median().reset_index()
summ.to_csv(out / "by_cond.csv", index=False)

pilot = pd.read_csv(paths.results / "pilot_units.csv")
scores = pd.read_csv(paths.scores / str(lc.scores_dir) / "unit_scores.csv")
units = df.merge(pilot[["unit_id", "kurtosis", "tok_entropy", "top_tok_share", "rank100_ratio"]], on="unit_id", how="left")
agents = units.merge(scores[["unit_id", "margin", "sim_spearman", "rate_pred", "blind_max_logratio", "n_parts"]],
                     on="unit_id", how="inner")
agents.to_csv(out / "agent_units_with_locality.csv", index=False)

med_rows, sp_rows = [], []
mediators = [str(lc.mediator), "k_star", "pr"]
for m in mediators:
    for y in ["sim_spearman", "margin", "rate_pred", "blind_max_logratio", "n_parts"]:
        med_rows += [{"data": "agents", **r} for r in mediation(agents, y, "cond", "neuron", m)]
        sp_rows.append({"data": "agents", **within_cond_spearman(agents, m, y)})
    for y in ["tok_entropy", "top_tok_share", "rank100_ratio"]:
        d = units[units.cond != "xdir"]
        med_rows += [{"data": "all_units", **r} for r in mediation(d, y, "cond", "neuron", m, n_boot=500)]
        sp_rows.append({"data": "all_units", **within_cond_spearman(d, m, y, n_boot=500)})
med = pd.DataFrame(med_rows)
med.to_csv(out / "mediation.csv", index=False)
sp = pd.DataFrame(sp_rows)
sp.to_csv(out / "within_cond_spearman.csv", index=False)

# robustness: is coherence just token repetition? sim_spearman ~ cond + layer + covariates, nested specifications
agents["log_kurtosis"] = np.log(agents["kurtosis"])
tokcov = ["top_tok_share", "tok_entropy", "log_kurtosis"]
specs = {"cond": [], "+tokens": tokcov, "+tokens+coh": tokcov + ["coh_excess"], "+tokens+pr": tokcov + ["pr"],
         "+coh+pr": ["coh_excess", "pr"]}
rd = agents.dropna(subset=tokcov + ["coh_excess", "pr", "sim_spearman"]).reset_index(drop=True)
brng = np.random.default_rng(0)
rob = []
for name, extra in specs.items():
    X, nm = design(rd, "cond", "neuron", extra)
    est = dict(zip(nm, ols(rd.sim_spearman.values, X)))
    bs = []
    for _ in range(1000):
        bd = rd.iloc[brng.integers(0, len(rd), len(rd))]
        Xb, nb = design(bd, "cond", "neuron", extra)
        bs.append(dict(zip(nb, ols(bd.sim_spearman.values, Xb))))
    for term in [t for t in nm if t.startswith("cond=") or t in extra]:
        v = np.array([b.get(term, np.nan) for b in bs])
        rob.append({"spec": name, "term": term, "est": est[term], "lo": np.nanpercentile(v, 5), "hi": np.nanpercentile(v, 95)})
rob = pd.DataFrame(rob)
rob.to_csv(out / "robustness_sim_spearman.csv", index=False)

# ----------------------------------------------------------------------------- report
lines = ["# Experiment B — per-unit locality (MLP-input space)", ""]
t = summ.pivot_table(index="cond", columns="layer", values=["coh_excess", "k_star", "pr"]).reindex(COND_ORDER)
lines += ["## Median per condition", "", "```", t.round(2).to_string(), "```", ""]
t = summ.pivot_table(index="cond", columns="layer", values=["dom_share", "dom_eff", "cos_gate"]).reindex(COND_ORDER).dropna(how="all")
lines += ["```", t.round(2).to_string(), "```", ""]
lines += ["Random token sets (same size): median coh = " +
          ", ".join(f"L{L} {g.coh.median():.4f}" for L, g in base.groupby("layer")) +
          "; k_star distribution: " + ", ".join(f"L{L} {dict(g.k_star.value_counts().sort_index())}" for L, g in base.groupby("layer")), ""]
pm = med[(med.mediator == str(lc.mediator)) & med.term.str.startswith("cond=")].copy()
pm["cond"] = pm.term.str.replace("cond=", "")
for data, ys in [("agents", ["sim_spearman", "margin"]), ("all_units", ["tok_entropy", "top_tok_share"])]:
    for y in ys:
        tt = pm[(pm.data == data) & (pm.y == y)].set_index("cond")[
            ["effect_total", "total_lo", "total_hi", "effect_direct", "direct_lo", "direct_hi", "attenuation"]]
        lines += [f"## Mediation by {lc.mediator}: {y} ({data}), effects vs neuron (90% bootstrap CI)", "", "```",
                  tt.reindex([c for c in COND_ORDER if c in tt.index]).round(3).to_string(), "```", ""]
lines += ["## Within-condition Spearman (rank within condition)", "", "```", sp.round(3).to_string(index=False), "```", ""]
rt = rob.assign(v=rob.apply(lambda r: f"{r.est:+.3f} [{r.lo:+.2f},{r.hi:+.2f}]", axis=1)).pivot(index="term", columns="spec", values="v")
lines += ["## Robustness: sim_spearman ~ cond + layer + covariates (90% bootstrap CI)", "", "```",
          rt[list(specs)].to_string(), "```"]
(out / "summary.md").write_text("\n".join(lines))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = [c for c in COND_ORDER if c in set(df.cond)]
    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 3.6), sharey=True)
    for ax, L in zip(np.atleast_1d(axes), layers):
        d = df[df.layer == L]
        data = [d[d.cond == c].coh_excess.values for c in conds]
        ax.boxplot(data, vert=False, widths=0.6, showfliers=False,
                   medianprops={"color": "#2a78d6", "lw": 2}, boxprops={"color": "#52514e"},
                   whiskerprops={"color": "#52514e"}, capprops={"color": "#52514e"})
        for i, v in enumerate(data):
            ax.scatter(v, np.full(len(v), i + 1) + rng.uniform(-0.2, 0.2, len(v)), s=6, color="#2a78d6", alpha=0.35, lw=0)
        ax.axvline(0, color="#a3a29d", lw=1, ls="--")
        ax.set_yticks(range(1, len(conds) + 1), conds, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"layer {L}", fontsize=9)
        ax.set_xlabel("coherence of top-100 inputs (0 = random tokens)", fontsize=8, color="#52514e")
        ax.tick_params(labelsize=8, colors="#52514e")
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    fig.tight_layout()
    fig.savefig(out / "coh_by_cond.png", dpi=150)
    plt.close(fig)


plot()
print((out / "summary.md").read_text())
