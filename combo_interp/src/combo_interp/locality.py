"""Experiment B: is each unit's set of strongest firings *local* in the MLP input space, and does that explain the
agents' neuron-vs-combination gap?

Per unit (primary pole, same rule and one-hit-per-document dedupe as `pilot.py`) take the top-N firings and the MLP
input x (post-norm input of gate_proj) at those tokens. On centered, unit-normalized x:

  coh           mean pairwise cosine of the top-N inputs (how tightly they sit together)
  coh_excess    coh minus the median coh of random-token sets of the same size (0 = no more local than random text;
                random sets have coh ~ 0 after centering, so this replaces the ratio coh_ratio, which is ill-defined)
  pr            participation ratio of the top-N Gram spectrum (effective dimensionality of the set)
  k_star, sil   spherical k-means k = 2..6, best silhouette; k_star = 1 if the best silhouette < SIL_MIN
  share4        share of the largest cluster at k = 4
  cos_gate      neurons: |cos(W_gate row, mean top-N x)| / median of the same for random gate rows
  dom_share     sparse combos: share of the top-N whose largest contribution comes from the most common constituent
  dom_eff       sparse combos: effective number of constituents that dominate some top firing (exp entropy)

Control condition `xdir`: random directions r in x-space used as units (z = r·x). Any single direction's tail is
somewhat local; neurons should be more local than that if the gate nonlinearity carves out a region.
"""
from __future__ import annotations

import numpy as np

from .export import dedupe_hits

SIL_MIN = 0.1        # fixed before looking at results
K_RANGE = range(2, 7)


# ----------------------------------------------------------------------------- top firings per unit
def primary_tops(big, mean_u, std_u, col, seq_len, n):
    """(sign, gids) of the unit's top-n firings on its primary pole, one per document. Same pole rule as pilot.py."""
    zpos = (big["pos_vals"][col, 0] - mean_u[col]) / std_u[col]
    zneg = (mean_u[col] - big["neg_vals"][col, 0]) / std_u[col]
    pole, sgn = ("pos", 1.0) if zpos >= zneg else ("neg", -1.0)
    sv = sgn * big[f"{pole}_vals"][col]
    if not np.isfinite(sv[0]) or sv[0] <= 0:
        return sgn, np.zeros(0, np.int64)
    hits = dedupe_hits(sv, big[f"{pole}_gid"][col], seq_len, 1, limit=n)
    return sgn, np.array([g for _, _, g in hits], dtype=np.int64)


# ----------------------------------------------------------------------------- one pass over the corpus (torch)
def capture_pass(model, store, want, R, n_top, skip_first=1, max_batch_tokens=8192, log=print):
    """One forward pass over `store`.
    want: {L: sorted unique gids} -> returns X[L] (len, d) float16 aligned with want[L].
    R:    {L: (d, n_ctrl) directions} -> per direction the top `n_top` documents by max_t r·x (one hit per document):
          ctrl[L] = dict(vals (n_top, n), gid (n_top, n), x (n_top, n, d) float16), sorted descending."""
    import torch
    from .hooks import MLPCapture

    layers = sorted(set(want) | set(R))
    device = next(model.parameters()).device
    Lq = store.seq_len
    X = {L: None for L in want}
    Rt = {L: torch.tensor(R[L], dtype=torch.float32, device=device) for L in R}
    ctrl = {L: None for L in R}
    nb = 0
    with MLPCapture(model, layers, capture_input=True) as cap:
        for batch in store.iter_batches(max_batch_tokens):
            ids = torch.from_numpy(batch["input_ids"]).to(device)
            lengths = torch.from_numpy(batch["lengths"]).to(device)
            ar = torch.arange(ids.shape[1], device=device)
            attn = ar[None, :] < lengths[:, None]
            valid = attn & (ar[None, :] >= skip_first)
            cap.run(ids, attn.long())
            gid = batch["seq_ids"][:, None] * Lq + np.arange(ids.shape[1])[None, :]
            for L in layers:
                x = cap.x[L].float()
                if L in want:
                    w = want[L]
                    j = np.clip(np.searchsorted(w, gid), 0, len(w) - 1)
                    hit = (w[j] == gid) & valid.cpu().numpy()
                    if hit.any():
                        if X[L] is None:
                            X[L] = np.zeros((len(w), x.shape[-1]), np.float16)
                        b, t = np.nonzero(hit)
                        X[L][j[b, t]] = x[b, t].cpu().numpy()
                if L in Rt:
                    s = (x @ Rt[L]).masked_fill(~valid[..., None], float("-inf"))      # (B, T, n)
                    v, t = s.max(1)                                                     # per document
                    B, n = v.shape
                    xs = x[torch.arange(B, device=device)[:, None], t]                  # (B, n, d)
                    g = torch.from_numpy(batch["seq_ids"]).to(device)[:, None] * Lq + t
                    c = ctrl[L]
                    if c is not None:
                        v = torch.cat([c["vals"], v]); g = torch.cat([c["gid"], g]); xs = torch.cat([c["x"], xs.half()])
                    k = min(n_top, v.shape[0])
                    v, o = v.topk(k, dim=0)
                    ctrl[L] = {"vals": v, "gid": g.gather(0, o), "x": xs.half()[o, torch.arange(n, device=device)[None, :]]}
            nb += 1
            if nb % 50 == 0:
                log(f"[locality] {nb} batches")
    ctrl = {L: {k: c[k].cpu().numpy() for k in c} for L, c in ctrl.items()}
    return X, ctrl


# ----------------------------------------------------------------------------- metrics (numpy)
def _unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def coherence(Xn):
    n = len(Xn)
    s = Xn.sum(0)
    return float((s @ s - n) / (n * (n - 1)))


def participation_ratio(Xn):
    ev = np.linalg.eigvalsh(Xn @ Xn.T)
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def _kmeans(Xn, k, rng, iters=30, restarts=3):
    best, best_obj = None, -np.inf
    for _ in range(restarts):
        C = Xn[rng.choice(len(Xn), k, replace=False)]
        for _ in range(iters):
            a = (Xn @ C.T).argmax(1)
            S = np.zeros_like(C)
            np.add.at(S, a, Xn)
            nrm = np.linalg.norm(S, axis=1, keepdims=True)
            C = np.where(nrm > 1e-9, S / np.maximum(nrm, 1e-12), Xn[rng.choice(len(Xn), k)])
        a = (Xn @ C.T).argmax(1)
        obj = (Xn @ C.T).max(1).sum()
        if obj > best_obj:
            best, best_obj = a, obj
    return best


def silhouette(Xn, a):
    D = 1 - Xn @ Xn.T
    labs = np.unique(a)
    if len(labs) < 2:
        return -1.0
    own = np.zeros(len(a)); other = np.full(len(a), np.inf)
    for c in labs:
        m = a == c
        mean_d = D[:, m].sum(1) / np.maximum(m.sum() - (a == c), 1)    # excludes self for members
        own[m] = mean_d[m]
        other[~m] = np.minimum(other[~m], D[~m][:, m].mean(1))
    single = np.array([(a == a[i]).sum() == 1 for i in range(len(a))])
    s = (other - own) / np.maximum(np.maximum(own, other), 1e-12)
    s[single] = 0.0
    return float(s.mean())


def set_metrics(Xn, rng):
    out = {"coh": coherence(Xn), "pr": participation_ratio(Xn)}
    best_k, best_s, share4 = 1, -1.0, np.nan
    for k in K_RANGE:
        if k >= len(Xn):
            break
        a = _kmeans(Xn, k, rng)
        s = silhouette(Xn, a)
        if s > best_s:
            best_k, best_s = k, s
        if k == 4:
            share4 = float(np.bincount(a).max() / len(a))
    out.update({"sil": best_s, "k_star": best_k if best_s >= SIL_MIN else 1, "share4": share4})
    return out


def dominance(Xraw, Wg, Wu, neurons, weights, sign):
    """For a sparse combo z = sign * sum_i w_i p_i: which constituent contributes most at each top firing."""
    import torch
    x = torch.tensor(Xraw, dtype=torch.float32)
    g = x @ torch.tensor(Wg[neurons].T); u = x @ torch.tensor(Wu[neurons].T)
    p = (torch.nn.functional.silu(g) * u).numpy()
    contrib = sign * p * np.asarray(weights)[None, :]
    dom = contrib.argmax(1)
    c = np.bincount(dom, minlength=len(neurons)) / len(dom)
    nz = c[c > 0]
    return float(c.max()), float(np.exp(-(nz * np.log(nz)).sum()))


# ----------------------------------------------------------------------------- regression helpers
def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def design(df, group_col, ref, extra=()):
    """Intercept + one-hot groups (minus ref) + layer FE + extra numeric columns. Returns (matrix, names)."""
    cols, names = [np.ones(len(df))], ["const"]
    for gname in sorted(set(df[group_col]) - {ref}):
        cols.append((df[group_col] == gname).values.astype(float)); names.append(f"{group_col}={gname}")
    for L in sorted(set(df.layer))[1:]:
        cols.append((df.layer == L).values.astype(float)); names.append(f"layer={L}")
    for e in extra:
        cols.append(df[e].values.astype(float)); names.append(e)
    return np.column_stack(cols), names


def mediation(df, y, group_col, ref, mediator, n_boot=2000, seed=0):
    """Group effects (vs ref) on y without and with the mediator; bootstrap CIs over units (stratified by group)."""
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=[y, mediator]).reset_index(drop=True)

    def fit(dd):
        X0, n0 = design(dd, group_col, ref)
        X1, n1 = design(dd, group_col, ref, [mediator])
        b0, b1 = ols(dd[y].values, X0), ols(dd[y].values, X1)
        r = {n: b for n, b in zip(n0, b0) if n.startswith(group_col)}
        r1 = {n: b for n, b in zip(n1, b1) if n.startswith(group_col)}
        return r, r1, b1[n1.index(mediator)]

    r0, r1, bm = fit(d)
    groups = [g for g in d[group_col].unique()]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(np.flatnonzero(d[group_col] == g), (d[group_col] == g).sum()) for g in groups])
        boots.append(fit(d.iloc[idx]))
    rows = []
    for name in r0:
        b0 = np.array([b[0].get(name, np.nan) for b in boots]); b1 = np.array([b[1].get(name, np.nan) for b in boots])
        rows.append({"y": y, "mediator": mediator, "term": name, "n": len(d),
                     "effect_total": r0[name], "effect_direct": r1[name],
                     "total_lo": np.nanpercentile(b0, 5), "total_hi": np.nanpercentile(b0, 95),
                     "direct_lo": np.nanpercentile(b1, 5), "direct_hi": np.nanpercentile(b1, 95),
                     "attenuation": 1 - r1[name] / r0[name] if r0[name] != 0 else np.nan,
                     "attenuation_lo": np.nanpercentile(1 - b1 / b0, 5), "attenuation_hi": np.nanpercentile(1 - b1 / b0, 95)})
    bms = np.array([b[2] for b in boots])
    rows.append({"y": y, "mediator": mediator, "term": f"slope:{mediator}", "n": len(d), "effect_total": np.nan,
                 "effect_direct": bm, "direct_lo": np.percentile(bms, 5), "direct_hi": np.percentile(bms, 95)})
    return rows


def within_cond_spearman(df, x, y, cond_col="cond", n_boot=2000, seed=0):
    """Spearman of x and y after ranking within each condition (does x predict y beyond condition?)."""
    from scipy.stats import spearmanr
    d = df.dropna(subset=[x, y]).copy()
    d["_rx"] = d.groupby(cond_col)[x].rank(pct=True)
    d["_ry"] = d.groupby(cond_col)[y].rank(pct=True)
    rho = spearmanr(d["_rx"], d["_ry"]).statistic
    rng = np.random.default_rng(seed)
    bs = [spearmanr(*d.iloc[rng.integers(0, len(d), len(d))][["_rx", "_ry"]].values.T).statistic for _ in range(n_boot)]
    return {"x": x, "y": y, "n": len(d), "rho": rho, "lo": np.nanpercentile(bs, 5), "hi": np.nanpercentile(bs, 95)}
