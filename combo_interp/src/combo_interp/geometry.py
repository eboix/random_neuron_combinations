"""Do neuron weights act as cluster centroids of the MLP input? (cf. Abedsoltan et al. 2026, "Revenge of Monosemanticity")

The paper finds that MLP neurons specialize to one cluster of the input: the (gate) weight places the neuron's firing
region inside a single cluster. If that holds in an LM, the rows of W_gate should partition the MLP input x (post-norm,
the tensor fed to gate_proj) better than random directions do. Here each set of M directions partitions tokens by
`argmax_j cos(w_j, x)` and the partition is scored against null sets that keep different parts of the structure:

  gate / up    rows of W_gate / W_up                       the test sets
  rand_iso     isotropic Gaussian                          weak baseline
  rand_cov     N(0, cov(x))                                matches the anisotropy of the data (main null)
  rand_rot     W_gate rows @ a random rotation             keeps the neurons' norms and mutual angles, drops alignment with x
  combo_k      Gaussian mix of k unit gate rows            the x-space analogue of the ksparse_k units
  rand_gcov    N(mean, cov) of the gate rows               keeps the gate rows' first and second moments, drops the rest
  data_pts     random tokens' x (fit half)                 data-aligned baseline (k-means++-style seeding)
  kmeans       spherical k-means on the fit half           upper bound

Data and centroids fit on documents with even seq_id, everything is evaluated on odd ones.

Two assignment rules (`assign` column):
  cos     argmax_j cos(c_j, x), centroids unit-norm (for equal-norm centroids this IS the Euclidean partition)
  euclid  argmin_j ||x - c_j||^2 on raw centered x with raw-scale centroids: the same sampled sets at their native
          relative norms, each set scaled once so its mean norm equals the mean ||x|| (data_pts = raw tokens, no
          rescaling; kmeans = Euclidean Lloyd k-means). Only differences in centroid norm separate it from `cos`.
  ip      argmax_j <c_j, x> on raw (centered unless center=False) x with rows at native norm: for weight rows, the
          neuron with the largest pre-activation. Weight-like sets only; rand_iso / combo_k get gate-row norms.
"""
from __future__ import annotations

import re

import numpy as np

SETS = ["gate", "up", "rand_iso", "rand_cov", "rand_rot", "data_pts", "kmeans"]   # + combo_k, rand_gcov
IP_SETS = {"gate", "up", "rand_iso", "rand_rot", "rand_gcov"}                      # + combo_k


# ----------------------------------------------------------------------------- capture (torch)
def capture_inputs(model, store, gids, layers, max_batch_tokens=8192, log=print):
    """MLP inputs x at the given global token ids. Returns {L: float16 (n, d)} in the order of `gids`."""
    import torch
    from .hooks import MLPCapture

    gids = np.asarray(gids, dtype=np.int64)
    seq_len = store.seq_len
    want = {}
    for i, g in enumerate(gids):
        want.setdefault(int(g // seq_len), []).append((int(g % seq_len), i))
    device = next(model.parameters()).device
    out = {L: None for L in layers}
    done = 0
    with MLPCapture(model, layers, capture_input=True) as cap:
        for batch in store.iter_batches(max_batch_tokens):
            rows = [r for r, s in enumerate(batch["seq_ids"]) if int(s) in want]
            if not rows:
                continue
            ids = torch.from_numpy(batch["input_ids"][rows]).to(device)
            lengths = torch.from_numpy(batch["lengths"][rows]).to(device)
            attn = (torch.arange(ids.shape[1], device=device)[None, :] < lengths[:, None]).long()
            cap.run(ids, attn)
            b_idx, t_idx, dst = [], [], []
            for b, r in enumerate(rows):
                for t, i in want[int(batch["seq_ids"][r])]:
                    b_idx.append(b); t_idx.append(t); dst.append(i)
            for L in layers:
                x = cap.x[L][b_idx, t_idx].float().cpu().numpy()
                if out[L] is None:
                    out[L] = np.zeros((len(gids), x.shape[1]), dtype=np.float16)
                out[L][dst] = x
            done += len(dst)
            log(f"[geometry] captured {done}/{len(gids)} tokens")
    return out


def mlp_weights(model, layer):
    from .model_utils import get_blocks
    mlp = get_blocks(model)[layer].mlp
    return (mlp.gate_proj.weight.detach().float().cpu().numpy(),
            mlp.up_proj.weight.detach().float().cpu().numpy())


# ----------------------------------------------------------------------------- labels
_STOP = {
    "en": "the of and to in is that for it with as was on are by this be".split(),
    "de": "der die und das ist nicht ein eine den mit von zu sich auf dem des im".split(),
    "fr": "le la les et des est une un du que pour dans qui en au sur pas".split(),
    "es": "el la los las y que de en del un una por con para es se".split(),
}
_CODE = re.compile(r"(^\s*(def |class |import |from \S+ import|#include|return\b|if \(|for \()|[{};]\s*$|==|!=|\+=|->)", re.M)


def doc_label(text: str) -> str:
    """Coarse document label from content: code / ja / ru / bn / de / fr / es / en / other. A heuristic
    (the corpus does not store sources per document and the builder's order cannot be replayed exactly)."""
    if not text:
        return "other"
    n = max(len(text), 1)
    if len(re.findall(r"[぀-ヿ一-鿿]", text)) / n > 0.2:
        return "ja"
    if len(re.findall(r"[Ѐ-ӿ]", text)) / n > 0.3:
        return "ru"
    if len(re.findall(r"[ঀ-৿]", text)) / n > 0.3:
        return "bn"
    lines = max(text.count("\n") + 1, 1)
    if len(_CODE.findall(text)) / lines > 0.25:
        return "code"
    words = re.findall(r"[a-zà-ÿ]+", text.lower())
    if not words:
        return "other"
    counts = {k: sum(w in set(v) for w in words) for k, v in _STOP.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] / len(words) > 0.05 else "other"


# ----------------------------------------------------------------------------- centroid sets
def unit_rows(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


def random_rotation(d, rng):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def spherical_kmeans(Xn, M, rng, iters=20, chunk=8192, device="cpu"):
    """Xn: unit rows. Returns (M, d) unit centroids (k-means++-lite: random data init)."""
    import torch
    X = torch.from_numpy(np.ascontiguousarray(Xn, dtype=np.float32)).to(device)
    C = X[torch.from_numpy(rng.choice(len(X), M, replace=False)).to(device)].clone()
    for _ in range(iters):
        a = assign(X, C, chunk)
        S = torch.zeros_like(C).index_add_(0, a, X)
        empty = S.norm(dim=1) < 1e-8
        if empty.any():          # re-seed empty clusters with random points
            S[empty] = X[torch.from_numpy(rng.choice(len(X), int(empty.sum()), replace=False)).to(device)]
        C = S / S.norm(dim=1, keepdim=True)
    return C.cpu().numpy()


def euclidean_kmeans(X, M, rng, iters=20, chunk=8192, device="cpu"):
    """Lloyd k-means on raw rows X (random data init). Returns (M, d) centroids."""
    import torch
    X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
    C = X[torch.from_numpy(rng.choice(len(X), M, replace=False)).to(device)].clone()
    for _ in range(iters):
        a = assign(X, C, chunk, bias=-0.5 * (C * C).sum(1))
        n = torch.bincount(a, minlength=M)
        S = torch.zeros_like(C).index_add_(0, a, X)
        C = S / n.clamp(min=1)[:, None].float()
        empty = n == 0
        if empty.any():          # re-seed empty clusters with random points
            C[empty] = X[torch.from_numpy(rng.choice(len(X), int(empty.sum()), replace=False)).to(device)]
    return C.cpu().numpy()


def assign(X, C, chunk=8192, return_max=False, bias=None):
    """argmax_j <x, c_j> + bias_j per row (torch tensors). C rows unit-norm => cosine for unit x;
    bias = -||c_j||^2 / 2 => nearest centroid in Euclidean distance."""
    import torch
    a, m = [], []
    for i in range(0, len(X), chunk):
        s = X[i:i + chunk] @ C.T
        if bias is not None:
            s = s + bias
        v, j = s.max(1)
        a.append(j); m.append(v)
    a = torch.cat(a)
    return (a, torch.cat(m)) if return_max else a


def make_set(name, M, rng, Wg, Wu, cov_sqrt, Xfit, Xfit_n, gstats=None, device="cpu"):
    """Raw-scale rows of one centroid set (kmeans: unit rows of spherical k-means). gstats = (mean, cov_sqrt) of the
    gate rows, for rand_gcov."""
    d = Wg.shape[1]
    if name == "rand_gcov":
        mu, s = gstats
        return mu + rng.standard_normal((M, d)) @ s
    if name == "gate":
        return Wg[rng.choice(len(Wg), M, replace=False)]
    if name == "up":
        return Wu[rng.choice(len(Wu), M, replace=False)]
    if name == "rand_iso":
        return rng.standard_normal((M, d))
    if name == "rand_cov":
        return rng.standard_normal((M, d)) @ cov_sqrt
    if name == "rand_rot":
        return Wg[rng.choice(len(Wg), M, replace=False)] @ random_rotation(d, rng)
    if name.startswith("combo_"):
        k = int(name.split("_")[1])
        G = unit_rows(Wg)
        idx = np.stack([rng.choice(len(G), k, replace=False) for _ in range(M)])
        return np.einsum("mk,mkd->md", rng.standard_normal((M, k)), G[idx])
    if name == "data_pts":
        return Xfit[rng.choice(len(Xfit), M, replace=False)]
    if name == "kmeans":
        return spherical_kmeans(Xfit_n, M, rng, device=device)
    raise ValueError(name)


# ----------------------------------------------------------------------------- metrics
def nmi(a, y):
    """Normalized mutual information (arithmetic-mean normalization) of two integer label arrays."""
    a = np.unique(a, return_inverse=True)[1]
    y = np.unique(y, return_inverse=True)[1]
    n = len(a)
    joint = np.bincount(a * (y.max() + 1) + y, minlength=(a.max() + 1) * (y.max() + 1)).reshape(a.max() + 1, -1) / n
    pa, py = joint.sum(1), joint.sum(0)
    nz = joint > 0
    mi = (joint[nz] * np.log(joint[nz] / np.outer(pa, py)[nz])).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    return float(2 * mi / (ha + hy)) if ha + hy > 0 else 0.0


def var_explained(X, a, M):
    """1 - within-cluster SS / total SS of the partition `a` of rows X (torch); cluster means from X itself."""
    import torch
    sizes = torch.bincount(a, minlength=M).to(X.dtype)      # exact for counts < 2^24
    means = torch.zeros((M, X.shape[1]), dtype=X.dtype, device=X.device).index_add_(0, a, X) / sizes.clamp(min=1)[:, None]
    ssw = ((X - means[a]) ** 2).sum()
    sst = ((X - X.mean(0)) ** 2).sum()
    return float(1 - ssw / sst)


def partition_metrics(Xn, C, labels, rng, Xc=None, rule="cos"):
    """Xn (n,d) unit rows (torch); Xc (n,d) raw rows (torch) for `ve_raw` and the euclid / ip rules.
    cos: C (M,d) unit rows, obj = mean max-cosine. euclid: C raw-scale rows, assignment by nearest centroid in Xc,
    dist = mean squared distance to it / mean ||x - mean||^2. ip: C rows at native norm, assignment by the largest
    inner product with Xc (for weight rows: the largest pre-activation). labels: {name: int array (n,), -1 = missing}."""
    import torch
    if rule == "euclid":
        a, m = assign(Xc, C, return_max=True, bias=-0.5 * (C * C).sum(1))
        d2 = (Xc * Xc).sum(1) - 2 * m
        fit = {"obj": float("nan"), "dist": float(d2.mean() / ((Xc - Xc.mean(0)) ** 2).sum(1).mean())}
    elif rule == "ip":
        a = assign(Xc, C)
        fit = {"obj": float("nan"), "dist": float("nan")}
    else:
        a, m = assign(Xn, C, return_max=True)
        fit = {"obj": float(m.mean()), "dist": float("nan")}
    M = C.shape[0]
    sizes = torch.bincount(a, minlength=M).cpu().double()   # float64 on the CPU (MPS has no float64)
    p = sizes[sizes > 0] / sizes.sum()
    a_np = a.cpu().numpy()
    res = {**fit, "ve": var_explained(Xn, a, M), "ve_raw": var_explained(Xc, a, M) if Xc is not None else float("nan"),
           "eff_k": float(torch.exp(-(p * p.log()).sum())), "used": int((sizes > 0).sum())}
    perm = rng.permutation(a_np)        # same cluster sizes, no relation to x: chance level of NMI
    for name, y in labels.items():
        ok = y >= 0
        res[f"nmi_{name}"] = nmi(a_np[ok], y[ok]) - nmi(perm[ok], y[ok])
    return res


# ----------------------------------------------------------------------------- driver
def centroid_eval(X, Wg, Wu, labels, fit_mask, Ms, reps, combo_ks=(2, 5, 20), center=True, seed=0, log=print,
                  assign_rules=("cos",), device="cpu"):
    """X (n,d) MLP inputs; fit_mask selects the fit half (eval = ~fit_mask). Returns a list of result rows.
    Wg = the neurons' input weights (gate rows; for a plain GELU MLP the first linear layer's rows). Wu = None
    (no gated value branch) drops the `up` set. assign_rules: subset of ("cos", "euclid", "ip"); "euclid" and "ip"
    reuse the sets drawn for "cos", and they and the rand_gcov set have their own rngs, so the other cos rows do not
    depend on whether they run. ip covers the weight-like sets only (IP_SETS); rows without a native norm (rand_iso,
    combo_k) get the norms of randomly drawn gate rows. device: where partitions and k-means are computed ("cuda" for
    large models; sets are still drawn with numpy on the CPU, so results match the CPU run up to float rounding)."""
    import torch
    rng = np.random.default_rng(seed)
    rng_e = np.random.default_rng(seed + 10_000)
    rng_g = np.random.default_rng(seed + 20_000)
    rng_i = np.random.default_rng(seed + 30_000)
    w, V = np.linalg.eigh(np.cov(Wg, rowvar=False))
    gstats = (Wg.mean(0), (V * np.sqrt(np.clip(w, 0, None))) @ V.T)
    g_norms = np.linalg.norm(Wg, axis=1)
    X = X.astype(np.float32)
    if center:
        X = X - X[fit_mask].mean(0)
    Xn = unit_rows(X)
    Xfit, Xfit_n = X[fit_mask], Xn[fit_mask]
    Xev = torch.from_numpy(np.ascontiguousarray(Xn[~fit_mask])).to(device)
    Xev_raw = torch.from_numpy(np.ascontiguousarray(X[~fit_mask])).to(device)
    x_norm = float(np.linalg.norm(Xfit, axis=1).mean())
    lab_ev = {k: v[~fit_mask] for k, v in labels.items()}
    w, V = np.linalg.eigh(np.cov(X[fit_mask], rowvar=False))
    cov_sqrt = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    sets = [s for s in SETS if not (s == "up" and Wu is None)] + [f"combo_{k}" for k in combo_ks] + ["rand_gcov"]
    rows = []
    for M in Ms:
        for r in range(reps.get(M, 1) if isinstance(reps, dict) else reps):
            for name in sets:
                if name in ("gate", "up") and M >= len(Wg) and r > 0:
                    continue        # the full weight set has no sampling variability
                srng = rng_g if name == "rand_gcov" else rng
                raw = make_set(name, M, srng, Wg, Wu, cov_sqrt, Xfit, Xfit_n, gstats, device=device)
                C = torch.from_numpy(unit_rows(raw).astype(np.float32)).to(device)
                if "cos" in assign_rules:
                    rows.append({"assign": "cos", "set": name, "M": M, "rep": r,
                                 **partition_metrics(Xev, C, lab_ev, srng, Xc=Xev_raw)})
                if "ip" in assign_rules and (name in IP_SETS or name.startswith("combo_")):
                    Ci = raw
                    if name == "rand_iso" or name.startswith("combo_"):
                        Ci = unit_rows(raw) * rng_i.choice(g_norms, M)[:, None]
                    Ci = torch.from_numpy(np.ascontiguousarray(Ci, dtype=np.float32)).to(device)
                    rows.append({"assign": "ip", "set": name, "M": M, "rep": r,
                                 **partition_metrics(Xev, Ci, lab_ev, rng_i, Xc=Xev_raw, rule="ip")})
                if "euclid" in assign_rules:
                    if name == "kmeans":
                        Ce = euclidean_kmeans(Xfit, M, rng_e, device=device)
                    elif name == "data_pts":
                        Ce = raw
                    else:
                        Ce = raw * (x_norm / np.linalg.norm(raw, axis=1).mean())
                    Ce = torch.from_numpy(np.ascontiguousarray(Ce, dtype=np.float32)).to(device)
                    rows.append({"assign": "euclid", "set": name, "M": M, "rep": r,
                                 **partition_metrics(Xev, Ce, lab_ev, rng_e, Xc=Xev_raw, rule="euclid")})
            log(f"[geometry] M={M} rep={r} done")
    return rows
