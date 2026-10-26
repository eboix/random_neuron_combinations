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
  data_pts     random tokens' x (fit half)                 data-aligned baseline (k-means++-style seeding)
  kmeans       spherical k-means on the fit half           upper bound

Data and centroids fit on documents with even seq_id, everything is evaluated on odd ones.
"""
from __future__ import annotations

import re

import numpy as np

SETS = ["gate", "up", "rand_iso", "rand_cov", "rand_rot", "data_pts", "kmeans"]


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


def spherical_kmeans(Xn, M, rng, iters=20, chunk=8192):
    """Xn: unit rows. Returns (M, d) unit centroids (k-means++-lite: random data init)."""
    import torch
    X = torch.from_numpy(np.ascontiguousarray(Xn, dtype=np.float32))
    C = X[torch.from_numpy(rng.choice(len(X), M, replace=False))].clone()
    for _ in range(iters):
        a = assign(X, C, chunk)
        S = torch.zeros_like(C).index_add_(0, a, X)
        empty = S.norm(dim=1) < 1e-8
        if empty.any():          # re-seed empty clusters with random points
            S[empty] = X[torch.from_numpy(rng.choice(len(X), int(empty.sum()), replace=False))]
        C = S / S.norm(dim=1, keepdim=True)
    return C.numpy()


def assign(X, C, chunk=8192, return_max=False):
    """argmax_j <x, c_j> per row (torch tensors). C rows unit-norm => cosine for unit x."""
    import torch
    a, m = [], []
    for i in range(0, len(X), chunk):
        s = X[i:i + chunk] @ C.T
        v, j = s.max(1)
        a.append(j); m.append(v)
    a = torch.cat(a)
    return (a, torch.cat(m)) if return_max else a


def make_set(name, M, rng, Wg, Wu, cov_sqrt, Xfit_n):
    d = Wg.shape[1]
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
        return Xfit_n[rng.choice(len(Xfit_n), M, replace=False)]
    if name == "kmeans":
        return spherical_kmeans(Xfit_n, M, rng)
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


def partition_metrics(Xn, C, labels, rng):
    """Xn (n,d) unit rows (torch), C (M,d) unit rows (torch). labels: {name: int array (n,), -1 = missing}."""
    import torch
    a, m = assign(Xn, C, return_max=True)
    M = C.shape[0]
    sizes = torch.bincount(a, minlength=M).double()
    p = sizes[sizes > 0] / sizes.sum()
    means = torch.zeros_like(C).index_add_(0, a, Xn) / sizes.clamp(min=1)[:, None].float()
    ssw = ((Xn - means[a]) ** 2).sum()
    sst = ((Xn - Xn.mean(0)) ** 2).sum()
    a_np = a.numpy()
    res = {"obj": float(m.mean()), "ve": float(1 - ssw / sst),
           "eff_k": float(torch.exp(-(p * p.log()).sum())), "used": int((sizes > 0).sum())}
    perm = rng.permutation(a_np)        # same cluster sizes, no relation to x: chance level of NMI
    for name, y in labels.items():
        ok = y >= 0
        res[f"nmi_{name}"] = nmi(a_np[ok], y[ok]) - nmi(perm[ok], y[ok])
    return res


# ----------------------------------------------------------------------------- driver
def centroid_eval(X, Wg, Wu, labels, fit_mask, Ms, reps, combo_ks=(2, 5, 20), center=True, seed=0, log=print):
    """X (n,d) MLP inputs; fit_mask selects the fit half (eval = ~fit_mask). Returns a list of result rows."""
    import torch
    rng = np.random.default_rng(seed)
    X = X.astype(np.float32)
    if center:
        X = X - X[fit_mask].mean(0)
    Xn = unit_rows(X)
    Xfit_n, Xev = Xn[fit_mask], torch.from_numpy(np.ascontiguousarray(Xn[~fit_mask]))
    lab_ev = {k: v[~fit_mask] for k, v in labels.items()}
    w, V = np.linalg.eigh(np.cov(X[fit_mask], rowvar=False))
    cov_sqrt = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    sets = SETS + [f"combo_{k}" for k in combo_ks]
    rows = []
    for M in Ms:
        for r in range(reps.get(M, 1) if isinstance(reps, dict) else reps):
            for name in sets:
                if name in ("gate", "up") and M >= len(Wg) and r > 0:
                    continue        # the full weight set has no sampling variability
                C = torch.from_numpy(unit_rows(make_set(name, M, rng, Wg, Wu, cov_sqrt, Xfit_n)).astype(np.float32))
                rows.append({"set": name, "M": M, "rep": r, **partition_metrics(Xev, C, lab_ev, rng)})
            log(f"[geometry] M={M} rep={r} done")
    return rows
