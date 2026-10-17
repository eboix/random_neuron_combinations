"""Unit generation. A *unit* is a direction a in R^{d_ff} plus offset b:  z(token) = p . a + b,
where p is the MLP hidden activation (input of down_proj). Conditions:

  neuron        a = e_j, b = 0                       (baseline; raw natural scale, not centered)
  dense_raw     random orthonormal directions (QR of Gaussian), centered, rescaled to unit variance
  dense_white   random orthonormal in the ZCA-whitened space (equal variance in every direction)
  ksparse_k     k random alive neurons, Gaussian weights on z-scored neurons, centered, unit variance
  related_k     a random seed neuron + its k-1 most correlated neurons, weights sign-aligned so the parts
                co-fire (positive control: combination of *related* neurons)

All combo units are normalised to unit variance and centered so their scale is comparable; the neuron
condition is deliberately left untouched (the essay stresses trusting a neuron's natural activation range).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io_utils import atomic_savez, read_json, write_json, write_jsonl


def zca_matrix(cov: np.ndarray, ridge: float, device: str = "auto") -> np.ndarray:
    """ZCA whitening matrix with ridge = ridge * mean eigenvalue. Uses GPU eigh if available."""
    use_torch = False
    if device != "numpy":
        try:
            import torch
            use_torch = torch.cuda.is_available()
        except ImportError:
            use_torch = False
    if use_torch:
        import torch
        c = torch.as_tensor(cov, dtype=torch.float32, device="cuda")
        ev, V = torch.linalg.eigh(c)
        ev = ev.clamp_min(0)
        ev_r = ev + ridge * ev.mean()
        W = (V * ev_r.rsqrt()) @ V.T
        return W.double().cpu().numpy()
    ev, V = np.linalg.eigh(cov.astype(np.float64))
    ev = np.clip(ev, 0, None)
    ev_r = ev + ridge * ev.mean()
    return (V / np.sqrt(ev_r)) @ V.T


def _finalize_cols(A, mean, cov, center=True):
    """Rescale columns to unit variance (under cov) and compute centering offsets."""
    A = np.asarray(A, dtype=np.float64)
    var = np.einsum("dm,dm->m", A, cov @ A)
    A = A / np.sqrt(np.maximum(var, 1e-30))
    b = -(mean @ A) if center else np.zeros(A.shape[1])
    return A, b


def make_unit_set(layer: int, mean: np.ndarray, cov: np.ndarray, ucfg, seed: int, zca: np.ndarray | None = None):
    """Returns dict(A (d,M) float32, b (M,), mean_u, std_u, meta list). Column order = condition order."""
    d = mean.shape[0]
    mean = mean.astype(np.float64)
    cov = cov.astype(np.float64)
    std = np.sqrt(np.clip(np.diag(cov), 0, None))
    alive = np.where(std > ucfg.min_std)[0]
    n = int(ucfg.per_condition)
    standardize = bool(ucfg.standardize_sparse_inputs)
    A_all, b_all, meta = [], [], []
    for ci, c in enumerate(ucfg.conditions):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(layer), ci]))
        kind, name = c["kind"], c["name"]
        k = int(c.get("k", 0)) if "k" in c else 0
        extra = [dict() for _ in range(n)]
        if kind == "neuron":
            idx = rng.choice(alive, size=n, replace=False)
            A = np.zeros((d, n)); A[idx, np.arange(n)] = 1.0
            b = np.zeros(n)
            for i, j in enumerate(idx):
                extra[i] = {"neurons": [int(j)], "weights": [1.0]}
        elif kind in ("dense_raw", "dense_white"):
            assert n <= d, "per_condition must be <= d_ff for orthonormal dense directions"
            Q, _ = np.linalg.qr(rng.standard_normal((d, n)))
            if kind == "dense_white":
                if zca is None:
                    raise ValueError("dense_white needs the ZCA matrix")
                Q = zca @ Q
            A, b = _finalize_cols(Q, mean, cov, center=True)
        elif kind == "ksparse":
            A = np.zeros((d, n))
            for i in range(n):
                idx = rng.choice(alive, size=k, replace=False)
                w = rng.standard_normal(k)
                if standardize:
                    w = w / std[idx]
                A[idx, i] = w
                extra[i] = {"neurons": [int(j) for j in idx]}
            A, b = _finalize_cols(A, mean, cov, center=True)
            for i in range(n):
                extra[i]["weights"] = [float(A[j, i]) for j in extra[i]["neurons"]]
        elif kind == "related":
            A = np.zeros((d, n))
            for i in range(n):
                j = int(rng.choice(alive))
                corr = cov[j, alive] / (std[j] * std[alive])
                corr[alive == j] = 0.0
                order = np.argsort(-np.abs(corr))[: k - 1]
                partners = alive[order]
                sg = np.sign(corr[order]); sg[sg == 0] = 1.0
                g = np.abs(rng.standard_normal(k))
                idx = np.concatenate([[j], partners])
                w = np.concatenate([[1.0], sg]) * g
                if standardize:
                    w = w / std[idx]
                A[idx, i] = w
                extra[i] = {"neurons": [int(x) for x in idx], "mean_abs_corr": float(np.abs(corr[order]).mean())}
            A, b = _finalize_cols(A, mean, cov, center=True)
            for i in range(n):
                extra[i]["weights"] = [float(A[j, i]) for j in extra[i]["neurons"]]
        else:
            raise ValueError(f"unknown condition kind {kind}")
        for i in range(n):
            meta.append({"cond": name, "kind": kind, "k": k, **extra[i]})
        A_all.append(A); b_all.append(b)
    A = np.concatenate(A_all, 1)
    b = np.concatenate(b_all)
    mean_u = mean @ A + b
    var_u = np.einsum("dm,dm->m", A, cov @ A)
    for col, m in enumerate(meta):
        m["layer"] = int(layer)
        m["col"] = col
        m["unit_id"] = f"L{layer}_{m['cond']}_{col:04d}"
    return {"A": A.astype(np.float32), "b": b.astype(np.float32), "mean_u": mean_u, "std_u": np.sqrt(var_u), "meta": meta}


def save_unit_set(us, outdir):
    outdir = Path(outdir)
    atomic_savez(outdir / "units.npz", A=us["A"], b=us["b"], mean_u=us["mean_u"], std_u=us["std_u"])
    write_jsonl(outdir / "meta.jsonl", us["meta"])


def load_unit_set(paths_or_dir, layer: int | None = None):
    from .io_utils import read_jsonl
    d = paths_or_dir.units(layer) if layer is not None else Path(paths_or_dir)
    z = np.load(d / "units.npz")
    return {"A": z["A"], "b": z["b"], "mean_u": z["mean_u"], "std_u": z["std_u"], "meta": read_jsonl(d / "meta.jsonl")}


def make_blind_key(all_unit_ids, seed: int):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 99991]))
    perm = rng.permutation(len(all_unit_ids))
    return {f"u{rank:05d}": all_unit_ids[i] for rank, i in enumerate(perm)}


def run_make_units(cfg, paths, log=print):
    layers = [int(l) for l in cfg.model.layers]
    need_zca = any(c["kind"] == "dense_white" for c in cfg.units.conditions)
    all_ids = []
    for L in layers:
        mean = np.load(paths.stats(L) / "mean.npy")
        cov = np.load(paths.stats(L) / "cov.npy")
        zca = zca_matrix(cov, float(cfg.units.whiten_ridge)) if need_zca else None
        us = make_unit_set(L, mean, cov, cfg.units, cfg.seed, zca)
        save_unit_set(us, paths.units(L))
        all_ids += [m["unit_id"] for m in us["meta"]]
        log(f"[units] layer {L}: {us['A'].shape[1]} units")
    if not paths.blind_key.exists():
        write_json(paths.blind_key, make_blind_key(all_ids, cfg.seed), indent=0)
        log(f"[units] wrote blind key -> {paths.blind_key}  (keep OUT of agent workspaces)")
