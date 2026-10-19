"""Per-unit "fires strongly" thresholds calibrated on random text (GPU job, minutes).

A probe text counts as firing when its max over tokens (positions >= skip_first) is high. A fixed fraction of the
corpus max is not comparable across units: a sparse neuron rarely reaches 30% of its max, a Gaussian-like dense
combo reaches it on most texts. Here random windows of corpus text are fed as standalone texts (exactly like probe
texts), giving for every unit and every length n the distribution of max over the first n tokens. Quantile q of
that distribution is the unit's bar for texts of length n: a random text of that length crosses it with
probability 1 - q, for every unit alike.

Output per layer: data/calibration/L{layer}.npz with thr_pos / thr_neg of shape (Q, T, M) (NaN for n <= skip).
"""
from __future__ import annotations

import numpy as np

from .io_utils import atomic_savez


def sample_windows(store, n_windows: int, window: int, rng) -> np.ndarray:
    """(n, window) token ids: random start in random documents at least `window` long."""
    long_docs = [s for s in range(store.n_seqs) if store.seq_length(s) >= window]
    if not long_docs:
        raise ValueError(f"no document with >= {window} tokens in {store.root}")
    out = []
    for _ in range(n_windows):
        s = int(long_docs[rng.integers(len(long_docs))])
        start = int(rng.integers(store.seq_length(s) - window + 1))
        out.append(store.get_tokens(s, start, start + window))
    return np.stack(out)


def window_thresholds(acts: np.ndarray, skip: int, quantiles) -> np.ndarray:
    """acts (N, T, M) per-token activations of N windows -> (Q, T, M): entry [q, n-1, m] = quantile q over windows of
    max(acts[:, skip:n, m]). Pass -acts for the negative pole."""
    a = np.asarray(acts, dtype=np.float32)
    q = np.asarray(quantiles, dtype=float)
    thr = np.full((len(q), a.shape[1], a.shape[2]), np.nan, dtype=np.float32)
    cmax = np.maximum.accumulate(a[:, skip:], axis=1)                  # max over tokens skip..n-1
    thr[:, skip:] = np.quantile(cmax, q, axis=0)
    return thr


def save_calibration(cfg, paths, layer: int, acts: np.ndarray):
    C, skip = cfg.calibration, int(cfg.cache.skip_first_tokens)
    q = [float(x) for x in C.quantiles]
    d = paths.data / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    atomic_savez(d / f"L{layer}.npz", quantiles=np.array(q), skip=np.array(skip), n_windows=np.array(acts.shape[0]),
                 thr_pos=window_thresholds(acts, skip, q), thr_neg=window_thresholds(-acts, skip, q))


def load_calibration(paths, layer: int):
    z = np.load(paths.data / "calibration" / f"L{layer}.npz")
    return {k: z[k] for k in z.files}


def run_calibration(cfg, paths, model=None, tok=None, log=print):
    import torch
    from .corpus import CorpusStore
    from .hooks import MLPCapture
    from .model_utils import load_model
    from .units import load_unit_set

    C = cfg.calibration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model, tok = load_model(cfg, device)
    device = next(model.parameters()).device
    layers = [int(l) for l in cfg.model.layers]
    W = int(C.window)
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), 31337]))
    ids_all = sample_windows(CorpusStore(paths.corpus(C.split)), int(C.n_windows), W, rng)
    us = {L: load_unit_set(paths, L) for L in layers}
    A = {L: torch.as_tensor(us[L]["A"], device=device, dtype=torch.float32) for L in layers}
    b = {L: torch.as_tensor(us[L]["b"], device=device, dtype=torch.float32) for L in layers}
    acts = {L: [] for L in layers}
    B = max(1, int(C.batch_tokens) // W)
    with MLPCapture(model, layers, stop_after_last=bool(cfg.get("runtime", {}).get("early_stop", True))) as cap:
        for i in range(0, len(ids_all), B):
            ids = torch.as_tensor(ids_all[i:i + B], device=device, dtype=torch.long)
            cap.run(ids, torch.ones_like(ids))
            for L in layers:
                acts[L].append((cap.p[L].float() @ A[L] + b[L]).cpu().numpy())
    for L in layers:
        save_calibration(cfg, paths, L, np.concatenate(acts[L]))
        log(f"[calibrate] layer {L}: {len(ids_all)} windows of {W} tokens")
