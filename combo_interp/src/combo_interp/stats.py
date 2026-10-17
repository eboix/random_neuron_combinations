"""Pass 1: streaming mean / covariance of the MLP hidden activation at the chosen layers."""
from __future__ import annotations

import numpy as np

from .io_utils import write_json


def cov_from_sums(n: int, s: np.ndarray, xtx: np.ndarray):
    """Population mean / covariance from raw sums (float64)."""
    mean = s / n
    cov = xtx / n - np.outer(mean, mean)
    cov = (cov + cov.T) / 2
    return mean, cov


class StreamingCov:
    def __init__(self, d, device="cuda"):
        import torch
        self.n = 0
        self.s = torch.zeros(d, dtype=torch.float64, device=device)
        self.xtx = torch.zeros(d, d, dtype=torch.float64, device=device)

    def update(self, p, chunk=4096):
        import torch
        for i in range(0, p.shape[0], chunk):
            x = p[i:i + chunk].float()
            self.n += x.shape[0]
            self.s += x.sum(0, dtype=torch.float64)
            self.xtx += (x.T @ x).double()

    def finalize(self):
        return cov_from_sums(self.n, self.s.cpu().numpy(), self.xtx.cpu().numpy())


def run_stats(cfg, paths, model=None, tok=None, log=print):
    import torch
    from .corpus import CorpusStore
    from .hooks import MLPCapture
    from .model_utils import load_model, mlp_width

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model, tok = load_model(cfg, device)
    device = next(model.parameters()).device
    store = CorpusStore(paths.corpus("train"))
    layers = [int(l) for l in cfg.model.layers]
    d = mlp_width(model)
    acc = {L: StreamingCov(d, device) for L in layers}
    skip = int(cfg.stats.skip_first_tokens)
    seen, nb = 0, 0
    with MLPCapture(model, layers, stop_after_last=bool(cfg.get("runtime", {}).get("early_stop", True))) as cap:
        for batch in store.iter_batches(int(cfg.stats.max_batch_tokens), int(cfg.stats.n_tokens)):
            ids = torch.from_numpy(batch["input_ids"]).to(device)
            lengths = torch.from_numpy(batch["lengths"]).to(device)
            ar = torch.arange(ids.shape[1], device=device)
            attn = ar[None, :] < lengths[:, None]
            valid = attn & (ar[None, :] >= skip)
            cap.run(ids, attn.long())
            for L in layers:
                acc[L].update(cap.p[L][valid])
            seen += int(valid.sum())
            nb += 1
            if nb % 50 == 0:
                log(f"[stats] {seen} tokens")
    for L in layers:
        mean, cov = acc[L].finalize()
        out = paths.stats(L)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "mean.npy", mean)
        np.save(out / "cov.npy", cov.astype(np.float32))
        write_json(out / "meta.json", {"n_tokens": int(acc[L].n), "d_ff": int(d), "layer": L})
        log(f"[stats] layer {L}: n={acc[L].n} d_ff={d}")
