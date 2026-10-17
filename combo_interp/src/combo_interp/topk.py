"""Streaming per-unit top-K over tokens (GPU). Scores are 'larger is better' (pass -acts for the negative pole).

State per unit m: vals[m, K] (descending), gid[m, K]. Optional payloads (e.g. activation windows) are carried
along for units in `pay_cols` and have shape (len(pay_cols), K, *extra).
"""
from __future__ import annotations


class TopKTracker:
    def __init__(self, M, K, device, pay_cols=None, pay_shapes=None):
        import torch
        self.K = K
        self.device = device
        self.vals = torch.full((M, K), float("-inf"), device=device)
        self.gid = torch.full((M, K), -1, dtype=torch.long, device=device)
        self.pay_cols = pay_cols
        self.pay = {}
        if pay_cols is not None:
            for name, shape in (pay_shapes or {}).items():
                self.pay[name] = torch.full((len(pay_cols), K, *shape), float("nan"), device=device)

    def update(self, scoresT, gids, payload_fn=None):
        """scoresT: (M, N) float32; gids: (N,) int64; payload_fn(r_pay (Mp,k)) -> {name: (Mp,k,*extra)}."""
        import torch
        M, N = scoresT.shape
        if N == 0:
            return
        k = min(self.K, N)
        v, r = torch.topk(scoresT, k, dim=1)                 # (M,k) candidates from this batch
        g = gids[r]
        cat_v = torch.cat([self.vals, v], 1)                 # (M, K+k)
        cat_g = torch.cat([self.gid, g], 1)
        sel_v, sel_i = torch.topk(cat_v, self.K, dim=1)      # survivors, indices into the K+k columns
        self.vals = sel_v
        self.gid = torch.gather(cat_g, 1, sel_i)
        if self.pay_cols is not None and payload_fn is not None:
            new = payload_fn(r[self.pay_cols])
            sel_ip = sel_i[self.pay_cols]                    # (Mp, K)
            for name, t_new in new.items():
                cat = torch.cat([self.pay[name], t_new.to(self.pay[name].dtype)], 1)   # (Mp, K+k, *extra)
                idx = sel_ip.reshape(*sel_ip.shape, *([1] * (cat.dim() - 2))).expand(-1, -1, *cat.shape[2:])
                self.pay[name] = torch.gather(cat, 1, idx)

    # checkpointing -----------------------------------------------------------------------------
    def state(self):
        return {"vals": self.vals.cpu(), "gid": self.gid.cpu(), "pay": {k: v.cpu() for k, v in self.pay.items()}}

    def load(self, st):
        self.vals = st["vals"].to(self.device)
        self.gid = st["gid"].to(self.device)
        self.pay = {k: v.to(self.device) for k, v in st["pay"].items()}
