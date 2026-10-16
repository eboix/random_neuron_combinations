"""TopK sparse autoencoder (Gao et al. 2024) over a block's post-GELU MLP hidden activations.

Features are a learned, thresholded linear readout in the same space as the neurons and the random
combinations: z = TopK(ReLU((x/scale - b_dec) @ W_enc + b_enc)), x_hat = (z @ W_dec + b_dec) * scale.
"""
import json
import torch, torch.nn as nn, torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d, f, k):
        super().__init__()
        self.d, self.f, self.k = d, f, k
        W = torch.randn(f, d); W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W)
        self.W_enc = nn.Parameter(W.t().clone().contiguous())
        self.b_enc = nn.Parameter(torch.zeros(f))
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.register_buffer('scale', torch.tensor(1.0))

    def pre(self, x):
        return (x / self.scale - self.b_dec) @ self.W_enc + self.b_enc

    def topk(self, pre, k=None):
        top = pre.topk(k or self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, top.indices, F.relu(top.values))

    def encode(self, x):
        return self.topk(self.pre(x))

    def decode_scaled(self, z):
        return z @ self.W_dec + self.b_dec

    def decode(self, z):
        return self.decode_scaled(z) * self.scale

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def save(self, path, extra=None):
        torch.save({'config': {'d': self.d, 'f': self.f, 'k': self.k}, 'state_dict': self.state_dict(), 'extra': extra or {}}, path)

    @classmethod
    def load(cls, path, device='cpu'):
        ck = torch.load(path, map_location=device)
        m = cls(**ck['config']); m.load_state_dict(ck['state_dict']); m.extra = ck.get('extra', {})
        return m.to(device)
