"""Ablate / steer / clamp a unit (a, b): z = p.a + b, where p is the MLP hidden activation.
For a neuron (a = e_j, b = 0) ablation reduces to zeroing neuron j. Provided for follow-up experiments
(steering tests); the main pipeline only needs read-only probing.
"""
from __future__ import annotations

from .model_utils import get_blocks


def make_unit_hook(a, b, mode: str, alpha: float = 0.0):
    import torch
    a = a.float()
    a2 = (a * a).sum()

    def hook(module, args):
        p = args[0]
        pf = p.float()
        z = pf @ a + b
        if mode == "ablate":
            delta = -z
        elif mode == "steer":
            delta = torch.full_like(z, alpha)
        elif mode == "clamp":
            delta = alpha - z
        else:
            raise ValueError(mode)
        return ((pf + (delta / a2).unsqueeze(-1) * a).to(p.dtype),)

    return hook


def apply_unit_intervention(model, layer: int, a, b, mode: str, alpha: float = 0.0):
    """Returns a hook handle; call handle.remove() to undo."""
    mlp = get_blocks(model)[layer].mlp
    return mlp.down_proj.register_forward_pre_hook(make_unit_hook(a.to(next(model.parameters()).device), b, mode, alpha))
