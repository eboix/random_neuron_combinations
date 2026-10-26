"""Capture the MLP hidden activation p = act(gate(x)) * up(x)  (= the input of down_proj).

For SwiGLU models this is *the neuron activation* in the sense of the essay. We optionally also capture the
gate / value pre-activations so sign regimes (the "double negative" echo) can be inspected for real neurons.
The forward pass is aborted right after the deepest requested layer (saves compute, never computes logits).
"""
from __future__ import annotations

from .model_utils import get_blocks


class StopForward(Exception):
    pass


class MLPCapture:
    def __init__(self, model, layers, capture_gate_up: bool = False, stop_after_last: bool = True,
                 capture_input: bool = False):
        self.model = model
        self.layers = sorted(set(int(l) for l in layers))
        self.capture_gate_up = capture_gate_up
        self.capture_input = capture_input      # x = the MLP input (post-norm), so gate_proj(x) is the gate pre-activation
        self.stop = stop_after_last
        self.p, self.g, self.u, self.x = {}, {}, {}, {}
        self._handles = []

    def __enter__(self):
        blocks = get_blocks(self.model)
        last = self.layers[-1]
        for L in self.layers:
            mlp = blocks[L].mlp
            self._handles.append(mlp.down_proj.register_forward_pre_hook(self._pre(L, self.stop and L == last)))
            if self.capture_gate_up:
                self._handles.append(mlp.gate_proj.register_forward_hook(self._out(self.g, L)))
                self._handles.append(mlp.up_proj.register_forward_hook(self._out(self.u, L)))
            if self.capture_input:
                self._handles.append(mlp.gate_proj.register_forward_pre_hook(self._in(self.x, L)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def _pre(self, L, stop):
        def hook(module, args):
            self.p[L] = args[0]
            if stop:
                raise StopForward()
        return hook

    def _in(self, store, L):
        def hook(module, args):
            store[L] = args[0]
        return hook

    def _out(self, store, L):
        def hook(module, args, output):
            store[L] = output
        return hook

    def run(self, input_ids, attention_mask=None):
        import torch
        self.p.clear(); self.g.clear(); self.u.clear(); self.x.clear()
        base = getattr(self.model, "model", self.model)   # skip lm_head entirely
        with torch.no_grad():
            try:
                base(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            except StopForward:
                pass
