from __future__ import annotations


def load_model(cfg, device="cuda"):
    """Load HF causal LM + tokenizer. Works for any model whose blocks look like Llama/Qwen:
    model.model.layers[i].mlp.{gate_proj, up_proj, down_proj}."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import transformers
    m = cfg.model
    tok = AutoTokenizer.from_pretrained(m.name)
    major_minor = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    dtype_kw = "dtype" if major_minor >= (4, 56) else "torch_dtype"     # renamed in 4.56; old name deprecated in 5.x
    model = AutoModelForCausalLM.from_pretrained(
        m.name, **{dtype_kw: getattr(torch, m.dtype)},
        attn_implementation=m.get("attn_implementation", "sdpa"))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def get_blocks(model):
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base.layers
    raise ValueError("unsupported architecture: expected model.model.layers")


def mlp_width(model) -> int:
    return get_blocks(model)[0].mlp.down_proj.in_features


def pad_id_of(tok) -> int:
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
