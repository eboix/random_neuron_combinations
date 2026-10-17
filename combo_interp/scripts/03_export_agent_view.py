#!/usr/bin/env python3
"""CPU: build blinded per-unit agent views + private ground truth from the caches (needs tokenizer + corpus)."""
from _common import setup

cfg, paths, args = setup(__doc__)
from transformers import AutoTokenizer  # noqa: E402
from combo_interp.export import build_agent_cache  # noqa: E402

build_agent_cache(cfg, paths, AutoTokenizer.from_pretrained(cfg.model.name))
