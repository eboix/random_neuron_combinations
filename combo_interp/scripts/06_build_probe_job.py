#!/usr/bin/env python3
"""CPU: write the probe-input file (the writer's max / predicted / near-miss / distant texts + null and random
baselines) for the single GPU probe round. Run after the write stage."""
from _common import setup

cfg, paths, args = setup(__doc__)
from transformers import AutoTokenizer  # noqa: E402
from combo_interp.scoring import build_probe_job  # noqa: E402

build_probe_job(cfg, paths, AutoTokenizer.from_pretrained(cfg.model.name))
