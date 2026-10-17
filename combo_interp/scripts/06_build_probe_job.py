#!/usr/bin/env python3
"""CPU: write the probe-input file for GPU round 1 (own/near/distant/null/random texts) or round 2 (blind-max texts)."""
from _common import setup


def extra(ap):
    ap.add_argument("--round", type=int, choices=[1, 2], required=True)


cfg, paths, args = setup(__doc__, extra)
tok = None
if args.round == 1:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.name)
from combo_interp.scoring import build_probe_job  # noqa: E402

build_probe_job(cfg, paths, args.round, tok)
