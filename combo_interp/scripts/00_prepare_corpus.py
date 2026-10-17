#!/usr/bin/env python3
"""Tokenize a mixture of web / code / multilingual text into document-aligned shards (needs internet + `datasets`).
Run on a machine with internet (laptop or login node), then rsync data/corpus to the cluster."""
from _common import setup

cfg, paths, args = setup(__doc__)
from transformers import AutoTokenizer  # noqa: E402
from combo_interp.corpus import build_corpus  # noqa: E402

tok = AutoTokenizer.from_pretrained(cfg.model.name)
for split in ("train", "holdout"):
    build_corpus(cfg, paths, split, tok)
