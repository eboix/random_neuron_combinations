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

import os, sys  # noqa: E402
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)   # abandoned `datasets` streaming iterators keep worker threads alive and the interpreter hangs at exit
