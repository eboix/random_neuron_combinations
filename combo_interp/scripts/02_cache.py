#!/usr/bin/env python3
"""Pass 2 (GPU): stream the corpus, project onto all units, keep top-k / windows / uniform sample. Resumable."""
from _common import setup


def extra(ap):
    ap.add_argument("--split", choices=["train", "holdout"], required=True)


cfg, paths, args = setup(__doc__, extra)
from combo_interp.cache import run_cache  # noqa: E402

run_cache(cfg, paths, args.split)
