#!/usr/bin/env python3
"""Pass 1 (GPU): activation mean/covariance per layer, then generate all units (neurons + combinations)."""
from _common import setup


def extra(ap):
    ap.add_argument("--stage", choices=["stats", "units", "both"], default="both")


cfg, paths, args = setup(__doc__, extra)
from combo_interp.stats import run_stats  # noqa: E402
from combo_interp.units import run_make_units  # noqa: E402

if args.stage in ("stats", "both"):
    run_stats(cfg, paths)
if args.stage in ("units", "both"):
    run_make_units(cfg, paths)
