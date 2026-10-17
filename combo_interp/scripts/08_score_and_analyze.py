#!/usr/bin/env python3
"""CPU: compute per-unit scores from probe outputs + agent results, then the equivalence analysis."""
from _common import setup


def extra(ap):
    ap.add_argument("--probe", nargs="+", required=True, help="probe output files (rounds 1 and 2)")


cfg, paths, args = setup(__doc__, extra)
from combo_interp.analyze import run_analysis  # noqa: E402
from combo_interp.scoring import compute_scores  # noqa: E402

compute_scores(cfg, paths, args.probe)
run_analysis(cfg, paths)
