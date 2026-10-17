#!/usr/bin/env python3
"""CPU: run an agent stage over the selected units. Stages: explain -> (GPU probe round 1) -> blindmax, simulate."""
from _common import setup


def extra(ap):
    ap.add_argument("--stage", choices=["explain", "blindmax", "simulate"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="only the first N selected units (use 3-5 for a trial)")
    ap.add_argument("--only", nargs="*", default=None, help="specific blind ids")
    ap.add_argument("--dry-run", action="store_true", help="only create the workspaces")


cfg, paths, args = setup(__doc__, extra)
from combo_interp.agents.run import run_stage  # noqa: E402

run_stage(cfg, paths, args.stage, limit=args.limit, only=args.only, dry_run=args.dry_run)
