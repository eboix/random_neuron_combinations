#!/usr/bin/env python3
"""GPU: score texts on units (the cluster side of the probe rounds)."""
from _common import setup


def extra(ap):
    ap.add_argument("--job-file", required=True)
    ap.add_argument("--out", required=True)


cfg, paths, args = setup(__doc__, extra)
from combo_interp.probe import run_probe  # noqa: E402

run_probe(cfg, paths, args.job_file, args.out)
