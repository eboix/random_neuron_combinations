#!/usr/bin/env python3
"""GPU: per-unit firing thresholds from random text windows (for scoring.threshold=random_quantile). Minutes."""
from _common import setup

cfg, paths, args = setup(__doc__)
from combo_interp.calibrate import run_calibration  # noqa: E402

run_calibration(cfg, paths)
