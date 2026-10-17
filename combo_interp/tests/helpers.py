import tempfile
import unittest
from pathlib import Path

from combo_interp.config import Paths, load_config

ROOT = Path(__file__).resolve().parents[1]


def smoke_cfg(tmp, extra=None):
    ov = [f"paths.data_root={tmp}/data", f"paths.private_root={tmp}/private", f"paths.jobs_dir={tmp}/jobs"] + (extra or [])
    cfg = load_config(ROOT / "configs" / "smoke.yaml", ov)
    return cfg, Paths(cfg)


def need(module):
    try:
        __import__(module)
    except ImportError:
        raise unittest.SkipTest(f"{module} not installed")
