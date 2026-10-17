import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from combo_interp.config import Paths, load_config  # noqa: E402


def setup(desc, extra=None):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override a config value, e.g. --set model.name=Qwen/Qwen3-4B")
    if extra:
        extra(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    return cfg, Paths(cfg), args
