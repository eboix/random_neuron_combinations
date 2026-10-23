#!/usr/bin/env python3
"""Pass 2 (GPU): stream the corpus, project onto all units, keep top-k / windows / uniform sample. Resumable.
SIGUSR1 (sent by slurm/job.sbatch shortly before the time limit) checkpoints and exits 75 so the job requeues."""
import signal
import sys
import threading

from _common import setup


def extra(ap):
    ap.add_argument("--split", choices=["train", "holdout"], required=True)


cfg, paths, args = setup(__doc__, extra)
from combo_interp.cache import run_cache  # noqa: E402

stop = threading.Event()
signal.signal(signal.SIGUSR1, lambda *_: stop.set())
if run_cache(cfg, paths, args.split, should_stop=stop.is_set) == "stopped":
    sys.exit(75)
