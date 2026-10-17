# Instructions for an agent working in this repo

* You have **no access to the GPU cluster** and must never try to reach it. For anything that needs a GPU, create a job with
  `python scripts/make_job.py ...`, then STOP and tell the human which `jobs/pending/NNN_*/COMMANDS.txt` to run. Do not assume
  a job ran until its outputs exist under `jobs/done/` or `data/`.
* Never read or print anything in `private/` (blinding key, ground truth) and never copy it into an agent workspace. Never
  look inside `data/agent_work/*` of units you are not currently handling.
* Never fabricate or "fill in" results; if a stage output is missing, say so.
* Run `pytest -q` before and after code changes. `tests/test_torch_tiny.py` is the safety net for the GPU code — keep it passing.
* Pre-registered equivalence margins (`analysis.equiv_margin`) must not be changed after agent results exist.
* Prefer small, reviewable changes; every config knob lives in `configs/default.yaml`.
