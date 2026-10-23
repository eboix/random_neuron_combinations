import os

# Tests build their own temp data roots; the cluster env (slurm/env.sh) points these at the real run, which would
# redirect test output into it and let tests see each other's files.
for _k in ("COMBO_DATA_ROOT", "COMBO_PRIVATE_ROOT", "COMBO_AGENT_WORK"):
    os.environ.pop(_k, None)
