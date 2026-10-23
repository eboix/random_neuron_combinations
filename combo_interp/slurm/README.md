# Running the study on a Slurm cluster

GPU and CPU stages run on the cluster (`slurm/submit.sh`); the Claude agents run on a laptop where the `claude` CLI is
signed in. Only the blinded agent views go to the laptop, so the blinding key and ground truth (`private/`) never sit
next to an agent workspace.

## One-time setup (cluster login node)
```bash
git clone <repo> ~/projects/RandomNeuronCombinations && cd ~/projects/RandomNeuronCombinations && git checkout rohan
cd combo_interp && $EDITOR slurm/env.sh                    # account, partition, GPU type, paths
python3 -m venv ~/venvs/combo && ~/venvs/combo/bin/pip install -e ".[gpu,dev]"   # python >= 3.10, not an alpha
bash slurm/submit.sh test                                  # CPU job: full test suite incl. the torch checks
```

## Phases
| # | where | command | what |
|---|---|---|---|
| 1 | cluster | `bash slurm/submit.sh data` | corpus + model download (CPU, needs internet) -> stats/units -> train cache, holdout cache, calibration -> export + pilot. Chained with `afterok` |
| 2 | laptop | sync down, `05_run_agents.py --stage explain`, sync up | explainer agents |
| 3 | cluster | `bash slurm/submit.sh probe1` | score the explainers' predicted / near-miss / distant texts |
| 4 | laptop | `--stage blindmax`, `--stage simulate`, sync up | the two explanation-only agents |
| 5 | cluster | `bash slurm/submit.sh probe2`, then `bash slurm/submit.sh score` | blind-max texts, then scores + equivalence tests |

If `cache_train` hits its time limit it leaves a checkpoint; `bash slurm/submit.sh cache_train` resumes it, then
`bash slurm/submit.sh export`.

## Laptop side
```bash
CL=you@cluster.host; R=combo_runs/qwen3_8b_v1/data; L=~/combo_runs/qwen3_8b_v1/data; cd combo_interp
rsync -av $CL:$R/agent_cache $L/                                    # views only; private/ stays on the cluster
COMBO_DATA_ROOT=$L python scripts/05_run_agents.py --config configs/cluster_qwen3_8b.yaml --stage explain --limit 5   # check cost first
COMBO_DATA_ROOT=$L python scripts/05_run_agents.py --config configs/cluster_qwen3_8b.yaml --stage explain
rsync -av $L/agent_results $CL:$R/                                  # then: submit.sh probe1 on the cluster
COMBO_DATA_ROOT=$L python scripts/05_run_agents.py --config configs/cluster_qwen3_8b.yaml --stage blindmax
COMBO_DATA_ROOT=$L python scripts/05_run_agents.py --config configs/cluster_qwen3_8b.yaml --stage simulate
rsync -av $L/agent_results $CL:$R/                                  # then: submit.sh probe2, submit.sh score
```
Agent stages are resumable (finished units are skipped). Keep the laptop awake (`caffeinate -i ...`).

`scripts/make_job.py` (per-job folders + rsync commands for clusters you cannot work on directly) still works.
