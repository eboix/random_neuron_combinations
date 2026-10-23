# Running the study on a Slurm cluster

GPU and CPU stages run on the cluster (`slurm/submit.sh`); the Claude agents run on a laptop where the `claude` CLI is
signed in. Only the blinded agent views go to the laptop, so the blinding key and ground truth (`private/`) never sit
next to an agent workspace.

## One-time setup (cluster login node)
Defaults in `slurm/env.sh` target Penn PARCC's Betty (`ssh <PennKey>@slurm_login.parcc.upenn.edu`).
```bash
srun --ntasks=1 --cpus-per-task=1 --mem=1G hostname        # fails -> PI must add you to a ColdFront project
git clone -b rohan https://github.com/eboix/random_neuron_combinations.git ~/projects/RandomNeuronCombinations
cd ~/projects/RandomNeuronCombinations/combo_interp && $EDITOR slurm/env.local.sh   # your paths/account (see env.sh)
module load anaconda3 && conda create -y -p ~/envs/py311 python=3.11          # Betty: CUDA 12.8 only
~/envs/py311/bin/python -m venv ~/venvs/combo
~/venvs/combo/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
~/venvs/combo/bin/pip install -e ".[gpu,dev]"
bash slurm/submit.sh test                                  # CPU job: full test suite incl. the torch checks
```
CPU stages go to `CI_CPU_PARTITION`, GPU stages to `CI_GPU_PARTITION` (default `dgx-b200`) with `--gpus=$CI_GPUS`.
Betty's CLI filter derives memory from CPUs and fixes CPUs per GPU per partition, so the scripts never pass `--mem`;
if you switch GPU partition, set `CI_GPU_CPUS` to match (table in `env.sh`).

## Phases
| # | where | command | what |
|---|---|---|---|
| 1 | cluster | `bash slurm/submit.sh data` | corpus + model download (CPU, needs internet) -> stats/units -> train cache, holdout cache, calibration -> export + pilot. Chained with `afterok` |
| 2 | laptop (+ cluster) | sync down; `05_run_agents.py --stage explain`, then `--stage write` and `--stage simulate`; sync up | all agents. If `agents.query.enabled`: first `bash slurm/submit.sh serve` and an SSH tunnel (below) |
| 3 | cluster | `bash slurm/submit.sh probe`, then `bash slurm/submit.sh score` | score the writer's texts, then scores + equivalence tests |

If `cache_train` hits its time limit it leaves a checkpoint; `bash slurm/submit.sh cache_train` resumes it, then
`bash slurm/submit.sh export`.

## Laptop side
```bash
CL=<PennKey>@slurm_login.parcc.upenn.edu; R=combo_runs/qwen3_8b_v1/data; L=~/combo_runs/qwen3_8b_v1/data; cd combo_interp
rsync -av $CL:$R/agent_cache $L/                                    # views only; private/ stays on the cluster
COMBO_DATA_ROOT=$L python scripts/05_run_agents.py --config configs/cluster_qwen3_8b.yaml --stage explain --limit 5   # check cost first
COMBO_DATA_ROOT=$L python scripts/05_run_agents.py --config configs/cluster_qwen3_8b.yaml --stage explain
COMBO_DATA_ROOT=$L python scripts/05_run_agents.py --config configs/cluster_qwen3_8b.yaml --stage write
COMBO_DATA_ROOT=$L python scripts/05_run_agents.py --config configs/cluster_qwen3_8b.yaml --stage simulate
rsync -av $L/agent_results $CL:$R/                                  # then: submit.sh probe, submit.sh score
```

### Model queries for the explainer (`agents.query.enabled: true`)
The query service needs the model and `private/`, so it runs on the cluster:
```bash
bash slurm/submit.sh serve                       # cluster; its log prints the node, e.g. node12
ssh -N -L 8765:node12:8765 you@cluster.host      # laptop, keep open while the explain stage runs
curl -s http://127.0.0.1:8765/health             # laptop: {"ok": true, ...}
```
The explain stage refuses to start if the service does not answer. Cancel the serve job (`scancel`) afterwards. The
query log (`data/query_log/`) stays on the cluster and is counted in the scores (`n_queries`).

Agent stages are resumable (finished units are skipped). Keep the laptop awake (`caffeinate -i ...`).

`scripts/make_job.py` (per-job folders + rsync commands for clusters you cannot work on directly) still works.
