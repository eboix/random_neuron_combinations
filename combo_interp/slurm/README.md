# Running the study on a Slurm cluster

Everything runs on the cluster through `slurm/submit.sh`, including the Claude agents (Slurm jobs that call the
Anthropic API). Running the agents on a laptop instead still works (end of this file).

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
curl -fsSL https://claude.ai/install.sh | bash             # agents: Claude Code CLI in ~/.local/bin
claude setup-token                                         # save the printed token to ~/.claude_token, then:
chmod 600 ~/.claude_token
```
CPU stages go to `CI_CPU_PARTITION`, GPU stages to `CI_GPU_PARTITION` (default `b200-mig45`, 45 GB) with `--gpus=$CI_GPUS`.
Betty's CLI filter derives memory from CPUs and fixes CPUs per GPU per partition, so the scripts never pass `--mem`;
if you switch GPU partition, set `CI_GPU_CPUS` to match (table in `env.sh`).

## Phases
| # | where | command | what |
|---|---|---|---|
| 1 | cluster | `bash slurm/submit.sh data` | corpus + model download (CPU, needs internet) -> stats/units -> train cache, holdout cache, calibration -> export + pilot. Chained with `afterok` |
| 2 | cluster | `bash slurm/submit.sh agents` | explain, then write and simulate. explain runs on a GPU with the model query service in the same job if `agents.query.enabled`, else on CPU. Resumable: after a timeout or a usage limit, `bash slurm/submit.sh explain` (or write / simulate) again |
| 3 | cluster | `bash slurm/submit.sh probe`, then `bash slurm/submit.sh score` | score the writer's texts, then scores + equivalence tests |

`cache_train` / `cache_holdout` handle their time limit themselves: 20 min before it, Slurm signals the job, the stage
writes a checkpoint, and the job requeues under the same job ID (log: `[job] checkpointed; requeueing ...`), so `export`
keeps waiting on it. At most `CI_MAX_REQUEUES` (5) times; after that, or if requeueing is refused, resume by hand with
`bash slurm/submit.sh cache_train`, then `scancel` the stuck export and `bash slurm/submit.sh export`.

## Agents on the cluster: what keeps them blind
Agents get only `Read`, `Write`, `Glob`, `Grep` (no shell, no web). Because the cluster filesystem also holds
`private/`, every `claude` call carries deny rules for `private/`, the data root and this repository (plus
`agents.deny_read`); the job log prints the list (`agents may not read: ...`). Workspaces live in `~/combo_agent_work`.
The sign-in token is read from `~/.claude_token` inside the job, never passed at submit time. Transcripts are kept by
the CLI under `~/.claude/projects/`; to check afterwards that no agent touched private files:
`grep -l "combo_runs/.*/private" ~/.claude/projects/*combo-agent-work*/*.jsonl` (expect no output).

## Alternative: agents on a laptop
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
