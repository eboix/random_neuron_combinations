#!/bin/bash
# Run one pipeline stage inside a Slurm job (submitted by slurm/submit.sh):  bash slurm/stage.sh <stage>
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$CI_VENV/bin/activate"
cd "$CI_CODE"
C="--config $CI_CONFIG"; D=$COMBO_DATA_ROOT
echo "[stage $1] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} job=${SLURM_JOB_ID:-} $(date)"
case "$1" in
  test)          python -m pytest -q ;;
  corpus)        # needs internet: streams the datasets, and pre-downloads the model so GPU nodes can run offline
                 python -c "from huggingface_hub import snapshot_download as s; import yaml; s(yaml.safe_load(open('$CI_CONFIG'))['model']['name'])"
                 python scripts/00_prepare_corpus.py $C ;;
  stats_units)   python scripts/01_stats_and_units.py $C ;;
  # exec: python must be the task process so the USR1 forwarded by srun reaches it (checkpoint + exit 75, job.sbatch)
  cache_train)   exec python scripts/02_cache.py $C --split train ;;
  cache_holdout) exec python scripts/02_cache.py $C --split holdout ;;
  calibrate)     python scripts/07b_calibrate.py $C ;;
  centroids)     # Experiment A (docs/experiment_A_centroids_qwen.md): {trained, random init} x {centered, uncentered x};
                 # every rule in geometry.assign (cos, euclid, ip). Resumable: finished runs are skipped.
                 for init in trained random; do
                   for center in true false; do
                     tag=$([ $center = true ] && echo centered || echo raw)
                     if [ -f "$D/results/geometry/$init/centroids_$tag.csv" ]; then echo "[centroids] $init/$tag exists, skipping"; continue; fi
                     python scripts/09_centroids.py $C --init $init --set geometry.center=$center
                   done
                 done ;;
  export)        python scripts/03_export_agent_view.py $C
                 python scripts/04_pilot.py $C ;;
  explain|write|simulate)
                 # Claude agents. No shell; deny rules keep them out of private/, the data root and this repo
                 # (agents/run.py). The sign-in token is read here, never passed at submit time.
                 [ -s "$HOME/.claude_token" ] || { echo "no ~/.claude_token: run 'claude setup-token' on the login node," \
                                                        "save the token to ~/.claude_token, chmod 600"; exit 2; }
                 export CLAUDE_CODE_OAUTH_TOKEN="$(cat "$HOME/.claude_token")"
                 A="--set agents.claude_cmd=$CI_CLAUDE"
                 if [ "$1" = explain ] && [ "$(ci_query_enabled)" = 1 ]; then     # model queries: service in this job
                   port=$((20000 + ${SLURM_JOB_ID:-0} % 20000))                      # GPU nodes are shared: per-job port
                   python scripts/05b_query_server.py $C --host 127.0.0.1 --port $port &
                   srv=$!; trap 'kill $srv 2>/dev/null' EXIT
                   for _ in $(seq 360); do                                           # model load: wait up to 30 min
                     curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
                     kill -0 $srv 2>/dev/null || { echo "[explain] query service exited"; exit 1; }
                     sleep 5
                   done
                   A="$A --set agents.query.url=http://127.0.0.1:$port"
                 fi
                 # CI_AGENT_ARGS (set at submit time, e.g. "--limit 3" for a trial) is passed on to 05_run_agents.py
                 python scripts/05_run_agents.py $C $A --stage "$1" ${CI_AGENT_ARGS:-} ;;   # resumable: finished units skipped
  serve)         # model query service for explainer agents on a laptop; reach it through an SSH tunnel
                 echo "[serve] on the laptop: ssh -N -L 8765:$(hostname):8765 <you>@<login node>"
                 python scripts/05b_query_server.py $C --host 0.0.0.0 --port 8765 ;;
  probe)         PD=$(ci_probe_dir)                                                  # per agent run (agents.run_tag)
                 python scripts/06_build_probe_job.py $C
                 python scripts/07_probe.py $C --job-file $PD/probe_in.jsonl --out $PD/probe_out.jsonl ;;
  score_sim)     python scripts/08b_sim_tables.py $C ;;                             # simulator scores + tables only
  score)         P="--probe $(ci_probe_dir)/probe_out.jsonl"
                 python scripts/08_score_and_analyze.py $C $P                                          # primary (config)
                 python scripts/08_score_and_analyze.py $C $P --set scoring.threshold=corpus_frac ;;    # sensitivity
  *) echo "unknown stage $1"; exit 2 ;;
esac
echo "[stage $1] done $(date)"
