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
  export)        python scripts/03_export_agent_view.py $C
                 python scripts/04_pilot.py $C ;;
  serve)         # model query service for explainer agents; reach it from the laptop through an SSH tunnel
                 echo "[serve] on the laptop: ssh -N -L 8765:$(hostname):8765 <you>@<login node>"
                 python scripts/05b_query_server.py $C --host 0.0.0.0 --port 8765 ;;
  probe)         python scripts/06_build_probe_job.py $C
                 python scripts/07_probe.py $C --job-file $D/probe/probe_in.jsonl --out $D/probe/probe_out.jsonl ;;
  score)         P="--probe $D/probe/probe_out.jsonl"
                 python scripts/08_score_and_analyze.py $C $P                                          # primary (config)
                 python scripts/08_score_and_analyze.py $C $P --set scoring.threshold=corpus_frac ;;    # sensitivity
  *) echo "unknown stage $1"; exit 2 ;;
esac
echo "[stage $1] done $(date)"
