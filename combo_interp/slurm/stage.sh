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
  cache_train)   python scripts/02_cache.py $C --split train ;;        # resumable: resubmit after a timeout
  cache_holdout) python scripts/02_cache.py $C --split holdout ;;
  calibrate)     python scripts/07b_calibrate.py $C ;;
  export)        python scripts/03_export_agent_view.py $C
                 python scripts/04_pilot.py $C ;;
  probe1)        python scripts/06_build_probe_job.py $C --round 1
                 python scripts/07_probe.py $C --job-file $D/probe/round1_in.jsonl --out $D/probe/round1_out.jsonl ;;
  probe2)        python scripts/06_build_probe_job.py $C --round 2
                 python scripts/07_probe.py $C --job-file $D/probe/round2_in.jsonl --out $D/probe/round2_out.jsonl ;;
  score)         P="--probe $D/probe/round1_out.jsonl $D/probe/round2_out.jsonl"
                 python scripts/08_score_and_analyze.py $C $P                                          # primary (config)
                 python scripts/08_score_and_analyze.py $C $P --set scoring.threshold=corpus_frac ;;    # sensitivity
  *) echo "unknown stage $1"; exit 2 ;;
esac
echo "[stage $1] done $(date)"
