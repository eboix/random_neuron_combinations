#!/bin/bash
# Submit a phase of the study from the cluster login node (run from the combo_interp folder):
#   bash slurm/submit.sh test     CPU: test suite (do this first on a new cluster setup)
#   bash slurm/submit.sh data     corpus -> stats_units -> {cache_train, cache_holdout, calibrate} -> export+pilot
#   bash slurm/submit.sh serve    model query service for the explainer agents (only if agents.query.enabled)
#   bash slurm/submit.sh probe    after the write / simulate results are synced up
#   bash slurm/submit.sh score    scores + equivalence analysis (primary threshold + corpus_frac sensitivity)
#   bash slurm/submit.sh <stage>  any single stage of slurm/stage.sh (e.g. cache_train to resume after a timeout)
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/env.sh

# name  kind  time  ->  sbatch resources
sub() {  # sub <stage> <cpu|gpu> <time> [dependency job ids...]
  local stage=$1 kind=$2 time=$3; shift 3
  local dep=""; [ $# -gt 0 ] && dep="--dependency=afterok:$(IFS=:; echo "$*")"
  # no --mem: Betty's CLI filter derives memory from CPUs and rejects any other ratio
  local res="--partition=$CI_CPU_PARTITION --cpus-per-task=$CI_CPU_CPUS"
  [ "$kind" = gpu ] && res="--partition=$CI_GPU_PARTITION --gpus=$CI_GPUS --cpus-per-task=$((CI_GPU_CPUS * CI_GPUS))"
  local acct=""; [ -n "$CI_ACCOUNT" ] && acct="--account=$CI_ACCOUNT"
  # cache stages checkpoint on USR1 (20 min before the limit) and requeue under the same job ID (slurm/job.sbatch)
  local rq=""; [[ "$stage" == cache_* ]] && rq="--signal=B:USR1@1200 --requeue --open-mode=append"
  sbatch --parsable --job-name="ci_$stage" $acct --time="$time" \
         $res $dep $rq --output="$CI_LOGS/%x_%j.out" slurm/job.sbatch "$stage"
}

# time limits (also used when one stage is resubmitted). Estimates, not measurements: check `sacct` after a run.
# Cache stages requeue themselves at the limit, so theirs can be tight; the others restart from scratch.
T() {
  case $1 in
    corpus) echo 06:00:00 ;;  stats_units) echo 02:00:00 ;;  cache_train) echo 08:00:00 ;;
    cache_holdout) echo 02:00:00 ;;  calibrate) echo 01:00:00 ;;  export) echo 03:00:00 ;;
  esac
}

case "${1:?phase or stage}" in
  data)
    c=$(sub corpus cpu $(T corpus))
    s=$(sub stats_units gpu $(T stats_units) "$c")
    t=$(sub cache_train gpu $(T cache_train) "$s")
    h=$(sub cache_holdout gpu $(T cache_holdout) "$s")
    k=$(sub calibrate gpu $(T calibrate) "$s")
    e=$(sub export cpu $(T export) "$t" "$h" "$k")
    echo "submitted: corpus=$c stats_units=$s cache_train=$t cache_holdout=$h calibrate=$k export=$e"
    echo "logs: $CI_LOGS   watch: squeue -u \$USER" ;;
  test)   echo "test=$(sub test cpu 01:00:00)" ;;
  serve)  echo "serve=$(sub serve gpu 12:00:00)  (cancel with scancel when the explain stage is done)" ;;
  probe)  echo "probe=$(sub probe gpu 03:00:00)" ;;
  score)  echo "score=$(sub score cpu 01:00:00)" ;;
  corpus|export) echo "$1=$(sub "$1" cpu $(T $1))" ;;
  stats_units|cache_train|cache_holdout|calibrate) echo "$1=$(sub "$1" gpu $(T $1))" ;;
  *) echo "unknown phase/stage $1"; exit 2 ;;
esac
