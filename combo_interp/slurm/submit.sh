#!/bin/bash
# Submit a phase of the study from the cluster login node (run from the combo_interp folder):
#   bash slurm/submit.sh test     CPU: test suite (do this first on a new cluster setup)
#   bash slurm/submit.sh data     corpus -> stats_units -> {cache_train, cache_holdout, calibrate} -> export+pilot;
#                                 also Experiment A (centroids) after cache_holdout
#   bash slurm/submit.sh centroids   Experiment A alone (resumable: finished runs are skipped)
#   bash slurm/submit.sh agents   Claude agents on the cluster: explain -> {write, simulate}. explain runs on a GPU
#                                 (model query service in the same job) only if agents.query.enabled, else on CPU
#   bash slurm/submit.sh explain|write|simulate   one agent stage (resumable: rerun after a timeout / usage limit)
#   bash slurm/submit.sh serve    query service for explainer agents running on a LAPTOP (SSH tunnel; see README)
#   bash slurm/submit.sh probe    after the write / simulate results exist
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
    corpus) echo 08:00:00 ;;  stats_units) echo 04:00:00 ;;  cache_train) echo 12:00:00 ;;
    cache_holdout) echo 04:00:00 ;;  calibrate) echo 02:00:00 ;;  export) echo 05:00:00 ;;  centroids) echo 10:00:00 ;;
    explain) echo 16:00:00 ;;  write|simulate) echo 08:00:00 ;;
  esac
}

explain_kind() { [ "$(ci_query_enabled)" = 1 ] && echo gpu || echo cpu; }

case "${1:?phase or stage}" in
  data)
    c=$(sub corpus cpu $(T corpus))
    s=$(sub stats_units gpu $(T stats_units) "$c")
    t=$(sub cache_train gpu $(T cache_train) "$s")
    h=$(sub cache_holdout gpu $(T cache_holdout) "$s")
    k=$(sub calibrate gpu $(T calibrate) "$s")
    e=$(sub export cpu $(T export) "$t" "$h" "$k")
    g=$(sub centroids gpu $(T centroids) "$h")
    echo "submitted: corpus=$c stats_units=$s cache_train=$t cache_holdout=$h calibrate=$k export=$e centroids=$g"
    echo "logs: $CI_LOGS   watch: squeue -u \$USER" ;;
  agents)
    x=$(sub explain $(explain_kind) $(T explain))
    w=$(sub write cpu $(T write) "$x")
    m=$(sub simulate cpu $(T simulate) "$x")
    echo "submitted: explain=$x write=$w simulate=$m   (then: submit.sh probe, submit.sh score)" ;;
  explain) echo "explain=$(sub explain $(explain_kind) $(T explain))" ;;
  write|simulate) echo "$1=$(sub "$1" cpu $(T $1))" ;;
  test)   echo "test=$(sub test cpu 01:00:00)" ;;
  serve)  echo "serve=$(sub serve gpu 12:00:00)  (cancel with scancel when the explain stage is done)" ;;
  probe)  echo "probe=$(sub probe gpu 05:00:00)" ;;
  score)  echo "score=$(sub score cpu 02:00:00)" ;;
  corpus|export) echo "$1=$(sub "$1" cpu $(T $1))" ;;
  stats_units|cache_train|cache_holdout|calibrate|centroids) echo "$1=$(sub "$1" gpu $(T $1))" ;;
  *) echo "unknown phase/stage $1"; exit 2 ;;
esac
