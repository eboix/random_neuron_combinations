#!/bin/bash
# Generic job wrapper; resources and the stage name are passed by slurm/submit.sh, e.g.
#   sbatch --job-name=ci_cache_train --partition=b200-mig45 --gpus=1 --cpus-per-task=6 ... slurm/job.sbatch cache_train
# Time-limit handling (cache stages; submit.sh adds --signal=B:USR1@... --requeue): Slurm sends USR1 to this script
# shortly before the limit, it is forwarded to the stage, which checkpoints and exits 75, and the job requeues itself
# under the SAME job ID, so jobs depending on it (export: afterok) keep waiting. At most CI_MAX_REQUEUES times.
#SBATCH --nodes=1
#SBATCH --ntasks=1
srun bash slurm/stage.sh "$1" &
pid=$!
asked=""
trap 'asked=1; echo "[job] time limit near: asking $1 to checkpoint ($(date))"; kill -USR1 $pid' USR1
wait $pid; rc=$?
[ -n "$asked" ] && { wait $pid; rc=$?; }       # the trap interrupts the first wait; this one gets the real exit code

if [ "$rc" -eq 75 ]; then
  n=${SLURM_RESTART_COUNT:-0}
  if [ "$n" -lt "${CI_MAX_REQUEUES:-5}" ]; then
    echo "[job] checkpointed; requeueing job $SLURM_JOB_ID (restart $((n + 1)) of max ${CI_MAX_REQUEUES:-5})"
    scontrol requeue "$SLURM_JOB_ID" && exit 0
    echo "[job] requeue failed; resume by hand: bash slurm/submit.sh $1 (then scancel and resubmit what waited on it)"
  else
    echo "[job] checkpointed, but already requeued $n times; resume by hand: bash slurm/submit.sh $1"
  fi
fi
exit "$rc"
