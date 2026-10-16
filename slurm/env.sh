#!/bin/bash
# Environment for steps launched into the node7 holder job (srun --jobid=... --overlap bash slurm/env.sh <cmd>).
set -euo pipefail
# node-local base: /scratch/tmp (node7/9/10), else /scratch2 (node6), else /scratch
if [ -d /scratch/tmp ] && mkdir -p /scratch/tmp/eboix 2>/dev/null; then BASE=/scratch/tmp/eboix
elif [ -d /scratch2 ] && mkdir -p /scratch2/eboix 2>/dev/null; then BASE=/scratch2/eboix
else mkdir -p /scratch/eboix && BASE=/scratch/eboix; fi
export BASE ROOT=$BASE/random_is_clean
export HF_HOME=$BASE/hf_cache
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=120
export TMPDIR=$BASE/tmp; mkdir -p "$TMPDIR" "$ROOT"
export XDG_CACHE_HOME=$BASE/xdg_cache
export TORCHINDUCTOR_CACHE_DIR=$BASE/inductor_cache
export TOKENIZERS_PARALLELISM=false
_ncpu="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-8}}"
export OMP_NUM_THREADS="$_ncpu" MKL_NUM_THREADS="$_ncpu"
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"
source /home/eboix/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONNOUSERSITE=1
cd /home/eboix/projects/random_is_clean
echo "[env] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} root=$ROOT $(date)"
exec "$@"
