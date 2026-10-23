#!/bin/bash
# Cluster settings for combo_interp jobs. EDIT THE BLOCK BELOW once; everything else reads from here.
# Defaults are for Penn PARCC's Betty cluster (https://parcc.upenn.edu/training/slurm/). Partitions there:
#   CPU: genoa-std-mem (genoa-lrg-mem above 256 GB)    GPU: b200-mig45 (45 GB MIG slice), b200-mig90, dgx-b200 (full B200)

# ---------------------------------------------------------------- edit me
export CI_ACCOUNT=                         # Slurm account from ColdFront; empty = your default account.
                                           #   list yours: sacctmgr show assoc user=$USER format=account%30
export CI_CPU_PARTITION=genoa-std-mem
export CI_GPU_PARTITION=b200-mig45         # 45 GB is plenty for Qwen3-8B in bf16
export CI_GPUS=1                           # passed as --gpus
export CI_GPU_MEM=64G                      # host RAM for GPU jobs; MIG partitions may cap memory per GPU
export CI_CONFIG=configs/cluster_qwen3_8b.yaml
export CI_CODE=$HOME/projects/RandomNeuronCombinations/combo_interp          # this folder, cloned on the cluster
# Run data (~8 GB) and the model cache (Qwen3-8B, ~16 GB) must be on shared storage (jobs land on different nodes).
# Home (/vast/home/...) is shared but capped at 50 GB; the PI's VAST project space is the roomier choice:
#   export CI_STORE=/vast/projects/<PI-PennKey>/default/$USER
export CI_STORE=${CI_STORE:-$HOME}
export CI_RUN=$CI_STORE/combo_runs/qwen3_8b_v1
export HF_HOME=${HF_HOME:-$CI_STORE/hf_cache}
# Betty supports CUDA 12.8 only; python from the anaconda3 module, torch from the cu128 index:
#   module load anaconda3 && conda create -y -p ~/envs/py311 python=3.11
#   ~/envs/py311/bin/python -m venv $CI_VENV && $CI_VENV/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
#   $CI_VENV/bin/pip install -e "$CI_CODE[gpu,dev]"
export CI_VENV=$HOME/venvs/combo
# ---------------------------------------------------------------- end

export COMBO_DATA_ROOT=$CI_RUN/data
export COMBO_PRIVATE_ROOT=$CI_RUN/private
export CI_LOGS=$CI_RUN/logs
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"
_ncpu="${SLURM_CPUS_PER_TASK:-8}"; export OMP_NUM_THREADS="$_ncpu" MKL_NUM_THREADS="$_ncpu"
mkdir -p "$COMBO_DATA_ROOT" "$COMBO_PRIVATE_ROOT" "$CI_LOGS" "$HF_HOME"
