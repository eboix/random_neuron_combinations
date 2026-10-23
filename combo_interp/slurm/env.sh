#!/bin/bash
# Cluster settings for combo_interp jobs. EDIT THE BLOCK BELOW once; everything else reads from here.
# Defaults mirror the team's working scripts (../../slurm/lm_sites.sbatch): account/partition whartonstat, L40S GPUs.

# ---------------------------------------------------------------- edit me
export CI_ACCOUNT=whartonstat
export CI_PARTITION=whartonstat            # or standby (preemptible, more GPUs)
export CI_GPU_GRES=gpu:l40s:1              # 48 GB is plenty for Qwen3-8B in bf16; gpu:a100:1 also works
export CI_CONFIG=configs/cluster_qwen3_8b.yaml
export CI_CODE=$HOME/projects/RandomNeuronCombinations/combo_interp          # this folder, cloned on the cluster
export CI_RUN=$HOME/combo_runs/qwen3_8b_v1                                    # SHARED storage (jobs land on different nodes); ~8 GB
export HF_HOME=${HF_HOME:-$HOME/hf_cache}                                     # Qwen3-8B is ~16 GB; must be shared too
export CI_VENV=$HOME/venvs/combo                                              # python -m venv $CI_VENV && $CI_VENV/bin/pip install -e "$CI_CODE[gpu,dev]"
# ---------------------------------------------------------------- end

export COMBO_DATA_ROOT=$CI_RUN/data
export COMBO_PRIVATE_ROOT=$CI_RUN/private
export CI_LOGS=$CI_RUN/logs
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"
_ncpu="${SLURM_CPUS_PER_TASK:-8}"; export OMP_NUM_THREADS="$_ncpu" MKL_NUM_THREADS="$_ncpu"
mkdir -p "$COMBO_DATA_ROOT" "$COMBO_PRIVATE_ROOT" "$CI_LOGS" "$HF_HOME"
