#!/bin/bash
# Cluster settings for combo_interp jobs. Defaults are for Penn PARCC's Betty cluster
# (https://parcc.upenn.edu/training/slurm/). Put YOUR values in slurm/env.local.sh (gitignored, so `git pull` never
# conflicts), e.g.
#   export CI_STORE=/vast/projects/<PI>/.../$USER
#   export CI_CODE=$CI_STORE/RandomNeuronCombinations/combo_interp
#   export CI_VENV=$CI_STORE/.venv
#   export HF_HOME=$CI_STORE/huggingface-cache
# Any variable below can be overridden there.
#
# Betty's Slurm CLI filter enforces fixed ratios per partition: memory is derived from CPUs (never pass --mem), and
# GPU jobs get a fixed number of CPUs per GPU:
#   dgx-b200 (full B200, 180 GB): 28 CPUs/GPU    b200-mig90 (90 GB): 14    b200-mig45 (45 GB): 6
#   genoa-std-mem: 5632 MB/CPU, any CPU count
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_here/env.local.sh" ] && source "$_here/env.local.sh"

export CI_ACCOUNT=${CI_ACCOUNT:-}                       # empty = your default account (sacctmgr show assoc user=$USER)
export CI_CPU_PARTITION=${CI_CPU_PARTITION:-genoa-std-mem}
export CI_CPU_CPUS=${CI_CPU_CPUS:-8}                    # -> 8 x 5632 MB = 44 GB
export CI_GPU_PARTITION=${CI_GPU_PARTITION:-b200-mig45} # 45 GB fits Qwen3-8B; dgx-b200 is ~4x the compute
export CI_GPUS=${CI_GPUS:-1}
export CI_GPU_CPUS=${CI_GPU_CPUS:-6}                    # CPUs per GPU required by the partition (table above)
export CI_CLAUDE=${CI_CLAUDE:-$HOME/.local/bin/claude}  # agent stages; sign-in token in ~/.claude_token (claude setup-token)
export CI_MAX_REQUEUES=${CI_MAX_REQUEUES:-5}            # cache stages requeue themselves at the time limit, at most this often
export CI_CONFIG=${CI_CONFIG:-configs/cluster_qwen3_8b.yaml}
export CI_CODE=${CI_CODE:-$HOME/projects/RandomNeuronCombinations/combo_interp}   # this folder, cloned on the cluster
# Run data (~8 GB) and the model cache (Qwen3-8B, ~16 GB) must be on shared storage (jobs land on different nodes).
# Home (/vast/home/...) is shared but capped at 50 GB; the PI's VAST project space is the roomier choice.
export CI_STORE=${CI_STORE:-$HOME}
export CI_RUN=${CI_RUN:-$CI_STORE/combo_runs/qwen3_8b_v1}
export HF_HOME=${HF_HOME:-$CI_STORE/hf_cache}
# Betty supports CUDA 12.8 only; python from the anaconda3 module, torch from the cu128 index:
#   module load anaconda3 && python -m venv $CI_VENV
#   $CI_VENV/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
#   $CI_VENV/bin/pip install -e "$CI_CODE[gpu,dev]"
export CI_VENV=${CI_VENV:-$HOME/venvs/combo}

export COMBO_DATA_ROOT=$CI_RUN/data
export COMBO_PRIVATE_ROOT=$CI_RUN/private
export CI_LOGS=$CI_RUN/logs
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"
_ncpu="${SLURM_CPUS_PER_TASK:-8}"; export OMP_NUM_THREADS="$_ncpu" MKL_NUM_THREADS="$_ncpu"
ci_query_enabled() {  # prints 1 if agents.query.enabled in $CI_CONFIG (explain then needs a GPU for the model)
  (cd "$CI_CODE" && "$CI_VENV/bin/python" -c "import sys; sys.path.insert(0, 'src'); from combo_interp.config import load_config
q = load_config('$CI_CONFIG').agents.get('query') or {}; print(int(bool(q.get('enabled'))))")
}
mkdir -p "$COMBO_DATA_ROOT" "$COMBO_PRIVATE_ROOT" "$CI_LOGS" "$HF_HOME"
