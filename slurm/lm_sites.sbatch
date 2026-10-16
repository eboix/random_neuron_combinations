#!/bin/bash
#SBATCH --job-name=ric_lm
#SBATCH --partition=whartonstat
#SBATCH --account=whartonstat
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/eboix/projects/random_is_clean/slurm/logs/%x_%j.out
# sbatch --job-name=sites_tiny --export=ALL,MODEL=facebook/convnext-tiny-224,LAYERS="stage3.block2 stage2.block8",SITES="mlp out resid",SAE_SITES="resid out" slurm/lm_sites.sbatch
: "${MODEL:?}" "${LAYERS:?}" "${SITES:?}" "${SAE_SITES:?}"
srun bash slurm/env.sh bash slurm/lm_sites.sh "$MODEL" "$LAYERS" "$SITES" "$SAE_SITES"
