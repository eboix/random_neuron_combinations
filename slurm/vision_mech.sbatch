#!/bin/bash
#SBATCH --job-name=ric_mech
#SBATCH --partition=whartonstat
#SBATCH --account=whartonstat
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/eboix/projects/random_is_clean/slurm/logs/%x_%j.out
# One model per job (submit all with slurm/vision_mech_all.sh):
#   sbatch --job-name=mech_tiny --export=ALL,MODEL=facebook/convnext-tiny-224,LAYERS="stage2.block8 stage3.block2" slurm/vision_mech.sbatch
: "${MODEL:?}" "${LAYERS:?}"
srun bash slurm/env.sh bash slurm/vision_mech.sh "$MODEL" "$LAYERS"
