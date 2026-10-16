#!/bin/bash
#SBATCH --job-name=ric_hold9
#SBATCH --partition=whartonstat
#SBATCH --account=whartonstat
#SBATCH --nodelist=node9
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/eboix/projects/random_is_clean/slurm/logs/%x_%j.out
# Holder allocation on node7: steps are launched into it with
#   srun --jobid=$JOBID --overlap bash -c '...'
echo "[hold] host=$(hostname) job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-?} start=$(date)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
df -h /scratch/tmp /tmp 2>&1
sleep 2d
