#!/bin/bash
# Generic job wrapper; resources and the stage name are passed by slurm/submit.sh, e.g.
#   sbatch --job-name=ci_cache_train --partition=b200-mig45 --gpus=1 ... slurm/job.sbatch cache_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
srun bash slurm/stage.sh "$1"
