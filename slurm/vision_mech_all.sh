#!/bin/bash
# Submit Experiments A + B for every vision model with published microscope / CLIP coherence results, at the same
# layers. One L40S job per model; jobs are independent (each syncs val/ to its node's scratch if needed).
#   bash slurm/vision_mech_all.sh            # submit
#   DRY=1 bash slurm/vision_mech_all.sh      # print the sbatch commands only
# Results: results/vision_mech/<model>/ (centroids/, locality/, resolve.json, clip_*.json); logs in slurm/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p slurm/logs results/vision_mech
submit() {   # job-name model layers
  cmd=(sbatch --job-name="$1" --export=ALL,MODEL="$2",LAYERS="$3" slurm/vision_mech.sbatch)
  if [ "${DRY:-0}" = 1 ]; then echo "${cmd[@]}"; else "${cmd[@]}"; fi
}
submit mech_cnx_tiny   facebook/convnext-tiny-224       "stage2.block8 stage3.block2"
submit mech_cnx_base   facebook/convnext-base-224       "stage2.block26 stage3.block2"
submit mech_cnx_large  facebook/convnext-large-224      "stage2.block26 stage3.block2"
submit mech_deit_small facebook/deit-small-patch16-224  "layer5 layer8 layer11"
submit mech_deit_base  facebook/deit-base-patch16-224   "layer5 layer8 layer11"
submit mech_vit_base   google/vit-base-patch16-224      "layer5 layer8 layer11"
