#!/bin/bash
# Build microscopes + coherence for one model (raw and equal-variance variants), given finished extractions.
#   bash slurm/model_pipeline.sh facebook/convnext-base-224 "stage3.block2 stage2.block26"
set -euo pipefail
M=$1; LAYERS=$2; N=${M#*/}; R=/scratch/tmp/eboix/random_is_clean
for L in $LAYERS; do
  python scripts/build_microscope.py --root $R --model $M --layer $L --n-real 32 --n-virt 32 --out microscope
  python scripts/build_microscope.py --root $R --model $M --acts-name $N-std --layer $L --n-real 32 --n-virt 32 --out microscope
  python scripts/coherence.py --micro microscope/$N --layer $L
  python scripts/coherence.py --micro microscope/$N-std --layer $L
done
echo "[pipeline] $N done"
