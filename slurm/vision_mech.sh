#!/bin/bash
# Experiments A + B (centroids, per-unit input locality, mediation of CLIP coherence / class purity) for one vision
# model, on the same ImageNet-1k val data as the published coherence results. Run inside a GPU job via slurm/env.sh:
#   bash slurm/vision_mech.sh facebook/convnext-tiny-224 "stage2.block8 stage3.block2"
# Idempotent: data sync is skipped when val/ is complete, finished stages are skipped (--resume).
# Results (csv/md/png/json, a few MB) are copied to results/vision_mech/<model>/ in the repo (home, not scratch).
set -uo pipefail
M=$1; LAYERS=$2
N=${M#*/}; R=${ROOT:-/scratch/tmp/eboix/random_is_clean}; STAGE=/shared_data0/eboix/ric_stage
if [ "$(ls $R/val 2>/dev/null | wc -l)" -lt 50000 ]; then
  until [ -f $STAGE/STAGED ]; do echo "[mech] waiting for staged data"; sleep 60; done
  echo "[mech] syncing val set from $STAGE"; mkdir -p $R
  cp $STAGE/val_labels.npy $STAGE/val_meta.json $R/ && tar xf $STAGE/val.tar -C $R || exit 1
fi
python -c "import torch, transformers, scipy, pandas, matplotlib; print('[mech] torch', torch.__version__, 'transformers', transformers.__version__)" || exit 1
echo "########## $(date) vision_mech $N layers=$LAYERS"
python scripts/vision_mech.py --root $R --model $M --layers $(echo $LAYERS | tr ' ' ',') --stages all --resume \
  --batch 250 --workers ${SLURM_CPUS_PER_TASK:-8} --export results/vision_mech || exit 1
# Experiment A with max-inner-product assignment (gate / combo_k / rand_rot / rand_iso / rand_gcov), -> centroids_cos_ip/
[ -f $R/mech/$N/centroids_cos_ip/summary_raw.md ] || python scripts/vision_mech.py --root $R --model $M \
  --layers $(echo $LAYERS | tr ' ' ',') --stages centroids --assign cos,ip --export results/vision_mech || exit 1
echo "[mech] $N done $(date)"
