#!/bin/bash
# Idempotent per-model pipeline over hook sites. Run inside a GPU job via slurm/env.sh:
#   bash slurm/vision_sites.sh facebook/convnext-tiny-224 "stage3.block2 stage2.block8" "mlp out resid" "out resid"
# Stages (each skipped when its output exists): data sync from shared staging -> extraction (raw + equal-variance)
# -> real/virt builds + coherence per layer x site -> SAE (collect, train, score, build, coherence) per layer x SAE site.
set -uo pipefail
M=$1; LAYERS=$2; SITES=$3; SAE_SITES=$4
N=${M#*/}; R=${ROOT:-/scratch/tmp/eboix/random_is_clean}; STAGE=/shared_data0/eboix/ric_stage
mkdir -p $R/imagenet-1k/data
if [ "$(ls $R/val 2>/dev/null | wc -l)" -lt 50000 ]; then until [ -f $STAGE/STAGED ]; do echo "[sites] waiting for staged data"; sleep 60; done; echo "[sites] syncing val set from $STAGE"; cp $STAGE/val_labels.npy $STAGE/val_meta.json $R/ && tar xf $STAGE/val.tar -C $R; fi
if [ "$(ls $R/imagenet-1k/data | grep -c train)" -lt 24 ]; then echo "[sites] syncing train shards"; rsync -a $STAGE/imagenet-1k/data/ $R/imagenet-1k/data/; fi
lname() { [ "$2" = mlp ] && echo "$1" || echo "$1.$2"; }
has_layer() { python -c "import json,sys; m=json.load(open('$1')); sys.exit(0 if any(l['name']=='$2' for l in m['layers']) else 1)" 2>/dev/null; }
need=0
for L in $LAYERS; do for S in $SITES $SAE_SITES; do has_layer $R/acts/$N/meta.json "$(lname $L $S)" || need=1; done; done
ALLSITES=$(echo "$SITES $SAE_SITES" | tr ' ' '\n' | sort -u | tr '\n' ',' | sed 's/,$//')
if [ $need = 1 ]; then
  echo "########## $(date) extract $N sites=$ALLSITES"
  python scripts/extract_acts.py --root $R --model $M --k-virtual 128 --sites $ALLSITES --blocks $(echo $LAYERS | tr ' ' ',') || exit 1
  python scripts/extract_acts.py --root $R --model $M --k-virtual 128 --sites $ALLSITES --blocks $(echo $LAYERS | tr ' ' ',') --virt-std-from $R/acts/$N --out-name $N-std || exit 1
fi
for L in $LAYERS; do for S in $SITES; do
  name=$(lname $L $S); slug=$(echo $name | tr . _)
  echo "########## $(date) build $N $name"
  [ -f microscope/$N/data/$slug.json ] || python scripts/build_microscope.py --root $R --model $M --layer $name --n-real 32 --n-virt 32 --out microscope || exit 1
  [ -f microscope/$N-std/data/$slug.json ] || python scripts/build_microscope.py --root $R --model $M --acts-name $N-std --layer $name --n-real 32 --n-virt 32 --out microscope || exit 1
  [ -f microscope/$N/data/${slug}_coherence.json ] || python scripts/coherence.py --micro microscope/$N --layer $name || exit 1
  [ -f microscope/$N-std/data/${slug}_coherence.json ] || python scripts/coherence.py --micro microscope/$N-std --layer $name || exit 1
done; done
for S in $SAE_SITES; do for L in $LAYERS; do
  name=$(lname $L $S); slug=$(echo $name | tr . _); SD=$R/sae/$N/$slug; mkdir -p $SD
  nimg=100000; case $L in stage2*|layer*) nimg=40000;; esac      # 14x14 maps: 196 positions/image
  echo "########## $(date) sae $N $name"
  if ! ls $SD/sae_x*_k32.pt >/dev/null 2>&1; then
    [ -f $SD/stats.json ] || python scripts/sae_collect.py --root $R --model $M --layer $name --n-images $nimg || exit 1
    python scripts/sae_train.py --acts $SD --expansion auto --k 32 --steps 10000 || exit 1
  fi
  CK=$(ls $SD/sae_x*_k32.pt | head -1)
  [ -f $R/acts/$N/${slug}_sae.json ] || python scripts/sae_extract.py --root $R --model $M --layer $name --sae $CK || exit 1
  [ -f microscope/$N-sae/data/$slug.json ] || python scripts/build_microscope.py --root $R --model $M --layer $name --kinds sae --n-sae 32 --sae $CK --acts-name $N-sae --stats-name $N --out microscope || exit 1
  [ -f microscope/$N-sae/data/${slug}_coherence.json ] || python scripts/coherence.py --micro microscope/$N-sae --layer $name || exit 1
  rm -f $SD/train_acts.npy
done; done
echo "[sites] $N done $(date)"
