#!/bin/bash
# Idempotent per-LM pipeline over hook sites (text analogue of vision_sites.sh). Run via slurm/env.sh:
#   bash slurm/lm_sites.sh Qwen/Qwen2.5-7B-Instruct "9,18,26" "mlp attn_out mlp_out resid" "resid mlp_out attn_out"
set -uo pipefail
M=$1; LAYERS=$2; SITES=$3; SAE_SITES=$4
N=${M#*/}; R=${ROOT:-/scratch/tmp/eboix/random_is_clean}; STAGE=/shared_data0/eboix/ric_stage
C=$R/text/sample/10BT/000_00000.parquet; A=$R/lm_acts/$N
if [ ! -f $C ]; then mkdir -p $(dirname $C); until [ -f $STAGE/text/000_00000.parquet ]; do echo "[lm] waiting for staged corpus"; sleep 60; done; cp $STAGE/text/000_00000.parquet $C; fi
has_layer() { python -c "import json,sys; m=json.load(open('$1')); sys.exit(0 if any(l['name']=='$2' for l in m['layers']) else 1)" 2>/dev/null; }
ALLSITES=$(echo "$SITES $SAE_SITES" | tr ' ' '\n' | sort -u | tr '\n' ',' | sed 's/,$//')
need=0; for L in $(echo $LAYERS | tr , ' '); do for S in $(echo $ALLSITES | tr , ' '); do has_layer $A/meta.json layer$L.$S || need=1; done; done
if [ $need = 1 ]; then
  echo "########## $(date) lm-extract $N layers=$LAYERS sites=$ALLSITES"
  python scripts/lm_extract.py --root $R --model $M --corpus $C --layers $LAYERS --sites $ALLSITES --n-docs 20000 --seq-len 256 --k-virtual 128 || exit 1
  python scripts/lm_extract.py --root $R --model $M --corpus $C --layers $LAYERS --sites $ALLSITES --n-docs 20000 --seq-len 256 --k-virtual 128 --virt-std-from $A --out-name $N-std || exit 1
fi
for L in $(echo $LAYERS | tr , ' '); do for S in $SITES; do
  name=layer$L.$S; slug=layer${L}_$S
  echo "########## $(date) lm-build $N $name"
  [ -f microscope/$N/data/$slug.json ] || python scripts/lm_build.py --root $R --model $M --layer $name --n-real 32 --n-virt 32 --out microscope || exit 1
  [ -f microscope/$N-std/data/$slug.json ] || python scripts/lm_build.py --root $R --model $M --layer $name --n-real 32 --n-virt 32 --acts-name $N-std --out microscope || exit 1
  [ -f microscope/$N/data/${slug}_coherence.json ] || python scripts/lm_coherence.py --micro microscope/$N --layer $name || exit 1
  [ -f microscope/$N-std/data/${slug}_coherence.json ] || python scripts/lm_coherence.py --micro microscope/$N-std --layer $name || exit 1
done; done
for S in $SAE_SITES; do for L in $(echo $LAYERS | tr , ' '); do
  name=layer$L.$S; slug=layer${L}_$S; SD=$R/lm_sae/$N/$slug; mkdir -p $SD
  echo "########## $(date) lm-sae $N $name"
  if ! ls $SD/sae_x*_k32.pt >/dev/null 2>&1; then
    [ -f $SD/stats.json ] || python scripts/lm_collect.py --root $R --model $M --corpus $C --layer $name --n-docs 20000 --doc-offset 20000 || exit 1
    python scripts/sae_train.py --acts $SD --expansion auto --k 32 --steps 10000 || exit 1
  fi
  CK=$(ls $SD/sae_x*_k32.pt | head -1)
  [ -f $A/${slug}_sae.json ] || python scripts/lm_sae_extract.py --root $R --model $M --layer $name --sae $CK || exit 1
  [ -f microscope/$N-sae/data/$slug.json ] || python scripts/lm_build.py --root $R --model $M --layer $name --kinds sae --n-sae 32 --sae $CK --acts-name $N-sae --stats-name $N --out microscope || exit 1
  [ -f microscope/$N-sae/data/${slug}_coherence.json ] || python scripts/lm_coherence.py --micro microscope/$N-sae --layer $name || exit 1
  rm -f $SD/train_acts.npy
done; done
echo "[lm] $N done $(date)"
