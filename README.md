# random_is_clean — are random neuron combinations as interpretable as neurons?

Reproduction of the Goodfire "Neural Networks Are Already Clean" dataset-example microscope
(https://already-clean.goodfire.com/microscope/medium.html) on a public ConvNeXt, extended with
**virtual neurons** (random linear combinations of a block's neurons), to test whether the
maximally-activating-example view is equally interpretable for neurons and for random directions.

Their "medium" model (7.06M params, 18 blocks, 8,832 post-SwiGLU coordinates) is a ConvNeXt with
depths [3,3,9,3] and widths [32,64,128,256]; its weights are not released. `facebook/convnext-tiny-224`
(28M, same depths, widths 3x larger, GELU MLP) is the closest public analogue, so "neuron" here means a
post-GELU MLP hidden coordinate of a block, exactly like their post-SwiGLU product coordinate.

## Pipeline (all on node7, data on node-local `/scratch/tmp/eboix/random_is_clean`)

| step | script | what |
|---|---|---|
| job | `slurm/hold_node7.sbatch` | 2-day holder job on node7 (standby, 1x A100, 44G); steps run via `srun --jobid=<id> --overlap --ntasks=1 bash slurm/env.sh <cmd>` |
| data | `scripts/prep_data.py` | HF `ILSVRC/imagenet-1k` parquet → `val/{00000..49999}.jpg` + `val_labels.npy` + `val_meta.json`; models to the HF cache. `--split train` pulls the 296 train shards (background, ~140 GB) |
| extract | `scripts/extract_acts.py` | one pass over the 50k val center crops (256→224, same geometry as theirs); for every block: per-image **max over positions**, argmax, min, argmin for all real units and for K=128 random unit-norm Gaussian directions (`--virt-std-from` = equal-variance variant), unit mean/std, predictions. 83 s on an A100; top-1 81.9 % (published 82.1 %) |
| build | `scripts/build_microscope.py` | per layer: 32 real (neuron 0 + 31 random) + 32 virtual units → example sheets (16×3 cells of 96 px, buckets top-16 / ranks 5-12 / 47-54 / 497-504 / 4 most negative, crops centred on the argmax with their 64/96/144/192 px stage windows), 24×24 heatmaps, per-layer JSON |
| metric | `scripts/coherence.py` | label-free proxy: mean pairwise CLIP ViT-B/32 cosine of a unit's top-16 crops vs. 16 random crops |
| sae | `scripts/sae_collect.py`, `sae_train.py`, `sae_extract.py`, `sae.py` | TopK SAE on the block's hidden activations (train images), scored on val; `build_microscope.py --kinds sae` |
| merge | `scripts/merge_variants.py` | folds the `-std` and `-sae` sibling dirs + coherence into one multi-model site (`models.js`, sprites per layer and kind), writes `index.html` |
| viewer | `microscope/viewer.html` | Goodfire-style microscope with a **blind mode** (shuffled units, type hidden, rate then reveal) |

Published viewer (model switcher: tiny / base / large): https://claude.ai/artifact/6FLPCVrxjZXZWCd9sPQ78W
(local copy: `microscope/index.html`, open over `python -m http.server` from `microscope/` or as a file).

## Results (2026-10-16, 32 units per kind per layer)

CLIP coherence of the top-16 crops (mean pairwise cosine, ViT-B/32; higher = more visually consistent;
"baseline" = 16 crops drawn at random from the layer's sheets). "random" = Gaussian unit-norm direction in
the block's hidden space; "eq-var" = coefficients scaled by 1/std of each unit first.

| model (top-1) | layer | real neurons | random | eq-var random | baseline | real vs random |
|---|---|---|---|---|---|---|
| tiny 28M (81.9 %) | stage3.block2 (last, 7×7×3072) | **0.696** (sd .075) | 0.619 | 0.624 | 0.613 | p = 7e-6, P(real>rand) = 0.83 |
| tiny | stage2.block8 (mid, 14×14×1536) | **0.695** (sd .062) | 0.668 | 0.674 | 0.643 | p = 0.17, P = 0.60 |
| base 89M (83.7 %) | stage3.block2 (last, 7×7×4096) | **0.681** (sd .071) | 0.611 | 0.621 | 0.613 | p = 3e-7, P = 0.87 |
| base | stage2.block26 (mid, 14×14×2048) | **0.743** (sd .053) | 0.679 | 0.682 | 0.655 | p = 5e-7, P = 0.87 |
| large 198M (84.2 %) | stage3.block2 (last, 7×7×6144) | **0.696** (sd .071) | 0.610 | 0.624 | 0.613 | p = 3e-8, P = 0.90 |
| large | stage2.block26 (mid, 14×14×3072) | **0.770** (sd .052) | 0.676 | 0.680 | 0.657 | p = 2e-9, P = 0.94 |

Top-16 class purity (share of the 16 crops from the modal ImageNet class) at tiny stage3.block2: real mean
0.40 (6/32 units ≥ 0.8, e.g. neuron 0 = badger, 1538 = goldfinch, 2499 = bobsled), random combos mean 0.08
(every one of 64 at the floor: 15–16 distinct classes in 16 crops). Even low-purity real units are coherent
on inspection (tiny neuron 2862: small yellow food morsels across many classes); random combos read as
mixtures of unrelated objects.

**Conclusion so far:** the hypothesis "equally interpretable" is rejected in every model at the last block
and in base/large at the mid-level block: random combinations' top examples are about as incoherent as
random crops, while most neurons are clean. Only tiny's mid-level block shows a small, non-significant gap.
The real-vs-random gap at the mid-level block *grows* with model size (+0.03 → +0.06 → +0.09): bigger
ConvNeXts have cleaner mid-level neurons, not messier ones. Equal-variance scaling makes no difference.

Why random directions fail (per-neuron decomposition of a random unit's response on its own top-16 images):
the largest contributing neuron carries only 3–7 % of the positive mass, and the dominant neuron differs
across 11–16 of the 16 images, so the set is 16 unrelated concepts; the unit's max is only ~2.6× its 99th
percentile vs ~5.5× for neurons (a CLT-like tail, not a detector). Last-block neurons are class-aligned
because the path neuron → W2 → residual → pool → LN → classifier is linear up to LN's normalisation: each
neuron has a fixed direct-logit vector (max/std over classes ≈ 11–14), and for 19/32 (tiny) and 15/32 (base)
visualised neurons its top-3 classes contain the modal top-16 class (`scripts/` ad-hoc check, see chat log).

### Vision transformers (DeiT-small / DeiT-base / ViT-B/16)

`scripts/model_adapters.py` makes the pipeline architecture-agnostic: for ViT/DeiT the neurons are the
post-GELU MLP hidden units of each transformer block at the 196 patch tokens (CLS/distillation tokens
dropped), a 14×14 map with stride 16 and a 144 px focus crop. Layers 5, 8, 11 of the 12.

| model (top-1) | layer | real | random | eq-var random | baseline | real vs random |
|---|---|---|---|---|---|---|
| DeiT-small 22M (80.2 %) | layer11 (last) | 0.652 | 0.641 | 0.626 | 0.617 | p = 0.26 |
| DeiT-small | layer8 | **0.727** | 0.685 | 0.648 | 0.625 | p = 0.006 (vs eq-var 3e-6) |
| DeiT-small | layer5 | 0.691 | 0.653 | **0.712** | 0.634 | p = 0.002 (vs eq-var 0.22) |
| DeiT-base / ViT-B/16 | | (pipelines running) | | | | |

ViTs look different from ConvNeXts. The last block's patch-token neurons are barely more coherent than
random directions and both sit near the random-crop baseline (the class read-out goes through the CLS token,
so last-layer patch MLP units are not the "class neurons" of a ConvNeXt). The clean neurons are in the middle
(layer 8). At layer 5 a random direction in the *standardised* basis can be as clean as a neuron: e.g.
eq-var unit 94 is a piano-keyboard detector spread over ~136 effective neurons (largest weight 4 %), so
early ViT layers hold redundant texture families that any even mixture picks up. Caveat on the eq-var
variant: where the unit-std distribution is heavy-tailed (DeiT layer 8: median effective neurons 36, up to
50 % of the weight on one rare neuron) it degenerates toward a few rare neurons; the builder now records
`effectiveNeurons` / `top1Share` per random unit and the viewer shows them.

### SAE features (third unit kind)

`scripts/sae_collect.py` → `sae_train.py` → `sae_extract.py`: a TopK sparse autoencoder (8× expansion,
k = 32, aux-k loss for dead latents, 10k steps of 4096 positions) trained on the same post-GELU hidden
activations at every position of 100k ImageNet **train** center crops, then scored on the 50k val images
with the same max-over-positions statistic. Features are a learned thresholded linear readout in the
same space as neurons and random directions, so all three are visualised identically; the builder samples
32 features that fire on ≥ 500 val images. Results: see the viewer's `SAE` units and the table below once
the chains finish (`slurm/logs/sae_*_s3b2.log`).

## Repository contents

* `scripts/` — the pipeline (see table above); `slurm/` — node7 job + env wrapper + per-model driver.
* `microscope/` — the self-contained viewer: `index.html` (or the artifact page `viewer.html`), `models.js`,
  and per model `meta.js`, `data/<layer>.js` (units, examples, 24×24 heatmaps, coherence) and
  `sprites/<layer>_<kind>.jpg` (all example sheets of a layer and unit kind stacked vertically).
  Open `microscope/index.html` in a browser, or `cd microscope && python -m http.server`.
  The per-unit sheets, raw heatmap binaries and the `-std` / `-sae` intermediate dirs are not committed
  (they are folded into the data files by `merge_variants.py`); the activations are node-local (below).
* The published artifact link above is private to its owner; the repo copy is the shareable one.

## Layout on node7 scratch

```
/scratch/tmp/eboix/random_is_clean/
  imagenet-1k/data/validation-*.parquet (+ train-*.parquet as they arrive)
  val/00000.jpg … 49999.jpg, val_labels.npy, val_meta.json
  acts/convnext-tiny-224/      {layer}_real_{max,min,argmax,argmin}.npy (50000 × D), {layer}_virt_*.npy (50000 × 128),
                               {layer}_virtual_dirs.npy, {layer}_unit_stats.npz, preds.npy, meta.json   (14 GB, all 18 blocks)
  acts/convnext-tiny-224-std/  same with equal-variance directions
```
HF cache (models, CLIP): `/scratch/tmp/eboix/hf_cache`.

## Re-running

```bash
sbatch slurm/hold_node7.sbatch                       # then J=<jobid>
R=/scratch/tmp/eboix/random_is_clean
srun --jobid=$J --overlap --ntasks=1 bash slurm/env.sh python scripts/extract_acts.py --root $R --model facebook/convnext-tiny-224
srun --jobid=$J --overlap --ntasks=1 bash slurm/env.sh python scripts/build_microscope.py --root $R --layer stage3.block2 --n-real 32 --n-virt 32 --out microscope
srun --jobid=$J --overlap --ntasks=1 bash slurm/env.sh python scripts/coherence.py --micro microscope/convnext-tiny-224 --layer stage3.block2
python scripts/merge_variants.py --main microscope/convnext-tiny-224 --std microscope/convnext-tiny-224-std --viewer microscope/viewer.html
```
Any other block works the same way (`--layer stage1.block2`, …); the extraction already covers all 18.
