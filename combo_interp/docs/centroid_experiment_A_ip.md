# Experiment A, max-inner-product assignment — does each token's most strongly driven neuron reveal structure?

*Qwen3-0.6B, layers 7 / 14 / 21, holdout split of `qwen3_0.6b_small_v1`. Run 2026-10-26. Companion to
`centroid_experiment_A.md` (same data, sets and metrics; this note covers only the max-inner-product rule). Code:
`src/combo_interp/geometry.py` (`assign_rules=("ip",)`, set `rand_gcov`), `scripts/09_centroids.py`; raw outputs in
`/Users/rohanbuluswar/combo_runs/qwen3_0.6b_small_v1/data/results/geometry/` (rows with `assign == "ip"` in
`*/centroids_centered.csv` and `*/centroids_raw.csv`).*

## Summary

Each token is assigned to the neuron whose gate pre-activation `W_gate[j]·x` is largest. The rows keep their native
norms, so this is exactly what the network computes, with no normalization choices. Partitions built this way from
the trained gate rows explain more of the input's variance and carry more information about the current and next
token than partitions built from the same rows randomly rotated, from isotropic random directions, or from
combinations of gate rows. With centered inputs this holds for variance explained in all 12 (layer, M) cells, and
for next-token information in 11–12 of them. At M = 3072 the margin over the rotated rows is 1.5–2.3× its size
under cosine assignment. Mixing neurons erodes the effect in order (k = 2 > 5 > 20),
and none of it appears in an untrained model.

The new null, random directions with **the gate rows' mean and covariance** (`rand_gcov`), is the strongest
competitor on variance explained. With all 3072 centroids and centered inputs it ties or beats the gate rows. But
its partitions are much more balanced (its norms hardly vary, unlike the gate rows'), and they know far less about
the text: about a quarter to two-fifths of the gate partition's next-token information. On uncentered inputs, what the
network actually sees, the gate rows beat `rand_gcov` in every cell, by up to 13.7 points.

## 1. Setup

* **Tokens and inputs.** As in the main document: the 50,178 tokens of the holdout uniform sample. `x` is the MLP input
  (post-RMSNorm, what `gate_proj` reads). Even documents form the fit half (used for the centering mean and the
  covariance), odd documents the evaluation half on which everything is scored.
* **Assignment rule `ip`.** Token → `argmax_j <c_j, x>`, centroids **not** normalized. For gate rows this is "the
  neuron with the largest gate pre-activation".
* **Two input variants.**
  * *Centered* (primary, as in the main document): x − mean(x). Equivalent to dropping each neuron's implicit offset
    `w_j·mean(x)`.
  * *Uncentered*: raw x, i.e. exactly the network's pre-activation `W_gate[j]·x`, offsets included.
* **Centroid sets.** M ∈ {32, 128, 512, 3072} rows, redrawn 10 / 10 / 5 / 3 times (the full gate and up sets at
  M = 3072 are drawn once). Under `ip` a row's norm changes the partition, so every set needs a norm convention:

| set | construction | norms |
|---|---|---|
| `gate` | M random rows of `W_gate` | native |
| `up` | M random rows of `W_up` | native (their own, about 14–15 % relative spread vs 22–28 % for gate) |
| `rand_rot` | the same gate rows × one random rotation | native gate norms; keeps all norms and angles, removes alignment with x. **Cleanest null** |
| `rand_gcov` | N(mean of gate rows, covariance of gate rows) | as drawn: mean norm matches the gate rows, but only ~3 % relative spread |
| `rand_iso` | isotropic Gaussian directions | given the norms of randomly drawn gate rows |
| `combo_k` (k = 2, 5, 20) | Gaussian-weighted sum of k random unit gate rows | given the norms of randomly drawn gate rows |

  The data-derived sets of the main document (`rand_cov`, `data_pts`, `kmeans`) are points in x-space, not linear
  functionals, so they have no natural place in this rule and are not included.
* **Metrics** (evaluation half, median over repetitions):
  * `ve_raw` (primary): share of the variance of x explained by the partition, 1 − within-cluster SS / total SS,
    with each cluster's spread measured around the mean of its own tokens. `ve` is the same on unit-normalized x; it is
    within 1 point of `ve_raw` everywhere and is not reported separately.
  * `eff_k`: effective number of clusters, exp(entropy of the cluster sizes).
  * `nmi_token`, `nmi_next`, `nmi_doc`: NMI of the partition with the current token, the next token and a coarse
    document label, minus the NMI of a random partition with the same cluster sizes.

## 2. Results, centered inputs

### 2.1 Variance explained

`ve_raw` (%), trained model, L7 / L14 / L21:

| M | gate | up | rand_gcov | rand_rot | rand_iso | combo_2 | combo_5 | combo_20 |
|---|---|---|---|---|---|---|---|---|
| 32 | **6.0 / 8.8 / 11.2** | 3.2 / 5.4 / 6.6 | 4.9 / 6.8 / 9.2 | 5.5 / 5.9 / 8.7 | 5.4 / 6.5 / 8.2 | 5.3 / 7.1 / 10.4 | 5.0 / 6.6 / 9.3 | 4.9 / 6.3 / 9.1 |
| 128 | **10.4 / 12.3 / 19.4** | 6.5 / 8.4 / 12.0 | 7.9 / 11.0 / 13.8 | 8.5 / 8.6 / 11.7 | 7.9 / 8.5 / 12.3 | 8.6 / 10.3 / 16.0 | 8.3 / 10.0 / 15.1 | 7.5 / 9.5 / 13.9 |
| 512 | **16.0 / 17.6 / 25.9** | 9.4 / 15.0 / 19.0 | 12.4 / 15.7 / 20.2 | 11.3 / 11.3 / 16.8 | 11.3 / 10.8 / 15.6 | 13.8 / 14.0 / 21.3 | 12.6 / 13.3 / 20.3 | 11.3 / 11.8 / 18.7 |
| 3072 | 25.7 / 24.2 / 32.5 | 17.3 / 25.3 / 31.6 | 25.4 / **28.0 / 33.1** | 17.8 / 14.9 / 22.2 | 17.6 / 15.0 / 21.6 | 22.5 / 18.7 / 30.0 | 21.1 / 19.0 / 27.4 | 18.6 / 16.8 / 24.7 |

Gate vs each null, out of 12 cells. "Median" = the gate median is higher. "Every rep" = every gate repetition beats
every repetition of the null.

| null | median | every rep |
|---|---|---|
| `rand_rot` | **12** (by 0.5–10.3 points) | 9 (not M = 32 at L7 and L21; M = 128 at L14) |
| `rand_iso` | 12 | 10 |
| `rand_gcov` | 10 (loses at M = 3072, L14 −3.8 and L21 −0.6; ties at L7 +0.3) | 6 |
| `up` | 11 (loses at M = 3072, L14) | — |
| `combo_2` / `combo_5` / `combo_20` | 12 / 12 / 12 | 7 / 7 / 8 |

* **Gate rows are aligned with the data under this rule too.** Against `rand_rot`, which has exactly the same norms
  and angles, the gate partition explains more variance in every cell. The margin grows with M, from
  +0.5 / +2.9 / +2.5 points at M = 32 to +7.9 / +9.3 / +10.3 at M = 3072. Under cosine assignment the same comparison
  gave +5.3 / +4.5 / +4.5 at M = 3072 (main document, §4.7), so keeping the native norms makes the gate's advantage
  1.5–2.3× larger. Which neuron a token drives *most strongly* is more informative than which gate row points closest to it.
* **Dilution is orderly:** gate > combo_2 > combo_5 > combo_20 holds in 11 / 12 cells (at M = 3072, L14, combo_5 is
  0.3 points above combo_2). This is cleaner than under cosine assignment (8 / 12).
* **The gate partitions are coarser than under cosine.** At M = 3072 the gate partition uses 311 / 200 / 328 effective
  clusters, against 791 / 769 / 494 under cosine. The larger-norm rows win more tokens. That is why the absolute
  `ve_raw` is lower than under cosine (25.7 vs 32.1 at L7), even though the margin over the nulls is larger.

### 2.2 The gate-covariance null

`rand_gcov` keeps everything about the gate rows that a Gaussian can: their mean and their full 1024 × 1024
covariance (participation ratio 528–618 of 1024 dimensions, so it is far from isotropic). It loses everything beyond
second order, and the identity of individual rows. Up to M = 512 the gate rows beat it in every cell (by 1.1–5.7
points). At M = 3072 it catches up and, at L14, wins. Two observations say this is not because it captures the data's
structure better:

* **It partitions much more evenly.** At M = 3072 its effective cluster count is 1954 / 1546 / 1374, against
  311 / 200 / 328 for gate. A Gaussian draw in 1024 dimensions has nearly constant norm (~3 % relative spread vs
  22–28 % for gate rows), so under `ip` it behaves almost like cosine assignment and spreads tokens over many cells.
  Finer, more even partitions explain more variance almost automatically.
* **It knows little about the text.** At M = 3072, its partition carries 0.012 / 0.015 / 0.028 next-token information,
  against 0.033 / 0.055 / 0.069 for gate. That is less than every other set, including `rand_rot` and `rand_iso` (2.3).

So the covariance of the gate rows explains part of their geometric advantage (it is strongly anisotropic and
partly aligned with the data's high-variance directions). It does not explain their alignment with linguistic
structure, which comes from the individual rows.

### 2.3 What the partitions know about the text

Chance-corrected NMI, trained, M = 3072, L7 / L14 / L21:

| set | `nmi_token` | `nmi_next` | `nmi_doc` |
|---|---|---|---|
| gate | **.081 / .129 / .135** | **.033 / .055** / .069 | **.069 / .077** / .132 |
| up | .051 / .069 / .102 | .022 / .040 / .063 | .032 / .046 / .109 |
| rand_gcov | .031 / .032 / .052 | .012 / .015 / .028 | .037 / .036 / .079 |
| rand_rot | .050 / .044 / .063 | .019 / .024 / .039 | .051 / .051 / .081 |
| rand_iso | .050 / .046 / .063 | .018 / .022 / .042 | .049 / .047 / .084 |
| combo_2 | .074 / .076 / .103 | .027 / .048 / **.070** | .058 / .052 / .127 |
| combo_5 | .064 / .061 / .088 | .024 / .034 / .060 | .052 / .055 / **.132** |
| combo_20 | .047 / .056 / .075 | .020 / .032 / .049 | .047 / .049 / .097 |

* Across all 12 cells, the gate median is highest against every null on `nmi_next` in 11–12 cells (12 against
  `rand_iso`, `rand_gcov`, `up`, `combo_5` and `combo_20`; 11 against `rand_rot` and `combo_2`).
* At L14 the gate partition carries 2.3–2.9× the next-token and token information of the rotated rows. `combo_2` comes
  closest, consistent with two-neuron mixtures keeping part of the alignment.

![ve_raw vs M, ip, centered](figures/geometry/ve_raw_ip_centered.png)

*Top: trained. Bottom: random init. Gate (blue) is on top in the trained model except at M = 3072, where `rand_gcov`
(teal) and, at L14, `up` (orange) catch up. In the random-init model all sets coincide.*

![nmi_next vs M, ip, centered](figures/geometry/nmi_next_ip_centered.png)

## 3. Results, uncentered inputs (the network's own pre-activations)

`ve_raw` (%), trained model, L7 / L14 / L21:

| M | gate | up | rand_gcov | rand_rot | rand_iso | combo_2 | combo_5 | combo_20 |
|---|---|---|---|---|---|---|---|---|
| 32 | 4.7 / 5.2 / 6.6 | 3.0 / 5.0 / 5.0 | 3.3 / 3.6 / 6.0 | 4.5 / **5.4 / 7.9** | 4.0 / 4.6 / 7.5 | 2.4 / 4.1 / 5.3 | 2.6 / 3.0 / 4.7 | 2.8 / 3.5 / 4.3 |
| 128 | **9.5 / 9.1 / 10.5** | 5.3 / 7.4 / 5.6 | 5.7 / 6.4 / 8.4 | 6.7 / 6.9 / 10.3 | 6.0 / 6.5 / 10.5 | 3.6 / 3.3 / 5.1 | 3.3 / 1.7 / 7.7 | 3.4 / 5.2 / 6.3 |
| 512 | **15.0 / 12.8 / 18.4** | 7.0 / 12.2 / 7.8 | 7.7 / 8.9 / 12.2 | 8.9 / 9.6 / 13.0 | 9.0 / 8.1 / 13.3 | 3.7 / 1.9 / 4.9 | 5.1 / 5.2 / 11.3 | 4.7 / 4.1 / 5.5 |
| 3072 | **25.7 / 20.9 / 27.1** | 1.7 / 15.6 / 11.7 | 12.0 / 12.1 / 15.4 | 12.7 / 10.9 / 17.4 | 11.3 / 11.9 / 17.3 | 3.1 / 2.6 / 6.5 | 7.0 / 3.1 / 10.6 | 5.9 / 6.2 / 11.6 |

* **With all neurons, the gate partition is clearly the best,** explaining 1.3–2× the variance of the best null at
  M = 3072 (L7: 25.7 vs ≤ 12.7; L14: 20.9 vs ≤ 15.6; L21: 27.1 vs ≤ 17.4). It beats `rand_gcov` in all 12 cells (by 0.6–13.7 points), `up` and all combos
  in all 12, and `rand_rot` in 10. The exceptions are the smallest sets (M = 32 at L14 and L21), where every partition
  explains under 8 %.
* **The other sets partly collapse.** Without centering, each row carries an offset `c_j·mean(x)`, and rows aligned
  with the mean input win almost everywhere. The gate partition still uses 339 / 151 / 98 effective clusters at
  M = 3072. Under cosine assignment, which keeps the mean but drops the norms, it collapsed to 1 / 10 / 14 (main
  document, §4.6). Up rows collapse at L7 (2.4 effective clusters, 1.7 % variance). Combinations collapse at every
  layer (3–31 effective clusters). So the network's own winner-take-all over gate pre-activations is far from
  degenerate, while for the nulls it mostly is.
* The ordering among the combos is noisy here (gate > combo_2 > combo_5 > combo_20 in only 1 / 12 cells), because
  the combos' partitions are near-degenerate. Gate beats every combo in every cell.
* Text purity follows the same pattern. The gate median is highest on `nmi_next` against `rand_gcov` and all combos
  in 12 / 12 cells, against `rand_rot` and `rand_iso` in 11 and against `up` in 10. At M = 3072 the gate partition
  carries .032 / .042 / .083 next-token information.

![ve_raw vs M, ip, uncentered](figures/geometry/ve_raw_ip_raw.png)

## 4. Random-init control

Same architecture, untrained weights, run through the same pipeline. No set stands out. Centered, gate − rand_rot
ranges from −0.8 to +1.3 points of `ve_raw`, and all eight sets agree to within 2.9 points in every cell.
Uncentered: −1.5 to +0.3, within 3.0. Effective cluster counts and all three NMIs are the same across sets. Gate
beats `rand_rot` in every repetition in only 1 (centered) and 0 (uncentered) of 12 cells. Everything in §2–3 is
learned.

## 5. Interpretation

* **Max inner product supports the main document's conclusions and sharpens them.** Letting each token go to the
  neuron it drives hardest, in the network's own units, gives partitions that are better aligned with the input and
  more informative about the text than any random set with the same geometry. Keeping the native norms *increases*
  the gate rows' margin over the rotated rows relative to cosine assignment. So the norms carry part of the learned
  structure, not just noise.
* **Combining neurons dilutes it,** monotonically in k when the partitions are not degenerate. That is the
  input-space counterpart of the `ksparse_k` dose-response in the agent study.
* **Second-order statistics of the gate rows explain part of the geometry but not the text alignment.** Random
  directions with the gate rows' covariance can match the gate rows on variance explained with many centroids
  (centered), but their partitions carry a quarter to two-fifths of the gate's next-token information. In the
  uncentered, network-faithful version they lose clearly.
* **Remaining limit.** This is a winner-take-all view: one neuron per token. The paper's claim is per neuron (each
  neuron fires within one cluster), and many neurons fire per token. A cluster served by many neurons appears here
  as a fine refinement, which the symmetric NMI penalizes.

## 6. Caveats

* **`rand_gcov` is not norm-matched.** Its norms hardly vary (~3 % relative spread vs 22–28 % for gate rows), which
  makes its partitions more balanced than the gate's. Part of its strong centered `ve_raw` at M = 3072 is that
  balance. A version with the gate rows' norms (shuffled) would isolate the covariance itself. `rand_iso` and `combo_k`
  do get gate-row norms.
* `up` keeps its own native norms, which differ from the gate's, so gate vs up mixes direction and norm.
* The data-derived references of the main document (`rand_cov`, `data_pts`, `kmeans`) do not apply to this rule, so
  there is no upper bound here. Variance explained is also not comparable across rules in absolute terms, because it
  rises with how evenly a partition uses its cells.
* NMI is symmetric (it is the V-measure) and so penalizes refinements of a clustering. Homogeneity, "each cell lies
  in one cluster", would be the cleaner purity measure.
* One model, three layers, ~25k evaluation tokens, 3–10 repetitions, no confidence intervals. At M = 32 the
  differences are within the repetition spread in several cells.

## 7. Reproduce

```bash
cd combo_interp
python scripts/09_centroids.py --config configs/local_small_study.yaml --init trained     # all rules incl. ip (~9 min on an M1 Pro)
python scripts/09_centroids.py --config configs/local_small_study.yaml --init random
python scripts/09_centroids.py --config configs/local_small_study.yaml --init trained --set geometry.center=false
python scripts/09_centroids.py --config configs/local_small_study.yaml --init random --set geometry.center=false
```

The rules come from `geometry.assign` in `configs/default.yaml`. Tables of every metric are in
`summary_centered.md` / `summary_raw.md` next to the raw outputs.
