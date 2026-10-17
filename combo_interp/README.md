# combo_interp — are random linear combinations of MLP neurons as interpretable as neurons?

Tests the claim of Cammarata's *"Neural Networks Are Already Clean"* (that a network's own neurons are special, legible
building blocks) by running the same kind of agentic explanation loop on **neurons** and on **random linear
combinations of neurons**, and asking whether the two are statistically equivalent.

> **Status of this code.** Everything that doesn't need torch has been executed and tested (units, export, blinding, agent
> runner with a fake backend, probe-job building, scoring, statistics, plots, CLI wiring). The torch-side code
> (`hooks.py`, `stats.py` run loop, `cache.py`, `topk.py`, `probe.py`, `interventions.py`) was written and desk-checked but
> **could not be executed where it was written**. `tests/test_torch_tiny.py` checks it against brute-force recomputation on a
> tiny random Llama on CPU. **Run `pytest -q` on your own machine before submitting anything to the cluster.**

## The unit and the conditions

A *unit* is a direction `a` plus offset `b`: `z(token) = p·a + b`, with `p` = the MLP hidden activation (input of `down_proj`,
`silu(gate)·up` for SwiGLU models). Per layer (default layers 9/18/27 of Qwen3-8B):

| condition | construction | scale |
|---|---|---|
| `neuron` | one coordinate of `p` | raw, uncentered (the neuron's natural range) |
| `dense_raw` | random orthonormal direction in raw coordinates | centered, unit variance |
| `dense_white` | random orthonormal direction in ZCA-whitened space (ridge) | centered, unit variance |
| `ksparse_{2,5,20,100}` | k random neurons, Gaussian weights on z-scored neurons | centered, unit variance |
| `related_{5,20}` | a seed neuron + its k−1 most correlated neurons, signs aligned so parts co-fire (**positive control**) | centered, unit variance |

`related_k` should be interpretable if "grouped because related" is what matters, while `ksparse_k` mixes unrelated
neurons; the k-dose-response shows where interpretability breaks (if it does).

## Pipeline

| # | stage | script | where | needs → produces |
|---|---|---|---|---|
| 0 | tokenize corpus | `00_prepare_corpus.py` | anywhere with internet | HF datasets → `data/corpus/{train,holdout}` |
| 1 | stats + generate units | `01_stats_and_units.py` | **GPU job** | corpus → `data/stats`, `data/units`, `private/blind_key.json` |
| 2a | cache (train) | `02_cache.py --split train` | **GPU job**, hours, resumable | → `data/cache/train` |
| 2b | cache (holdout) | `02_cache.py --split holdout` | **GPU job**, short | → `data/cache/holdout` |
| 3 | export blinded agent views | `03_export_agent_view.py` | CPU (needs tokenizer, corpus) | caches → `data/agent_cache`, private ground truth |
| 4 | agent-free pilot | `04_pilot.py` | CPU | caches → `data/results/pilot_*` |
| 5a | agents: explain | `05_run_agents.py --stage explain` | CPU + Claude | views → `data/agent_results/explain` |
| 6a | probe round 1 | `06_build_probe_job.py --round 1` → **GPU job** `07_probe.py` | | explanations → probe scores |
| 5b | agents: blindmax, simulate | `05_run_agents.py --stage blindmax|simulate` | CPU + Claude | |
| 6b | probe round 2 | `06_build_probe_job.py --round 2` → **GPU job** | | blind-max texts → probe scores |
| 8 | score + analyze | `08_score_and_analyze.py --probe r1.jsonl r2.jsonl` | CPU | → `data/results/{equivalence.csv,summary.md,dose_response.png,...}` |

GPU jobs: **5** (stats_units, cache_train, cache_holdout, probe r1, probe r2). Everything else is local.
`python scripts/make_job.py <stage> --name X` writes `jobs/pending/NNN_X/{run.sbatch,request.md,COMMANDS.txt}`;
nothing touches the cluster — you review, then run `COMMANDS.txt` yourself (2FA included). See `docs/PROTOCOL.md`.

## Quick start

```bash
pip install -e ".[gpu,dev]"            # python>=3.10
pytest -q                              # 1) numpy tests  2) torch tiny-model tests (CPU, ~1-2 min)  <- do this first

# on a machine with internet (edit configs/default.yaml first: model.name, cluster.*; dataset ids are unverified!)
huggingface-cli download Qwen/Qwen3-8B --local-dir /path/to/Qwen3-8B     # then set model.name to that path
python scripts/00_prepare_corpus.py

python scripts/make_job.py stats_units   --name stats_units    # review, run COMMANDS.txt, fetch results
python scripts/make_job.py cache_holdout --name cache_holdout
python scripts/make_job.py cache_train   --name cache_train

python scripts/03_export_agent_view.py
python scripts/04_pilot.py                                        # agent-free sanity check, cheap
python scripts/05_run_agents.py --stage explain --limit 5         # trial: check cost in data/agent_results/explain/_run_log.jsonl
python scripts/05_run_agents.py --stage explain
python scripts/06_build_probe_job.py --round 1
python scripts/make_job.py probe --name probe_r1 --input data/probe/round1_in.jsonl     # run, fetch jobs/done/.../probe_out.jsonl
python scripts/05_run_agents.py --stage blindmax
python scripts/05_run_agents.py --stage simulate
python scripts/06_build_probe_job.py --round 2
python scripts/make_job.py probe --name probe_r2 --input data/probe/round2_in.jsonl
python scripts/08_score_and_analyze.py --probe jobs/done/*probe_r1/probe_out.jsonl jobs/done/*probe_r2/probe_out.jsonl
```
Dry-run everything without a GPU or API spend: `--config configs/smoke.yaml` (agent backend `fake`).

## What the agents do (and how the comparison is kept fair)

* **Blinding.** Agents get an opaque id (`u01234`), the layer, and examples. Never the condition, k, neurons, kurtosis, or raw
  activation scales (only *fractions of the unit's corpus maximum*). The key lives in `private/` and must never be inside an
  agent workspace. An automated test fails if condition names leak into a view. Identical instructions for all conditions.
* **Explainer** sees top examples (one per document) with per-token activation tags, tiered examples (strong/moderate/weak),
  random-token baselines, token statistics, the other pole. It writes: explanation, `fires_on`, `parts` (independent triggers →
  compactness), confidence, and 5 predicted-fire / 5 near-miss / 5 distant texts **before any measurement, with no feedback**.
* **Blind-max writer** sees only the explanation and writes 3 texts meant to maximally fire the unit (essay's "blind maximum").
* **Simulator** sees only the explanation and 20 held-out contexts (10 strong from the held-out corpus, 10 random) and scores 0-10.
* No revision loop in v1 (feedback would turn the measurement into an optimisation and add a GPU round).

Metrics per unit (primary pole; `r` = signed activation / the unit's corpus max): `rate_pred`, `rate_near`, `rate_distant`
(fraction with `r >= 0.3`), `rate_null` (other units' texts: chance), `rate_rand` (random document prefixes), `margin = rate_pred - rate_near`,
`blind_max_logratio`, `sim_spearman`, plus `n_parts` (compactness).

**Statistics** (`analyze.py`): per condition vs `neuron`, layer-stratified bootstrap CI of the difference in each metric; TOST-style
verdict (`equivalent` / `combo_worse` / `combo_better` / `inconclusive`) against margins **fixed in the config before you see
agent results** (`analysis.equiv_margin`). Also OLS effects adjusting for layer and log-kurtosis, a breakdown by `n_parts`, and a
dose-response plot. "Equally interpretable" is only meaningful at comparable complexity, so check `n_parts_breakdown.csv`.

## Design decisions (all editable in `configs/default.yaml`)

| choice | value | why / caveat |
|---|---|---|
| model, layers | Qwen3-8B, 9/18/27 | below 8B neurons get broader (essay); fits one 40-80 GB GPU. Swap freely (any Llama/Qwen-style MLP) |
| position 0 | excluded everywhere | attention-sink massive activations would own every top-k list |
| units cached / explained | 200 / 40 per (layer, condition) | caching is cheap, agents are the cost: 40 × 9 conds × 3 layers = 1080 explain sessions. Pilot uses all 1800/layer |
| neuron scale | raw, uncentered | essay: trust natural ranges. Combos are normalised for comparability — a stated asymmetry |
| `dense_raw` vs `dense_white` | both | raw random directions are dominated by high-variance directions (massive-activation features); whitening removes that; compare the two |
| `related_k` | top-correlated by Pearson | Pearson is outlier-sensitive; position 0 excluded to limit that |
| corpus | 50M train tokens, mixed web/edu/code/7 languages, doc-aligned, 2048 ctx | long context matters (essay); holdout split by `doc_index % 10` |
| cache | top-1024 (value+gid), top-64 with ±(48,16)-token activation windows, 200k uniform tokens, moments | one pass gives everything agents need; no second GPU pass for windows |
| strong firing | `r >= 0.3` of corpus max | essay's convention |
| agent runner | `claude -p` headless (CLI), model alias `sonnet` | flags in `agents/run.py::_claude` — verify with `claude --help` |

## Caveats you should know about

1. Torch-side code is verified only by the test suite you run (see top). Cluster-specific issues (module names, offline HF) are in `cluster.setup`.
2. HF dataset ids/configs in the config could not be checked offline.
3. Agent isolation is by folder, not by OS: an agent with `Bash` could in principle read `private/`. For a clean study run agents in a container/separate user with only `data/agent_work` mounted.
4. 9 conditions × 4 metrics = 36 tests, no multiplicity correction; the pre-registered margins and the TOST framing are the guard. Units are treated as independent.
5. The explainer, blind-max writer and simulator are the same model family: shared blind spots (e.g. the noun bias the essay describes) affect all conditions but not necessarily equally.
6. Interventions (`interventions.py`: ablate/steer/clamp a unit) are implemented and tested but not wired into the agent rounds — steering is the essay's strongest evidence type and a natural v2.

## Layout

```
configs/        default.yaml (full study), smoke.yaml (tiny)
src/combo_interp/
  corpus.py     tokenized doc-aligned store + builder      units.py     unit generation (numpy)
  hooks.py      MLP hidden-activation capture               stats.py     pass 1
  topk.py       streaming top-k (GPU)                       cache.py     pass 2
  probe.py      GPU text scoring                            export.py    blinded views + ground truth
  pilot.py      agent-free metrics                          scoring.py   probe jobs + per-unit scores
  analyze.py    equivalence statistics, plots               interventions.py
  agents/       prompts, validation, runner, unit_tools.py (stdlib helper copied into each workspace)
scripts/        00..08 stage CLIs, make_job.py              jobs/pending|done   handoff folders
tests/          numpy tests, synthetic pipeline, torch tiny-model verification
docs/PROTOCOL.md   the manual cluster handoff            CLAUDE.md   rules for an agent working in this repo
```
