"""End-to-end run of everything except the torch model passes, using synthetic activations that follow the
exact on-disk schema of the real cache. Needs pandas/scipy/matplotlib only."""
import json
import tempfile

import numpy as np
import pandas as pd

from combo_interp.agents.run import run_stage
from combo_interp.analyze import run_analysis
from combo_interp.export import build_agent_cache
from combo_interp.io_utils import read_json
from combo_interp.pilot import pilot_metrics, summarize_pilot
from combo_interp.scoring import build_probe_job, compute_scores

from .helpers import smoke_cfg
from .synth import (FakeTokenizer, SynthModel, emulate_cache, emulate_calibration, emulate_probe,
                    emulate_stats_and_units, make_corpus)


def build_world(tmp):
    cfg, paths = smoke_cfg(tmp)
    make_corpus(paths.corpus("train"), 120, 96, 0)
    make_corpus(paths.corpus("holdout"), 40, 96, 1)
    model, tok = SynthModel(96, seed=3), FakeTokenizer()
    emulate_stats_and_units(cfg, paths, model)
    emulate_cache(cfg, paths, model, "train")
    emulate_cache(cfg, paths, model, "holdout")
    return cfg, paths, model, tok


def test_full_downstream():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, paths, model, tok = build_world(tmp)

        # ---- export / blinding
        build_agent_cache(cfg, paths, tok, log=lambda *_: None)
        index, sel = read_json(paths.agent_cache / "index.json"), read_json(paths.agent_cache / "selection.json")
        ref, truth = read_json(paths.unit_ref), read_json(paths.holdout_truth)
        assert len(index) > 100 and len(sel) == 6 * 6 * 2           # 6 per (layer, condition)
        assert set(sel) <= set(ref)
        blob = (paths.agent_cache / f"{sel[0]}.json").read_text() + (paths.agent_cache / f"{sel[0]}.md").read_text()
        for leak in ("dense", "ksparse", "related", "neuron", "kurtosis", "cond", "rarity", "distinct"):  # agents must not see these
            assert leak not in blob.lower(), f"leak: {leak}"
        assert "frac_tokens_ge_10pct_of_max" in ref[sel[0]]                          # kept privately instead
        d = read_json(paths.agent_cache / f"{sel[0]}.json")
        assert d["examples"]["primary"] and "[[" in d["examples"]["primary"][0]["text"]
        assert {ref[b]["primary"] for b in sel} == {1, -1}, "need both poles to test sign handling"
        for b in sel:                                                               # both poles: fractions in [0, 1]
            v = read_json(paths.agent_cache / f"{b}.json")
            assert all(0 <= e["frac"] <= 1.0001 for e in v["examples"]["primary"]), (b, ref[b]["primary"])
            assert max(truth[b]) > 0.3, (b, ref[b]["primary"], truth[b])          # strong holdout items stay positive
        assert len(truth[sel[0]]) == len(read_json(paths.agent_cache / "holdout" / f"{sel[0]}.json"))

        groups = read_json(paths.holdout_groups)                                    # simulator item groups
        nt, nr = int(cfg.export.holdout_top_items), int(cfg.export.holdout_random_items)
        assert all(groups[b].count("top") == nt and groups[b].count("random") == nr for b in sel)
        assert all(len(groups[b]) == len(truth[b]) for b in sel)

        # ---- agents (fake backend): explain -> write, simulate -> one probe round (numpy emulation of the GPU job)
        quiet = lambda *_: None
        run_stage(cfg, paths, "explain", log=quiet)
        assert len(list(paths.agent_results("explain").glob("u*.json"))) == len(sel)
        assert "predicted_fire" not in read_json(next(paths.agent_results("explain").glob("u*.json")))
        run_stage(cfg, paths, "write", log=quiet)
        run_stage(cfg, paths, "simulate", log=quiet)
        p1 = build_probe_job(cfg, paths, tok, log=quiet)
        rows = [json.loads(l) for l in open(p1)]
        assert {r["kind"] for r in rows} == {"blindmax", "pred", "near", "distant", "null", "rand"}
        o1 = paths.data / "probe" / "probe_out.jsonl"
        emulate_probe(cfg, paths, p1, o1, model, tok)
        o2 = o1                                                                     # (single round now)

        # ---- scores + analysis
        df = compute_scores(cfg, paths, [o1], log=quiet)
        assert len(df) == len(sel)
        for c in ("margin", "blind_max_logratio", "sim_spearman", "sim_spearman_top", "sim_spearman_random", "sim_auc",
                  "sim_rating_top", "sim_rating_random", "rate_null", "rate_rand", "n_parts", "n_queries"):
            assert c in df.columns, c
        assert df.sim_auc.between(0, 1).all() and abs(df.sim_auc.mean() - 0.5) < 0.15   # fake simulator rates at random
        neu = df[df.cond == "neuron"]
        assert neu.rate_pred.mean() > neu.rate_rand.mean() + 0.2, "fake explainer should beat random text on neurons"
        assert neu.margin.mean() > 0.2
        eq = run_analysis(cfg, paths, log=quiet)
        assert set(eq.cond) == set(df.cond) - {"neuron"}
        res = paths.results / "corpus_frac_0.3"
        assert (res / "dose_response.png").exists() and (res / "summary.md").exists()

        # ---- per-unit thresholds calibrated on random text: every unit crosses on ~5% of random texts
        cfg_q, _ = smoke_cfg(tmp, ["scoring.threshold=random_quantile"])
        emulate_calibration(cfg_q, paths, model)
        dq = compute_scores(cfg_q, paths, [o1], log=quiet)
        assert (paths.scores / "random_q0.95" / "unit_scores.csv").exists() and (dq.threshold == "random_q0.95").all()
        assert dq.rate_rand.mean() < 0.2, dq.groupby("cond").rate_rand.mean().to_dict()
        assert (dq.blind_max_logratio.fillna(0) == df.blind_max_logratio.fillna(0)).all()   # threshold-free metric unchanged
        run_analysis(cfg_q, paths, log=quiet)
        assert (paths.results / "random_q0.95" / "summary.md").exists()
        print("rate_rand by cond, corpus_frac vs random_q0.95:",
              df.groupby("cond").rate_rand.mean().round(2).to_dict(), dq.groupby("cond").rate_rand.mean().round(2).to_dict())

        # ---- agent-free pilot: neurons fire on specific tokens, dense directions do not
        pdf = pilot_metrics(cfg, paths, log=quiet)
        med = pdf.groupby("cond").tok_entropy.median()
        assert med["neuron"] < med["dense_raw"], med.to_dict()
        assert len(summarize_pilot(pdf, n_boot=50)) > 0
        print("pilot median token entropy by condition:", {k: round(v, 3) for k, v in med.items()})
        print("fake-explainer margin by condition:", df.groupby("cond").margin.mean().round(2).to_dict())


def test_claude_backend_refuses_workspace_under_claude_md():
    from pathlib import Path
    import pytest
    from combo_interp.agents.run import instruction_files_above
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "CLAUDE.md").write_text("project rules")
        cfg, paths = smoke_cfg(tmp, ["agents.backend=claude"])
        assert instruction_files_above(paths.agent_work("explain").parent) == [Path(tmp).resolve() / "CLAUDE.md"]
        with pytest.raises(RuntimeError, match="breaks blinding"):
            run_stage(cfg, paths, "explain", log=lambda *_: None)
