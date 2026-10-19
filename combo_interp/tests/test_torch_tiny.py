"""Verifies the GPU-side code against brute-force recomputation on a tiny random Llama (runs on CPU, ~1 min).
RUN THIS FIRST on your own machine:   pytest -q tests/test_torch_tiny.py
(It could not be executed in the sandbox the code was written in: no torch there.)"""
import json
import tempfile

import numpy as np

from combo_interp.corpus import CorpusStore
from combo_interp.units import load_unit_set

from .helpers import need, smoke_cfg
from .synth import FakeTokenizer, make_corpus


def _model():
    need("torch"); need("transformers")
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    c = LlamaConfig(vocab_size=259, hidden_size=32, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
                    num_key_value_heads=4, max_position_embeddings=256, pad_token_id=0)
    return LlamaForCausalLM(c).eval()


def _world(tmp):
    from combo_interp.cache import run_cache
    from combo_interp.stats import run_stats
    from combo_interp.units import run_make_units
    cfg, paths = smoke_cfg(tmp)
    make_corpus(paths.corpus("train"), 60, 96, 0)
    make_corpus(paths.corpus("holdout"), 20, 96, 1)
    model, tok = _model(), FakeTokenizer()
    q = lambda *_: None
    run_stats(cfg, paths, model=model, tok=tok, log=q)
    run_make_units(cfg, paths, log=q)
    return cfg, paths, model, tok, q


def _brute(cfg, paths, model, layer):
    import torch
    from combo_interp.hooks import MLPCapture
    store = CorpusStore(paths.corpus("train"))
    us = load_unit_set(paths, layer)
    A, b = torch.tensor(us["A"]), torch.tensor(us["b"])
    skip, Lq = int(cfg.cache.skip_first_tokens), store.seq_len
    Z, gids = [], []
    with MLPCapture(model, [layer]) as cap:
        for seq in range(store.n_seqs):
            n = store.seq_length(seq)
            cap.run(torch.tensor(store.get_tokens(seq, 0, n))[None])
            Z.append((cap.p[layer][0].float() @ A + b)[skip:].numpy())
            gids += [seq * Lq + t for t in range(skip, n)]
    return np.concatenate(Z), np.array(gids), us


def test_cache_matches_bruteforce_and_resume():
    need("torch"); need("transformers")
    import torch
    from combo_interp.cache import run_cache
    with tempfile.TemporaryDirectory() as tmp:
        cfg, paths, model, tok, q = _world(tmp)
        run_cache(cfg, paths, "train", model=model, tok=tok, log=q, _stop_after=2)     # "timeout" after 2 batches
        assert (paths.data / "cache" / "train" / "ckpt.pt").exists()
        run_cache(cfg, paths, "train", model=model, tok=tok, log=q)                    # resume -> finish
        assert not (paths.data / "cache" / "train" / "ckpt.pt").exists()
        for L in cfg.model.layers:
            Z, gids, us = _brute(cfg, paths, model, int(L))
            big = np.load(paths.cache("train", int(L)) / "topk_big.npz")
            K = 5
            assert np.allclose(big["pos_vals"][:, :K], -np.sort(-Z, 0)[:K].T, rtol=1e-4, atol=1e-5)
            assert np.allclose(big["neg_vals"][:, :K], np.sort(Z, 0)[:K].T, rtol=1e-4, atol=1e-5)
            assert np.array_equal(big["pos_gid"][:, 0], gids[Z.argmax(0)])
            assert np.array_equal(big["neg_gid"][:, 0], gids[Z.argmin(0)])
            prof = np.load(paths.cache("train", int(L)) / "topk_prof.npz")
            ci = int(np.where(prof["offsets"] == 0)[0][0])
            assert np.allclose(prof["pos_win"][:, :, ci], prof["pos_vals"], rtol=1e-4, atol=1e-5)
            assert np.allclose(prof["neg_win"][:, :, ci], prof["neg_vals"], rtol=1e-4, atol=1e-5)
            # the window really is the unit's activation sequence around the token
            g2r = {int(g): i for i, g in enumerate(gids)}
            m, r = 0, 0
            row = g2r[int(prof["pos_gid"][m, r])]
            for wi, o in enumerate(prof["offsets"]):
                v = prof["pos_win"][m, r, wi]
                if np.isfinite(v):
                    assert abs(v - Z[row + int(o), m]) < 1e-4
            # SwiGLU identity at the firing token of every neuron unit: silu(gate) * value == activation
            for u in us["meta"]:
                if u["kind"] == "neuron":
                    g, up, val = (prof["pos_gate"][u["col"], 0], prof["pos_up"][u["col"], 0], prof["pos_vals"][u["col"], 0])
                    assert abs(g / (1 + np.exp(-g)) * up - val) < 1e-4, (u["unit_id"], g, up, val)
            uni = np.load(paths.cache("train", int(L)) / "uniform.npz")
            rows = [g2r[int(g)] for g in uni["gid"]]
            assert len(rows) > 0 and np.allclose(uni["acts"].astype(np.float32), Z[rows], rtol=2e-3, atol=2e-3)
            mom = np.load(paths.cache("train", int(L)) / "moments.npz")
            assert int(mom["n"]) == len(Z) and np.allclose(mom["s1"], Z.sum(0), rtol=1e-4, atol=1e-3)
            assert np.allclose(mom["s2"], (Z.astype(np.float64) ** 2).sum(0), rtol=1e-4, atol=1e-3)


def test_downstream_on_real_cache_and_probe():
    """Cache from the torch pass -> export -> fake agents -> probe via torch -> scores (full chain)."""
    need("torch"); need("transformers")
    import torch
    from combo_interp.agents.run import run_stage
    from combo_interp.analyze import run_analysis
    from combo_interp.cache import run_cache
    from combo_interp.export import build_agent_cache
    from combo_interp.hooks import MLPCapture
    from combo_interp.io_utils import read_json, read_jsonl
    from combo_interp.probe import run_probe
    from combo_interp.scoring import build_probe_job, compute_scores
    with tempfile.TemporaryDirectory() as tmp:
        cfg, paths, model, tok, q = _world(tmp)
        run_cache(cfg, paths, "train", model=model, tok=tok, log=q)
        run_cache(cfg, paths, "holdout", model=model, tok=tok, log=q)
        build_agent_cache(cfg, paths, tok, log=q)
        run_stage(cfg, paths, "explain", log=q)
        p1 = build_probe_job(cfg, paths, 1, tok, log=q)
        o1 = paths.data / "probe" / "round1_out.jsonl"
        run_probe(cfg, paths, p1, o1, model=model, tok=tok, log=q)
        # probe == direct computation
        rows = read_jsonl(o1)
        r = rows[7]
        L, col = next((int(L), m["col"]) for L in cfg.model.layers for m in load_unit_set(paths, int(L))["meta"]
                      if m["unit_id"] == r["unit_id"])
        us = load_unit_set(paths, L)
        text = next(x["text"] for x in read_jsonl(p1) if x["probe_id"] == r["probe_id"])
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with MLPCapture(model, [L]) as cap:
            cap.run(torch.tensor(ids)[None])
            z = (cap.p[L][0].float() @ torch.tensor(us["A"][:, col]) + float(us["b"][col])).numpy()
        assert np.allclose(r["acts"], z, rtol=1e-3, atol=1e-3) and abs(r["max_act"] - z[1:].max()) < 1e-3
        run_stage(cfg, paths, "blindmax", log=q)
        p2 = build_probe_job(cfg, paths, 2, tok, log=q)
        o2 = paths.data / "probe" / "round2_out.jsonl"
        run_probe(cfg, paths, p2, o2, model=model, tok=tok, log=q)
        run_stage(cfg, paths, "simulate", log=q)
        df = compute_scores(cfg, paths, [o1, o2], log=q)
        assert len(df) > 0 and df.margin.notna().any()
        run_analysis(cfg, paths, log=q)

        # calibration via torch == direct computation on the same windows
        from combo_interp.calibrate import load_calibration, run_calibration, sample_windows, window_thresholds
        run_calibration(cfg, paths, model=model, tok=tok, log=q)
        C = cfg.calibration
        rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), 31337]))
        ids = sample_windows(CorpusStore(paths.corpus(C.split)), int(C.n_windows), int(C.window), rng)
        with MLPCapture(model, [L]) as cap:
            cap.run(torch.tensor(ids))
            z = (cap.p[L].float() @ torch.tensor(us["A"]) + torch.tensor(us["b"])).numpy()
        cal = load_calibration(paths, L)
        skip = int(cfg.cache.skip_first_tokens)
        assert np.allclose(cal["thr_pos"], window_thresholds(z, skip, C.quantiles), rtol=1e-3, atol=1e-3, equal_nan=True)
        cfg_q, _ = smoke_cfg(tmp, ["scoring.threshold=random_quantile"])
        dq = compute_scores(cfg_q, paths, [o1, o2], log=q)
        assert dq.rate_rand.notna().any() and dq.rate_rand.mean() < 0.3


def test_interventions():
    need("torch")
    import torch
    from combo_interp.interventions import make_unit_hook
    torch.manual_seed(0)
    p = torch.randn(4, 7, 96)
    a = torch.randn(96); b = 0.3
    z0 = p @ a + b
    z_abl = make_unit_hook(a, b, "ablate")(None, (p,))[0] @ a + b
    z_st = make_unit_hook(a, b, "steer", alpha=2.5)(None, (p,))[0] @ a + b
    z_cl = make_unit_hook(a, b, "clamp", alpha=-1.0)(None, (p,))[0] @ a + b
    assert torch.allclose(z_abl, torch.zeros_like(z0), atol=1e-4)
    assert torch.allclose(z_st, z0 + 2.5, atol=1e-4)
    assert torch.allclose(z_cl, torch.full_like(z0, -1.0), atol=1e-4)
    # orthogonal directions are untouched
    v = torch.randn(96); v = v - (v @ a) / (a @ a) * a
    h = make_unit_hook(a, b, "ablate")(None, (p,))[0]
    assert torch.allclose(h @ v, p @ v, atol=1e-4)
