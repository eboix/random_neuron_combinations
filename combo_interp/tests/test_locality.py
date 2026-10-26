"""Experiment B helpers (combo_interp.locality): set metrics, mediation, and the one-pass capture on a tiny Llama."""
import tempfile

import numpy as np
import pandas as pd

from .helpers import need


def _unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_coherence_and_pr_match_bruteforce():
    from combo_interp.locality import coherence, participation_ratio
    rng = np.random.default_rng(0)
    X = _unit(rng.standard_normal((30, 8)))
    G = X @ X.T
    assert abs(coherence(X) - G[~np.eye(30, dtype=bool)].mean()) < 1e-9
    ev = np.linalg.eigvalsh(G)
    assert abs(participation_ratio(X) - ev.sum() ** 2 / (ev ** 2).sum()) < 1e-6


def test_set_metrics_find_modes():
    from combo_interp.locality import set_metrics
    rng = np.random.default_rng(1)
    d = 64
    centers = _unit(rng.standard_normal((3, d)))
    X3 = _unit(np.repeat(centers, 40, axis=0) + 0.05 * rng.standard_normal((120, d)))
    X1 = _unit(centers[:1] + 0.05 * rng.standard_normal((120, d)))
    Xr = _unit(rng.standard_normal((120, d)))
    m3, m1, mr = (set_metrics(X, rng) for X in (X3, X1, Xr))
    assert m3["k_star"] == 3 and m3["sil"] > 0.5
    assert mr["k_star"] == 1                       # no structure -> silhouettes below SIL_MIN
    assert m1["coh"] > m3["coh"] > mr["coh"]
    assert m1["pr"] > m3["pr"] or m3["pr"] < mr["pr"]


def test_mediation_recovers_full_attenuation():
    from combo_interp.locality import mediation
    rng = np.random.default_rng(2)
    n = 60
    df = pd.DataFrame({"cond": np.repeat(["neuron", "combo"], n), "layer": np.tile([7, 14], n)})
    df["m"] = np.where(df.cond == "neuron", 1.0, 0.0) + 0.3 * rng.standard_normal(2 * n)
    df["y"] = 2.0 * df.m + 0.1 * rng.standard_normal(2 * n)       # y depends on the condition only through m
    r = {x["term"]: x for x in mediation(df, "y", "cond", "neuron", "m", n_boot=200)}
    t = r["cond=combo"]
    assert t["effect_total"] < -1.5 and abs(t["effect_direct"]) < 0.2 and t["attenuation"] > 0.85
    assert abs(r["slope:m"]["effect_direct"] - 2.0) < 0.2


def test_capture_pass_matches_bruteforce():
    need("torch"); need("transformers")
    import torch

    from combo_interp.corpus import CorpusStore
    from combo_interp.hooks import MLPCapture
    from combo_interp.locality import capture_pass

    from .synth import make_corpus
    from .test_torch_tiny import _model
    model = _model()
    with tempfile.TemporaryDirectory() as tmp:
        make_corpus(tmp + "/c", 8, 40, 0)
        store = CorpusStore(tmp + "/c")
        Lq = store.seq_len
        want = {1: np.array(sorted([0 * Lq + 3, 2 * Lq + 7, 5 * Lq + 1, 7 * Lq + 2]))}
        R = {2: np.random.default_rng(0).standard_normal((32, 3)).astype(np.float32)}
        X, ctrl = capture_pass(model, store, want, R, n_top=4, skip_first=1, max_batch_tokens=64, log=lambda *_: None)
        xs = {}
        for s in range(store.n_seqs):
            with MLPCapture(model, [1, 2], capture_input=True) as cap:
                cap.run(torch.tensor(store.get_tokens(s, 0, store.seq_length(s)))[None])
            xs[s] = {L: cap.x[L][0].float().numpy() for L in (1, 2)}
        for i, g in enumerate(want[1]):
            s, t = divmod(int(g), Lq)
            assert np.allclose(X[1][i], xs[s][1][t], atol=2e-2)
        for j in range(3):                     # per direction: top-4 documents by max_{t>=1} r.x
            best = sorted(((float((xs[s][2][1:] @ R[2][:, j]).max()), s) for s in xs), reverse=True)[:4]
            assert np.allclose(ctrl[2]["vals"][:, j], [v for v, _ in best], atol=1e-3)
            for r_, (v, s) in enumerate(best):
                s2, t = divmod(int(ctrl[2]["gid"][r_, j]), Lq)
                assert s2 == s
                assert np.allclose(ctrl[2]["x"][r_, j], xs[s][2][t], atol=2e-2)
