"""Centroid test (combo_interp.geometry) on a planted mixture, and MLP-input capture on a tiny random Llama."""
import tempfile

import numpy as np

from .helpers import need


def _mixture(K=16, d=32, n_per=200, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal((K, d))
    mu /= np.linalg.norm(mu, axis=1, keepdims=True)
    y = np.repeat(np.arange(K), n_per)
    X = mu[y] + noise * rng.standard_normal((len(y), d)) / np.sqrt(d)
    return X, y, mu, rng


def test_nmi_basics():
    from combo_interp.geometry import nmi
    a = np.array([0, 0, 1, 1, 2, 2])
    assert abs(nmi(a, a) - 1) < 1e-9
    assert abs(nmi(a, 5 - a) - 1) < 1e-9          # label names do not matter
    assert nmi(np.zeros(6, int), a) == 0.0


def test_planted_centroids_beat_nulls():
    need("torch")
    from combo_interp.geometry import centroid_eval
    X, y, mu, rng = _mixture()
    # "gate" = the true cluster means, "up" = random directions
    Wg = mu
    Wu = rng.standard_normal(Wg.shape)
    fit = rng.random(len(y)) < 0.5
    rows = centroid_eval(X, Wg, Wu, {"cluster": y}, fit, Ms=[16], reps=3, combo_ks=(5,), center=False,
                         log=lambda *_: None)
    med = {s: np.median([r["obj"] for r in rows if r["set"] == s]) for s in {r["set"] for r in rows}}
    nmi = {s: np.median([r["nmi_cluster"] for r in rows if r["set"] == s]) for s in med}
    for null in ["rand_iso", "rand_cov", "rand_rot", "up", "combo_5"]:
        assert med["gate"] > med[null] + 0.05, (null, med)
        assert nmi["gate"] > nmi[null], (null, nmi)
    assert med["kmeans"] > med["rand_cov"]
    assert nmi["kmeans"] > 0.5


def test_euclid_assignment():
    need("torch")
    from combo_interp.geometry import centroid_eval
    X, y, mu, rng = _mixture()
    Wu = rng.standard_normal(mu.shape)
    fit = rng.random(len(y)) < 0.5
    kw = dict(Ms=[16], reps=3, combo_ks=(5,), center=False, log=lambda *_: None)
    rows = centroid_eval(3 * X, mu, Wu, {"cluster": y}, fit, assign_rules=("cos", "euclid"), **kw)
    eu = [r for r in rows if r["assign"] == "euclid"]
    assert len(eu) == len(rows) // 2
    med = {s: np.median([r["ve_raw"] for r in eu if r["set"] == s]) for s in {r["set"] for r in eu}}
    dist = {s: np.median([r["dist"] for r in eu if r["set"] == s]) for s in med}
    for null in ["rand_iso", "rand_cov", "rand_rot", "up", "combo_5"]:
        assert med["gate"] > med[null] + 0.05, (null, med)
    assert dist["kmeans"] < dist["rand_cov"]
    # the cos rows are unchanged by also running euclid
    cos_only = centroid_eval(3 * X, mu, Wu, {"cluster": y}, fit, **kw)
    assert [r["obj"] for r in cos_only if r["set"] == "gate"] == [r["obj"] for r in rows
                                                                if r["assign"] == "cos" and r["set"] == "gate"]


def test_ip_assignment():
    need("torch")
    from combo_interp.geometry import centroid_eval
    X, y, mu, rng = _mixture()
    norms = rng.uniform(0.5, 2.0, len(mu))[:, None]       # native norms vary; ip must still find the planted means
    fit = rng.random(len(y)) < 0.5
    rows = centroid_eval(X, mu * norms, rng.standard_normal(mu.shape), {"cluster": y}, fit, Ms=[16], reps=3,
                         combo_ks=(5,), center=False, log=lambda *_: None, assign_rules=("cos", "ip"))
    ip = [r for r in rows if r["assign"] == "ip"]
    assert {r["set"] for r in ip} == {"gate", "up", "rand_iso", "rand_rot", "rand_gcov", "combo_5"}
    med = {s: np.median([r["ve_raw"] for r in ip if r["set"] == s]) for s in {r["set"] for r in ip}}
    for null in ["rand_iso", "rand_rot", "rand_gcov", "up", "combo_5"]:
        assert med["gate"] > med[null] + 0.05, (null, med)
    assert any(r["set"] == "rand_gcov" for r in rows if r["assign"] == "cos")


def test_capture_inputs_match_gate():
    need("torch"); need("transformers")
    import torch

    from combo_interp.corpus import CorpusStore
    from combo_interp.geometry import capture_inputs
    from combo_interp.hooks import MLPCapture

    from .synth import make_corpus
    from .test_torch_tiny import _model
    model = _model()
    with tempfile.TemporaryDirectory() as tmp:
        make_corpus(tmp + "/c", 6, 40, 0)
        store = CorpusStore(tmp + "/c")
        gids = np.array([0 * store.seq_len + 3, 2 * store.seq_len + 7, 5 * store.seq_len + 1, 2 * store.seq_len + 2])
        X = capture_inputs(model, store, gids, [1, 2], max_batch_tokens=64, log=lambda *_: None)
        for L in [1, 2]:
            for i, g in enumerate(gids):
                s, t = divmod(int(g), store.seq_len)
                with MLPCapture(model, [L], capture_gate_up=True) as cap:
                    cap.run(torch.tensor(store.get_tokens(s, 0, store.seq_length(s)))[None])
                g_ref = cap.g[L][0, t]
                g_hat = model.model.layers[L].mlp.gate_proj(torch.tensor(X[L][i], dtype=torch.float32))
                assert torch.allclose(g_hat, g_ref, atol=2e-2), (L, i)    # X is stored as float16
