"""Pure-numpy tests (no torch needed)."""
import numpy as np

from combo_interp.agents.run import strip_marks
from combo_interp.analyze import verdict
from combo_interp.corpus import CorpusStore
from combo_interp.export import dedupe_hits, pick_spread, render_marked, tier_of
from combo_interp.stats import cov_from_sums
from combo_interp.units import make_blind_key, make_unit_set, zca_matrix

from .helpers import smoke_cfg
from .synth import make_corpus
import tempfile


def _acts(d=64, N=20000, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((N, 8))
    P = np.stack([np.maximum(0, 0.7 * base[:, j % 8] + 0.7 * rng.standard_normal(N) - 1.0) for j in range(d)], 1)
    return P


def test_cov_from_sums():
    P = _acts()
    mean, cov = cov_from_sums(len(P), P.sum(0), P.T @ P)
    assert np.allclose(mean, P.mean(0)) and np.allclose(cov, np.cov(P.T, bias=True))


def test_units_properties():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, _ = smoke_cfg(tmp)
    P = _acts()
    mean, cov = P.mean(0), np.cov(P.T, bias=True)
    us = make_unit_set(3, mean, cov, cfg.units, 0, zca_matrix(cov, 0.05, "numpy"))
    A, b = us["A"].astype(float), us["b"].astype(float)
    Z = P @ A + b
    conds = np.array([m["cond"] for m in us["meta"]])
    assert np.allclose(Z.std(0), us["std_u"], rtol=1e-3) and np.allclose(Z.mean(0), us["mean_u"], atol=1e-6)
    for c in set(conds) - {"neuron"}:
        assert np.allclose(Z[:, conds == c].std(0), 1.0, atol=1e-3), c            # unit variance
        assert np.allclose(Z[:, conds == c].mean(0), 0.0, atol=1e-6), c           # centered
    for i in np.where(conds == "neuron")[0]:                                       # neuron == raw column
        assert np.allclose(Z[:, i], P[:, us["meta"][i]["neurons"][0]])
    nnz = (A != 0).sum(0)
    assert all(nnz[conds == "ksparse_5"] == 5) and all(nnz[conds == "ksparse_2"] == 2)
    assert len({m["unit_id"] for m in us["meta"]}) == len(us["meta"])
    # determinism
    us2 = make_unit_set(3, mean, cov, cfg.units, 0, zca_matrix(cov, 0.05, "numpy"))
    assert np.array_equal(us["A"], us2["A"])
    # blind key hides structure and is a bijection
    ids = [m["unit_id"] for m in us["meta"]]
    key = make_blind_key(ids, 0)
    assert sorted(key.values()) == sorted(ids)


def test_dedupe_and_spread():
    Lq = 100
    gids = np.array([5 * Lq + 10, 5 * Lq + 12, 7 * Lq + 3, -1, 9 * Lq + 50])
    vals = np.array([5.0, 4.0, 3.0, -np.inf, 1.0])
    out = dedupe_hits(vals, gids, Lq, max_per_seq=1)
    assert [g for _, _, g in out] == [5 * Lq + 10, 7 * Lq + 3, 9 * Lq + 50]
    out2 = dedupe_hits(vals, gids, Lq, max_per_seq=2, min_gap=1)
    assert 5 * Lq + 12 in [g for _, _, g in out2]
    assert len(pick_spread(list(range(100)), 10)) == 10 and pick_spread([1, 2], 10) == [1, 2]
    assert tier_of(0.6, [0.5, 0.3, 0.1]) == "strong" and tier_of(0.05, [0.5, 0.3, 0.1]) is None


def test_render_and_strip():
    s = render_marked(["a", " b", "\n", "c"], [0.0, 0.5, np.nan, 1.0], center=3, thresh=0.1)
    assert s == "a« b|0.50»↵[[c|1.00]]"
    assert strip_marks(s) == "a b\nc"


def test_verdict():
    assert verdict(-0.05, 0.05, 0.1) == "equivalent"
    assert verdict(-0.3, -0.2, 0.1) == "combo_worse"
    assert verdict(0.2, 0.3, 0.1) == "combo_better"
    assert verdict(-0.2, 0.05, 0.1) == "inconclusive"


def test_corpus_batches_cover_everything_once():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, paths = smoke_cfg(tmp)
        make_corpus(paths.corpus("train"), 40, 64, 0)
        st = CorpusStore(paths.corpus("train"))
        seen = []
        for b in st.iter_batches(256):
            assert b["input_ids"].shape[0] == len(b["lengths"]) == len(b["seq_ids"])
            for ids, n, s in zip(b["input_ids"], b["lengths"], b["seq_ids"]):
                assert np.array_equal(ids[:n], st.get_tokens(int(s), 0, int(n)))
                seen.append(int(s))
        assert sorted(seen) == list(range(st.n_seqs))
