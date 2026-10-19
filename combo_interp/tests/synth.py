"""Synthetic stand-ins so the whole pipeline can be exercised without a GPU / real model / network.

FakeTokenizer      byte-level (id = byte + 3), pad=0, eos=1
make_corpus        random token streams with local structure
SynthModel         fake 'MLP hidden activation' P(ids) = token-specific + previous-token-specific sparse features
emulate_*          numpy re-implementations of stats / cache / probe writing the SAME file schema as the torch code
"""
from __future__ import annotations

import numpy as np

from combo_interp.corpus import CorpusStore, write_corpus
from combo_interp.io_utils import atomic_savez, write_json, read_jsonl, write_jsonl
from combo_interp.stats import cov_from_sums
from combo_interp.units import (load_unit_set, make_blind_key, make_unit_set, save_unit_set, zca_matrix)

VOCAB = 259


class FakeTokenizer:
    pad_token_id, eos_token_id = 0, 1

    def __call__(self, texts, add_special_tokens=False, truncation=True, max_length=None):
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        ids = []
        for t in texts:
            x = [b + 3 for b in t.encode("utf-8", "replace")]
            ids.append(x[:max_length] if max_length else x)
        return {"input_ids": ids[0] if single else ids}

    def decode(self, ids):
        bs = bytes([max(0, min(255, int(i) - 3)) for i in ids if int(i) >= 3])
        return bs.decode("utf-8", "replace")


def make_corpus(root, n_seqs, seq_len, seed, pad_id=0):
    rng = np.random.default_rng(seed)
    letters = np.arange(ord("a"), ord("z") + 1) + 3
    docs = []
    for _ in range(n_seqs):
        n = int(rng.integers(seq_len // 2, seq_len + 1))
        ids = rng.choice(np.concatenate([letters, np.array([ord(" ") + 3] * 6), np.array([ord(".") + 3, ord(",") + 3])]), size=n)
        docs.append(ids.astype(np.int32))
    return write_corpus(root, docs, seq_len, pad_id, shard_seqs=max(8, n_seqs // 3))


class SynthModel:
    """P(ids) -> (T, d_ff). Each neuron prefers 1-2 tokens (current) and sometimes a previous token."""

    def __init__(self, d_ff, seed=0):
        rng = np.random.default_rng(seed)
        self.d = d_ff
        self.Wt = np.zeros((VOCAB, d_ff))
        self.Wp = np.zeros((VOCAB, d_ff))
        letters = np.arange(ord("a"), ord("z") + 1) + 3
        for j in range(d_ff):
            for t in rng.choice(letters, size=int(rng.integers(1, 3)), replace=False):
                self.Wt[t, j] = rng.uniform(2, 6)
            if j % 3 == 0:
                self.Wp[rng.choice(letters), j] = rng.uniform(1, 3)
        self.scale = rng.uniform(0.5, 2.0, size=d_ff)
        self.seed = seed

    def P(self, ids, layer=0):
        ids = np.asarray(ids)
        prev = np.concatenate([[0], ids[:-1]])
        base = (self.Wt[ids] + self.Wp[prev]) * self.scale * (1 + 0.15 * layer)
        # deterministic pseudo-noise from token ids (so P is a pure function of the text)
        noise = np.sin(np.outer(ids * 0.37 + prev * 0.11, np.arange(1, self.d + 1)) * 12.9898) * 0.25
        return np.maximum(base + noise, -0.2)


def _valid_tokens(store, seq_id, skip):
    n = store.seq_length(seq_id)
    return store.get_tokens(seq_id, 0, n), np.arange(skip, n)


def emulate_stats_and_units(cfg, paths, model: SynthModel):
    store = CorpusStore(paths.corpus("train"))
    skip = int(cfg.stats.skip_first_tokens)
    ids_all = []
    for L in cfg.model.layers:
        s, xtx, n = 0, 0, 0
        for seq in range(store.n_seqs):
            ids, pos = _valid_tokens(store, seq, skip)
            P = model.P(ids, L)[pos]
            s = s + P.sum(0); xtx = xtx + P.T @ P; n += len(P)
        mean, cov = cov_from_sums(n, s, xtx)
        paths.stats(L).mkdir(parents=True, exist_ok=True)
        np.save(paths.stats(L) / "mean.npy", mean); np.save(paths.stats(L) / "cov.npy", cov.astype(np.float32))
        zca = zca_matrix(cov, float(cfg.units.whiten_ridge), device="numpy")
        us = make_unit_set(int(L), mean, cov, cfg.units, cfg.seed, zca)
        save_unit_set(us, paths.units(L))
        ids_all += [m["unit_id"] for m in us["meta"]]
    write_json(paths.blind_key, make_blind_key(ids_all, cfg.seed))


def emulate_cache(cfg, paths, model: SynthModel, split: str):
    cc = cfg.cache
    sp = cc.holdout if split == "holdout" else cc
    k_big, k_prof = int(sp.get("k_big", cc.k_big)), int(sp.get("k_prof", cc.k_prof))
    n_unif = int(sp.get("uniform_samples", cc.uniform_samples))
    before, after, skip = int(cc.window_before), int(cc.window_after), int(cc.skip_first_tokens)
    store = CorpusStore(paths.corpus(split))
    Lq = store.seq_len
    rng = np.random.default_rng(1 if split == "train" else 2)
    offsets = np.arange(-before, after + 1)
    for L in cfg.model.layers:
        us = load_unit_set(paths, int(L))
        A, b = us["A"].astype(np.float64), us["b"].astype(np.float64)
        M = A.shape[1]
        acts_seq, gids, rows = {}, [], []
        for seq in range(store.n_seqs):
            ids, pos = _valid_tokens(store, seq, skip)
            Z = model.P(ids, L) @ A + b                         # (len, M) incl. position 0 (masked below)
            acts_seq[seq] = Z
            for p in pos:
                gids.append(seq * Lq + p)
            rows.append(Z[skip:])
        Zall = np.concatenate(rows); gids = np.array(gids, dtype=np.int64)
        N = len(Zall)
        mu, sd = us["mean_u"], np.maximum(us["std_u"], 1e-12)
        z = (Zall - mu) / sd
        mom = {"n": np.array(N), "s1": Zall.sum(0), "s2": (Zall ** 2).sum(0), "s3": (Zall ** 3).sum(0), "s4": (Zall ** 4).sum(0),
               "gt3": (z > 3).sum(0), "lt3": (z < -3).sum(0), "gt5": (z > 5).sum(0), "lt5": (z < -5).sum(0)}
        def topk(score):                                           # score (N,M) -> idx (M,K) sorted desc
            K = min(k_big, N)
            idx = np.argsort(-score, axis=0, kind="stable")[:K].T
            vals = np.take_along_axis(score.T, idx, 1)
            return vals, idx
        pv, pi = topk(Zall); nv, ni = topk(-Zall)
        pad = lambda a, fill: np.concatenate([a, np.full((M, k_big - a.shape[1]), fill, a.dtype)], 1) if a.shape[1] < k_big else a
        d = paths.cache(split, int(L)); d.mkdir(parents=True, exist_ok=True)
        atomic_savez(d / "topk_big.npz", pos_vals=pad(pv, -np.inf), pos_gid=pad(gids[pi], -1),
                     neg_vals=pad(-nv, np.inf), neg_gid=pad(gids[ni], -1))
        if k_prof > 0:
            kw = {}
            for pole, (v, ix, sgn) in (("pos", (pv, pi, 1)), ("neg", (nv, ni, -1))):
                K = min(k_prof, v.shape[1])
                win = np.full((M, k_prof, len(offsets)), np.nan, np.float32)
                g = np.full((M, k_prof), -1, np.int64); vv = np.full((M, k_prof), -np.inf if pole == "pos" else np.inf)
                for m in range(M):
                    for r in range(K):
                        gid = gids[ix[m, r]]; seq, p = divmod(int(gid), Lq)
                        Z = acts_seq[seq]
                        for wi, o in enumerate(offsets):
                            q = p + o
                            if skip <= q < len(Z):
                                win[m, r, wi] = Z[q, m]
                        g[m, r] = gid; vv[m, r] = sgn * v[m, r]
                kw[f"{pole}_vals"], kw[f"{pole}_gid"], kw[f"{pole}_win"] = vv, g, win
                kw[f"{pole}_gate"] = np.full((M, k_prof), np.nan); kw[f"{pole}_up"] = np.full((M, k_prof), np.nan)
            atomic_savez(d / "topk_prof.npz", offsets=offsets, **kw)
        sel = rng.choice(N, size=min(n_unif, N), replace=False)
        atomic_savez(d / "uniform.npz", acts=Zall[sel].astype(np.float16), gid=gids[sel])
        atomic_savez(d / "moments.npz", **mom)
        write_json(d / "meta.json", {"split": split, "n_tokens": int(N), "seq_len": Lq, "layer": int(L)})


def emulate_probe(cfg, paths, job_file, out_file, model: SynthModel, tok):
    skip = int(cfg.cache.skip_first_tokens)
    rows = read_jsonl(job_file)
    us = {int(L): load_unit_set(paths, int(L)) for L in cfg.model.layers}
    col = {m["unit_id"]: (L, m["col"]) for L, u in us.items() for m in u["meta"]}
    out = []
    for r in rows:
        L, c = col[r["unit_id"]]
        ids = tok(r["text"], add_special_tokens=False, truncation=True, max_length=int(cfg.probe.max_tokens))["input_ids"]
        if len(ids) == 0:
            v = np.zeros(0)
        else:
            v = model.P(np.array(ids), L) @ us[L]["A"][:, c].astype(np.float64) + us[L]["b"][c]
        vv = v[skip:]
        out.append({**{k: x for k, x in r.items() if k != "text"}, "n_tokens": len(ids),
                    "max_act": float(vv.max()) if vv.size else None, "min_act": float(vv.min()) if vv.size else None,
                    "acts": [round(float(x), 4) for x in v], "tokens": [tok.decode([t]) for t in ids]})
    write_jsonl(out_file, out)


def emulate_calibration(cfg, paths, model: SynthModel):
    """numpy version of calibrate.run_calibration (same window sampling, same output files)."""
    from combo_interp.calibrate import sample_windows, save_calibration
    C = cfg.calibration
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), 31337]))
    ids = sample_windows(CorpusStore(paths.corpus(C.split)), int(C.n_windows), int(C.window), rng)
    for L in [int(l) for l in cfg.model.layers]:
        us = load_unit_set(paths, L)
        acts = np.stack([model.P(w, L) @ us["A"].astype(np.float64) + us["b"] for w in ids])
        save_calibration(cfg, paths, L, acts)
