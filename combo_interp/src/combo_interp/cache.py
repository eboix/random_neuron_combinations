"""Pass 2: stream the corpus once, project the MLP hidden activation onto every unit of every layer, and keep

  moments.npz   streaming moments (mean/var/kurtosis/sparsity) per unit
  topk_big.npz  K_big strongest firings per pole per unit: (value, gid)
  topk_prof.npz K_prof strongest firings per pole per unit WITH a per-token activation window around them
                (+ SwiGLU gate/value at the firing token for neuron units)
  uniform.npz   activations of all units on a uniform random sample of tokens (bulk of the distribution)

Layout of unit columns matches data/units/L*/units.npz. Resumable (checkpoint every cfg.cache.ckpt_every batches).
"""
from __future__ import annotations

import os

import numpy as np

from .config import Paths
from .io_utils import atomic_savez, write_json


def gather_windows(acts_full, b_idx, t_idx, rows, offsets, lengths, skip_first):
    """acts_full (B,T,M); rows (M,k) indices into the valid-token list -> (M,k,W) activations of unit m
    in a window around each selected token (NaN outside the real sequence / before skip_first)."""
    import torch
    M, k = rows.shape
    b = b_idx[rows]                                        # (M,k)
    t = t_idx[rows]
    tt = t.unsqueeze(-1) + offsets.view(1, 1, -1)          # (M,k,W)
    ok = (tt >= skip_first) & (tt < lengths[b].unsqueeze(-1))
    tt_c = tt.clamp(0, acts_full.shape[1] - 1)
    col = torch.arange(M, device=acts_full.device).view(M, 1, 1)
    out = acts_full[b.unsqueeze(-1), tt_c, col]
    return torch.where(ok, out, torch.full_like(out, float("nan")))


def run_cache(cfg, paths: Paths, split: str, model=None, tok=None, log=print, _stop_after=None):
    import torch
    from .corpus import CorpusStore
    from .hooks import MLPCapture
    from .model_utils import load_model
    from .topk import TopKTracker
    from .units import load_unit_set

    cc = cfg.cache
    sp = cc.holdout if split == "holdout" else cc
    n_tokens = int(sp["tokens"] if split == "holdout" else cc.train_tokens)
    k_big, k_prof = int(sp.get("k_big", cc.k_big)), int(sp.get("k_prof", cc.k_prof))
    n_unif = int(sp.get("uniform_samples", cc.uniform_samples))
    before, after = int(cc.window_before), int(cc.window_after)
    skip = int(cc.skip_first_tokens)
    record_gu = bool(cc.record_gate_up) and k_prof > 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model, tok = load_model(cfg, device)
    device = next(model.parameters()).device
    store = CorpusStore(paths.corpus(split))
    L_seq = store.seq_len
    layers = [int(l) for l in cfg.model.layers]
    out_root = paths.data / "cache" / split
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- per-layer state
    S = {}
    offsets = torch.arange(-before, after + 1, device=device)
    W = len(offsets)
    for L in layers:
        us = load_unit_set(paths, L)
        M = us["A"].shape[1]
        meta = us["meta"]
        ncols = [m["col"] for m in meta if m["kind"] == "neuron"]
        nids = [m["neurons"][0] for m in meta if m["kind"] == "neuron"]
        st = {
            "M": M,
            "A": torch.as_tensor(us["A"], device=device, dtype=torch.float32),
            "b": torch.as_tensor(us["b"], device=device, dtype=torch.float32),
            "mu": torch.as_tensor(us["mean_u"], device=device, dtype=torch.float32),
            "sd": torch.as_tensor(np.maximum(us["std_u"], 1e-12), device=device, dtype=torch.float32),
            "ncols": torch.as_tensor(ncols, device=device, dtype=torch.long),
            "nids": torch.as_tensor(nids, device=device, dtype=torch.long),
            "big_pos": TopKTracker(M, k_big, device), "big_neg": TopKTracker(M, k_big, device),
            "mom": {k: torch.zeros(M, dtype=torch.float64, device=device) for k in ("s1", "s2", "s3", "s4")},
            "cnt": {k: torch.zeros(M, dtype=torch.long, device=device) for k in ("gt3", "lt3", "gt5", "lt5")},
            "uni": [],
        }
        if k_prof > 0:
            allc = torch.arange(M, device=device)
            shapes = {"win": (W,), "gate": (), "up": ()}
            st["prof_pos"] = TopKTracker(M, k_prof, device, allc, shapes)
            st["prof_neg"] = TopKTracker(M, k_prof, device, allc, shapes)
        S[L] = st

    uni_gid, n_seen, start_batch = [], 0, 0
    ckpt_path = out_root / "ckpt.pt"
    fingerprint = {"split": split, "layers": layers, "M": [S[L]["M"] for L in layers], "k_big": k_big, "k_prof": k_prof,
                   "n_tokens": n_tokens, "n_unif": n_unif, "seed": int(cfg.seed), "window": [before, after], "skip": skip,
                   "model": str(cfg.model.name)}
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck.get("fingerprint") != fingerprint:
            raise RuntimeError(f"{ckpt_path} was written with a different config/unit set; delete it to start over "
                               f"(or restore the old config to resume).\n old: {ck.get('fingerprint')}\n new: {fingerprint}")
        start_batch, n_seen, uni_gid = ck["bi"] + 1, ck["n_seen"], ck["uni_gid"]
        for L in layers:
            c = ck["S"][L]
            S[L]["big_pos"].load(c["big_pos"]); S[L]["big_neg"].load(c["big_neg"])
            if k_prof > 0:
                S[L]["prof_pos"].load(c["prof_pos"]); S[L]["prof_neg"].load(c["prof_neg"])
            for k in S[L]["mom"]:
                S[L]["mom"][k] = c["mom"][k].to(device)
            for k in S[L]["cnt"]:
                S[L]["cnt"][k] = c["cnt"][k].to(device)
            S[L]["uni"] = c["uni"]
        log(f"[cache:{split}] resumed from batch {start_batch} ({n_seen} tokens)")

    ufrac = min(1.0, n_unif / max(1, n_tokens))

    def save_ckpt(bi):
        ck = {"bi": bi, "n_seen": n_seen, "uni_gid": uni_gid, "S": {}, "fingerprint": fingerprint}
        for L in layers:
            s = S[L]
            ck["S"][L] = {"big_pos": s["big_pos"].state(), "big_neg": s["big_neg"].state(),
                          "mom": {k: v.cpu() for k, v in s["mom"].items()},
                          "cnt": {k: v.cpu() for k, v in s["cnt"].items()}, "uni": s["uni"]}
            if k_prof > 0:
                ck["S"][L]["prof_pos"] = s["prof_pos"].state(); ck["S"][L]["prof_neg"] = s["prof_neg"].state()
        torch.save(ck, str(ckpt_path) + ".tmp")
        os.replace(str(ckpt_path) + ".tmp", ckpt_path)

    # ---- main loop
    with MLPCapture(model, layers, capture_gate_up=record_gu, stop_after_last=bool(cfg.get("runtime", {}).get("early_stop", True))) as cap:
        for bi, batch in enumerate(store.iter_batches(int(cc.max_batch_tokens), n_tokens)):
            if bi < start_batch:
                continue
            ids = torch.from_numpy(batch["input_ids"]).to(device)
            lengths = torch.from_numpy(batch["lengths"]).to(device)
            seq_ids = torch.from_numpy(batch["seq_ids"]).to(device)
            B, T = ids.shape
            ar = torch.arange(T, device=device)
            attn = ar[None, :] < lengths[:, None]
            valid = attn & (ar[None, :] >= skip)
            cap.run(ids, attn.long())
            b_idx, t_idx = valid.nonzero(as_tuple=True)
            N = int(b_idx.numel())
            if N == 0:
                continue
            gids = seq_ids[b_idx] * L_seq + t_idx
            gen = torch.Generator(device=device).manual_seed(int(cfg.seed) * 1_000_003 + bi)
            ur = (torch.rand(N, generator=gen, device=device) < ufrac).nonzero().squeeze(1)
            uni_gid.append(gids[ur].cpu())
            n_seen += N
            for L in layers:
                s = S[L]
                acts_full = torch.matmul(cap.p[L].float(), s["A"]) + s["b"]      # (B,T,M)
                acts = acts_full[b_idx, t_idx]                                    # (N,M)
                a64 = acts.double()
                a2 = a64 * a64
                s["mom"]["s1"] += a64.sum(0); s["mom"]["s2"] += a2.sum(0)
                s["mom"]["s3"] += (a2 * a64).sum(0); s["mom"]["s4"] += (a2 * a2).sum(0)
                z = (acts - s["mu"]) / s["sd"]
                s["cnt"]["gt3"] += (z > 3).sum(0); s["cnt"]["lt3"] += (z < -3).sum(0)
                s["cnt"]["gt5"] += (z > 5).sum(0); s["cnt"]["lt5"] += (z < -5).sum(0)
                s["uni"].append(acts[ur].clamp(-6e4, 6e4).half().cpu())
                sT = acts.T.contiguous()                                          # (M,N)
                s["big_pos"].update(sT, gids)
                s["big_neg"].update(-sT, gids)
                if k_prof > 0:
                    def payload(rows, L=L, s=s, acts_full=acts_full):
                        Mm, k = rows.shape
                        out = {"win": gather_windows(acts_full, b_idx, t_idx, rows, offsets, lengths, skip)}
                        gate = torch.full((Mm, k), float("nan"), device=device)
                        up = torch.full((Mm, k), float("nan"), device=device)
                        if record_gu and s["ncols"].numel() > 0:
                            rn = rows[s["ncols"]]                                 # (Mn,k)
                            gate[s["ncols"]] = cap.g[L][b_idx[rn], t_idx[rn], s["nids"].view(-1, 1)].float()
                            up[s["ncols"]] = cap.u[L][b_idx[rn], t_idx[rn], s["nids"].view(-1, 1)].float()
                        out["gate"], out["up"] = gate, up
                        return out
                    s["prof_pos"].update(sT, gids, payload)
                    s["prof_neg"].update(-sT, gids, payload)
                del acts_full, acts, a64, a2, z, sT
            if (bi + 1) % int(cc.ckpt_every) == 0:
                save_ckpt(bi)
                log(f"[cache:{split}] batch {bi + 1}, {n_seen} tokens (checkpoint)")
            if _stop_after is not None and bi + 1 >= _stop_after:      # test hook: simulate a job timeout
                save_ckpt(bi)
                return

    # ---- write outputs
    ug = torch.cat(uni_gid).numpy() if uni_gid else np.zeros(0, dtype=np.int64)
    for L in layers:
        s = S[L]
        d = paths.cache(split, L)
        d.mkdir(parents=True, exist_ok=True)
        mom = {k: v.cpu().numpy() for k, v in s["mom"].items()}
        cnt = {k: v.cpu().numpy() for k, v in s["cnt"].items()}
        atomic_savez(d / "moments.npz", n=np.array(n_seen), **mom, **cnt)
        atomic_savez(d / "topk_big.npz",
                     pos_vals=s["big_pos"].vals.cpu().numpy(), pos_gid=s["big_pos"].gid.cpu().numpy(),
                     neg_vals=-s["big_neg"].vals.cpu().numpy(), neg_gid=s["big_neg"].gid.cpu().numpy())
        if k_prof > 0:
            kw = {}
            for pole, tr in (("pos", s["prof_pos"]), ("neg", s["prof_neg"])):
                kw[f"{pole}_vals"] = (tr.vals if pole == "pos" else -tr.vals).cpu().numpy()
                kw[f"{pole}_gid"] = tr.gid.cpu().numpy()
                kw[f"{pole}_win"] = tr.pay["win"].cpu().numpy().astype(np.float32)
                kw[f"{pole}_gate"] = tr.pay["gate"].cpu().numpy()
                kw[f"{pole}_up"] = tr.pay["up"].cpu().numpy()
            atomic_savez(d / "topk_prof.npz", offsets=offsets.cpu().numpy(), **kw)
        uni_acts = torch.cat(s["uni"]).numpy() if s["uni"] else np.zeros((0, s["M"]), dtype=np.float16)
        atomic_savez(d / "uniform.npz", acts=uni_acts, gid=ug)
        write_json(d / "meta.json", {"split": split, "n_tokens": int(n_seen), "seq_len": int(L_seq),
                                     "k_big": k_big, "k_prof": k_prof, "layer": L,
                                     "skip_first": skip, "window_before": before, "window_after": after})
    if ckpt_path.exists():
        ckpt_path.unlink()
    log(f"[cache:{split}] done: {n_seen} tokens")
