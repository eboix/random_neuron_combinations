"""GPU job: score arbitrary texts on specific units. Input jsonl rows: {probe_id, unit_id, text, kind}.
Output jsonl rows add max_act / min_act (over tokens at positions >= skip_first), per-token acts and tokens."""
from __future__ import annotations

import numpy as np

from .io_utils import read_jsonl, write_jsonl


def run_probe(cfg, paths, job_file, out_file, model=None, tok=None, log=print):
    import torch
    from .hooks import MLPCapture
    from .model_utils import load_model, pad_id_of
    from .units import load_unit_set

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model, tok = load_model(cfg, device)
    device = next(model.parameters()).device
    rows = read_jsonl(job_file)
    skip = int(cfg.cache.skip_first_tokens)
    max_tok, budget = int(cfg.probe.max_tokens), int(cfg.probe.batch_tokens)
    pad = pad_id_of(tok)

    uset = {int(L): load_unit_set(paths, int(L)) for L in cfg.model.layers}
    col_of = {}
    for L, us in uset.items():
        for m in us["meta"]:
            col_of[m["unit_id"]] = (L, m["col"])

    by_layer = {}
    for i, r in enumerate(rows):
        L, col = col_of[r["unit_id"]]
        by_layer.setdefault(L, []).append((i, col))
    results = [None] * len(rows)

    for L, items in by_layer.items():
        A = torch.as_tensor(uset[L]["A"], device=device, dtype=torch.float32)
        bb = torch.as_tensor(uset[L]["b"], device=device, dtype=torch.float32)
        enc = {i: tok(rows[i]["text"], add_special_tokens=False, truncation=True, max_length=max_tok)["input_ids"]
               for i, _ in items}
        items = sorted(items, key=lambda x: -len(enc[x[0]]))
        pos = 0
        with MLPCapture(model, [L], stop_after_last=bool(cfg.get("runtime", {}).get("early_stop", True))) as cap:
            while pos < len(items):
                T = max(1, len(enc[items[pos][0]]))
                Bn = max(1, budget // T)
                chunk = items[pos:pos + Bn]
                pos += len(chunk)
                ids = torch.full((len(chunk), T), pad, dtype=torch.long)
                msk = torch.zeros((len(chunk), T), dtype=torch.long)
                for r, (i, _) in enumerate(chunk):
                    e = enc[i]
                    ids[r, :len(e)] = torch.tensor(e, dtype=torch.long)
                    msk[r, :len(e)] = 1
                ids, msk = ids.to(device), msk.to(device)
                cap.run(ids, msk)
                p = cap.p[L].float()                                         # (B,T,d_ff)
                cols = torch.as_tensor([c for _, c in chunk], device=device)
                a = A[:, cols]                                               # (d_ff,B)
                acts = torch.einsum("btf,fb->bt", p, a) + bb[cols][:, None]  # (B,T)
                acts = acts.cpu().numpy()
                for r, (i, _) in enumerate(chunk):
                    n = len(enc[i])
                    v = acts[r, :n]
                    vv = v[skip:] if n > skip else np.zeros(0)
                    results[i] = {
                        **rows[i], "n_tokens": n,
                        "max_act": float(vv.max()) if vv.size else None,
                        "min_act": float(vv.min()) if vv.size else None,
                        "acts": [round(float(x), 4) for x in v],
                        "tokens": [tok.decode([t]) for t in enc[i]],
                    }
        log(f"[probe] layer {L}: {len(items)} texts")
    for r in results:
        r.pop("text", None)
    write_jsonl(out_file, results)
    log(f"[probe] wrote {out_file}")
