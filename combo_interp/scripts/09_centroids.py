#!/usr/bin/env python3
"""CPU/GPU: do the rows of W_gate partition the MLP input better than random centroids? (see combo_interp.geometry)

  python scripts/09_centroids.py --config ... --init trained     # capture x (one model pass) + evaluate
  python scripts/09_centroids.py --config ... --init random      # same architecture, untrained weights
  python scripts/09_centroids.py --config ... --report           # summary.md + plot from both
"""
import json
import time

from _common import setup


def extra(ap):
    ap.add_argument("--init", choices=["trained", "random"], default="trained")
    ap.add_argument("--report", action="store_true", help="only summarize existing results")
    ap.add_argument("--recapture", action="store_true", help="recompute the captured MLP inputs")


cfg, paths, args = setup(__doc__, extra)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from combo_interp.geometry import centroid_eval, doc_label  # noqa: E402
from combo_interp.io_utils import write_json  # noqa: E402

g = cfg.geometry
layers = [int(l) for l in cfg.model.layers]
out_root = paths.results / "geometry"
tag = "centered" if g.center else "raw"


def report():
    from combo_interp.geometry_report import write_report
    frames = [pd.read_csv(f) for f in sorted(out_root.glob(f"*/centroids_{tag}.csv"))]
    if not frames:
        raise SystemExit(f"no results under {out_root}")
    write_report(pd.concat(frames), out_root, tag)
    print((out_root / f"summary_{tag}.md").read_text())


if args.report:
    report()
    raise SystemExit

import torch  # noqa: E402

from combo_interp.corpus import CorpusStore  # noqa: E402
from combo_interp.geometry import capture_inputs, mlp_weights  # noqa: E402
from combo_interp.model_utils import load_model  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{time.strftime('%H:%M:%S')} [centroids] init={args.init} center={g.center} device={device}", flush=True)
model, tok = load_model(cfg, device)
if args.init == "random":
    from transformers import AutoModelForCausalLM
    config = model.config
    del model                       # free the trained weights first (two 8B models do not fit a 45 GB GPU slice)
    torch.cuda.empty_cache() if device == "cuda" else None
    torch.manual_seed(int(g.random_init_seed))
    model = AutoModelForCausalLM.from_config(config, dtype=getattr(torch, cfg.model.dtype)).to(device).eval()

store = CorpusStore(paths.corpus(g.split))
gids = np.load(paths.cache(g.split, layers[0]) / "uniform.npz")["gid"].astype(np.int64)
cap_dir = paths.data / "geometry" / args.init
cap_dir.mkdir(parents=True, exist_ok=True)
if args.recapture or not all((cap_dir / f"X_L{L}.npy").exists() for L in layers):
    X = capture_inputs(model, store, gids, layers, int(cfg.cache.max_batch_tokens))
    for L in layers:
        np.save(cap_dir / f"X_L{L}.npy", X[L])
    np.save(cap_dir / "gid.npy", gids)

# labels: current token, next token, coarse document label
seq, pos = gids // store.seq_len, gids % store.seq_len
tok_id = np.array([store.get_tokens(s, p, p + 1)[0] for s, p in zip(seq, pos)])
nxt = [store.get_tokens(s, p + 1, p + 2) for s, p in zip(seq, pos)]
next_id = np.array([t[0] if len(t) else -1 for t in nxt])
doc_names = {int(s): doc_label(tok.decode(store.get_tokens(int(s), 0, 2000))) for s in np.unique(seq)}
names = sorted(set(doc_names.values()))
doc = np.array([names.index(doc_names[int(s)]) for s in seq])
labels = {"token": tok_id, "next": next_id, "doc": doc}
fit = seq % 2 == 0

out = out_root / args.init
out.mkdir(parents=True, exist_ok=True)
write_json(out / "doc_labels.json", {"counts_tokens": {n: int((doc == i).sum()) for i, n in enumerate(names)}})
rows = []
for L in layers:
    part = out / f"centroids_{tag}_L{L}.csv"       # per-layer results: a rerun after a timeout skips finished layers
    if part.exists() and not args.recapture:
        print(f"L{L} done earlier ({part.name}), skipping", flush=True)
        rows += pd.read_csv(part).to_dict("records")
        continue
    layer_rows = []
    X = np.load(cap_dir / f"X_L{L}.npy")
    Wg, Wu = mlp_weights(model, L)
    Ms = sorted({min(int(m), Wg.shape[0]) for m in g.Ms})
    reps = {min(int(k), Wg.shape[0]): int(v) for k, v in dict(g.reps).items()}
    for r in centroid_eval(X, Wg, Wu, labels, fit, Ms, reps, combo_ks=list(g.combo_ks), center=bool(g.center),
                           seed=int(cfg.seed) + L, log=lambda m, L=L: print(f"{time.strftime('%H:%M:%S')} L{L} {m}", flush=True),
                           assign_rules=tuple(g.assign), device=device):
        layer_rows.append({"init": args.init, "layer": L, **r})
    pd.DataFrame(layer_rows).to_csv(part, index=False)
    rows += layer_rows
pd.DataFrame(rows).to_csv(out / f"centroids_{tag}.csv", index=False)
print(json.dumps({"rows": len(rows), "out": str(out)}))
report()
