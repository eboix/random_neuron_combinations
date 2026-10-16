#!/usr/bin/env python
"""Collect one site's per-token activations of a causal LM on held-out documents (disjoint from the
scan set via --doc-offset) into an fp16 memmap for SAE training. Position 0 and pads are skipped;
documents are written in a random order.

    python scripts/lm_collect.py --root <root> --model Qwen/Qwen2.5-7B-Instruct --corpus <parquet> \
        --layer layer18.resid --n-docs 20000 --doc-offset 20000
Output: <root>/lm_sae/<model>/<slug>/train_acts.npy (n_tokens, D) fp16 + stats.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from lm_extract import load_docs, site_hooks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--model', required=True); ap.add_argument('--corpus', required=True)
    ap.add_argument('--layer', required=True, help='e.g. layer18.resid'); ap.add_argument('--n-docs', type=int, default=20000); ap.add_argument('--doc-offset', type=int, default=20000)
    ap.add_argument('--seq-len', type=int, default=256); ap.add_argument('--min-tokens', type=int, default=64); ap.add_argument('--batch', type=int, default=16); ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    root = Path(args.root); mname = args.model.split('/')[-1]; slug = args.layer.replace('.', '_')
    out = root / 'lm_sae' / mname / slug; out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model); dev = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    t0 = time.time()
    docs = load_docs(args.corpus, tok, args.n_docs, args.seq_len, args.min_tokens, args.doc_offset)
    N, T = docs.shape; valid = docs >= 0; valid[:, 0] = False; n_tok = int(valid.sum())
    blk, site = args.layer.split('.'); li = int(blk.replace('layer', ''))
    name, slug2, site, _, mod, kind = site_hooks(model, [li], [site])[0]
    cap = {}
    if kind == 'pre': mod.register_forward_pre_hook(lambda m, inp: cap.__setitem__('h', inp[0]))
    else: mod.register_forward_hook(lambda m, inp, o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))
    with torch.no_grad(): model(input_ids=torch.tensor(docs[:1, :8].clip(min=0), device=dev))
    D = cap['h'].shape[-1]
    acts = np.lib.format.open_memmap(out / 'train_acts.npy', 'w+', np.float16, (n_tok, D))
    order = np.random.default_rng(args.seed).permutation(N); pos = 0; ssum = torch.zeros(D, dtype=torch.float64, device=dev); nsum = 0.0
    print(f'[lm-collect] {args.model} {args.layer}: {N} docs, {n_tok} tokens, width {D}', flush=True)
    with torch.no_grad():
        for b0 in range(0, N, args.batch):
            idx = order[b0:b0 + args.batch]; ids = torch.tensor(docs[idx], device=dev); mask = ids >= 0
            model(input_ids=ids.clamp(min=0), attention_mask=mask.long())
            v = torch.tensor(valid[idx], device=dev); h = cap['h'][v]                    # (n_valid, D)
            ssum += h.float().sum(0).double(); nsum += h.float().norm(dim=-1).sum().item()
            hn = h.half().cpu().numpy(); acts[pos:pos + len(hn)] = hn; pos += len(hn)
            if (b0 // args.batch) % 100 == 0: print(f'[lm-collect] {b0 + len(idx)}/{N} docs', flush=True)
    acts.flush(); assert pos == n_tok, (pos, n_tok)
    json.dump({'model': args.model, 'layer': args.layer, 'n_docs': N, 'positions': n_tok, 'D': D, 'mean': (ssum / n_tok).cpu().tolist(), 'mean_norm': nsum / n_tok,
               'corpus': args.corpus, 'doc_offset': args.doc_offset}, open(out / 'stats.json', 'w'))
    print(f'[lm-collect] done {n_tok} tokens -> {out} in {time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__':
    main()
