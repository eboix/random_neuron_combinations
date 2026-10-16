#!/usr/bin/env python
"""Text analogue of extract_acts.py: scan documents through a causal LM and, for each chosen decoder
layer and hook site, record per document the maximum over tokens of every unit (and its argmax token),
plus the same for K random unit-norm directions ("virtual neurons"), unit mean/std, and the token ids.

Sites per decoder layer (Llama/Qwen-style blocks):
  mlp       SwiGLU product act(gate)*up, the input of down_proj (the MLP "neurons", width I)
  attn_out  output of o_proj (the attention block's write, width D)
  mlp_out   output of down_proj (the MLP block's write, width D)
  resid     residual stream after the layer (width D)
Position 0 is excluded from max/min (attention-sink token).

    python scripts/lm_extract.py --root <root> --model Qwen/Qwen2.5-7B-Instruct --corpus <parquet> \
        --layers 9,18,26 --sites mlp,attn_out,mlp_out,resid --n-docs 20000 --seq-len 256
Outputs <root>/lm_acts/<model>/: docs_tokens.npy (N, T) int32 (-1 = pad), {slug}_real_{max,min,argmax,argmin}.npy,
{slug}_virt_*.npy, {slug}_virtual_dirs.npy, {slug}_unit_stats.npz, meta.json
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
import pyarrow.parquet as pq

def load_docs(corpus, tok, n_docs, seq_len, min_tokens, offset):
    """First n_docs documents (after `offset` accepted ones) with at least min_tokens tokens, truncated to seq_len."""
    pf = pq.ParquetFile(corpus); ids = []; skipped = 0
    for rg in range(pf.num_row_groups):
        texts = pf.read_row_group(rg, columns=['text']).column('text').to_pylist()
        enc = tok(texts, add_special_tokens=True, truncation=True, max_length=seq_len)['input_ids']
        for e in enc:
            if len(e) < min_tokens: continue
            if skipped < offset: skipped += 1; continue
            ids.append(e)
            if len(ids) >= n_docs: break
        if len(ids) >= n_docs: break
    arr = np.full((len(ids), seq_len), -1, dtype=np.int32)
    for i, e in enumerate(ids): arr[i, :len(e)] = e
    return arr

def site_hooks(model, layer_idx, sites):
    """Return list of (name, slug, site, module, kind) where kind is 'pre' (capture input) or 'out'."""
    blocks = model.model.layers
    out = []
    for i in layer_idx:
        blk = blocks[i]
        for s in sites:
            if s == 'mlp': out.append((f'layer{i}.mlp', f'layer{i}_mlp', s, i, blk.mlp.down_proj, 'pre'))
            elif s == 'attn_out': out.append((f'layer{i}.attn_out', f'layer{i}_attn_out', s, i, blk.self_attn.o_proj, 'out'))
            elif s == 'mlp_out': out.append((f'layer{i}.mlp_out', f'layer{i}_mlp_out', s, i, blk.mlp.down_proj, 'out'))
            elif s == 'resid': out.append((f'layer{i}.resid', f'layer{i}_resid', s, i, blk, 'out'))
            else: raise ValueError(s)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--model', required=True); ap.add_argument('--corpus', required=True)
    ap.add_argument('--layers', required=True); ap.add_argument('--sites', default='mlp,attn_out,mlp_out,resid')
    ap.add_argument('--n-docs', type=int, default=20000); ap.add_argument('--doc-offset', type=int, default=0)
    ap.add_argument('--seq-len', type=int, default=256); ap.add_argument('--min-tokens', type=int, default=64)
    ap.add_argument('--batch', type=int, default=16); ap.add_argument('--k-virtual', type=int, default=128); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-name', default=''); ap.add_argument('--virt-std-from', default='')
    args = ap.parse_args()
    root = Path(args.root); mname = args.model.split('/')[-1]
    out = root / 'lm_acts' / (args.out_name or mname); out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    dev = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    t0 = time.time()
    tokens_path = out / 'docs_tokens.npy'
    if tokens_path.exists() and np.load(tokens_path, mmap_mode='r').shape == (args.n_docs, args.seq_len):
        docs = np.load(tokens_path)
    else:
        docs = load_docs(args.corpus, tok, args.n_docs, args.seq_len, args.min_tokens, args.doc_offset); np.save(tokens_path, docs)
    N, T = docs.shape
    print(f'[lm-extract] {args.model}: {N} docs x {T} tokens ({time.time()-t0:.0f}s to tokenize)', flush=True)
    layer_idx = [int(x) for x in args.layers.split(',')]
    hooks = site_hooks(model, layer_idx, args.sites.split(','))
    captured = {}
    for j, (name, slug, site, blk, mod, kind) in enumerate(hooks):
        if kind == 'pre': mod.register_forward_pre_hook((lambda j: lambda m, inp: captured.__setitem__(j, inp[0]))(j))
        else: mod.register_forward_hook((lambda j: lambda m, inp, o: captured.__setitem__(j, o[0] if isinstance(o, tuple) else o))(j))
    # probe widths
    with torch.no_grad():
        model(input_ids=torch.tensor(docs[:1, :8].clip(min=0), device=dev))
    g = torch.Generator().manual_seed(args.seed); store = {}; layers_meta = []
    for j, (name, slug, site, blk, mod, kind) in enumerate(hooks):
        D = captured[j].shape[-1]
        Wd = torch.randn(D, args.k_virtual, generator=g)
        if args.virt_std_from:
            sd = np.load(Path(args.virt_std_from) / f'{slug}_unit_stats.npz')['std']; Wd = Wd / torch.tensor(np.maximum(sd, 1e-3), dtype=torch.float32)[:, None]
        Wd = Wd / Wd.norm(dim=0, keepdim=True); np.save(out / f'{slug}_virtual_dirs.npy', Wd.numpy())
        s = {'dirs': Wd.to(dev, torch.bfloat16)}
        s['real_max'] = np.lib.format.open_memmap(out / f'{slug}_real_max.npy', 'w+', np.float32, (N, D))
        s['real_min'] = np.lib.format.open_memmap(out / f'{slug}_real_min.npy', 'w+', np.float16, (N, D))
        s['real_argmax'] = np.lib.format.open_memmap(out / f'{slug}_real_argmax.npy', 'w+', np.int16, (N, D))
        s['real_argmin'] = np.lib.format.open_memmap(out / f'{slug}_real_argmin.npy', 'w+', np.int16, (N, D))
        for key in ['max', 'min', 'argmax', 'argmin']:
            s[f'virt_{key}'] = np.lib.format.open_memmap(out / f'{slug}_virt_{key}.npy', 'w+', np.int16 if key.startswith('arg') else np.float32, (N, args.k_virtual))
        s['sum'] = torch.zeros(D, dtype=torch.float64, device=dev); s['sumsq'] = torch.zeros(D, dtype=torch.float64, device=dev); s['count'] = 0
        store[j] = s; layers_meta.append({'name': name, 'slug': slug, 'site': site, 'block': blk, 'D': D})
        print(f'  {name}: width {D}', flush=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    with torch.no_grad():
        for b0 in range(0, N, args.batch):
            ids = torch.tensor(docs[b0:b0 + args.batch], device=dev); mask = ids >= 0
            model(input_ids=ids.clamp(min=0), attention_mask=mask.long())
            valid = mask.clone(); valid[:, 0] = False                       # exclude position 0
            vf = valid.float()
            for j in store:
                h = captured[j].float(); s = store[j]                     # (B, T, D)
                big = torch.where(valid[..., None], h, torch.full_like(h, -1e9)); mx = big.max(1)
                small = torch.where(valid[..., None], h, torch.full_like(h, 1e9)); mn = small.min(1)
                s['real_max'][b0:b0 + len(ids)] = mx.values.cpu().numpy(); s['real_argmax'][b0:b0 + len(ids)] = mx.indices.cpu().numpy().astype(np.int16)
                s['real_min'][b0:b0 + len(ids)] = mn.values.cpu().numpy().astype(np.float16); s['real_argmin'][b0:b0 + len(ids)] = mn.indices.cpu().numpy().astype(np.int16)
                s['sum'] += (h * vf[..., None]).sum((0, 1)).double(); s['sumsq'] += (h * h * vf[..., None]).sum((0, 1)).double(); s['count'] += int(vf.sum().item())
                p = (captured[j] @ s['dirs']).float()                     # (B, T, K)
                pb = torch.where(valid[..., None], p, torch.full_like(p, -1e9)).max(1); ps = torch.where(valid[..., None], p, torch.full_like(p, 1e9)).min(1)
                s['virt_max'][b0:b0 + len(ids)] = pb.values.cpu().numpy(); s['virt_argmax'][b0:b0 + len(ids)] = pb.indices.cpu().numpy().astype(np.int16)
                s['virt_min'][b0:b0 + len(ids)] = ps.values.cpu().numpy(); s['virt_argmin'][b0:b0 + len(ids)] = ps.indices.cpu().numpy().astype(np.int16)
            if (b0 // args.batch) % 100 == 0:
                print(f'[lm-extract] {b0 + len(ids)}/{N} docs {(b0 + len(ids))/(time.time()-t0):.0f} docs/s', flush=True)
    for j, lm in enumerate(layers_meta):
        s = store[j]; mean = (s['sum'] / s['count']).cpu().numpy(); var = (s['sumsq'] / s['count']).cpu().numpy() - mean ** 2
        np.savez(out / f'{lm["slug"]}_unit_stats.npz', mean=mean, std=np.sqrt(np.maximum(var, 0)), count=s['count'])
        for k, v in s.items():
            if isinstance(v, np.memmap): v.flush()
    old = json.load(open(out / 'meta.json')) if (out / 'meta.json').exists() else {}
    names = {l['name'] for l in layers_meta}
    meta = {'model': args.model, 'modality': 'text', 'corpus': str(args.corpus), 'n_docs': N, 'seq_len': T, 'doc_offset': args.doc_offset, 'k_virtual': args.k_virtual, 'seed': args.seed,
            'spatialStatistic': 'maximum activation over tokens (position 0 excluded) per document',
            'virtual': ('K Gaussian unit-norm directions in the raw activation space' if not args.virt_std_from else 'K unit-norm directions with Gaussian coefficients scaled by 1/std of each unit'),
            'layers': [l for l in old.get('layers', []) if l['name'] not in names] + layers_meta}
    json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
    print(f'[lm-extract] done in {time.time()-t0:.0f}s -> {out}', flush=True)

if __name__ == '__main__':
    main()
