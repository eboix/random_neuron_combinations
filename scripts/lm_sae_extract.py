#!/usr/bin/env python
"""Score every SAE feature on the scan documents (docs_tokens.npy of lm_extract.py): per document the
maximum feature activation over tokens (position 0 excluded) and its argmax, the number of documents each
feature fires on, and the SAE's FVU on the scan set.

    python scripts/lm_sae_extract.py --root <root> --model Qwen/Qwen2.5-7B-Instruct --layer layer18.resid --sae <ckpt>
Writes into <root>/lm_acts/<model>/: {slug}_sae_max.npy (N,F) f32, {slug}_sae_argmax.npy (N,F) i16, {slug}_sae_active.npy, {slug}_sae.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from sae import TopKSAE
from lm_extract import site_hooks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--model', required=True); ap.add_argument('--layer', required=True); ap.add_argument('--sae', required=True)
    ap.add_argument('--batch', type=int, default=16); ap.add_argument('--acts-name', default='')
    args = ap.parse_args()
    root = Path(args.root); mname = args.model.split('/')[-1]; slug = args.layer.replace('.', '_')
    acts = root / 'lm_acts' / (args.acts_name or mname)
    docs = np.load(acts / 'docs_tokens.npy'); N, T = docs.shape; valid_all = docs >= 0; valid_all[:, 0] = False
    from transformers import AutoModelForCausalLM
    dev = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    blk, site = args.layer.split('.'); li = int(blk.replace('layer', ''))
    name, _, site, _, mod, kind = site_hooks(model, [li], [site])[0]
    cap = {}
    if kind == 'pre': mod.register_forward_pre_hook(lambda m, inp: cap.__setitem__('h', inp[0]))
    else: mod.register_forward_hook(lambda m, inp, o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))
    sae = TopKSAE.load(args.sae, dev).eval(); Fdim = sae.f
    smax = np.lib.format.open_memmap(acts / f'{slug}_sae_max.npy', 'w+', np.float32, (N, Fdim))
    samax = np.lib.format.open_memmap(acts / f'{slug}_sae_argmax.npy', 'w+', np.int16, (N, Fdim))
    active = torch.zeros(Fdim, dtype=torch.int64, device=dev); sse = 0.0; sst = 0.0; t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, N, args.batch):
            ids = torch.tensor(docs[b0:b0 + args.batch], device=dev); mask = ids >= 0
            model(input_ids=ids.clamp(min=0), attention_mask=mask.long())
            h = cap['h'].float(); v = torch.tensor(valid_all[b0:b0 + len(ids)], device=dev)
            z = sae.encode(h)                                                    # (B, T, F)
            hv = h[v]; sse += (sae.decode(z)[v] - hv).pow(2).sum().item(); sst += (hv - hv.mean(0)).pow(2).sum().item()
            zb = torch.where(v[..., None], z, torch.full_like(z, -1.0)); mx = zb.max(1)
            smax[b0:b0 + len(ids)] = mx.values.clamp(min=0).cpu().numpy(); samax[b0:b0 + len(ids)] = mx.indices.cpu().numpy().astype(np.int16)
            active += (mx.values > 0).sum(0)
            if (b0 // args.batch) % 100 == 0: print(f'[lm-sae-extract] {b0 + len(ids)}/{N}', flush=True)
    smax.flush(); samax.flush(); act = active.cpu().numpy(); np.save(acts / f'{slug}_sae_active.npy', act.astype(np.int32))
    info = {'sae': str(args.sae), 'features': Fdim, 'k': sae.k, 'val_fvu': sse / sst, 'alive_ge1': int((act >= 1).sum()), 'alive_ge500': int((act >= 500).sum()),
            'alive_ge100': int((act >= 100).sum()), 'extra': {k: v for k, v in getattr(sae, 'extra', {}).items() if k != 'log'}}
    json.dump(info, open(acts / f'{slug}_sae.json', 'w'), indent=1)
    print(f'[lm-sae-extract] done in {time.time()-t0:.0f}s: scan FVU {info["val_fvu"]:.4f}, alive>=100 docs {info["alive_ge100"]}/{Fdim}', flush=True)

if __name__ == '__main__':
    main()
