#!/usr/bin/env python
"""Score every SAE feature on the 50k ImageNet-1k validation center crops: per image, the maximum
feature activation over positions and its argmax (same statistic as for neurons), plus how many
images each feature fires on and the validation FVU of the SAE.

    python scripts/sae_extract.py --root <root> --model facebook/convnext-tiny-224 --layer stage3.block2 \
        --sae <root>/sae/convnext-tiny-224/stage3_block2/sae_x8_k32.pt
Writes into <root>/acts/<model>/: {slug}_sae_max.npy (N,F) f32, {slug}_sae_argmax.npy (N,F) i16,
{slug}_sae_active.npy (F) int32, {slug}_sae.json
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from sae import TopKSAE
from extract_acts import ValSet
from model_adapters import get_adapter
from torch.utils.data import DataLoader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--model', default='facebook/convnext-tiny-224')
    ap.add_argument('--layer', default='stage3.block2'); ap.add_argument('--sae', required=True)
    ap.add_argument('--batch-size', type=int, default=250); ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()
    root = Path(args.root); slug = args.layer.replace('.', '_'); mname = args.model.split('/')[-1]
    out = root / 'acts' / mname
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dev = 'cuda'
    from transformers import AutoModelForImageClassification
    model = AutoModelForImageClassification.from_pretrained(args.model).to(dev).eval()
    ad = get_adapter(model); to_spatial = ad['to_spatial']
    mod = next(l for l in ad['layers'] if l['name'] == args.layer)['module']
    cap = {}; mod.register_forward_hook(lambda mo, i, o: cap.__setitem__('h', to_spatial(o)))
    sae = TopKSAE.load(args.sae, dev).eval(); Fdim = sae.f
    ds = ValSet(root, ad['preprocess']); N = len(ds)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    smax = np.lib.format.open_memmap(out / f'{slug}_sae_max.npy', 'w+', np.float32, (N, Fdim))
    samax = np.lib.format.open_memmap(out / f'{slug}_sae_argmax.npy', 'w+', np.int16, (N, Fdim))
    active = torch.zeros(Fdim, dtype=torch.int64, device=dev)
    sse = 0.0; sst = 0.0; t0 = time.time(); done = 0
    with torch.no_grad():
        for x, idx in dl:
            model(x.to(dev, non_blocking=True)); h = cap['h'].flatten(1, 2)      # (B, HW, D)
            z = sae.encode(h)                                                     # (B, HW, F)
            recon = sae.decode(z)
            sse += (recon - h).pow(2).sum().item(); sst += (h - h.mean((0, 1))).pow(2).sum().item()
            mx = z.max(1)
            i0, i1 = idx[0].item(), idx[-1].item() + 1
            smax[i0:i1] = mx.values.cpu().numpy(); samax[i0:i1] = mx.indices.cpu().numpy().astype(np.int16)
            active += (mx.values > 0).sum(0)
            done += x.shape[0]
            if done % 10000 < args.batch_size: print(f'[sae-extract] {done}/{N} {done/(time.time()-t0):.0f} img/s', flush=True)
    smax.flush(); samax.flush()
    act = active.cpu().numpy(); np.save(out / f'{slug}_sae_active.npy', act.astype(np.int32))
    info = {'sae': str(args.sae), 'features': Fdim, 'k': sae.k, 'val_fvu': sse / sst, 'alive_ge1': int((act >= 1).sum()),
            'alive_ge500': int((act >= 500).sum()), 'median_active_images': float(np.median(act)), 'extra': getattr(sae, 'extra', {})}
    info['extra'].pop('log', None)
    json.dump(info, open(out / f'{slug}_sae.json', 'w'), indent=1)
    print(f'[sae-extract] done in {time.time()-t0:.0f}s: val FVU {info["val_fvu"]:.4f}, features alive on >=500 images: {info["alive_ge500"]}/{Fdim}', flush=True)

if __name__ == '__main__':
    main()
