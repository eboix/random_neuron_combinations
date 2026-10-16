#!/usr/bin/env python
"""One pass over the ImageNet-1k validation set (50k center crops) through a HF ConvNeXt.

For every block we hook the post-GELU MLP hidden activations -- the "neurons" of the
Goodfire microscope (their post-SwiGLU product coordinate) -- and record per image:
  * real units:    max over positions, argmax position, min over positions
  * virtual units: the same for K fixed random unit-norm directions in the hidden space
                   (Gaussian in raw activation space, seeded per layer), i.e. random
                   linear combinations of the block's neurons
plus running mean/var per unit, predictions, and the direction matrices.

    python scripts/extract_acts.py --root /scratch/tmp/eboix/random_is_clean \
        --model facebook/convnext-tiny-224 --k-virtual 128
"""
import argparse, json, os, re, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model_adapters import get_adapter, torch_transform, IMAGENET_MEAN, IMAGENET_STD

MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
DEFAULT_PRE = {'resize': 256, 'crop': 224, 'mean': MEAN, 'std': STD}   # Goodfire / ConvNeXt geometry

class ValSet(Dataset):
    def __init__(self, root, pre=None):
        self.files = sorted((Path(root) / 'val').glob('*.jpg'))
        self.tf = torch_transform(pre or DEFAULT_PRE)
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert('RGB')), i

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--model', default='facebook/convnext-tiny-224')
    ap.add_argument('--k-virtual', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=250)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0, help='debug: only first N images')
    ap.add_argument('--virt-std-from', default='', help='acts dir of a previous run: scale each Gaussian coefficient by 1/std of that unit (equal-variance random combination)')
    ap.add_argument('--out-name', default='', help='output subdir name under acts/ (default: model name)')
    args = ap.parse_args()
    root = Path(args.root)
    out = root / 'acts' / (args.out_name or args.model.split('/')[-1]); out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False  # keep activations exact fp32
    torch.backends.cudnn.allow_tf32 = False
    dev = 'cuda'
    from transformers import AutoModelForImageClassification
    model = AutoModelForImageClassification.from_pretrained(args.model).to(dev).eval()
    adapter = get_adapter(model); layers = adapter['layers']; to_spatial = adapter['to_spatial']
    assert layers, 'no blocks found'
    captured = {}
    def mk(i):
        def hook(mod, inp, outp): captured[i] = to_spatial(outp)
        return hook
    for i, L in enumerate(layers): L['module'].register_forward_hook(mk(i))

    ds = ValSet(root, adapter['preprocess'])
    n = len(ds) if not args.limit else min(args.limit, len(ds))
    if args.limit: ds.files = ds.files[:n]
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    print(f'[extract] {args.model}: {len(layers)} blocks, {n} images, K={args.k_virtual}', flush=True)

    # probe shapes with one image
    with torch.no_grad():
        model(ds[0][0][None].to(dev))
    g = torch.Generator().manual_seed(args.seed)
    store = {}
    for i, L in enumerate(layers):
        h = captured[i]; B, H, W, D = h.shape
        L.update({'H': H, 'W': W, 'D': D, 'stride': 224 // H})
        Wd = torch.randn(D, args.k_virtual, generator=g)
        if args.virt_std_from:
            sd = np.load(Path(args.virt_std_from) / f'{L["slug"]}_unit_stats.npz')['std']
            Wd = Wd / torch.tensor(np.maximum(sd, 1e-3), dtype=torch.float32)[:, None]
        Wd = Wd / Wd.norm(dim=0, keepdim=True)
        L['dirs'] = Wd.to(dev)
        np.save(out / f'{L["slug"]}_virtual_dirs.npy', Wd.numpy())
        s = {}
        s['real_max'] = np.lib.format.open_memmap(out / f'{L["slug"]}_real_max.npy', 'w+', np.float32, (n, D))
        s['real_min'] = np.lib.format.open_memmap(out / f'{L["slug"]}_real_min.npy', 'w+', np.float16, (n, D))
        s['real_argmax'] = np.lib.format.open_memmap(out / f'{L["slug"]}_real_argmax.npy', 'w+', np.int16, (n, D))
        s['real_argmin'] = np.lib.format.open_memmap(out / f'{L["slug"]}_real_argmin.npy', 'w+', np.int16, (n, D))
        for key in ['max', 'min', 'argmax', 'argmin']:
            dt = np.int16 if key.startswith('arg') else np.float32
            s[f'virt_{key}'] = np.lib.format.open_memmap(out / f'{L["slug"]}_virt_{key}.npy', 'w+', dt, (n, args.k_virtual))
        s['sum'] = torch.zeros(D, dtype=torch.float64, device=dev)
        s['sumsq'] = torch.zeros(D, dtype=torch.float64, device=dev)
        s['count'] = 0
        store[i] = s
        print(f'  {L["name"]}: {H}x{W}x{D} stride {L["stride"]}', flush=True)
    preds = np.zeros(n, dtype=np.int16); logits_top = np.zeros(n, dtype=np.float32)

    t0 = time.time(); done = 0
    with torch.no_grad():
        for x, idx in dl:
            x = x.to(dev, non_blocking=True)
            logits = model(x).logits
            i0, i1 = idx[0].item(), idx[-1].item() + 1
            preds[i0:i1] = logits.argmax(1).cpu().numpy(); logits_top[i0:i1] = logits.max(1).values.cpu().numpy()
            for i, L in enumerate(layers):
                h = captured[i].flatten(1, 2)                     # (B, HW, D)
                s = store[i]
                mx = h.max(1); mn = h.min(1)
                s['real_max'][i0:i1] = mx.values.cpu().numpy(); s['real_argmax'][i0:i1] = mx.indices.cpu().numpy().astype(np.int16)
                s['real_min'][i0:i1] = mn.values.cpu().numpy().astype(np.float16); s['real_argmin'][i0:i1] = mn.indices.cpu().numpy().astype(np.int16)
                s['sum'] += h.sum((0, 1)).double(); s['sumsq'] += (h * h).sum((0, 1)).double(); s['count'] += h.shape[0] * h.shape[1]
                p = h @ L['dirs']                                  # (B, HW, K)
                pmx = p.max(1); pmn = p.min(1)
                s['virt_max'][i0:i1] = pmx.values.cpu().numpy(); s['virt_argmax'][i0:i1] = pmx.indices.cpu().numpy().astype(np.int16)
                s['virt_min'][i0:i1] = pmn.values.cpu().numpy(); s['virt_argmin'][i0:i1] = pmn.indices.cpu().numpy().astype(np.int16)
            done += x.shape[0]
            if done % 5000 < args.batch_size:
                print(f'[extract] {done}/{n} {done/(time.time()-t0):.0f} img/s', flush=True)
    labels = np.load(root / 'val_labels.npy')[:n]
    acc = float((preds == labels).mean())
    meta = {'model': args.model, 'arch': adapter['arch'], 'preprocess': adapter['preprocess'], 'n_images': n, 'k_virtual': args.k_virtual, 'seed': args.seed,
            'top1_acc': acc, 'spatialStatistic': 'maximum post-GELU MLP hidden activation per image',
            'virtual': ('K Gaussian unit-norm directions in the raw hidden space (seed per run, generator shared across layers in block order)' if not args.virt_std_from else
                        'K unit-norm directions with Gaussian coefficients scaled by 1/std of each unit (equal-variance random combination), seed per run'),
            'layers': [{k: v for k, v in L.items() if k not in ('module', 'dirs')} for L in layers]}
    for i, L in enumerate(layers):
        s = store[i]
        mean = (s['sum'] / s['count']).cpu().numpy(); var = (s['sumsq'] / s['count']).cpu().numpy() - mean ** 2
        np.savez(out / f'{L["slug"]}_unit_stats.npz', mean=mean, std=np.sqrt(np.maximum(var, 0)), count=s['count'])
        for k, v in s.items():
            if isinstance(v, np.memmap): v.flush()
    np.save(out / 'preds.npy', preds); np.save(out / 'logits_top.npy', logits_top)
    json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
    print(f'[extract] done in {time.time()-t0:.0f}s; top-1 acc {acc:.4f}; outputs in {out}', flush=True)

if __name__ == '__main__':
    main()
