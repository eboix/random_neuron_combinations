#!/usr/bin/env python
"""Collect one block's post-GELU MLP hidden activations at every spatial position, on ImageNet-1k
TRAIN images (shortest edge 256, center crop 224, same geometry as the microscope), into an fp16
memmap for SAE training. Images are written in a random order so contiguous slices are shuffled.

    python scripts/sae_collect.py --root /scratch/tmp/eboix/random_is_clean \
        --model facebook/convnext-tiny-224 --layer stage3.block2 --n-images 100000
Output: <root>/sae/<model>/<slug>/train_acts.npy  (n_images*HW, D) fp16, plus stats.json
"""
import argparse, os, io, json, re, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch
from PIL import Image
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
from model_adapters import get_adapter, pil_input, find_layer

PRE = {'resize': 256, 'crop': 224, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}   # set from the adapter before the Pool forks

def decode(b):
    return np.asarray(pil_input(Image.open(io.BytesIO(b)).convert('RGB'), PRE), dtype=np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--model', default='facebook/convnext-tiny-224')
    ap.add_argument('--layer', default='stage3.block2'); ap.add_argument('--n-images', type=int, default=100000)
    ap.add_argument('--batch', type=int, default=256); ap.add_argument('--workers', type=int, default=max(2, int(os.environ.get('SLURM_CPUS_PER_TASK', '12')) - 2))
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    root = Path(args.root); slug = args.layer.replace('.', '_'); mname = args.model.split('/')[-1]
    out = root / 'sae' / mname / slug; out.mkdir(parents=True, exist_ok=True)
    shards = sorted((root / 'imagenet-1k' / 'data').glob('train-*.parquet'))
    assert shards, 'no train shards yet'
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dev = 'cuda'
    from transformers import AutoModelForImageClassification
    model = AutoModelForImageClassification.from_pretrained(args.model).to(dev).eval()
    ad = get_adapter(model); PRE.update(ad['preprocess']); lyr = find_layer(ad, args.layer); to_spatial = lyr['to_spatial']; mod = lyr['module']
    cap = {}; mod.register_forward_hook(lambda mo, i, o: cap.__setitem__('h', to_spatial(o)))
    mean_t = torch.tensor(PRE['mean'], device=dev).view(1, 3, 1, 1); std_t = torch.tensor(PRE['std'], device=dev).view(1, 3, 1, 1)
    with torch.no_grad():
        model(torch.zeros(1, 3, PRE['crop'], PRE['crop'], device=dev)); _, H, W, D = cap['h'].shape
    HW = H * W; N = args.n_images
    acts = np.lib.format.open_memmap(out / 'train_acts.npy', 'w+', np.float16, (N * HW, D))
    perm = np.random.default_rng(args.seed).permutation(N)
    ssum = torch.zeros(D, dtype=torch.float64, device=dev); nsum = 0.0; count = 0
    done = 0; t0 = time.time(); used = []
    pool = Pool(args.workers)
    for sh in shards:
        if done >= N: break
        tbl = pq.read_table(sh, columns=['image'])
        bs = tbl.column('image').combine_chunks().field('bytes').to_pylist()
        bs = bs[:N - done]; used.append(sh.name)
        for i in range(0, len(bs), args.batch):
            chunk = bs[i:i + args.batch]
            x = torch.from_numpy(np.stack(pool.map(decode, chunk, chunksize=8))).to(dev).permute(0, 3, 1, 2).float() / 255
            x = (x - mean_t) / std_t
            with torch.no_grad():
                model(x); h = cap['h'].flatten(1, 2)              # (B, HW, D)
            ssum += h.sum((0, 1)).double(); nsum += h.norm(dim=-1).sum().item(); count += h.shape[0] * HW
            hn = h.half().cpu().numpy()
            for j in range(hn.shape[0]):
                s = perm[done + j]; acts[s * HW:(s + 1) * HW] = hn[j]
            done += hn.shape[0]
            if done % 10240 < args.batch:
                print(f'[collect] {done}/{N} {done/(time.time()-t0):.0f} img/s', flush=True)
    pool.close(); acts.flush()
    stats = {'model': args.model, 'layer': args.layer, 'n_images': done, 'HW': HW, 'D': D, 'positions': done * HW,
             'mean': (ssum / count).cpu().tolist(), 'mean_norm': nsum / count, 'shards': used, 'preprocess': PRE, 'arch': ad['arch']}
    json.dump(stats, open(out / 'stats.json', 'w'))
    print(f'[collect] done {done} images -> {out} in {time.time()-t0:.0f}s (mean norm {stats["mean_norm"]:.3f})', flush=True)

if __name__ == '__main__':
    main()
