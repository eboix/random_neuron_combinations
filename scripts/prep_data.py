#!/usr/bin/env python
"""Download ImageNet-1k (HF ILSVRC/imagenet-1k parquet shards) and ConvNeXt weights to
node-local scratch, and unpack the validation split into individual JPEG files
(original bytes, no re-encode) plus labels for fast random access.

    python scripts/prep_data.py --root /scratch/tmp/eboix/random_is_clean --split validation
    python scripts/prep_data.py --root ... --split train          # background, ~140 GB
"""
import argparse, json, os, sys, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--split', default='validation', choices=['validation', 'train', 'test'])
    ap.add_argument('--models', default='facebook/convnext-tiny-224,facebook/convnext-base-224')
    ap.add_argument('--no-unpack', action='store_true')
    args = ap.parse_args()
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    t0 = time.time()
    for m in [x for x in args.models.split(',') if x]:
        p = snapshot_download(repo_id=m, allow_patterns=['*.json', '*.safetensors', '*.bin', '*.txt'])
        print(f'[prep] model {m} -> {p}', flush=True)
    ds_dir = root / 'imagenet-1k'
    p = snapshot_download(repo_id='ILSVRC/imagenet-1k', repo_type='dataset', local_dir=str(ds_dir),
                          allow_patterns=[f'data/{args.split}-*.parquet', 'classes.py', 'README.md'],
                          max_workers=8)
    print(f'[prep] dataset split={args.split} -> {p} ({time.time()-t0:.0f}s)', flush=True)
    if args.no_unpack or args.split != 'validation':
        return
    import pyarrow.parquet as pq
    out = root / 'val'; out.mkdir(exist_ok=True)
    files = sorted((ds_dir / 'data').glob('validation-*.parquet'))
    meta, i = [], 0
    for f in files:
        tbl = pq.read_table(f)
        imgs = tbl.column('image').combine_chunks()
        labels = tbl.column('label').to_pylist()
        bs = imgs.field('bytes'); ps = imgs.field('path')
        for j in range(len(tbl)):
            b = bs[j].as_py(); path = ps[j].as_py()
            with open(out / f'{i:05d}.jpg', 'wb') as fh:
                fh.write(b)
            meta.append({'index': i, 'file': path, 'label': labels[j]})
            i += 1
        print(f'[prep] unpacked {f.name}: total {i} ({time.time()-t0:.0f}s)', flush=True)
    json.dump(meta, open(root / 'val_meta.json', 'w'))
    import numpy as np
    np.save(root / 'val_labels.npy', np.array([m['label'] for m in meta], dtype=np.int16))
    print(f'[prep] done: {i} validation images in {out}', flush=True)

if __name__ == '__main__':
    main()
