#!/usr/bin/env python
"""Label-free interpretability proxy: visual coherence of a unit's top-16 crops.

Embeds every cell of each example sheet with CLIP ViT-B/32 and reports, per unit, the mean
pairwise cosine similarity among its top-16 crops (cells 0-15), among its 99%-bucket crops
(cells 32-39), and a baseline from 16 crops drawn at random across all units of the layer.

    python scripts/coherence.py --micro microscope/convnext-tiny-224 --layer stage3.block2
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--micro', required=True)
    ap.add_argument('--layer', required=True)
    ap.add_argument('--clip', default='openai/clip-vit-base-patch32')
    args = ap.parse_args()
    micro = Path(args.micro); slug = args.layer.replace('.', '_')
    d = json.load(open(micro / 'data' / f'{slug}.json'))
    meta = json.load(open(micro / 'meta.json')); cell = meta['exampleSheet']['cell']; cols = meta['exampleSheet']['columns']
    from transformers import CLIPModel, CLIPProcessor
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained(args.clip).to(dev).eval(); proc = CLIPProcessor.from_pretrained(args.clip)
    embs = {}
    for n in d['neurons']:
        sheet = Image.open(micro / n['sheet']).convert('RGB')
        y0 = n.get('sheetRow', 0) * meta['exampleSheet']['rows'] * cell   # sprite-packed sheets (after merge) stack units vertically
        crops = [sheet.crop(((k % cols) * cell, y0 + (k // cols) * cell, (k % cols + 1) * cell, y0 + (k // cols + 1) * cell)) for k in range(44)]
        with torch.no_grad():
            e = model.get_image_features(**proc(images=crops, return_tensors='pt').to(dev))
        embs[n['id']] = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
    def coh(E):
        S = E @ E.T; m = ~np.eye(len(E), dtype=bool); return float(S[m].mean())
    rng = np.random.default_rng(0); allc = np.concatenate([embs[n['id']][:40] for n in d['neurons']])
    base = np.mean([coh(allc[rng.choice(len(allc), 16, replace=False)]) for _ in range(500)])
    out = {'baselineRandom16': base, 'units': {}}
    for n in d['neurons']:
        E = embs[n['id']]
        out['units'][n['id']] = {'kind': n['kind'], 'top16': coh(E[:16]), 'p999': coh(E[24:32]), 'p99': coh(E[32:40])}
    for kind in sorted({u['kind'] for u in out['units'].values()}):
        v = np.array([u['top16'] for u in out['units'].values() if u['kind'] == kind])
        p = np.array([u['p99'] for u in out['units'].values() if u['kind'] == kind])
        print(f'{args.layer} {kind}: top16 CLIP coherence mean={v.mean():.3f} median={np.median(v):.3f} (min {v.min():.3f}, max {v.max():.3f}); p99 bucket mean={p.mean():.3f}; random-16 baseline={base:.3f}')
    json.dump(out, open(micro / 'data' / f'{slug}_coherence.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
