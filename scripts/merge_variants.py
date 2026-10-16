#!/usr/bin/env python
"""Assemble the multi-model microscope site under microscope/:

  * folds the equal-variance ("std") virtual units of <model>-std into <model> as kind 'virt_std'
  * attaches CLIP coherence scores, recomputes the blind order
  * packs the per-unit example sheets of each (layer, kind) into one vertical sprite JPEG
    (sprites/<slug>_<kind>.jpg) so the published site stays well under the file-count limits
  * writes data/<slug>.json + data/<slug>.js (keyed "<model dir>/<slug>"), meta.js, and a top-level
    models.js + index.html (the viewer wrapped in a document skeleton)

    python scripts/merge_variants.py --site microscope --viewer microscope/viewer.html \
        --models convnext-tiny-224,convnext-base-224,convnext-large-224
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

def merge_model(site, mdir, seed):
    main_dir = site / mdir; std_dir = site / f'{mdir}-std'
    meta = json.load(open(main_dir / 'meta.json'))
    if meta.get('modality') != 'text': (main_dir / 'sprites').mkdir(exist_ok=True)
    for L in meta['layers']:
        slug = L['slug']
        d = json.load(open(main_dir / 'data' / f'{slug}.json'))
        coh = {}
        cp = main_dir / 'data' / f'{slug}_coherence.json'
        if cp.exists():
            c = json.load(open(cp)); coh.update(c['units']); d['coherenceBaseline'] = c['baselineRandom16']
        d['neurons'] = [n for n in d['neurons'] if n['kind'] in ('real', 'virt')]
        # fold in sibling variant dirs: <model>-std (virt -> virt_std) and <model>-sae (sae)
        for suffix, src_kind, dst_kind, count_key in [('-std', 'virt', 'virt_std', 'nVirtStd'), ('-sae', 'sae', 'sae', 'nSae')]:
            vdir = site / f'{mdir}{suffix}'
            if not (vdir / 'data' / f'{slug}.json').exists(): continue
            ds = json.load(open(vdir / 'data' / f'{slug}.json'))
            cps = vdir / 'data' / f'{slug}_coherence.json'
            cohs = json.load(open(cps))['units'] if cps.exists() else {}
            for n in ds['neurons']:
                if n['kind'] != src_kind: continue
                n2 = dict(n); n2['kind'] = dst_kind; n2['id'] = f'{dst_kind}:{n["channel"]}'; n2['sheetFile'] = str(vdir / 'examples' / slug / f'{src_kind}_{n["channel"]:04d}.jpg')
                if n['id'] in cohs: coh[n2['id']] = cohs[n['id']]
                d['neurons'].append(n2)
            L[count_key] = sum(1 for n in d['neurons'] if n['kind'] == dst_kind)
            vm = json.load(open(vdir / 'meta.json')); vl = next((l for l in vm['layers'] if l['slug'] == slug), {})
            if 'sae' in vl: L['sae'] = vl['sae']
        text = d.get('modality') == 'text'
        for n in d['neurons']:
            if not text and 'sheetFile' not in n: n['sheetFile'] = str(main_dir / 'examples' / slug / f'{n["kind"]}_{n["channel"]:04d}.jpg')  # builder's per-unit sheet (idempotent)
            if n['id'] in coh: n['coherence'] = {k: coh[n['id']][k] for k in ('top16', 'p999', 'p99')}
        # sprites: one vertical stack per kind, in unit order (image models only)
        for kind in ([] if text else sorted({n['kind'] for n in d['neurons']})):
            ns = [n for n in d['neurons'] if n['kind'] == kind]
            sheets = [Image.open(n['sheetFile']).convert('RGB') for n in ns]
            w, h = sheets[0].size
            sprite = Image.new('RGB', (w, h * len(sheets)))
            for i, (n, im) in enumerate(zip(ns, sheets)):
                sprite.paste(im, (0, i * h)); n['sheet'] = f'sprites/{slug}_{kind}.jpg'; n['sheetRow'] = i; n['sheetRows'] = len(sheets)
            sprite.save(main_dir / 'sprites' / f'{slug}_{kind}.jpg', quality=90)
        for n in d['neurons']: n.pop('sheetFile', None)
        d['blindOrder'] = np.random.default_rng(seed).permutation(len(d['neurons'])).tolist()
        json.dump(d, open(main_dir / 'data' / f'{slug}.json', 'w'))
        (main_dir / 'data' / f'{slug}.js').write_text('window.MICRO_DATA=window.MICRO_DATA||{};window.MICRO_DATA[' + json.dumps(f'{mdir}/{slug}') + ']=' + json.dumps(d) + ';')
        print(f'[merge] {mdir}/{slug}: {len(d["neurons"])} units, coherence for {sum("coherence" in n for n in d["neurons"])}, sprites {sorted({n["sheet"] for n in d["neurons"]})}')
    meta['dir'] = mdir
    meta['virtualStd'] = "equal-variance variant: Gaussian coefficients divided by each unit's activation std before normalising"
    meta['coherence'] = 'mean pairwise CLIP ViT-B/32 cosine similarity among the top-16 crops (label-free coherence); baseline = 16 crops drawn at random across all units of the layer'
    json.dump(meta, open(main_dir / 'meta.json', 'w'), indent=1)
    (main_dir / 'meta.js').write_text('window.MICRO_META=' + json.dumps(meta) + ';')
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', default='microscope'); ap.add_argument('--viewer', default='microscope/viewer.html')
    ap.add_argument('--models', required=True, help='comma list of model dirs under --site (each may have a -std sibling)')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    site = Path(args.site)
    models = [merge_model(site, m, args.seed) for m in args.models.split(',') if (site / m / 'meta.json').exists()]
    (site / 'models.js').write_text('window.MICRO_MODELS=' + json.dumps(models) + ';')
    body = Path(args.viewer).read_text()
    (site / 'index.html').write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n' + body + '\n</body>\n</html>\n')
    print(f'[merge] {len(models)} models -> {site / "models.js"}, {site / "index.html"}')

if __name__ == '__main__':
    main()
