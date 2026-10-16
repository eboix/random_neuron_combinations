#!/usr/bin/env python
"""Build Goodfire-style microscope assets (example sheets, heatmaps, per-layer JSON) for a
set of real neurons and virtual neurons (random linear combinations) of one ConvNeXt block,
from the per-image statistics written by extract_acts.py.

Bucket convention copied from already-clean.goodfire.com (medium microscope):
  top16 = ranks 1-16, p9999 = ranks 5-12, p999 = ranks 47-54, p99 = ranks 497-504 (by the
  image's maximum spatial activation), bottom4 = the 4 lowest spatial minima.
Sheet: 16 columns x 3 rows of 96 px crops, 44 cells (40 positive + 4 negative).
Crops: focus window per stage [64, 96, 144, 192] px centred on the argmax position.

    python scripts/build_microscope.py --root /scratch/tmp/eboix/random_is_clean \
        --model facebook/convnext-tiny-224 --layer stage3.block2 --n-real 32 --n-virt 32 --out microscope
"""
import argparse, base64, json, re, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from model_adapters import get_adapter, pil_input

FOCUS = [64, 96, 144, 192]
BUCKETS = [('top16', 'Top 16', 'strongest image maxima', list(range(1, 17))),
           ('p9999', '99.99%', 'around the 5th-highest image', list(range(5, 13))),
           ('p999', '99.9%', 'around the 50th-highest image', list(range(47, 55))),
           ('p99', '99%', 'around the 500th-highest image', list(range(497, 505)))]
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

def load_input(root, idx, cache, pre):
    """The exact model input (e.g. shortest edge 256, center crop 224) as a PIL image."""
    if idx in cache: return cache[idx]
    im = pil_input(Image.open(root / 'val' / f'{idx:05d}.jpg').convert('RGB'), pre)
    if len(cache) > 4000: cache.clear()
    cache[idx] = im
    return im

def crop_cell(im, pos, H, W, stride, win, cell):
    IN = im.size[0]
    py, px = divmod(int(pos), W)
    cy, cx = (py + 0.5) * stride, (px + 0.5) * stride
    l = int(round(min(max(cx - win / 2, 0), IN - win))); t = int(round(min(max(cy - win / 2, 0), IN - win)))
    return im.crop((l, t, l + win, t + win)).resize((cell, cell), Image.BILINEAR), (l, t, win)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--model', default='facebook/convnext-tiny-224')
    ap.add_argument('--layer', default='stage3.block2')
    ap.add_argument('--n-real', type=int, default=32)
    ap.add_argument('--n-virt', type=int, default=32)
    ap.add_argument('--real-units', default='', help='explicit comma list (overrides --n-real sampling except unit 0 always kept)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='microscope')
    ap.add_argument('--cell', type=int, default=96)
    ap.add_argument('--no-heatmaps', action='store_true')
    ap.add_argument('--acts-name', default='', help='acts subdir (default: model name); also used as the output subdir')
    ap.add_argument('--kinds', default='real,virt', help='comma subset of real,virt,sae')
    ap.add_argument('--n-sae', type=int, default=32); ap.add_argument('--sae', default='', help='TopK SAE checkpoint (needed for sae heatmaps)')
    ap.add_argument('--sae-min-active', type=int, default=500, help='sample SAE features that fire on at least this many val images')
    ap.add_argument('--stats-name', default='', help='acts subdir holding the per-image stats (default: --acts-name / model name)')
    args = ap.parse_args()
    root = Path(args.root); mname = args.acts_name or args.model.split('/')[-1]
    acts = root / 'acts' / (args.stats_name or mname)
    kinds = args.kinds.split(',')
    meta = json.load(open(acts / 'meta.json'))
    L = next(l for l in meta['layers'] if l['name'] == args.layer)
    slug, H, W, D, stride, stage = L['slug'], L['H'], L['W'], L['D'], L['stride'], L['stage']
    win = L.get('focus', FOCUS[stage])
    pre = meta.get('preprocess', {'resize': 256, 'crop': 224, 'mean': list(MEAN), 'std': list(STD)})
    out = Path(args.out) / mname; (out / 'data').mkdir(parents=True, exist_ok=True)
    (out / 'examples' / slug).mkdir(parents=True, exist_ok=True); (out / 'heatmaps' / slug).mkdir(parents=True, exist_ok=True)
    labels = np.load(root / 'val_labels.npy')
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    names = {int(k): v.split(',')[0] for k, v in cfg.id2label.items()}

    real_max = np.load(acts / f'{slug}_real_max.npy', mmap_mode='r'); real_min = np.load(acts / f'{slug}_real_min.npy', mmap_mode='r')
    real_amax = np.load(acts / f'{slug}_real_argmax.npy', mmap_mode='r'); real_amin = np.load(acts / f'{slug}_real_argmin.npy', mmap_mode='r')
    virt_max = np.load(acts / f'{slug}_virt_max.npy', mmap_mode='r'); virt_min = np.load(acts / f'{slug}_virt_min.npy', mmap_mode='r')
    virt_amax = np.load(acts / f'{slug}_virt_argmax.npy', mmap_mode='r'); virt_amin = np.load(acts / f'{slug}_virt_argmin.npy', mmap_mode='r')
    dirs = np.load(acts / f'{slug}_virtual_dirs.npy'); K = dirs.shape[1]
    ustats = np.load(acts / f'{slug}_unit_stats.npz')
    N = real_max.shape[0]
    rng = np.random.default_rng(args.seed)
    if args.real_units:
        real_units = [int(x) for x in args.real_units.split(',')]
    else:
        real_units = [0] + sorted(rng.choice(np.arange(1, D), size=args.n_real - 1, replace=False).tolist())
    virt_units = sorted(rng.choice(np.arange(K), size=min(args.n_virt, K), replace=False).tolist())
    sae_units, sae_max, sae_amax, sae_active, sae_info, sae_model = [], None, None, None, {}, None
    if 'sae' in kinds:
        sae_max = np.load(acts / f'{slug}_sae_max.npy', mmap_mode='r'); sae_amax = np.load(acts / f'{slug}_sae_argmax.npy', mmap_mode='r')
        sae_active = np.load(acts / f'{slug}_sae_active.npy'); sae_info = json.load(open(acts / f'{slug}_sae.json'))
        alive = np.flatnonzero(sae_active >= args.sae_min_active)
        sae_units = sorted(rng.choice(alive, size=min(args.n_sae, len(alive)), replace=False).tolist())
    units = ([('real', u) for u in real_units] if 'real' in kinds else []) + ([('virt', u) for u in virt_units] if 'virt' in kinds else []) + [('sae', u) for u in sae_units]
    print(f'[build] {args.model} {args.layer}: {H}x{W}x{D} stride {stride} win {win}; kinds {kinds}: {len(real_units)} real, {len(virt_units)} virtual, {len(sae_units)} sae units', flush=True)

    model = None
    if not args.no_heatmaps:
        import torch
        from transformers import AutoModelForImageClassification
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = AutoModelForImageClassification.from_pretrained(args.model).to(dev).eval()
        cap = {}
        ad = get_adapter(model); to_spatial = ad['to_spatial']
        mod = next(l for l in ad['layers'] if l['name'] == args.layer)['module']
        mod.register_forward_hook(lambda m, i, o: cap.__setitem__('h', to_spatial(o)))
        dirs_t = torch.tensor(dirs, device=dev)
        if 'sae' in kinds:
            import sys; sys.path.insert(0, str(Path(__file__).parent)); from sae import TopKSAE
            sae_model = TopKSAE.load(args.sae, dev).eval()

    cache = {}; neurons = []; t0 = time.time(); heat_b64 = None
    cols, rows, cell = 16, 3, args.cell
    for kind, u in units:
        neg_desc = 'lowest spatial response anywhere in an image'
        if kind == 'real':
            vmax, vmin, amax, amin = real_max[:, u].astype(np.float32), real_min[:, u].astype(np.float32), real_amax[:, u], real_amin[:, u]
        elif kind == 'virt':
            vmax, vmin, amax, amin = virt_max[:, u].astype(np.float32), virt_min[:, u].astype(np.float32), virt_amax[:, u], virt_amin[:, u]
        else:  # SAE features are non-negative: the negative bucket shows images with the lowest image maximum
            vmax, amax = sae_max[:, u].astype(np.float32), sae_amax[:, u]; vmin, amin = vmax, amax; neg_desc = 'images where the feature is least active (lowest image maximum)'
        order = np.argsort(-vmax, kind='stable')            # rank r (1-based) -> image order[r-1]
        order_min = np.argsort(vmin, kind='stable')
        sheet = Image.new('RGB', (cols * cell, rows * cell), (239, 233, 220))
        examples, cells_flat, c = [], [], 0
        for bid, label, desc, ranks in BUCKETS:
            cells = []
            for r in ranks:
                idx = int(order[r - 1]); pos = int(amax[idx])
                im = load_input(root, idx, cache, pre)
                crop, box = crop_cell(im, pos, H, W, stride, win, cell)
                sheet.paste(crop, ((c % cols) * cell, (c // cols) * cell)); c += 1
                cells.append({'imageIndex': idx, 'datasetLabel': names[int(labels[idx])], 'rank': r, 'rankDirection': 'descending',
                              'quantile': 1 - (r - 1) / N, 'value': float(vmax[idx]), 'position': pos, 'box': box})
            examples.append({'bucketId': bid, 'label': label, 'description': desc, 'source': 'maximum', 'cells': cells}); cells_flat += cells
        cells = []
        for r in range(1, 5):
            idx = int(order_min[r - 1]); pos = int(amin[idx])
            im = load_input(root, idx, cache, pre)
            crop, box = crop_cell(im, pos, H, W, stride, win, cell)
            sheet.paste(crop, ((c % cols) * cell, (c // cols) * cell)); c += 1
            cells.append({'imageIndex': idx, 'datasetLabel': names[int(labels[idx])], 'rank': r, 'rankDirection': 'ascending',
                          'quantile': (r - 1) / N, 'value': float(vmin[idx]), 'position': pos, 'box': box})
        examples.append({'bucketId': 'bottom4', 'label': 'Most negative', 'description': neg_desc, 'source': 'minimum', 'cells': cells})
        fname = f'{kind}_{u:04d}'
        sheet.save(out / 'examples' / slug / f'{fname}.jpg', quality=88)
        if model is not None:
            import torch
            ims = [load_input(root, cl['imageIndex'], cache, pre) for cl in cells_flat]
            pm, ps = np.array(pre['mean']), np.array(pre['std'])
            x = torch.tensor(np.stack([(np.asarray(im, dtype=np.float32) / 255 - pm) / ps for im in ims]).transpose(0, 3, 1, 2), dtype=torch.float32, device=dev)
            with torch.no_grad():
                model(x); h = cap['h']                          # (40, H, W, D)
                m = h[..., u] if kind == 'real' else (h @ dirs_t[:, u] if kind == 'virt' else sae_model.encode(h)[..., u])
                m = m.clamp(min=0); m = m / m.flatten(1).max(1).values.clamp(min=1e-6)[:, None, None]   # (40, H, W), per-image max = 1
                IN = ims[0].size[0]
                up = torch.nn.functional.interpolate(m[:, None], size=(IN, IN), mode='bilinear', align_corners=False)   # map over the model input
                # overlay must match the crop window of each cell: cut the same (l, t, win) box out of the upsampled map
                m = torch.cat([torch.nn.functional.interpolate(up[i:i + 1, :, bt:bt + bw, bl:bl + bw], size=(24, 24), mode='bilinear', align_corners=False)
                               for i, (bl, bt, bw) in enumerate(cl['box'] for cl in cells_flat)])[:, 0]
                hm = (m * 255).round().byte().cpu().numpy().tobytes()
                (out / 'heatmaps' / slug / f'{fname}.bin').write_bytes(hm)
                heat_b64 = base64.b64encode(hm).decode()
        q = lambda r: float(vmax[order[r - 1]])
        stats = {'max': q(1), 'p9999': q(5), 'p999': q(50), 'p99': q(500), 'min': float(vmin[order_min[0]]),
                 'imageMaxMean': float(vmax.mean()), 'imageMaxStd': float(vmax.std())}
        if kind == 'real':
            stats.update({'mean': float(ustats['mean'][u]), 'std': float(ustats['std'][u])})
        elif kind == 'virt':
            coef = dirs[:, u]; top = np.argsort(-np.abs(coef))[:8]
            stats['topCoefficients'] = [[int(i), float(coef[i])] for i in top]
            w2 = coef ** 2; stats['effectiveNeurons'] = float(w2.sum() ** 2 / (w2 ** 2).sum()); stats['top1Share'] = float(w2.max() / w2.sum())
        else:
            coef = sae_model.W_dec[u].detach().cpu().numpy() if sae_model is not None else np.zeros(D); top = np.argsort(-np.abs(coef))[:8]
            stats['topCoefficients'] = [[int(i), float(coef[i])] for i in top]; stats['activeImages'] = int(sae_active[u])
        neurons.append({'id': f'{kind}:{u}', 'kind': kind, 'channel': int(u), 'stats': stats, 'examples': examples,
                        'sheet': f'examples/{slug}/{fname}.jpg', 'heatmaps24': heat_b64})
        print(f'  {fname}: max {stats["max"]:.3f} p99 {stats["p99"]:.3f} top label {cells_flat[0]["datasetLabel"]!r} ({time.time()-t0:.0f}s)', flush=True)
    # blind order: a seeded shuffle of the units, identity hidden by the viewer until revealed
    blind = rng.permutation(len(neurons)).tolist()
    data = {'layer': args.layer, 'slug': slug, 'shape': [H, W, D], 'stride': stride, 'stage': stage, 'focusWindow': win,
            'imageCount': N, 'kVirtual': K, 'neurons': neurons, 'blindOrder': blind}
    json.dump(data, open(out / 'data' / f'{slug}.json', 'w'))
    (out / 'data' / f'{slug}.js').write_text('window.MICRO_DATA=window.MICRO_DATA||{};window.MICRO_DATA[' + json.dumps(slug) + ']=' + json.dumps(data) + ';')
    mpath = out / 'meta.json'
    m = json.load(open(mpath)) if mpath.exists() else {'title': f'{mname} ImageNet-1k val microscope', 'model': args.model, 'dataset': 'ImageNet-1k validation, 50,000 center crops (256 -> 224)',
             'spatialStatistic': meta['spatialStatistic'], 'virtual': meta['virtual'], 'top1Acc': meta['top1_acc'], 'exampleSheet': {'cell': cell, 'columns': cols, 'rows': rows, 'cells': 44, 'positiveCells': 40},
             'cropGeometry': {'inputCenterCrop': pre['crop'], 'sourceImage': pre['resize'], 'focusCropByStage': FOCUS, 'preprocess': pre, 'arch': meta.get('arch', 'convnext')}, 'buckets': [{'id': b[0], 'label': b[1], 'description': b[2], 'count': len(b[3])} for b in BUCKETS] + [{'id': 'bottom4', 'label': 'Most negative', 'count': 4}],
             'layers': []}
    m['layers'] = [l for l in m['layers'] if l['name'] != args.layer] + [{'name': args.layer, 'slug': slug, 'stage': stage, 'focus': win, 'channels': D, 'kVirtual': K, 'data': f'data/{slug}.json',
                   'nReal': len(real_units) if 'real' in kinds else 0, 'nVirt': len(virt_units) if 'virt' in kinds else 0, 'nSae': len(sae_units),
                   **({'sae': {k: sae_info.get(k) for k in ('features', 'k', 'val_fvu', 'alive_ge500')}} if sae_units else {})}]
    m['layers'].sort(key=lambda l: l['name'])
    json.dump(m, open(mpath, 'w'), indent=1)
    (out / 'meta.js').write_text('window.MICRO_META=' + json.dumps(m) + ';')
    print(f'[build] wrote {out} in {time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__':
    main()
