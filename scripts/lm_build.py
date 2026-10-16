#!/usr/bin/env python
"""Text microscope builder: for real units, random directions and SAE features of one LM site, the
bucketed maximally activating documents, each shown as a token window around the argmax token with
per-token activations (the text analogue of the crop + heatmap).

    python scripts/lm_build.py --root <root> --model Qwen/Qwen2.5-7B-Instruct --layer layer18.resid \
        --n-real 32 --n-virt 32 --out microscope            # kinds real,virt
    python scripts/lm_build.py ... --kinds sae --n-sae 32 --sae <ckpt> --acts-name <model>-sae --stats-name <model>
Writes microscope/<name>/data/<slug>.json (+ .js) with modality 'text' and updates microscope/<name>/meta.json.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from lm_extract import site_hooks

BUCKETS = [('top16', 'Top 16', 'strongest document maxima', list(range(1, 17))),
           ('p9999', '99.99%', 'around the 5th-highest document', list(range(5, 13))),
           ('p999', '99.9%', 'around the 50th-highest document', list(range(47, 55))),
           ('p99', '99%', 'around the 500th-highest document', list(range(497, 505)))]
LEFT, RIGHT = 24, 8   # window tokens before / after the argmax token

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--model', required=True); ap.add_argument('--layer', required=True)
    ap.add_argument('--n-real', type=int, default=32); ap.add_argument('--n-virt', type=int, default=32); ap.add_argument('--n-sae', type=int, default=32)
    ap.add_argument('--kinds', default='real,virt'); ap.add_argument('--sae', default=''); ap.add_argument('--sae-min-active', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default='microscope')
    ap.add_argument('--acts-name', default='', help='lm_acts subdir for real/virt stats; also the output subdir name'); ap.add_argument('--stats-name', default='')
    args = ap.parse_args()
    root = Path(args.root); mname = args.acts_name or args.model.split('/')[-1]; kinds = args.kinds.split(',')
    acts = root / 'lm_acts' / (args.stats_name or mname); meta = json.load(open(acts / 'meta.json'))
    L = next(l for l in meta['layers'] if l['name'] == args.layer); slug, D = L['slug'], L['D']
    docs = np.load(acts / 'docs_tokens.npy'); N, T = docs.shape
    out = Path(args.out) / mname; (out / 'data').mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    rng = np.random.default_rng(args.seed)
    units = []
    if 'real' in kinds or 'virt' in kinds:
        real_max = np.load(acts / f'{slug}_real_max.npy', mmap_mode='r'); real_min = np.load(acts / f'{slug}_real_min.npy', mmap_mode='r')
        real_amax = np.load(acts / f'{slug}_real_argmax.npy', mmap_mode='r'); real_amin = np.load(acts / f'{slug}_real_argmin.npy', mmap_mode='r')
        virt_max = np.load(acts / f'{slug}_virt_max.npy', mmap_mode='r'); virt_min = np.load(acts / f'{slug}_virt_min.npy', mmap_mode='r')
        virt_amax = np.load(acts / f'{slug}_virt_argmax.npy', mmap_mode='r'); virt_amin = np.load(acts / f'{slug}_virt_argmin.npy', mmap_mode='r')
        dirs = np.load(acts / f'{slug}_virtual_dirs.npy'); K = dirs.shape[1]; ustats = np.load(acts / f'{slug}_unit_stats.npz')
        real_units = [0] + sorted(rng.choice(np.arange(1, D), size=args.n_real - 1, replace=False).tolist()) if 'real' in kinds else []
        virt_units = sorted(rng.choice(np.arange(K), size=min(args.n_virt, K), replace=False).tolist()) if 'virt' in kinds else []
        units += [('real', u) for u in real_units] + [('virt', u) for u in virt_units]
    else:
        dirs = np.zeros((D, 0), dtype=np.float32); K = meta.get('k_virtual', 0); real_units = virt_units = []
    sae_units, sae_info = [], {}
    if 'sae' in kinds:
        sae_max = np.load(acts / f'{slug}_sae_max.npy', mmap_mode='r'); sae_amax = np.load(acts / f'{slug}_sae_argmax.npy', mmap_mode='r')
        sae_active = np.load(acts / f'{slug}_sae_active.npy'); sae_info = json.load(open(acts / f'{slug}_sae.json'))
        alive = np.flatnonzero(sae_active >= args.sae_min_active)
        sae_units = sorted(rng.choice(alive, size=min(args.n_sae, len(alive)), replace=False).tolist()); units += [('sae', u) for u in sae_units]
    print(f'[lm-build] {args.model} {args.layer}: width {D}; {len(real_units)} real, {len(virt_units)} virtual, {len(sae_units)} sae units over {N} docs', flush=True)
    dev = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    blk, site = args.layer.split('.'); li = int(blk.replace('layer', ''))
    _, _, _, _, mod, kind = site_hooks(model, [li], [site])[0]
    cap = {}
    if kind == 'pre': mod.register_forward_pre_hook(lambda m, inp: cap.__setitem__('h', inp[0]))
    else: mod.register_forward_hook(lambda m, inp, o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))
    dirs_t = torch.tensor(dirs, device=dev, dtype=torch.bfloat16)
    sae_model = None
    if sae_units:
        from sae import TopKSAE; sae_model = TopKSAE.load(args.sae, dev).eval()
    valid_all = docs >= 0; valid_all[:, 0] = False
    neurons = []; t0 = time.time()
    for kd, u in units:
        if kd == 'real': vmax, vmin, amax, amin = real_max[:, u].astype(np.float32), real_min[:, u].astype(np.float32), real_amax[:, u], real_amin[:, u]
        elif kd == 'virt': vmax, vmin, amax, amin = virt_max[:, u].astype(np.float32), virt_min[:, u].astype(np.float32), virt_amax[:, u], virt_amin[:, u]
        else: vmax, amax = sae_max[:, u].astype(np.float32), sae_amax[:, u]; vmin, amin = vmax, amax
        order = np.argsort(-vmax, kind='stable'); order_min = np.argsort(vmin, kind='stable')
        cells = []
        for bid, label, desc, ranks in BUCKETS:
            for r in ranks:
                d = int(order[r - 1]); cells.append({'bucketId': bid, 'docIndex': d, 'pos': int(amax[d]), 'rank': r, 'rankDirection': 'descending', 'quantile': 1 - (r - 1) / N, 'value': float(vmax[d])})
        for r in range(1, 5):
            d = int(order_min[r - 1]); cells.append({'bucketId': 'bottom4', 'docIndex': d, 'pos': int(amin[d]), 'rank': r, 'rankDirection': 'ascending', 'quantile': (r - 1) / N, 'value': float(vmin[d])})
        # re-forward the 44 documents to get this unit's per-token activation
        dix = [c['docIndex'] for c in cells]; ids = torch.tensor(docs[dix], device=dev); mask = ids >= 0
        with torch.no_grad():
            model(input_ids=ids.clamp(min=0), attention_mask=mask.long()); h = cap['h']
            a = h[..., u].float() if kd == 'real' else ((h @ dirs_t[:, u]).float() if kd == 'virt' else sae_model.encode(h.float())[..., u])
        a = a.cpu().numpy()
        for i, c in enumerate(cells):
            p = c['pos']; n_valid = int(mask[i].sum()); lo, hi = max(0, p - LEFT), min(n_valid, p + RIGHT + 1)
            toks = [tok.decode([t]) for t in docs[c['docIndex'], lo:hi].tolist()]
            c.update({'tokens': toks, 'acts': [round(float(x), 3) for x in a[i, lo:hi]], 'focus': p - lo, 'docMax': float(a[i, 1:n_valid].max()), 'datasetLabel': ''.join(toks[max(0, p - lo - 3):p - lo + 2]).replace('\n', ' ')})
        examples = [{'bucketId': b[0], 'label': b[1], 'description': b[2], 'source': 'maximum', 'cells': [c for c in cells if c['bucketId'] == b[0]]} for b in BUCKETS]
        examples.append({'bucketId': 'bottom4', 'label': 'Most negative' if kd != 'sae' else 'Least active', 'description': 'lowest token activation anywhere in a document' if kd != 'sae' else 'documents where the feature is least active', 'source': 'minimum', 'cells': [c for c in cells if c['bucketId'] == 'bottom4']})
        q = lambda r: float(vmax[order[r - 1]])
        stats = {'max': q(1), 'p9999': q(5), 'p999': q(50), 'p99': q(500), 'min': float(vmin[order_min[0]]), 'imageMaxMean': float(vmax.mean()), 'imageMaxStd': float(vmax.std())}
        if kd == 'real': stats.update({'mean': float(ustats['mean'][u]), 'std': float(ustats['std'][u])})
        elif kd == 'virt':
            coef = dirs[:, u]; top = np.argsort(-np.abs(coef))[:8]; stats['topCoefficients'] = [[int(i), float(coef[i])] for i in top]
            w2 = coef ** 2; stats['effectiveNeurons'] = float(w2.sum() ** 2 / (w2 ** 2).sum()); stats['top1Share'] = float(w2.max() / w2.sum())
        else:
            coef = sae_model.W_dec[u].detach().float().cpu().numpy(); top = np.argsort(-np.abs(coef))[:8]
            stats['topCoefficients'] = [[int(i), float(coef[i])] for i in top]; stats['activeImages'] = int(sae_active[u])
            w2 = coef ** 2; stats['effectiveNeurons'] = float(w2.sum() ** 2 / (w2 ** 2).sum()); stats['top1Share'] = float(w2.max() / w2.sum())
        neurons.append({'id': f'{kd}:{u}', 'kind': kd, 'channel': int(u), 'stats': stats, 'examples': examples, 'sheet': None})
        print(f'  {kd}_{u:05d}: max {stats["max"]:.3f} p99 {stats["p99"]:.3f} top: {cells[0]["datasetLabel"]!r} ({time.time()-t0:.0f}s)', flush=True)
    data = {'modality': 'text', 'layer': args.layer, 'slug': slug, 'site': L['site'], 'shape': [1, 1, D], 'imageCount': N, 'kVirtual': K, 'neurons': neurons,
            'blindOrder': rng.permutation(len(neurons)).tolist(), 'window': [LEFT, RIGHT]}
    json.dump(data, open(out / 'data' / f'{slug}.json', 'w'))
    mpath = out / 'meta.json'
    m = json.load(open(mpath)) if mpath.exists() else {'title': f'{mname} text microscope', 'model': args.model, 'modality': 'text', 'dataset': f'{meta["n_docs"]:,} FineWeb-edu documents x {meta["seq_len"]} tokens',
             'spatialStatistic': meta['spatialStatistic'], 'virtual': meta['virtual'], 'top1Acc': None, 'exampleSheet': {'cell': 0, 'columns': 16, 'rows': 3, 'cells': 44, 'positiveCells': 40},
             'cropGeometry': {'window': [LEFT, RIGHT]}, 'buckets': [{'id': b[0], 'label': b[1], 'description': b[2], 'count': len(b[3])} for b in BUCKETS] + [{'id': 'bottom4', 'label': 'Most negative', 'count': 4}], 'layers': []}
    m['layers'] = [l for l in m['layers'] if l['name'] != args.layer] + [{'name': args.layer, 'slug': slug, 'site': L['site'], 'stage': 0, 'channels': D, 'kVirtual': K, 'data': f'data/{slug}.json',
                   'nReal': len(real_units), 'nVirt': len(virt_units), 'nSae': len(sae_units), **({'sae': {k: sae_info.get(k) for k in ('features', 'k', 'val_fvu', 'alive_ge100', 'alive_ge500')}} if sae_units else {})}]
    m['layers'].sort(key=lambda l: (int(l['name'].split('.')[0].replace('layer', '')), l['name']))
    json.dump(m, open(mpath, 'w'), indent=1)
    print(f'[lm-build] wrote {out} in {time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__':
    main()
