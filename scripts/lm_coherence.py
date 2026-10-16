#!/usr/bin/env python
"""Label-free coherence for text units: embed each unit's top-16 token windows with a sentence-embedding
model (mean-pooled MiniLM) and report the mean pairwise cosine, plus a baseline from 16 windows drawn at
random across all units of the layer. Same output format as coherence.py.

    python scripts/lm_coherence.py --micro microscope/Qwen2.5-7B-Instruct --layer layer18.resid
"""
import argparse, json
from pathlib import Path
import numpy as np, torch

def embed(texts, tok, model, dev):
    out = []
    for i in range(0, len(texts), 64):
        b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors='pt').to(dev)
        with torch.no_grad():
            h = model(**b).last_hidden_state; m = b['attention_mask'][..., None].float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(torch.nn.functional.normalize(e, dim=-1).cpu())
    return torch.cat(out).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--micro', required=True); ap.add_argument('--layer', required=True); ap.add_argument('--embedder', default='sentence-transformers/all-MiniLM-L6-v2')
    args = ap.parse_args()
    micro = Path(args.micro); slug = args.layer.replace('.', '_'); d = json.load(open(micro / 'data' / f'{slug}.json'))
    from transformers import AutoTokenizer, AutoModel
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(args.embedder); model = AutoModel.from_pretrained(args.embedder).to(dev).eval()
    embs = {}
    for n in d['neurons']:
        cells = [c for ex in n['examples'] for c in ex['cells']]
        embs[n['id']] = embed([''.join(c['tokens']) for c in cells], tok, model, dev)
    def coh(E):
        S = E @ E.T; m = ~np.eye(len(E), dtype=bool); return float(S[m].mean())
    rng = np.random.default_rng(0); allc = np.concatenate([embs[n['id']][:40] for n in d['neurons']])
    base = float(np.mean([coh(allc[rng.choice(len(allc), 16, replace=False)]) for _ in range(500)]))
    out = {'baselineRandom16': base, 'units': {}}
    for n in d['neurons']:
        E = embs[n['id']]; out['units'][n['id']] = {'kind': n['kind'], 'top16': coh(E[:16]), 'p999': coh(E[24:32]), 'p99': coh(E[32:40])}
    for kind in sorted({u['kind'] for u in out['units'].values()}):
        v = np.array([u['top16'] for u in out['units'].values() if u['kind'] == kind]); p = np.array([u['p99'] for u in out['units'].values() if u['kind'] == kind])
        print(f'{args.layer} {kind}: top16 text coherence mean={v.mean():.3f} median={np.median(v):.3f} (min {v.min():.3f}, max {v.max():.3f}); p99 bucket mean={p.mean():.3f}; random-16 baseline={base:.3f}')
    json.dump(out, open(micro / 'data' / f'{slug}_coherence.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
