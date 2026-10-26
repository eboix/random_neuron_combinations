#!/usr/bin/env python
"""Experiments A and B of combo_interp (docs/centroid_experiment_A.md, docs/locality_experiment_B.md) on the
vision models of this repo: do neuron input weights act as centroids of the MLP input (A), and are a unit's top
images local in MLP-input space, and does that explain the CLIP coherence / class purity gap (B)?

Same data as the published coherence/purity results: ImageNet-1k val, 50k center crops, per-image maximum over
positions of the post-GELU MLP hidden unit. The "MLP input" x is the input of the block's first MLP linear layer
(ConvNeXt: post-LN input of pwconv1 at each position; ViT/DeiT: LN2 output at each patch token), so W_in[j]·x + b_j
is exactly neuron j's pre-activation.

Stages (each reads the previous stage's outputs under <root>/mech/<model>/):
  capture   pass 1: x at one random position per image (+ untrained-model copy with --init random); per-unit mean/std
            over all positions; hidden covariance from 4 random positions per image
  centroids Experiment A on the captured x (combo_interp.geometry.centroid_eval)
  units     unit sets per layer: neuron, related_5/20, ksparse_2/5/20/100 (combo_interp.units), dense_raw / dense_std
            (the published virt / virt_std directions, all seed-order hypotheses), xdir (random directions in x space)
  maxes     pass 2: per-image max / argmax over positions for every unit
  resolve   match the published virt / virt_std units by their top-16 images; fix the final unit list
  topcap    pass 3: x and the hidden vector h at each unit's top-100 (image, argmax position)
  clip      CLIP ViT-B/32 coherence of each unit's top-16 crops (same crops as build_microscope.py) + class purity
  locality  Experiment B metrics, mediation of clip coherence / purity, report

    python scripts/vision_mech.py --root ~/combo_runs/vision --model facebook/convnext-tiny-224 \
        --layers stage2.block8,stage3.block2 --stages all
"""
import argparse, json, os, re, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'combo_interp' / 'src'))

STAGES = ['capture', 'centroids', 'units', 'maxes', 'resolve', 'topcap', 'clip', 'locality']
FOCUS_CONVNEXT = [64, 96, 144, 192]
COND_ORDER = ['neuron', 'related_5', 'related_20', 'ksparse_2', 'ksparse_5', 'ksparse_20', 'ksparse_100',
              'dense_raw', 'dense_std', 'xdir']


# ----------------------------------------------------------------------------- data (torchvision-exact preprocessing)
def tv_resize_crop(im, pre):
    """torchvision Resize(r) [+ CenterCrop(c)] on a PIL image, reproduced exactly (same size rounding)."""
    r, c = pre['resize'], pre['crop']
    w, h = im.size
    if isinstance(r, (list, tuple)):
        return im.resize((r[1], r[0]), Image.BILINEAR)
    if w <= h:
        nw, nh = r, int(r * h / w)
    else:
        nw, nh = int(r * w / h), r
    im = im.resize((nw, nh), Image.BILINEAR)
    top, left = int(round((nh - c) / 2.0)), int(round((nw - c) / 2.0))
    return im.crop((left, top, left + c, top + c))


class ValSet:
    def __init__(self, root, pre, indices=None):
        self.root, self.pre = Path(root), pre
        self.idx = np.arange(len(list((self.root / 'val').glob('*.jpg')))) if indices is None else np.asarray(indices)
        self.mean = np.array(pre['mean'], np.float32)[:, None, None]
        self.std = np.array(pre['std'], np.float32)[:, None, None]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        import torch
        j = int(self.idx[i])
        im = tv_resize_crop(Image.open(self.root / 'val' / f'{j:05d}.jpg').convert('RGB'), self.pre)
        a = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy((a - self.mean) / self.std), j


def loader(root, pre, indices=None, bs=125, workers=6):
    from torch.utils.data import DataLoader
    return DataLoader(ValSet(root, pre, indices), batch_size=bs, num_workers=workers,
                      persistent_workers=False, pin_memory=False)


# ----------------------------------------------------------------------------- model parts
def device():
    import torch
    return 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')


def load_model(name, init='trained', seed=0):
    import torch
    from transformers import AutoConfig, AutoModelForImageClassification
    if init == 'random':
        torch.manual_seed(seed)
        m = AutoModelForImageClassification.from_config(AutoConfig.from_pretrained(name))
    else:
        m = AutoModelForImageClassification.from_pretrained(name)
    return m.to(device()).eval()


def preprocess(model):
    """The preprocessing the published runs used (model_adapters.get_adapter)."""
    from model_adapters import get_adapter
    return get_adapter(model, sites=['mlp'])['preprocess']


def mlp_parts(model, layer):
    """(in_linear, act_module, to_spatial, focus) for a layer name like stage3.block2 (ConvNeXt) or layer11 (ViT/DeiT)."""
    arch = model.config.model_type
    mods = dict(model.named_modules())
    if arch == 'convnext':
        s, b = map(int, re.fullmatch(r'stage(\d+)\.block(\d+)', layer).groups())
        base = f'convnext.encoder.stages.{s}.layers.{b}'
        return mods[base + '.pwconv1'], mods[base + '.act'], (lambda o: o), FOCUS_CONVNEXT[s]
    if arch in ('vit', 'deit'):
        b = int(re.fullmatch(r'layer(\d+)', layer).group(1))
        H = model.config.image_size // model.config.patch_size
        old, new = f'{arch}.encoder.layer.{b}.intermediate', f'{arch}.layers.{b}.mlp'   # transformers <5 / >=5 names
        lin, act = (mods[old + '.dense'], mods[old + '.intermediate_act_fn']) if old + '.dense' in mods else \
                   (mods[new + '.fc1'], mods[new + '.activation_fn'])

        def tok(o):
            o = o[0] if isinstance(o, tuple) else o
            B, T, D = o.shape
            return o[:, T - H * H:, :].reshape(B, H, H, D)
        return lin, act, tok, 144
    raise ValueError(arch)


class Capture:
    """Hooks x (input of the first MLP linear) and h (post-GELU hidden) at the given layers, as (B, H, W, ·)."""

    def __init__(self, model, layers):
        self.x, self.h, self.hs = {}, {}, []
        for L in layers:
            lin, act, ts, _ = mlp_parts(model, L)
            self.hs.append(lin.register_forward_pre_hook(lambda m, a, L=L, ts=ts: self.x.__setitem__(L, ts(a[0]))))
            self.hs.append(act.register_forward_hook(lambda m, a, o, L=L, ts=ts: self.h.__setitem__(L, ts(o))))

    def close(self):
        for h in self.hs:
            h.remove()


def site_layers(model, sites):
    """(name, site, D) of every hook site in the order model_adapters.get_adapter lists them."""
    arch, cfg = model.config.model_type, model.config
    out = []
    if arch == 'convnext':
        for st, depth in enumerate(cfg.depths):
            C = cfg.hidden_sizes[st]
            for b in range(depth):
                for site in ['mlp', 'out', 'resid']:
                    if site in sites:
                        out.append((f'stage{st}.block{b}', site, 4 * C if site == 'mlp' else C))
    else:
        for b in range(cfg.num_hidden_layers):
            for site in ['mlp', 'attn_out', 'mlp_out', 'resid']:
                if site in sites:
                    out.append((f'layer{b}', site, cfg.intermediate_size if site == 'mlp' else cfg.hidden_size))
    return out


def virt_dir_hypotheses(model, layers, K=128, seed=0):
    """The published virtual directions come from extract_acts.py: one torch Generator(seed), randn(D, K) per hooked
    layer in adapter order, unit-normalised (virt_std: divided by the unit std first). Which layers were hooked
    depends on the run (all blocks with sites=mlp, or --blocks <published layers> with all sites as in
    slurm/vision_sites.sh), so each order is generated and resolved later by matching top images.
    Returns {hyp: {layer: (D, K) raw Gaussian}}."""
    import torch
    full = ['mlp', 'out', 'resid'] if model.config.model_type == 'convnext' else ['mlp', 'attn_out', 'mlp_out', 'resid']
    out = {}
    for hyp, sites, restrict in [('all_mlp', ['mlp'], False), ('blocks_mlp', ['mlp'], True), ('blocks_all_sites', full, True)]:
        g = torch.Generator().manual_seed(seed)
        out[hyp] = {}
        for name, site, D in site_layers(model, sites):
            if restrict and name not in layers:
                continue
            W = torch.randn(D, K, generator=g).numpy()
            if site == 'mlp' and name in layers:
                out[hyp][name] = W
    return out


# ----------------------------------------------------------------------------- stages
def stage_capture(a, M, init):
    import torch
    model = load_model(a.model, init, a.seed)
    pre = preprocess(model)
    labels = np.load(a.root / 'val_labels.npy')
    n = len(labels) if not a.limit else a.limit
    cap = Capture(model, a.layers)
    acc = {}
    rng_pos = np.random.default_rng(a.seed + (1 if init == 'random' else 0))
    t0, done = time.time(), 0
    with torch.no_grad():
        for xb, idx in loader(a.root, pre, np.arange(n), a.batch, a.workers):
            model(xb.to(device()))
            idx = idx.numpy()
            for L in a.layers:
                x, h = cap.x[L], cap.h[L]
                B, H, W, C = x.shape
                D = h.shape[-1]
                if L not in acc:
                    acc[L] = {'X': np.zeros((n, C), np.float16), 'pos': np.zeros(n, np.int16), 'sum': np.zeros(D), 'sumsq': np.zeros(D),
                              'cnt': 0, 'csum': np.zeros(D), 'cxx': np.zeros((D, D)), 'ccnt': 0, 'HW': (H, W)}
                s = acc[L]
                p = rng_pos.integers(0, H * W, B)
                xf = x.reshape(B, H * W, C)
                s['X'][idx] = xf[torch.arange(B), torch.as_tensor(p, device=xf.device)].float().cpu().numpy()
                s['pos'][idx] = p
                if init == 'trained':
                    hf = h.reshape(B, H * W, D).float()
                    s['sum'] += hf.sum((0, 1)).cpu().numpy().astype(np.float64)
                    s['sumsq'] += (hf * hf).sum((0, 1)).cpu().numpy().astype(np.float64)
                    s['cnt'] += B * H * W
                    q = torch.as_tensor(rng_pos.integers(0, H * W, (B, 4)), device=hf.device)
                    hs = hf[torch.arange(B, device=hf.device)[:, None], q].reshape(-1, D)
                    s['csum'] += hs.sum(0).cpu().numpy().astype(np.float64)
                    s['cxx'] += (hs.T @ hs).cpu().numpy().astype(np.float64)
                    s['ccnt'] += hs.shape[0]
            done += len(idx)
            if done % 5000 < a.batch:
                print(f'[capture:{init}] {done}/{n} {done / (time.time() - t0):.0f} img/s', flush=True)
    cap.close()
    for L in a.layers:
        s = acc[L]
        lin = mlp_parts(model, L)[0]
        out = {'X': s['X'], 'pos': s['pos'], 'labels': labels[:n], 'HW': np.array(s['HW']),
               'W_in': lin.weight.detach().float().cpu().numpy(), 'b_in': lin.bias.detach().float().cpu().numpy()}
        if init == 'trained':
            mean = s['sum'] / s['cnt']
            cm = s['csum'] / s['ccnt']
            out.update(mean=mean, std=np.sqrt(np.maximum(s['sumsq'] / s['cnt'] - mean ** 2, 0)),
                       cov_mean=cm, cov=s['cxx'] / s['ccnt'] - np.outer(cm, cm))
        np.savez(M / f'capture_{init}_{L.replace(".", "_")}.npz', **out)
    print(f'[capture:{init}] done in {time.time() - t0:.0f}s', flush=True)


def stage_centroids(a, M):
    """Experiment A with the exact setup of the LM results (combo_interp/scripts/09_centroids.py, configs/default.yaml
    `geometry:`): centered and uncentered MLP inputs, assignment rules cos (primary) and euclid, M in
    {32, 128, 512, d_ff} with 10 / 10 / 5 / 3 repetitions, combo_2/5/20, trained and random-init models, scored and
    reported by combo_interp.geometry_report.write_report. W_in rows (pwconv1 / fc1) play the role of the gate rows;
    there is no `up` set (plain GELU MLP). Labels: ImageNet class (in place of the LM's document label) and spatial
    position (in place of the next token); fit / eval split = even / odd image index (LM: even / odd document)."""
    import pandas as pd
    from combo_interp.geometry import centroid_eval
    from combo_interp.geometry_report import write_report
    rules = tuple(a.assign.split(','))
    out = M / ('centroids' if rules == ('cos', 'euclid') else 'centroids_' + '_'.join(rules))
    out.mkdir(exist_ok=True)
    for tag, center in [('centered', True), ('raw', False)]:
        rows = []
        for init in ['trained', 'random']:
            for L in a.layers:
                f = M / f'capture_{init}_{L.replace(".", "_")}.npz'
                if not f.exists():
                    print(f'[centroids] missing {f.name}, skipped'); continue
                z = np.load(f)
                X, Wg = z['X'], z['W_in']
                fit = np.arange(len(X)) % 2 == 0
                labels = {'class': z['labels'].astype(np.int64), 'pos': z['pos'].astype(np.int64)}
                cap_m = min(Wg.shape[0], int(fit.sum()) // 2)          # only binds on tiny debug runs
                Ms = sorted({min(m, cap_m) for m in [32, 128, 512, 3072]})
                reps = {min(k, cap_m): v for k, v in {32: 10, 128: 10, 512: 5, 3072: 3}.items()}
                for r in centroid_eval(X, Wg, None, labels, fit, Ms, reps, combo_ks=(2, 5, 20), center=center,
                                       seed=a.seed + a.layers.index(L), assign_rules=rules,
                                       log=lambda m, L=L, init=init: print(f'{tag} {init} {L} {m}', flush=True)):
                    rows.append({'init': init, 'layer': L, **r})
                if 'ip' in rules:
                    rows += gate_bias_rows(X, Wg, z['b_in'], labels, fit, Ms, center, init, L)
        df = pd.DataFrame(rows)
        df.to_csv(out / f'centroids_{tag}.csv', index=False)
        # the shared report knows the LM label names; map class -> doc slot, position -> next-token slot, then rename
        write_report(df.rename(columns={'nmi_class': 'nmi_doc', 'nmi_pos': 'nmi_next'}).assign(nmi_token=np.nan), out, tag)
        for f in out.glob(f'*_{tag}.*'):
            if f.suffix in ('.md', '.csv'):
                t = f.read_text().replace('nmi_doc', 'nmi_class').replace('nmi_next', 'nmi_pos').replace(
                    'doc label / token / next token', 'ImageNet class / spatial position').replace('held-out tokens', 'held-out positions')
                f.write_text(t)
            if f.name.startswith(('nmi_doc', 'nmi_next')):
                f.rename(out / f.name.replace('nmi_doc', 'nmi_class').replace('nmi_next', 'nmi_pos'))
        print((out / f'summary_{tag}.md').read_text())


def gate_bias_rows(X, Wg, b, labels, fit, Ms, center, init, L, seed=0):
    """ip with the first layer's bias: argmax_j (W_in[j]·x + b_j), the neuron with the largest pre-activation
    (meaningful on uncentered x). Same subsets of rows as the gate set would draw are not reproducible here, so M
    random rows per repetition (all rows at M = d_ff); scored like combo_interp.geometry.partition_metrics."""
    import torch
    from combo_interp.geometry import assign, nmi, unit_rows, var_explained
    rng = np.random.default_rng(seed + 40_000)
    X = X.astype(np.float32)
    if center:
        X = X - X[fit].mean(0)
    Xc = torch.from_numpy(np.ascontiguousarray(X[~fit]))
    Xn = torch.from_numpy(np.ascontiguousarray(unit_rows(X[~fit])))
    lab = {k: v[~fit] for k, v in labels.items()}
    rows = []
    for Mv in Ms:
        for r in range(1 if Mv >= len(Wg) else {32: 10, 128: 10, 512: 5}.get(Mv, 3)):
            idx = rng.choice(len(Wg), Mv, replace=False)
            C = torch.from_numpy(np.ascontiguousarray(Wg[idx], dtype=np.float32))
            a_ = assign(Xc, C, bias=torch.from_numpy(b[idx].astype(np.float32)))
            sizes = torch.bincount(a_, minlength=Mv).double()
            p = sizes[sizes > 0] / sizes.sum()
            an = a_.numpy(); perm = rng.permutation(an)
            rows.append({'init': init, 'layer': L, 'assign': 'ip', 'set': 'gate_bias', 'M': Mv, 'rep': r,
                         'obj': np.nan, 'dist': np.nan, 've': var_explained(Xn, a_, Mv), 've_raw': var_explained(Xc, a_, Mv),
                         'eff_k': float(torch.exp(-(p * p.log()).sum())), 'used': int((sizes > 0).sum()),
                         **{f'nmi_{k}': nmi(an, y) - nmi(perm, y) for k, y in lab.items()}})
    return rows


def published_units(a, L):
    """{'real': [channels], 'virt': [ids], 'virt_std': [ids]} and their published top-16 image indices + coherence."""
    slug = L.replace('.', '_')
    pubs = {}
    for sub, kinds in [(a.name, ('real', 'virt')), (a.name + '-std', ('virt',))]:
        d = a.repo / 'microscope' / sub / 'data'
        if not (d / f'{slug}.js').exists():
            continue
        s = (d / f'{slug}.js').read_text()
        j = json.loads(s[s.index(']=') + 2:].rstrip().rstrip(';'))
        coh = json.loads((d / f'{slug}_coherence.json').read_text()) if (d / f'{slug}_coherence.json').exists() else {'units': {}}
        for u in j['neurons']:
            kind = u['kind']
            if kind not in kinds and kind != 'virt_std':
                continue
            key = 'virt_std' if (kind == 'virt' and sub.endswith('-std')) or kind == 'virt_std' else kind
            top = [c['imageIndex'] for c in next(e for e in u['examples'] if e['bucketId'] == 'top16')['cells']]
            labs = [c.get('datasetLabel') for c in next(e for e in u['examples'] if e['bucketId'] == 'top16')['cells']]
            pubs.setdefault(key, {})[int(u['channel'])] = {'top16': top, 'labels': labs,
                                                           'coh': coh['units'].get(u['id'], {}).get('top16')}
    return pubs


def stage_units(a, M):
    from combo_interp.units import make_unit_set
    model = load_model(a.model)
    hyps = virt_dir_hypotheses(model, a.layers, K=128, seed=0)
    for L in a.layers:
        slug = L.replace('.', '_')
        z = np.load(M / f'capture_trained_{slug}.npz')
        mean, std, cov = z['mean'], z['std'], z['cov']
        cov = cov.copy()
        D = len(mean)
        pubs = published_units(a, L)
        rng = np.random.default_rng(a.seed + 101)
        cols, meta = [], []

        def add(vec, m):
            cols.append(vec.astype(np.float32)); meta.append(m)
        # neurons: the published ones first, then random others up to n_per
        pub_real = sorted(pubs.get('real', {}))
        others = [j for j in rng.permutation(np.where(std > 1e-6)[0]) if j not in pub_real]
        for j in (pub_real + others)[:max(a.n_per, len(pub_real))]:
            e = np.zeros(D); e[j] = 1
            add(e, {'cond': 'neuron', 'neurons': [int(j)], 'weights': [1.0], 'published': j in pub_real})
        # sparse / related combinations exactly as in combo_interp (z-scored inputs, Gaussian / sign-aligned weights)
        ucfg = SimpleNamespace(min_std=1e-6, per_condition=a.n_per, standardize_sparse_inputs=True, conditions=[
            {'name': 'related_5', 'kind': 'related', 'k': 5}, {'name': 'related_20', 'kind': 'related', 'k': 20},
            {'name': 'ksparse_2', 'kind': 'ksparse', 'k': 2}, {'name': 'ksparse_5', 'kind': 'ksparse', 'k': 5},
            {'name': 'ksparse_20', 'kind': 'ksparse', 'k': 20}, {'name': 'ksparse_100', 'kind': 'ksparse', 'k': 100}])
        cov_full = cov.copy()
        cov_full[np.diag_indices(D)] = np.maximum(np.diag(cov_full), 0)
        us = make_unit_set(a.layers.index(L), z['cov_mean'], cov_full, ucfg, a.seed)
        for c, m in enumerate(us['meta']):
            add(us['A'][:, c], {'cond': m['cond'], 'neurons': m['neurons'], 'weights': m['weights'], 'k': m['k']})
        # published random directions under each seed-order hypothesis (raw and equal-variance)
        for hyp, dirs in hyps.items():
            Wr = dirs[L] / np.linalg.norm(dirs[L], axis=0, keepdims=True)
            Ws = dirs[L] / np.maximum(std, 1e-3)[:, None]
            Ws = Ws / np.linalg.norm(Ws, axis=0, keepdims=True)
            for k in range(Wr.shape[1]):
                add(Wr[:, k], {'cond': f'cand_raw:{hyp}', 'virt': k})
                add(Ws[:, k], {'cond': f'cand_std:{hyp}', 'virt': k})
        A = np.stack(cols, 1)
        C = z['X'].shape[1]
        R = np.random.default_rng(a.seed + 7).standard_normal((C, a.n_per)).astype(np.float32)
        np.savez(M / f'units_{slug}.npz', A=A, R=R)
        (M / f'units_{slug}.json').write_text(json.dumps(meta))
        print(f'[units] {L}: {A.shape[1]} hidden-space units + {R.shape[1]} xdir', flush=True)


def stage_maxes(a, M):
    import torch
    model = load_model(a.model)
    pre = preprocess(model)
    cap = Capture(model, a.layers)
    n = len(np.load(a.root / 'val_labels.npy')) if not a.limit else a.limit
    U = {L: np.load(M / f'units_{L.replace(".", "_")}.npz') for L in a.layers}
    At = {L: torch.tensor(U[L]['A'], device=device()) for L in a.layers}
    Rt = {L: torch.tensor(U[L]['R'], device=device()) for L in a.layers}
    out = {L: {k: np.zeros((n, w), dt) for k, w, dt in [('zmax', At[L].shape[1], np.float32), ('zarg', At[L].shape[1], np.int16),
                                                         ('rmax', Rt[L].shape[1], np.float32), ('rarg', Rt[L].shape[1], np.int16)]}
           for L in a.layers}
    t0, done = time.time(), 0
    with torch.no_grad():
        for xb, idx in loader(a.root, pre, np.arange(n), a.batch, a.workers):
            model(xb.to(device()))
            idx = idx.numpy()
            for L in a.layers:
                B = len(idx)
                h = cap.h[L].reshape(B, -1, cap.h[L].shape[-1]).float()
                x = cap.x[L].reshape(B, -1, cap.x[L].shape[-1]).float()
                v, i = (h @ At[L]).max(1)
                out[L]['zmax'][idx] = v.cpu().numpy(); out[L]['zarg'][idx] = i.cpu().numpy()
                v, i = (x @ Rt[L]).max(1)
                out[L]['rmax'][idx] = v.cpu().numpy(); out[L]['rarg'][idx] = i.cpu().numpy()
            done += len(idx)
            if done % 5000 < a.batch:
                print(f'[maxes] {done}/{n} {done / (time.time() - t0):.0f} img/s', flush=True)
    cap.close()
    for L in a.layers:
        np.savez(M / f'maxes_{L.replace(".", "_")}.npz', **out[L])


def stage_resolve(a, M):
    """Pick the seed-order hypothesis whose candidate directions reproduce the published units' top-16 images."""
    report = {}
    for L in a.layers:
        slug = L.replace('.', '_')
        meta = json.loads((M / f'units_{slug}.json').read_text())
        mx = np.load(M / f'maxes_{slug}.npz')
        zmax = mx['zmax']
        pubs = published_units(a, L)
        top16 = lambda c: set(np.argsort(-zmax[:, c], kind='stable')[:16].tolist())
        chk = {}
        for c, m in enumerate(meta):
            if m['cond'] == 'neuron' and m.get('published'):
                chk.setdefault('real', []).append(len(top16(c) & set(pubs['real'][m['neurons'][0]]['top16'])) / 16)
        best = {}
        for kind, key in [('raw', 'virt'), ('std', 'virt_std')]:
            scores = {}
            for c, m in enumerate(meta):
                if m['cond'].startswith(f'cand_{kind}:') and m['virt'] in pubs.get(key, {}):
                    hyp = m['cond'].split(':')[1]
                    scores.setdefault(hyp, []).append(len(top16(c) & set(pubs[key][m['virt']]['top16'])) / 16)
            chk.update({f'{key}:{h}': float(np.mean(v)) for h, v in scores.items()})
            best[kind] = max(scores, key=lambda h: np.mean(scores[h])) if scores else 'all_mlp'
        chk['real'] = float(np.mean(chk.get('real', [np.nan])))
        # final unit list: everything except the candidate sets of the losing hypotheses; dense_* = winner's 128
        # directions, published ones first, capped at max(n_per, #published)
        keep, final = [], []
        for kind, cond, key in [('raw', 'dense_raw', 'virt'), ('std', 'dense_std', 'virt_std')]:
            cands = [(c, m) for c, m in enumerate(meta) if m['cond'] == f'cand_{kind}:{best[kind]}']
            pubset = set(pubs.get(key, {}))
            cands.sort(key=lambda cm: (cm[1]['virt'] not in pubset, cm[1]['virt']))
            for c, m in cands[:max(a.n_per, len(pubset))]:
                final.append((c, {'cond': cond, 'virt': m['virt'], 'published': m['virt'] in pubset}))
        final = [(c, m) for c, m in enumerate(meta) if not m['cond'].startswith('cand_')] + final
        (M / f'final_{slug}.json').write_text(json.dumps([{'col': c, **m} for c, m in final]))
        report[L] = {'top16_overlap_with_published': chk, 'chosen': best}
        print(f'[resolve] {L}: {json.dumps(report[L])}', flush=True)
    (M / 'resolve.json').write_text(json.dumps(report, indent=1))


def top_list(vals, args_, n):
    o = np.argsort(-vals, kind='stable')[:n]
    return o, args_[o]


def stage_topcap(a, M):
    import torch
    model = load_model(a.model)
    pre = preprocess(model)
    need, plan = {}, {}
    for L in a.layers:
        slug = L.replace('.', '_')
        final = json.loads((M / f'final_{slug}.json').read_text())
        mx = np.load(M / f'maxes_{slug}.npz')
        plan[L] = []
        for u in final:
            im, pos = top_list(mx['zmax'][:, u['col']], mx['zarg'][:, u['col']], a.n_top)
            plan[L].append(('z', u['col'], im, pos))
        for j in range(mx['rmax'].shape[1]):
            im, pos = top_list(mx['rmax'][:, j], mx['rarg'][:, j], a.n_top)
            plan[L].append(('r', j, im, pos))
        pairs = sorted({(int(i), int(p)) for _, _, ims, ps in plan[L] for i, p in zip(ims, ps)})
        need[L] = {pr: k for k, pr in enumerate(pairs)}
    imgs = sorted({i for L in a.layers for (i, _) in need[L]})
    by_img = {L: {} for L in a.layers}
    for L in a.layers:
        for (i, p), k in need[L].items():
            by_img[L].setdefault(i, []).append((p, k))
    cap = Capture(model, a.layers)
    store = {L: None for L in a.layers}
    t0, done = time.time(), 0
    with torch.no_grad():
        for xb, idx in loader(a.root, pre, imgs, a.batch, a.workers):
            model(xb.to(device()))
            for L in a.layers:
                B = len(idx)
                x = cap.x[L].reshape(B, -1, cap.x[L].shape[-1]).float().cpu().numpy()
                h = cap.h[L].reshape(B, -1, cap.h[L].shape[-1]).float().cpu().numpy()
                if store[L] is None:
                    P = len(need[L])
                    store[L] = {'X': np.zeros((P, x.shape[-1]), np.float16), 'H': np.zeros((P, h.shape[-1]), np.float16)}
                for b, i in enumerate(idx.numpy()):
                    for p, k in by_img[L].get(int(i), []):
                        store[L]['X'][k] = x[b, p]; store[L]['H'][k] = h[b, p]
            done += len(idx)
            if done % 5000 < a.batch:
                print(f'[topcap] {done}/{len(imgs)} images', flush=True)
    cap.close()
    for L in a.layers:
        slug = L.replace('.', '_')
        index = {f'{t}{c}': [need[L][(int(i), int(p))] for i, p in zip(ims, ps)] for t, c, ims, ps in plan[L]}
        tops = {f'{t}{c}': {'img': ims.tolist(), 'pos': ps.tolist()} for t, c, ims, ps in plan[L]}
        np.savez(M / f'topcap_{slug}.npz', X=store[L]['X'], H=store[L]['H'])
        (M / f'topcap_{slug}.json').write_text(json.dumps({'index': index, 'tops': tops}))
    print(f'[topcap] done in {time.time() - t0:.0f}s', flush=True)


def crop(root, idx, pos, H, W, focus, pre, cache):
    """The same crop as build_microscope.crop_cell: focus window centred on the argmax position, resized to 96 px."""
    if idx not in cache:
        if len(cache) > 3000:
            cache.clear()
        cache[idx] = tv_resize_crop(Image.open(root / 'val' / f'{idx:05d}.jpg').convert('RGB'), pre)
    im = cache[idx]
    IN, stride = im.size[0], im.size[0] // H
    py, px = divmod(int(pos), W)
    cy, cx = (py + 0.5) * stride, (px + 0.5) * stride
    l = int(round(min(max(cx - focus / 2, 0), IN - focus))); t = int(round(min(max(cy - focus / 2, 0), IN - focus)))
    return im.crop((l, t, l + focus, t + focus)).resize((96, 96), Image.BILINEAR)


def stage_clip(a, M):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    dev = device()
    clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(dev).eval()
    proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model_cfg = load_model(a.model)
    pre = preprocess(model_cfg)
    labels = np.load(a.root / 'val_labels.npy')
    cache = {}

    def embed(crops):
        with torch.no_grad():
            e = clip.get_image_features(**proc(images=crops, return_tensors='pt').to(dev))
            e = e.pooler_output if hasattr(e, 'pooler_output') else e
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()

    def coh(E):
        S = E @ E.T
        return float(S[~np.eye(len(E), dtype=bool)].mean())
    for L in a.layers:
        slug = L.replace('.', '_')
        _, _, _, focus = mlp_parts(model_cfg, L)
        HW = np.load(M / f'capture_trained_{slug}.npz')['HW']
        tops = json.loads((M / f'topcap_{slug}.json').read_text())['tops']
        rows = {}
        for key, t in tops.items():
            ims, ps = t['img'][:16], t['pos'][:16]
            E = embed([crop(a.root, i, p, HW[0], HW[1], focus, pre, cache) for i, p in zip(ims, ps)])
            lab = labels[ims]
            rows[key] = {'clip_coh': coh(E), 'purity': float(np.bincount(lab).max() / len(lab))}
        rng = np.random.default_rng(0)
        base = []
        for _ in range(100):
            ims = rng.choice(len(labels), 16, replace=False)
            E = embed([crop(a.root, int(i), int(rng.integers(0, HW[0] * HW[1])), HW[0], HW[1], focus, pre, cache) for i in ims])
            base.append(coh(E))
        (M / f'clip_{slug}.json').write_text(json.dumps({'units': rows, 'baseline_random16': float(np.median(base))}))
        print(f'[clip] {L}: {len(rows)} units, random-16 baseline {np.median(base):.3f}', flush=True)


def stage_locality(a, M):
    import pandas as pd
    from combo_interp.locality import _unit, design, mediation, ols, set_metrics, within_cond_spearman
    rng = np.random.default_rng(a.seed + 2024)
    rows, base_rows, valid = [], [], []
    model = load_model(a.model)
    for L in a.layers:
        slug = L.replace('.', '_')
        z = np.load(M / f'capture_trained_{slug}.npz')
        mu = z['X'].astype(np.float32).mean(0)
        Xu = _unit(z['X'].astype(np.float32) - mu)
        Wn = _unit(z['W_in'])
        U = np.load(M / f'units_{slug}.npz')
        A = U['A']
        final = {f'z{u["col"]}': u for u in json.loads((M / f'final_{slug}.json').read_text())}
        tc = np.load(M / f'topcap_{slug}.npz')
        Xt, Ht = tc['X'].astype(np.float32), tc['H'].astype(np.float32)
        tj = json.loads((M / f'topcap_{slug}.json').read_text())
        cl = json.loads((M / f'clip_{slug}.json').read_text())
        pubs = published_units(a, L)
        n = a.n_top
        base = pd.DataFrame([set_metrics(Xu[rng.choice(len(Xu), n, replace=False)], rng) for _ in range(a.n_random_sets)])
        base_coh = float(base.coh.median())
        base_rows += [{'layer': L, **r} for r in base.to_dict('records')]
        for key, ix in tj['index'].items():
            if key.startswith('r'):
                meta = {'cond': 'xdir', 'unit': key}
            else:
                meta = {**final[key], 'unit': key}
            Xn = _unit(Xt[ix] - mu)
            r = {'layer': L, 'unit': key, 'cond': meta['cond'], 'published': bool(meta.get('published', False)),
                 **set_metrics(Xn, rng)}
            r['coh_excess'] = r['coh'] - base_coh
            m = _unit(Xn.mean(0, keepdims=True))[0]
            if meta['cond'] == 'neuron':
                j = meta['neurons'][0]
                r['cos_win'] = float(abs(Wn[j] @ m) / np.median(np.abs(Wn[rng.choice(len(Wn), 200, replace=False)] @ m)))
            if key.startswith('z') and meta['cond'] != 'neuron':
                a_col = A[:, int(key[1:])]
                nz = np.flatnonzero(a_col)
                contrib = Ht[ix][:, nz] * a_col[nz][None, :]
                dom = contrib.argmax(1)
                c = np.bincount(dom, minlength=len(nz)) / len(dom)
                p = c[c > 0]
                r['dom_share'], r['dom_eff'] = float(c.max()), float(np.exp(-(p * np.log(p)).sum()))
            r.update(cl['units'].get(key, {}))
            r['clip_coh_excess'] = r.get('clip_coh', np.nan) - cl['baseline_random16']
            if meta['cond'] in ('neuron', 'dense_raw', 'dense_std') and meta.get('published'):
                pk = {'neuron': 'real', 'dense_raw': 'virt', 'dense_std': 'virt_std'}[meta['cond']]
                pid = meta['neurons'][0] if meta['cond'] == 'neuron' else meta['virt']
                r['published_clip_coh'] = pubs.get(pk, {}).get(pid, {}).get('coh')
            rows.append(r)
        print(f'[locality] {L}: {sum(r["layer"] == L for r in rows)} units', flush=True)
    df = pd.DataFrame(rows)
    out = M / 'locality'
    out.mkdir(exist_ok=True)
    df.to_csv(out / 'unit_locality.csv', index=False)
    pd.DataFrame(base_rows).to_csv(out / 'random_sets.csv', index=False)
    analyze(df, out, a)


def analyze(df, out, a):
    import pandas as pd
    from combo_interp.locality import design, mediation, ols, within_cond_spearman
    df = df.copy()
    df['layer_name'] = df.layer
    df['layer'] = df.layer.map({L: i for i, L in enumerate(a.layers)})
    summ = df.groupby(['cond', 'layer_name'])[['coh_excess', 'pr', 'clip_coh', 'purity', 'dom_share', 'dom_eff', 'cos_win']].median().reset_index()
    summ.to_csv(out / 'by_cond.csv', index=False)
    hid = df[df.cond != 'xdir']
    med, sp = [], []
    for m in ['coh_excess', 'pr']:
        for y in ['clip_coh', 'purity']:
            med += mediation(hid, y, 'cond', 'neuron', m)
            sp.append(within_cond_spearman(hid, m, y))
    med, sp = pd.DataFrame(med), pd.DataFrame(sp)
    med.to_csv(out / 'mediation.csv', index=False)
    sp.to_csv(out / 'within_cond_spearman.csv', index=False)
    # robustness: does input-space coherence predict CLIP coherence beyond class purity?
    rob = []
    rng = np.random.default_rng(0)
    for name, extra in {'cond': [], '+purity': ['purity'], '+purity+coh': ['purity', 'coh_excess'], '+coh': ['coh_excess']}.items():
        d = hid.dropna(subset=['clip_coh', 'purity', 'coh_excess']).reset_index(drop=True)
        X, nm = design(d, 'cond', 'neuron', extra)
        est = dict(zip(nm, ols(d.clip_coh.values, X)))
        bs = []
        for _ in range(1000):
            bd = d.iloc[rng.integers(0, len(d), len(d))]
            Xb, nb = design(bd, 'cond', 'neuron', extra)
            bs.append(dict(zip(nb, ols(bd.clip_coh.values, Xb))))
        for t in [t for t in nm if t.startswith('cond=') or t in extra]:
            v = np.array([b.get(t, np.nan) for b in bs])
            rob.append({'spec': name, 'term': t, 'est': est[t], 'lo': np.nanpercentile(v, 5), 'hi': np.nanpercentile(v, 95)})
    rob = pd.DataFrame(rob)
    rob.to_csv(out / 'robustness_clip_coh.csv', index=False)
    val = df[df.published_clip_coh.notna()] if 'published_clip_coh' in df else df.iloc[:0]
    lines = [f'# Vision Experiment B — {a.model}', '']
    if len(val):
        from scipy.stats import pearsonr
        lines += [f'Reproduction: our CLIP top-16 coherence vs published, {len(val)} units: Pearson r = '
                  f'{pearsonr(val.clip_coh, val.published_clip_coh.astype(float)).statistic:.3f}, mean abs diff '
                  f'{np.mean(np.abs(val.clip_coh - val.published_clip_coh.astype(float))):.3f}', '']
    t = summ.pivot_table(index='cond', columns='layer_name', values=['coh_excess', 'pr', 'clip_coh', 'purity'])
    lines += ['## Median per condition', '', '```', t.reindex([c for c in COND_ORDER if c in t.index]).round(3).to_string(), '```', '']
    t = summ.pivot_table(index='cond', columns='layer_name', values=['dom_share', 'dom_eff', 'cos_win'])
    lines += ['```', t.reindex([c for c in COND_ORDER if c in t.index]).dropna(how='all').round(2).to_string(), '```', '']
    pm = med[(med.mediator == 'coh_excess') & med.term.str.startswith('cond=')].copy()
    pm['cond'] = pm.term.str.replace('cond=', '')
    for y in ['clip_coh', 'purity']:
        tt = pm[pm.y == y].set_index('cond')[['effect_total', 'total_lo', 'total_hi', 'effect_direct', 'direct_lo', 'direct_hi', 'attenuation']]
        lines += [f'## Mediation by coh_excess: {y}, effects vs neuron (90% bootstrap CI)', '', '```',
                  tt.reindex([c for c in COND_ORDER if c in tt.index]).round(3).to_string(), '```', '']
    lines += ['## Within-condition Spearman', '', '```', sp.round(3).to_string(index=False), '```', '']
    rt = rob.assign(v=rob.apply(lambda r: f'{r.est:+.3f} [{r.lo:+.2f},{r.hi:+.2f}]', axis=1)).pivot(index='term', columns='spec', values='v')
    lines += ['## Robustness: clip_coh ~ cond + layer + covariates', '', '```', rt.to_string(), '```']
    (out / 'summary.md').write_text('\n'.join(lines))
    plot(df, out, a)
    print((out / 'summary.md').read_text())


def plot(df, out, a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conds = [c for c in COND_ORDER if c in set(df.cond)]
    fig, axes = plt.subplots(1, len(a.layers), figsize=(4.4 * len(a.layers), 3.8), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, L in zip(axes[0], a.layers):
        d = df[df.layer_name == L]
        data = [d[d.cond == c].coh_excess.values for c in conds]
        ax.boxplot(data, vert=False, widths=0.6, showfliers=False, medianprops={'color': '#2a78d6', 'lw': 2},
                   boxprops={'color': '#52514e'}, whiskerprops={'color': '#52514e'}, capprops={'color': '#52514e'})
        for i, v in enumerate(data):
            ax.scatter(v, np.full(len(v), i + 1) + rng.uniform(-0.2, 0.2, len(v)), s=6, color='#2a78d6', alpha=0.35, lw=0)
        ax.axvline(0, color='#a3a29d', lw=1, ls='--')
        ax.set_yticks(range(1, len(conds) + 1), conds, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(L, fontsize=9)
        ax.set_xlabel('coherence of top-100 MLP inputs (0 = random positions)', fontsize=8, color='#52514e')
        ax.tick_params(labelsize=8, colors='#52514e')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.grid(axis='x', color='#e6e5e0', lw=0.6)
    fig.tight_layout()
    fig.savefig(out / 'coh_by_cond.png', dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, type=Path, help='holds val/ and val_labels.npy; outputs go to <root>/mech/<model>/')
    ap.add_argument('--model', default='facebook/convnext-tiny-224')
    ap.add_argument('--layers', default='stage2.block8,stage3.block2')
    ap.add_argument('--stages', default='all')
    ap.add_argument('--n-per', type=int, default=50, help='units per condition and layer (published units always included)')
    ap.add_argument('--n-top', type=int, default=100)
    ap.add_argument('--n-random-sets', type=int, default=200)
    ap.add_argument('--batch', type=int, default=125)
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='debug: only the first N images')
    ap.add_argument('--assign', default='cos,euclid', help='centroids stage: assignment rules (cos, euclid, ip); '
                    'anything but the default writes to centroids_<rules>/')
    ap.add_argument('--resume', action='store_true', help='skip stages already completed (marker files .done_<stage>)')
    ap.add_argument('--export', type=Path, default=None, help='copy the small result files (csv/md/png/json) here at the end')
    a = ap.parse_args()
    a.root = a.root.expanduser()
    a.layers = a.layers.split(',')
    a.name = a.model.split('/')[-1]
    a.repo = Path(__file__).resolve().parents[1]
    M = a.root / 'mech' / a.name
    M.mkdir(parents=True, exist_ok=True)
    stages = STAGES if a.stages == 'all' else a.stages.split(',')
    for s in stages:
        if a.resume and (M / f'.done_{s}').exists():
            print(f'########## {s} already done, skipped', flush=True)
            continue
        t0 = time.time()
        print(f'########## {s} {a.name} {time.strftime("%H:%M:%S")}', flush=True)
        if s == 'capture':
            stage_capture(a, M, 'trained'); stage_capture(a, M, 'random')
        else:
            globals()[f'stage_{s}'](a, M)
        (M / f'.done_{s}').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
        print(f'########## {s} done in {time.time() - t0:.0f}s', flush=True)
    if a.export:
        import shutil
        dst = a.export.expanduser() / a.name
        for f in [M / 'resolve.json', *M.glob('final_*.json'), *M.glob('clip_*.json'),
                  *M.glob('centroids*/*'), *(M / 'locality').glob('*')]:
            if f.is_file():
                rel = f.relative_to(M)
                (dst / rel.parent).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst / rel)
        print(f'[export] results copied to {dst}', flush=True)


if __name__ == '__main__':
    main()
