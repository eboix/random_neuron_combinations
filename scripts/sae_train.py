#!/usr/bin/env python
"""Train a TopK SAE on collected activations (sae_collect.py) with the aux-k loss for dead latents.

    python scripts/sae_train.py --acts <root>/sae/convnext-tiny-224/stage3_block2 --expansion 8 --k 32 --steps 10000
Writes <acts>/sae_x<expansion>_k<k>.pt and a training log json.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from sae import TopKSAE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acts', required=True); ap.add_argument('--expansion', type=int, default=8); ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--steps', type=int, default=10000); ap.add_argument('--batch', type=int, default=4096); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--resample-every', type=int, default=500, help='re-initialise dead latents to high-residual inputs every N steps (0 = off), until 75%% of training')
    ap.add_argument('--dead-steps', type=int, default=500); ap.add_argument('--aux-k', type=int, default=512); ap.add_argument('--aux-coef', type=float, default=1 / 32)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--holdout', type=float, default=0.02)
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = 'cuda'
    acts_dir = Path(args.acts); X = np.load(acts_dir / 'train_acts.npy', mmap_mode='r'); N, D = X.shape
    n_hold = int(N * args.holdout); n_train = N - n_hold          # last slice held out (images are in random order)
    f = args.expansion * D
    sae = TopKSAE(D, f, args.k).to(dev)
    # scale so that E||x|| = sqrt(D); b_dec = mean of scaled data
    samp = torch.tensor(np.asarray(X[np.sort(rng.choice(n_train, 65536, replace=False))]), dtype=torch.float32, device=dev)
    scale = samp.norm(dim=1).mean() / D ** 0.5
    sae.scale.fill_(scale.item()); sae.b_dec.data = (samp / scale).mean(0)
    base_var = ((samp / scale - sae.b_dec).pow(2).sum(1)).mean().item()
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=6.25e-10)
    last_fired = torch.zeros(f, dtype=torch.long, device=dev)
    log = []; t0 = time.time()
    print(f'[sae] D={D} F={f} k={args.k} train positions {n_train} holdout {n_hold} scale {scale.item():.4f}', flush=True)
    for step in range(1, args.steps + 1):
        s = int(rng.integers(0, n_train - args.batch))
        x = torch.tensor(np.asarray(X[s:s + args.batch]), dtype=torch.float32, device=dev) / sae.scale
        xc = x - sae.b_dec
        pre = xc @ sae.W_enc + sae.b_enc
        z = sae.topk(pre)
        recon = sae.decode_scaled(z)
        mse = (recon - x).pow(2).sum(1).mean()
        loss = mse / base_var
        fired = (z > 0).any(0); last_fired[fired] = step
        dead = (step - last_fired) > args.dead_steps
        aux = torch.tensor(0.0, device=dev)
        if dead.any() and step > args.dead_steps:
            e = (x - recon).detach()
            pre_dead = torch.where(dead[None, :], pre, torch.full_like(pre, -1e9))
            z_aux = sae.topk(pre_dead, k=min(args.aux_k, int(dead.sum())))
            aux = ((z_aux @ sae.W_dec) - e).pow(2).sum(1).mean() / e.pow(2).sum(1).mean().clamp(min=1e-6)
            loss = loss + args.aux_coef * aux
        opt.zero_grad(set_to_none=True); loss.backward()
        # remove the gradient component parallel to each decoder row, then renormalise rows
        with torch.no_grad():
            g = sae.W_dec.grad; sae.W_dec.grad = g - (g * sae.W_dec).sum(1, keepdim=True) * sae.W_dec
        opt.step(); sae.normalize_decoder()
        if args.resample_every and step % args.resample_every == 0 and step <= 0.75 * args.steps and dead.any():
            with torch.no_grad():   # Anthropic-style resampling: dead latents -> directions of poorly reconstructed inputs
                idx_dead = dead.nonzero().flatten(); nd = len(idx_dead)
                e2 = (x - recon).pow(2).sum(1)
                pick = torch.multinomial(e2 / e2.sum(), nd, replacement=nd > e2.numel())
                v = x[pick] - sae.b_dec; v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-6)
                alive = ~dead
                enc_norm = sae.W_enc.data[:, alive].norm(dim=0).mean().item() if alive.any() else 1.0
                sae.W_dec.data[idx_dead] = v
                sae.W_enc.data[:, idx_dead] = (v * enc_norm * 0.2).t()
                sae.b_enc.data[idx_dead] = 0
                for p_, sl in [(sae.W_enc, (slice(None), idx_dead)), (sae.W_dec, (idx_dead, slice(None))), (sae.b_enc, (idx_dead,))]:
                    st = opt.state.get(p_, {})
                    for key in ('exp_avg', 'exp_avg_sq'):
                        if key in st: st[key][sl] = 0
                last_fired[idx_dead] = step
                print(f'[sae] step {step}: resampled {nd} dead latents', flush=True)
        if step % 250 == 0 or step == args.steps:
            fvu = (mse / base_var).item(); nd = int(dead.sum())
            log.append({'step': step, 'fvu': fvu, 'dead': nd, 'aux': float(aux)})
            print(f'[sae] step {step} fvu {fvu:.4f} dead {nd}/{f} aux {float(aux):.3f} ({time.time()-t0:.0f}s)', flush=True)
    # held-out evaluation
    with torch.no_grad():
        xs = torch.tensor(np.asarray(X[n_train:n_train + min(n_hold, 65536)]), dtype=torch.float32, device=dev)
        recon, z = sae(xs)
        fvu_h = ((recon - xs).pow(2).sum(1).mean() / (xs - xs.mean(0)).pow(2).sum(1).mean()).item()
        l0 = (z > 0).float().sum(1).mean().item()
        ever = (z > 0).any(0).float().mean().item()
    out = acts_dir / f'sae_x{args.expansion}_k{args.k}.pt'
    extra = {'args': vars(args), 'holdout_fvu': fvu_h, 'holdout_l0': l0, 'holdout_alive_frac': ever, 'dead_at_end': int(dead.sum()), 'log': log}
    sae.save(out, extra)
    json.dump(extra, open(acts_dir / f'sae_x{args.expansion}_k{args.k}.json', 'w'), indent=1)
    print(f'[sae] done: holdout FVU {fvu_h:.4f}, L0 {l0:.1f}, dead {int(dead.sum())}/{f} -> {out} ({time.time()-t0:.0f}s)', flush=True)

if __name__ == '__main__':
    main()
