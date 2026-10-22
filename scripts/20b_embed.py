"""Stage `embed`: train the neighbourhood IWAE (scripts/embed_iwae.py) on 3 x 3 patches of forest land sampled over the whole extent (config embed:
latent_dim, hidden, k_iw, patches, tiles, epochs, batch, lr, seed). 10 % of the patches are held out: the log shows the IWAE bound (nats per patch) on
train and held-out patches per epoch, and the share of held-out feature variance the posterior-mean reconstruction explains, per input group.
embed.enabled = false: writes a marker only (the stage exists in the graph so `sp` can depend on it). Output <OUT>/embed.pt (+ embed_log.json)."""
import os, sys, json, time, numpy as np, torch
from rasterio.windows import Window
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import Featurizer, open_sources, CFG
from stands_cfg import OUT, IN
import embed_iwae as EI
E = CFG['embed']; os.makedirs(OUT, exist_ok=True)
if not E['enabled']:
    json.dump(dict(enabled=False), open(f'{OUT}/embed_log.json', 'w')); print('embed disabled'); sys.exit(0)
torch.manual_seed(E['seed']); rng = np.random.default_rng(E['seed']); torch.set_num_threads(int(os.environ.get('NPROC', 4)))
fit = dict(np.load(f"{IN('fit')}/feature_fit.npz")); fz = Featurizer(fit); src = open_sources(); H, W = src['land'].shape
tiles = [(r, c) for r in range(0, H - 1024, 1024) for c in range(0, W - 1024, 1024)]; rng.shuffle(tiles)
P = []; got = 0; per = E['patches'] // E['tiles']; t0 = time.time()
for r, c in tiles:
    win = Window(c, r, 1024, 1024); land = src['land'].read(1, window=win); fl = (land >= 1) & (land <= 4)
    if fl.mean() < 0.2: continue
    raw = fz.raw(src, win); X, _ = EI.pixel_inputs(raw, fit)
    ok = fl.copy(); ok[[0, -1], :] = False; ok[:, [0, -1]] = False                       # centre pixels with a full 3 x 3 neighbourhood
    ii = rng.choice(np.flatnonzero(ok.ravel()), min(per, int(ok.sum())), replace=False); rr, cc = np.unravel_index(ii, ok.shape)
    P.append(np.stack([X[rr + dy, cc + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 1))   # (n, 9, CIN)
    got += 1
    if got >= E['tiles']: break
P = np.concatenate(P).reshape(-1, 3, 3, EI.CIN).transpose(0, 3, 1, 2).astype(np.float32)      # (N, CIN, 3, 3)
P = torch.from_numpy(np.ascontiguousarray(P)); N = P.shape[0]; perm = torch.randperm(N); nv = N // 10; va, trn = perm[:nv], perm[nv:]
print(f'patches {N} from {got} tiles ({time.time()-t0:.0f}s); availability a25/a22/tm/soil {P[:, EI.NF:, 1, 1].mean(0).numpy().round(3)}', flush=True)
model = EI.IWAE(E['latent_dim'], E['hidden']); opt = torch.optim.Adam(model.parameters(), lr=E['lr']); log = []
def bound(idx):
    model.eval(); tot = 0.0
    with torch.no_grad():
        for b in range(0, len(idx), 4096): tot += float(model.iwae_loss(P[idx[b:b + 4096]], E['k_iw'])[0]) * len(idx[b:b + 4096])
    model.train(); return tot / len(idx)
for ep in range(E['epochs']):
    t1 = time.time(); sh = trn[torch.randperm(len(trn))]
    for b in range(0, len(sh), E['batch']):
        loss, _ = model.iwae_loss(P[sh[b:b + E['batch']]], E['k_iw']); opt.zero_grad(); loss.backward(); opt.step()
    bt, bv = bound(trn[:20000]), bound(va); log.append(dict(epoch=ep + 1, train=bt, val=bv))
    print(f'epoch {ep+1}: -IWAE bound train {bt:.1f} | held-out {bv:.1f} nats / patch ({time.time()-t1:.0f}s)', flush=True)
# reconstruction quality of the posterior mean on held-out patches: share of variance explained per group (centre pixel)
with torch.no_grad():
    mu, _ = model.encode_patches(P[va]); m, _ = model.decode(mu); tgt = P[va][:, :EI.NF, 1, 1]; rec = m[:, 4]; av = P[va][:, EI.NF:, 1, 1]
    r2 = {}; off = 0
    for g, (nm, k) in enumerate(EI.GROUPS):
        sel = av[:, g] > 0; t_, p_ = tgt[sel, off:off + k], rec[sel, off:off + k]
        r2[nm] = round(float(1 - ((t_ - p_) ** 2).sum() / ((t_ - t_.mean(0)) ** 2).sum()), 3); off += k
    zs = mu.numpy(); act = (zs.var(0) > 0.01).sum()
print(f'held-out centre-pixel R² of the posterior-mean reconstruction by group {r2}; active latent dims {act} of {E["latent_dim"]} (var > 0.01); '
      f'latent sd {zs.std(0).round(2)}', flush=True)
torch.save(dict(state=model.state_dict(), D=E['latent_dim'], H=E['hidden'], cfg=E), f'{OUT}/embed.pt')
json.dump(dict(enabled=True, log=log, r2=r2, active=int(act), n_patches=int(N)), open(f'{OUT}/embed_log.json', 'w'), indent=1)
print('saved', f'{OUT}/embed.pt')
