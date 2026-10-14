"""Fit the feature transforms (PCA bases, scales) on a random sample of forest-land pixels -> data/derived/feature_fit.npz"""
import sys, os, numpy as np; sys.path.insert(0, os.path.dirname(__file__))
from rasterio.windows import Window
from features import *
rng = np.random.default_rng(1)
src = open_sources(); H, W = src['land'].shape
fz = Featurizer()
samples = {k: [] for k in ['a25', 'a22', 'soil', 'tma']}
tiles = [(r, c) for r in range(0, H, 1024) for c in range(0, W, 1024)]
rng.shuffle(tiles); got = 0
for r, c in tiles:
    win = Window(c, r, min(1024, W - c), min(1024, H - r))
    land = src['land'].read(1, window=win)
    if (land > 0).mean() < 0.2: continue
    rr = fz.raw(src, win); m = land > 0
    sel = rng.choice(np.flatnonzero(m.ravel()), 4000, replace=False)
    for k in samples: samples[k].append(rr[k].reshape(-1, rr[k].shape[-1])[sel])
    got += 1
    if got >= 60: break
S = {k: np.concatenate(v) for k, v in samples.items()}
def pca(X, k):
    X = X[np.isfinite(X).all(1)]; mu = X.mean(0); U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    ev = s ** 2 / (X.shape[0] - 1); return mu, Vt[:k], ev
k = CFG['segmentation']['aef_pcs']
mu, comp, ev = pca(S['a25'], k)
print('AEF2025 PCA explained', ev[:k].sum() / ev.sum(), 'first 8', ev[:8].sum() / ev.sum())
fit = dict(aef_mean=mu, aef_comp=comp, aef_scale=np.sqrt(ev[:k].sum()), aef_scale8=np.sqrt(ev[:8].sum()))
a25, a22 = S['a25'], S['a22']; ok = np.isfinite(a25).all(1) & np.isfinite(a22).all(1)
cos = (a25[ok] * a22[ok]).sum(1) / np.sqrt((a25[ok] ** 2).sum(1) * (a22[ok] ** 2).sum(1))
print('cos(2022,2025) quantiles', np.quantile(cos, [.01, .05, .25, .5, .75, .95]))
fit['change_sd'] = max(np.std(1 - cos), 1e-3)
T = S['tma']; T = T[np.isfinite(T).all(1)]
fit['tm_mean'] = T.mean(0); fit['tm_sd'] = T.std(0) + 1e-6
Z = S['soil']; Z = Z[np.isfinite(Z).all(1)]
fit['soil_mean'] = Z.mean(0); fit['soil_sd'] = Z.std(0) + 1e-6
smu, scomp, sev = pca((Z - fit['soil_mean']) / fit['soil_sd'], 3)
print('soil PCA explained', sev[:3].sum() / sev.sum())
fit['soil_comp'] = scomp; fit['soil_scale'] = np.sqrt(sev[:3].sum())
np.savez(f'{ROOT}/data/derived/feature_fit.npz', **fit); print('saved; samples', {k: v.shape for k, v in S.items()})
