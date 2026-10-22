"""Stage A: SLIC superpixels per 2048-px tile and per land status, inside forest land minus barrier pixels.
Writes <OUT>/sp.tif (int32 global superpixel id, 0 = none) and <OUT>/sp_stats.npz
(id, n, status, feature sums). Superpixels never cross barriers, statuses or tile edges; the global RAG merge
(stage C) joins them across tile edges. Window (driver) = the tiles whose origin lies inside (r0, c0, r1, c1). Ids are assigned in tile order
(deterministic: the merge's tie-breaking depends on ids)."""
import sys, os, time, numpy as np, rasterio
from rasterio.windows import Window
from multiprocessing import Pool
from skimage.segmentation import slic
sys.path.insert(0, os.path.dirname(__file__))
from features import *
from common import PROFILE
from stands_cfg import OUT, IN, WINDOW
import embed_iwae as EI
_emb = None
T = int(CFG['segmentation']['sp_tile'])     # superpixel tile size (superpixels never cross tile edges, the RAG merge joins them)
SEG = CFG['segmentation']; SP_PX = max(4, int(round(SEG.get('superpixel_ha', 1.0) * 1e4 / 900)))
from stands_cfg import barrier_files
BARRIERS = barrier_files(CFG)                    # configured barriers; a missing raster is an error, not a silent skip
_fz = None
from scipy.ndimage import uniform_filter
def feature_space(groups, X, raw, smooth=0):
    """A feature space from column groups: the scaled stack of features() (aef, aef_old, aef_change, treemap, soil; TreeMap optionally smoothed by a
    masked n x n mean), the raw AlphaEarth embeddings (aef25_raw, aef22_raw: unit-norm, nodata -> 0) or the IWAE latent (latent: posterior mean)."""
    dims = [('aef', _fit['aef_comp'].shape[0]), ('aef_old', 8), ('aef_change', 1), ('treemap', len(TM_ATTRS)), ('soil', _fit['soil_comp'].shape[0])]
    assert sum(k for _, k in dims) == X.shape[-1], 'feature block layout changed'
    parts, off = {}, 0
    for g, k in dims: parts[g] = X[..., off:off + k]; off += k
    if smooth > 0 and 'treemap' in groups:
        v = np.isfinite(raw['tma']).all(-1).astype(np.float32); den = uniform_filter(v, smooth, mode='constant')
        parts['treemap'] = np.stack([uniform_filter(parts['treemap'][..., j] * v, smooth, mode='constant') / np.maximum(den, 1e-6) for j in range(parts['treemap'].shape[-1])], -1) * v[..., None]
    if 'aef25_raw' in groups: parts['aef25_raw'] = np.nan_to_num(raw['a25'])
    if 'aef22_raw' in groups: parts['aef22_raw'] = np.nan_to_num(raw['a22'])
    if 'latent' in groups:
        if _emb is None: raise RuntimeError('latent requested but the embed stage is disabled (embed.enabled: false)')
        parts['latent'] = EI.encode_tile(_emb, raw, _fit)
    bad = set(groups) - set(parts)
    if bad: raise KeyError(f'unknown feature groups {bad}')
    return np.concatenate([parts[g] for g in groups], -1).astype(np.float32)
def slic_input(X, raw): return feature_space(SEG['slic_groups'], X, raw, SEG['treemap_smooth_px'])

def init():
    global _fz, _src, _fit, _bar
    global _emb
    _fit = dict(np.load(f"{IN('fit')}/feature_fit.npz"))
    _emb = None
    if 'latent' in SEG['slic_groups'] + SEG['merge_groups']:
        import torch; torch.set_num_threads(1); _emb = EI.load(f"{IN('embed')}/embed.pt")[0]
    _fz = Featurizer(_fit); _src = open_sources()
    _bar = [rasterio.open(b) for b in BARRIERS]

def work(job):
    ti, r, c, H, W = job
    win = Window(c, r, min(T, W - c), min(T, H - r))
    land = _src['land'].read(1, window=win)
    fl = (land >= 1) & (land <= 4)
    if fl.sum() < 4: return ti, r, c, None, None
    bar = np.zeros(land.shape, bool)
    for b in _bar: bar |= b.read(1, window=win) > 0
    raw = _fz.raw(_src, win); X = _fz.features(raw); Y = slic_input(X, raw)
    M = Y if SEG['merge_groups'] == SEG['slic_groups'] and SEG['treemap_smooth_px'] == 0 else feature_space(SEG['merge_groups'], X, raw)   # merge feature space
    lab = np.zeros(land.shape, np.int32); nxt = 1
    for st in (1, 2, 3, 4):
        m = fl & ~bar & (land == st)
        npx = int(m.sum())
        if npx == 0: continue
        if npx < SP_PX * 2:
            from scipy import ndimage
            l, n = ndimage.label(m); l[l > 0] += nxt - 1; lab = np.where(m, l, lab); nxt += n; continue
        comp = SEG['slic_compactness']
        if SEG['slic_scale'] == 'fixed': comp = comp * float(Y[m].max() - Y[m].min()) / SEG['slic_range']   # undo scikit-image's per-tile min-max
        s = slic(Y, n_segments=max(1, npx // SP_PX), compactness=comp, mask=m,
                 channel_axis=-1, convert2lab=False, enforce_connectivity=True, start_label=1, max_num_iter=10)
        s = s.astype(np.int32); s[s > 0] += nxt - 1
        lab = np.where(m, s, lab); nxt = int(lab.max()) + 1
        # SLIC with enforce_connectivity can leave mask pixels at 0 when tiny; label them as their own pieces
    from scipy import ndimage
    miss = fl & ~bar & (lab == 0)
    if miss.any():
        l, n = ndimage.label(miss); lab = np.where(miss, l + nxt - 1, lab); nxt += n
    # split any non-4-connected superpixel (SLIC connectivity is 8-ish) into 4-connected parts
    ids = np.unique(lab[lab > 0])
    out = np.zeros_like(lab); k = 0
    objs = ndimage.find_objects(lab)
    for i, sl in enumerate(objs, start=1):
        if sl is None: continue
        sub = lab[sl] == i
        l, n = ndimage.label(sub)
        o = out[sl]; o[sub] = l[sub] + k; k += n
    lab = out
    n = np.bincount(lab.ravel(), minlength=k + 1)[1:]
    flat = lab.ravel(); keep = flat > 0
    S = np.zeros((k, M.shape[-1]), np.float64)                 # feature sums in the MERGE feature space (merge_groups)
    np.add.at(S, flat[keep] - 1, M.reshape(-1, M.shape[-1])[keep])
    stt = np.zeros(k, np.uint8); stt[flat[keep] - 1] = land.ravel()[keep]
    return ti, r, c, lab, (n, stt, S)

if __name__ == '__main__':
    t0 = time.time()
    with rasterio.open(f"{IN('clean')}/land_clean.tif") as s: prof = s.profile; H, W = s.shape
    jobs = [(i, r, c, H, W) for i, (r, c) in enumerate((r, c) for r in range(0, H, T) for c in range(0, W, T))]
    win = WINDOW or (tuple(map(int, sys.argv[1:5])) if len(sys.argv) > 4 else None)     # pilot window r0 c0 r1 c1 (pixels)
    if win:
        r0, c0, r1, c1 = win; jobs = [j for j in jobs if r0 - T < j[1] < r1 and c0 - T < j[2] < c1]   # every tile overlapping the window
    prof = dict(prof, **PROFILE, dtype='int32', nodata=0)
    ids, ns, sts, Ss = [], [], [], []; off = 0
    os.makedirs(OUT, exist_ok=True)
    with rasterio.open(f'{OUT}/sp.tif', 'w', **prof) as d, Pool(int(os.environ.get('NPROC', 6)), initializer=init) as p:
        for j, (ti, r, c, lab, st) in enumerate(p.imap(work, jobs)):
            if lab is None: continue
            n, stt, S = st
            g = np.where(lab > 0, lab + off, 0).astype(np.int32)
            d.write(g, 1, window=Window(c, r, lab.shape[1], lab.shape[0]))
            ids.append(np.arange(off + 1, off + 1 + n.size)); ns.append(n); sts.append(stt); Ss.append(S.astype(np.float32))
            off += n.size
            print(f'tile {j+1}/{len(jobs)} sp {n.size} total {off} {time.time()-t0:.0f}s', flush=True)
    np.savez(f'{OUT}/sp_stats.npz', id=np.concatenate(ids), n=np.concatenate(ns),
             status=np.concatenate(sts), S=np.concatenate(Ss))
    print('done', off, time.time() - t0)
