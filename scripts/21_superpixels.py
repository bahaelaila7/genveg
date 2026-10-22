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

HALO = int(SEG['sp_seam_halo'])          # > 0: seam reconciliation (SLIC on the tile + HALO px; superpixels cut by a tile seam are re-joined, see stitch)
SIDES = ('E', 'W', 'S', 'N')
def work(job):
    ti, r, c, H, W = job
    th, tw = min(T, H - r), min(T, W - c)
    ra, ca = max(r - HALO, 0), max(c - HALO, 0); rb, cb = min(r + th + HALO, H), min(c + tw + HALO, W)
    cy0, cx0 = r - ra, c - ca                                                   # the tile core inside the (expanded) window
    win = Window(ca, ra, cb - ca, rb - ra)
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
    strips = None
    if HALO > 0:                                                # keep the core; describe the halo for the seam stitching
        E = lab; core = E[cy0:cy0 + th, cx0:cx0 + tw]
        lab = np.zeros_like(core); k = 0                        # core pieces (a superpixel cut by the core edge may split)
        for i, sl in enumerate(ndimage.find_objects(core), start=1):
            if sl is None: continue
            sub = core[sl] == i; l, n_ = ndimage.label(sub); o = lab[sl]; o[sub] = l[sub] + k; k += n_
        land = land[cy0:cy0 + th, cx0:cx0 + tw]; M = M[cy0:cy0 + th, cx0:cx0 + tw]
        strips = {}
        for sd in SIDES:                                        # halo strip (outside the core, along the side) and its seam band (core edge)
            if sd == 'E': hal, band = E[cy0:cy0 + th, cx0 + tw:], (core[:, -1], lab[:, -1]); edge = lab[:, -HALO:]
            elif sd == 'W': hal, band = E[cy0:cy0 + th, :cx0], (core[:, 0], lab[:, 0]); edge = lab[:, :HALO]
            elif sd == 'S': hal, band = E[cy0 + th:, cx0:cx0 + tw], (core[-1, :], lab[-1, :]); edge = lab[-HALO:, :]
            else: hal, band = E[:cy0, cx0:cx0 + tw], (core[0, :], lab[0, :]); edge = lab[:HALO, :]
            emap = {}                                           # expanded-window superpixel -> its core piece on the seam band (mode)
            for e_, p_ in zip(band[0].tolist(), band[1].tolist()):
                if e_ > 0 and p_ > 0: emap.setdefault(e_, {}).setdefault(p_, 0); emap[e_][p_] += 1
            lut = np.zeros(int(E.max()) + 1, np.int64)
            for e_, d_ in emap.items(): lut[e_] = max(d_, key=d_.get)
            strips[sd] = (lut[hal] if hal.size else hal.astype(np.int64), edge.copy())
    n = np.bincount(lab.ravel(), minlength=k + 1)[1:]
    flat = lab.ravel(); keep = flat > 0
    S = np.zeros((k, M.shape[-1]), np.float64)                 # feature sums in the MERGE feature space (merge_groups)
    np.add.at(S, flat[keep] - 1, M.reshape(-1, M.shape[-1])[keep])
    stt = np.zeros(k, np.uint8); stt[flat[keep] - 1] = land.ravel()[keep]
    return ti, r, c, lab, (n, stt, S, strips)

def stitch(TS, N, n, st, S, sp_path):
    """Seam reconciliation: superpixel a of tile A and b of the neighbouring tile B are one superpixel cut by the seam when A's SLIC run extends a
    into B's core mostly onto b (>= 50 % of a's halo part) AND B's run extends b into A's core mostly onto a (>= 50 %), with the same status.
    Union-find over the pairs; sp.tif and the statistics are rewritten with the joined ids (compacted, tile order kept)."""
    par = np.arange(N + 1)
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    def frac_pairs(ext, core):                                  # (a, b) -> share of a's halo pixels that fall on b's core pixels
        m = ext > 0
        if not m.any(): return {}
        a_, b_ = ext[m], core[m]; tot = np.bincount(a_, minlength=N + 1)
        u, cnt = np.unique(np.stack([a_, b_], 1), axis=0, return_counts=True)
        return {(int(x), int(y)): c_ / tot[x] for (x, y), c_ in zip(u.tolist(), cnt.tolist()) if y > 0}
    joins = 0
    for (r, c), A in TS.items():
        for sd, nb, op in (('E', (r, c + T), 'W'), ('S', (r + T, c), 'N')):
            B = TS.get(nb)
            if B is None: continue
            a_ext, a_edge = A[sd]; b_ext, b_edge = B[op]
            if sd == 'E': w_ = min(a_ext.shape[1], b_edge.shape[1], b_ext.shape[1], a_edge.shape[1]); f1 = frac_pairs(a_ext[:, :w_], b_edge[:, :w_]); f2 = frac_pairs(b_ext[:, -w_:], a_edge[:, -w_:])
            else: w_ = min(a_ext.shape[0], b_edge.shape[0], b_ext.shape[0], a_edge.shape[0]); f1 = frac_pairs(a_ext[:w_], b_edge[:w_]); f2 = frac_pairs(b_ext[-w_:], a_edge[-w_:])
            for (a, b), q in f1.items():
                if q >= 0.5 and f2.get((b, a), 0) >= 0.5 and st[a - 1] == st[b - 1]:
                    ra_, rb_ = find(a), find(b)
                    if ra_ != rb_: par[max(ra_, rb_)] = min(ra_, rb_); joins += 1
    root = np.array([find(x) for x in range(N + 1)]); u, newid = np.unique(root[1:], return_inverse=True)
    lut = np.zeros(N + 1, np.int64); lut[1:] = newid + 1; K = u.size
    nn = np.bincount(newid, n, K); SS = np.zeros((K, S.shape[1]), np.float64); np.add.at(SS, newid, S.astype(np.float64)); SS = SS.astype(np.float32)
    stt = np.zeros(K, st.dtype); stt[newid] = st
    with rasterio.open(sp_path, 'r+') as d:
        for _, w in d.block_windows(1):
            a = d.read(1, window=w)
            if a.any(): d.write(lut[a].astype(np.int32), 1, window=w)
    print(f'seam stitching: {joins} cross-seam joins; superpixels {N} -> {K}', flush=True)
    return np.arange(1, K + 1), nn, stt, SS

if __name__ == '__main__':
    t0 = time.time()
    with rasterio.open(f"{IN('clean')}/land_clean.tif") as s: prof = s.profile; H, W = s.shape
    jobs = [(i, r, c, H, W) for i, (r, c) in enumerate((r, c) for r in range(0, H, T) for c in range(0, W, T))]
    win = WINDOW or (tuple(map(int, sys.argv[1:5])) if len(sys.argv) > 4 else None)     # pilot window r0 c0 r1 c1 (pixels)
    if win:
        r0, c0, r1, c1 = win; jobs = [j for j in jobs if r0 - T < j[1] < r1 and c0 - T < j[2] < c1]   # every tile overlapping the window
    prof = dict(prof, **PROFILE, dtype='int32', nodata=0)
    ids, ns, sts, Ss = [], [], [], []; off = 0; TS = {}
    os.makedirs(OUT, exist_ok=True)
    with rasterio.open(f'{OUT}/sp.tif', 'w', **prof) as d, Pool(int(os.environ.get('NPROC', 6)), initializer=init) as p:
        for j, (ti, r, c, lab, st) in enumerate(p.imap(work, jobs)):
            if lab is None: continue
            n, stt, S, strips = st
            if strips is not None:
                TS[(r, c)] = {sd: (np.where(a > 0, a + off, 0), np.where(b > 0, b + off, 0)) for sd, (a, b) in strips.items()}
            g = np.where(lab > 0, lab + off, 0).astype(np.int32)
            d.write(g, 1, window=Window(c, r, lab.shape[1], lab.shape[0]))
            ids.append(np.arange(off + 1, off + 1 + n.size)); ns.append(n); sts.append(stt); Ss.append(S.astype(np.float32))
            off += n.size
            print(f'tile {j+1}/{len(jobs)} sp {n.size} total {off} {time.time()-t0:.0f}s', flush=True)
    ID, NS, ST_, SS = np.concatenate(ids), np.concatenate(ns), np.concatenate(sts), np.concatenate(Ss)
    if HALO > 0 and off > 0:
        ID, NS, ST_, SS = stitch(TS, off, NS, ST_, SS, f'{OUT}/sp.tif')
    np.savez(f'{OUT}/sp_stats.npz', id=ID, n=NS, status=ST_, S=SS)
    print('done', len(ID), time.time() - t0)
