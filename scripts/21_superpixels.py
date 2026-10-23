"""Stage A: SLIC superpixels per sp_tile (512-px) tile and per land status, inside forest land minus barrier pixels.
Writes <OUT>/sp.tif (int32 global superpixel id, 0 = none) and <OUT>/sp_stats.npz
(id, n, status, feature sums). Superpixels never cross barriers or statuses; with sp_seam_mode phased they cross tile seams (no seam
artefacts), with stitch / no halo they are cut at tile edges. Window (driver) = every tile overlapping (r0, c0, r1, c1). Ids are assigned in tile order
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

HALO = int(SEG['sp_seam_halo'])          # > 0: seam reconciliation on SLIC of the tile + HALO px (mode `sp_seam_mode`)
MODE = str(SEG.get('sp_seam_mode', 'phased'))  # phased: tiles in 4 checkerboard phases; a tile segments only pixels not yet assigned by earlier phases (its
                                               # superpixels grow around theirs) and keeps every superpixel whose centroid is not in a later-phase tile, with
                                               # all its pixels (also across the seam); stitch: tiles keep their cores, cut superpixels re-joined (stitch)
if MODE not in ('phased', 'stitch'): raise ValueError(f"segmentation.sp_seam_mode must be 'phased' or 'stitch', not {MODE!r}")
SIDES = ('E', 'W', 'S', 'N')
def work(job):
    ti, r, c, H, W = job[:5]; fixed, later = (job[5], job[6]) if len(job) > 5 else (None, None)   # phased: already-assigned pixels / later-phase tiles
    th, tw = min(T, H - r), min(T, W - c)
    ra, ca = max(r - HALO, 0), max(c - HALO, 0); rb, cb = min(r + th + HALO, H), min(c + tw + HALO, W)
    cy0, cx0 = r - ra, c - ca                                                   # the tile core inside the (expanded) window
    win = Window(ca, ra, cb - ca, rb - ra)
    land = _src['land'].read(1, window=win)
    fl = (land >= 1) & (land <= 4)
    if fl.sum() < 4: return ti, r, c, None, None
    bar = np.zeros(land.shape, bool)
    for b in _bar: bar |= b.read(1, window=win) == 1          # barrier rasters: 1 = line, 0 = none, 255 = nodata (outside the AOI)
    raw = _fz.raw(_src, win); X = _fz.features(raw); Y = slic_input(X, raw)
    M = Y if SEG['merge_groups'] == SEG['slic_groups'] and SEG['treemap_smooth_px'] == 0 else feature_space(SEG['merge_groups'], X, raw)   # merge feature space
    lab = np.zeros(land.shape, np.int32); nxt = 1
    for st in (1, 2, 3, 4):
        m = fl & ~bar & (land == st)
        if fixed is not None: m &= ~fixed
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
    miss = fl & ~bar & (lab == 0) & (~fixed if fixed is not None else True)
    for st in (1, 2, 3, 4):                                        # per status: a leftover piece never mixes statuses
        ms = miss & (land == st)
        if ms.any(): l, n = ndimage.label(ms); lab = np.where(ms, l + nxt - 1, lab); nxt += n
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
    if HALO > 0 and MODE == 'phased':                           # keep every superpixel whose centroid is not in a later-phase tile (all its pixels)
        k = int(lab.max()); rr, cc = np.nonzero(lab)
        cnt = np.bincount(lab[rr, cc], minlength=k + 1); cy = np.bincount(lab[rr, cc], rr, k + 1) / np.maximum(cnt, 1); cx = np.bincount(lab[rr, cc], cc, k + 1) / np.maximum(cnt, 1)
        iy = np.clip(np.rint(cy).astype(int), 0, lab.shape[0] - 1); ix = np.clip(np.rint(cx).astype(int), 0, lab.shape[1] - 1)
        owned = ~later[iy, ix]; owned[0] = False
        return ti, r, c, np.where(owned[lab], lab, 0), (ra, ca)
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

def stats_work(job):
    """Second pass of the own mode: per-superpixel pixel count, status and feature sums (merge feature space) of one tile core of the final sp.tif."""
    ti, r, c, H, W, sp_path = job
    th, tw = min(T, H - r), min(T, W - c); win = Window(c, r, tw, th)
    with rasterio.open(sp_path) as d: lab = d.read(1, window=win).astype(np.int64)
    if not lab.any(): return None
    land = _src['land'].read(1, window=win)
    raw = _fz.raw(_src, win); X = _fz.features(raw)
    M = feature_space(SEG['merge_groups'], X, raw) if not (SEG['merge_groups'] == SEG['slic_groups'] and SEG['treemap_smooth_px'] == 0) else slic_input(X, raw)
    flat = lab.ravel(); keep = flat > 0; u, inv = np.unique(flat[keep], return_inverse=True)
    n = np.bincount(inv, minlength=u.size); S = np.zeros((u.size, M.shape[-1]), np.float64); np.add.at(S, inv, M.reshape(-1, M.shape[-1])[keep])
    lv = land.ravel()[keep]; st_min = np.full(u.size, 255, np.int64); st_max = np.zeros(u.size, np.int64)
    np.minimum.at(st_min, inv, lv); np.maximum.at(st_max, inv, lv)
    return u, n, S, st_min, st_max

def run_phased(jobs, H, W, prof, sp_path, nproc):
    """Checkerboard phases over the processed tiles (phase = (tile row parity, tile column parity)). In phase k every tile segments its expanded window
    (tile + HALO) without the pixels assigned in phases < k and keeps the superpixels whose centroid is not in a tile of a later phase, with all their
    pixels inside processed tiles (same-phase tiles are T apart, so their windows never overlap). Forest pixels left unassigned become their own
    4-connected pieces per status. Every label is split into 4-connected pieces and renumbered in raster order; statistics from stats_work."""
    from skimage.measure import label as cc_label
    from scipy import ndimage
    rcs = [(j[1], j[2]) for j in jobs]
    R0 = min(r for r, c in rcs); C0 = min(c for r, c in rcs); R1 = max(min(r + T, H) for r, c in rcs); C1 = max(min(c + T, W) for r, c in rcs)
    hh, ww = R1 - R0, C1 - C0
    ph = lambda r, c: ((r // T) % 2) * 2 + (c // T) % 2
    PH = np.full((hh, ww), -1, np.int8)                                     # phase of the tile at each pixel (-1 = not processed)
    for r, c in rcs: PH[r - R0:min(r + T, H) - R0, c - C0:min(c + T, W) - C0] = ph(r, c)
    G = np.zeros((hh, ww), np.int64); off = 0
    def crop(A, ra, rb, ca, cb, fill):                                      # global window [ra:rb, ca:cb] of an extent array (outside the extent: fill)
        out = np.full((rb - ra, cb - ca), fill, A.dtype); ya, yb, xa, xb = max(ra - R0, 0), min(rb - R0, hh), max(ca - C0, 0), min(cb - C0, ww)
        if yb > ya and xb > xa: out[ya - (ra - R0):yb - (ra - R0), xa - (ca - C0):xb - (ca - C0)] = A[ya:yb, xa:xb]
        return out
    with Pool(nproc, initializer=init) as p:
        for k in range(4):
            jk = []
            for j in jobs:
                ti, r, c = j[0], j[1], j[2]
                if ph(r, c) != k: continue
                th, tw = min(T, H - r), min(T, W - c); ra, ca = max(r - HALO, 0), max(c - HALO, 0); rb, cb = min(r + th + HALO, H), min(c + tw + HALO, W)
                P_ = crop(PH, ra, rb, ca, cb, -1)
                fixed = (crop(G, ra, rb, ca, cb, 0) > 0) | (P_ < 0)               # assigned earlier, or outside the processed tiles
                jk.append(j[:5] + (fixed, P_ > k))
            for ti, r, c, lab, info in p.imap(work, jk):
                if lab is None: continue
                ra, ca = info; ys, xs = np.nonzero(lab); gy, gx = ys + ra - R0, xs + ca - C0
                ok = (gy >= 0) & (gy < hh) & (gx >= 0) & (gx < ww); ys, xs, gy, gx = ys[ok], xs[ok], gy[ok], gx[ok]
                ok = (PH[gy, gx] >= 0) & (G[gy, gx] == 0); ys, xs, gy, gx = ys[ok], xs[ok], gy[ok], gx[ok]
                G[gy, gx] = lab[ys, xs] + off; off += int(lab.max())
            print(f'phase {k + 1}/4: {len(jk)} tiles, superpixels so far {off}', flush=True)
    win = Window(C0, R0, ww, hh)                                             # unassigned forest pixels: own pieces per status
    land = _src_land.read(1, window=win); bar = np.zeros((hh, ww), bool)
    for b in BARRIERS:
        with rasterio.open(b) as d: bar |= d.read(1, window=win) == 1
    gap = (G == 0) & (PH >= 0) & (land >= 1) & (land <= 4) & ~bar; ngap = int(gap.sum())
    for st in (1, 2, 3, 4):
        l, n_ = ndimage.label(gap & (land == st)); G = np.where(l > 0, l + off, G); off += n_
    F = cc_label(G, background=0, connectivity=1).astype(np.int32)
    with rasterio.open(sp_path, 'w', **prof) as d: d.write(F, 1, window=win)
    K = int(F.max()); n = np.zeros(K, np.int64); S = None; smin = np.full(K, 255, np.int64); smax = np.zeros(K, np.int64)
    with Pool(nproc, initializer=init) as p:
        for res in p.imap(stats_work, [j[:5] + (sp_path,) for j in jobs]):
            if res is None: continue
            u, n_, S_, a_, b_ = res
            if S is None: S = np.zeros((K, S_.shape[1]), np.float64)
            n[u - 1] += n_; S[u - 1] += S_; smin[u - 1] = np.minimum(smin[u - 1], a_); smax[u - 1] = np.maximum(smax[u - 1], b_)
    assert (n > 0).all() and (smin == smax).all(), 'phased superpixels must be non-empty and of one land status'
    print(f'phased seams: {K} superpixels (halo {HALO} px; {ngap} unassigned forest px filled as own pieces)', flush=True)
    return np.arange(1, K + 1), n, smin.astype(np.uint8), S.astype(np.float32)

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
    if HALO > 0 and MODE == 'phased':
        _src_land = rasterio.open(f"{IN('clean')}/land_clean.tif")
        ID, NS, ST_, SS = run_phased(jobs, H, W, prof, f'{OUT}/sp.tif', int(os.environ.get('NPROC', 6)))
        np.savez(f'{OUT}/sp_stats.npz', id=ID, n=NS, status=ST_, S=SS)
        print('done', len(ID), time.time() - t0); sys.exit(0)
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
