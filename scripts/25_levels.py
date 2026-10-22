"""Stage `levels` (hierarchical segmentation C, user 2026-10-22): nested stand levels on the superpixel extent.
  fine units    the Ward merge of rag_merge on the superpixel graph RESTRICTED to edges inside one stand (a fine unit never crosses a stand boundary),
                threshold bisected to levels.fine_median_ha (area-weighted median), MMU absorption below levels.fine_mmu_ha (inside the stand);
                barrier / sliver pixels of a stand join the fine unit of an adjacent pixel of the same stand (grown)
  stands        the finalize stage's stands (unchanged)
  compartments  the Ward merge of whole stands on the stand adjacency graph (features = area-weighted stand means of the merge feature space),
                threshold bisected to levels.coarse_median_ha; statuses may mix unless levels.coarse_status_hard; barrier lines are not crossed unless
                levels.coarse_across_barriers (adjacency through barrier pixels is ignored); size cap levels.coarse_max_ha
Nesting is checked (every fine unit inside one stand, every stand inside one compartment). Held-out share per level (as stands_metrics.py, interior units,
equal-count blocks) in levels.json. Outputs <OUT>/fine.tif, compartments.tif (int32), levels.gpkg (layers fine, compartments), levels.json."""
import os, sys, json, time, numpy as np, pandas as pd, rasterio, geopandas as gpd
from rasterio.windows import Window
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stands_cfg import CFG, IN, OUT, WINDOW, ROOT, sp_extent, mmu_px, barrier_files
from common import PROFILE
from rag_merge import merge
t0 = time.time(); PXHA = 0.09; LV = CFG['levels']; SEG = CFG['segmentation']; os.makedirs(OUT, exist_ok=True)
r0, c0, r1, c1 = sp_extent(CFG, WINDOW); win = Window(c0, r0, c1 - c0, r1 - r0)
ST = rasterio.open(f"{IN('finalize')}/stands.tif"); A = ST.read(1, window=win).astype(np.int64)
SP = rasterio.open(f"{IN('sp')}/sp.tif").read(1, window=win).astype(np.int64)
L = rasterio.open(f"{IN('clean')}/land_clean.tif").read(1, window=win)
stt = np.load(f"{IN('sp')}/sp_stats.npz"); spn = stt['n'].astype(np.float64); spS = stt['S'].astype(np.float64); spst = stt['status'].astype(np.int64)
bar = np.zeros(A.shape, bool)
for b in barrier_files(CFG):
    with rasterio.open(b) as bs: bar |= bs.read(1, window=win) > 0
def wmedian_ha(sz): s = np.sort(sz) * PXHA; return s[np.searchsorted(np.cumsum(s), s.sum() / 2)]
def bisect(fn, target, lo=1e-3, hi=1e5, it=16):
    """threshold whose area-weighted median unit size is the target (fn(thr) -> unit sizes in px)."""
    for _ in range(it):
        mid = np.sqrt(lo * hi); med = wmedian_ha(fn(mid))
        if med < target: lo = mid
        else: hi = mid
    return np.sqrt(lo * hi)
def pairs4(X, keep=None):
    out = []
    for a, b, ka, kb in ((X[:, :-1], X[:, 1:], None if keep is None else keep[:, :-1], None if keep is None else keep[:, 1:]),
                         (X[:-1], X[1:], None if keep is None else keep[:-1], None if keep is None else keep[1:])):
        m = (a != b) & (a > 0) & (b > 0)
        if keep is not None: m &= ka & kb
        out.append(np.stack([a[m], b[m]], 1))
    P = np.concatenate(out); P = np.sort(P, 1); u, c = np.unique(P, axis=0, return_counts=True); return u[:, 0], u[:, 1], c.astype(float)
# ---------------- superpixel -> stand (V4 of verify: a superpixel lies in exactly one stand)
m = SP > 0; sp_ids = np.unique(SP[m]); ps = np.unique(np.stack([SP[m], A[m]], 1), axis=0)
assert len(ps) == len(sp_ids), 'a superpixel spans several stands'
sp2st = dict(zip(ps[:, 0].tolist(), ps[:, 1].tolist())); loc = {s: i for i, s in enumerate(sp_ids)}
n_ = spn[sp_ids - 1]; S_ = spS[sp_ids - 1]; st_ = spst[sp_ids - 1]; stand_of = np.array([sp2st[s] for s in sp_ids])
li = np.zeros(SP.max() + 1, np.int64); li[sp_ids] = np.arange(sp_ids.size)
# perimeters in the extent: 4n - 2 * internal pixel pairs
inner = np.zeros(sp_ids.size);
for a, b in ((SP[:, :-1], SP[:, 1:]), (SP[:-1], SP[1:])):
    k = (a == b) & (a > 0); np.add.at(inner, li[a[k]], 1)
per = 4 * n_ - 2 * inner
ea, eb, el = pairs4(SP); ea, eb = li[ea], li[eb]; same = stand_of[ea] == stand_of[eb]; fa, fb, fl_ = ea[same], eb[same], el[same]
FMMU = int(round(LV['fine_mmu_ha'] / PXHA))
def fine_run(thr):
    return merge(n_, S_, st_, fa, fb, fl_, per, thr, 1e12, shape_w=SEG['shape_weight'], min_n=FMMU, status_hard=True)
thr_f = bisect(lambda t: np.bincount(fine_run(t), n_), LV['fine_median_ha']); lab_f = fine_run(thr_f)
F = np.zeros(A.shape, np.int64); F[m] = lab_f[li[SP[m]]] + 1
# barrier / sliver pixels of a stand: grow the fine unit of an adjacent pixel of the same stand
hole = (A > 0) & (F == 0)
for _ in range(A.shape[0] + A.shape[1]):
    if not hole.any(): break
    found = np.zeros(A.shape, bool); cand = np.zeros_like(F)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):                  # neighbour (y+dy, x+dx), no wrap-around
        nf = np.zeros_like(F); na = np.zeros_like(A)
        ys, yd = (slice(dy, None), slice(None, -dy)) if dy > 0 else ((slice(None, dy), slice(-dy, None)) if dy < 0 else (slice(None), slice(None)))
        xs, xd = (slice(dx, None), slice(None, -dx)) if dx > 0 else ((slice(None, dx), slice(-dx, None)) if dx < 0 else (slice(None), slice(None)))
        nf[yd, xd] = F[ys, xs]; na[yd, xd] = A[ys, xs]
        ok = hole & ~found & (nf > 0) & (na == A); cand[ok] = nf[ok]; found |= ok
    if not found.any(): break
    F[found] = cand[found]; hole &= ~found
nh = int(hole.sum())
if nh:                                                                  # stand pixels with no superpixel in their stand: the stand itself is one fine unit
    F[hole] = F.max() + 1 + np.unique(A[hole], return_inverse=True)[1]
print(f'fine units: threshold {thr_f:.4g}, {np.unique(F[F > 0]).size} units, area-weighted median {wmedian_ha(np.bincount(F[F > 0])[1:][np.bincount(F[F > 0])[1:] > 0]):.1f} ha '
      f'({time.time()-t0:.0f}s)', flush=True)
# ---------------- compartments: Ward on the stand graph
sid = np.unique(A[A > 0]); si = np.zeros(A.max() + 1, np.int64); si[sid] = np.arange(sid.size)
sn = np.bincount(si[A[A > 0]], minlength=sid.size).astype(float)
sS = np.zeros((sid.size, S_.shape[1])); np.add.at(sS, si[stand_of], S_)                 # superpixel feature sums per stand
sp_n_in = np.bincount(si[stand_of], n_, sid.size); sS = sS / np.maximum(sp_n_in, 1)[:, None] * sn[:, None]   # stand mean x stand area
sst = np.zeros(sid.size, np.int64); cnt = np.zeros((sid.size, 5)); np.add.at(cnt, (si[A[A > 0]], np.clip(L[A > 0], 0, 4)), 1); cnt[:, 0] = 0; sst = cnt.argmax(1)
keep = None if LV['coarse_across_barriers'] else ~bar
ca, cb, cl = pairs4(A, keep); ca, cb = si[ca], si[cb]
sinner = np.zeros(sid.size)
for a, b in ((A[:, :-1], A[:, 1:]), (A[:-1], A[1:])):
    k = (a == b) & (a > 0); np.add.at(sinner, si[a[k]], 1)
sper = 4 * sn - 2 * sinner
cst = sst if LV['coarse_status_hard'] else np.ones_like(sst)               # statuses may mix: one common status for the merge
def coarse_run(thr):
    return merge(sn, sS, cst, ca, cb, cl, sper, thr, LV['coarse_max_ha'] / PXHA, shape_w=SEG['shape_weight'], min_n=0, status_hard=True)
thr_c = bisect(lambda t: np.bincount(coarse_run(t), sn), LV['coarse_median_ha']); lab_c = coarse_run(thr_c)
C = np.zeros(A.shape, np.int64); C[A > 0] = lab_c[si[A[A > 0]]] + 1
print(f'compartments: threshold {thr_c:.4g}, {lab_c.max() + 1} compartments, area-weighted median {wmedian_ha(np.bincount(lab_c, sn)):.1f} ha ({time.time()-t0:.0f}s)', flush=True)
# ---------------- nesting
fs = np.unique(np.stack([F[A > 0], A[A > 0]], 1), axis=0); sc = np.unique(np.stack([A[A > 0], C[A > 0]], 1), axis=0)
nest_f = len(fs) == np.unique(fs[:, 0]).size; nest_c = len(sc) == np.unique(sc[:, 0]).size
cover = bool(((F > 0) == (A > 0)).all() and ((C > 0) == (A > 0)).all())
print(f'nesting: every fine unit in one stand {nest_f}; every stand in one compartment {nest_c}; same coverage {cover}; fine units without a superpixel {nh} px', flush=True)
assert nest_f and nest_c and cover, 'levels not nested'
# ---------------- held-out share per level (interior units, equal-count blocks)
rr, cc = np.mgrid[0:A.shape[0], 0:A.shape[1]]
def share_of(U):
    edge = np.unique(np.concatenate([U[0], U[-1], U[:, 0], U[:, -1]])); fl = (L >= 1) & (L <= 4) & (U > 0) & ~np.isin(U, edge); K = np.unique(U[fl]).size
    blocks = lambda s: ((rr // s) * (A.shape[1] // s + 1) + cc // s) * 8 + L.astype(np.int64)
    side = min(range(2, 300), key=lambda s: abs(np.unique(blocks(s)[fl]).size - K)); B = blocks(side)
    def r2(lab, x, mk):
        _, inv = np.unique(lab[mk], return_inverse=True); v = x[mk]; n = np.bincount(inv); s = np.bincount(inv, v); s2 = np.bincount(inv, v * v)
        return 1 - (s2 - s * s / n).sum() / ((v - v.mean()) ** 2).sum()
    out = {}
    for rel in CFG['metrics']['heldout']:
        p = f'{ROOT}/data/{rel}'
        if not os.path.exists(p): continue
        with rasterio.open(p) as s: x = s.read(1, window=win).astype(np.float64); nd = s.nodata
        ok = fl & np.isfinite(x) & ((x != nd) if nd is not None and not np.isnan(nd) else True)
        rs, rb = r2(U, x, ok), r2(B, x, ok)
        if 1 - rb >= CFG['metrics']['headroom_min']: out[os.path.basename(rel).split('.')[0]] = round(float((rs - rb) / (1 - rb)), 4)
    return dict(units=int(K), mean_share=round(float(np.mean(list(out.values()))), 4), share=out)
res = dict(fine=dict(threshold=float(thr_f), **share_of(F)), stands=share_of(A), compartments=dict(threshold=float(thr_c), **share_of(C)),
           fine_wmedian_ha=float(wmedian_ha(np.bincount(F[F > 0])[1:][np.bincount(F[F > 0])[1:] > 0])), compartment_wmedian_ha=float(wmedian_ha(np.bincount(lab_c, sn))),
           nested=True)
json.dump(res, open(f'{OUT}/levels.json', 'w'), indent=1)
print('held-out mean share per level: fine {} | stands {} | compartments {}'.format(res['fine']['mean_share'], res['stands']['mean_share'], res['compartments']['mean_share']), flush=True)
# ---------------- rasters + polygons
prof = dict(ST.profile, **PROFILE, dtype='int32', nodata=0)
for nm, X in (('fine', F), ('compartments', C)):
    with rasterio.open(f'{OUT}/{nm}.tif', 'w', **prof) as d: d.write(X.astype(np.int32), 1, window=win)
tr = rasterio.windows.transform(win, ST.transform)
for nm, X, parent in (('fine', F, A), ('compartments', C, None)):
    geoms = {}
    for gj, v in shapes(X.astype(np.int32), mask=X > 0, connectivity=4, transform=tr): geoms.setdefault(int(v), []).append(shape(gj))
    ids = sorted(geoms); area = np.bincount(X[X > 0]); stat = {}
    cntL = np.zeros((X.max() + 1, 5)); np.add.at(cntL, (X[X > 0], np.clip(L[X > 0], 0, 4)), 1); cntL[:, 0] = 0
    df = pd.DataFrame({'unit_id': ids, 'area_ha': [area[i] * PXHA for i in ids], 'status': [int(cntL[i].argmax()) for i in ids]})
    if parent is not None:
        par = dict(zip(fs[:, 0].tolist(), fs[:, 1].tolist())); df['stand_id'] = [par[i] for i in ids]
    else:
        nst = pd.Series(sc[:, 1]).value_counts(); df['n_stands'] = [int(nst.get(i, 0)) for i in ids]
    gdf = gpd.GeoDataFrame(df, geometry=[unary_union(geoms[i]) if len(geoms[i]) > 1 else geoms[i][0] for i in ids], crs=ST.crs)
    gdf.to_file(f'{OUT}/levels.gpkg', layer=nm, driver='GPKG')
    print(f'{nm}: {len(gdf)} polygons ({time.time()-t0:.0f}s)', flush=True)
print('done')
