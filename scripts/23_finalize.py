"""Stage D: stands raster + polygons.
  <OUT>/stands.tif   int32 stand id per pixel = sp2stand[superpixel]; forest-land pixels without a superpixel (the 1-px barrier lines
               rasterized inside forest land, and slivers) are grown into 4-ADJACENT stands (same land status first, up to
               finalize.hole_dist_px rings, then any), so the forest area is complete, every barrier line is split between the stands on
               each side and stands stay connected; pixels no stand can reach become stands of their own.
  <OUT>/stands.gpkg  one polygon per stand (dissolved over processing windows) with attributes:
               stand_id, status (1 forest / 2 harvested / 3 regenerating / 4 forest_unmapped), area_ha, loss_year (mode, harvested),
               tm_main (TreeMap TM_ID with the largest pixel share), tm_share (its share), tm_n (number of distinct TM_IDs),
               BALIVE / STANDHT / QMD / TPA_LIVE / CANOPYPCT / SOFTWOOD_FRAC (TreeMap pixel means over TreeMap pixels),
               soil means, lon / lat of the centroid (EPSG:4326).
Usage: python scripts/23_finalize.py [r0 c0 r1 c1]   (pilot window in pixels; default = the bounding box of sp.tif's non-zero area)"""
import sys, os, time, json, numpy as np, pandas as pd, rasterio, geopandas as gpd
from rasterio.windows import Window
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape
sys.path.insert(0, os.path.dirname(__file__))
from features import CFG, ROOT, SOILS, TM_ATTRS, vat_table
from common import PROFILE
from stands_cfg import OUT, IN, WINDOW
t0 = time.time(); PXHA = 900 / 1e4
sp2 = np.load(f"{IN('merge')}/sp2stand.npy")
spr = rasterio.open(f"{IN('sp')}/sp.tif"); land = rasterio.open(f"{IN('clean')}/land_clean.tif")
H, W = spr.shape
from stands_cfg import sp_extent, mmu_px, barrier_files
if WINDOW or len(sys.argv) > 4:
    r0, c0, r1, c1 = sp_extent(CFG, WINDOW or tuple(map(int, sys.argv[1:5])))   # the superpixel extent: stands are never cut by the window
else:
    rows = []; cols = []
    for r in range(0, H, 2048):
        x = spr.read(1, window=Window(0, r, W, min(2048, H - r))) > 0
        if x.any():
            rr = np.flatnonzero(x.any(1)); cc = np.flatnonzero(x.any(0)); rows += [r + rr[0], r + rr[-1] + 1]; cols += [cc[0], cc[-1] + 1]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
r1 = min(r1, H); c1 = min(c1, W)
print(f'window rows {r0}:{r1} cols {c0}:{c1}', flush=True)
T = 4096; HALO = 64
def grow_holes(st, ld, fl, dmax):
    """Forest-land pixels without a stand (barrier lines, slivers) join a 4-ADJACENT stand, grown outward ring by ring: first only from neighbours of
    the same land status (up to dmax rings), then from any neighbour, so every assigned pixel touches its stand (stands stay connected).
    Returns the stands and the pixels no stand can reach."""
    hole = fl & (st == 0)
    for same in (True, False):
        for _ in range(dmax if same else st.shape[0] + st.shape[1]):
            if not hole.any(): return st, hole
            cand = np.zeros_like(st); found = np.zeros(st.shape, bool)
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nb = np.zeros_like(st); nl = np.zeros_like(ld)
                ys, yd = (slice(dy, None), slice(None, -dy)) if dy > 0 else ((slice(None, dy), slice(-dy, None)) if dy < 0 else (slice(None), slice(None)))
                xs, xd = (slice(dx, None), slice(None, -dx)) if dx > 0 else ((slice(None, dx), slice(-dx, None)) if dx < 0 else (slice(None), slice(None)))
                nb[yd, xd] = st[ys, xs]; nl[yd, xd] = ld[ys, xs]
                ok = hole & ~found & (nb > 0) & ((nl == ld) if same else True)
                cand[ok] = nb[ok]; found |= ok
            if not found.any(): break
            st[found] = cand[found]; hole &= ~found
    return st, hole
NEXT_ID = [int(sp2.max()) + 1]                                              # ids for unreachable slivers
prof = dict(spr.profile, **PROFILE, dtype='int32', nodata=0)
name = CFG['name']
os.makedirs(OUT, exist_ok=True)
dst = rasterio.open(f'{OUT}/stands.tif', 'w', **prof)
# --- stands raster with barrier / sliver reassignment (halo so the nearest stand can sit across a tile edge)
for r in range(r0, r1, T):
    for c in range(c0, c1, T):
        ra, ca = max(r - HALO, 0), max(c - HALO, 0); rb, cb = min(r + T + HALO, H), min(c + T + HALO, W)
        win = Window(ca, ra, cb - ca, rb - ra)
        sp = spr.read(1, window=win); ld = land.read(1, window=win)
        st = sp2[sp].astype(np.int32); st[sp == 0] = 0
        fl = (ld >= 1) & (ld <= 4)
        st, rest = grow_holes(st, ld, fl, CFG['finalize']['hole_dist_px'])
        if rest.any():                                                     # forest pixels no stand can reach (enclosed slivers): stands of their own
            lab, n = ndimage.label(rest); st[rest] = lab[rest] + NEXT_ID[0] - 1; NEXT_ID[0] += n
        iy, ix = r - ra, c - ca; h, w = min(T, r1 - r), min(T, c1 - c)
        dst.write(st[iy:iy + h, ix:ix + w], 1, window=Window(c, r, w, h))
        print(f'stands raster tile r{r} c{c} {time.time()-t0:.0f}s', flush=True)
dst.close()
# --- generalisation: fragments below the MMU (mostly slivers boxed in between road / parcel lines, which cannot merge on their own side) are
#     merged into the neighbour with the longest shared boundary (config: same land status only when status_is_hard, else same status preferred;
#     across barrier lines only when finalize.mmu_across_barriers). A fragment merges only into a LARGER neighbour (or an equal one with a higher id), so
#     merges are acyclic; chains a -> b -> c are resolved to their end. Status of a stand = majority land status of its pixels. In-memory (pilot windows).
MMU_PX = mmu_px(CFG); HARD = CFG['segmentation']['status_is_hard']; FIN = CFG['finalize']
BARRIERS = barrier_files(CFG)
def majority_status(A, L):
    k = int(A.max()) + 1; cnt = np.zeros((k, 5), np.int64)
    np.add.at(cnt, (A.ravel(), np.clip(L.ravel(), 0, 4)), 1); cnt[:, 0] = 0
    return cnt.argmax(1), cnt.sum(1)
GEN = dict(passes=0, merged=0, left_small=0, skipped=False)
if (r1 - r0) * (c1 - c0) > 120_000_000:
    GEN['skipped'] = True; print('!! MMU generalisation SKIPPED: extent > 120 Mpx (in-memory step); verify V5 fails', flush=True)
else:
    with rasterio.open(f'{OUT}/stands.tif', 'r+') as ds:
        win = Window(c0, r0, c1 - c0, r1 - r0); A = ds.read(1, window=win).astype(np.int64); L = land.read(1, window=win)
        bar = np.zeros(A.shape, bool)
        if not FIN['mmu_across_barriers']:
            for b in BARRIERS:
                with rasterio.open(b) as bs: bar |= bs.read(1, window=win) > 0
        for it in range(FIN['mmu_max_passes']):
            stat, size = majority_status(A, L); small = (size < MMU_PX) & (size > 0); small[0] = False
            if not small.any(): break
            pairs = []
            for a, b, ba, bb in ((A[:, :-1], A[:, 1:], bar[:, :-1], bar[:, 1:]), (A[:-1], A[1:], bar[:-1], bar[1:])):
                m = (a != b) & (a > 0) & (b > 0) & ~ba & ~bb; pairs.append(np.stack([a[m], b[m]], 1)); pairs.append(np.stack([b[m], a[m]], 1))
            P2 = np.concatenate(pairs); P2 = P2[small[P2[:, 0]]]
            if HARD: P2 = P2[stat[P2[:, 0]] == stat[P2[:, 1]]]
            P2 = P2[(size[P2[:, 1]] > size[P2[:, 0]]) | ((size[P2[:, 1]] == size[P2[:, 0]]) & (P2[:, 1] > P2[:, 0]))]   # acyclic: only toward larger
            if len(P2) == 0: break
            u, c = np.unique(P2, axis=0, return_counts=True)
            same = (stat[u[:, 0]] == stat[u[:, 1]]).astype(np.int64)
            o = np.lexsort((size[u[:, 1]], c, same, u[:, 0]))        # per fragment: same status, then longest boundary, then larger neighbour
            last = np.ones(len(o), bool); last[:-1] = u[o[:-1], 0] != u[o[1:], 0]
            best = u[o[last]]
            lut_ = np.arange(int(A.max()) + 1); lut_[best[:, 0]] = best[:, 1]
            while True:
                nl = lut_[lut_]
                if np.array_equal(nl, lut_): break
                lut_ = nl
            A = lut_[A]; GEN['passes'] = it + 1; GEN['merged'] += len(best)
            print(f'MMU generalisation pass {it+1}: merged {len(best)} fragments < {MMU_PX} px', flush=True)
        stat, size = majority_status(A, L); GEN['left_small'] = int(((size < MMU_PX) & (size > 0))[1:].sum())
        ds.write(A.astype(np.int32), 1, window=win)
json.dump(GEN, open(f'{OUT}/generalisation.json', 'w'))
print('generalisation', GEN, flush=True)
# --- per-stand attributes (pixel accumulation) + polygons
v = vat_table(); ids = v.TM_ID.values.astype(np.int64); lut = np.full(int(ids.max()) + 1, -1, np.int64); lut[ids] = np.arange(ids.size)
A = np.stack([v[a].astype(float).values for a in TM_ATTRS], 1)
srs = rasterio.open(f'{OUT}/stands.tif'); tmr = rasterio.open(f'{ROOT}/data/treemap/tm_id.tif'); lyr = rasterio.open(f"{IN('land')}/loss_year.tif")
soil = {k: rasterio.open(f'{ROOT}/data/soils/{k}.tif') for k in SOILS}
acc = {}; tm_pairs = []; loss_pairs = []; geoms = {}
for r in range(r0, r1, T):
    for c in range(c0, c1, T):
        h, w = min(T, r1 - r), min(T, c1 - c); win = Window(c, r, w, h)
        st = srs.read(1, window=win); m = st > 0
        if not m.any(): continue
        ld = land.read(1, window=win); tm = tmr.read(1, window=win).astype(np.int64); ly = lyr.read(1, window=win)
        sid = st[m].astype(np.int64)
        df = pd.DataFrame({'sid': sid, 'status': ld[m]})
        tl = tm[m]; vi = np.where((tl > 0) & (tl < lut.size), lut[np.clip(tl, 0, lut.size - 1)], -1)
        for j, a in enumerate(TM_ATTRS):
            x = np.where(vi >= 0, A[np.maximum(vi, 0), j], np.nan); df[a] = x
        for k in SOILS:
            x = soil[k].read(1, window=win).astype(np.float64)[m]; nd = soil[k].nodata
            if nd is not None: x[x == nd] = np.nan
            df[k] = x
        df['n'] = 1.0; df['tmok'] = vi >= 0
        g = df.groupby('sid')
        part = pd.concat([g.n.sum(), g.tmok.sum().rename('ntm'), g[[*TM_ATTRS, *SOILS]].sum(min_count=1),
                          g[[*TM_ATTRS, *SOILS]].count().add_suffix('_c')], axis=1)
        acc[(r, c)] = part
        st_tab = pd.crosstab(df.sid, df.status); acc[(r, c, 'st')] = st_tab
        tm_pairs.append(pd.DataFrame({'sid': sid[vi >= 0], 'tm': tl[vi >= 0]}).value_counts().rename('px').reset_index())
        lm = ld[m] == 2
        if lm.any(): loss_pairs.append(pd.DataFrame({'sid': sid[lm], 'ly': ly[m][lm]}).value_counts().rename('px').reset_index())
        tr = rasterio.windows.transform(win, srs.transform)
        for gj, val in shapes(st, mask=m, connectivity=4, transform=tr):
            geoms.setdefault(int(val), []).append(shape(gj))
        print(f'attributes/polygons tile r{r} c{c} stands so far {len(geoms)} {time.time()-t0:.0f}s', flush=True)
P = pd.concat([acc[k] for k in acc if len(k) == 2]).groupby(level=0).sum(min_count=1)
C = pd.concat([acc[k] for k in acc if len(k) == 3]).groupby(level=0).sum().fillna(0)
out = pd.DataFrame(index=P.index); out.index.name = 'stand_id'
out['status'] = C.idxmax(axis=1).astype(int); out['area_ha'] = P.n * PXHA
for a in [*TM_ATTRS, *SOILS]: out[a] = P[a] / P[a + '_c'].replace(0, np.nan)
out['tm_frac'] = P.ntm / P.n
tp = pd.concat(tm_pairs).groupby(['sid', 'tm']).px.sum().reset_index()
tot = tp.groupby('sid').px.sum(); top = tp.sort_values(['px', 'tm'], ascending=[False, True], kind='stable').drop_duplicates('sid').set_index('sid')
out['tm_main'] = top.tm; out['tm_share'] = top.px / tot; out['tm_n'] = tp.groupby('sid').size()
if loss_pairs:
    lp = pd.concat(loss_pairs).groupby(['sid', 'ly']).px.sum().reset_index().sort_values(['px', 'ly'], ascending=[False, True], kind='stable').drop_duplicates('sid').set_index('sid')
    out['loss_year'] = (lp.ly.astype(int) + 2000).where(lp.ly > 0).where(out.status == 2)          # harvested stands only (mode of their loss years)
from shapely.ops import unary_union
gs = gpd.GeoSeries([unary_union(geoms[s]) if len(geoms[s]) > 1 else geoms[s][0] for s in out.index], index=out.index, crs=srs.crs)
gdf = gpd.GeoDataFrame(out, geometry=gs)
cen = gdf.geometry.representative_point().to_crs(4326); gdf['lon'] = cen.x; gdf['lat'] = cen.y
fn = f'{OUT}/stands.gpkg'
gdf.reset_index().to_file(fn, layer='stands', driver='GPKG')
a = gdf.area_ha.values
print(f'wrote {fn}: {len(gdf)} stands; area ha 5/25/50/75/95 %: {np.quantile(a, [.05, .25, .5, .75, .95]).round(1)}; '
      f'status counts {gdf.status.value_counts().to_dict()}; {time.time()-t0:.0f}s')
