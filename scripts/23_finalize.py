"""Stage D: stands raster + polygons.
  stands.tif   int32 stand id per pixel = sp2stand[superpixel]; forest-land pixels without a superpixel (the 1-px barrier lines
               rasterized inside forest land, and slivers) are given the id of the NEAREST stand pixel of the same land status
               (else any status), so the forest area is complete and every barrier line is split between the stands on each
               side (no stand crosses a barrier: the reassigned pixels only extend stands that already touch the line).
  stands_<name>.gpkg  one polygon per stand (dissolved over processing windows) with attributes:
               stand_id, status (1 forest / 2 harvested / 3 regenerating / 4 forest_unmapped), area_ha, loss_year (mode, harvested),
               tm_main (TreeMap TM_ID with the largest pixel share), tm_share (its share), tm_n (number of distinct TM_IDs),
               BALIVE / STANDHT / QMD / TPA_LIVE / CANOPYPCT / SOFTWOOD_FRAC (TreeMap pixel means over TreeMap pixels),
               soil means, lon / lat of the centroid (EPSG:4326).
Usage: python scripts/23_finalize.py [r0 c0 r1 c1]   (pilot window in pixels; default = the bounding box of sp.tif's non-zero area)"""
import sys, os, time, numpy as np, pandas as pd, rasterio, geopandas as gpd
from rasterio.windows import Window
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape
sys.path.insert(0, os.path.dirname(__file__))
from features import CFG, ROOT, SOILS, TM_ATTRS, vat_table
from common import PROFILE
t0 = time.time(); PXHA = 900 / 1e4
sp2 = np.load(f'{ROOT}/data/derived/sp2stand.npy')
spr = rasterio.open(f'{ROOT}/data/derived/sp.tif'); land = rasterio.open(f'{ROOT}/data/derived/land_clean.tif')
H, W = spr.shape
if len(sys.argv) > 4:
    r0, c0, r1, c1 = map(int, sys.argv[1:5])
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
prof = dict(spr.profile, **PROFILE, dtype='int32', nodata=0)
name = CFG['name']
dst = rasterio.open(f'{ROOT}/data/derived/stands.tif', 'w', **prof)
# --- stands raster with barrier / sliver reassignment (halo so the nearest stand can sit across a tile edge)
for r in range(r0, r1, T):
    for c in range(c0, c1, T):
        ra, ca = max(r - HALO, 0), max(c - HALO, 0); rb, cb = min(r + T + HALO, H), min(c + T + HALO, W)
        win = Window(ca, ra, cb - ca, rb - ra)
        sp = spr.read(1, window=win); ld = land.read(1, window=win)
        st = sp2[sp].astype(np.int32); st[sp == 0] = 0
        fl = (ld >= 1) & (ld <= 4); hole = fl & (st == 0)
        if hole.any() and (st > 0).any():
            # nearest stand pixel of the same status, else of any status
            for s_ in (1, 2, 3, 4):
                hs = hole & (ld == s_); src_ = (st > 0) & (ld == s_)
                if hs.any() and src_.any():
                    d, (ir, ic) = ndimage.distance_transform_edt(~src_, return_indices=True)
                    ok = hs & (d <= 3); st[ok] = st[ir[ok], ic[ok]]
            hole = fl & (st == 0)
            if hole.any():
                _, (ir, ic) = ndimage.distance_transform_edt(st == 0, return_indices=True); st[hole] = st[ir[hole], ic[hole]]
        iy, ix = r - ra, c - ca; h, w = min(T, r1 - r), min(T, c1 - c)
        dst.write(st[iy:iy + h, ix:ix + w], 1, window=Window(c, r, w, h))
        print(f'stands raster tile r{r} c{c} {time.time()-t0:.0f}s', flush=True)
dst.close()
# --- generalisation: fragments below the MMU (mostly slivers boxed in between road / parcel lines, which cannot merge on their own side) are
#     merged into the neighbour with the longest shared boundary (same land status preferred), even across a barrier line. In-memory (pilot windows).
MMU_PX = int(round(CFG['segmentation']['mmu_ha'] / PXHA))
if (r1 - r0) * (c1 - c0) <= 120_000_000:
    with rasterio.open(f'{ROOT}/data/derived/stands.tif', 'r+') as ds:
        win = Window(c0, r0, c1 - c0, r1 - r0); A = ds.read(1, window=win); L = land.read(1, window=win)
        for it in range(6):
            ids, cnt = np.unique(A[A > 0], return_counts=True); size = dict(zip(ids.tolist(), cnt.tolist()))
            small = set(i for i, c in size.items() if c < MMU_PX)
            if not small: break
            stat = {}; flat = A.ravel(); lf = L.ravel()
            for i, s_ in zip(flat[flat > 0], lf[flat > 0]): stat.setdefault(int(i), s_)
            pairs = []
            for a, b in ((A[:, :-1], A[:, 1:]), (A[:-1], A[1:])):
                m = (a != b) & (a > 0) & (b > 0); pairs.append(np.stack([a[m], b[m]], 1)); pairs.append(np.stack([b[m], a[m]], 1))
            P2 = np.concatenate(pairs); P2 = P2[np.isin(P2[:, 0], list(small))]
            if len(P2) == 0: break
            u, c = np.unique(P2, axis=0, return_counts=True)
            best = {}
            for (a, b), n_ in zip(u.tolist(), c.tolist()):
                key = (stat.get(b) == stat.get(a), n_, size.get(b, 0))       # same status first, then longest boundary, then larger
                if a not in best or key > best[a][0]: best[a] = (key, b)
            remap = {a: b for a, (_, b) in best.items() if not (b in small and size[b] < size[a])}
            if not remap: break
            lut_ = np.arange(int(A.max()) + 1); 
            for a, b in remap.items(): lut_[a] = b
            for _ in range(4): lut_ = lut_[lut_]                                 # chains a -> b -> c
            A = lut_[A]
            print(f'MMU generalisation pass {it+1}: merged {len(remap)} fragments < {MMU_PX} px', flush=True)
        ds.write(A.astype(np.int32), 1, window=win)
# --- per-stand attributes (pixel accumulation) + polygons
v = vat_table(); ids = v.TM_ID.values.astype(np.int64); lut = np.full(int(ids.max()) + 1, -1, np.int64); lut[ids] = np.arange(ids.size)
A = np.stack([v[a].astype(float).values for a in TM_ATTRS], 1)
srs = rasterio.open(f'{ROOT}/data/derived/stands.tif'); tmr = rasterio.open(f'{ROOT}/data/treemap/tm_id.tif'); lyr = rasterio.open(f'{ROOT}/data/derived/loss_year.tif')
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
tot = tp.groupby('sid').px.sum(); top = tp.sort_values('px', ascending=False).drop_duplicates('sid').set_index('sid')
out['tm_main'] = top.tm; out['tm_share'] = top.px / tot; out['tm_n'] = tp.groupby('sid').size()
if loss_pairs:
    lp = pd.concat(loss_pairs).groupby(['sid', 'ly']).px.sum().reset_index().sort_values('px', ascending=False).drop_duplicates('sid').set_index('sid')
    out['loss_year'] = (lp.ly.astype(int) + 2000).where(lp.ly > 0)
from shapely.ops import unary_union
gs = gpd.GeoSeries([unary_union(geoms[s]) if len(geoms[s]) > 1 else geoms[s][0] for s in out.index], index=out.index, crs=srs.crs)
gdf = gpd.GeoDataFrame(out, geometry=gs)
cen = gdf.geometry.representative_point().to_crs(4326); gdf['lon'] = cen.x; gdf['lat'] = cen.y
os.makedirs(f'{ROOT}/out', exist_ok=True); fn = f'{ROOT}/out/stands_{name}.gpkg'
gdf.reset_index().to_file(fn, layer='stands', driver='GPKG')
a = gdf.area_ha.values
print(f'wrote {fn}: {len(gdf)} stands; area ha 5/25/50/75/95 %: {np.quantile(a, [.05, .25, .5, .75, .95]).round(1)}; '
      f'status counts {gdf.status.value_counts().to_dict()}; {time.time()-t0:.0f}s')
