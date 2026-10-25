"""Stand footprints of the generative-vegetation mapping set (genveg pipeline stage stands_fp; moved out of g02b 2026-10-21 so the mapping set follows
the configured stand-delineation run without rebuilding the training table): the stands of the run (common.stands_run: STANDS_RUN) that intersect
Baker County, up to 4 footprints (3x3 x 30 m) per stand drawn without replacement (seed 0) from the 3-px grid, plus one at the central pixel of
every stand the grid misses (slivers), with the candidate features of g02b.feats (identical code to the plots' candidates). Writes data/proto/gafl/stands.parquet (stand_id, row, col) and stand_feat.npy (float16)."""
import os, sys, numpy as np, pandas as pd, rasterio, geopandas as gpd
from rasterio.windows import Window
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g02b_build_gafl as B
from stand_runs import stands_run
if __name__ == '__main__':
    SR_ = stands_run(); print('stand run', SR_['dir'], flush=True)
    B.init(); OUT = B.OUT
    cty = gpd.read_file(f'{B.ROOT}/data/derived/baker_county.gpkg').to_crs(B._S['land'].crs).geometry.iloc[0]
    st = gpd.read_file(SR_['gpkg'], layer='stands'); st = st[st.intersects(cty)]
    sr_ = rasterio.open(SR_['tif']); x0, y0, x1, y1 = st.total_bounds; r0, c0 = sr_.index(x0, y1); r1, c1 = sr_.index(x1, y0)
    dR, dC = B.stand_offset(sr_)       # AEF90 (Region 4 rasters): feature grid = grid_r4, a multiple of 3 px off the stand raster; windows on the 90 m blocks
    if B.AEF90: r0, c0 = 3 * (r0 // 3), 3 * (c0 // 3); r1, c1 = 3 * (r1 // 3) + 2, 3 * (c1 // 3) + 2
    SR = sr_.read(1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1), boundless=True, fill_value=0); rr, cc, F = B.feats(r0 + dR, c0 + dC, r1 - r0 + 1, c1 - c0 + 1)
    sid = SR[rr, cc]; kp = np.isin(sid, st.stand_id.values); rr, cc, F, sid = rr[kp], cc[kp], F[kp], sid[kp]
    rg = np.random.default_rng(0); sel = np.array([i for _, ix in pd.Series(np.arange(len(sid))).groupby(sid) for i in rg.choice(ix.values, min(4, len(ix)), replace=False)])
    # stands the 3-px footprint grid misses (slivers, mostly 1-2 px): one footprint at the stand pixel nearest the stand's mean position
    fr, fc, Fx, sx = [], [], [], []
    for s_ in sorted(set(st.stand_id.values) - set(sid[sel])):
        yy, xx = np.nonzero(SR == s_)
        if len(yy) == 0: continue
        k = np.argmin((yy - yy.mean()) ** 2 + (xx - xx.mean()) ** 2); r_, c_ = int(yy[k]) + r0, int(xx[k]) + c0
        if B.AEF90: r_, c_ = 3 * (r_ // 3) + 1, 3 * (c_ // 3) + 1          # (AEF90: the 90 m block holding that pixel)
        rr1, cc1, F1 = B.feats(r_ - 1 + dR, c_ - 1 + dC, 3, 3)               # the centre of a 3 x 3 window = that pixel (if forest land)
        if len(rr1): fr.append(r_); fc.append(c_); Fx.append(F1[0]); sx.append(s_)
    print(f'sliver stands given a footprint at their central pixel: {len(sx)}', flush=True)
    for f in ('stands.parquet', 'stand_feat.npy'):
        if os.path.islink(f'{OUT}/{f}'): os.unlink(f'{OUT}/{f}')
    S_ = pd.DataFrame({'stand_id': np.concatenate([sid[sel], np.array(sx, sid.dtype)]), 'row': np.concatenate([rr[sel] + r0, fr]).astype(int),
                       'col': np.concatenate([cc[sel] + c0, fc]).astype(int), 'kind': ['grid'] * len(sel) + ['sliver'] * len(sx)})
    S_.to_parquet(f'{OUT}/stands.parquet'); np.save(f'{OUT}/stand_feat.npy', np.concatenate([F[sel], np.array(Fx, F.dtype).reshape(-1, F.shape[1])]).astype(np.float16))
    sid = S_.stand_id.values; sel = np.arange(len(sid))
    missing = set(st.stand_id) - set(sid)
    print('stand footprints', len(sel), 'stands', len(set(sid[sel])), f'(stands of the county without a footprint pixel: {len(missing)})')
