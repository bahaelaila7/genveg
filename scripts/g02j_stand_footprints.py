"""Stand footprints of the generative-vegetation mapping set (genveg pipeline stage stands_fp; moved out of g02b 2026-10-21 so the mapping set follows
the configured stand-delineation run without rebuilding the training table): the stands of the run (common.stands_run: STANDS_RUN) that intersect
Baker County, up to 4 footprints (3x3 x 30 m) per stand drawn without replacement (seed 0), with the candidate features of g02b.feats (identical code
to the plots' candidates). Writes data/proto/gafl/stands.parquet (stand_id, row, col) and stand_feat.npy (float16)."""
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
    SR = sr_.read(1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)); rr, cc, F = B.feats(r0, c0, r1 - r0 + 1, c1 - c0 + 1)
    sid = SR[rr, cc]; kp = np.isin(sid, st.stand_id.values); rr, cc, F, sid = rr[kp], cc[kp], F[kp], sid[kp]
    rg = np.random.default_rng(0); sel = np.array([i for _, ix in pd.Series(np.arange(len(sid))).groupby(sid) for i in rg.choice(ix.values, min(4, len(ix)), replace=False)])
    for f in ('stands.parquet', 'stand_feat.npy'):
        if os.path.islink(f'{OUT}/{f}'): os.unlink(f'{OUT}/{f}')
    pd.DataFrame({'stand_id': sid[sel], 'row': rr[sel] + r0, 'col': cc[sel] + c0}).to_parquet(f'{OUT}/stands.parquet'); np.save(f'{OUT}/stand_feat.npy', F[sel].astype(np.float16))
    missing = set(st.stand_id) - set(sid[sel])
    print('stand footprints', len(sel), 'stands', len(set(sid[sel])), f'(stands of the county without a footprint pixel: {len(missing)})')
