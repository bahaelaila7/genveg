"""Targeted AlphaEarth fetch for the generative-vegetation likelihood (user OK 2026-10-20, "continue"): same-year embeddings at the FIA visits of the
well-localised plots (FIA-dated events pin >= 50 % of the prior to <= 50 footprints; data/proto/gafl/localisation.parquet), years 2017-2025 except
2022 / 2025 (local mosaics). For each (plot, year): the plot's fuzz-disk window on the 30 m TreeMap grid (same window as g02b), read from the remote AlphaEarth
COGs (20 m level-0 overview, HTTP range reads via /vsicurl, bottom-up rows flipped as in 02_aef.py), averaged to 30 m in de-quantised space, renormalised and
re-quantised → data/proto/gafl/visit_aef/<st>_<u>_<c>_<p>_<year>.npy (int8 [64, h, w], -128 = nodata; FIA plot identity, NOT pid, which shifts
between table rebuilds; the window is the plot's g02b disk window, row0/col0 = plot pixel - RP). Run g02d_localisation.py first. Resumable."""
import os, sys, time, numpy as np, pandas as pd, rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import Affine
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from common import grid, ROOT
tr, W, H, crs = grid(); D = f'{ROOT}/data/proto/gafl'; OUT = f'{D}/visit_aef'; os.makedirs(OUT, exist_ok=True)
R = 1600.0; RP = int(R // 30) + 2
LS = pd.read_parquet(f'{D}/localisation.parquet'); loc = set(LS.index[(LS.n_hi > 0) & (LS.n_hi <= 50) & (LS.mass_hi >= 0.5)])
V = pd.read_parquet(f'{D}/visits.parquet'); P = pd.read_parquet(f'{D}/plots.parquet').set_index('pid')
v = V[V.pid.isin(loc)].copy(); v['yi'] = v.year.astype(int)
need = v[(v.yi >= 2017) & (v.yi <= 2025) & ~v.yi.isin([2022, 2025])][['pid', 'yi']].drop_duplicates().values.tolist()
IDX = pd.read_parquet(f'{ROOT}/data/aef/aef_index.parquet')
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tiff', GDAL_HTTP_MULTIPLEX='YES', GDAL_HTTP_VERSION='2',
           GDAL_HTTP_MAX_RETRY=5, GDAL_HTTP_RETRY_DELAY=3, CPL_VSIL_CURL_CACHE_SIZE=100000000, GDAL_CACHEMAX=256)
def dq(a): x = a.astype(np.float32); o = np.sign(x) * (x / 127.5) ** 2; o[a == -128] = np.nan; return o
def job(a):
    pid, y = a; p = P.loc[pid]; fn = f'{OUT}/{int(p.st)}_{int(p.u)}_{int(p.c)}_{int(p.p)}_{y}.npy'
    if os.path.exists(fn): return a, 'skip', 0.0
    t0 = time.time()
    with rasterio.open(f'{ROOT}/data/derived/land_clean.tif') as s: row, col = s.index(p.x, p.y)
    r0, c0 = row - RP, col - RP; h = w = 2 * RP + 1
    wt = rasterio.windows.transform(Window(c0, r0, w, h), tr); bx = (wt.c, wt.f + wt.e * h, wt.c + wt.a * w, wt.f)
    dst = np.full((64, h, w), np.nan, np.float32); n = 0
    with rasterio.Env(**ENV):
        lon, lat = p.lon, p.lat; dd = 0.05                                  # prefilter the global index by WGS84 bounds (~33k files per year)
        cand = IDX[(IDX.year == y) & (IDX.wgs84_west <= lon + dd) & (IDX.wgs84_east >= lon - dd) & (IDX.wgs84_south <= lat + dd) & (IDX.wgs84_north >= lat - dd)]
        for _, f in cand.iterrows():
            l, b, rr, t = transform_bounds('EPSG:5070', f.crs, *bx, densify_pts=21)
            if rr < f.utm_west or l > f.utm_east or t < f.utm_south or b > f.utm_north: continue
            m = 3
            cc0 = max(int(np.floor((l - f.utm_west) / 20)) - m, 0); cc1 = min(int(np.ceil((rr - f.utm_west) / 20)) + m, 4096)
            rr0 = max(int(np.floor((b - f.utm_south) / 20)) - m, 0); rr1 = min(int(np.ceil((t - f.utm_south) / 20)) + m, 4096)
            if cc1 <= cc0 or rr1 <= rr0: continue
            u = '/vsicurl/' + f.path.replace('s3://us-west-2.opendata.source.coop', 'https://data.source.coop')
            with rasterio.open(u, overview_level=0) as s: arr = s.read(window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0))
            if (arr[0] == -128).all(): continue
            arr = arr[:, ::-1]; st = Affine(20, 0, f.utm_west + cc0 * 20, 0, -20, f.utm_south + rr1 * 20)
            for k in range(64):
                reproject(dq(arr[k]), dst[k], src_transform=st, src_crs=f.crs, dst_transform=wt, dst_crs='EPSG:5070', resampling=Resampling.average,
                          src_nodata=np.nan, dst_nodata=np.nan, init_dest_nodata=False)
            n += 1
    bad = np.isnan(dst[0]); dst[:, bad] = 0; nr = np.sqrt((dst ** 2).sum(0)); bad |= nr == 0; nr[nr == 0] = 1; dst /= nr
    q = (np.sign(dst) * np.rint(127.5 * np.sqrt(np.abs(dst)))).clip(-127, 127).astype(np.int8); q[:, bad] = -128
    np.save(fn + '.part.npy', q); os.replace(fn + '.part.npy', fn)
    return a, f'ok src {n}', time.time() - t0
if __name__ == '__main__':
    t0 = time.time(); print('windows to fetch', len(need), flush=True)
    with ThreadPoolExecutor(int(os.environ.get('NT', 10))) as ex:
        for i, (a, st, dt) in enumerate(ex.map(job, need)):
            if i % 25 == 0: print(f'{i+1}/{len(need)} {a} {st} {dt:.1f}s elapsed {time.time()-t0:.0f}s', flush=True)
    print('done', time.time() - t0)
