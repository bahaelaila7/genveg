"""AlphaEarth annual embeddings -> Region 4 90 m footprint means (docs/GENVEG_V4_R4.md §B). Per 90 m output window (256 x 256 px = 768 x 768 at 30 m):
  1. the 20 m overview (level 0) of every intersecting source COG is read via /vsicurl (nothing stored locally), de-quantised, averaged to the 30 m
     grid_r4 lattice, renormalised to unit length and int8-quantised EXACTLY as scripts/02_aef.py (the 30 m product of GA+FL). Each 90 m cell takes
     its pixels from the source tiles of the UTM zone that contains the cell centre (zone = floor((lon + 180) / 6) + 1 of the centre's longitude;
     G51.24 / E15-4: the zone tiles overlap near -90 / -84 / -78 and the old last-tile-wins order mixed two tiles inside one footprint); among the
     tiles of that zone the first valid pixel in path order counts. A cell no tile of its own zone covers at all takes the other tiles' pixels (same
     order), so a coverage gap of the zone tiles stays filled; the per-tile count of such cells is the tile tag `zone_fallback_cells`;
  2. de-quantised again (sign(v)*(v/127.5)^2) and averaged over each 3x3 block, nodata excluded (= the footprint mean m3n of g02b.feats);
  3. re-quantised for storage with the same scheme (sign*rint(127.5*sqrt|v|)); -128 = no valid 30 m pixel or outside the AOI.
Per-window tiles in data/aef90/tiles_<year>/ (resumable), then assembled into data/aef90/aef90_<year>.tif (64 bands int8, grid_r4_90).
  .venv/bin/python -I scripts/r4_aef90.py <year> [nproc=4]   (AEF90_CACHE_GB: local source budget, default 25)        (then: ... r4_aef90.py <year> assemble)"""
import os as _os; _SM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _WS = _os.environ.get('WORKSPACE', _os.path.dirname(_SM))   # relocatable paths
import sys; sys.path.insert(0, _SM + '/scripts')
import os, glob, re, json, time, numpy as np, pandas as pd, rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window, transform as wtransform
from rasterio.transform import Affine
from multiprocessing import Pool
from pyproj import Transformer
from common import ROOT, PROFILE
G90 = json.load(open(f'{ROOT}/data/grid/grid_r4_90.json')); TR90 = Affine(*G90['transform']); W90, H90 = G90['width'], G90['height']
T = 256                                                            # output window (90 m px); sources are local copies (fetched ahead within a disk budget)
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tiff', GDAL_HTTP_MULTIPLEX='YES', GDAL_HTTP_VERSION='2',
           GDAL_CACHEMAX=128, CPL_VSIL_CURL_CACHE_SIZE=33554432,   # small caches: ~0.6 GB per worker (shared machine)
           GDAL_HTTP_MAX_RETRY=8, GDAL_HTTP_RETRY_DELAY=5, GDAL_NUM_THREADS=1,
           CPL_VSIL_CURL_CHUNK_SIZE=10485760, GDAL_HTTP_MERGE_CONSECUTIVE_RANGES='YES')   # 10 MB chunks: ~4x faster reads
OUT = f'{ROOT}/data/aef90'
def https(p): return '/vsicurl/' + p.replace('s3://us-west-2.opendata.source.coop', 'https://data.source.coop')
CACHE = f'{OUT}/cache'
def local(p): return f'{CACHE}/' + p.split('/annual/')[1].replace('/', '_')            # sparse local copy of the 20 m overview byte span (02a scheme)
import io, requests, tifffile
_S = requests.Session()
def _rng(u, a, b):
    for k in range(8):
        try:
            r = _S.get(u, headers={'Range': f'bytes={a}-{b}'}, timeout=180)
            if r.status_code in (200, 206) and len(r.content) == b - a + 1: return r.content
        except Exception: pass
        time.sleep(3 + 5 * k)
    raise RuntimeError(f'range fail {u} {a}')
def fetch(p):
    """the 20 m overview byte span of one COG into a sparse local file (same offsets as the remote file; GDAL reads it as the COG). Returns bytes on disk."""
    lp = local(p)
    if os.path.exists(lp + '.done'): a, b = map(int, open(lp + '.done').read().split()); return b - a
    u = p.replace('s3://us-west-2.opendata.source.coop', 'https://data.source.coop')
    head = _rng(u, 0, 4 * 1024 * 1024 - 1)
    with tifffile.TiffFile(io.BytesIO(head)) as t: pg = t.pages[1]; offs = pg.dataoffsets; cnt = pg.databytecounts
    a = min(offs); b = max(o + c for o, c in zip(offs, cnt)); size = int(_S.head(u).headers['content-length'])
    with open(lp, 'wb') as f:
        f.truncate(size); f.seek(0); f.write(head)
        for s_ in range(a, b, 8 << 20): f.seek(s_); f.write(_rng(u, s_, min(s_ + (8 << 20), b) - 1))
    open(lp + '.done', 'w').write(f'{a} {b}'); return b - a
def tdir(y): return f'{OUT}/tiles_{y}'
def tile_path(y, r, c): return f'{tdir(y)}/t_{r:05d}_{c:05d}.tif'
ZONE_RULE = 'G51.24 cell-centre UTM zone'
def tile_done(p):
    """a finished window: an empty marker, or a tile written under the current zone rule (tiles of the old last-tile-wins rule are redone)"""
    if os.path.exists(p + '.empty'): return True
    if not os.path.exists(p): return False
    with rasterio.open(p) as t_: return t_.tags().get('zone_rule') == ZONE_RULE
def dq(a):
    v = a.astype(np.float32); out = np.sign(v) * (v / 127.5) ** 2; out[a == -128] = np.nan; return out
def quant(x): return (np.sign(x) * np.rint(127.5 * np.sqrt(np.abs(x)))).clip(-127, 127).astype(np.int8)
IDX = None
def utm_zone_of_crs(crs):
    """the UTM zone of a northern-hemisphere source tile's CRS (EPSG:326zz)"""
    code = int(str(crs).split(':')[1])
    if not 32601 <= code <= 32660: raise ValueError(f'source tile CRS {crs} is not a northern UTM zone')
    return code - 32600
_LL = Transformer.from_crs('EPSG:5070', 'EPSG:4326', always_xy=True)
def cell_zones(wt90, h, w):
    """UTM zone of the CENTRE of every 90 m cell (h x w) of an output window with transform wt90, from its longitude (G51.24)"""
    rr, cc = np.mgrid[0:h, 0:w]; x, y = wt90 * (cc + 0.5, rr + 0.5); lon, _ = _LL.transform(x, y)
    return (np.floor((np.asarray(lon) + 180) / 6).astype(int) + 1).reshape(h, w)
def place(dst_k, alt_k, tmp, own):
    """one band of one source tile into the window: its valid pixels fill the still-empty pixels of the cells of its own zone (dst) or, for the
    other cells, of the fallback layer (alt)"""
    ok = ~np.isnan(tmp); m = ok & own & np.isnan(dst_k); dst_k[m] = tmp[m]; m = ok & ~own & np.isnan(alt_k); alt_k[m] = tmp[m]
def fallback(dst, alt, h, w):
    """cells without any valid pixel from a tile of their own zone take the fallback layer; returns the number of such cells that it fills"""
    empty = np.isnan(dst[0]).reshape(h, 3, w, 3).all(axis=(1, 3)); e30 = np.repeat(np.repeat(empty, 3, 0), 3, 1)
    n = int((empty & ~np.isnan(alt[0]).reshape(h, 3, w, 3).all(axis=(1, 3))).sum()); dst[:, e30] = alt[:, e30]; return n
def needs(job):
    """source COGs (rows of IDX) intersecting the window: the exact test of the loop below"""
    y, r, c = job; h = min(T, H90 - r); w = min(T, W90 - c)
    wt90 = wtransform(Window(c, r, w, h), TR90); bx = (wt90.c, wt90.f - 90 * h, wt90.c + 90 * w, wt90.f)
    gw, gs, ge, gn = transform_bounds('EPSG:5070', 'EPSG:4326', *bx, densify_pts=21); m_ = 0.05
    cand = IDX[(IDX.wgs84_east >= gw - m_) & (IDX.wgs84_west <= ge + m_) & (IDX.wgs84_north >= gs - m_) & (IDX.wgs84_south <= gn + m_)]
    out = []
    for i, f in cand.iterrows():
        l, b, rr, t = transform_bounds('EPSG:5070', f.crs, *bx, densify_pts=21)
        if not (rr < f.utm_west or l > f.utm_east or t < f.utm_south or b > f.utm_north): out.append(i)
    return out
def work(job):
    y, r, c = job
    out = tile_path(y, r, c)
    if tile_done(out): return job, 'skip', 0
    t0 = time.time(); h = min(T, H90 - r); w = min(T, W90 - c)
    with rasterio.open(f'{ROOT}/data/grid/aoi_r4_mask90.tif') as m: mm = m.read(1, window=Window(c, r, w, h))
    if mm.sum() == 0: open(out + '.empty', 'w').close(); return job, 'empty', 0
    wt90 = wtransform(Window(c, r, w, h), TR90); wt = Affine(30, 0, wt90.c, 0, -30, wt90.f); h3, w3 = 3 * h, 3 * w
    bx = (wt.c, wt.f - 30 * h3, wt.c + 30 * w3, wt.f)
    dst = np.full((64, h3, w3), np.nan, np.float32); alt = np.full((64, h3, w3), np.nan, np.float32); tmp = np.empty((h3, w3), np.float32); nsrc = 0
    z30 = np.repeat(np.repeat(cell_zones(wt90, h, w), 3, 0), 3, 1)                 # zone of each 30 m pixel's 90 m cell (G51.24)
    gw, gs, ge, gn = transform_bounds('EPSG:5070', 'EPSG:4326', *bx, densify_pts=21); m_ = 0.05      # vectorised prefilter on the lon/lat boxes
    cand = IDX[(IDX.wgs84_east >= gw - m_) & (IDX.wgs84_west <= ge + m_) & (IDX.wgs84_north >= gs - m_) & (IDX.wgs84_south <= gn + m_)].sort_values('path')
    with rasterio.Env(**ENV):
        for _, f in cand.iterrows():
            l, b, rr, t = transform_bounds('EPSG:5070', f.crs, *bx, densify_pts=21)
            if rr < f.utm_west or l > f.utm_east or t < f.utm_south or b > f.utm_north: continue
            mg = 3
            c0 = max(int(np.floor((l - f.utm_west) / 20)) - mg, 0); c1 = min(int(np.ceil((rr - f.utm_west) / 20)) + mg, 4096)
            r0 = max(int(np.floor((b - f.utm_south) / 20)) - mg, 0); r1 = min(int(np.ceil((t - f.utm_south) / 20)) + mg, 4096)
            if c1 <= c0 or r1 <= r0: continue
            for k_ in range(4):
                try:
                    with rasterio.open(local(f.path), overview_level=0) as s: a = s.read(window=Window(c0, r0, c1 - c0, r1 - r0)); break
                except Exception as e:
                    if k_ == 3: raise
                    time.sleep(10)
            if (a[0] == -128).all(): continue
            a = a[:, ::-1]                                             # bottom-up -> north-up (as 02_aef.py)
            st = Affine(20, 0, f.utm_west + c0 * 20, 0, -20, f.utm_south + r1 * 20); own = z30 == utm_zone_of_crs(f.crs)
            for k in range(64):
                tmp.fill(np.nan)
                reproject(dq(a[k]), tmp, src_transform=st, src_crs=f.crs, dst_transform=wt, dst_crs='EPSG:5070', resampling=Resampling.average,
                          src_nodata=np.nan, dst_nodata=np.nan, init_dest_nodata=False)
                place(dst[k], alt[k], tmp, own)
            nsrc += 1
    nfb = fallback(dst, alt, h, w); del alt
    # 30 m product exactly as 02_aef.py: renormalise, quantise
    bad = np.isnan(dst[0]); dst[:, bad] = 0
    n = np.sqrt((dst ** 2).sum(0)); bad |= (n == 0); n[n == 0] = 1; dst /= n
    q = quant(dst); q[:, bad] = -128
    # footprint (3x3) means of the de-quantised 30 m values, nodata excluded
    D = dq(q).reshape(64, h, 3, w, 3); V = ~np.isnan(D[:1])
    S = np.nansum(D, axis=(2, 4)); N = V.sum(axis=(2, 4))
    M = S / np.maximum(N, 1); q90 = quant(M); q90[:, (N[0] == 0) | (mm == 0)] = -128
    os.makedirs(tdir(y), exist_ok=True)
    with rasterio.open(out + '.part', 'w', **dict(PROFILE, width=w, height=h, count=64, dtype='int8', crs='EPSG:5070', transform=wt90, nodata=-128,
                                                   zstd_level=3, INTERLEAVE='PIXEL')) as d:
        d.write(q90); d.update_tags(zone_rule=ZONE_RULE, zone_fallback_cells=str(nfb))
    os.replace(out + '.part', out)
    return job, f'{nsrc} src', time.time() - t0
def assemble(y):
    fs = sorted(glob.glob(f'{tdir(y)}/t_*.tif')); old = [f for f in fs if not tile_done(f)]
    if old: sys.exit(f'{len(old)} tiles of {tdir(y)} predate the G51.24 zone rule (e.g. {old[0]}): rerun `r4_aef90.py {y}` first')
    prof = dict(driver='GTiff', width=W90, height=H90, count=64, dtype='int8', crs='EPSG:5070', transform=TR90, nodata=-128, tiled=True, blockxsize=512,
                blockysize=512, compress='ZSTD', zstd_level=9, predictor=2, INTERLEAVE='PIXEL', BIGTIFF='YES', SPARSE_OK='TRUE')
    with rasterio.open(f'{OUT}/aef90_{y}.tif', 'w', **prof) as d:
        for k in range(64): d.set_band_description(k + 1, f'A{k:02d}')
        d.update_tags(source='AlphaEarth Foundations annual v1 (tge-labs/aef, source.coop)', year=str(y), dequant='sign(v)*(v/127.5)**2',
                      processing='20m overview -> 30m EPSG:5070 average (float), unit-normalised, int8 (02_aef.py); 3x3 mean of de-quantised 30m (nodata excluded) -> 90m, int8; '
                                 'each 90m cell from the source tiles of the UTM zone of its centre longitude (G51.24)')
        for f in fs:
            r, c = map(int, re.findall(r't_(\d+)_(\d+)', f)[0])
            with rasterio.open(f) as s: d.write(s.read(), window=Window(c, r, s.width, s.height))
    print(y, len(fs), 'tiles assembled')
if __name__ == '__main__':
    y = int(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == 'assemble': assemble(y); sys.exit(0)
    npr = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    BUDGET = float(os.environ.get('AEF90_CACHE_GB', 25)) * 1e9
    I = pd.read_parquet(f'{ROOT}/data/aef/aef_index.parquet'); I = I[I.year == y]
    with rasterio.open(f'{ROOT}/data/grid/aoi_r4_mask90.tif') as m: B = transform_bounds('EPSG:5070', 'EPSG:4326', *m.bounds); MM = m.read(1)
    IDX = I[(I.wgs84_east >= B[0]) & (I.wgs84_west <= B[2]) & (I.wgs84_north >= B[1]) & (I.wgs84_south <= B[3])].reset_index(drop=True)
    os.makedirs(tdir(y), exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    jobs = [(y, r, c) for r in range(0, H90, T) for c in range(0, W90, T)
            if MM[r:r + T, c:c + T].any() and not tile_done(tile_path(y, r, c))]
    need = {j: needs(j) for j in jobs}
    for j in [j for j in jobs if not need[j]]: open(tile_path(*j) + '.empty', 'w').close()      # no source at all (ocean edge)
    jobs = [j for j in jobs if need[j]]
    used = sorted({i for j in jobs for i in need[j]}, key=lambda i: (IDX.wgs84_west[i], IDX.wgs84_south[i]))
    rank = {i: k for k, i in enumerate(used)}; jobs.sort(key=lambda j: max(rank[i] for i in need[j]))
    left = {i: 0 for i in used}
    for j in jobs:
        for i in need[j]: left[i] += 1
    print(f'{y}: {len(IDX)} source COGs in the bbox, {len(used)} needed by {len(jobs)} windows; cache budget {BUDGET / 1e9:.0f} GB', flush=True)
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    dl = ThreadPoolExecutor(6); fut = {}; ready = {}; nxt = 0; cache = 0.0; t0 = time.time(); done = 0
    pend = deque(jobs); running = {}
    with Pool(npr) as p:
        while pend or running:
            for i in [i for i, f_ in fut.items() if f_.done()]: ready[i] = fut.pop(i).result(); cache += ready[i]
            stalled = not running and not any(all(i in ready for i in need[j]) for j in list(pend)[:50])
            while nxt < len(used) and len(fut) < 6 and (cache + 0.45e9 * (len(fut) + 1) <= BUDGET or (stalled and not fut)):
                fut[used[nxt]] = dl.submit(fetch, IDX.path[used[nxt]]); nxt += 1; stalled = False
            while pend and len(running) < 2 * npr and all(i in ready for i in need[pend[0]]):
                j = pend.popleft(); running[j] = p.apply_async(work, (j,))
            for j in [j for j, rr in running.items() if rr.ready()]:
                _, st, dt = running.pop(j).get(); done += 1
                for i in need[j]:
                    left[i] -= 1
                    if left[i] == 0:
                        for f_ in (local(IDX.path[i]), local(IDX.path[i]) + '.done'):
                            if os.path.exists(f_): os.remove(f_)
                        cache -= ready.pop(i)
                if done % 20 == 0 or done == len(jobs):
                    print(f'{done}/{len(jobs)} windows; sources fetched {nxt}/{len(used)}; cache {cache / 1e9:.1f} GB; elapsed {time.time() - t0:.0f}s', flush=True)
            time.sleep(0.5)
