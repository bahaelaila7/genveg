"""USGS 3DEP 1 arc-second DEM tiles (~30 m) covering the GA+FL AOI -> data/terrain/raw/USGS_1_<tile>.tif (resumable; tiles that do not
exist upstream, e.g. open ocean, are skipped). Source: https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/current/<tile>/USGS_1_<tile>.tif,
tile name n<lat of the north edge>w<lon of the west edge> (1 x 1 degree)."""
import os, sys, math, time, requests, geopandas as gpd
from shapely.geometry import box
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT
OUT = f'{ROOT}/data/terrain/raw'; os.makedirs(OUT, exist_ok=True)
aoi = gpd.read_file(f'{ROOT}/data/grid/aoi_gafl.gpkg').to_crs(4326).geometry.union_all().buffer(0.05)
lo0, la0, lo1, la1 = aoi.bounds
tiles = [(n, w) for n in range(math.floor(la0) + 1, math.ceil(la1) + 1) for w in range(-math.floor(lo0), -math.floor(lo1) - 1, -1)
         if aoi.intersects(box(-w, n - 1, -w + 1, n))]
URL = 'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/current/{t}/USGS_1_{t}.tif'
def job(nw):
    t = f'n{nw[0]:02d}w{nw[1]:03d}'; fn = f'{OUT}/USGS_1_{t}.tif'
    if os.path.exists(fn): return t, 'skip'
    for k in range(4):
        try:
            r = requests.get(URL.format(t=t), stream=True, timeout=120)
            if r.status_code == 404: return t, 'absent'
            r.raise_for_status()
            with open(fn + '.part', 'wb') as f:
                for b in r.iter_content(1 << 22): f.write(b)
            os.replace(fn + '.part', fn); return t, 'ok'
        except Exception as e:
            time.sleep(5 * (k + 1)); err = str(e)[:80]
    return t, 'failed ' + err
if __name__ == '__main__':
    print(f'{len(tiles)} candidate tiles', flush=True); res = {}
    with ThreadPoolExecutor(6) as ex:
        for t, s in ex.map(job, tiles): res[s.split()[0]] = res.get(s.split()[0], 0) + 1; print(t, s, flush=True)
    print('done', res)
