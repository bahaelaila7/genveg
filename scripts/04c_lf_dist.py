"""LANDFIRE annual disturbance 1999-2025 (CONUS, 30 m, EPSG:5070) cropped to the TreeMap grid: data/landfire/dist/dist_<year>.tif (int16 code,
0 = none) + data/landfire/dist/dist_codes.csv (the union of the yearly attribute tables: code -> disturbance type, severity, source).
Source: data/landfire/raw/annual_dist/USAnnualDisturbance_1999_present.zip (nested per-year zips), read without unpacking the GeoTIFFs."""
import sys, os, re, zipfile, tempfile, time, numpy as np, pandas as pd, rasterio
from rasterio.windows import Window
sys.path.insert(0, os.path.dirname(__file__))
from common import *
t0 = time.time(); tr, W, H, crs = grid()
SRC = f'{ROOT}/data/landfire/raw/annual_dist/USAnnualDisturbance_1999_present.zip'; OUT = f'{ROOT}/data/landfire/dist'; os.makedirs(OUT, exist_ok=True)
outer = zipfile.ZipFile(SRC); codes = []
for name in sorted(n for n in outer.namelist() if n.endswith('_CONUS.zip')):
    yy = int(re.search(r'Dist(\d\d)_CONUS', name).group(1)); year = 1900 + yy if yy >= 90 else 2000 + yy
    fn = f'{OUT}/dist_{year}.tif'
    with tempfile.TemporaryDirectory(dir=f'{ROOT}/data/landfire/raw') as td:
        inner = os.path.join(td, 'y.zip')
        with outer.open(name) as s, open(inner, 'wb') as d:
            while (b := s.read(1 << 24)): d.write(b)
        z = zipfile.ZipFile(inner); members = z.namelist()
        tif = [m for m in members if m.lower().endswith('.tif')][0]
        csvs = [m for m in members if m.lower().endswith('.csv')]
        if csvs:
            t = pd.read_csv(z.open(csvs[0])); t['year'] = year; codes.append(t)
        if not os.path.exists(fn):
            with rasterio.open(f'/vsizip/{inner}/{tif}') as s:
                col = round((tr.c - s.transform.c) / 30); row = round((s.transform.f - tr.f) / 30)
                assert abs((tr.c - s.transform.c) / 30 - col) < 1e-6 and abs((s.transform.f - tr.f) / 30 - row) < 1e-6, 'grid not aligned'
                prof = dict(PROFILE, width=W, height=H, count=1, dtype='int16', crs=crs, transform=tr, nodata=None)
                n = 0
                with rasterio.open(fn, 'w', **prof) as o:
                    for r in range(0, H, 2048):
                        h = min(2048, H - r)
                        a = s.read(1, window=Window(col, row + r, W, h), boundless=True, fill_value=0).astype(np.int32)
                        a[(a < 0) | (a == s.nodata if s.nodata is not None else False)] = 0
                        o.write(a.astype(np.int16), 1, window=Window(0, r, W, h)); n += int((a > 0).sum())
            print(f'{year}: {name.split("/")[-1]} disturbed px {n:,} ({time.time()-t0:.0f}s)', flush=True)
C = pd.concat(codes, ignore_index=True); C.columns = [c.strip() for c in C.columns]
C.to_csv(f'{OUT}/dist_codes.csv', index=False); print('codes', C.shape, C.columns.tolist()[:12])
