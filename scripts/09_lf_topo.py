"""LANDFIRE topography (LF2020 Elev / SlpD / Asp, 30 m, the CONUS grid of EVT and the disturbance products — user, 2026-10-22: use LANDFIRE's
30 m terrain rather than 3DEP) and LANDFIRE EVC 2022 (existing vegetation cover), cropped to the TreeMap grid WITHOUT resampling (integer pixel
offset asserted, as in 04c_lf_dist.py). Derived:
  data/terrain/elevation.tif (m), slope.tif (deg), northness.tif = cos(aspect)·sin(slope), eastness.tif = sin(aspect)·sin(slope) (flat: 0),
  tpi300.tif / tpi2k.tif = elevation − mean elevation in a 21 x 21 / 135 x 135 px window (≈ 300 m / 2 km; nodata-aware), all float32;
  data/landfire/evc2022.tif (uint16 code) and data/derived/tree_cover.tif (EVC 110-199 -> tree cover %, else 0; uint8).
Sources: data/landfire/raw/topo/LF2020_{Elev,SlpD,Asp}_CONUS.zip (landfire.gov/data-downloads/CONUS_Topo_LF2020/), <WORKSPACE>/LF2022_EVC_CONUS/Tif."""
import os, sys, zipfile, time, numpy as np, rasterio
from rasterio.windows import Window
from scipy.ndimage import uniform_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import grid, ROOT, WS, PROFILE
tr, W, H, crs = grid(); OUT = f'{ROOT}/data/terrain'; os.makedirs(OUT, exist_ok=True); T, HALO = 2048, 70
def src_path(name):
    z = f'{ROOT}/data/landfire/raw/topo/{name}.zip'; tif = [m for m in zipfile.ZipFile(z).namelist() if m.lower().endswith('.tif')][0]
    return f'/vsizip/{z}/{tif}'
def crop(path, out, dtype, nodata_out, fn=None):
    with rasterio.open(path) as s:
        col = round((tr.c - s.transform.c) / 30); row = round((s.transform.f - tr.f) / 30)
        assert abs((tr.c - s.transform.c) / 30 - col) < 1e-6 and abs((s.transform.f - tr.f) / 30 - row) < 1e-6, f'{path}: grid not aligned'
        prof = dict(PROFILE, width=W, height=H, count=1, dtype=dtype, crs=crs, transform=tr, nodata=nodata_out)
        with rasterio.open(out, 'w', **prof) as o:
            for r in range(0, H, 2048):
                h = min(2048, H - r); a = s.read(1, window=Window(col, row + r, W, h), boundless=True, fill_value=s.nodata if s.nodata is not None else -9999)
                o.write((fn(a, s.nodata) if fn else a).astype(dtype), 1, window=Window(0, r, W, h))
    print('cropped', os.path.basename(out), f'{time.time() - t0:.0f}s', flush=True)
def nodata_f(a, nd): a = a.astype(np.float32); a[(a == nd) | (a < -1000)] = np.nan; return a
if __name__ == '__main__':
    t0 = time.time()
    crop(src_path('LF2020_Elev_CONUS'), f'{OUT}/elevation.tif', 'float32', np.nan, nodata_f)
    crop(src_path('LF2020_SlpD_CONUS'), f'{OUT}/slope.tif', 'float32', np.nan, nodata_f)
    crop(src_path('LF2020_Asp_CONUS'), f'{OUT}/aspect.tif', 'float32', np.nan, lambda a, nd: np.where((a == nd) | (a < -1), np.nan, a).astype(np.float32))
    evc = f'{WS}/LF2022_EVC_CONUS/Tif/LF2022_EVC_CONUS.tif'
    crop(evc, f'{ROOT}/data/landfire/evc2022.tif', 'int16', -9999)
    prof = dict(PROFILE, width=W, height=H, count=1, dtype='float32', crs=crs, transform=tr, nodata=np.nan)
    outs = {k: rasterio.open(f'{OUT}/{k}.tif', 'w', **prof) for k in ('northness', 'eastness', 'tpi300', 'tpi2k')}
    tc = rasterio.open(f'{ROOT}/data/derived/tree_cover.tif', 'w', **dict(prof, dtype='uint8', nodata=255))
    el, sl, asp, ev = (rasterio.open(p) for p in (f'{OUT}/elevation.tif', f'{OUT}/slope.tif', f'{OUT}/aspect.tif', f'{ROOT}/data/landfire/evc2022.tif'))
    for r in range(0, H, T):
        for c in range(0, W, T):
            h, w = min(T, H - r), min(T, W - c); win = Window(c, r, w, h)
            s_ = np.radians(sl.read(1, window=win)); a_ = asp.read(1, window=win); flat = np.isnan(a_) | (a_ < 0)
            outs['northness'].write(np.where(flat, 0, np.cos(np.radians(a_)) * np.sin(s_)).astype(np.float32), 1, window=win)
            outs['eastness'].write(np.where(flat, 0, np.sin(np.radians(a_)) * np.sin(s_)).astype(np.float32), 1, window=win)
            E = el.read(1, window=Window(c - HALO, r - HALO, w + 2 * HALO, h + 2 * HALO), boundless=True, fill_value=np.nan); ok = np.isfinite(E); E0 = np.where(ok, E, 0)
            for k, n in (('tpi300', 21), ('tpi2k', 135)):
                m = uniform_filter(E0, n, mode='constant') / np.maximum(uniform_filter(ok.astype(np.float32), n, mode='constant'), 1e-6)
                outs[k].write((E - m)[HALO:HALO + h, HALO:HALO + w].astype(np.float32), 1, window=win)
            v = ev.read(1, window=win); tc.write(np.where((v >= 110) & (v <= 199), v - 100, 0).astype(np.uint8), 1, window=win)
        print('row', r, f'{time.time() - t0:.0f}s', flush=True)
    for o in list(outs.values()) + [tc]: o.close()
    print('done', f'{time.time() - t0:.0f}s')
