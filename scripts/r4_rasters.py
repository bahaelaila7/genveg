"""Region 4 covariate rasters for the genveg table (docs/GENVEG_V4_R4.md §B), mirroring the GA+FL scripts on the Region 4 grids, into data/r4/:
  hansen_dl  Hansen GFC 2024 v1.12 granules for the Region 4 box (adds 30N_100W, 40N_100W to data/hansen/raw)          (03_hansen_dl.py)
  hansen     lossyear (mode), treecover2000 (average), datamask (nearest) -> grid_r4 30 m (255 outside the AOI)          (03_hansen.py)
  dist       LANDFIRE annual disturbance 1999-2025 (CONUS archive) cropped to grid_r4 (aligned lattice), int16 code       (04c_lf_dist.py)
  land       forest-land classes 0/1/2/3/4/9 with the rule of 10_forest_land.py and config/pilot.yml forest_land (TreeMap 2022 CONUS,
             LANDFIRE EVT 2022 CONUS, Hansen); NOT speckle-cleaned (11_land_clean.py has no Region 4 run): the table reads only footprint centres
  soils90    SOLUS 0-30 cm depth-weighted layers + restriction / lithic depth, bilinear -> grid_r4_90 (= the 30 m centre-pixel value)   (05_soils.py)
  soilfill   Region 4 forest-land soil medians (g02b nodata fill) -> data/r4/soils90/fill.json
  .venv/bin/python -I scripts/r4_rasters.py <hansen_dl|hansen|dist|land|soils90|soilfill> [nproc]"""
import os as _os; _SM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _WS = _os.environ.get('WORKSPACE', _os.path.dirname(_SM))   # relocatable paths
import sys; sys.path.insert(0, _SM + '/scripts')
import os, re, glob, json, math, time, zipfile, tempfile, yaml, numpy as np, pandas as pd, rasterio, requests
from rasterio.windows import Window, from_bounds, transform as wtransform
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from multiprocessing import Pool
from common import ROOT, PROFILE
O = f'{ROOT}/data/r4'; os.makedirs(O, exist_ok=True)
G = json.load(open(f'{ROOT}/data/grid/grid_r4.json')); TR = Affine(*G['transform']); W, H = G['width'], G['height']
G9 = json.load(open(f'{ROOT}/data/grid/grid_r4_90.json')); TR9 = Affine(*G9['transform']); W9, H9 = G9['width'], G9['height']
CRS = 'EPSG:5070'
def aoi30(r, c, h, w):
    """AOI flag of 30 m pixels (from the 90 m mask; the AOI is defined at 90 m)"""
    r9, c9 = r // 3, c // 3; h9 = (r + h - 1) // 3 - r9 + 1; w9 = (c + w - 1) // 3 - c9 + 1
    with rasterio.open(f'{ROOT}/data/grid/aoi_r4_mask90.tif') as m: a = m.read(1, window=Window(c9, r9, w9, h9), boundless=True, fill_value=0)
    a = np.repeat(np.repeat(a, 3, 0), 3, 1); return a[r - 3 * r9:r - 3 * r9 + h, c - 3 * c9:c - 3 * c9 + w] == 1
# ---------------------------------------------------------------- Hansen
HP = {'lossyear': Resampling.mode, 'treecover2000': Resampling.average, 'datamask': Resampling.nearest}
def hansen_dl():
    for la in (30, 40):
        for lo in (100, 90, 80):
            for p in HP:
                n = f'Hansen_GFC-2024-v1.12_{p}_{la:02d}N_{lo:03d}W.tif'; d = f'{ROOT}/data/hansen/raw/{n}'
                if os.path.exists(d): continue
                r = requests.get(f'https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/{n}', stream=True, timeout=600)
                print(n, r.status_code, flush=True)
                if r.status_code != 200: continue
                with open(d + '.part', 'wb') as f:
                    for ch in r.iter_content(1 << 22): f.write(ch)
                os.rename(d + '.part', d)
T = 4096
def hansen_work(a):
    prod, r, c = a; h = min(T, H - r); w = min(T, W - c); wt = wtransform(Window(c, r, w, h), TR)
    m = aoi30(r, c, h, w)
    if not m.any(): return prod, r, c, None
    out = np.full((h, w), 255, np.uint8)
    for f in sorted(glob.glob(f'{ROOT}/data/hansen/raw/Hansen_GFC-2024-v1.12_{prod}_*.tif')):
        with rasterio.open(f) as s:
            reproject(rasterio.band(s, 1), out, dst_transform=wt, dst_crs=CRS, resampling=HP[prod], init_dest_nodata=False, dst_nodata=255, src_nodata=None)
    out[~m] = 255; return prod, r, c, out
def hansen(npr):
    for prod in HP:
        prof = dict(PROFILE, width=W, height=H, count=1, dtype='uint8', crs=CRS, transform=TR, nodata=255)
        jobs = [(prod, r, c) for r in range(0, H, T) for c in range(0, W, T)]
        with rasterio.open(f'{O}/hansen_{prod}.tif', 'w', **prof) as d, Pool(npr) as p:
            for pr, r, c, o in p.imap_unordered(hansen_work, jobs):
                if o is not None: d.write(o, 1, window=Window(c, r, o.shape[1], o.shape[0]))
        print(prod, 'done', flush=True)
# ---------------------------------------------------------------- LANDFIRE annual disturbance
def dist():
    SRC = f'{ROOT}/data/landfire/raw/annual_dist/USAnnualDisturbance_1999_present.zip'; OD = f'{O}/dist'; os.makedirs(OD, exist_ok=True)
    outer = zipfile.ZipFile(SRC)
    for name in sorted(n for n in outer.namelist() if n.endswith('_CONUS.zip')):
        yy = int(re.search(r'Dist(\d\d)_CONUS', name).group(1)); year = 1900 + yy if yy >= 90 else 2000 + yy; fn = f'{OD}/dist_{year}.tif'
        if os.path.exists(fn): continue
        with tempfile.TemporaryDirectory(dir=f'{ROOT}/data/landfire/raw') as td:
            inner = os.path.join(td, 'y.zip')
            with outer.open(name) as s, open(inner, 'wb') as d:
                while (b := s.read(1 << 24)): d.write(b)
            z = zipfile.ZipFile(inner); tif = [m for m in z.namelist() if m.lower().endswith('.tif')][0]
            with rasterio.open(f'/vsizip/{inner}/{tif}') as s:
                col = round((TR.c - s.transform.c) / 30); row = round((s.transform.f - TR.f) / 30)
                assert abs((TR.c - s.transform.c) / 30 - col) < 1e-6 and abs((s.transform.f - TR.f) / 30 - row) < 1e-6, 'grid not aligned'
                prof = dict(PROFILE, width=W, height=H, count=1, dtype='int16', crs=CRS, transform=TR, nodata=None)
                n = 0
                with rasterio.open(fn + '.part', 'w', **prof) as o:
                    for r in range(0, H, 2048):
                        h = min(2048, H - r)
                        a = s.read(1, window=Window(col, row + r, W, h), boundless=True, fill_value=0).astype(np.int32)   # (as 04c_lf_dist.py)
                        a[(a < 0) | (a == s.nodata if s.nodata is not None else False)] = 0
                        a[~aoi30(r, 0, h, W)] = 0; n += int((a > 0).sum()); o.write(a.astype(np.int16), 1, window=Window(0, r, W, h))
                os.replace(fn + '.part', fn)
        print(year, 'disturbed px', n, flush=True)
# ---------------------------------------------------------------- forest-land classes (10_forest_land.py rule)
def land():
    cfg = yaml.safe_load(open(f'{ROOT}/config/pilot.yml'))['forest_land']
    evt = pd.read_csv(f'{ROOT}/data/landfire/evt_classes.csv').rename(columns={'value': 'VALUE'}); evt = evt[evt.VALUE >= 0].reset_index(drop=True)
    lut_size = int(max(evt.VALUE.max(), 32767)) + 1
    def lut(mask): t = np.zeros(lut_size, bool); t[evt.VALUE[mask].values] = True; return t
    excl = evt.EVT_LF.isin(cfg['exclude_evt_lf']) | evt.EVT_PHYS.str.startswith(tuple(cfg['exclude_evt_phys_prefix']))
    L_EXCL = lut(excl); L_TREE = lut((evt.EVT_LF == 'Tree') & ~excl)
    L_DIST = lut(evt.EVT_NAME.str.contains('|'.join(cfg['disturbed_evt_name'])) & ~excl); L_SHRUBHERB = lut(evt.EVT_LF.isin(['Shrub', 'Herb']) & ~excl)
    tm = rasterio.open(_WS + '/FVSclaude/treemap/Data/TreeMap2022_CONUS.tif'); ev_ = rasterio.open(f'{ROOT}/data/landfire/raw/LF2022_EVT_CONUS/Tif/LF2022_EVT_CONUS.tif')
    off = {k: (round((TR.c - s.transform.c) / 30), round((s.transform.f - TR.f) / 30)) for k, s in (('tm', tm), ('evt', ev_))}
    hs = {k: rasterio.open(f'{O}/hansen_{k}.tif') for k in HP}
    prof = dict(PROFILE, width=W, height=H, count=1, dtype='uint8', crs=CRS, transform=TR, nodata=None)
    out = rasterio.open(f'{O}/land.tif', 'w', **prof); counts = np.zeros(10, np.int64); rf = cfg['recent_loss_from'] - 2000; t0 = time.time()
    for r in range(0, H, 1024):
        h = min(1024, H - r); w = Window(0, r, W, h)
        aoi = aoi30(r, 0, h, W)
        if not aoi.any(): out.write(np.zeros((h, W), np.uint8), 1, window=w); continue
        t_ = tm.read(1, window=Window(off['tm'][0], off['tm'][1] + r, W, h), boundless=True, fill_value=0)
        ev = ev_.read(1, window=Window(off['evt'][0], off['evt'][1] + r, W, h), boundless=True, fill_value=-1).astype(np.int64); ev[(ev < 0) | (ev >= lut_size)] = 0
        loss = hs['lossyear'].read(1, window=w); tc = hs['treecover2000'].read(1, window=w); land_ok = hs['datamask'].read(1, window=w) == 1
        loss = np.where(loss == 255, 0, loss); tc = np.where(tc == 255, 0, tc)
        in_tm = (t_ != tm.nodata) & (t_ != 0)
        excl_ = L_EXCL[ev]; former = tc >= cfg['treecover2000_min']
        recent = (loss >= rf) & former; older = (loss > 0) & (loss < rf) & former
        c = np.zeros((h, W), np.uint8)
        c[L_TREE[ev] & ~in_tm] = 4
        c[~in_tm & ~excl_ & (older | L_DIST[ev] | (L_SHRUBHERB[ev] & (tc >= cfg['shrub_treecover2000_min'])))] = 3
        c[in_tm] = 1; c[recent & ~excl_] = 2; c[(loss > 0) & former & excl_ & ~in_tm] = 9
        c[~aoi | ~land_ok] = 0
        out.write(c, 1, window=w); counts += np.bincount(c.ravel(), minlength=10)
        if (r // 1024) % 8 == 0: print(f'row {r}/{H} {time.time() - t0:.0f}s', flush=True)
    out.close(); names = {0: 'none', 1: 'forest', 2: 'harvested', 3: 'regenerating', 4: 'forest_unmapped', 9: 'converted'}
    rep = {names[k]: dict(px=int(counts[k]), km2=round(counts[k] * 900 / 1e6, 1)) for k in names}
    json.dump(rep, open(f'{O}/land_summary.json', 'w'), indent=1); print(json.dumps(rep, indent=1))
# ---------------------------------------------------------------- soils at 90 m (bilinear = the 30 m centre value)
S = _WS + '/SOLUS/SOLUS100/'; DEP = [(0, 5), (5, 10), (15, 15)]
LAYERED = {'clay030': ('claytotal_{d}_cm_p', 1.0), 'sand030': ('sandtotal_{d}_cm_p', 1.0), 'soc030': ('soc_{d}_cm_p', 1000.0),
           'ph030': ('ph1to1h2o_{d}_cm_p', 100.0), 'awc030': ('hydro/derived/awc_{d}_cm', None)}
SINGLE = {'restrdepth': 'resdept_all_cm_p', 'lithdepth': 'anylithicdpt_cm_p'}
T9 = 2048
def rd(s, win, scale):
    a = s.read(1, window=win, boundless=True).astype('float32'); nd = s.nodata
    a = np.where(a == nd, np.nan, a) if nd is not None else a
    return a / scale if scale else a
def soils_work(a):
    r, c = a; h = min(T9, H9 - r); w = min(T9, W9 - c); wt = wtransform(Window(c, r, w, h), TR9)
    with rasterio.open(f'{ROOT}/data/grid/aoi_r4_mask90.tif') as m: mm = m.read(1, window=Window(c, r, w, h))
    if mm.sum() == 0: return r, c, None
    L, B, R, Tp = wt.c, wt.f + wt.e * h, wt.c + wt.a * w, wt.f
    with rasterio.open(S + 'claytotal_0_cm_p.tif') as s0:
        win = from_bounds(L - 300, B - 300, R + 300, Tp + 300, s0.transform).round_offsets().round_lengths(); st = s0.window_transform(win)
    def warp(src):
        d = np.full((h, w), np.nan, np.float32)
        reproject(src, d, src_transform=st, src_crs=CRS, dst_transform=wt, dst_crs=CRS, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        d[mm == 0] = np.nan; return d
    res = {}
    for k, (pat, sc) in LAYERED.items():
        acc = 0; wsum = 0
        for top, th in DEP:
            with rasterio.open(S + pat.format(d=top) + '.tif') as s: a_ = rd(s, win, sc)
            acc = acc + np.nan_to_num(a_) * th; wsum = wsum + np.where(np.isnan(a_), 0, th)
        res[k] = warp(np.where(wsum > 0, acc / np.maximum(wsum, 1), np.nan).astype('float32'))
    for k, f in SINGLE.items():
        with rasterio.open(S + f + '.tif') as s: res[k] = warp(rd(s, win, None))
    return r, c, res
def soils90(npr):
    OD = f'{O}/soils90'; os.makedirs(OD, exist_ok=True); names = list(LAYERED) + list(SINGLE)
    prof = dict(PROFILE, width=W9, height=H9, count=1, dtype='float32', crs=CRS, transform=TR9, nodata=-9999.0, predictor=3)
    outs = {n: rasterio.open(f'{OD}/{n}.tif', 'w', **prof) for n in names}
    with Pool(npr) as p:
        for r, c, res in p.imap_unordered(soils_work, [(r, c) for r in range(0, H9, T9) for c in range(0, W9, T9)]):
            if res is None: continue
            for n, v in res.items(): outs[n].write(np.where(np.isnan(v), -9999.0, v).astype('float32'), 1, window=Window(c, r, v.shape[1], v.shape[0]))
    for o in outs.values(): o.close()
    print('soils90 done', flush=True)
def soilfill():
    """Region 4 forest-land medians of the soil layers (the g02b nodata fill, as SOIL_FILL holds the GA+FL ones): 90 m pixels whose block-centre 30 m
    land class is forest land (1-4) and whose soil value is valid -> data/r4/soils90/fill.json"""
    from features import SOILS
    lnd = rasterio.open(f'{O}/land.tif'); so = {k: rasterio.open(f'{O}/soils90/{k}.tif') for k in SOILS}; vals = {k: [] for k in SOILS}
    for r9 in range(0, H9, 1024):
        h9 = min(1024, H9 - r9); L = lnd.read(1, window=Window(0, 3 * r9, W, 3 * h9), boundless=True, fill_value=0)[1::3, 1::3][:h9, :W9]
        f = (L >= 1) & (L <= 4)
        for k in SOILS:
            v = so[k].read(1, window=Window(0, r9, W9, h9)); v = v[f & (v > -9000)]; vals[k].append(v[::7])          # every 7th pixel (median)
    fill = {k: round(float(np.median(np.concatenate(v))), 4) for k, v in vals.items()}
    json.dump(fill, open(f'{O}/soils90/fill.json', 'w'), indent=1); print('soil fill', fill, flush=True)
if __name__ == '__main__':
    st = sys.argv[1]; npr = int(sys.argv[2]) if len(sys.argv) > 2 else 2; t0 = time.time()
    dict(hansen_dl=lambda: hansen_dl(), hansen=lambda: hansen(npr), dist=lambda: dist(), land=lambda: land(), soils90=lambda: soils90(npr), soilfill=lambda: soilfill())[st]()
    print(st, 'finished', round(time.time() - t0), 's', flush=True)
