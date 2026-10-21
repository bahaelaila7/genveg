"""Forest land + status on the TreeMap grid, keeping clear-cut (temporarily non-forest) land.
Outputs data/derived/land.tif (uint8 class) and data/derived/loss_year.tif (uint8, 0 = none, else 2000+v).
Classes:
  0  not forest land (outside AOI, developed, agriculture, water, barren, natural non-forest)
  1  forest            TreeMap 2022 forest pixel, no stand-replacing loss since recent_loss_from
  2  harvested         recent loss (>= recent_loss_from) on former forest (treecover2000 >= min), not converted
  3  regenerating      not mapped by TreeMap, former forest: older loss (2001..recent-1), LANDFIRE recently-disturbed /
                       plantation / ruderal-forest type, or shrub/herb with high treecover2000 (cut before 2001)
  4  forest_unmapped   LANDFIRE tree life form, no TreeMap imputation, no loss
  9  converted         loss on former forest now developed/agriculture/water (reported, not forest land)
Precedence (as implemented; documented 2026-10-22 after gv_verify_upstream U6): 4, then 3 (outside TreeMap, not excluded), then 1 (any TreeMap
pixel), then 2 (recent loss on former forest, EVT not excluded — overrides 1), then 9 (loss on former forest, excluded EVT, outside TreeMap);
finally 0 outside the AOI and on Hansen datamask non-land (water). A TreeMap pixel with recent loss but an excluded 2022 EVT therefore stays 1.
"""
import sys, os, json, time, yaml, numpy as np, pandas as pd, rasterio
from rasterio.windows import Window
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, PROFILE

cfg = yaml.safe_load(open(f'{ROOT}/config/pilot.yml'))['forest_land']
os.makedirs(f'{ROOT}/data/derived', exist_ok=True)
evt = pd.read_csv(f'{ROOT}/data/landfire/evt_classes.csv').rename(columns={'value': 'VALUE'})
evt = evt[evt.VALUE >= 0].reset_index(drop=True)   # -9999 Fill-NoData row (raster nodata is handled below)
lut_size = int(max(evt.VALUE.max(), 32767)) + 1
def lut(mask):
    t = np.zeros(lut_size, bool); t[evt.VALUE[mask].values] = True; return t
excl = evt.EVT_LF.isin(cfg['exclude_evt_lf']) | evt.EVT_PHYS.str.startswith(tuple(cfg['exclude_evt_phys_prefix']))
L_EXCL = lut(excl)
L_TREE = lut((evt.EVT_LF == 'Tree') & ~excl)
L_DIST = lut(evt.EVT_NAME.str.contains('|'.join(cfg['disturbed_evt_name'])) & ~excl)
L_SHRUBHERB = lut(evt.EVT_LF.isin(['Shrub', 'Herb']) & ~excl)

src = {k: rasterio.open(f'{ROOT}/data/{p}') for k, p in dict(
    tm='treemap/tm_id.tif', aoi='grid/aoi_mask.tif', evt='landfire/evt2022.tif',
    loss='hansen/lossyear.tif', tc='hansen/treecover2000.tif', dm='hansen/datamask.tif').items()}
ref = src['tm']
for k, s in src.items():
    assert s.transform == ref.transform and s.shape == ref.shape, f'{k} not on grid'
H, W = ref.shape
prof = dict(PROFILE, width=W, height=H, count=1, dtype='uint8', crs=ref.crs, transform=ref.transform, nodata=None)
out = rasterio.open(f'{ROOT}/data/derived/land.tif', 'w', **prof)
outl = rasterio.open(f'{ROOT}/data/derived/loss_year.tif', 'w', **prof)
counts = np.zeros(10, np.int64); t0 = time.time()
rf = cfg['recent_loss_from'] - 2000
for r in range(0, H, 2048):
    w = Window(0, r, W, min(2048, H - r))
    tm = src['tm'].read(1, window=w); aoi = src['aoi'].read(1, window=w) == 1
    ev = src['evt'].read(1, window=w).astype(np.int64); ev[(ev < 0) | (ev >= lut_size)] = 0
    loss = src['loss'].read(1, window=w); tc = src['tc'].read(1, window=w)
    land_ok = src['dm'].read(1, window=w) == 1
    in_tm = (tm != ref.nodata) & (tm != 0)
    excl_ = L_EXCL[ev]; former = tc >= cfg['treecover2000_min']
    recent = (loss >= rf) & former; older = (loss > 0) & (loss < rf) & former
    c = np.zeros(tm.shape, np.uint8)
    c[L_TREE[ev] & ~in_tm] = 4
    c[~in_tm & ~excl_ & (older | L_DIST[ev] | (L_SHRUBHERB[ev] & (tc >= cfg['shrub_treecover2000_min'])))] = 3
    c[in_tm] = 1
    c[recent & ~excl_] = 2
    c[(loss > 0) & former & excl_ & ~in_tm] = 9
    c[~aoi | ~land_ok] = 0
    out.write(c, 1, window=w); outl.write(np.where(aoi, loss, 0).astype(np.uint8), 1, window=w)
    counts += np.bincount(c.ravel(), minlength=10)
    print(f'row {r}/{H} {time.time()-t0:.0f}s', flush=True)
out.close(); outl.close()
names = {0: 'none', 1: 'forest', 2: 'harvested', 3: 'regenerating', 4: 'forest_unmapped', 9: 'converted'}
rep = {names[k]: dict(px=int(counts[k]), km2=round(counts[k] * 900 / 1e6, 1)) for k in names}
json.dump(rep, open(f'{ROOT}/data/derived/land_summary.json', 'w'), indent=1); print(json.dumps(rep, indent=1))
