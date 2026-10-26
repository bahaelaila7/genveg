"""Independent verification of the UPSTREAM pipeline LOGIC (user, 2026-10-22: "check the correctness of the pipeline"; checking the downloads
themselves is unnecessary): grid alignment of every derived raster, the forest-land classes against their written rules, the stands raster against
the polygons, the stand-mapping internals (g05 landscape nodes), and the Forest Explorer display. PASS / FAIL per check; exit 1 on any failure.
  python scripts/gv_verify_upstream.py [n_points=200]"""
import sys, os, json, math, zipfile, re, numpy as np, pandas as pd, rasterio, yaml
from rasterio.windows import Window
from rasterio.transform import Affine
from pyproj import Transformer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from stand_runs import stands_run
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); WS = os.environ.get('WORKSPACE', os.path.dirname(ROOT)); D = f"{ROOT}/data/proto/{os.environ.get('DATA', 'gafl')}"
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 200; rng = np.random.default_rng(7); RES = []
def check(name, ok, detail=''):
    RES.append((name, bool(ok))); print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
G = json.load(open(f'{ROOT}/data/grid/grid.json')); TR = Affine(*G['transform']); CRS = G['crs']; W, H = G['width'], G['height']
CFG = yaml.safe_load(open(f"{stands_run()['dir']}/config_resolved.yml"))   # the rules the stand run was built with
def dq(a): v = a.astype(np.float32); o = np.sign(v) * (v / 127.5) ** 2; o[a == -128] = np.nan; return o
# ======================= U1 every derived raster on the grid of grid.json
SRN = stands_run()                                       # the stand-delineation run (STANDS_RUN) the generative-vegetation stages use
fs = {'land_clean': SRN['land_clean'], 'land': SRN['land'], 'stands': SRN['tif'], 'aef_2022': 'aef/aef_2022.tif', 'aef_2025': 'aef/aef_2025.tif',
      'lossyear': 'hansen/lossyear.tif', 'treecover2000': 'hansen/treecover2000.tif', 'evt2022': 'landfire/evt2022.tif', 'tm_id': 'treemap/tm_id.tif',
      **{f'dist_{y}': f'landfire/dist/dist_{y}.tif' for y in (2000, 2012, 2024, 2025)}, **{k: f'soils/{k}.tif' for k in ('clay030', 'sand030', 'soc030', 'awc030', 'ph030', 'restrdepth')}}
bad = [k for k, p in fs.items() if (lambda s: not (s.transform.almost_equals(TR) and s.width == W and s.height == H and str(s.crs) == str(rasterio.crs.CRS.from_user_input(CRS))))(rasterio.open(p if os.path.isabs(p) else f'{ROOT}/data/{p}'))]
check('U1 all derived rasters share the grid of grid.json (transform, size, CRS)', not bad, f'{len(fs)} rasters; off-grid: {bad}')
land = rasterio.open(SRN['land_clean'])
# random forest-land pixels (sampled in blocks to keep reads cheap)
pts = []
while len(pts) < NP:
    r0, c0 = int(rng.integers(0, H - 512)), int(rng.integers(0, W - 512)); L = land.read(1, window=Window(c0, r0, 512, 512)); ix = np.argwhere((L >= 1) & (L <= 4))
    if len(ix): k = ix[rng.integers(len(ix))]; pts.append((r0 + int(k[0]), c0 + int(k[1])))
pts = np.array(pts); XY = np.array([TR * (c + 0.5, r + 0.5) for r, c in pts]); LL = np.array(Transformer.from_crs(CRS, 4326, always_xy=True).transform(XY[:, 0], XY[:, 1])).T
# (the prepared rasters are not re-checked against their downloaded sources: user, 2026-10-22, 'unnecessary')
# ======================= U6 forest-land classes re-derived from the WRITTEN rules (10_forest_land docstring + config), on land.tif (before speckle cleaning)
fl = CFG['forest_land']; evt = pd.read_csv(f'{ROOT}/data/landfire/evt_classes.csv').rename(columns={'value': 'VALUE'}); evt = evt[evt.VALUE >= 0].set_index('VALUE')
excl = evt.EVT_LF.isin(fl['exclude_evt_lf']) | evt.EVT_PHYS.astype(str).str.startswith(tuple(fl['exclude_evt_phys_prefix']))
dist_ = evt.EVT_NAME.astype(str).str.contains('|'.join(fl['disturbed_evt_name'])); tree_ = evt.EVT_LF == 'Tree'; sh_ = evt.EVT_LF.isin(['Shrub', 'Herb'])
src = {k: rasterio.open(p if os.path.isabs(p) else f'{ROOT}/data/{p}') for k, p in dict(land=SRN['land'], tm='treemap/tm_id.tif', evt='landfire/evt2022.tif', loss='hansen/lossyear.tif',
                                                          tc='hansen/treecover2000.tif', aoi='grid/aoi_mask.tif', dm='hansen/datamask.tif').items()}
conf = {}; pts6 = [(int(rng.integers(0, H)), int(rng.integers(0, W))) for _ in range(4000)]
for r, c in pts6:
    v = {k: s.read(1, window=Window(c, r, 1, 1))[0, 0] for k, s in src.items()}
    if not v['aoi']: continue
    e = int(v['evt']); ex = bool(excl.get(e, False)); tm_ok = int(v['tm']) > 0 and (src['tm'].nodata is None or v['tm'] != src['tm'].nodata)
    ly = int(v['loss']); ly = 2000 + ly if 0 < ly < 100 else 0; former = int(v['tc']) >= fl['treecover2000_min']; recent = ly >= fl['recent_loss_from']; older = 0 < ly < fl['recent_loss_from']
    if int(v['dm']) != 1: exp = 0                                  # Hansen datamask non-land (documented after the first run)
    elif ly and former and ex and not tm_ok: exp = 9
    elif tm_ok: exp = 2 if (recent and former and not ex) else 1
    elif ex: exp = 0
    elif recent and former: exp = 2
    elif (older and former) or bool(dist_.get(e, False)) or (bool(sh_.get(e, False)) and int(v['tc']) >= fl['shrub_treecover2000_min']): exp = 3
    elif bool(tree_.get(e, False)): exp = 4
    else: exp = 0
    conf[(exp, int(v['land']))] = conf.get((exp, int(v['land'])), 0) + 1
n6 = sum(conf.values()); agree6 = sum(n for (a, b), n in conf.items() if a == b) / max(n6, 1)
check('U6 forest-land classes (land.tif) = the written rules re-derived from TreeMap / EVT / Hansen at random AOI pixels', agree6 > 0.99,
      f'{n6} pixels, agreement {agree6:.4f}; disagreements (rule, land.tif): { {k: v for k, v in conf.items() if k[0] != k[1]} }')
# ======================= U1r / U6r Region 4 rasters (AEF90 = 1, docs/GENVEG_V4_R4.md §B): grids, and the forest-land classes of data/r4/land.tif re-derived
#                          from the written rules at random AOI pixels, reading the CONUS TreeMap / EVT by COORDINATES (r4_rasters.land uses pixel offsets)
AEF90 = os.environ.get('AEF90', '0') == '1'
if AEF90:
    G4 = json.load(open(f'{ROOT}/data/grid/grid_r4.json')); G49 = json.load(open(f'{ROOT}/data/grid/grid_r4_90.json')); TR4, TR49 = Affine(*G4['transform']), Affine(*G49['transform'])
    f30 = ['r4/land.tif', 'r4/hansen_lossyear.tif', 'r4/hansen_treecover2000.tif', 'r4/hansen_datamask.tif', *[f'r4/dist/dist_{y}.tif' for y in (2000, 2012, 2024, 2025)]]
    f90 = ['grid/aoi_r4_mask90.tif', 'aef90/aef90_2022.tif', 'aef90/aef90_2025.tif', *[f'r4/soils90/{k}.tif' for k in ('clay030', 'sand030', 'soc030', 'awc030', 'ph030', 'restrdepth')]]
    on = lambda p_, t_, g_: (lambda s_: s_.transform.almost_equals(t_) and s_.width == g_['width'] and s_.height == g_['height'] and s_.crs.to_epsg() == 5070)(rasterio.open(f'{ROOT}/data/{p_}'))
    bad = [p_ for p_ in f30 if not on(p_, TR4, G4)] + [p_ for p_ in f90 if not on(p_, TR49, G49)]
    lat = (TR4.c - TR.c) / 30, (TR.f - TR4.f) / 30
    check('U1r Region 4 rasters on grid_r4 (30 m) / grid_r4_90 (90 m); grid_r4 on the TreeMap lattice of grid.json, a multiple of 3 px off', not bad and all(abs(v - round(v)) < 1e-6 and round(v) % 3 == 0 for v in lat),
          f'{len(f30) + len(f90)} rasters; off-grid {bad}; offset {lat}')
    flr = yaml.safe_load(open(f'{ROOT}/config/pilot.yml'))['forest_land']        # the rules r4_rasters.land applies
    excl4 = evt.EVT_LF.isin(flr['exclude_evt_lf']) | evt.EVT_PHYS.astype(str).str.startswith(tuple(flr['exclude_evt_phys_prefix']))
    dist4 = evt.EVT_NAME.astype(str).str.contains('|'.join(flr['disturbed_evt_name']))
    s4 = dict(land=rasterio.open(f'{ROOT}/data/r4/land.tif'), loss=rasterio.open(f'{ROOT}/data/r4/hansen_lossyear.tif'), tc=rasterio.open(f'{ROOT}/data/r4/hansen_treecover2000.tif'),
              dm=rasterio.open(f'{ROOT}/data/r4/hansen_datamask.tif'), aoi=rasterio.open(f'{ROOT}/data/grid/aoi_r4_mask90.tif'))
    tmC = rasterio.open(WS + '/FVSclaude/treemap/Data/TreeMap2022_CONUS.tif'); evC = rasterio.open(f'{ROOT}/data/landfire/raw/LF2022_EVT_CONUS/Tif/LF2022_EVT_CONUS.tif')
    conf4 = {}; n4 = 0
    am = s4['aoi'].read(1); ia = np.flatnonzero(am.ravel() == 1); del am           # AOI 90 m pixels -> a random 30 m pixel inside each sampled block
    pk = rng.choice(ia, 4000, replace=False); P4 = [(3 * (q // G49['width']) + int(rng.integers(0, 3)), 3 * (q % G49['width']) + int(rng.integers(0, 3))) for q in pk]
    P4.sort()                                                                          # row order: the strip-organised TreeMap CONUS re-uses its decoded rows
    def px(src, x_, y_, fill):                                                         # one pixel by coordinates (plain window read; boundless reads build a VRT: slow)
        r_, c_ = src.index(x_, y_)
        return src.read(1, window=Window(c_, r_, 1, 1))[0, 0] if 0 <= r_ < src.height and 0 <= c_ < src.width else fill
    for r, c in P4:
        n4 += 1; x_, y_ = TR4 * (c + 0.5, r + 0.5)
        v = {k: s_.read(1, window=Window(c, r, 1, 1))[0, 0] for k, s_ in s4.items() if k != 'aoi'}
        tmv = px(tmC, x_, y_, 0); e = int(px(evC, x_, y_, -1))
        tm_ok = int(tmv) > 0 and (tmC.nodata is None or tmv != tmC.nodata); ex = bool(excl4.get(e, False))
        ly = int(v['loss']); ly = 2000 + ly if 0 < ly < 100 else 0; tcv = 0 if int(v['tc']) == 255 else int(v['tc'])
        former = tcv >= flr['treecover2000_min']; recent = ly >= flr['recent_loss_from']; older = 0 < ly < flr['recent_loss_from']
        if int(v['dm']) != 1: exp = 0
        elif ly and former and ex and not tm_ok: exp = 9
        elif tm_ok: exp = 2 if (recent and former and not ex) else 1
        elif ex: exp = 0
        elif recent and former: exp = 2
        elif (older and former) or bool(dist4.get(e, False)) or (bool(sh_.get(e, False)) and tcv >= flr['shrub_treecover2000_min']): exp = 3
        elif bool(tree_.get(e, False)): exp = 4
        else: exp = 0
        conf4[(exp, int(v['land']))] = conf4.get((exp, int(v['land'])), 0) + 1
    ag4 = sum(n for (a, b), n in conf4.items() if a == b) / n4
    check('U6r Region 4 forest-land classes (data/r4/land.tif) = the written rules re-derived (CONUS TreeMap / EVT by coordinates, R4 Hansen) at random AOI pixels', ag4 > 0.99,
          f'{n4} pixels, agreement {ag4:.4f}; disagreements (rule, land.tif): { {k: v for k, v in conf4.items() if k[0] != k[1]} }')
# ======================= U7 stands raster vs the stand polygons (gpkg): pixel centre inside its stand's polygon; polygon area ≈ pixel area
import geopandas as gpd
from shapely.geometry import Point
st = gpd.read_file(SRN['gpkg'], layer='stands').set_index('stand_id'); sr = rasterio.open(SRN['tif']); ins, n7 = 0, 0
for _ in range(300):
    sid = int(rng.choice(st.index.values)); geom = st.loc[sid].geometry; x, y = geom.representative_point().coords[0]; r, c = sr.index(x, y)
    v = int(sr.read(1, window=Window(c, r, 1, 1))[0, 0]); n7 += 1; ins += int(v == sid)
sa = st.sample(60, random_state=1); arr = []
for sid, row in sa.iterrows():
    x0, y0, x1, y1 = row.geometry.bounds; r0, c0 = sr.index(x0, y1); r1, c1 = sr.index(x1, y0)
    A = sr.read(1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)); arr.append(abs((A == sid).sum() * 900 / row.geometry.area - 1))
check('U7 stand polygons: a point inside each polygon falls on its own id in stands.tif; polygon area = its pixel count x 900 m2', ins / n7 > 0.99 and np.median(arr) < 0.02,
      f'{n7} polygons: id match {ins / n7:.3f}; |area ratio − 1| median {np.median(arr):.4f}, max {max(arr):.3f}')
# ======================= U8 stand-mapping internals (g05): each stand's own landscape (disk at its centroid) recomputed from the rasters with independent code (full disk; g05 uses a
#                         uniform 256-subsample, so compared as an unbiased estimate: signed errors across nodes)
if os.environ.get('GV_SKIP_U8') == '1':                     # genveg_pipeline stage vup runs before the stand mapping; U8 runs in stage verify
    print('   (U8 skipped here: it checks the stand mapping, run after it in the verify stage)')
elif os.path.exists(f'{D}/stand_nodes.npz') and 'node' in pd.read_parquet(f'{D}/stand_veg_v3.parquet').columns:
    import pickle
    NZ = np.load(f'{D}/stand_nodes.npz'); km = pickle.load(open(f'{D}/bayes_kmeans.pkl', 'rb')); a22, a25 = rasterio.open(f'{ROOT}/data/aef/aef_2022.tif'), rasterio.open(f'{ROOT}/data/aef/aef_2025.tif')
    if AEF90:       # Region 4 features: footprints = 90 m blocks of grid_r4 (centres 1 mod 3), AlphaEarth = the 90 m pixel; stand pixels shifted by the grid offset
        a22, a25 = rasterio.open(f'{ROOT}/data/aef90/aef90_2022.tif'), rasterio.open(f'{ROOT}/data/aef90/aef90_2025.tif'); land4 = rasterio.open(f'{ROOT}/data/r4/land.tif')
        st_tr = rasterio.open(SRN['tif']).transform; oR, oC = round((land4.transform.f - st_tr.f) / 30), round((st_tr.c - land4.transform.c) / 30)
    RP = int(1600 // 30) + 2; dd8 = []; sv = pd.read_parquet(f'{D}/stand_veg_v3.parquet')
    Ssub = pd.read_parquet(f'{D}/stands.parquet'); cen = Ssub.groupby('stand_id')[['row', 'col']].mean()
    okn = (np.abs(NZ['nodes'][sv.node.values] - np.rint(cen.loc[sv.stand_id.values].values)).max(1) == 0).mean()   # each stand's landscape is centred on its own centroid
    for i in rng.choice(len(NZ['nodes']), 20, replace=False):
        r, c = NZ['nodes'][i]
        if AEF90:
            R_, C_ = r + oR, c + oC; ar_ = np.arange(R_ - RP - 3, R_ + RP + 4); ac_ = np.arange(C_ - RP - 3, C_ + RP + 4); ar_, ac_ = ar_[ar_ % 3 == 1], ac_[ac_ % 3 == 1]
            AR, AC = np.meshgrid(ar_, ac_, indexing='ij'); AR, AC = AR.ravel(), AC.ravel()
            L = land4.read(1, window=Window(C_ - RP - 3, R_ - RP - 3, 2 * RP + 7, 2 * RP + 7), boundless=True, fill_value=0)[AR - (R_ - RP - 3), AC - (C_ - RP - 3)]
            m = (L >= 1) & (L <= 4) & (np.hypot(AR - R_, AC - C_) * 30 <= 1600); AR, AC = AR[m], AC[m]
            if len(AR) < 20: continue
            def p9(A): return np.stack([np.nan_to_num(dq(A.read(window=Window(x // 3, y // 3, 1, 1))).reshape(64)) for y, x in zip(AR, AC)])
            e22, e25 = p9(a22), p9(a25); rr = AR
        else:
            L = land.read(1, window=Window(c - RP, r - RP, 2 * RP + 1, 2 * RP + 1), boundless=True, fill_value=0)
            rr, cc = np.mgrid[1:2 * RP:3, 1:2 * RP:3]; rr, cc = rr.ravel(), cc.ravel(); m = (L[rr, cc] >= 1) & (L[rr, cc] <= 4) & (np.hypot(rr - RP, cc - RP) * 30 <= 1600)
            rr, cc = rr[m], cc[m]
            if len(rr) < 20: continue
            A2 = dq(a22.read(window=Window(c - RP, r - RP, 2 * RP + 1, 2 * RP + 1), boundless=True, fill_value=-128)); A5 = dq(a25.read(window=Window(c - RP, r - RP, 2 * RP + 1, 2 * RP + 1), boundless=True, fill_value=-128))
            def m3(A): return np.stack([np.nan_to_num(np.nanmean(A[:, y - 1:y + 2, x - 1:x + 2].reshape(64, -1), 1)) for y, x in zip(rr, cc)])
            e22, e25 = m3(A2), m3(A5)
        hist = np.bincount(km.predict(e25.astype(np.float32)), minlength=32) / len(rr)
        x_ind = np.r_[e22.mean(0), e25.mean(0), hist]; dd8.append(NZ['XL'][i][:160] - x_ind)
    dd8 = np.array(dd8); sd8 = dd8.std(0, ddof=1); okd = sd8 > 1e-9; t8 = np.abs(dd8.mean(0)[okd]) / (sd8[okd] / np.sqrt(len(dd8)))
    check('U8 g05: every stand landscape is centred on the stand centroid; landscape (AlphaEarth means + cluster histogram) = independent full-disk recomputation (unbiased)',
          okn == 1.0 and (t8 > 3.5).mean() < 0.05 and np.abs(dd8[:, :128]).mean() < 0.02, f'own-centroid agreement {okn:.3f}; {len(dd8)} nodes: dims with significant bias {(t8 > 3.5).mean():.3f}, mean |Δ AEF| {np.abs(dd8[:, :128]).mean():.4f}')
else:
    print('   (U8 skipped: stand_nodes.npz / stand_veg_v3.node not yet written by g05)')
# ======================= U9 Forest Explorer display: unit labels of the generative attributes match g07's conversions
js = open(os.path.join(WS, 'fe_pan/apps/forest-explorer/public/stands.js')).read()
lab = dict(re.findall(r"(GEN_[A-Z_]+): '([^']+)'", js[js.index('const u = {'):js.index('}', js.index('const u = {'))]))
exp9 = {'GEN_BA': 'ft²/ac', 'GEN_TPA': 'trees/ac', 'GEN_QMD': 'in', 'GEN_BA_SD': 'ft²/ac'}
check('U9 Forest Explorer stand popup / legend units for GEN_* = the exporter conversions (ft²/ac, trees/ac, in)', all(lab.get(k) == v for k, v in exp9.items()), f'labels {lab}')
nf = sum(1 for _, ok in RES if not ok); print(f'\n{len(RES) - nf} / {len(RES)} upstream checks PASS'); sys.exit(1 if nf else 0)
