"""Independent verification of the UPSTREAM pipeline LOGIC (user, 2026-10-22: "check the correctness of the pipeline"; checking the downloads
themselves is unnecessary): grid alignment of every derived raster, the forest-land classes against their written rules, the stands raster against
the polygons, the stand-mapping internals (g05 landscape nodes), the Region 4 derived rasters, and the Forest Explorer display. PASS / FAIL per check;
exit 1 on any failure. VERIFY_INPUTS: the exact list of paths main() reads (the driver fingerprints it, G45.7); importing the module runs no check.
  python scripts/gv_verify_upstream.py [n_points=200]

PRE-REGISTERED THRESHOLDS (one line each: check — threshold — rationale). [r14] = first set in audit round 14 (before any run of it; G47.8).
  U1    exact (transform, size, CRS)       every derived raster is written on grid.json by construction
  U1r   exact; offset a multiple of 3 px   Region 4 grids: grid_r4 on the TreeMap lattice, grid_r4_90 = grid_r4 aggregated 3x3
  U6    agreement > 0.99                   the forest-land classes are a deterministic rule of their inputs; the residual is resampling ties at class edges
  U6r   agreement > 0.99                   the same rule in Region 4 (CONUS TreeMap / EVT read by coordinates)
  U7    id match > 0.99, median |area ratio − 1| < 0.02   rasterisation of the polygons (pixel-centre rule; small stands quantise)
  U8    own-centroid agreement = 1; < 5 % of ALL descriptor dims with |t| > 3.5 [r14: every block — AlphaEarth means, cluster histogram, disturbance
        class + Hansen fractions per period, soils]; mean |Δ| AlphaEarth < 0.02, disturbance / Hansen fractions < 0.03 [r14], standardised soils < 0.1 [r14]
        g05 uses a uniform 256-footprint subsample of the disk: an unbiased estimate (signed errors across nodes), sampling noise ~0.01-0.03 per dim
  U9    exact labels                       the FE display must show g07's units
  U10r  [r14] Region 4 derived rasters against independent routes on sampled pixels (AEF90 only):
        (a) where the GA+FL products exist (the same sources through 03_hansen / 04c_lf_dist / 05_soils): dist and hansen_* equal on >= 99.5 % (treecover2000,
            GDAL `average`: within 1 — revised after the first run: its rounding depends on the warp window, the differences were exactly ±1 on 5.5 %),
            soils90 = the 30 m soils pixel at the 90 m block centre within rel 1e-3 on >= 99 %;
        (b) everywhere from the SOURCES: dist = the CONUS archive by coordinates (6 years) exactly on 100 %; hansen own reprojection (source pixels
            whose centres fall in the 30 m pixel: mode / mean / centre) — agreement must PEAK at offset (0, 0) among ±1 px shifts (lossyear on loss
            pixels, treecover by mean |Δ|), datamask agreement >= 0.99; restrdepth = own bilinear of SOLUS100 at the 90 m centre, rel 1e-3 on >= 99 %
            of pixels with 4 valid neighbours. Every failing line ends with a provenance label (G47.9)."""
import sys, os, json, math, zipfile, re, numpy as np, pandas as pd, rasterio, yaml
from rasterio.windows import Window
from rasterio.transform import Affine
from pyproj import Transformer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from stand_runs import stands_run
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); WS = os.environ.get('WORKSPACE', os.path.dirname(ROOT)); D = f"{ROOT}/data/proto/{os.environ.get('DATA', 'gafl')}"
AEF90 = os.environ.get('AEF90', '0') == '1'
SOILS6 = ('clay030', 'sand030', 'soc030', 'awc030', 'ph030', 'restrdepth')
U10_YEARS = (2001, 2008, 2014, 2019, 2023, 2025)                           # (U10r: the archive years compared)
DIST_ARCHIVE = 'landfire/raw/annual_dist/USAnnualDisturbance_1999_present.zip'
def _years():
    try: return json.load(open(f'{D}/meta.json'))['years']
    except Exception: return list(range(2000, 2026))
def _verify_inputs():
    """every path main() reads under the current environment (G45.7: the driver fingerprints exactly this list; module constant, importable without running)"""
    try: SR_ = stands_run()
    except FileNotFoundError: SR_ = {}
    soils = SOILS6; d_ = lambda x: f'{ROOT}/data/{x}'; yrs = _years()
    out = [d_('grid/grid.json'), d_('aef/aef_2022.tif'), d_('aef/aef_2025.tif'), d_('hansen/lossyear.tif'), d_('hansen/treecover2000.tif'), d_('hansen/datamask.tif'),
           d_('landfire/evt2022.tif'), d_('landfire/evt_classes.csv'), d_('treemap/tm_id.tif'), d_('grid/aoi_mask.tif'), os.path.join(WS, 'fe_pan/apps/forest-explorer/public/stands.js'),
           *[d_(f'landfire/dist/dist_{y}.tif') for y in (2000, 2012, 2024, 2025)], *[d_(f'soils/{k}.tif') for k in soils]]
    if os.environ.get('GV_SKIP_U8') != '1':                       # U8 (stand mapping internals): the map outputs + what the landscape is recomputed from
        out += [f'{D}/stand_nodes.npz', f'{D}/bayes_kmeans.pkl', f'{D}/stand_veg_v3.parquet', f'{D}/stands.parquet', f'{D}/meta.json', f'{D}/cand_feat.npy',
                d_('landfire/dist/dist_codes.csv')]
        out += [d_(f'r4/dist/dist_{y}.tif') for y in yrs] + [d_('r4/soils90/fill.json')] if AEF90 else [d_(f'landfire/dist/dist_{y}.tif') for y in yrs]
    out += [SR_[k] for k in ('land_clean', 'land', 'tif', 'gpkg') if k in SR_] + ([f"{SR_['dir']}/config_resolved.yml"] if 'dir' in SR_ else [])
    if AEF90:
        out += [d_('grid/grid_r4.json'), d_('grid/grid_r4_90.json'), d_('grid/aoi_r4_mask90.tif'), d_('aef90/aef90_2022.tif'), d_('aef90/aef90_2025.tif'), f'{ROOT}/config/pilot.yml',
                d_('r4/land.tif'), d_('r4/hansen_lossyear.tif'), d_('r4/hansen_treecover2000.tif'), d_('r4/hansen_datamask.tif'), *[d_(f'r4/dist/dist_{y}.tif') for y in (2000, 2012, 2024, 2025)],
                *[d_(f'r4/soils90/{k}.tif') for k in soils], WS + '/FVSclaude/treemap/Data/TreeMap2022_CONUS.tif', d_('landfire/raw/LF2022_EVT_CONUS/Tif/LF2022_EVT_CONUS.tif')]
        # U10r: the GA+FL products, the sources (Hansen granules, the LANDFIRE archive, SOLUS100 restriction depth)
        import glob as _gl
        out += [d_(f'landfire/dist/dist_{y}.tif') for y in U10_YEARS] + [d_(f'r4/dist/dist_{y}.tif') for y in U10_YEARS] + [d_(DIST_ARCHIVE), WS + '/SOLUS/SOLUS100/resdept_all_cm_p.tif'] + \
               sorted(_gl.glob(d_('hansen/raw/Hansen_GFC-2024-v1.12_*_*N_*W.tif')))
    return sorted(set(out))
VERIFY_INPUTS = _verify_inputs()
VERIFY_CODE = [f'{ROOT}/scripts/{f}' for f in ('gv_verify.py', 'stand_runs.py')]     # (U8 imports gv_verify's LANDFIRE decoding; stand_runs: the stand run)
def main():
    NP = int(sys.argv[1]) if len(sys.argv) > 1 else 200; rng = np.random.default_rng(7); RES = []
    def check(name, ok, detail=''):          # (G47.9: these checks read rasters / map outputs, which carry no table code hash)
        RES.append((name, bool(ok))); lab = '' if ok else '  [divergent (upstream rasters / map outputs carry no table code hash: the products disagree with the rule)]'
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}{lab}", flush=True)
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
        # (r13 E-F9) grid_r4_90 = grid_r4 aggregated 3x3: the same origin, a 3x larger pixel, the size covering grid_r4
        g9ok = abs(TR49.c - TR4.c) < 1e-6 and abs(TR49.f - TR4.f) < 1e-6 and abs(TR49.a - 3 * TR4.a) < 1e-9 and abs(TR49.e - 3 * TR4.e) < 1e-9 and TR49.b == 0 and TR49.d == 0 and \
               G49['width'] == -(-G4['width'] // 3) and G49['height'] == -(-G4['height'] // 3)
        check('U1r Region 4 rasters on grid_r4 (30 m) / grid_r4_90 (90 m); grid_r4 on the TreeMap lattice of grid.json, a multiple of 3 px off; grid_r4_90 origin = grid_r4 origin, pixel 3x',
              not bad and g9ok and all(abs(v - round(v)) < 1e-6 and round(v) % 3 == 0 for v in lat),
              f'{len(f30) + len(f90)} rasters; off-grid {bad}; offset {lat}; grid_r4_90 origin {(TR49.c, TR49.f)} vs {(TR4.c, TR4.f)}, pixel {TR49.a} vs {TR4.a}, size {G49["width"]}x{G49["height"]} vs {G4["width"]}x{G4["height"]}')
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
    # ======================= U10r (r14 G47.8) the Region 4 derived rasters (dist, hansen_*, soils90) against independent routes on sampled pixels (pre-registered above)
    if AEF90:
        import zipfile, glob as _gl
        AOI9 = rasterio.open(f'{ROOT}/data/grid/aoi_r4_mask90.tif'); a9 = AOI9.read(1); ia9 = np.flatnonzero(a9.ravel() == 1); W9 = a9.shape[1]; del a9
        r10 = np.random.default_rng(1010); pk9 = r10.choice(ia9, 3000, replace=False)
        P10 = sorted((3 * (q // W9) + int(r10.integers(0, 3)), 3 * (q % W9) + int(r10.integers(0, 3))) for q in pk9)       # random 30 m AOI pixels (row order)
        XY10 = np.array([TR4 * (c + 0.5, r + 0.5) for r, c in P10]); st10 = {}
        # (a) the GA+FL products (03_hansen / 04c_lf_dist / 05_soils on the TreeMap lattice): exact equality where both exist inside both AOIs
        gaoi = rasterio.open(f'{ROOT}/data/grid/aoi_mask.tif'); inG = []
        for (r, c), (x_, y_) in zip(P10, XY10):
            rg, cg = gaoi.index(x_, y_)
            if 0 <= rg < gaoi.height and 0 <= cg < gaoi.width and gaoi.read(1, window=Window(cg, rg, 1, 1))[0, 0] == 1: inG.append((r, c, rg, cg))
        eqa = {}
        for nm, f4, fg in [('lossyear', 'r4/hansen_lossyear.tif', 'hansen/lossyear.tif'), ('treecover2000', 'r4/hansen_treecover2000.tif', 'hansen/treecover2000.tif'),
                           ('datamask', 'r4/hansen_datamask.tif', 'hansen/datamask.tif')] + [(f'dist_{y}', f'r4/dist/dist_{y}.tif', f'landfire/dist/dist_{y}.tif') for y in U10_YEARS]:
            s4_, sg_ = rasterio.open(f'{ROOT}/data/{f4}'), rasterio.open(f'{ROOT}/data/{fg}')
            tol_ = 1 if nm == 'treecover2000' else 0                          # (GDAL average: window-dependent rounding, ±1)
            eqa[nm] = (len(inG), float(np.mean([abs(int(s4_.read(1, window=Window(c, r, 1, 1))[0, 0]) - int(sg_.read(1, window=Window(cg, rg, 1, 1))[0, 0])) <= tol_ for r, c, rg, cg in inG])) if inG else np.nan)
        so_rel = []
        for k in SOILS6:                                  # the 90 m soil pixel = the 30 m soils pixel at the 90 m block centre (bilinear of the same SOLUS grid at the same point)
            s9, sg_ = rasterio.open(f'{ROOT}/data/r4/soils90/{k}.tif'), rasterio.open(f'{ROOT}/data/soils/{k}.tif')
            for r, c, _, _ in inG[:300]:
                x9, y9 = TR4 * (3 * (c // 3) + 1.5, 3 * (r // 3) + 1.5); rg, cg = sg_.index(x9, y9)
                v9 = float(s9.read(1, window=Window(c // 3, r // 3, 1, 1))[0, 0]); vg = float(sg_.read(1, window=Window(cg, rg, 1, 1))[0, 0])
                if v9 > -9000 and vg > -9000: so_rel.append(abs(v9 - vg) / max(abs(vg), 1e-3))
        so_rel = np.array(so_rel)
        okA = bool(inG) and all(v_[1] >= 0.995 for v_ in eqa.values()) and len(so_rel) > 0 and (so_rel <= 1e-3).mean() >= 0.99
        # (b) from the SOURCES, everywhere: the LANDFIRE CONUS archive by coordinates; Hansen granules (EPSG:4326) by an own reprojection; SOLUS100 restriction depth by own bilinear
        AOI_ok = lambda r, c: AOI9.read(1, window=Window(c // 3, r // 3, 1, 1))[0, 0] == 1
        zo = zipfile.ZipFile(f'{ROOT}/data/{DIST_ARCHIVE}'); nm10 = {int(re.search(r'Dist(\d\d)_CONUS', n).group(1)): n for n in zo.namelist() if n.endswith('_CONUS.zip')}
        eqb = {}; P10d = P10[::5]
        for y in U10_YEARS:
            yy = y - 2000 if y >= 2000 else y - 1900; n_ = nm10.get(yy)
            if n_ is None: eqb[y] = (0, np.nan); continue
            tif = [m for m in zipfile.ZipFile(zo.open(n_)).namelist() if m.lower().endswith('.tif')][0]
            s4_ = rasterio.open(f'{ROOT}/data/r4/dist/dist_{y}.tif'); agr = []
            with rasterio.open('/vsizip/{/vsizip/' + f'{ROOT}/data/{DIST_ARCHIVE}/{n_}' + '}/' + tif) as so_:
                for (r, c), (x_, y_) in zip(P10d, XY10[::5]):
                    rs_, cs_ = so_.index(x_, y_); v = int(so_.read(1, window=Window(cs_, rs_, 1, 1))[0, 0]) if 0 <= rs_ < so_.height and 0 <= cs_ < so_.width else 0
                    v = 0 if (v < 0 or (so_.nodata is not None and v == so_.nodata) or not AOI_ok(r, c)) else v
                    agr.append(int(s4_.read(1, window=Window(c, r, 1, 1))[0, 0]) == v)
            eqb[y] = (len(agr), float(np.mean(agr)))
        to4326 = Transformer.from_crs(CRS, 4326, always_xy=True); fr4326 = Transformer.from_crs(4326, CRS, always_xy=True)
        gran = {p_: sorted(_gl.glob(f'{ROOT}/data/hansen/raw/Hansen_GFC-2024-v1.12_{p_}_*N_*W.tif')) for p_ in ('lossyear', 'treecover2000', 'datamask')}
        gsrc = {p_: [rasterio.open(f_) for f_ in fs_] for p_, fs_ in gran.items()}
        def own_hansen(p_, x_, y_):
            """the 30 m pixel centred at (x_, y_) (EPSG:5070) from the source granule: the source pixels whose centres fall inside it -> mode (lossyear) / mean
            (treecover2000) / the pixel at the centre (datamask)"""
            lo_, la_ = to4326.transform(x_, y_)
            g_ = next((s_ for s_ in gsrc[p_] if s_.bounds.left <= lo_ < s_.bounds.right and s_.bounds.bottom < la_ <= s_.bounds.top), None)
            if g_ is None: return None
            if p_ == 'datamask': rr_, cc_ = g_.index(lo_, la_); return int(g_.read(1, window=Window(cc_, rr_, 1, 1))[0, 0])
            rr_, cc_ = g_.index(lo_, la_); A_ = g_.read(1, window=Window(cc_ - 3, rr_ - 3, 7, 7), boundless=True, fill_value=0)
            jj_, ii_ = np.meshgrid(np.arange(cc_ - 3, cc_ + 4), np.arange(rr_ - 3, rr_ + 4)); lon_, lat_ = g_.transform * (jj_ + 0.5, ii_ + 0.5)
            X_, Y_ = fr4326.transform(lon_, lat_); m_ = (np.abs(np.asarray(X_) - x_) <= 15) & (np.abs(np.asarray(Y_) - y_) <= 15)
            if not m_.any(): return None
            v_ = A_[m_]
            if p_ == 'treecover2000': return float(v_.mean())
            u_, n_ = np.unique(v_, return_counts=True); return int(u_[np.argmax(n_)])
        h4 = {p_: rasterio.open(f'{ROOT}/data/r4/hansen_{p_}.tif') for p_ in gran}
        lyw = h4['lossyear']; Pl = []
        while len(Pl) < 300:                          # lossyear: pixels with loss (stratified: a random AOI block, its loss pixels)
            r_, c_ = P10[int(r10.integers(len(P10)))]; B_ = lyw.read(1, window=Window(c_ - 32, r_ - 32, 64, 64), boundless=True, fill_value=255)
            ij = np.argwhere((B_ > 0) & (B_ < 100))
            if len(ij): k_ = ij[r10.integers(len(ij))]; Pl.append((r_ - 32 + int(k_[0]), c_ - 32 + int(k_[1])))
        agree_l = {}; dtc = {}
        ol_ = [(r, c, own_hansen('lossyear', *(TR4 * (c + 0.5, r + 0.5)))) for r, c in Pl[:200]]; ot_ = [(r, c, own_hansen('treecover2000', *(TR4 * (c + 0.5, r + 0.5)))) for r, c in P10[:200]]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                a_ = [int(lyw.read(1, window=Window(c + dc, r + dr, 1, 1))[0, 0]) == o_ for r, c, o_ in ol_ if o_ is not None]
                t_ = [abs(v4 - o_) for r, c, o_ in ot_ if o_ is not None for v4 in [int(h4['treecover2000'].read(1, window=Window(c + dc, r + dr, 1, 1))[0, 0])] if v4 != 255]
                agree_l[(dr, dc)] = round(float(np.mean(a_)), 3) if a_ else np.nan; dtc[(dr, dc)] = round(float(np.mean(t_)), 2) if t_ else np.nan
        dm_ = [int(h4['datamask'].read(1, window=Window(c, r, 1, 1))[0, 0]) == own_hansen('datamask', *(TR4 * (c + 0.5, r + 0.5))) for r, c in P10[:400]]
        best_l = max(agree_l, key=lambda k_: agree_l[k_]); best_t = min(dtc, key=lambda k_: dtc[k_])
        sol = rasterio.open(WS + '/SOLUS/SOLUS100/resdept_all_cm_p.tif'); s9r = rasterio.open(f'{ROOT}/data/r4/soils90/restrdepth.tif'); rel_r = []
        for r, c in P10[:600]:
            x9, y9 = TR4 * (3 * (c // 3) + 1.5, 3 * (r // 3) + 1.5); fx = (x9 - sol.transform.c) / sol.transform.a - 0.5; fy = (y9 - sol.transform.f) / sol.transform.e - 0.5
            i0, j0 = int(math.floor(fy)), int(math.floor(fx)); B_ = sol.read(1, window=Window(j0, i0, 2, 2), boundless=True, fill_value=sol.nodata if sol.nodata is not None else 0).astype(float)
            if sol.nodata is not None and (B_ == sol.nodata).any(): continue
            wy, wx = fy - i0, fx - j0; own = B_[0, 0] * (1 - wy) * (1 - wx) + B_[0, 1] * (1 - wy) * wx + B_[1, 0] * wy * (1 - wx) + B_[1, 1] * wy * wx
            v9 = float(s9r.read(1, window=Window(c // 3, r // 3, 1, 1))[0, 0])
            if v9 > -9000: rel_r.append(abs(v9 - own) / max(abs(own), 1e-3))
        rel_r = np.array(rel_r)
        okB = all(n_ > 0 and v_ == 1.0 for n_, v_ in eqb.values()) and best_l == (0, 0) and best_t == (0, 0) and np.mean(dm_) >= 0.99 and len(rel_r) > 0 and (rel_r <= 1e-3).mean() >= 0.99
        check('U10r Region 4 derived rasters (dist, hansen_*, soils90) = independent routes: (a) the GA+FL products where both exist; (b) the sources (LANDFIRE archive, Hansen granules, SOLUS100)',
              okA and okB, f'(a) {len(inG)} pixels in both AOIs: equal share {eqa}; soils90 vs the 30 m pixel: rel <= 1e-3 on {(so_rel <= 1e-3).mean() if len(so_rel) else float("nan"):.4f} of {len(so_rel)}; '
              f'(b) dist vs the archive {{year: (pixels, equal share)}} {eqb}; lossyear agreement on loss pixels by offset {{(dr, dc): share}} {agree_l} (best {best_l}); '
              f'treecover mean |Δ| by offset {dtc} (best {best_t}); datamask agreement {np.mean(dm_):.4f}; restrdepth own bilinear rel <= 1e-3 on {(rel_r <= 1e-3).mean() if len(rel_r) else float("nan"):.4f} of {len(rel_r)}')
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
    #                         uniform 256-subsample, so compared as an unbiased estimate: signed errors across nodes). (r14 G47.8) EVERY block of the descriptor: AlphaEarth
    #                         means, cluster histogram, per period (BINS) the share of footprints with each LANDFIRE class / Hansen loss in it, mean standardised soils
    if os.environ.get('GV_SKIP_U8') == '1':                     # genveg_pipeline stage vup runs before the stand mapping; U8 runs in stage verify
        print('   (U8 skipped here: it checks the stand mapping, run after it in the verify stage)')
    elif os.path.exists(f'{D}/stand_nodes.npz') and os.path.exists(f'{D}/stand_veg_v3.parquet') and 'node' in pd.read_parquet(f'{D}/stand_veg_v3.parquet').columns:   # (r13 F#11: guard)
        import pickle, gv_verify as GV
        NZ = np.load(f'{D}/stand_nodes.npz'); km = pickle.load(open(f'{D}/bayes_kmeans.pkl', 'rb')); a22, a25 = rasterio.open(f'{ROOT}/data/aef/aef_2022.tif'), rasterio.open(f'{ROOT}/data/aef/aef_2025.tif')
        M8 = json.load(open(f'{D}/meta.json')); YRS, CLS8 = M8['years'], M8['classes']; NC8 = len(CLS8); SO8 = M8['soils']; lk8, _ = GV.landfire_codes(f'{ROOT}/data/landfire/dist/dist_codes.csv', CLS8)
        BINS8 = [(2000, 2009), (2010, 2016), (2017, 2021), (2022, 2025)]                # the descriptor's disturbance periods (docs: g04 landscape)
        CF8 = np.load(f'{D}/cand_feat.npy', mmap_mode='r'); _so8 = np.asarray(CF8[:, -len(SO8):], dtype=np.float64); smu8, ssd8 = _so8.mean(0), _so8.std(0) + 1e-6; del _so8   # (the model's soil scaling)
        if AEF90:
            dist8 = {y: rasterio.open(f'{ROOT}/data/r4/dist/dist_{y}.tif') for y in YRS}; han8 = rasterio.open(f'{ROOT}/data/r4/hansen_lossyear.tif')
            soil8 = {k: rasterio.open(f'{ROOT}/data/r4/soils90/{k}.tif') for k in SO8}; FILL8 = json.load(open(f'{ROOT}/data/r4/soils90/fill.json')); SD8 = 3
        else:
            dist8 = {y: rasterio.open(f'{ROOT}/data/landfire/dist/dist_{y}.tif') for y in YRS}; han8 = rasterio.open(f'{ROOT}/data/hansen/lossyear.tif')
            soil8 = {k: rasterio.open(f'{ROOT}/data/soils/{k}.tif') for k in SO8}; FILL8 = {'clay030': 7.58, 'sand030': 83.2, 'soc030': 0.964, 'awc030': 0.052, 'ph030': 5.18, 'restrdepth': 200.0}; SD8 = 1
        if AEF90:       # Region 4 features: footprints = 90 m blocks of grid_r4 (centres 1 mod 3), AlphaEarth = the 90 m pixel; stand pixels shifted by the grid offset
            a22, a25 = rasterio.open(f'{ROOT}/data/aef90/aef90_2022.tif'), rasterio.open(f'{ROOT}/data/aef90/aef90_2025.tif'); land4 = rasterio.open(f'{ROOT}/data/r4/land.tif')
            st_tr = rasterio.open(SRN['tif']).transform; oR, oC = round((land4.transform.f - st_tr.f) / 30), round((st_tr.c - land4.transform.c) / 30)
        def fp_desc(AR, AC):
            """the disturbance + soil blocks of the descriptor for footprints at absolute feature-grid pixels (AR, AC): per period the share of footprints with a
            LANDFIRE class (any pixel of its 3x3 window / 90 m block, event calendar year in the period) and with Hansen loss there; mean standardised soils"""
            r0_ = (3 * (AR // 3)) if AEF90 else AR - 1; c0_ = (3 * (AC // 3)) if AEF90 else AC - 1; R0, C0 = int(r0_.min()), int(c0_.min()); hh, ww = int(r0_.max()) - R0 + 3, int(c0_.max()) - C0 + 3
            ev = np.zeros((len(AR), len(YRS), NC8 + 1), bool); hw_ = han8.read(1, window=Window(C0, R0, ww, hh), boundless=True, fill_value=0)
            for y in YRS:
                W_ = dist8[y].read(1, window=Window(C0, R0, ww, hh), boundless=True, fill_value=0)
                for dr in range(3):
                    for dc in range(3):
                        cod = W_[r0_ - R0 + dr, c0_ - C0 + dc]; hv = hw_[r0_ - R0 + dr, c0_ - C0 + dc]
                        for i_ in np.flatnonzero(cod > 0):
                            c_, ey = lk8.get((y, int(cod[i_])), (None, y))
                            if c_ is not None and YRS[0] <= ey <= YRS[-1]: ev[i_, ey - YRS[0], c_] = True
                        if y == YRS[0]:
                            for i_ in np.flatnonzero((hv > 0) & (hv < 100)):
                                if YRS[0] <= 2000 + int(hv[i_]) <= YRS[-1]: ev[i_, 2000 + int(hv[i_]) - YRS[0], NC8] = True
            out = []
            for lo, hi in BINS8:
                sl = slice(lo - YRS[0], hi - YRS[0] + 1); out += list(ev[:, sl, :NC8].any(1).mean(0)) + [float(ev[:, sl, NC8].any(1).mean())]
            so = np.array([[(lambda v: FILL8[k] if v <= -9000 else v)(float(soil8[k].read(1, window=Window(int(c) // SD8, int(r) // SD8, 1, 1))[0, 0])) for k in SO8] for r, c in zip(AR, AC)])
            return np.r_[out, ((so - smu8) / ssd8).mean(0)]
        RP = int(1600 // 30) + 2; dd8 = []; sv = pd.read_parquet(f'{D}/stand_veg_v3.parquet')
        Ssub = pd.read_parquet(f'{D}/stands.parquet'); cen = Ssub.groupby('stand_id')[['row', 'col']].mean()
        okn = (np.abs(NZ['nodes'][sv.node.values] - np.rint(cen.loc[sv.stand_id.values].values)).max(1) == 0).mean()   # each stand's landscape is centred on its own centroid
        nxl = 128 + 32 + len(BINS8) * (NC8 + 1) + len(SO8); shp8 = NZ['XL'].shape[1] == nxl
        for i in rng.choice(len(NZ['nodes']), 20, replace=False) if shp8 else []:
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
                e22, e25 = m3(A2), m3(A5); AR, AC = rr + r - RP, cc + c - RP
            hist = np.bincount(km.predict(e25.astype(np.float32)), minlength=32) / len(rr)
            x_ind = np.r_[e22.mean(0), e25.mean(0), hist, fp_desc(np.asarray(AR), np.asarray(AC))]; dd8.append(NZ['XL'][i] - x_ind)
        dd8 = np.array(dd8).reshape(-1, nxl); ok8 = len(dd8) >= 10
        if ok8:
            sd8 = dd8.std(0, ddof=1); okd = sd8 > 1e-9; t8 = np.abs(dd8.mean(0)[okd]) / (sd8[okd] / np.sqrt(len(dd8))); i_d = slice(160, 160 + len(BINS8) * (NC8 + 1)); i_s = slice(nxl - len(SO8), nxl)
            mA, mD, mS = np.abs(dd8[:, :128]).mean(), np.abs(dd8[:, i_d]).mean(), np.abs(dd8[:, i_s]).mean(); fsig = float((t8 > 3.5).mean())
        check('U8 g05: every stand landscape is centred on the stand centroid; landscape (every block: AlphaEarth means, cluster histogram, disturbance / Hansen period fractions, soils) = independent full-disk recomputation (unbiased)',
              shp8 and ok8 and okn == 1.0 and fsig < 0.05 and mA < 0.02 and mD < 0.03 and mS < 0.1,
              f'own-centroid agreement {okn:.3f}; descriptor width {NZ["XL"].shape[1]} (expected {nxl}); {len(dd8)} nodes' +
              (f': dims with significant bias {fsig:.3f} (of {int(okd.sum())}; by block AEF {(t8[:int(okd[:128].sum())] > 3.5).mean():.3f}), mean |Δ| AEF {mA:.4f}, '
               f'disturbance / Hansen {mD:.4f}, soils {mS:.4f}' if ok8 else ' (fewer than 10 usable nodes)'))
    else:
        print('   (U8 skipped: stand_nodes.npz / stand_veg_v3.node not yet written by g05)')
        if os.environ.get('GV_REQUIRE_MAP') == '1': check('U8 stand-mapping outputs present (required after the map stage)', False, 'stand_nodes.npz / stand_veg_v3.parquet node column missing')
    # ======================= U9 Forest Explorer display: unit labels of the generative attributes match g07's conversions
    js = open(os.path.join(WS, 'fe_pan/apps/forest-explorer/public/stands.js')).read()
    lab = dict(re.findall(r"(GEN_[A-Z_]+): '([^']+)'", js[js.index('const u = {'):js.index('}', js.index('const u = {'))]))
    exp9 = {'GEN_BA': 'ft²/ac', 'GEN_TPA': 'trees/ac', 'GEN_QMD': 'in', 'GEN_BA_SD': 'ft²/ac'}
    check('U9 Forest Explorer stand popup / legend units for GEN_* = the exporter conversions (ft²/ac, trees/ac, in)', all(lab.get(k) == v for k, v in exp9.items()), f'labels {lab}')
    nf = sum(1 for _, ok in RES if not ok); print(f'\n{len(RES) - nf} / {len(RES)} upstream checks PASS'); sys.exit(1 if nf else 0)
if __name__ == '__main__': main()
