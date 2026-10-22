"""Prototype layers for Forest Explorer (user, 2026-10-22: "add all resulting prototype layers ... so that I can visualize the results").
Reads config/fe_layers.yml and writes, under <WORKSPACE>/<out>/, one file per layer + manifest.json (read by the FE 'Prototype layers' panel,
fe_pan apps/forest-explorer/public/proto.js). Layer kinds:
  stands        polygons of a stand-delineation run (data/derived/runs/<run>/finalize/stands.gpkg) clipped to the area (24_export_fe.py)
  geojson_file  an existing GeoJSON (e.g. the generative-vegetation pipeline's FE file)
  raster        a grid raster in the raster window reprojected to EPSG:4326 (PNG + bounds); styles: land (status classes), edges (boundaries of an
                id raster), aef_rgb (AlphaEarth PCs 1-3 of feature_fit as RGB), ramp / diverging (value range), latent_rgb (IWAE embedding PCs 1-3)
  freerun       HBAD3P free-run plots (freerun_val.csv of a fit) as points at the PUBLIC (fuzzed) FIA plot coordinates, relative errors per variable
Sources: "run:<run>/<stage>/<file>" (stand run), "data:<path>" (standmap data/), plain paths relative to WORKSPACE. Missing sources: skipped + logged.
  .venv/bin/python -I scripts/fe_proto_layers.py [config/fe_layers.yml] [layer id ...]"""
import os, sys, json, subprocess, numpy as np, pandas as pd, rasterio, yaml, duckdb
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_bounds
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, WS, grid
CFG = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.yml') else f'{ROOT}/config/fe_layers.yml'))
ONLY = [a for a in sys.argv[1:] if not a.endswith('.yml')]
OUT = os.path.join(WS, CFG['out']); os.makedirs(OUT, exist_ok=True)
TR, GW, GH, CRS = grid()
STATUS = {1: ('forest', '#2e7d32'), 2: ('harvested (loss since 2017)', '#e65100'), 3: ('regenerating', '#f9a825'), 4: ('forest (not in TreeMap)', '#6a1b9a'), 9: ('converted', '#9e9e9e')}
VIRIDIS = ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725']; DIVERG = ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b']

def src_path(s):
    if s.startswith('run:'): return os.path.join(ROOT, 'data', 'derived', 'runs', s[4:])
    if s.startswith('data:'): return os.path.join(ROOT, 'data', s[5:])
    return os.path.join(WS, s)
def hexrgb(h): return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)
def ramp(v, stops, lo, hi):
    c = np.stack([hexrgb(h) for h in stops]); t = np.clip((v - lo) / (hi - lo), 0, 1) * (len(stops) - 1); i = np.minimum(t.astype(int), len(stops) - 2); f = (t - i)[..., None]
    return c[i] * (1 - f) + c[i + 1] * f

# ---------------- rasters: window of the grid -> EPSG:4326 PNG
r0, c0, r1, c1 = CFG['raster_window']; WIN = Window(c0, r0, c1 - c0, r1 - r0); WTR = rasterio.windows.transform(WIN, TR)
L_, B_, R_, T_ = rasterio.windows.bounds(WIN, TR); BB = transform_bounds(CRS, 'EPSG:4326', L_, B_, R_, T_, densify_pts=50)
RES = CFG['raster_res_deg']; DW, DH = int(np.ceil((BB[2] - BB[0]) / RES)), int(np.ceil((BB[3] - BB[1]) / RES)); DTR = from_bounds(*BB, DW, DH)
def warp(a, nearest):
    out = np.full((DH, DW), np.nan if a.dtype.kind == 'f' else 0, a.dtype)
    reproject(a, out, src_transform=WTR, src_crs=CRS, dst_transform=DTR, dst_crs='EPSG:4326', resampling=Resampling.nearest if nearest else Resampling.bilinear,
              src_nodata=np.nan if a.dtype.kind == 'f' else 0, dst_nodata=np.nan if a.dtype.kind == 'f' else 0)
    return out
def forest_mask():
    p = src_path('run:gafl_pilot/clean/land_clean.tif'); a = rasterio.open(p).read(1, window=WIN); return warp(((a >= 1) & (a <= 4)).astype(np.uint8), True) > 0
def save_png(rgba, lid):
    Image.fromarray(rgba.astype(np.uint8), 'RGBA').save(f'{OUT}/{lid}.png', optimize=True)
    return dict(type='image', url=f'/proto/{lid}.png', coordinates=[[BB[0], BB[3]], [BB[2], BB[3]], [BB[2], BB[1]], [BB[0], BB[1]]])
def raster_layer(L):
    p = src_path(L['source']); st = L['style']
    if not os.path.exists(p): return None, f'missing {p}'
    s = rasterio.open(p); assert s.transform == TR and s.shape == (GH, GW), f'{p} not on the grid'
    rgba = np.zeros((DH, DW, 4))
    if st == 'land':
        a = warp(s.read(1, window=WIN).astype(np.uint8), True)
        for k, (_, col) in STATUS.items(): m = a == k; rgba[m, :3] = hexrgb(col); rgba[m, 3] = 255
        legend = [dict(label=l, color=c) for l, c in STATUS.values()]
    elif st == 'edges':
        a = s.read(1, window=WIN).astype(np.int64); e = np.zeros(a.shape, np.uint8)
        e[:, :-1] |= (a[:, :-1] != a[:, 1:]) & (a[:, :-1] > 0); e[:-1] |= (a[:-1] != a[1:]) & (a[:-1] > 0)
        w = warp(e, True) > 0; rgba[w] = [20, 20, 20, 230]; legend = [dict(label='superpixel boundary', color='#141414')]
    elif st in ('aef_rgb', 'latent_rgb'):
        if st == 'aef_rgb':
            fit = dict(np.load(src_path(L['fit']))); q = s.read(window=WIN).astype(np.float32); bad = (q == -128).any(0)
            x = np.sign(q) * (q / 127.5) ** 2; X = np.moveaxis(x, 0, -1)
            P = (np.nan_to_num(X - fit['aef_mean'])) @ fit['aef_comp'][:3].T
        else:
            X = np.moveaxis(s.read(window=WIN).astype(np.float32), 0, -1); bad = ~np.isfinite(X).all(-1); Xf = np.nan_to_num(X)
            v = Xf[~bad]; mu = v.mean(0); U, S_, Vt = np.linalg.svd(v[::max(1, len(v) // 200000)] - mu, full_matrices=False); P = (Xf - mu) @ Vt[:3].T
        P[bad] = np.nan; rgb = np.zeros((DH, DW, 3))
        for j in range(3):
            w = warp(P[..., j].astype(np.float32), False); lo, hi = np.nanpercentile(w, [2, 98]); rgb[..., j] = np.clip((w - lo) / (hi - lo), 0, 1) * 255
        ok = np.isfinite(rgb).all(-1) & forest_mask(); rgba[ok, :3] = rgb[ok]; rgba[ok, 3] = 255
        legend = [dict(label='RGB = principal components 1 / 2 / 3 (2-98 % stretch), forest land only', color=None)]
    else:
        a = s.read(1, window=WIN).astype(np.float32); nd = s.nodata
        if nd is not None and not np.isnan(nd): a[a == nd] = np.nan
        w = warp(a, False); lo, hi = L['range']; ok = np.isfinite(w) & forest_mask()
        rgba[ok, :3] = ramp(w[ok], VIRIDIS if st == 'ramp' else DIVERG, lo, hi); rgba[ok, 3] = 255
        legend = [dict(ramp=VIRIDIS if st == 'ramp' else DIVERG, lo=lo, hi=hi, unit=L.get('unit', ''))]
    d = save_png(rgba, L['id']); d['legend'] = legend; d['note'] = 'forest land of the Baker superpixel extent; ~30 m, reprojected to EPSG:4326'
    return d, None

# ---------------- vector layers
STAND_PROPS = [dict(name='status', label='status', categories={str(k): dict(label=l, color=c) for k, (l, c) in STATUS.items()}),
               dict(name='area_ha', label='area', unit='ha', range=[0, 40]), dict(name='BALIVE', label='basal area (TreeMap)', unit='ft²/ac', range=[0, 150]),
               dict(name='STANDHT', label='stand height (TreeMap)', unit='ft', range=[0, 100]), dict(name='QMD', label='QMD (TreeMap)', unit='in', range=[0, 16]),
               dict(name='CANOPYPCT', label='canopy cover (TreeMap)', unit='%', range=[0, 100]), dict(name='SOFTWOOD_FRAC', label='softwood share', unit='', range=[0, 1]),
               dict(name='tm_n', label='TreeMap plots in the stand', unit='', range=[1, 30]), dict(name='awc030', label='available water 0-30 cm', unit='cm/cm', range=[0, 0.2])]
GEN_PROPS = [dict(name='GEN_BA', label='basal area — generative', unit='ft²/ac', range=[0, 150]), dict(name='GEN_TPA', label='trees per acre — generative', unit='trees/ac', range=[0, 600]),
             dict(name='GEN_QMD', label='QMD — generative', unit='in', range=[0, 16]), dict(name='GEN_BA_SD', label='basal area — sample spread', unit='ft²/ac', range=[0, 100])]
def stands_layer(L):
    g = os.path.join(ROOT, 'data', 'derived', 'runs', L['run'], 'finalize', 'stands.gpkg')
    if not os.path.exists(g): return None, f'missing {g}'
    fn = f"{OUT}/{L['id']}.geojson"
    r = subprocess.run([sys.executable, '-I', os.path.join(ROOT, 'scripts', '24_export_fe.py'), os.path.join(ROOT, CFG['area']), fn, g], capture_output=True, text=True)
    if r.returncode != 0: return None, 'export failed: ' + r.stderr[-400:]
    summ = json.load(open(os.path.join(ROOT, 'data', 'derived', 'runs', L['run'], 'summary.json')))
    return dict(type='fill', url=f"/proto/{L['id']}.geojson", props=STAND_PROPS, color='status', note=r.stdout.strip().splitlines()[-1][:200] + f" | overrides {summ.get('sets') or 'none'}"), None
def geojson_layer(L):
    p = os.path.join(ROOT, L['path'])
    if not os.path.exists(p): return None, f'missing {p}'
    d = json.load(open(p)); json.dump(d, open(f"{OUT}/{L['id']}.geojson", 'w'), separators=(',', ':'))
    has_gen = any('GEN_BA' in f['properties'] for f in d['features'][:50])
    return dict(type='fill', url=f"/proto/{L['id']}.geojson", props=(GEN_PROPS if has_gen else []) + STAND_PROPS, color='GEN_BA' if has_gen else 'status',
                note=f"{len(d['features'])} stands"), None
_plots = None
def plot_coords():
    global _plots
    if _plots is None:
        c = duckdb.connect(os.path.join(WS, 'landis_region4', 'data_r4', 'slim_r4.duckdb'), read_only=True)
        _plots = c.sql('select STATECD st, UNITCD u, COUNTYCD c, PLOT p, avg(LAT) lat, avg(LON) lon from PLOT group by all').df()
    return _plots
def freerun_layer(L):
    p = os.path.join(WS, L['fit'], 'freerun_val.csv')
    if not os.path.exists(p): return None, f'missing {p}'
    d = pd.read_csv(p)
    if 'st' not in d.columns: return None, f'{p} has no plot ids (rerun the freerun stage)'
    w = d.pivot_table(index=['st', 'u', 'c', 'p', 'T'], columns='var', values=['obs1', 'obsL', 'sim']).reset_index()
    w.columns = ['_'.join(x).strip('_') for x in w.columns]
    w = w.merge(plot_coords(), on=['st', 'u', 'c', 'p'], how='inner'); feats = []
    for r in w.itertuples(index=False):
        pr = dict(plot=f'{r.st}_{r.u}_{r.c}_{r.p}', years=round(r.T, 1))
        for v in ('BA', 'N12', 'QMD', 'LH'):
            o1, oL, s_ = getattr(r, f'obs1_{v}'), getattr(r, f'obsL_{v}'), getattr(r, f'sim_{v}')
            pr[f'{v}_obs_first'] = None if pd.isna(o1) else round(float(o1), 2); pr[f'{v}_obs_last'] = None if pd.isna(oL) else round(float(oL), 2)
            pr[f'{v}_sim'] = None if pd.isna(s_) else round(float(s_), 2)
            pr[f'err_{v}'] = None if (pd.isna(s_) or pd.isna(oL) or oL <= 0) else round(100 * (s_ - oL) / oL, 1)
            pr[f'errp_{v}'] = None if (pd.isna(o1) or pd.isna(oL) or oL <= 0) else round(100 * (o1 - oL) / oL, 1)
        feats.append(dict(type='Feature', properties=pr, geometry=dict(type='Point', coordinates=[round(r.lon, 5), round(r.lat, 5)])))
    json.dump(dict(type='FeatureCollection', features=feats), open(f"{OUT}/{L['id']}.geojson", 'w'), separators=(',', ':'))
    props = [dict(name=f'err_{v}', label=f'{lab}: simulated vs observed at the last visit', unit='% error', range=[-60, 60], diverging=True) for v, lab in
             (('BA', 'basal area'), ('N12', 'stems ≥ 12.7 cm'), ('QMD', 'QMD ≥ 12.7 cm'), ('LH', 'Lorey height'))] + \
            [dict(name=f'errp_{v}', label=f'{v}: persistence (first visit) vs last visit', unit='% error', range=[-60, 60], diverging=True) for v in ('BA', 'N12')]
    return dict(type='circle', url=f"/proto/{L['id']}.geojson", props=props, color='err_BA',
                note=f'{len(feats)} val plots at the PUBLIC (fuzzed) FIA coordinates; free run first -> last visit'), None

def latent_layer(L):
    """IWAE latent (posterior mean) of the raster window, encoded tile by tile with a 1-px halo, shown as its principal components 1-3 (RGB)."""
    mp = src_path(L['model'])
    if not os.path.exists(mp): return None, f'missing {mp}'
    import torch, embed_iwae as EI
    from features import Featurizer, open_sources
    torch.set_num_threads(4); model, ck = EI.load(mp)
    run_dir = os.path.dirname(os.path.dirname(os.path.realpath(mp)))              # data/derived/cache (the embed dir's parent)
    fit = dict(np.load(src_path(L['fit']))); fz = Featurizer(fit); src = open_sources()
    Z = np.full((r1 - r0, c1 - c0, ck['D']), np.nan, np.float32); T = 512
    for r in range(r0, r1, T):
        for c in range(c0, c1, T):
            h, w = min(T, r1 - r), min(T, c1 - c); win = Window(c - 1, r - 1, w + 2, h + 2)
            raw = fz.raw(src, win); z = EI.encode_tile(model, raw, fit)[1:-1, 1:-1]
            Z[r - r0:r - r0 + h, c - c0:c - c0 + w] = z
    fl = forest_mask(); v = Z.reshape(-1, Z.shape[-1]); v = v[np.isfinite(v).all(1)][::50]; mu = v.mean(0)
    _, _, Vt = np.linalg.svd(v - mu, full_matrices=False); P = (Z - mu) @ Vt[:3].T
    rgba = np.zeros((DH, DW, 4)); rgb = np.zeros((DH, DW, 3))
    for j in range(3):
        wv = warp(P[..., j].astype(np.float32), False); lo, hi = np.nanpercentile(wv[fl], [2, 98]); rgb[..., j] = np.clip((wv - lo) / (hi - lo), 0, 1) * 255
    ok = np.isfinite(rgb).all(-1) & fl; rgba[ok, :3] = rgb[ok]; rgba[ok, 3] = 255
    d = save_png(rgba, L['id']); d['legend'] = [dict(label=f"RGB = latent principal components 1 / 2 / 3 ({ck['D']}-d IWAE posterior mean, 90 m neighbourhood), forest land only", color=None)]
    d['note'] = f"embedding {os.path.relpath(os.path.realpath(mp), ROOT)}"
    return d, None

KINDS = dict(stands=stands_layer, geojson_file=geojson_layer, raster=raster_layer, freerun=freerun_layer, latent=latent_layer)
mf = f'{OUT}/manifest.json'; man = json.load(open(mf)) if os.path.exists(mf) else dict(layers=[])
old = {l['id']: l for l in man['layers']}; layers = []
for L in CFG['layers']:
    if ONLY and L['id'] not in ONLY:
        if L['id'] in old: layers.append(old[L['id']])
        continue
    d, err = KINDS[L['kind']](L)
    if d is None: print(f"skip {L['id']}: {err}", flush=True); continue
    d.update(id=L['id'], group=L['group'], label=L['label']); layers.append(d); print(f"layer {L['id']} ({d['type']}): {d.get('note', '')}", flush=True)
json.dump(dict(built=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'), layers=layers), open(mf, 'w'), indent=1)
print(f'{len(layers)} layers -> {OUT}/manifest.json')
