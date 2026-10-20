"""Generative-vegetation prototype (Baker County FL), step 1: training table.
Plots: FIA plots (Pan R4 interval product + FIADB public coordinates) within BUF km of Baker County, all visits:
  visit state = live trees at v1 of every interval + the plot's last v2 (alive tracked trees + first-tallied ingrowth); natural + planted.
  cohort tokens per visit (plot level, the 4 subplots pooled): species key (top-K in the area, else _S/_H) x 10-yr age class →
  N (trees/ha), QMD (cm), mean H (m), mean CR (fraction; NaN at a final v2), mean age; plus the visit year.
  remeasurement pairs = (visit k, visit k+1) of the same plot.
Candidates: 90 m footprints (3x3 px, stride 3) on forest land within R m of the public coordinate; per candidate: AlphaEarth 2022 + 2025 (3x3 mean,
  dequantised, 64 + 64), disturbance classes per year 2000-2025 (Hansen loss + LANDFIRE annual: harvest, thin/mechanical, fire, insects/disease,
  weather, chemical, development; LANDFIRE severity 1-3), SOLUS 0-30 cm (6).
Prior per (plot, candidate): FIA-reported dated treatments/disturbances (TRTYR*/DSTRBYR* of every visit) matched by a candidate event in year ±1 → x8;
  a candidate stand-replacing event (Hansen loss / LANDFIRE clearcut-harvest) inside the plot's undisturbed, aged history (after first-visit year −
  STDAGE + 2, with no FIA-reported event) → x0.1. Normalised per plot.
Mapping set: Baker stands (out/stands_gafl_pilot.gpkg ∩ county): up to 4 footprints per stand (largest-area interior pixels) with the same features.
Output: data/proto/baker/{plots,visits,tokens,pairs,cands,stands}.parquet + cand_feat.npy / stand_feat.npy (float16)."""
import sys, os, json, numpy as np, pandas as pd, duckdb, rasterio, pyarrow.ipc as ip, geopandas as gpd
from rasterio.windows import Window
from pyproj import Transformer
sys.path.insert(0, os.path.dirname(__file__))
from features import ROOT, SOILS, dequant
BUF = float(os.environ.get('BUF_KM', 50)); R = float(os.environ.get('RAD_M', 1600)); OUT = f'{ROOT}/data/proto/baker'; os.makedirs(OUT, exist_ok=True)
P4 = '/workspace/landis_hbad3/results/hbad2_r4'
iv = ip.open_file(f'{P4}/intervals.arrow').read_all().to_pandas(); tr = ip.open_file(f'{P4}/interval_trees.arrow').read_all().to_pandas()
ig = ip.open_file(f'{P4}/interval_ingrowth.arrow').read_all().to_pandas()
con = duckdb.connect('/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb', read_only=True)
pl = con.execute("""select STATECD st, UNITCD u, COUNTYCD c, PLOT p, avg(LAT) lat, avg(LON) lon from PLOT where STATECD in (12, 13) group by all""").df()
cond = con.execute("""select p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, p.MEASYEAR y, c.STDAGE, c.TRTCD1, c.TRTYR1, c.TRTCD2, c.TRTYR2, c.DSTRBCD1, c.DSTRBYR1, c.DSTRBCD2, c.DSTRBYR2
                      from COND c join PLOT p on c.PLT_CN = p.CN where p.STATECD in (12, 13) and c.COND_STATUS_CD = 1""").df()
cty = gpd.read_file(f'{ROOT}/data/derived/baker_county.gpkg'); land = rasterio.open(f'{ROOT}/data/derived/land_clean.tif')
cty = cty.to_crs(land.crs); geo = cty.geometry.iloc[0]
tf = Transformer.from_crs(4326, land.crs, always_xy=True)
pl['x'], pl['y'] = tf.transform(pl.lon.values, pl.lat.values)
from shapely.geometry import Point
near = gpd.GeoSeries([Point(x, y) for x, y in zip(pl.x, pl.y)], crs=land.crs).distance(geo) <= BUF * 1000
pl = pl[near.values].reset_index(drop=True); print('plots within', BUF, 'km:', len(pl), flush=True)
key = ['st', 'u', 'c', 'p']
ivp = iv.merge(pl[key], on=key); ivp['d1'] = pd.to_datetime(ivp.d1); ivp['d2'] = pd.to_datetime(ivp.d2)
t = tr[tr.iid.isin(ivp.iid)].merge(ivp[['iid', *key, 'k', 'd1', 'd2', 'planted']], on='iid')
# species keys
cnt = t.sym.value_counts(); top = list(cnt.index[:int(os.environ.get('TOPK', 12))])
kf = lambda s, g: s if s in top else '_' + g
KEYS = sorted(top) + ['_H', '_S']; kix = {k: i for i, k in enumerate(KEYS)}
def tokens(df, D, H, CR, A, W):
    d = pd.DataFrame({'key': [kix[kf(s, g)] for s, g in zip(df.sym, df.grp)], 'D': D, 'H': H, 'CR': CR, 'A': A, 'W': W})
    d = d[(d.D > 0) & (d.W > 0)]; d['ac'] = np.clip((d.A // 10).astype(int), 0, 15)
    g = d.groupby(['key', 'ac']).apply(lambda x: pd.Series({'N': x.W.sum(), 'QMD': np.sqrt((x.W * x.D ** 2).sum() / x.W.sum()), 'H': (x.W * x.H).sum() / x.W.sum(),
                                                             'CR': (x.W * x.CR).sum() / x.W.sum() if x.CR.notna().all() else np.nan, 'A': (x.W * x.A).sum() / x.W.sum()}), include_groups=False)
    return g.reset_index()
visits = []; toks = []
for pk, g in ivp.groupby(key):
    g = g.sort_values('k')
    for r in g.itertuples():
        x = t[t.iid == r.iid]
        visits.append(dict(zip(key, pk), vid=len(visits), year=r.d1.year + r.d1.dayofyear / 366, planted=bool(r.planted)))
        tk = tokens(x, x.dia_cm.values, x.ht_m.values, x.cr.values / 100, x.age.values.astype(float), x.w_ha.values); tk['vid'] = visits[-1]['vid']; toks.append(tk)
    r = g.iloc[-1]; x = t[(t.iid == r.iid) & (t.out == 'alive') & (t.dia2_cm > 0)]; gi = ig[ig.iid == r.iid]
    dt = r['dt']
    D = np.r_[x.dia2_cm.values, gi.dia_cm.values]; H = np.r_[x.ht2_m.values, gi.ht_m.values]; A = np.r_[x.age.values + dt, gi.age.values]; W = np.r_[x.w2_ha.values, gi.w_ha.values]
    xx = pd.concat([x[['sym', 'grp']], gi[['sym', 'grp']]])
    visits.append(dict(zip(key, pk), vid=len(visits), year=r.d2.year + r.d2.dayofyear / 366, planted=bool(r.planted)))
    tk = tokens(xx, D, H, np.full(len(D), np.nan), A.astype(float), W); tk['vid'] = visits[-1]['vid']; toks.append(tk)
V = pd.DataFrame(visits); TK = pd.concat(toks, ignore_index=True)
V = V.merge(pl[[*key, 'lat', 'lon', 'x', 'y']], on=key)
V['pid'] = V.groupby(key).ngroup()
pairs = V.sort_values(['pid', 'year']).groupby('pid').apply(lambda g: pd.DataFrame({'v1': g.vid.values[:-1], 'v2': g.vid.values[1:]}), include_groups=False).reset_index(level=0)
print('visits', len(V), 'tokens', len(TK), 'pairs', len(pairs), 'plots', V.pid.nunique(), 'keys', KEYS, flush=True)
# ---- disturbance classes per year
codes = pd.read_csv(f'{ROOT}/data/landfire/dist/dist_codes.csv')
CLS = {'harvest': ['Clearcut', 'Harvest'], 'thin': ['Thinning', 'Mastication', 'Other Mechanical', 'Mechanical Remove', 'Mechanical Add'],
       'fire': ['Wildfire', 'Fire', 'Prescribed Fire', 'Wildland Fire Use', 'Wildland Fire', 'Unknown/Possibly Fire'],
       'insdis': ['Insects/Disease', 'Insects', 'Disease', 'Biological'], 'weather': ['Weather'], 'chem': ['Herbicide', 'Chemical', 'Insecticide'], 'dev': ['Development']}
cid = {t_: i + 1 for i, (c_, ts) in enumerate(CLS.items()) for t_ in ts}
SEV = {'Low': 1, 'Low-Medium': 1.5, 'Medium-Low': 1.5, 'Medium': 2, 'Medium-High': 2.5, 'High-Medium': 2.5, 'High': 3}
lut = {}
for r in codes.itertuples():
    yr = int(r.year); lut[(yr, int(r.VALUE))] = (cid.get(str(r.DIST_TYPE).strip(), 0), SEV.get(str(r.SEVERITY).strip(), 1.0))
YEARS = list(range(2000, 2026)); NC = len(CLS)
dsrc = {y: rasterio.open(f'{ROOT}/data/landfire/dist/dist_{y}.tif') for y in YEARS}
hloss = rasterio.open(f'{ROOT}/data/hansen/lossyear.tif')
a22 = rasterio.open(f'{ROOT}/data/aef/aef_2022.tif'); a25 = rasterio.open(f'{ROOT}/data/aef/aef_2025.tif')
soil = {k: rasterio.open(f'{ROOT}/data/soils/{k}.tif') for k in SOILS}
NF = 64 + 64 + len(YEARS) * (NC + 2) + len(SOILS)       # AEF22, AEF25, per year (one-hot class, severity, Hansen loss), soils
def feats(r0, c0, h, w):
    """features of all 3x3 footprints in a window: returns (centres (rows, cols), F [n, NF])."""
    win = Window(c0, r0, w, h)
    A2 = dequant(a22.read(window=win)); A5 = dequant(a25.read(window=win)); L = land.read(1, window=win); HL = hloss.read(1, window=win)
    DS = np.stack([dsrc[y].read(1, window=win) for y in YEARS]); S = np.stack([soil[k].read(1, window=win).astype(np.float32) for k in SOILS])
    rr, cc = np.mgrid[1:h - 1:3, 1:w - 1:3]; rr, cc = rr.ravel(), cc.ravel()
    ok = (L[rr, cc] >= 1) & (L[rr, cc] <= 4); rr, cc = rr[ok], cc[ok]
    if len(rr) == 0: return rr, cc, np.zeros((0, NF), np.float32)
    def m3(X):   # 3x3 mean of a (k, H, W) array at the centres
        s = sum(X[:, rr + dr, cc + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)) / 9.0; return s.T
    F = [np.nan_to_num(m3(A2)), np.nan_to_num(m3(A5))]
    yrs = []
    for j, y in enumerate(YEARS):
        code = DS[j, rr, cc]; cl = np.array([lut.get((y, int(v)), (0, 0))[0] if v > 0 else 0 for v in code]); sv = np.array([lut.get((y, int(v)), (0, 0))[1] if v > 0 else 0 for v in code])
        oh = np.zeros((len(rr), NC), np.float32); oh[cl > 0, cl[cl > 0] - 1] = 1
        yrs += [oh, sv[:, None].astype(np.float32), (HL[rr, cc] == y - 2000).astype(np.float32)[:, None]]
    F += yrs; F.append(np.nan_to_num(S[:, rr, cc].T))
    return rr, cc, np.concatenate(F, 1).astype(np.float32)
# ---- candidates per plot
fia_ev = {}
for r in cond.itertuples():
    k = (r.st, r.u, r.c, r.p)
    for code, yr in ((r.TRTCD1, r.TRTYR1), (r.TRTCD2, r.TRTYR2), (r.DSTRBCD1, r.DSTRBYR1), (r.DSTRBCD2, r.DSTRBYR2)):
        if pd.notna(code) and code > 0 and pd.notna(yr) and 2000 <= yr <= 2025: fia_ev.setdefault(k, set()).add(int(yr))
first = V.groupby('pid').year.min(); age0 = cond.groupby(['st', 'u', 'c', 'p']).STDAGE.max()
RP = int(R // 30) + 2; cand_rows = []; CF = []
for pid, g in V.groupby('pid'):
    r = g.iloc[0]; row, col = land.index(r.x, r.y)
    rr, cc, F = feats(row - RP, col - RP, 2 * RP + 1, 2 * RP + 1)
    d = np.hypot(rr - RP, cc - RP) * 30; m = d <= R; rr, cc, F, d = rr[m], cc[m], F[m], d[m]
    if len(rr) == 0: continue
    ev = fia_ev.get((r.st, r.u, r.c, r.p), set())
    per = F[:, 128:128 + len(YEARS) * (NC + 2)].reshape(len(rr), len(YEARS), NC + 2)
    sr = (per[:, :, 0] + per[:, :, 1] + per[:, :, NC + 1] > 0)                     # stand-replacing-ish: harvest / thin / Hansen loss
    w = np.ones(len(rr))
    for yv in ev:
        j = yv - 2000; hit = sr[:, max(j - 1, 0):j + 2].any(1); w *= np.where(hit, 8.0, 1.0)
    a0 = age0.get((r.st, r.u, r.c, r.p), np.nan)
    if pd.notna(a0) and not ev:
        since = int(first[pid] - a0 + 2) - 2000
        if since < len(YEARS): w *= np.where(sr[:, max(since, 0):int(first[pid]) - 2000 + 1].any(1), 0.1, 1.0)
    w /= w.sum()
    cand_rows.append(pd.DataFrame({'pid': pid, 'cid': np.arange(len(CF) and sum(len(f) for f in CF), sum(len(f) for f in CF) + len(rr)), 'dist_m': d, 'prior': w}))
    CF.append(F.astype(np.float16))
C = pd.concat(cand_rows, ignore_index=True); CFa = np.concatenate(CF)
print('candidates', len(C), 'per plot median', C.groupby('pid').size().median(), 'features', CFa.shape, flush=True)
# ---- Baker stands: up to 4 footprints per stand
st = gpd.read_file(f'{ROOT}/out/stands_gafl_pilot.gpkg', layer='stands'); st = st[st.intersects(geo)]
sraster = rasterio.open(f'{ROOT}/data/derived/stands.tif')
x0, y0, x1, y1 = st.total_bounds; r0, c0 = sraster.index(x0, y1); r1, c1 = sraster.index(x1, y0)
SR = sraster.read(1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1))
rr, cc, F = feats(r0, c0, r1 - r0 + 1, c1 - c0 + 1)
sid = SR[rr, cc]; keep = np.isin(sid, st.stand_id.values); rr, cc, F, sid = rr[keep], cc[keep], F[keep], sid[keep]
rng = np.random.default_rng(0); sel = []
for s_, idx in pd.Series(np.arange(len(sid))).groupby(sid):
    sel += list(rng.choice(idx.values, min(4, len(idx)), replace=False))
sel = np.array(sel); S_ = pd.DataFrame({'stand_id': sid[sel], 'row': rr[sel] + r0, 'col': cc[sel] + c0})
print('stand footprints', len(S_), 'stands', S_.stand_id.nunique(), flush=True)
pd.DataFrame({'pid': V.pid.unique()}).merge(V.drop_duplicates('pid')[['pid', *key, 'lat', 'lon', 'x', 'y']], on='pid').to_parquet(f'{OUT}/plots.parquet')
V.to_parquet(f'{OUT}/visits.parquet'); TK.to_parquet(f'{OUT}/tokens.parquet'); pairs.to_parquet(f'{OUT}/pairs.parquet')
C.to_parquet(f'{OUT}/cands.parquet'); np.save(f'{OUT}/cand_feat.npy', CFa); S_.to_parquet(f'{OUT}/stands.parquet'); np.save(f'{OUT}/stand_feat.npy', F[sel].astype(np.float16))
json.dump(dict(keys=KEYS, years=YEARS, classes=list(CLS), soils=SOILS, nf=NF, buf_km=BUF, radius_m=R), open(f'{OUT}/meta.json', 'w'))
print('done')
