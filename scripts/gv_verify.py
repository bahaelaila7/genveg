"""Independent verification of the generative-vegetation pipeline (user: "triple check again. I don't trust it yet").
Every check recomputes a stage from the raw sources by a DIFFERENT code path (SQL instead of pandas, direct raster reads at stored coordinates, FIADB COND
instead of curated_trees for events) and compares. Output: PASS / FAIL lines + numbers; exit code 1 if anything fails.
  python scripts/gv_verify.py [n_sample=300]"""
import sys, os, json, math, numpy as np, pandas as pd, duckdb, rasterio
from rasterio.windows import Window
from pyproj import Transformer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from stand_runs import stands_run
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); WS = os.environ.get('WORKSPACE', os.path.dirname(ROOT)); D = f"{ROOT}/data/proto/{os.environ.get('DATA', 'gafl')}"; NS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng = np.random.default_rng(42); RES = []
def check(name, ok, detail=''):
    RES.append((name, bool(ok))); print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
M = json.load(open(f'{D}/meta.json')); from region import region_check; region_check(M); KEYS = M['keys']; YEARS = M['years']; NC = len(M['classes']); NY = len(YEARS)
# region switches (as g02b): REGION = FIA STATECDs; AEF90 = Region 4 rasters (AlphaEarth / soils at 90 m: a footprint = one 90 m pixel, centres = 1 mod 3)
REGION = tuple(int(x) for x in os.environ.get('REGION', '12,13').split(',')); ST_SQL = f"({','.join(map(str, REGION))})"; AEF90 = os.environ.get('AEF90', '0') == '1'
LAND = f'{ROOT}/data/r4/land.tif' if AEF90 else stands_run()['land_clean']
AEFP = {y: f'{ROOT}/data/aef90/aef90_{y}.tif' if AEF90 else f'{ROOT}/data/aef/aef_{y}.tif' for y in (2022, 2025)}
SOILP = (lambda k: f'{ROOT}/data/r4/soils90/{k}.tif') if AEF90 else (lambda k: f'{ROOT}/data/soils/{k}.tif')
DISTP = (lambda y: f'{ROOT}/data/r4/dist/dist_{y}.tif') if AEF90 else (lambda y: f'{ROOT}/data/landfire/dist/dist_{y}.tif')
HANSP = f'{ROOT}/data/r4/hansen_lossyear.tif' if AEF90 else f'{ROOT}/data/hansen/lossyear.tif'
key = ['st', 'u', 'c', 'p']; V = pd.read_parquet(f'{D}/visits.parquet'); TK = pd.read_parquet(f'{D}/tokens.parquet'); PR = pd.read_parquet(f'{D}/pairs.parquet')
P = pd.read_parquet(f'{D}/plots.parquet'); C = pd.read_parquet(f'{D}/cands.parquet'); CF = np.load(f'{D}/cand_feat.npy', mmap_mode='r')
con = duckdb.connect(f'{WS}/landis_region4/data_r4/slim_r4.duckdb', read_only=True)
# ======================= T1 visits and cohort tokens (SQL aggregation, independent of the pandas builder)
top = [r[0] for r in con.execute(f"""select r.SPECIES_SYMBOL from curated_trees t join REF_SPECIES r on r.SPCD=t.SPCD where t.STATECD in {ST_SQL} and t.MEASDATE>='2000-01-01'
                                    and t.STATUSCD=1 group by 1 order by count(*) desc, 1 limit {len(KEYS) - 2}""").fetchall()]
check(f'T1b species keys = top-{len(KEYS) - 2} live species (SQL) + _H/_S', sorted(top) + ['_H', '_S'] == KEYS, f'{sorted(top)[:5]}…')
samp = V.sample(min(NS, len(V)), random_state=1)
reg = duckdb.connect(); reg.register('samp', samp[key + ['vid']].assign(yr=samp.year))
q = f"""select t.STATECD st, t.UNITCD u, t.COUNTYCD c, t.PLOT p, t.MEASDATE md, r.SPECIES_SYMBOL sym, r.SFTWD_HRDWD grp, t.DIA dia, t.HT ht, t.CR cr, t.TPA_UNADJ tpa,
              t.estimated_age age, t.STATUSCD status, t.cond_stdorgcd org, t.CONDID cond
       from curated_trees t join REF_SPECIES r on r.SPCD = t.SPCD where t.STATECD in {ST_SQL} and t.MEASDATE >= '2000-01-01'"""
T = con.execute(q).df(); T['md'] = pd.to_datetime(T.md)
# ---- the visit set, rebuilt independently from FIADB (pandas aggregation of per-condition rows; string dates):
#      visits with curated trees on a forested condition  ∪  forested FIADB visits of the same plots with no live tree in FIADB TREE;
#      forested share of the SAMPLED area (non-sampled status 5 excluded) >= 0.1
_fx = duckdb.connect(f'{WS}/FIA/FIASQLITE2PGSQL/FIADB.duckdb', read_only=True)
_cc = _fx.execute(f"""select p.CN pcn, p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, printf('%04d-%02d-%02d', p.MEASYEAR::int, p.MEASMON::int, p.MEASDAY::int) ds,
        c.CONDID cond, c.COND_STATUS_CD cs, c.SUBPPROP_UNADJ sp, c.MICRPROP_UNADJ mp, c.STDORGCD org from PLOT p join COND c on c.PLT_CN = p.CN
        where p.STATECD in {ST_SQL} and p.MEASYEAR >= 2000""").df()
_nl = _fx.execute(f"select PLT_CN pcn, count(*) nl from TREE where STATECD in {ST_SQL} and STATUSCD = 1 group by 1").df()
_f = _cc.assign(sp1=np.where(_cc.cs == 1, _cc.sp, 0), sp5=np.where(_cc.cs == 5, _cc.sp, 0), mp1=np.where(_cc.cs == 1, _cc.mp, 0), mp5=np.where(_cc.cs == 5, _cc.mp, 0),
                org1=np.where(_cc.cs == 1, _cc.org, np.nan)).groupby(['pcn'] + key + ['ds']).agg(sp1=('sp1', 'sum'), sp5=('sp5', 'sum'), mp1=('mp1', 'sum'), mp5=('mp5', 'sum'), org=('org1', 'max')).reset_index()
_f = _f.merge(_nl, on='pcn', how='left').fillna({'nl': 0}); _f = _f[_f.sp1 > 0]
_f['fs'] = np.minimum(_f.sp1 / (1 - _f.sp5), 1.0); _f['fm'] = np.where(_f.mp1 > 0, np.minimum(_f.mp1 / (1 - _f.mp5), 1.0), _f.fs)      # shares of the SAMPLED area (visit filter)
_f['fsw'] = np.minimum(_f.sp1, 1.0); _f['fmw'] = np.where(_f.mp1 > 0, np.minimum(_f.mp1, 1.0), _f.fsw)          # shares of the WHOLE plot: TPA_UNADJ expands to it (densities)
T['ds'] = T.md.dt.strftime('%Y-%m-%d'); T = T.merge(_cc[key + ['ds', 'cond', 'cs']], on=key + ['ds', 'cond'], how='left'); nT = len(T); T = T[T.cs == 1]
_tv = T[key + ['ds']].drop_duplicates(); _bare = _f[_f.nl == 0].merge(_tv[key].drop_duplicates(), on=key).merge(_tv.assign(_i=1), on=key + ['ds'], how='left')
_bare = _bare[_bare._i.isna()][key + ['ds']]
EX = pd.concat([_tv, _bare]).merge(_f[key + ['ds', 'fs', 'fm', 'fsw', 'fmw', 'org']], on=key + ['ds'], how='inner'); EX = EX[EX.fs >= 0.1]
EX['md'] = pd.to_datetime(EX.ds); EX = EX.sort_values(key + ['md']).reset_index(drop=True); EX['vid'] = np.arange(len(EX)); EX['year'] = EX.md.dt.year + (EX.md.dt.dayofyear - 0.5) / np.where(EX.md.dt.is_leap_year, 366.0, 365.0)
check('T1a visit set = curated forested-condition visits ∪ bare FIADB visits, share >= 0.1 (rebuilt from FIADB)', len(EX) == len(V),
      f'independent {len(EX)} (bare {len(_bare)}) table {len(V)}; non-forest-condition tree records {nT - len(T)}')
Vs = V.sort_values('vid')
check('T1c visit ids reproduce (sorted plot, date order)', len(EX) == len(V) and (EX[key].values == Vs[key].values).all() and (np.abs(EX.year.values - Vs.year.values) < 1e-9).all())
T = T.merge(EX[key + ['ds', 'vid', 'fs', 'fm']], on=key + ['ds']); Ts = T[T.vid.isin(samp.vid)]
live = Ts[(Ts.status == 1) & (Ts.dia > 0) & (Ts.tpa > 0) & (Ts.age > 0) & (Ts.ht > 0)].copy()
live['key'] = [KEYS.index(s) if s in top else KEYS.index('_' + (g if g in ('S', 'H') else 'H')) for s, g in zip(live.sym, live.grp)]
live['share'] = np.where(live.dia < 5, live.fmw, live.fsw)
live['ac'] = np.clip((live.age // 10).astype(int), 0, 15); live['w'] = live.tpa * 2.47105 / live.share; live['D'] = live.dia * 2.54; live['H'] = live.ht * 0.3048
print(f'   (sampled visits with forested share < 0.95: {(live.groupby("vid").fs.first() < 0.95).mean():.3f})')
agg = live.groupby(['vid', 'key', 'ac']).apply(lambda x: pd.Series({'N': x.w.sum(), 'QMD': math.sqrt((x.w * x.D ** 2).sum() / x.w.sum()), 'H': (x.w * x.H).sum() / x.w.sum()}), include_groups=False).reset_index()
tk = TK[TK.vid.isin(samp.vid)].merge(agg, on=['vid', 'key', 'ac'], how='outer', suffixes=('', '_ind'), indicator=True)
check('T1d token set identical to independent aggregation (vid, key, age class)', (tk._merge == 'both').all(), f"{tk._merge.value_counts().to_dict()}")
b = tk[tk._merge == 'both']
check('T1e token N / QMD / H equal (rtol 1e-6)', np.allclose(b.N, b.N_ind, rtol=1e-6) and np.allclose(b.QMD, b.QMD_ind, rtol=1e-6) and np.allclose(b.H, b.H_ind, rtol=1e-6),
      f'max rel N {np.max(np.abs(b.N / b.N_ind - 1)):.2e} QMD {np.max(np.abs(b.QMD / b.QMD_ind - 1)):.2e}')
emp = set(samp.vid) - set(live.vid); tkv = set(TK[TK.vid.isin(samp.vid)].vid)
check('T1f visits without qualifying live trees have no tokens (and are kept)', not (emp & tkv) and emp <= set(V.vid), f'empty in sample {len(emp)}')
ba_ind = live.groupby('vid').apply(lambda x: (x.tpa / x.share * 0.005454154 * x.dia ** 2).sum() * 0.229568, include_groups=False)
tt = TK[TK.vid.isin(ba_ind.index)]; ba_tok = tt.groupby('vid').apply(lambda x: (x.N * math.pi / 4 * (x.QMD / 100) ** 2).sum(), include_groups=False)
check('T1g plot BA per forested area (FIA formula ft2/ac -> m2/ha) = BA from tokens', np.allclose(ba_ind.sort_index(), ba_tok.reindex(ba_ind.index).values, rtol=1e-6), f'n {len(ba_ind)}')
pv = V.sort_values(['pid', 'year']); exp_pairs = set(zip(pv.vid.values[:-1], pv.vid.values[1:])) - {(a, b_) for a, b_, p1, p2 in zip(pv.vid.values[:-1], pv.vid.values[1:], pv.pid.values[:-1], pv.pid.values[1:]) if p1 != p2}
check('T1h pairs = consecutive visits of each plot', set(zip(PR.v1, PR.v2)) == exp_pairs, f'{len(PR)} pairs')
pl_ind = EX.set_index('vid').org.fillna(0) == 1
check('T1i planted flag = COND STDORGCD 1 (forested conditions) of the visit', (V.set_index('vid').planted.reindex(pl_ind.index) == pl_ind).all(), f'planted share {pl_ind.mean():.3f}')
fv = duckdb.connect(f'{WS}/FIA/FIASQLITE2PGSQL/FIADB.duckdb', read_only=True).execute(f"""select distinct p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, p.MEASYEAR my
      from PLOT p join COND c on c.PLT_CN = p.CN where p.STATECD in {ST_SQL} and p.MEASYEAR >= 2000 and c.COND_STATUS_CD = 1""").df()
ours = V[key].assign(my=V.year.astype(int)).drop_duplicates(); inplots = fv.merge(V[key].drop_duplicates(), on=key)
miss = inplots.merge(ours, on=key + ['my'], how='left', indicator=True); nm = (miss._merge == 'left_only').sum()
nplots_f = len(fv[key].drop_duplicates()); nplots_o = len(V[key].drop_duplicates())
check('T1j FIADB forested visits of our plots that are absent from the table (bare visits with no tree record?) < 2 %', nm / len(inplots) < 0.02,
      f'missing {nm} of {len(inplots)}; forested FIADB plots {nplots_f}, in table {nplots_o} (curated product subset)')
# ======================= T2 plot coordinates
fc = duckdb.connect(f'{WS}/FIA/FIASQLITE2PGSQL/FIADB.duckdb', read_only=True)
pl = fc.execute(f"select STATECD st, UNITCD u, COUNTYCD c, PLOT p, MEASYEAR my, LAT lat, LON lon from PLOT where STATECD in {ST_SQL} and MEASYEAR >= 2000").df()
_sp = pl.groupby(key).agg(la0=('lat', 'min'), la1=('lat', 'max'), lo0=('lon', 'min'), lo1=('lon', 'max')).reset_index()
check('T2a0 a plot key\'s annual-inventory rows (MEASYEAR >= 2000) agree on the coordinates (< 0.2 km)', ((_sp.la1 - _sp.la0) * 111 < 0.2).mean() > 0.999 and ((_sp.lo1 - _sp.lo0) * 92 < 0.2).mean() > 0.999,
      f'{len(_sp)} keys; spread > 0.2 km: {int((((_sp.la1 - _sp.la0) * 111 >= 0.2) | ((_sp.lo1 - _sp.lo0) * 92 >= 0.2)).sum())}')
pl = pl.groupby(key).agg(lat=('lat', 'mean'), lon=('lon', 'mean')).reset_index()
m = P.merge(pl, on=key, suffixes=('', '_f'))
check('T2a plot lat/lon = FIADB PLOT mean of the annual-inventory rows', np.allclose(m.lat, m.lat_f) and np.allclose(m.lon, m.lon_f), f'n {len(m)}')
land = rasterio.open(LAND); tf = Transformer.from_crs(4326, land.crs, always_xy=True); x, y = tf.transform(m.lon.values, m.lat.values)
check('T2b projected x/y (EPSG:5070) recomputed', np.allclose(x, m.x, atol=1e-3) and np.allclose(y, m.y, atol=1e-3))
if AEF90:          # point in the state's polygon (TIGER counties dissolved), 2 km tolerance (FIA coordinate fuzz)
    import geopandas as gpd
    stp = gpd.read_file(f'{WS}/TIGER/tl_2024_us_county/tl_2024_us_county.shp'); stp = stp[stp.STATEFP.astype(int).isin(REGION)].to_crs(land.crs).dissolve('STATEFP').geometry
    pts = gpd.GeoSeries(gpd.points_from_xy(x, y), crs=land.crs); okst = np.mean([stp[f'{st_:02d}'].distance(pt) <= 2000 for st_, pt in zip(m.st.astype(int), pts)])
    check('T2c plots lie in their state (state polygon, 2 km tolerance)', okst > 0.995, f'{okst:.4f}')
else:
    okst = ((m.st == 12) & (m.lat < 31.1) | (m.st == 13) & (m.lat > 30.3)).mean()
    check('T2c plots lie in their state (latitude band check)', okst > 0.99, f'{okst:.4f}')
# ======================= T3 candidates: direct raster reads at stored coordinates
pids = rng.choice(C.pid.unique(), min(60, C.pid.nunique()), replace=False); Cs = C[C.pid.isin(pids)]
Pi = P.set_index('pid')
rows_ok = []; dist_err = []
for pid, g in Cs.groupby('pid'):
    r0, c0 = land.index(Pi.loc[pid].x, Pi.loc[pid].y); dist_err.append(np.max(np.abs(np.hypot(g.row - r0, g.col - c0) * 30 - g.dist_m)))
check('T3a stored dist_m = distance(plot pixel, candidate pixel); all <= 1600 m', max(dist_err) < 1e-3 and (Cs.dist_m <= 1600).all(), f'max err {max(dist_err):.2e}')
lc = np.array([land.read(1, window=Window(c, r, 1, 1))[0, 0] for r, c in zip(Cs.row, Cs.col)])
check('T3b every candidate centre is forest land (classes 1-4)', ((lc >= 1) & (lc <= 4)).all(), f'classes {np.unique(lc)}')
sub = Cs.sample(min(400, len(Cs)), random_state=3); X = np.asarray(CF[sub.cid.values], dtype=np.float32)
a22 = rasterio.open(AEFP[2022]); a25 = rasterio.open(AEFP[2025])
def dq(a): v = a.astype(np.float32); o = np.sign(v) * (v / 127.5) ** 2; o[a == -128] = np.nan; return o
def aef_fp(src, r, c):             # the footprint's AlphaEarth: 3x3 mean of the 30 m mosaic, or (AEF90) the 90 m pixel of its block
    if AEF90: return np.nan_to_num(dq(src.read(window=Window(c // 3, r // 3, 1, 1))).reshape(64))
    return np.nan_to_num(np.nanmean(dq(src.read(window=Window(c - 1, r - 1, 3, 3))).reshape(64, -1), 1))
if AEF90: check('T3c0 footprint centres are 90 m block centres (row, col = 1 mod 3)', ((C.row % 3 == 1) & (C.col % 3 == 1)).all())
e22 = np.stack([aef_fp(a22, r, c) for r, c in zip(sub.row, sub.col)]); e25 = np.stack([aef_fp(a25, r, c) for r, c in zip(sub.row, sub.col)])
err = max(np.abs(e22 - X[:, :64]).max(), np.abs(e25 - X[:, 64:128]).max())
check('T3c AlphaEarth 2022 / 2025 = 3x3 mean of the mosaics at the stored centre' + (' (AEF90: the 90 m pixel)' if AEF90 else ''), err < 2e-3, f'max abs err {err:.2e} (float16 storage)')
if AEF90:          # the 90 m route against the GA+FL 30 m mosaics (independent: 3x3 means of 02_aef.py's 30 m product), candidates inside that extent
    g30 = {y: rasterio.open(f'{ROOT}/data/aef/aef_{y}.tif') for y in (2022, 2025)}; tl = land.transform
    ci = C[C.st.isin([12, 13])] if 'st' in C.columns else C.merge(P[['pid', 'st']], on='pid').query('st in (12, 13)')
    ci = ci.sample(min(300, len(ci)), random_state=4); e3, e9 = [], []
    for r, c, cid_ in zip(ci.row, ci.col, ci.cid):
        xx, yy = tl * (c + 0.5, r + 0.5); r3, c3 = g30[2025].index(xx, yy)
        if not (1 <= r3 < g30[2025].height - 1 and 1 <= c3 < g30[2025].width - 1): continue
        m30 = np.nanmean(dq(g30[2025].read(window=Window(c3 - 1, r3 - 1, 3, 3))).reshape(64, -1), 1)
        if np.isnan(m30).any(): continue
        e3.append(m30); e9.append(np.asarray(CF[cid_, 64:128], dtype=np.float32))
    e3, e9 = np.array(e3), np.array(e9); cs = (e3 * e9).sum(1) / (np.linalg.norm(e3, axis=1) * np.linalg.norm(e9, axis=1) + 1e-9)
    check('T3c2 AEF90 footprint (2025) = 3x3 mean of the GA+FL 30 m mosaic (re-quantisation tolerance)', len(e3) > 50 and np.median(np.abs(e3 - e9).max(1)) < 0.02 and np.quantile(cs, 0.05) > 0.98,
          f'{len(e3)} footprints; median max|Δ| {np.median(np.abs(e3 - e9).max(1)):.4f}, cosine q05 {np.quantile(cs, .05):.4f}')
SOILS = M['soils']; FILL = json.load(open(f'{ROOT}/data/r4/soils90/fill.json')) if AEF90 else {'clay030': 7.58, 'sand030': 83.2, 'soc030': 0.964, 'awc030': 0.052, 'ph030': 5.18, 'restrdepth': 200.0}
SDIV = 3 if AEF90 else 1               # (AEF90: soils on the 90 m grid)
so = np.array([[(lambda v: FILL[k] if v <= -9000 else v)(float(rasterio.open(SOILP(k)).read(1, window=Window(c // SDIV, r // SDIV, 1, 1))[0, 0])) for k in SOILS]
               for r, c in zip(sub.row[:120], sub.col[:120])])
Xs = X[:120, -len(SOILS):]; se = np.abs(so - Xs) / (np.abs(so) * 1e-3 + 1e-3)
check('T3d soils = centre pixel, nodata -> regional median (float16 storage)', se.max() < 1 and (np.asarray(CF[::97, -len(SOILS):]) > -9000).all(), f'max err / tol {se.max():.2f}; no -9999 left')
# --- LANDFIRE: independent decoding of the code tables (type; burned 'Unknown' -> fire; 'Increased Green' is no disturbance; event year = CALENDAR_YR
#     of the fiscal-year products, else the file year), evidence = fraction of the footprint's 3x3 pixels
codes = pd.read_csv(f'{ROOT}/data/landfire/dist/dist_codes.csv', low_memory=False); CLS = M['classes']
cmap = {'harvest': {'Clearcut', 'Harvest'}, 'thin': {'Thinning', 'Mastication', 'Other Mechanical', 'Mechanical Remove', 'Mechanical Add', 'Mechanical Unknown'},
        'fire': {'Wildfire', 'Fire', 'Prescribed Fire', 'Wildland Fire Use', 'Wildland Fire', 'Unknown/Possibly Fire'}, 'insdis': {'Insects/Disease', 'Insects', 'Disease'},
        'weather': {'Weather'}, 'chem': {'Herbicide', 'Chemical', 'Insecticide', 'Biological'}, 'dev': {'Development'}, 'unknown': {'Unknown', 'Unknown/Fire Doubtful'}}
codes['desc'] = (codes.DESCRIPTION.fillna('') + ' ' + codes.DESCRIPTIO.fillna('')).str.replace(r'\s+', ' ', regex=True)
codes['ey'] = codes.CALENDAR_YR.fillna(codes.year).astype(int)
def cls_of(r):
    t = str(r.DIST_TYPE).strip()
    if 'Green' in str(r.SEVERITY): return None
    if t == 'Unknown' and any(p in r.desc for p in ('indicates burned', 'burn probability is 95% or greater', 'most likely fire')): t = 'Wildfire'
    return next((i for i, k in enumerate(CLS) if t in cmap[k]), None)
lk = {(int(r.year), int(r.VALUE)): (cls_of(r), int(r.ey)) for r in codes.itertuples()}
print(f"   (fiscal-year codes placed in the previous calendar year: {sum(1 for (fy, _), (c_, ey) in lk.items() if c_ is not None and ey != fy)})")
hl = rasterio.open(HANSP); LF = {yr: rasterio.open(DISTP(yr)) for yr in YEARS}; bad = 0; nev = 0; nh = 0; maxd = 0
for (r, c), xrow in zip(zip(sub.row[:150], sub.col[:150]), X[:150]):
    per = xrow[128:128 + NY * (NC + 2)].reshape(NY, NC + 2); hw = hl.read(1, window=Window(c - 1, r - 1, 3, 3)); exp = np.zeros((NY, NC))
    for yr in YEARS:
        win = LF[yr].read(1, window=Window(c - 1, r - 1, 3, 3))
        for code in win.ravel():
            if code > 0:
                c_, ey = lk.get((yr, int(code)), (None, yr))
                if c_ is not None and YEARS[0] <= ey <= YEARS[-1]: exp[ey - YEARS[0], c_] += 1 / 9
    exp = np.minimum(exp, 1); he = np.array([((hw == yr - 2000) & (hw > 0)).sum() / 9 for yr in YEARS])
    dd_ = max(np.abs(per[:, :NC] - exp).max(), np.abs(per[:, NC + 1] - he).max()); maxd = max(maxd, dd_); bad += int(dd_ > 2e-3)
    nev += int((exp > 0).sum()); nh += int((he > 0).sum())
check('T3e disturbance fractions (3x3) at the event calendar year (LANDFIRE) and Hansen fractions = direct raster lookups', bad == 0,
      f'150 candidates x {NY} years, class-years with evidence {nev}, Hansen-years {nh}, max |Δ| {maxd:.1e}, candidates off {bad}')
# --- prior: re-implemented from the SPECIFICATION (not the code): FIADB COND events by type, records of one physical event (years chaining within 2 yr)
#     merged into one cluster (union of evidence and windows, max likelihood ratio, ×2 for a cut matched by a harvest class); negative periods per FIA
#     protocol (since the previous measurement, PREV_PLT_CN, or the 5-yr lookback of a new plot) for visits without a code that could hide stand-replacing
#     evidence; stand-replacing = harvest class or Hansen loss on >= half the footprint -> ×0.25. Stored prior must be ∝ w x incl_w.
ix = {k: i for i, k in enumerate(CLS)}; HH = NC + 1
# the STRICT data-derived rules (g09; 2026-10-22), written out independently of g02b.EVK
SPEC = {('T', 10): (['harvest', 'thin', 'unknown', 'H'], (-1, 1), 4), ('T', 20): (['harvest', 'chem', 'fire', 'unknown', 'H'], (-1, 1), 4),     # cutting / site prep / regeneration
        ('T', 30): (['harvest', 'chem', 'unknown', 'H'], (-2, 1), 4), ('T', 40): (['harvest', 'fire', 'unknown', 'H'], (-2, 1), 4),           # accept LANDFIRE 'unknown' (2026-10-22)
        ('T', 50): (['thin', 'chem'], (-1, 1), 2)}
for cd in range(10, 23): SPEC[('D', cd)] = (['insdis'], (-1, 1), 2)
for cd in (30, 31): SPEC[('D', cd)] = (['fire'], (-1, 1), 2)
SPEC[('D', 32)] = (['fire', 'H'], (-1, 1), 4)
for cd in range(50, 55): SPEC[('D', cd)] = (['weather'], (-1, 1), 2)
SPEC[('D', 80)] = (['harvest', 'dev'], (-1, 1), 2)          # DSTRBCD 70 (unknown) is not matched
HARMLESS = {12, 22, 31, 40, 41, 42, 43, 44, 45, 46, 60}
evr = _fx.execute(f"""select p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, p.MEASYEAR my, p.MEASMON mm, p.MEASDAY dd, c.COND_STATUS_CD cs, q.MEASYEAR pmy,
                    c.TRTCD1, c.TRTYR1, c.TRTCD2, c.TRTYR2, c.TRTCD3, c.TRTYR3, c.DSTRBCD1, c.DSTRBYR1, c.DSTRBCD2, c.DSTRBYR2, c.DSTRBCD3, c.DSTRBYR3
                    from COND c join PLOT p on c.PLT_CN = p.CN left join PLOT q on q.CN = p.PREV_PLT_CN where p.STATECD in {ST_SQL} and p.MEASYEAR >= 2000""").df()
evr = evr[evr.cs == 1]; evr['vk'] = evr.my * 10000 + evr.mm * 100 + evr.dd
ratios, nneg, ncl = [], 0, 0; REC = {}
for pid, g in Cs.groupby('pid'):
    k_ = tuple(int(v) for v in Pi.loc[pid][key]); e = evr[(evr.st == k_[0]) & (evr.u == k_[1]) & (evr.c == k_[2]) & (evr.p == k_[3])]
    Xg = np.asarray(CF[g.cid.values], dtype=np.float32)[:, 128:128 + NY * (NC + 2)].reshape(len(g), NY, NC + 2); w = np.ones(len(g))
    recs = set(); hiding = set()
    for r in e.itertuples():
        for t_, a_, b_ in (('T', 'TRTCD1', 'TRTYR1'), ('T', 'TRTCD2', 'TRTYR2'), ('T', 'TRTCD3', 'TRTYR3'), ('D', 'DSTRBCD1', 'DSTRBYR1'), ('D', 'DSTRBCD2', 'DSTRBYR2'), ('D', 'DSTRBCD3', 'DSTRBYR3')):
            cd, yr = getattr(r, a_), getattr(r, b_)
            if pd.isna(cd) or cd <= 0: continue
            if t_ == 'T' or int(cd) not in HARMLESS: hiding.add(r.vk)
            if (t_, int(cd)) in SPEC and pd.notna(yr) and 2000 <= yr <= 2025: recs.add((int(yr), t_, int(cd)))
    cl = []                                       # chain records by year (gap <= 2)
    for yr, t_, cd in sorted(recs):
        cols_, (lo, hi), lr = SPEC[(t_, cd)]; cs_ = {HH if k == 'H' else ix[k] for k in cols_}
        if cl and yr - cl[-1][4] <= 2: a = cl[-1]; cl[-1] = (a[0] | cs_, min(a[1], yr + lo), max(a[2], yr + hi), max(a[3], lr), yr, a[5] or (t_ == 'T' and cd == 10))
        else: cl.append((cs_, yr + lo, yr + hi, lr, yr, t_ == 'T' and cd == 10))
    for cs_, lo, hi, lr, _, cut in cl:
        a, b = max(lo - 2000, 0), min(hi - 2000, NY - 1)
        if b < a: continue
        w = w * np.where(Xg[:, a:b + 1][:, :, sorted(cs_)].sum(-1).sum(-1) > 0, lr, 1.0)
        if cut: w = w * np.where(Xg[:, a:b + 1, ix['harvest']].sum(-1) > 0, 2.0, 1.0)
    ncl += len(cl)
    vv = e.groupby('vk').agg(my=('my', 'first'), pmy=('pmy', 'first')).sort_index(); srv = np.maximum(Xg[:, :, ix['harvest']], Xg[:, :, HH]) >= 0.5; per_ = []
    for j in range(len(vv)):
        if vv.index[j] in hiding: continue
        if j > 0: per_.append((int(vv.my.iloc[j - 1]) + 1, int(vv.my.iloc[j]) - 1))
        elif pd.notna(vv.pmy.iloc[0]): per_.append((int(vv.pmy.iloc[0]) + 1, int(vv.my.iloc[0]) - 1))
        else: per_.append((int(vv.my.iloc[0]) - 5, int(vv.my.iloc[0]) - 1))
    for a, b in per_:
        a, b = max(a - 2000, 0), min(b - 2000, NY - 1)
        if b >= a: w = w * np.where(srv[:, a:b + 1].any(1), 0.25, 1.0); nneg += 1
    # radial location prior (FIA perturbation: RADIAL_P of the mass within 0.5 mile, uniform per zone; written out from the spec, not imported)
    rp_ = float(M.get('radial_p', 0.9)); rad = np.where(g.dist_m.values <= 804.672, rp_ / (np.pi * 804.672 ** 2), (1 - rp_) / (np.pi * (1600.0 ** 2 - 804.672 ** 2)))
    q = g.prior.values / (w * rad * g.incl_w.values); ratios.append(q.max() / q.min()); REC[pid] = (cl, per_)
check('T3f prior ∝ (clustered typed FIA events x FIA-protocol negative evidence) x radial location prior x incl_w — re-implemented from the spec on FIADB COND', max(ratios) < 1 + 2e-3,
      f'{len(ratios)} plots ({ncl} event clusters, {nneg} negative periods), max ratio spread {max(ratios):.5f}')
# --- semantic alignment of the time axes: AlphaEarth change 2022->2025 must concentrate on candidates whose stand-replacing evidence falls in 2023-2025
smp = np.sort(rng.choice(len(C), 200000, replace=False)); XX = np.asarray(CF[smp], dtype=np.float32)
cosv = (XX[:, :64] * XX[:, 64:128]).sum(1) / (np.linalg.norm(XX[:, :64], axis=1) * np.linalg.norm(XX[:, 64:128], axis=1) + 1e-9)
pp = XX[:, 128:128 + NY * (NC + 2)].reshape(-1, NY, NC + 2); anyd = (pp[:, :, :NC].sum(-1) + pp[:, :, HH]) > 0
lfm = pp[:, :, [ix['harvest'], ix['unknown']]].max(-1) >= 0.5; hm_ = pp[:, :, HH] >= 0.5; single = anyd.sum(1) == 1      # one disturbed year only
c_clean = float(np.median(cosv[~anyd[:, 2015 - 2000:].any(1)]))
tabL = {yr: (int((lfm[:, yr - 2000] & single).sum()), round(float(np.median(cosv[lfm[:, yr - 2000] & single])), 3) if (lfm[:, yr - 2000] & single).sum() > 20 else None) for yr in range(2017, 2026)}
tabH = {yr: (int((hm_[:, yr - 2000] & single).sum()), round(float(np.median(cosv[hm_[:, yr - 2000] & single])), 3) if (hm_[:, yr - 2000] & single).sum() > 20 else None) for yr in range(2017, 2025)}
pre = [v[1] for y, v in tabL.items() if y <= 2021 and v[1] is not None] + [v[1] for y, v in tabH.items() if y <= 2021 and v[1] is not None]
post = [v[1] for y, v in tabL.items() if y in (2023, 2024) and v[1] is not None] + [v[1] for y, v in tabH.items() if y in (2023, 2024) and v[1] is not None]
check('T3i time axes: LANDFIRE harvest/unknown (calendar year) and Hansen events of 2023-24 lower the AlphaEarth 2022->2025 cosine; events <= 2021 do not',
      len(post) > 0 and len(pre) > 0 and max(post) < min(pre) and max(post) < c_clean - 0.05,
      f'no disturbance since 2015: {c_clean:.3f}; LANDFIRE {{yr: (n, cos)}} {tabL}; Hansen {tabH}')
hok = [((g.prior / g.incl_w)[g.hi].min() >= (g.prior / g.incl_w)[~g.hi].max() * (1 - 1e-4)) for _, g in Cs.groupby('pid') if g.hi.any() and (~g.hi).any()]
check('T3h hi flag = a per-footprint prior threshold (monotone)', all(hok), f'{len(hok)} plots with both classes')
sys.path.insert(0, f'{ROOT}/scripts'); import g02b_build_gafl as G2; G2.init()
cnt_err = []
for pid, g in Cs.groupby('pid'):
    r0, c0 = land.index(Pi.loc[pid].x, Pi.loc[pid].y); RP = int(1600 // 30) + 2
    L = land.read(1, window=Window(c0 - RP - 3, r0 - RP - 3, 2 * RP + 7, 2 * RP + 7), boundless=True, fill_value=0)       # (window origin r0 - RP - 3)
    if AEF90: ar_ = np.arange(r0 - RP - 3, r0 + RP + 4); ac_ = np.arange(c0 - RP - 3, c0 + RP + 4); ar_, ac_ = ar_[ar_ % 3 == 1], ac_[ac_ % 3 == 1]   # 90 m block centres
    else: ar_, ac_ = np.arange(r0 - RP + 1, r0 + RP, 3), np.arange(c0 - RP + 1, c0 + RP, 3)                                    # the plot window's own stride-3 lattice
    AR, AC = np.meshgrid(ar_, ac_, indexing='ij'); AR, AC = AR.ravel(), AC.ravel(); Lc = L[AR - (r0 - RP - 3), AC - (c0 - RP - 3)]
    fl = (Lc >= 1) & (Lc <= 4) & (np.hypot(AR - r0, AC - c0) * 30 <= 1600)
    cnt_err.append(abs(g.incl_w.sum() - fl.sum()) / fl.sum())
check('T3g inclusion weights sum to the full forest-footprint count of the disk', max(cnt_err) < 1e-4, f'max rel err {max(cnt_err):.2e}')
# ======================= T4 model inputs (g03b data tensors) and unit tests of the model pieces
os.environ.update(DATA=os.environ.get('DATA', 'gafl'), EPOCHS='0', LOAD='0', NT=os.environ.get('NT', '4')); _a = sys.argv; sys.argv = ['g03b']
SRC = open(f'{ROOT}/scripts/g03b_train_gafl.py').read(); SRC = SRC[:SRC.index('TEMP = 0.05')]; ns_ = {'__file__': f'{ROOT}/scripts/g03b_train_gafl.py'}
exec(compile(SRC, 'g03b', 'exec'), ns_); sys.argv = _a; import torch
tv = samp.vid.values[:200]
G4 = open(f'{ROOT}/scripts/g04_bayes_downscale.py').read(); exec(compile(G4[G4.index('# decoded summaries'):G4.index('t0 = time.time()')], 'g04', 'exec'), ns_)
o = ns_['obs_summ'](torch.tensor(tv)); ba_t = pd.Series(o[1].numpy(), index=tv)
ba_i = ba_ind.reindex(tv).fillna(0.0)
check('T4a obs_summ BA (model targets) = FIA-formula BA (independent)', np.allclose(ba_t.values, ba_i.values, rtol=1e-4, atol=1e-6), f'max abs diff {np.max(np.abs(ba_t.values - ba_i.values)):.2e}')
tph_i = live.groupby('vid').w.sum().reindex(tv).fillna(0.0); check('T4b obs_summ TPH = sum of TPA_UNADJ x 2.471 (independent)', np.allclose(o[0].numpy(), tph_i.values, rtol=1e-4, atol=1e-6))
NK, NA = len(KEYS), 16; v_ = torch.tensor(tv[:50]); lp_perfect = torch.where(ns_['slot_p'][v_] > 0, torch.tensor(60.0), torch.tensor(-60.0)) + torch.log(ns_['POSW'])
s_pred = ns_['summaries'](lp_perfect, ns_['slot_v'][v_]); o2 = ns_['obs_summ'](v_)
check('T4c decoder summaries on perfect slots (with calibration) = observed summaries', all(np.allclose(s_pred[i].numpy(), o2[i].numpy(), rtol=1e-4, atol=1e-5) for i in range(3)))
tr_p = set(ns_['vpid'][ns_['tr_v']]); te_p = set(ns_['vpid'][ns_['te_v']])
check('T4d train / test plots disjoint (spatial blocks)', not (tr_p & te_p), f'train plots {len(tr_p)} test plots {len(te_p)}')
flow = type(ns_['flow'])(); Dz = ns_['D_Z']; z0 = torch.zeros(3, Dz); dist = torch.zeros(3, NY, NC + 2); dist[1, 2010 - 2000, 0] = 1; soil = torch.zeros(3, ns_['SOIL'].shape[1])
with torch.no_grad():
    for p_ in flow.J.parameters(): p_.fill_(0.0)
    flow.J[-1].bias.fill_(1.0); flow.A.fill_(0.5)                                    # jump = tanh(1) on every dim when an event fires; non-zero attractor
    st = flow(z0, torch.tensor([2005.0, 2005.0, 2015.0]), [2009, 2010, 2011], dist, soil)
    j10 = (st[2010][1] - st[2010][0]).abs().mean().item(); j09 = (st[2009][1] - st[2009][0]).abs().max().item()
    moved = st[2011][0].abs().min().item(); frozen = st[2011][2].abs().max().item()
check('T4e flow: event jump = tanh(1) exactly at its year, none before; a visit-2015 state is untouched in 2011 while a visit-2005 state has flowed',
      j09 < 1e-7 and abs(j10 - math.tanh(1.0)) < 1e-5 and moved > 0 and frozen == 0, f'diff@2009 {j09:.1e}, jump@2010 {j10:.5f} (tanh 1 = 0.76159), flowed {moved:.3f}, frozen {frozen}')
with torch.no_grad():
    st2 = flow(torch.zeros(2, Dz), torch.tensor([2005.0, 2015.0]), [2011], torch.zeros(2, NY, NC + 2), torch.zeros(2, ns_['SOIL'].shape[1]))
check('T4f flow: a sample whose visit is after the stop year keeps its start state', (st2[2011][1] == 0).all().item())
with torch.no_grad():
    dist3 = torch.zeros(3, NY, NC + 2); dist3[:, 2010 - 2000, 0] = 1
    st3 = flow(torch.zeros(3, Dz), torch.tensor([2010.25, 2010.75, 2009.5]), [2010], dist3, torch.zeros(3, ns_['SOIL'].shape[1]))
    nojump = flow(torch.zeros(1, Dz), torch.tensor([2010.25]), [2010], torch.zeros(1, NY, NC + 2), torch.zeros(1, ns_['SOIL'].shape[1]))[2010]
    j_early = (st3[2010][0] - nojump[0]).abs().mean().item(); j_late = (st3[2010][1] - flow(torch.zeros(1, Dz), torch.tensor([2010.75]), [2010], torch.zeros(1, NY, NC + 2), torch.zeros(1, ns_['SOIL'].shape[1]))[2010][0]).abs().max().item()
    m_a = flow(torch.zeros(1, Dz), torch.tensor([2010.0]), [2010], torch.zeros(1, NY, NC + 2), torch.zeros(1, ns_['SOIL'].shape[1]))[2010].abs().mean().item()
    m_b = nojump.abs().mean().item()
check('T4h flow timing: a visit-year event counts for a first-half visit only; the first year integrates only the part after the visit',
      abs(j_early - math.tanh(1.0)) < 1e-5 and j_late < 1e-7 and 0 < m_b < m_a, f'jump(2010.25) {j_early:.5f}, jump(2010.75) {j_late:.1e}, drift full year {m_a:.3f} > 0.75 yr {m_b:.3f}')
pid_t = max(ns_['cand_of'], key=lambda k: ns_['cand_of'][k][1].max()); cid_t, pr_t = ns_['cand_of'][pid_t]; np.random.seed(0)
draws = np.concatenate([ns_['sample_cands']([pid_t], 48)[0] for _ in range(400)]); fr = pd.Series(draws).value_counts().reindex(cid_t).fillna(0).values / len(draws)
check('T4i MIL candidates are iid draws from the prior (empirical frequency of the most concentrated plot)', np.abs(fr - pr_t).max() < 0.02, f'max prior {pr_t.max():.3f}, empirical {fr[np.argmax(pr_t)]:.3f}, max |Δ| {np.abs(fr - pr_t).max():.4f}')
with torch.no_grad():
    zA = torch.zeros(2, Dz); stE = flow(zA, torch.tensor([2010.2, 2010.2]), [2014], torch.zeros(2, NY, NC + 2), torch.zeros(2, ns_['SOIL'].shape[1]), end=torch.tensor([2014.3, 2014.9]))
    stF = flow(zA[:1], torch.tensor([2010.2]), [2014, 2015], torch.zeros(1, NY, NC + 2), torch.zeros(1, ns_['SOIL'].shape[1]))
    dE = torch.zeros(1, NY, NC + 2); dE[0, 2014 - 2000, 0] = 1
    e1 = flow(zA[:1], torch.tensor([2010.2]), [2014], dE, torch.zeros(1, ns_['SOIL'].shape[1]), end=torch.tensor([2014.3]))[2014]
    e2 = flow(zA[:1], torch.tensor([2010.2]), [2014], dE, torch.zeros(1, ns_['SOIL'].shape[1]), end=torch.tensor([2014.9]))[2014]
    e2n = flow(zA[:1], torch.tensor([2010.2]), [2014], torch.zeros(1, NY, NC + 2), torch.zeros(1, ns_['SOIL'].shape[1]), end=torch.tensor([2014.9]))[2014]
check('T4j flow end time (pair targets): the state is taken AT the end date (between end-of-2013 and end-of-2014 states), end-year events only if before the end',
      (stE[2014][0] - stE[2014][1]).abs().max() > 0 and (e1 - stE[2014][0]).abs().max() < 1e-6 and abs((e2 - e2n).abs().mean().item() - math.tanh(1.0)) < 1e-5,
      f'Δ(end 2014.3 vs 2014.9) {(stE[2014][0] - stE[2014][1]).abs().max():.3f}; event of 2014 with end 2014.3 ignored, with end 2014.9 applied')
zn = torch.randn(4096, Dz); zs = zn * 3
check('T4g SIGReg: ~0 for N(0, I), large for a scaled cloud', ns_['sigreg'](zn).item() < 0.2 * ns_['sigreg'](zs).item(), f'{ns_["sigreg"](zn).item():.4f} vs {ns_["sigreg"](zs).item():.4f}')
# ======================= T5 landscape descriptors: inclusion-weighted subsample vs the full disk
exec(compile(G4[G4.index('from sklearn.cluster import'):G4.index('XL = {}')], 'g04', 'exec'), ns_)
AEF, DIST, SOIL = ns_['AEF'], ns_['DIST'], ns_['SOIL']; smu, ssd = ns_['smu'], ns_['ssd']; land_ = ns_['landscape']; dW, dU = [], []
for pid in pids:
    g = C[C.pid == pid]; cid = g.cid.values
    if g.incl_w.max() <= 1: continue                       # disk not subsampled: trivially exact
    xs = land_(AEF[cid], DIST[cid], SOIL[cid], g.incl_w.values); xu = land_(AEF[cid], DIST[cid], SOIL[cid])
    r0, c0 = land.index(Pi.loc[pid].x, Pi.loc[pid].y); RP = int(1600 // 30) + 2
    if AEF90: wr0, wc0, hw = 3 * ((r0 - RP) // 3), 3 * ((c0 - RP) // 3), 2 * RP + 7
    else: wr0, wc0, hw = r0 - RP, c0 - RP, 2 * RP + 1
    rr, cc, F = G2.feats(wr0, wc0, hw, hw); F = F[np.hypot(rr + wr0 - r0, cc + wc0 - c0) * 30 <= 1600]; F[:, -len(SOILS):] = (F[:, -len(SOILS):] - smu) / ssd
    F = torch.tensor(F); xf = land_(F[:, :128], F[:, 128:128 + NY * (NC + 2)].reshape(-1, NY, NC + 2), F[:, -len(SOILS):])
    dW.append(xs - xf); dU.append(xu - xf)
dW, dU = np.array(dW), np.array(dU)
def frac_sig(dd_):                                         # share of descriptor dims whose mean signed error across plots is significant (|t| > 3.5)
    sd_ = dd_.std(0, ddof=1); ok_ = sd_ > 1e-9; t_ = np.abs(dd_.mean(0)[ok_]) / (sd_[ok_] / np.sqrt(len(dd_))); return float((t_ > 3.5).mean())
fw, fu = frac_sig(dW), frac_sig(dU)
check('T5 inclusion-weighted landscape is an UNBIASED estimate of the full-disk landscape (signed errors across subsampled plots; unweighted shown for power)',
      fw < 0.05, f'{len(dW)} subsampled plots: dims with significant bias — weighted {fw:.3f}, unweighted {fu:.3f}; mean |Δ| weighted {np.abs(dW).mean():.4f}')
# ======================= T6 stand mapping inputs
S = pd.read_parquet(f'{D}/stands.parquet'); SF = np.load(f'{D}/stand_feat.npy', mmap_mode='r'); ssub = S.sample(200, random_state=5); Xs = np.asarray(SF[ssub.index.values], dtype=np.float32)
sraster = rasterio.open(stands_run()['tif'])
sid_chk = np.array([sraster.read(1, window=Window(c, r, 1, 1))[0, 0] for r, c in zip(ssub.row, ssub.col)])
check('T6a stand footprints lie in their stand (stands.tif)', (sid_chk == ssub.stand_id.values).all())
dR6, dC6 = G2.stand_offset(sraster)          # (AEF90: the feature grid is grid_r4)
e25s = np.stack([aef_fp(a25, r + dR6, c + dC6) for r, c in zip(ssub.row, ssub.col)])
check('T6b stand footprint AlphaEarth 2025 = mosaic 3x3 mean' + (' (AEF90: the 90 m pixel)' if AEF90 else ''), np.abs(e25s - Xs[:, 64:128]).max() < 2e-3, f'max abs err {np.abs(e25s - Xs[:, 64:128]).max():.2e}')
check('T6c unit conversions (m2/ha->ft2/ac, /ha->/ac, cm->in)', abs(1 / 0.229568 - 4.356) < 1e-3 and abs(2.47105 - 2.471) < 1e-3)
# ======================= T7 downstream keys (pid is a row number of THIS table)
LS = pd.read_parquet(f'{D}/localisation.parquet'); jl = LS.join(P.set_index('pid')[key], rsuffix='_t')
check('T7a localisation rows carry the same FIA plot identity as plots.parquet', all((jl[k] == jl[k + '_t']).all() for k in key) and len(LS) == C.pid.nunique(), f'{len(LS)} plots')
if 'n_clusters' in LS.columns:     # g02g localisation: n_hi = footprints consistent with the plot's whole FIA history
    CHh = pd.read_parquet(f'{D}/cand_hist.parquet'); kh = C.merge(CHh, on=['cid', 'pid']).query('cons').groupby('pid').incl_w.sum().reindex(LS.index).fillna(0)
    check('T7b localisation n_hi = consistent footprints (Σ incl_w over cand_hist.cons); mass_hi = 1 iff non-empty', np.allclose(kh, LS.n_hi) and ((LS.mass_hi == 1) == (LS.n_hi > 0)).all(),
          f'{len(LS)} plots, non-empty {int((LS.n_hi > 0).sum())}')
else:
    Pq = P.set_index('pid').reindex(LS.index); check('T7b localisation n_hi / mass_hi = the full-disk numbers of g02b (plots)', np.allclose(LS.n_hi, Pq.n_hi) and np.allclose(LS.mass_hi, Pq.mass_hi))
    kh = C[C.hi].groupby('pid').incl_w.sum().reindex(LS.index).fillna(0); full = C.groupby('pid').incl_w.max().reindex(LS.index) == 1
    check('T7c kept high-prior footprints = n_hi where the disk was not subsampled', np.allclose(kh[full], LS.n_hi[full]), f'{int(full.sum())} plots unsampled')
import glob; fns = [os.path.basename(f)[:-4] for f in glob.glob(f'{D}/visit_aef/*.npy') if '.part' not in f]
kn = set(tuple(map(int, f.split('_'))) for f in fns if f.count('_') == 4)
lp = LS[(LS.n_hi > 0) & (LS.n_hi <= 50) & (LS.mass_hi >= 0.5)]; vl = V[V.pid.isin(lp.index)].assign(yi=lambda x: x.year.astype(int))
need = {(int(r.st), int(r.u), int(r.c), int(r.p), int(r.yi)) for r in vl.itertuples() if 2017 <= r.yi <= 2025 and r.yi not in (2022, 2025)}
check('T7d visit_aef: all files named by FIA plot identity (coverage of the localised plots reported; not yet consumed by any model)', len(kn) == len(fns),
      f'{len(fns)} files (of current table plots {sum(1 for k in kn if k[:4] in set(map(tuple, P[key].astype(int).values)))}); needed {len(need)}, missing {len(need - kn)}')
# ======================= T8 FIA-anchored remeasurement pairs (g03b.pair_cand) vs an independent recomputation from FIADB + the spec
src8 = open(f'{ROOT}/scripts/g03b_train_gafl.py').read(); ns_['V'] = pd.read_parquet(f'{D}/visits.parquet'); os.environ['PAIR_ANCHOR'] = 'period'; os.environ['SEVERITY_ANCHOR'] = '0'
ns_['cand_of'] = {pid: (g.cid.values, g.prior.values) for pid, g in C.groupby('pid')}      # T8 tests the period rule on g02b's candidates
exec(compile(src8[src8.index('vpair = '):src8.index("print(f'FIA-anchored pairs")], 'g03b', 'exec'), ns_)      # (vpair applies g03b's forest-share filter fs_ok)
PC = ns_['pair_cand']; EXv = EX.set_index('vid'); rng8 = np.random.default_rng(8); bad8 = 0; n8 = 0; nev8 = 0
for v1 in rng8.choice(list(PC), 300, replace=False):
    v2 = ns_['vpair'][v1]; pid = int(V.pid.values[v1]); k_ = tuple(int(x) for x in Pi.loc[pid][key]); y1, y2 = float(V.year.values[v1]), float(V.year.values[v2])
    g = C[C.pid == pid]; Xg = np.asarray(CF[g.cid.values], dtype=np.float32)[:, 128:128 + NY * (NC + 2)].reshape(len(g), NY, NC + 2)
    e = evr[(evr.st == k_[0]) & (evr.u == k_[1]) & (evr.c == k_[2]) & (evr.p == k_[3])]; recs = set(); hid2 = False
    d2 = EXv.loc[v2].md; vk2 = d2.year * 10000 + d2.month * 100 + d2.day
    for r in e.itertuples():
        for t_, a_, b_ in (('T', 'TRTCD1', 'TRTYR1'), ('T', 'TRTCD2', 'TRTYR2'), ('T', 'TRTCD3', 'TRTYR3'), ('D', 'DSTRBCD1', 'DSTRBYR1'), ('D', 'DSTRBCD2', 'DSTRBYR2'), ('D', 'DSTRBCD3', 'DSTRBYR3')):
            cd, yr = getattr(r, a_), getattr(r, b_)
            if pd.isna(cd) or cd <= 0: continue
            if r.vk == vk2 and (t_ == 'T' or int(cd) not in HARMLESS): hid2 = True
            if (t_, int(cd)) in SPEC and pd.notna(yr) and 2000 <= yr <= 2025 and y1 <= yr + 0.5 <= y2: recs.add((int(yr), t_, int(cd)))
    cl = []
    for yr, t_, cd in sorted(recs):
        cols_, (lo, hi), lr = SPEC[(t_, cd)]; cs_ = {HH if kk == 'H' else ix[kk] for kk in cols_}
        if cl and yr - cl[-1][4] <= 2: a0 = cl[-1]; cl[-1] = (a0[0] | cs_, min(a0[1], yr + lo), max(a0[2], yr + hi), max(a0[3], lr), yr)
        else: cl.append((cs_, yr + lo, yr + hi, lr, yr))
    m = np.ones(len(g), bool)
    for cs_, lo, hi, lr, _ in cl:                      # sequential narrowing; a constraint that would leave no candidate is skipped (spec)
        if lr < 4: continue
        a0, b0 = max(lo - 2000, 0), min(hi - 2000, NY - 1)
        if b0 >= a0:
            nm_ = m & (Xg[:, a0:b0 + 1][:, :, sorted(cs_)].sum(-1).sum(-1) > 0)
            if nm_.any(): m = nm_
    nev8 += any(c[3] >= 4 for c in cl)
    if not hid2:
        a0, b0 = int(math.floor(y1)) + 1 - 2000, int(math.floor(y2)) - 1 - 2000
        if b0 >= max(a0, 0):
            nm_ = m & ~(np.maximum(Xg[:, max(a0, 0):b0 + 1, ix['harvest']], Xg[:, max(a0, 0):b0 + 1, HH]) >= 0.5).any(1)
            if nm_.any(): m = nm_
    n8 += 1; bad8 += set(g.cid.values[m]) != set(PC[v1][0].tolist())
check('T8 FIA-anchored pair histories (g03b.pair_cand) = independent recomputation from FIADB records + spec', bad8 == 0, f'{n8} pairs ({nev8} with detectable events), mismatches {bad8}')
# T8b (a) evaluation safety: with LOAD=1 the severity rule must be OFF (pair sets = the records-only sets of T8); (b) training (LOAD=0, SEVERITY 1,
# default PAIR_ANCHOR=both): every stand-replacing pair (detectable FIA event, BA(v2) < 0.3 BA(v1)) whose consistent set CONTAINS a candidate with
# stand-replacing evidence in the window uses only such candidates (non-vacuous: pairs without any are not counted)
PC_rec = {k: set(v_[0].tolist()) for k, v_ in ns_['pair_cand'].items()}
os.environ['LOAD'] = '1'; os.environ['SEVERITY_ANCHOR'] = '1'; exec(compile(src8[src8.index('vpair = '):src8.index("print(f'FIA-anchored pairs")], 'g03b', 'exec'), ns_)      # (vpair applies g03b's forest-share filter fs_ok)
off_ok = all(set(ns_['pair_cand'][k][0].tolist()) == PC_rec[k] for k in list(PC_rec)[:3000])
os.environ['LOAD'] = '0'; os.environ['PAIR_ANCHOR'] = 'both'; ns_['C'] = C.merge(pd.read_parquet(f'{D}/cand_hist.parquet'), on=['cid', 'pid'], how='left').assign(cons=lambda x: x.cons.fillna(False).astype(bool))
ns_['C']['prior0'] = ns_['C'].prior; ns_['cand_hist'] = {pid: (g.prior0.values, g.cons.values) for pid, g in ns_['C'].groupby('pid')}
exec(compile(src8[src8.index('vpair = '):src8.index("print(f'FIA-anchored pairs")], 'g03b', 'exec'), ns_)      # (vpair applies g03b's forest-share filter fs_ok)
PCs = ns_['pair_cand']; BAtk = TK.assign(ba=TK.N * math.pi / 4 * (TK.QMD / 100) ** 2).groupby('vid').ba.sum(); okr = 0; nrep = 0
for v1, v2 in list(ns_['vpair'].items()):
    if v1 not in PCs or BAtk.get(v1, 0) <= 1.0 or BAtk.get(v2, 0) / BAtk.get(v1, 1) >= 0.3: continue
    pid = int(V.pid.values[v1]); y1, y2 = float(V.year.values[v1]), float(V.year.values[v2])
    cl_ = [c for c in G2.clusters([(k, y) for k, y in ns_['ev_of'].get(pid, []) if y1 <= y + 0.5 <= y2]) if c['lr'] >= 4]
    if not cl_: continue
    allc = C[C.pid == pid].cid.values; Xp = np.asarray(CF[allc], dtype=np.float32)[:, 128:128 + NY * (NC + 2)].reshape(len(allc), NY, NC + 2); w_ = np.zeros(len(allc), bool)
    for c in cl_:
        lo, hi = max(c['lo'] - 2000, 0), min(c['hi'] - 2000, NY - 1) + 1
        if hi > lo: w_ |= (np.maximum(Xp[:, lo:hi, ix['harvest']], Xp[:, lo:hi, HH]) >= 0.5).any(1)
    m_ev = np.ones(len(allc), bool)                     # event-consistent set (sequential narrowing with skip, independent code)
    for c in cl_:
        lo, hi = max(c['lo'] - 2000, 0), min(c['hi'] - 2000, NY - 1) + 1
        if hi > lo:
            nm_ = m_ev & (Xp[:, lo:hi][:, :, sorted(c['cols'])].sum(-1) > 0).any(1)
            if nm_.any(): m_ev = nm_
    sel = np.isin(allc, PCs[v1][0]); strong = set(allc[w_ & m_ev])
    if not strong: continue
    nrep += 1; okr += set(allc[sel]) <= strong       # selected ⊆ event-consistent candidates with stand-replacing evidence
    if nrep >= 300: break
check('T8b severity anchoring: OFF under LOAD=1 (evaluation sets = records-only sets); ON in training (PAIR=both): stand-replacing pairs use only candidates with stand-replacing evidence',
      off_ok and okr == nrep and nrep > 0, f'evaluation sets unchanged: {off_ok}; {nrep} stand-replacing pairs with strong candidates checked, violations {nrep - okr}')
# ======================= T9 plot-level location posterior (g02f) recomputed from the specification with the fitted parameters
if os.path.exists(f'{D}/cand_prior.parquet'):
    def bcls(m, iw, inn):      # background per distance class (inner 800 m disk / ring), clipped as specified
        out = np.zeros(len(m))
        for cl_ in (inn, ~inn):
            if cl_.any(): out[cl_] = np.clip((iw[cl_] * m[cl_]).sum() / iw[cl_].sum(), 0.5 / iw[cl_].sum(), 1 - 1e-3)
        return out
    th = json.load(open(f'{D}/location_em.json'))['theta']; CP = pd.read_parquet(f'{D}/cand_prior.parquet').set_index('cid'); GRPV = {8: 'strong', 4: 'medium', 3: 'weak'}; err9 = []
    for pid, g in Cs.groupby('pid'):
        Xg = np.asarray(CF[g.cid.values], dtype=np.float32)[:, 128:128 + NY * (NC + 2)].reshape(len(g), NY, NC + 2); iw = g.incl_w.values.astype(float); inn = g.dist_m.values <= 800
        lw = np.log(np.where(inn, th['pi'] / iw[inn].sum(), (1 - th['pi']) / iw[~inn].sum()) * iw) if inn.any() and (~inn).any() else np.log(iw / iw.sum())
        cl, per_ = REC[pid]
        for c_ in cl:
            cs_, lo, hi, lr = c_[0], c_[1], c_[2], c_[3]; a0, b0 = max(lo - 2000, 0), min(hi - 2000, NY - 1)
            if b0 < a0: continue
            m = Xg[:, a0:b0 + 1][:, :, sorted(cs_)].sum(-1).sum(-1) > 0; qq = th['q'][GRPV[int(lr)]]
            if 0 < (iw * m).sum() / iw.sum() < 1: bf = bcls(m, iw, inn); lw = lw + np.where(m, np.log(qq / bf), np.log((1 - qq) / (1 - bf)))
        srv = np.maximum(Xg[:, :, ix['harvest']], Xg[:, :, HH]) >= 0.5
        for a0, b0 in per_:
            a0, b0 = max(a0 - 2000, 0), min(b0 - 2000, NY - 1)
            if b0 < a0: continue
            m = srv[:, a0:b0 + 1].any(1)
            if 0 < (iw * m).sum() / iw.sum() < 1: bf = bcls(m, iw, inn); lw = lw + np.where(m, np.log(th['f'] / bf), np.log((1 - th['f']) / (1 - bf)))
        lin = np.logaddexp.reduce(lw); post = np.exp(lw - lin); rin = 1 - np.exp(np.log(th['s']) - np.logaddexp(np.log(1 - th['s']) + lin, np.log(th['s'])))
        p = rin * post + (1 - rin) * iw / iw.sum(); err9.append(np.abs(p - CP.loc[g.cid.values].prior_loc.values).max() / p.max())
    check('T9 plot-level location posterior (g02f) = recomputation from the spec + fitted EM parameters', max(err9) < 1e-3, f'{len(err9)} plots, max rel err {max(err9):.1e}; theta q {th["q"]} f {th["f"]:.3f} pi {th["pi"]:.3f} s {th["s"]:.3f}')
# ======================= T10 FIA-history-consistent candidates (g02g) recomputed from the spec (T3f's independent records)
if os.path.exists(f'{D}/cand_hist.parquet'):
    CH = pd.read_parquet(f'{D}/cand_hist.parquet').set_index('cid'); bad10 = 0; ne10 = 0
    for pid, g in Cs.groupby('pid'):
        Xg = np.asarray(CF[g.cid.values], dtype=np.float32)[:, 128:128 + NY * (NC + 2)].reshape(len(g), NY, NC + 2); cl, per_ = REC[pid]; m = np.ones(len(g), bool)
        srv = np.maximum(Xg[:, :, ix['harvest']], Xg[:, :, HH]) >= 0.5; cons10 = []
        for c_ in cl:
            if c_[3] < 4: continue
            a0, b0 = max(c_[1] - 2000, 0), min(c_[2] - 2000, NY - 1)
            if b0 >= a0: cons10.append((c_[1], Xg[:, a0:b0 + 1][:, :, sorted(c_[0])].sum(-1).sum(-1) > 0))
        for a0, b0 in per_:
            a1, b1 = max(a0 - 2000, 0), min(b0 - 2000, NY - 1)
            if b1 >= a1: cons10.append((a0, ~srv[:, a1:b1 + 1].any(1)))
        for _, x in sorted(cons10, key=lambda t: t[0]):          # time order; a constraint no remaining footprint satisfies is skipped (spec)
            if (m & x).any(): m = m & x
        bad10 += int((CH.loc[g.cid.values].cons.values != m).sum() > 0); ne10 += int(not m.any())
    check('T10 FIA-history-consistent candidates (g02g) = independent recomputation (detectable clusters + undisturbed periods, time order, unsatisfiable skipped)',
          bad10 == 0, f'{Cs.pid.nunique()} plots, mismatching plots {bad10}, empty sets {ne10}')
# ======================= T11 Forest Explorer export (g07): GEN_* = stand map x unit conversions
FEJ = os.environ.get('FE_GEOJSON', os.path.join(os.path.dirname(ROOT), 'fe_pan/apps/forest-explorer/public/stands/baker_fl.geojson'))   # genveg_pipeline: config fe.geojson
if os.path.exists(FEJ) and os.path.exists(f'{D}/stand_veg_v3.parquet'):
    SV = pd.read_parquet(f'{D}/stand_veg_v3.parquet').set_index('stand_id'); fe = json.load(open(FEJ))['features']; e11 = []; n11 = 0
    for f in fe:
        p = f['properties']
        if 'GEN_BA' not in p: continue
        r = SV.loc[int(p['stand_id'])]; n11 += 1
        e11.append(max(abs(p['GEN_BA'] - r.ba / 0.229568), abs(p['GEN_TPA'] - r.tph / 2.47105), abs(p['GEN_QMD'] - r.qmd / 2.54) * 10, abs(p['GEN_BA_SD'] - r.ba_sd / 0.229568)))
        if p['GEN_DOM'] != r.dom_key: e11.append(99)
    check('T11 Forest Explorer stand layer: GEN_BA / TPA / QMD / BA_SD / DOM = stand map (g05) x conversions (m2/ha->ft2/ac, /ha->/ac, cm->in)', n11 > 0 and max(e11) < 0.11,
          f'{n11} stands, max abs diff {max(e11) if e11 else float("nan"):.3f} (rounding 0.05 / 0.005)')
# ======================= T11b stand cohort tables (g05, the TreeMap-independent initial state of the simulation engines) carry the CALIBRATED stand summary
if os.path.exists(f'{D}/stand_cohorts_v3.parquet') and os.path.exists(f'{D}/stand_veg_v3.parquet'):
    Cc = pd.read_parquet(f'{D}/stand_cohorts_v3.parquet'); Sv = pd.read_parquet(f'{D}/stand_veg_v3.parquet').set_index('stand_id')
    ct_ = Cc.groupby('stand_id').tph.sum(); cb_ = (Cc.tph * math.pi / 4 * (Cc.qmd_cm / 100) ** 2).groupby(Cc.stand_id).sum(); jj = Sv[['tph', 'ba']].join(pd.DataFrame({'ct': ct_, 'cb': cb_}), how='inner')
    jj = jj[(jj.tph > 0) & (jj.ba > 0)]; rt = (jj.ct / jj.tph - 1).abs().max(); rb = (jj.cb / jj.ba - 1).abs().max()
    check('T11b stand cohort tables sum to the calibrated stand TPH and BA (g05)', rt < 1e-6 and rb < 1e-6, f'{len(jj)} stands; max |ratio - 1| TPH {rt:.1e} BA {rb:.1e}')
# ======================= T12 summary-reconstruction targets (g03b OBS_*) and the loss at a perfect prediction
vv12 = torch.tensor(samp.vid.values[:300]); o12 = ns_['obs_summ'](vv12)
okt = all(np.allclose(ns_[k][vv12].numpy(), o12[i].numpy(), rtol=1e-5, atol=1e-6) for i, k in enumerate(['OBS_T', 'OBS_B', 'OBS_Q']))
lp_perf = torch.where(ns_['slot_p'][vv12] > 0, torch.tensor(60.0), torch.tensor(-60.0)) + torch.log(ns_['POSW'])
se12 = ns_['summ_err'](lp_perf, ns_['slot_v'][vv12], vv12)
check('T12 summary-loss targets = obs_summ (TPH, BA, QMD; BA independently = FIA formula by T4a) and the loss vanishes at a perfect prediction (incl. the Bray term)',
      okt and float(se12.max()) < 1e-6, f'max loss at perfect prediction {float(se12.max()):.2e}')
# ======================= T13 pixel-exact georeferencing of the targeted AlphaEarth windows (g02c): their 3x3 means must match the 2022 mosaic best at offset 0
import glob as _gl
LSg = pd.read_parquet(f'{D}/localisation.parquet'); RPw = int(1600 // 30) + 2; acc13 = {}; nv13 = 0
for fnw in sorted(_gl.glob(f'{D}/visit_aef/*_2021.npy') + _gl.glob(f'{D}/visit_aef/*_2023.npy'))[:60]:
    st_, u_, c_, p_, yr_ = map(int, os.path.basename(fnw)[:-4].split('_')); pp = P[(P.st == st_) & (P.u == u_) & (P.c == c_) & (P.p == p_)]
    if not len(pp): continue
    pid = int(pp.pid.iloc[0]); g = C[C.pid == pid].sample(min(40, (C.pid == pid).sum()), random_state=0); Wn = dq(np.load(fnw)); r0, c0 = land.index(pp.x.iloc[0], pp.y.iloc[0])
    M22 = np.stack([np.nan_to_num(np.nanmean(dq(a22.read(window=Window(c - 1, r - 1, 3, 3))).reshape(64, -1), 1)) for r, c in zip(g.row, g.col)])
    M22 /= np.linalg.norm(M22, axis=1, keepdims=True) + 1e-9; nv13 += 1
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            rr = g.row.values - (r0 - RPw) + dr; cc = g.col.values - (c0 - RPw) + dc; ok_ = (rr >= 1) & (rr < Wn.shape[1] - 1) & (cc >= 1) & (cc < Wn.shape[2] - 1)
            Wv = np.stack([np.nan_to_num(np.nanmean(Wn[:, r - 1:r + 2, c - 1:c + 2].reshape(64, -1), 1)) for r, c in zip(rr[ok_], cc[ok_])]); Wv /= np.linalg.norm(Wv, axis=1, keepdims=True) + 1e-9
            acc13.setdefault((dr, dc), []).append(float(np.median((Wv * M22[ok_]).sum(1))))
med13 = {k: np.median(v) for k, v in acc13.items()}; best13 = max(med13, key=med13.get) if med13 else None
if best13 is None: print('SKIP  T13 no targeted AlphaEarth windows (g02c) in this table')
else: check('T13 targeted AlphaEarth windows are pixel-exact: agreement with the 2022 mosaic peaks at offset (0, 0) among ±2 px shifts', best13 == (0, 0),
      f'{nv13} windows; median cosine at (0,0) {med13[(0, 0)]:.4f}, best {best13} {med13[best13]:.4f}, at (+1,0) {med13[(1, 0)]:.4f}, (0,+1) {med13[(0, 1)]:.4f}')
# ======================= T14 no leakage: the joint AlphaEarth location posterior (g08) exists only for TRAINING plots (spatial blocks + buffer of g03b)
if os.path.exists(f'{D}/cand_loc_aef.parquet'):
    LA = pd.read_parquet(f'{D}/cand_loc_aef.parquet'); trp = set(ns_['vpid'][ns_['tr_v']]); bad14 = set(LA.pid) - trp
    check('T14 same-year AlphaEarth location posterior only for training plots (no held-out / buffer plot)', not bad14, f'{LA.pid.nunique()} plots, outside training {len(bad14)}')
nf = sum(1 for _, ok in RES if not ok); print(f'\n{len(RES) - nf} / {len(RES)} checks PASS'); sys.exit(1 if nf else 0)
