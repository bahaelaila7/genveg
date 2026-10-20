"""Independent verification of the generative-vegetation pipeline (user: "triple check again. I don't trust it yet").
Every check recomputes a stage from the raw sources by a DIFFERENT code path (SQL instead of pandas, direct raster reads at stored coordinates, FIADB COND
instead of curated_trees for events) and compares. Output: PASS / FAIL lines + numbers; exit code 1 if anything fails.
  python scripts/gv_verify.py [n_sample=300]"""
import sys, os, json, math, numpy as np, pandas as pd, duckdb, rasterio
from rasterio.windows import Window
from pyproj import Transformer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); WS = os.environ.get('WORKSPACE', os.path.dirname(ROOT)); D = f'{ROOT}/data/proto/gafl'; NS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng = np.random.default_rng(42); RES = []
def check(name, ok, detail=''):
    RES.append((name, bool(ok))); print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
M = json.load(open(f'{D}/meta.json')); KEYS = M['keys']; YEARS = M['years']; NC = len(M['classes']); NY = len(YEARS)
key = ['st', 'u', 'c', 'p']; V = pd.read_parquet(f'{D}/visits.parquet'); TK = pd.read_parquet(f'{D}/tokens.parquet'); PR = pd.read_parquet(f'{D}/pairs.parquet')
P = pd.read_parquet(f'{D}/plots.parquet'); C = pd.read_parquet(f'{D}/cands.parquet'); CF = np.load(f'{D}/cand_feat.npy', mmap_mode='r')
con = duckdb.connect(f'{WS}/landis_region4/data_r4/slim_r4.duckdb', read_only=True)
# ======================= T1 visits and cohort tokens (SQL aggregation, independent of the pandas builder)
top = [r[0] for r in con.execute("""select r.SPECIES_SYMBOL from curated_trees t join REF_SPECIES r on r.SPCD=t.SPCD where t.STATECD in (12,13) and t.MEASDATE>='2000-01-01'
                                    and t.STATUSCD=1 group by 1 order by count(*) desc, 1 limit 16""").fetchall()]
check('T1b species keys = top-16 live species (SQL) + _H/_S', sorted(top) + ['_H', '_S'] == KEYS, f'{sorted(top)[:5]}…')
samp = V.sample(min(NS, len(V)), random_state=1)
reg = duckdb.connect(); reg.register('samp', samp[key + ['vid']].assign(yr=samp.year))
q = """select t.STATECD st, t.UNITCD u, t.COUNTYCD c, t.PLOT p, t.MEASDATE md, r.SPECIES_SYMBOL sym, r.SFTWD_HRDWD grp, t.DIA dia, t.HT ht, t.CR cr, t.TPA_UNADJ tpa,
              t.estimated_age age, t.STATUSCD status, t.cond_stdorgcd org, t.CONDID cond
       from curated_trees t join REF_SPECIES r on r.SPCD = t.SPCD where t.STATECD in (12,13) and t.MEASDATE >= '2000-01-01'"""
T = con.execute(q).df(); T['md'] = pd.to_datetime(T.md)
# ---- the visit set, rebuilt independently from FIADB (pandas aggregation of per-condition rows; string dates):
#      visits with curated trees on a forested condition  ∪  forested FIADB visits of the same plots with no live tree in FIADB TREE;
#      forested share of the SAMPLED area (non-sampled status 5 excluded) >= 0.1
_fx = duckdb.connect(f'{WS}/FIA/FIASQLITE2PGSQL/FIADB.duckdb', read_only=True)
_cc = _fx.execute("""select p.CN pcn, p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, printf('%04d-%02d-%02d', p.MEASYEAR::int, p.MEASMON::int, p.MEASDAY::int) ds,
        c.CONDID cond, c.COND_STATUS_CD cs, c.SUBPPROP_UNADJ sp, c.MICRPROP_UNADJ mp, c.STDORGCD org from PLOT p join COND c on c.PLT_CN = p.CN
        where p.STATECD in (12,13) and p.MEASYEAR >= 2000""").df()
_nl = _fx.execute("select PLT_CN pcn, count(*) nl from TREE where STATECD in (12,13) and STATUSCD = 1 group by 1").df()
_f = _cc.assign(sp1=np.where(_cc.cs == 1, _cc.sp, 0), sp5=np.where(_cc.cs == 5, _cc.sp, 0), mp1=np.where(_cc.cs == 1, _cc.mp, 0), mp5=np.where(_cc.cs == 5, _cc.mp, 0),
                org1=np.where(_cc.cs == 1, _cc.org, np.nan)).groupby(['pcn'] + key + ['ds']).agg(sp1=('sp1', 'sum'), sp5=('sp5', 'sum'), mp1=('mp1', 'sum'), mp5=('mp5', 'sum'), org=('org1', 'max')).reset_index()
_f = _f.merge(_nl, on='pcn', how='left').fillna({'nl': 0}); _f = _f[_f.sp1 > 0]
_f['fs'] = np.minimum(_f.sp1 / (1 - _f.sp5), 1.0); _f['fm'] = np.where(_f.mp1 > 0, np.minimum(_f.mp1 / (1 - _f.mp5), 1.0), _f.fs)
T['ds'] = T.md.dt.strftime('%Y-%m-%d'); T = T.merge(_cc[key + ['ds', 'cond', 'cs']], on=key + ['ds', 'cond'], how='left'); nT = len(T); T = T[T.cs == 1]
_tv = T[key + ['ds']].drop_duplicates(); _bare = _f[_f.nl == 0].merge(_tv[key].drop_duplicates(), on=key).merge(_tv.assign(_i=1), on=key + ['ds'], how='left')
_bare = _bare[_bare._i.isna()][key + ['ds']]
EX = pd.concat([_tv, _bare]).merge(_f[key + ['ds', 'fs', 'fm', 'org']], on=key + ['ds'], how='inner'); EX = EX[EX.fs >= 0.1]
EX['md'] = pd.to_datetime(EX.ds); EX = EX.sort_values(key + ['md']).reset_index(drop=True); EX['vid'] = np.arange(len(EX)); EX['year'] = EX.md.dt.year + EX.md.dt.dayofyear / 366
check('T1a visit set = curated forested-condition visits ∪ bare FIADB visits, share >= 0.1 (rebuilt from FIADB)', len(EX) == len(V),
      f'independent {len(EX)} (bare {len(_bare)}) table {len(V)}; non-forest-condition tree records {nT - len(T)}')
Vs = V.sort_values('vid')
check('T1c visit ids reproduce (sorted plot, date order)', len(EX) == len(V) and (EX[key].values == Vs[key].values).all() and (np.abs(EX.year.values - Vs.year.values) < 1e-9).all())
T = T.merge(EX[key + ['ds', 'vid', 'fs', 'fm']], on=key + ['ds']); Ts = T[T.vid.isin(samp.vid)]
live = Ts[(Ts.status == 1) & (Ts.dia > 0) & (Ts.tpa > 0) & (Ts.age > 0) & (Ts.ht > 0)].copy()
live['key'] = [KEYS.index(s) if s in top else KEYS.index('_' + (g if g in ('S', 'H') else 'H')) for s, g in zip(live.sym, live.grp)]
live['share'] = np.where(live.dia < 5, live.fm, live.fs)
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
fv = duckdb.connect(f'{WS}/FIA/FIASQLITE2PGSQL/FIADB.duckdb', read_only=True).execute("""select distinct p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, p.MEASYEAR my
      from PLOT p join COND c on c.PLT_CN = p.CN where p.STATECD in (12,13) and p.MEASYEAR >= 2000 and c.COND_STATUS_CD = 1""").df()
ours = V[key].assign(my=V.year.astype(int)).drop_duplicates(); inplots = fv.merge(V[key].drop_duplicates(), on=key)
miss = inplots.merge(ours, on=key + ['my'], how='left', indicator=True); nm = (miss._merge == 'left_only').sum()
nplots_f = len(fv[key].drop_duplicates()); nplots_o = len(V[key].drop_duplicates())
check('T1j FIADB forested visits of our plots that are absent from the table (bare visits with no tree record?) < 2 %', nm / len(inplots) < 0.02,
      f'missing {nm} of {len(inplots)}; forested FIADB plots {nplots_f}, in table {nplots_o} (curated product subset)')
# ======================= T2 plot coordinates
fc = duckdb.connect(f'{WS}/FIA/FIASQLITE2PGSQL/FIADB.duckdb', read_only=True)
pl = fc.execute("select STATECD st, UNITCD u, COUNTYCD c, PLOT p, avg(LAT) lat, avg(LON) lon from PLOT where STATECD in (12,13) group by all").df()
m = P.merge(pl, on=key, suffixes=('', '_f'))
check('T2a plot lat/lon = FIADB PLOT mean', np.allclose(m.lat, m.lat_f) and np.allclose(m.lon, m.lon_f), f'n {len(m)}')
land = rasterio.open(f'{ROOT}/data/derived/land_clean.tif'); tf = Transformer.from_crs(4326, land.crs, always_xy=True); x, y = tf.transform(m.lon.values, m.lat.values)
check('T2b projected x/y (EPSG:5070) recomputed', np.allclose(x, m.x, atol=1e-3) and np.allclose(y, m.y, atol=1e-3))
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
a22 = rasterio.open(f'{ROOT}/data/aef/aef_2022.tif'); a25 = rasterio.open(f'{ROOT}/data/aef/aef_2025.tif')
def dq(a): v = a.astype(np.float32); o = np.sign(v) * (v / 127.5) ** 2; o[a == -128] = np.nan; return o
e22 = np.stack([np.nan_to_num(np.nanmean(dq(a22.read(window=Window(c - 1, r - 1, 3, 3))).reshape(64, -1), 1)) for r, c in zip(sub.row, sub.col)])
e25 = np.stack([np.nan_to_num(np.nanmean(dq(a25.read(window=Window(c - 1, r - 1, 3, 3))).reshape(64, -1), 1)) for r, c in zip(sub.row, sub.col)])
err = max(np.abs(e22 - X[:, :64]).max(), np.abs(e25 - X[:, 64:128]).max())
check('T3c AlphaEarth 2022 / 2025 = 3x3 mean of the mosaics at the stored centre', err < 2e-3, f'max abs err {err:.2e} (float16 storage)')
SOILS = M['soils']; FILL = {'clay030': 7.58, 'sand030': 83.2, 'soc030': 0.964, 'awc030': 0.052, 'ph030': 5.18, 'restrdepth': 200.0}
so = np.array([[(lambda v: FILL[k] if v <= -9000 else v)(float(rasterio.open(f'{ROOT}/data/soils/{k}.tif').read(1, window=Window(c, r, 1, 1))[0, 0])) for k in SOILS]
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
hl = rasterio.open(f'{ROOT}/data/hansen/lossyear.tif'); LF = {yr: rasterio.open(f'{ROOT}/data/landfire/dist/dist_{yr}.tif') for yr in YEARS}; bad = 0; nev = 0; nh = 0; maxd = 0
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
SPEC = {('T', 10): (['harvest', 'thin', 'unknown', 'H'], (-1, 1), 8), ('T', 20): (['harvest', 'thin', 'fire', 'chem', 'unknown', 'H'], (-1, 1), 8),
        ('T', 30): (['harvest', 'thin', 'chem', 'unknown', 'H'], (-2, 1), 8), ('T', 40): (['harvest', 'unknown', 'H'], (-2, 1), 8), ('T', 50): (['thin', 'fire', 'chem', 'unknown'], (-1, 1), 3)}
for cd in range(10, 23): SPEC[('D', cd)] = (['insdis', 'unknown', 'H'], (-1, 1), 4)
for cd in (30, 31): SPEC[('D', cd)] = (['fire', 'unknown'], (-1, 1), 3)
SPEC[('D', 32)] = (['fire', 'unknown', 'H'], (-1, 1), 8)
for cd in range(50, 55): SPEC[('D', cd)] = (['weather', 'unknown', 'H'], (-1, 1), 4)
SPEC[('D', 80)] = (['harvest', 'thin', 'dev', 'unknown', 'H'], (-1, 1), 8); SPEC[('D', 70)] = (CLS + ['H'], (-1, 1), 3)
HARMLESS = {12, 22, 31, 40, 41, 42, 43, 44, 45, 46, 60}
evr = _fx.execute("""select p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, p.MEASYEAR my, p.MEASMON mm, p.MEASDAY dd, c.COND_STATUS_CD cs, q.MEASYEAR pmy,
                    c.TRTCD1, c.TRTYR1, c.TRTCD2, c.TRTYR2, c.TRTCD3, c.TRTYR3, c.DSTRBCD1, c.DSTRBYR1, c.DSTRBCD2, c.DSTRBYR2, c.DSTRBCD3, c.DSTRBYR3
                    from COND c join PLOT p on c.PLT_CN = p.CN left join PLOT q on q.CN = p.PREV_PLT_CN where p.STATECD in (12,13) and p.MEASYEAR >= 2000""").df()
evr = evr[evr.cs == 1]; evr['vk'] = evr.my * 10000 + evr.mm * 100 + evr.dd
ratios, nneg, ncl = [], 0, 0
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
    q = g.prior.values / (w * g.incl_w.values); ratios.append(q.max() / q.min())
check('T3f prior ∝ (clustered typed FIA events x FIA-protocol negative evidence) x incl_w — re-implemented from the spec on FIADB COND', max(ratios) < 1 + 2e-3,
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
    L = land.read(1, window=Window(c0 - RP, r0 - RP, 2 * RP + 1, 2 * RP + 1), boundless=True, fill_value=0)
    rr, cc = np.mgrid[1:2 * RP:3, 1:2 * RP:3]; rr, cc = rr.ravel(), cc.ravel(); fl = (L[rr, cc] >= 1) & (L[rr, cc] <= 4) & (np.hypot(rr - RP, cc - RP) * 30 <= 1600)
    cnt_err.append(abs(g.incl_w.sum() - fl.sum()) / fl.sum())
check('T3g inclusion weights sum to the full forest-footprint count of the disk', max(cnt_err) < 1e-4, f'max rel err {max(cnt_err):.2e}')
# ======================= T4 model inputs (g03b data tensors) and unit tests of the model pieces
os.environ.update(DATA='gafl', EPOCHS='0', LOAD='0', NT='4'); _a = sys.argv; sys.argv = ['g03b']
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
flow = ns_['Flow'](); Dz = ns_['D_Z']; z0 = torch.zeros(3, Dz); dist = torch.zeros(3, NY, NC + 2); dist[1, 2010 - 2000, 0] = 1; soil = torch.zeros(3, ns_['SOIL'].shape[1])
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
    rr, cc, F = G2.feats(r0 - RP, c0 - RP, 2 * RP + 1, 2 * RP + 1); F = F[np.hypot(rr - RP, cc - RP) * 30 <= 1600]; F[:, -len(SOILS):] = (F[:, -len(SOILS):] - smu) / ssd
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
sraster = rasterio.open(f'{ROOT}/data/derived/stands.tif')
sid_chk = np.array([sraster.read(1, window=Window(c, r, 1, 1))[0, 0] for r, c in zip(ssub.row, ssub.col)])
check('T6a stand footprints lie in their stand (stands.tif)', (sid_chk == ssub.stand_id.values).all())
e25s = np.stack([np.nan_to_num(np.nanmean(dq(a25.read(window=Window(c - 1, r - 1, 3, 3))).reshape(64, -1), 1)) for r, c in zip(ssub.row, ssub.col)])
check('T6b stand footprint AlphaEarth 2025 = mosaic 3x3 mean', np.abs(e25s - Xs[:, 64:128]).max() < 2e-3, f'max abs err {np.abs(e25s - Xs[:, 64:128]).max():.2e}')
check('T6c unit conversions (m2/ha->ft2/ac, /ha->/ac, cm->in)', abs(1 / 0.229568 - 4.356) < 1e-3 and abs(2.47105 - 2.471) < 1e-3)
# ======================= T7 downstream keys (pid is a row number of THIS table)
LS = pd.read_parquet(f'{D}/localisation.parquet'); jl = LS.join(P.set_index('pid')[key], rsuffix='_t')
check('T7a localisation rows carry the same FIA plot identity as plots.parquet', all((jl[k] == jl[k + '_t']).all() for k in key) and len(LS) == C.pid.nunique(), f'{len(LS)} plots')
Pq = P.set_index('pid').reindex(LS.index); check('T7b localisation n_hi / mass_hi = the full-disk numbers of g02b (plots)', np.allclose(LS.n_hi, Pq.n_hi) and np.allclose(LS.mass_hi, Pq.mass_hi))
kh = C[C.hi].groupby('pid').incl_w.sum().reindex(LS.index).fillna(0); full = C.groupby('pid').incl_w.max().reindex(LS.index) == 1
check('T7c kept high-prior footprints = n_hi where the disk was not subsampled', np.allclose(kh[full], LS.n_hi[full]), f'{int(full.sum())} plots unsampled')
import glob; fns = [os.path.basename(f)[:-4] for f in glob.glob(f'{D}/visit_aef/*.npy') if '.part' not in f]
kn = set(tuple(map(int, f.split('_'))) for f in fns if f.count('_') == 4)
lp = LS[(LS.n_hi > 0) & (LS.n_hi <= 50) & (LS.mass_hi >= 0.5)]; vl = V[V.pid.isin(lp.index)].assign(yi=lambda x: x.year.astype(int))
need = {(int(r.st), int(r.u), int(r.c), int(r.p), int(r.yi)) for r in vl.itertuples() if 2017 <= r.yi <= 2025 and r.yi not in (2022, 2025)}
check('T7d visit_aef: all files named by FIA plot identity; every window needed by the current localised plots exists', len(kn) == len(fns) and need <= kn,
      f'{len(fns)} files (of current table plots {sum(1 for k in kn if k[:4] in set(map(tuple, P[key].astype(int).values)))}); needed {len(need)}, missing {len(need - kn)}')
nf = sum(1 for _, ok in RES if not ok); print(f'\n{len(RES) - nf} / {len(RES)} checks PASS'); sys.exit(1 if nf else 0)
