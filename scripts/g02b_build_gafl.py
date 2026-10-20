"""Generative-vegetation prototype v2 (all GA+FL plots), training table. Same content as g02_build_proto.py but:
  visits   = every (plot, MEASDATE) of curated_trees (slim_r4, read-only) for GA+FL since 2000, single-visit plots included; live trees with
             estimated_age, DIA, HT, CR, TPA_UNADJ (the ages / filters of the Pan interval product); species key = top-K species, else _S/_H
  pairs    = consecutive visits of a plot
  cands    = up to NCAND footprints per plot (all with an elevated prior + a random fill), features as in g02 (AlphaEarth 2022/25, disturbance
             history 2000-2025 by class + severity + Hansen, soils); prior: dated FIA TRT/DSTRB events matched by a candidate harvest / mechanical /
             Hansen event in year ±1 → ×8 (and a LANDFIRE harvest-class match ×2 more); stand-replacing candidate events inside an undisturbed
             aged history → ×0.1
  conc     = per plot, prior mass on its top 10 % of candidates (localisation confidence; curriculum)
Mapping set: Baker stands as in g02. Output data/proto/gafl/…"""
import sys, os, json, numpy as np, pandas as pd, duckdb, rasterio, geopandas as gpd
from rasterio.windows import Window
from pyproj import Transformer
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from features import ROOT, SOILS, dequant
R = 1600.0; NCAND = int(os.environ.get('NCAND', 256)); OUT = f'{ROOT}/data/proto/gafl'; os.makedirs(OUT, exist_ok=True)
YEARS = list(range(2000, 2026))
CLS = {'harvest': ['Clearcut', 'Harvest'], 'thin': ['Thinning', 'Mastication', 'Other Mechanical', 'Mechanical Remove', 'Mechanical Add'],
       'fire': ['Wildfire', 'Fire', 'Prescribed Fire', 'Wildland Fire Use', 'Wildland Fire', 'Unknown/Possibly Fire'],
       'insdis': ['Insects/Disease', 'Insects', 'Disease', 'Biological'], 'weather': ['Weather'], 'chem': ['Herbicide', 'Chemical', 'Insecticide'], 'dev': ['Development']}
NC = len(CLS); NF = 64 + 64 + len(YEARS) * (NC + 2) + len(SOILS)
_S = {}
def init():
    codes = pd.read_csv(f'{ROOT}/data/landfire/dist/dist_codes.csv')
    cid = {t_: i + 1 for i, (c_, ts) in enumerate(CLS.items()) for t_ in ts}
    SEV = {'Low': 1, 'Low-Medium': 1.5, 'Medium-Low': 1.5, 'Medium': 2, 'Medium-High': 2.5, 'High-Medium': 2.5, 'High': 3}
    lut = {}
    for r in codes.itertuples(): lut[(int(r.year), int(r.VALUE))] = (cid.get(str(r.DIST_TYPE).strip(), 0), SEV.get(str(r.SEVERITY).strip(), 1.0))
    _S.update(lut=lut, land=rasterio.open(f'{ROOT}/data/derived/land_clean.tif'), d={y: rasterio.open(f'{ROOT}/data/landfire/dist/dist_{y}.tif') for y in YEARS},
              h=rasterio.open(f'{ROOT}/data/hansen/lossyear.tif'), a22=rasterio.open(f'{ROOT}/data/aef/aef_2022.tif'), a25=rasterio.open(f'{ROOT}/data/aef/aef_2025.tif'),
              soil={k: rasterio.open(f'{ROOT}/data/soils/{k}.tif') for k in SOILS})
def feats(r0, c0, h, w):
    s = _S; win = Window(c0, r0, w, h)
    L = s['land'].read(1, window=win, boundless=True, fill_value=0)
    rr, cc = np.mgrid[1:h - 1:3, 1:w - 1:3]; rr, cc = rr.ravel(), cc.ravel(); ok = (L[rr, cc] >= 1) & (L[rr, cc] <= 4); rr, cc = rr[ok], cc[ok]
    if len(rr) == 0: return rr, cc, np.zeros((0, NF), np.float32)
    A2 = dequant(s['a22'].read(window=win, boundless=True, fill_value=-128)); A5 = dequant(s['a25'].read(window=win, boundless=True, fill_value=-128))
    HL = s['h'].read(1, window=win, boundless=True, fill_value=0); DS = np.stack([s['d'][y].read(1, window=win, boundless=True, fill_value=0) for y in YEARS])
    SO = np.stack([s['soil'][k].read(1, window=win, boundless=True, fill_value=0).astype(np.float32) for k in SOILS])
    m3 = lambda X: (sum(X[:, rr + dr, cc + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)) / 9.0).T
    F = [np.nan_to_num(m3(A2)), np.nan_to_num(m3(A5))]
    for j, y in enumerate(YEARS):
        code = DS[j, rr, cc]; cl = np.zeros(len(rr), int); sv = np.zeros(len(rr))
        for k in np.flatnonzero(code > 0): cl[k], sv[k] = s['lut'].get((y, int(code[k])), (0, 0))
        oh = np.zeros((len(rr), NC), np.float32); oh[cl > 0, cl[cl > 0] - 1] = 1
        F += [oh, sv[:, None].astype(np.float32), (HL[rr, cc] == y - 2000).astype(np.float32)[:, None]]
    F.append(np.nan_to_num(SO[:, rr, cc].T)); return rr, cc, np.concatenate(F, 1).astype(np.float32)
def plot_job(a):
    pid, x, y, ev, first, a0 = a; s = _S; row, col = s['land'].index(x, y); RP = int(R // 30) + 2
    rr, cc, F = feats(row - RP, col - RP, 2 * RP + 1, 2 * RP + 1)
    d = np.hypot(rr - RP, cc - RP) * 30; m = d <= R; F, d = F[m], d[m]
    if len(d) == 0: return pid, None, None, None, 0.0
    per = F[:, 128:128 + len(YEARS) * (NC + 2)].reshape(len(d), len(YEARS), NC + 2)
    sr = (per[:, :, 0] + per[:, :, 1] + per[:, :, NC + 1]) > 0; hv = per[:, :, 0] > 0
    w = np.ones(len(d))
    for yv in ev:
        j = yv - 2000; lo, hi = max(j - 1, 0), j + 2
        w *= np.where(sr[:, lo:hi].any(1), 8.0, 1.0) * np.where(hv[:, lo:hi].any(1), 2.0, 1.0)
    if not np.isnan(a0) and not ev:
        since = int(first - a0 + 2) - 2000; last = int(first) - 2000
        if since < len(YEARS) and last >= 0: w *= np.where(sr[:, max(since, 0):last + 1].any(1), 0.1, 1.0)
    w /= w.sum(); conc = float(np.sort(w)[::-1][:max(1, len(w) // 10)].sum())
    if len(d) > NCAND:     # keep every candidate with an elevated prior, fill at random
        hi_ = np.flatnonzero(w > np.median(w) * 1.5); rest = np.setdiff1d(np.arange(len(d)), hi_)
        keep = hi_[np.argsort(-w[hi_])][:NCAND]; nfill = NCAND - len(keep)
        if nfill > 0: keep = np.r_[keep, np.random.default_rng(pid).choice(rest, nfill, replace=False)]
        mass_kept = w[keep].sum(); F, d, w = F[keep], d[keep], w[keep] / mass_kept
    return pid, F.astype(np.float16), d.astype(np.float32), w.astype(np.float32), conc
if __name__ == '__main__':
    con = duckdb.connect('/workspace/landis_region4/data_r4/slim_r4.duckdb', read_only=True)
    t = con.execute("""select t.STATECD st, t.UNITCD u, t.COUNTYCD c, t.PLOT p, t.SUBP sb, t.MEASDATE md, t.DIA, t.HT, t.CR, t.TPA_UNADJ tpa, t.estimated_age age, r.SPECIES_SYMBOL sym, r.SFTWD_HRDWD grp,
                              t.cond_stdage stdage, t.TRTCD1, t.TRTYR1, t.TRTCD2, t.TRTYR2, t.DSTRBCD1, t.DSTRBYR1, t.DSTRBCD2, t.DSTRBYR2, t.cond_stdorgcd org
                       from curated_trees t join REF_SPECIES r on r.SPCD = t.SPCD
                       where t.STATECD in (12, 13) and t.STATUSCD = 1 and t.MEASDATE >= '2000-01-01' order by t.STATECD, t.UNITCD, t.COUNTYCD, t.PLOT, t.MEASDATE, t.SUBP, t.TREE""").df()
    t = t[(t.DIA > 0) & (t.tpa > 0)].copy(); t['md'] = pd.to_datetime(t.md); key = ['st', 'u', 'c', 'p']
    cnt = t.sym.value_counts(); top = list(cnt.index[:int(os.environ.get('TOPK', 16))]); KEYS = sorted(top) + ['_H', '_S']; kix = {k: i for i, k in enumerate(KEYS)}
    t['key'] = [kix[s] if s in top else kix['_' + (g if g in ('S', 'H') else 'H')] for s, g in zip(t.sym, t.grp)]
    t['D'] = t.DIA * 2.54; t['H'] = t.HT * 0.3048; t['W'] = t.tpa * 2.47105; t['CRf'] = t.CR / 100; t['A'] = t.age.astype(float)
    t = t[(t.A > 0) & (t.H > 0)]
    vis = t[key + ['md']].drop_duplicates().sort_values(key + ['md']).reset_index(drop=True); vis['vid'] = np.arange(len(vis))
    vis['year'] = vis.md.dt.year + vis.md.dt.dayofyear / 366; t = t.merge(vis[key + ['md', 'vid']], on=key + ['md'])
    t['ac'] = np.clip((t.A // 10).astype(int), 0, 15); t['WD2'] = t.W * t.D ** 2; t['WH'] = t.W * t.H; t['WA'] = t.W * t.A
    t['crok'] = t.CRf.notna() & (t.CRf > 0); t['WCR'] = np.where(t.crok, t.W * t.CRf, 0); t['WCRW'] = np.where(t.crok, t.W, 0)
    g = t.groupby(['vid', 'key', 'ac'])[['W', 'WD2', 'WH', 'WA', 'WCR', 'WCRW']].sum().reset_index()
    TK = pd.DataFrame({'vid': g.vid, 'key': g.key, 'ac': g.ac, 'N': g.W, 'QMD': np.sqrt(g.WD2 / g.W), 'H': g.WH / g.W, 'A': g.WA / g.W,
                       'CR': np.where(g.WCRW > 0, g.WCR / g.WCRW.replace(0, np.nan), np.nan)})
    pv = vis.groupby(key).ngroup(); vis['pid'] = pv
    org = t.groupby('vid').org.first(); vis['planted'] = vis.vid.map(org).fillna(1).values == 2
    pairs = vis.sort_values(['pid', 'year']).groupby('pid').apply(lambda x: pd.DataFrame({'v1': x.vid.values[:-1], 'v2': x.vid.values[1:]}), include_groups=False).reset_index(level=0)
    print(f'GA+FL: plots {vis.pid.nunique()} visits {len(vis)} tokens {len(TK)} pairs {len(pairs)} keys {KEYS}', flush=True)
    fc = duckdb.connect('/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb', read_only=True)
    pl = fc.execute("select STATECD st, UNITCD u, COUNTYCD c, PLOT p, avg(LAT) lat, avg(LON) lon from PLOT where STATECD in (12, 13) group by all").df()
    P = vis.drop_duplicates('pid')[['pid', *key]].merge(pl, on=key, how='left'); land = rasterio.open(f'{ROOT}/data/derived/land_clean.tif')
    tf = Transformer.from_crs(4326, land.crs, always_xy=True); P['x'], P['y'] = tf.transform(P.lon.values, P.lat.values); P = P[P.lat.notna()]
    ev = {}; tt = t.drop_duplicates(['vid', 'TRTCD1', 'TRTYR1', 'TRTCD2', 'TRTYR2', 'DSTRBCD1', 'DSTRBYR1', 'DSTRBCD2', 'DSTRBYR2'])
    pid_of = vis.set_index('vid').pid
    for r in tt.itertuples():
        for code, yr in ((r.TRTCD1, r.TRTYR1), (r.TRTCD2, r.TRTYR2), (r.DSTRBCD1, r.DSTRBYR1), (r.DSTRBCD2, r.DSTRBYR2)):
            if pd.notna(code) and code > 0 and pd.notna(yr) and 2000 <= yr <= 2025: ev.setdefault(pid_of[r.vid], set()).add(int(yr))
    first = vis.groupby('pid').year.min(); a0 = t.groupby(t.vid.map(pid_of)).stdage.max()
    jobs = [(int(r.pid), r.x, r.y, ev.get(r.pid, set()), float(first[r.pid]), float(a0.get(r.pid, np.nan))) for r in P.itertuples()]
    rows = []; CF = []; n0 = 0; concs = {}
    with Pool(int(os.environ.get('NPROC', 6)), initializer=init) as pool:
        for i, (pid, F, d, w, conc) in enumerate(pool.imap(plot_job, jobs, chunksize=8)):
            if F is None: continue
            rows.append(pd.DataFrame({'pid': pid, 'cid': np.arange(n0, n0 + len(d)), 'dist_m': d, 'prior': w})); CF.append(F); n0 += len(d); concs[pid] = conc
            if i % 500 == 0: print(f'plots {i}/{len(jobs)} candidates {n0}', flush=True)
    C = pd.concat(rows, ignore_index=True); P['conc'] = P.pid.map(concs)
    vis.drop(columns=['md']).to_parquet(f'{OUT}/visits.parquet'); TK.to_parquet(f'{OUT}/tokens.parquet'); pairs.to_parquet(f'{OUT}/pairs.parquet'); P.to_parquet(f'{OUT}/plots.parquet')
    C.to_parquet(f'{OUT}/cands.parquet'); np.save(f'{OUT}/cand_feat.npy', np.concatenate(CF))
    for f in ('stands.parquet', 'stand_feat.npy'):
        os.path.exists(f'{OUT}/{f}') or os.symlink(f'{ROOT}/data/proto/baker/{f}', f'{OUT}/{f}')
    json.dump(dict(keys=KEYS, years=YEARS, classes=list(CLS), soils=SOILS, nf=NF, radius_m=R, ncand=NCAND), open(f'{OUT}/meta.json', 'w'))
    print(f'done: candidates {len(C)} plots {C.pid.nunique()}; localisation concentration quantiles {np.quantile(list(concs.values()), [.1, .5, .9]).round(3)}')
