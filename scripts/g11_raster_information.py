"""FAST information-content test (no neural training; user, 2026-10-22: tie FIA vegetation to rasters on the ground — soils, AlphaEarth, Tessera,
TreeMap, LANDFIRE, terrain). Which raster families, summarised over a plot's 1.6 km disk, predict the plot's FIA vegetation, and does WHERE in the
disk matter (public point / uniform / FIA-history-consistent footprints)?
Unit: each plot's most recent visit since 2015 (closest to the 2022 / 2025 rasters; forested share >= 0.25).
Targets: log BA, log TPH, log QMD (per forested ha, +small offsets), softwood BA share; dominant species key (BA); planted (STDORGCD 1).
Families (per candidate footprint, then a placement-weighted mean):
  aef      AlphaEarth 2022 + 2025 (128)                        soil     SOLUS 0-30 cm (6)
  dist     disturbance in the 5 / 15 yr before the visit: harvest|Hansen>=0.5, any Hansen / 'unknown', fire (6)
  evc      LANDFIRE EVC 2022 tree cover (mean, share >= 50 %) (2)
  evt      LANDFIRE EVT 2022 physiognomy shares (Conifer, Hardwood, Conifer-Hardwood, Riparian, Shrubland, Grassland, other) (7)
  treemap  TreeMap 2022 ln BA, H, ln QMD, cover, ln TPA, softwood share, ln stand age proxy absent (6) — partly CIRCULAR (built from FIA plots)
  terrain  elevation, slope, northness, eastness, TPI 300 m, TPI 2 km (6) — when data/terrain/*.tif exist
Placements: public (candidate nearest the public point), uniform (incl_w), history (g02g consistent set; else uniform).
Model: sklearn HistGradientBoosting, 5-fold spatial-block CV (10 km blocks). Scores: regression R2 and median |error|; classification accuracy /
balanced accuracy; nulls for reference. Writes data/proto/gafl/raster_information.json."""
import os, sys, json, math, numpy as np, pandas as pd, rasterio
from rasterio.windows import Window
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g02b_build_gafl as G2
from features import ROOT, vat_table
D = f'{ROOT}/data/proto/gafl'; NY, NC = len(G2.YEARS), G2.NC; Y0 = G2.YEARS[0]; IX = G2.IX; H_ = G2.H_; RP = int(1600 // 30) + 2
TERR = [k for k in ('elevation', 'slope', 'northness', 'eastness', 'tpi300', 'tpi2k') if os.path.exists(f'{ROOT}/data/terrain/{k}.tif')]
HAS_EVC = os.path.exists(f'{ROOT}/data/derived/tree_cover.tif')        # LANDFIRE EVC 2022 tree cover (09_lf_topo.py)
EVTG = ['Conifer', 'Hardwood', 'Conifer-Hardwood', 'Riparian', 'Shrubland', 'Grassland']
def init():
    global C, CF, CH, VAT, EV, src, GIX
    C = pd.read_parquet(f'{D}/cands.parquet').sort_values('cid').reset_index(drop=True); CF = np.load(f'{D}/cand_feat.npy', mmap_mode='r')
    global GIX; GIX = C.groupby('pid').indices                 # plot -> row positions (one pass; avoids a 2.2M-row scan per plot)
    CH = pd.read_parquet(f'{D}/cand_hist.parquet').set_index('cid').cons
    v = vat_table(); A = np.c_[np.log1p(v.BALIVE.clip(lower=0)), v.STANDHT, np.log1p(v.QMD.clip(lower=0)), v.CANOPYPCT, np.log1p(v.TPA_LIVE.clip(lower=0)), v.SOFTWOOD_FRAC]
    VAT = (dict(zip(v.TM_ID.values.astype(np.int64), range(len(v)))), np.nan_to_num(A.astype(np.float32)))
    e = pd.read_csv(f'{ROOT}/data/landfire/evt_classes.csv'); e = e[e.value >= 0]
    EV = np.full(int(max(e.value.max(), 32767)) + 1, len(EVTG), np.int16)
    for i, g in enumerate(EVTG): EV[e.value[e.EVT_PHYS == g].values] = i
    src = {k: rasterio.open(f'{ROOT}/data/{p}') for k, p in dict(tm='treemap/tm_id.tif', evt='landfire/evt2022.tif').items()}
    for k in TERR: src[k] = rasterio.open(f'{ROOT}/data/terrain/{k}.tif')
    if HAS_EVC: src['evc'] = rasterio.open(f'{ROOT}/data/derived/tree_cover.tif')
def plot_job(a):
    pid, yv = a
    if pid not in GIX: return pid, None
    g = C.iloc[GIX[pid]]
    per = np.asarray(CF[g.cid.values], dtype=np.float32); n = len(g); iw = g.incl_w.values.astype(float)
    r0, c0 = int(g.row.min()) - 1, int(g.col.min()) - 1; h, w = int(g.row.max()) - r0 + 2, int(g.col.max()) - c0 + 2; rr, cc = g.row.values - r0, g.col.values - c0
    tm = src['tm'].read(1, window=Window(c0, r0, w, h), boundless=True, fill_value=0)[rr, cc].astype(np.int64)
    idx = np.array([VAT[0].get(int(t), -1) for t in tm]); TMf = np.where((idx >= 0)[:, None], VAT[1][np.clip(idx, 0, None)], np.nan); tm_ok = idx >= 0
    ev = src['evt'].read(1, window=Window(c0, r0, w, h), boundless=True, fill_value=0)[rr, cc].astype(np.int64); ev = EV[np.clip(ev, 0, len(EV) - 1)]
    EVf = np.eye(len(EVTG) + 1)[ev]
    ds = per[:, 128:128 + NY * (NC + 2)].reshape(n, NY, NC + 2); y = int(math.floor(yv)); feats_d = []
    for back in (5, 15):
        a_, b_ = max(y - back - Y0, 0), min(y - Y0, NY - 1)
        sl = ds[:, a_:b_ + 1] if b_ >= a_ else np.zeros((n, 1, NC + 2), np.float32)
        feats_d += [(np.maximum(sl[:, :, IX['harvest']], sl[:, :, H_]) >= 0.5).any(1), ((sl[:, :, H_] > 0) | (sl[:, :, IX['unknown']] > 0)).any(1), (sl[:, :, IX['fire']] > 0).any(1)]
    DIf = np.stack(feats_d, 1).astype(float)
    TEf = np.stack([src[k].read(1, window=Window(c0, r0, w, h), boundless=True, fill_value=0)[rr, cc] for k in TERR], 1).astype(float) if TERR else np.zeros((n, 0))
    if HAS_EVC:
        tcv = src['evc'].read(1, window=Window(c0, r0, w, h), boundless=True, fill_value=0)[rr, cc].astype(float); ECf = np.c_[tcv / 100, (tcv >= 50).astype(float)]
    else: ECf = np.zeros((n, 0))
    fam = {'aef': per[:, :128].astype(float), 'soil': per[:, -6:].astype(float), 'dist': DIf, 'evt': EVf, 'evc': ECf, 'treemap': TMf, 'terrain': TEf}
    cons = CH.reindex(g.cid.values).fillna(False).values.astype(bool)
    W = {'public': np.eye(n)[int(np.argmin(g.dist_m.values))], 'uniform': iw, 'history': iw * cons if cons.any() else iw}
    out = {}
    for pl, wv in W.items():
        for f, X in fam.items():
            if X.shape[1] == 0: continue
            if f == 'treemap':
                wv2 = wv * tm_ok; out[(pl, f)] = np.nansum(np.nan_to_num(X) * wv2[:, None], 0) / max(wv2.sum(), 1e-12) if wv2.sum() > 0 else np.full(X.shape[1], np.nan)
            else: out[(pl, f)] = (X * wv[:, None]).sum(0) / wv.sum()
    return pid, out
if __name__ == '__main__':
    V = pd.read_parquet(f'{D}/visits.parquet'); TK = pd.read_parquet(f'{D}/tokens.parquet'); P = pd.read_parquet(f'{D}/plots.parquet')
    M = json.load(open(f'{D}/meta.json')); KEYS = M['keys']
    V = V[(V.year >= 2015) & (V.fsub >= 0.25)]; last = V.loc[V.groupby('pid').year.idxmax()]
    tk = TK.assign(ba=TK.N * math.pi / 4 * (TK.QMD / 100) ** 2, d2=TK.N * TK.QMD ** 2)
    agg = tk.groupby('vid').agg(ba=('ba', 'sum'), tph=('N', 'sum'), d2=('d2', 'sum'))
    soft_keys = {i for i, k in enumerate(KEYS) if k in ('PIEL', 'PITA', 'PIPA2', 'PICL', 'TAAS', '_S')}
    sw = tk[tk.key.isin(soft_keys)].groupby('vid').ba.sum(); dom = tk.groupby(['vid', 'key']).ba.sum().reset_index().sort_values('ba').groupby('vid').key.last()
    T = last.set_index('vid')[['pid', 'year', 'planted']].join(agg).fillna({'ba': 0, 'tph': 0, 'd2': 0})
    T['lba'] = np.log(T.ba + 0.5); T['ltph'] = np.log(T.tph + 10); T['lqmd'] = np.log(np.sqrt(T.d2 / T.tph.clip(lower=1e-6)) + 1)
    T['soft'] = (sw.reindex(T.index).fillna(0) / T.ba.clip(lower=1e-6)).clip(0, 1); T['dom'] = dom.reindex(T.index).fillna(-1).astype(int)
    print(f'plots {len(T)}; terrain layers {TERR}', flush=True)
    with Pool(int(os.environ.get('NPROC', 6)), initializer=init) as pool: res = dict(pool.map(plot_job, list(zip(T.pid.values, T.year.values)), chunksize=16))
    T = T[T.pid.map(lambda p: res.get(p) is not None)]
    Pk = P.set_index('pid'); blk = ((Pk.x // 10000).astype(int) * 1000 + (Pk.y // 10000).astype(int)).reindex(T.pid).values
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    fams = ['aef', 'soil', 'dist', 'evt'] + (['evc'] if HAS_EVC else []) + ['treemap'] + (['terrain'] if TERR else [])
    def X_of(pl, fs): return np.hstack([np.vstack([res[p][(pl, f)] for p in T.pid]) for f in fs])
    targets = {'lba': 'reg', 'ltph': 'reg', 'lqmd': 'reg', 'soft': 'reg', 'dom': 'clf', 'planted': 'clf'}
    top = T.dom.value_counts().index[:int(os.environ.get('G11_DOM_TOP', 12))]; T['dom'] = np.where(T.dom.isin(top), T.dom, -2)   # dominant key: top classes, -2 = other (a boosted classifier fits one tree per class)
    gkf = GroupKFold(5); out = {}
    def cv(X, tgt, kind):
        y = T[tgt].values.astype(float if kind == 'reg' else int); pred = np.zeros(len(y)) if kind == 'reg' else np.zeros(len(y), int)
        for tr, te in gkf.split(X, y, blk):
            m = (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0) if kind == 'reg' else
                 HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0))
            m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
        if kind == 'reg':
            return {'R2': float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 'medAE': float(np.median(np.abs(pred - y)))}
        bal = float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))
        return {'acc': float(np.mean(pred == y)), 'bal_acc': bal}
    configs = [('uniform', [f]) for f in fams] + [('uniform', fams), ('uniform', [f for f in fams if f != 'treemap'])] + \
              [('public', [f for f in fams if f != 'treemap']), ('history', [f for f in fams if f != 'treemap'])] + [('uniform', [f for f in fams if f not in (x, 'treemap')]) for x in fams if x != 'treemap']
    for pl, fs in configs:
        X = X_of(pl, fs); key = f"{pl}:{'+'.join(fs) if len(fs) < len(fams) - 1 else ('ALL' if len(fs) == len(fams) else 'ALL-' + '-'.join(sorted(set(fams) - set(fs))))}"
        out[key] = {t: cv(X, t, k) for t, k in targets.items()}
        print(key, {t: {kk: round(vv, 3) for kk, vv in v.items()} for t, v in out[key].items()}, flush=True)
    null = {t: ({'R2': 0.0, 'medAE': float(np.median(np.abs(T[t] - T[t].median())))} if k == 'reg' else {'acc': float(T[t].value_counts(normalize=True).iloc[0])}) for t, k in targets.items()}
    out['null'] = null; print('null', null)
    json.dump(out, open(f'{D}/raster_information.json', 'w'), indent=1)
