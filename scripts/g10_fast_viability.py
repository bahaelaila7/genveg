"""FAST viability test of FIA <-> LANDFIRE / Hansen matching, no neural training (user, 2026-10-22: "prototype viability much faster without full training").
Question: does placing a plot by its FIA records (history-consistent footprints) make the candidate's disturbance evidence predictive of FIA's
measured change, better than the public point, uniform placement, or the FIA records alone?
Per remeasurement pair (both visits with forested share >= 0.25, BA(v1) > 1 m2/ha), target y = log((BA2 + 0.5) / (BA1 + 0.5)).
Candidate evidence state over the period [floor(y1), floor(y2)]: SR (harvest class or Hansen on >= half the footprint) > partial (any Hansen /
'unknown' pixel) > fire (fire class) > none. A placement gives weights over the plot's kept candidates -> expected state shares p_s.
Predictors (least squares on TRAINING pairs, the g03b spatial blocks; scored on held-out pairs):
  zero        y = 0
  records     y ~ FIA record group of the period (cut / regeneration / crown fire / weak / none) + gap        (no satellite evidence)
  ev:<place>  y ~ p_s(placement) + gap                                                                             (evidence only)
  both:<place> y ~ records + p_s(placement) + gap
Placements: public (candidate nearest the public point), uniform (incl_w), history (g02g history-consistent ∩ this period's detectable
records, sequential with skip; RULES variant). Rule variants (env RULES or all): old (permissive), strict, strict_unknown (strict + 'unknown' for
treatments). Prints median |error| by group; writes data/proto/gafl/fast_viability.json."""
import os, sys, json, math, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g02b_build_gafl as G2, g02g_history_consistency as GG
from features import ROOT
D = f'{ROOT}/data/proto/gafl'; NY, NC = len(G2.YEARS), G2.NC; Y0 = G2.YEARS[0]; IX = G2.IX; H_ = G2.H_
C = pd.read_parquet(f'{D}/cands.parquet'); CF = np.load(f'{D}/cand_feat.npy', mmap_mode='r'); V = pd.read_parquet(f'{D}/visits.parquet').set_index('vid')
P = pd.read_parquet(f'{D}/plots.parquet'); PR = pd.read_parquet(f'{D}/pairs.parquet'); TK = pd.read_parquet(f'{D}/tokens.parquet')
E = pd.read_parquet(f'{D}/fia_events.parquet'); N = pd.read_parquet(f'{D}/fia_negs.parquet')
ev = {p: list(zip(g.kind, g.year.astype(int))) for p, g in E.groupby('pid')}; ng = {p: list(zip(g.y_lo.astype(int), g.y_hi.astype(int))) for p, g in N.groupby('pid')}
BA = TK.assign(b=TK.N * math.pi / 4 * (TK.QMD / 100) ** 2).groupby('vid').b.sum()
# spatial split exactly as g03b (10 km blocks, rng(1), 20 % test)
P['blk'] = (P.x // 10000).astype(int) * 1000 + (P.y // 10000).astype(int); blks = P.blk.unique(); rng = np.random.default_rng(1)
test_blk = set(rng.choice(blks, max(1, int(0.2 * len(blks))), replace=False)); test_pid = set(P.pid[P.blk.isin(test_blk)])
VAR = {'old': {'T10': ([IX['harvest'], IX['thin'], IX['unknown'], H_], (-1, 1), 8.0), 'T20': ([IX['harvest'], IX['thin'], IX['fire'], IX['chem'], IX['unknown'], H_], (-1, 1), 8.0),
               'T30': ([IX['harvest'], IX['thin'], IX['chem'], IX['unknown'], H_], (-2, 1), 8.0), 'T40': ([IX['harvest'], IX['unknown'], H_], (-2, 1), 8.0),
               'T50': ([IX['thin'], IX['fire'], IX['chem'], IX['unknown']], (-1, 1), 3.0), 'Dfire': ([IX['fire'], IX['unknown']], (-1, 1), 3.0),
               'Dcrown': ([IX['fire'], IX['unknown'], H_], (-1, 1), 8.0), 'Dins': ([IX['insdis'], IX['unknown'], H_], (-1, 1), 4.0),
               'Dwx': ([IX['weather'], IX['unknown'], H_], (-1, 1), 4.0), 'Dhum': ([IX['harvest'], IX['thin'], IX['dev'], IX['unknown'], H_], (-1, 1), 8.0)},
       'strict': dict(G2.EVK)}
VAR['strict_unknown'] = {k: ((v[0] + [IX['unknown']]) if k in ('T10', 'T20', 'T30', 'T40') else v[0], v[1], v[2]) for k, v in VAR['strict'].items()}
GROUP = lambda kinds: 'regen' if kinds & {'T20', 'T30', 'T40'} else 'cut' if 'T10' in kinds else 'crown' if 'Dcrown' in kinds else 'weak' if kinds else 'none'
# pairs
pairs = []
for v1, v2 in zip(PR.v1.values, PR.v2.values):
    if V.fsub[v1] < 0.25 or V.fsub[v2] < 0.25: continue
    b1 = BA.get(v1, 0.0)
    if b1 <= 1.0: continue
    p_ = int(V.pid[v1]); y1, y2 = float(V.year[v1]), float(V.year[v2]); kinds = {k for k, y in ev.get(p_, []) if y1 <= y + 0.5 <= y2}
    pairs.append((v1, v2, p_, y1, y2, math.log((BA.get(v2, 0.0) + 0.5) / (b1 + 0.5)), GROUP(kinds)))
print(f'pairs {len(pairs)} (groups {pd.Series([p[6] for p in pairs]).value_counts().to_dict()})', flush=True)
byp = {}
for i, p in enumerate(pairs): byp.setdefault(p[2], []).append(i)
order = C.sort_values('cid'); grp_c = dict(list(order.groupby('pid')))
def states(per, a, b):        # per [n, years, NC+2] -> one-hot state over years a..b inclusive
    a, b = max(a - Y0, 0), min(b - Y0, NY - 1); sl = per[:, a:b + 1]
    sr = (np.maximum(sl[:, :, IX['harvest']], sl[:, :, H_]) >= 0.5).any(1); part = ((sl[:, :, H_] > 0) | (sl[:, :, IX['unknown']] > 0)).any(1); fi = (sl[:, :, IX['fire']] > 0).any(1)
    s = np.zeros((len(per), 4)); s[sr, 0] = 1; s[~sr & part, 1] = 1; s[~sr & ~part & fi, 2] = 1; s[~sr & ~part & ~fi, 3] = 1; return s
results = {}
for vname in (os.environ.get('RULES', 'old,strict,strict_unknown')).split(','):
    G2.EVK = VAR[vname]                                    # monkeypatch: clusters / consistent() read G2.EVK at call time
    feats = {pl: np.zeros((len(pairs), 4)) for pl in ('public', 'uniform', 'history')}
    for pid, ixs in byp.items():
        if pid not in grp_c: continue
        g = grp_c[pid]; per = np.asarray(CF[g.cid.values], dtype=np.float32)[:, 128:128 + NY * (NC + 2)].reshape(len(g), NY, NC + 2); iw = g.incl_w.values.astype(float)
        cons, _ = GG.consistent(per, ev.get(pid, []), ng.get(pid, [])); near = int(np.argmin(g.dist_m.values))
        for i in ixs:
            _, _, _, y1, y2, _, _ = pairs[i]; S = states(per, int(math.floor(y1)), int(math.floor(y2)))
            m = cons.copy()
            for c in G2.clusters([(k, y) for k, y in ev.get(pid, []) if y1 <= y + 0.5 <= y2]):
                if c['lr'] < 4: continue
                lo, hi = max(c['lo'] - Y0, 0), min(c['hi'] - Y0, NY - 1) + 1
                if hi > lo:
                    x = (per[:, lo:hi][:, :, sorted(c['cols'])].sum(-1) > 0).any(1)
                    if (m & x).any(): m = m & x
            for pl, w in (('public', np.eye(len(g))[near]), ('uniform', iw), ('history', iw * m)):
                feats[pl][i] = (w[:, None] * S).sum(0) / w.sum()
    y = np.array([p[5] for p in pairs]); gap = np.array([p[4] - p[3] for p in pairs]); grp = np.array([p[6] for p in pairs]); te = np.array([p[2] in test_pid for p in pairs])
    G1 = pd.get_dummies(pd.Series(grp)).reindex(columns=['none', 'weak', 'cut', 'regen', 'crown'], fill_value=0).values.astype(float)
    def fit_eval(X):
        X = np.c_[X, gap, np.ones(len(y))]; beta, *_ = np.linalg.lstsq(X[~te], y[~te], rcond=None); return X @ beta
    preds = {'zero': np.zeros(len(y)), 'records': fit_eval(G1)}
    for pl in ('public', 'uniform', 'history'):
        preds[f'ev:{pl}'] = fit_eval(feats[pl][:, :3]); preds[f'both:{pl}'] = fit_eval(np.c_[G1, feats[pl][:, :3]])
    tab = {}
    for gname in ('regen', 'cut', 'crown', 'weak', 'none', 'all'):
        mk = te & ((grp == gname) if gname != 'all' else True)
        if mk.sum() < 10: continue
        tab[gname] = {'n': int(mk.sum()), **{k: float(np.median(np.abs(v[mk] - y[mk]))) for k, v in preds.items()}}
    # DISCRIMINATION: does the stand-replacing evidence share under a placement separate FIA-measured stand-replacing outcomes (BA2 < 0.3 BA1) from
    # partial ones? AUC on held-out pairs (ranks; ties 0.5), by record group
    def auc(score, lab):
        sc = score[lab == 1]; sn = score[lab == 0]
        if len(sc) < 5 or len(sn) < 5: return None
        r = pd.Series(np.r_[sc, sn]).rank().values; return float((r[:len(sc)].sum() - len(sc) * (len(sc) + 1) / 2) / (len(sc) * len(sn)))
    lab = (y < math.log(0.3 * 1.0 + 1e-9) + 0.0).astype(int)                 # approx BA2/BA1 < 0.3 (log ratio with +0.5 offsets ~ same)
    tab['AUC_stand_replacing'] = {}
    for gname in ('cut', 'regen', 'none', 'events'):
        mk = (te if os.environ.get('AUC_ALL') != '1' else np.ones(len(te), bool)) & ((grp == gname) if gname != 'events' else np.isin(grp, ['cut', 'regen']))
        tab['AUC_stand_replacing'][gname] = {'n': int(mk.sum()), 'share_SR': float(lab[mk].mean()), **{pl: auc(feats[pl][mk, 0] + 0.5 * feats[pl][mk, 1], lab[mk]) for pl in ('public', 'uniform', 'history')}}
    # AUC_ALL=1: the score has no fitted parameters, so all pairs can be used
    pidv = np.array([p[2] for p in pairs]); mk = (te if os.environ.get('AUC_ALL') != '1' else np.ones(len(te), bool)) & np.isin(grp, ['cut', 'regen']); ip = pidv[mk]; up = np.unique(ip); rgb = np.random.default_rng(3); diffs = {'history-uniform': [], 'history-public': []}
    sH, sU, sP, lb = [(feats[pl][mk, 0] + 0.5 * feats[pl][mk, 1]) for pl in ('history', 'uniform', 'public')] + [lab[mk]]
    for _ in range(int(os.environ.get('NBOOT', 300))):              # paired bootstrap over PLOTS (pairs of a plot are not independent)
        pick = rgb.choice(up, len(up)); idx = np.concatenate([np.flatnonzero(ip == q) for q in pick])
        aH, aU, aP = auc(sH[idx], lb[idx]), auc(sU[idx], lb[idx]), auc(sP[idx], lb[idx])
        if None not in (aH, aU, aP): diffs['history-uniform'].append(aH - aU); diffs['history-public'].append(aH - aP)
    tab['AUC_paired_bootstrap_events'] = {k: [float(np.mean(v)), float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in diffs.items()}
    print('AUC (stand-replacing outcome vs evidence share SR + 0.5 partial):', json.dumps(tab['AUC_stand_replacing']), flush=True)
    print('paired bootstrap (plots) of the AUC difference on event pairs [mean, 2.5 %, 97.5 %]:', json.dumps(tab['AUC_paired_bootstrap_events']), flush=True)
    results[vname] = tab
    print(f'\n== rules {vname}: held-out median |error| of log BA change'); print(pd.DataFrame({k: v for k, v in tab.items() if not k.startswith('AUC')}).T.round(3).to_string(), flush=True)
json.dump(results, open(f'{D}/fast_viability.json', 'w'), indent=1)
