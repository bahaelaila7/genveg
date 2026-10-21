"""What the DATA says about matching FIA records to LANDFIRE / Hansen evidence (user, 2026-10-22: "vet the process and the data really well … what the
data says"). For every typed FIA event (g02e: plot, kind, year; also the raw FIA code) and every evidence type e (8 LANDFIRE classes, Hansen loss,
stand-replacing = harvest class or Hansen on >= half the footprint) at coverage threshold t (any pixel / >= 3 of 9 / >= 5 of 9):
  f(o)  = share of the plot's disk footprints (Horvitz-Thompson weights incl_w) showing e in year event + o, o = -4..+4
  base  = the same share averaged over the plot's QUIET years (>= 4 years from any of its FIA events; controls the landscape)
  lift(o) = Σ f(o) / Σ base over events (ratio of means)          — association strength and its TIMING
  inner / ring lift at o = 0, ±1: inner 800 m disk vs 800-1600 m ring — a stronger inner lift means the evidence is LOCAL to the plot
Also FIA's own measured outcome by kind: BA(v2) / BA(v1) of the remeasurement pairs whose period holds the event (median, share < 0.3, > 0.6).
Writes data/proto/gafl/fia_landfire_association.csv (kind, code, evidence, threshold, offset, lift, n) + outcome table; prints summaries."""
import os, sys, math, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g02b_build_gafl as G2
from features import ROOT
D = f'{ROOT}/data/proto/gafl'; NY, NC = len(G2.YEARS), G2.NC; Y0 = G2.YEARS[0]; OFF = list(range(-4, 5))
C = pd.read_parquet(f'{D}/cands.parquet'); CF = np.load(f'{D}/cand_feat.npy', mmap_mode='r'); V = pd.read_parquet(f'{D}/visits.parquet')
E = pd.read_parquet(f'{D}/fia_events.parquet')
# raw FIA codes per (plot, year) for the per-code breakdown (from FIADB via the shared functions)
import duckdb
from features import WS
fc = duckdb.connect(f'{WS}/FIA/FIASQLITE2PGSQL/FIADB.duckdb', read_only=True); FE = G2.fiadb_events(fc); P = pd.read_parquet(f'{D}/plots.parquet')
pidk = dict(zip(map(tuple, P[['st', 'u', 'c', 'p']].values), P.pid)); raw = []
for r in FE.itertuples():
    k_ = (r.st, r.u, r.c, r.p)
    if k_ not in pidk: continue
    for cc_, yy_, trt in (('TRTCD1', 'TRTYR1', 1), ('TRTCD2', 'TRTYR2', 1), ('TRTCD3', 'TRTYR3', 1), ('DSTRBCD1', 'DSTRBYR1', 0), ('DSTRBCD2', 'DSTRBYR2', 0), ('DSTRBCD3', 'DSTRBYR3', 0)):
        cd, yr = getattr(r, cc_), getattr(r, yy_)
        if pd.notna(cd) and cd > 0 and pd.notna(yr) and 2000 <= yr <= 2025: raw.append((pidk[k_], ('T' if trt else 'D') + str(int(cd)), int(yr)))
RAW = pd.DataFrame(raw, columns=['pid', 'code', 'year']).drop_duplicates()
EV_ALL = {p: set(g.year) for p, g in RAW.groupby('pid')}
EVID = {G2.CLS_NAMES[i] if hasattr(G2, 'CLS_NAMES') else list(G2.CLS)[i]: i for i in range(NC)}; EVID['hansen'] = G2.H_; EVID['stand_replacing'] = -1
TH = {'any': 1e-6, '>=3/9': 3 / 9 - 1e-6, '>=5/9': 5 / 9 - 1e-6}
IXc = G2.IX; H_ = G2.H_
def kind_of(code): return G2.ev_kind(code[0] == 'T', int(code[1:]))
OLD = {k: (set(v[0]), v[1]) for k, v in G2.EVK.items()}
NEW = {'T10': ({IXc['harvest'], IXc['thin'], H_}, (-1, 1)), 'T20': ({IXc['harvest'], IXc['chem'], IXc['fire'], H_}, (-1, 1)),
       'T30': ({IXc['harvest'], IXc['chem'], H_}, (-2, 1)), 'T40': ({IXc['harvest'], IXc['fire'], H_}, (-2, 1)), 'T50': ({IXc['thin'], IXc['chem']}, (-1, 1)),
       'Dfire': ({IXc['fire']}, (-1, 1)), 'Dcrown': ({IXc['fire'], H_}, (-1, 1)), 'Dins': ({IXc['insdis']}, (-1, 1)), 'Dwx': ({IXc['weather']}, (-1, 1)),
       'Dhum': ({IXc['harvest'], IXc['dev']}, (-1, 1)), 'Dunk': (set(), (-1, 1))}
RULES = {'old': OLD, 'new': NEW}
racc = {}           # (rule, code, zone) -> [sum share passing at the FIA window, sum base share, n]
acc = {}            # (code, evidence, threshold, offset, zone) -> [sum f, sum base, n]
for pid, g in RAW.groupby('pid'):
    cg = C[C.pid == pid]
    if not len(cg): continue
    per = np.asarray(CF[cg.cid.values], dtype=np.float32)[:, 128:128 + NY * (NC + 2)].reshape(len(cg), NY, NC + 2); iw = cg.incl_w.values.astype(float)
    inn = cg.dist_m.values <= 800; quiet = [j for j, y in enumerate(G2.YEARS) if all(abs(y - e) >= 4 for e in EV_ALL[pid])]
    if len(quiet) < 3: continue
    evm = {}
    for en, ix in EVID.items():
        x = np.maximum(per[:, :, G2.IX['harvest']], per[:, :, G2.H_]) if ix == -1 else per[:, :, ix]
        for tn, t in TH.items():
            if ix == -1 and tn != '>=5/9': continue
            evm[(en, tn)] = x >= (0.5 - 1e-6 if ix == -1 else t)                  # [cands, years] evidence indicator
    for code, years in g.groupby('code').year:
        kd = kind_of(code)
        for y in set(years):
            for rn, R_ in RULES.items():          # share of footprints a RULE lets through in the record's window vs the same window shape in quiet years
                if kd is None or kd not in R_ or not R_[kd][0]: continue
                cols, (dlo, dhi) = R_[kd]; X = per[:, :, sorted(cols)].sum(-1) > 0
                def win_share(yc, zm):
                    lo, hi = max(yc + dlo - Y0, 0), min(yc + dhi - Y0, NY - 1)
                    w = iw * zm; return float((w * X[:, lo:hi + 1].any(1)).sum() / w.sum()) if hi >= lo and w.sum() > 0 else np.nan
                for zone, zm in (('all', np.ones(len(iw), bool)), ('inner', inn), ('ring', ~inn)):
                    f0 = win_share(y, zm); bq = np.nanmean([win_share(yq, zm) for yq in (Y0 + j for j in quiet)])
                    if np.isfinite(f0) and np.isfinite(bq): a_ = racc.setdefault((rn, code, zone), [0.0, 0.0, 0]); a_[0] += f0; a_[1] += bq; a_[2] += 1
            for (en, tn), M in evm.items():
                for zone, zm in (('all', np.ones(len(iw), bool)), ('inner', inn), ('ring', ~inn)):
                    w = iw * zm
                    if w.sum() <= 0: continue
                    sh = (w[:, None] * M).sum(0) / w.sum(); base = sh[quiet].mean()
                    for o in OFF if zone == 'all' else (-1, 0, 1):
                        j = y + o - Y0
                        if 0 <= j < NY:
                            a = acc.setdefault((code, en, tn, o, zone), [0.0, 0.0, 0]); a[0] += sh[j]; a[1] += base; a[2] += 1
PSEUDO = 0.002      # floor on the mean base share (0.2 % of the disk): rare classes with a ~zero base no longer give absurd lifts
rows = [dict(code=k[0], evidence=k[1], threshold=k[2], offset=k[3], zone=k[4], lift=(v[0] / v[2] + PSEUDO) / (v[1] / v[2] + PSEUDO), f=v[0] / v[2], base=v[1] / v[2], n=v[2]) for k, v in acc.items()]
A = pd.DataFrame(rows); A.to_csv(f'{D}/fia_landfire_association.csv', index=False)
# FIA's own outcome by code: BA(v2)/BA(v1) of the pairs whose period holds the event
TK = pd.read_parquet(f'{D}/tokens.parquet'); BA = (TK.assign(b=TK.N * math.pi / 4 * (TK.QMD / 100) ** 2).groupby('vid').b.sum()); PR = pd.read_parquet(f'{D}/pairs.parquet')
Vy = V.set_index('vid').year; Vp = V.set_index('vid').pid; out = []
for v1, v2 in zip(PR.v1, PR.v2):
    b1 = BA.get(v1, 0.0)
    if b1 <= 1.0: continue
    r_ = BA.get(v2, 0.0) / b1; p_ = Vp[v1]; y1, y2 = Vy[v1], Vy[v2]
    codes = set(RAW[(RAW.pid == p_) & (RAW.year + 0.5 >= y1) & (RAW.year + 0.5 <= y2)].code) or {'none'}
    for cd in codes: out.append((cd, r_))
O = pd.DataFrame(out, columns=['code', 'ratio']).groupby('code').ratio.agg(n='size', median='median', share_lt_0_3=lambda x: float((x < 0.3).mean()), share_gt_0_6=lambda x: float((x > 0.6).mean()))
O.to_csv(f'{D}/fia_outcome_by_code.csv')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('FIA-measured outcome BA(v2)/BA(v1) by FIA code (pairs whose period holds the event):'); print(O[O.n >= 30].sort_values('median').round(3).to_string())
top = A[(A.zone == 'all') & (A.n >= 40)]
piv = top[top.offset == 0].pivot_table(index='code', columns=['evidence', 'threshold'], values='lift')
print('\nlift at the FIA year (offset 0), disk level, by code x evidence x coverage threshold:'); print(piv.round(2).to_string())
print('\ntiming of the strongest evidence per code (any-pixel threshold): lift by offset')
for code, g in top[top.threshold.isin(['any', '>=5/9'])].groupby('code'):
    best = g[g.offset == 0].sort_values('lift', ascending=False).head(2)
    for _, b in best.iterrows():
        prof = g[(g.evidence == b.evidence) & (g.threshold == b.threshold)].set_index('offset').lift.reindex(OFF)
        print(f'  {code:5s} {b.evidence:16s} {b.threshold:6s} n {int(b.n):4d}: ' + ' '.join(f'{o:+d}:{v:.2f}' for o, v in prof.items()))
loc = A[(A.zone != 'all') & (A.offset.isin([-1, 0, 1])) & (A.n >= 40)].groupby(['code', 'evidence', 'threshold', 'zone']).lift.mean().unstack('zone')
loc['inner_over_ring'] = loc.inner / loc.ring
print('\nlocality (mean lift over offsets -1..+1, inner 800 m vs ring) for the strongest code x evidence pairs:')
print(loc.sort_values('inner', ascending=False).head(25).round(2).to_string())
R = pd.DataFrame([dict(rule=k[0], code=k[1], zone=k[2], passing=v[0] / v[2], base=v[1] / v[2], lift=(v[0] / v[2] + PSEUDO) / (v[1] / v[2] + PSEUDO), n=v[2]) for k, v in racc.items()])
R.to_csv(f'{D}/fia_rule_comparison.csv', index=False)
Rp = R[R.n >= 30].pivot_table(index='code', columns=['rule', 'zone'], values=['passing', 'lift'])
print('\nRULE SETS (old g02b.EVK vs stricter): share of disk footprints passing in the record window ("passing"), lift vs quiet years, inner / ring:')
print(Rp.round(3).to_string())
