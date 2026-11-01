"""Per-region derivation of the FIA-record -> evidence table (docs/DECISIONS.md G34 / G51.6-7), and the association analysis it rests on (user,
2026-10-22: "vet the process and the data really well … what the data says").
  .venv/bin/python -I scripts/g09_fia_landfire_association.py [--out DIR] [--association] [--legacy-table]
DATA = the region's table, REGION / AEF90 as that table (region_check). A region without an EVK_TABLES entry is bootstrapped with a table built
under EVK_PROVISIONAL=1 (g02b: the GA+FL rows; the candidate sample is design-based, so the measurement below does not depend on the provisional
lr). The FIA records are g02e's fia_events of THAT table build (provenance checked); FIADB (common.fiadb(), read-only) supplies the raw codes of
the quiet years under the table's canonical plot keys (key_remap.parquet).

Rows (classes / window / detectability per record kind) follow the derivation RULES below, the 2026-10-22 GA+FL analysis written as rules:
  - mechanism classes per kind (MECH: cutting -> harvest / thin, site preparation -> harvest / chemical / fire, …);
  - Hansen loss for the kinds whose FIA-measured outcome is stand-replacing (STAND_REPLACING: cutting, site preparation, artificial / natural
    regeneration, crown fire); these are also the DETECTABLE kinds (the hard constraints; G51.7 keeps detectability fixed);
  - LANDFIRE 'unknown' (MIICA change without an assigned cause) for cutting / site preparation / regeneration (CHANGE);
  - window (-1, +1) around the FIA year, (-2, +1) for regeneration (LEAD2: planting / regeneration follow the cut).
lr = the MEASURED inner-zone lift of the row (LR_METHOD): per record (plot, kind, year), f = the incl_w-weighted share of the plot's candidate
footprints within R_IN of the public coordinate showing any of the row's classes (> 0 of the 3x3) in [year + dlo, year + dhi]; base = the mean of
the same share over the plot's QUIET windows (centre years >= 4 yr from any FIA code year of the plot, >= 3 of them); lift = (mean f + PSEUDO) /
(mean base + PSEUDO) over the row's records (ratio of means; PSEUDO = 0.002 of the zone, the floor of the association analysis: a class that passes
~0.01 % of the quiet footprints, insects / weather / development, would otherwise weigh ×5-31 on a few records). A row with fewer than N_MIN records
takes the pooled lift of the rows of its detectability; a lift below 1 is floored at 1 (no evidence weight); lr is rounded to 3 decimals.
Writes OUT/evk_proposal.json (region, the table's build id and evk_provisional, the rows with their measurement, the EVK_TABLES entry as Python text
for the coordinator to commit, its evk_json / evk_hash). Self-test on GA+FL (REGION 12,13): the rules reproduce EVK_TABLES[(12, 13)]'s classes /
windows / detectability exactly (including 'unknown'), and the measured lr equal its committed lr; exit 1 otherwise.
--association: also the association analysis per FIA code (evidence class x coverage threshold x offset, disk / inner / ring; FIA's measured
outcome BA(v2) / BA(v1) by code; the table's rule vs the derived rules) -> OUT/fia_landfire_association.csv, fia_outcome_by_code.csv,
fia_rule_comparison.csv. --legacy-table: accept a table built before G51.5 (no build_id / provenance; e.g. gafl_v4), reported as such."""
import os, sys, math, json, argparse, numpy as np, pandas as pd, duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g02b_build_gafl as G2
from features import ROOT
from common import fiadb, fiadb_con
from region import region_check, region_set
NY, NC = len(G2.YEARS), G2.NC; Y0 = G2.YEARS[0]; OFF = list(range(-4, 5)); IXc = G2.IX; H_ = G2.H_
MECH = {'T10': ['harvest', 'thin'], 'T20': ['harvest', 'chem', 'fire'], 'T30': ['harvest', 'chem'], 'T40': ['harvest', 'fire'],
        'T50': ['thin', 'chem'], 'Dfire': ['fire'], 'Dcrown': ['fire'], 'Dins': ['insdis'], 'Dwx': ['weather'], 'Dhum': ['harvest', 'dev']}
STAND_REPLACING = {'T10', 'T20', 'T30', 'T40', 'Dcrown'}
CHANGE = {'T10', 'T20', 'T30', 'T40'}
LEAD2 = {'T30', 'T40'}
N_MIN = 30; PSEUDO = 0.002
LR_METHOD = ('g09 (G51.7): inner-zone lift = (mean f + PSEUDO) / (mean base + PSEUDO) over the row\'s FIA records, f = incl_w share of the candidate '
             f'footprints within R_IN = {G2.R_IN} m showing any of the row\'s classes (> 0 of 3x3) in [year + dlo, year + dhi], base = mean of that share '
             f'over the plot\'s quiet windows (>= 4 yr from any FIA code year, >= 3 windows), PSEUDO = {PSEUDO}; rows with < {N_MIN} records take the '
             'pooled lift of their detectability; floored at 1; rounded to 3 decimals')
def lift_of(sf, sb, n): return (sf / n + PSEUDO) / (sb / n + PSEUDO) if n else np.nan
def rule_rows():
    """kind -> (class indices, (dlo, dhi), detectable) from the derivation rules"""
    out = {}
    for k, m in MECH.items():
        cols = [IXc[c] for c in m] + ([IXc['unknown']] if k in CHANGE else []) + ([H_] if k in STAND_REPLACING else [])
        out[k] = (cols, (-2, 1) if k in LEAD2 else (-1, 1), k in STAND_REPLACING)
    return out
def cname(i): return 'H' if i == H_ else list(G2.CLS)[i]
def load(D, legacy):
    META = json.load(open(f'{D}/meta.json')); region_check(META)
    if legacy and not META.get('build_id'): print(f'!! {D}: legacy table (no build_id / provenance, built before G51.5): records read unchecked', flush=True)
    else: G2.check_prov(f'{D}/fia_events.parquet', META)
    C = pd.read_parquet(f'{D}/cands.parquet'); CF = np.load(f'{D}/cand_feat.npy', mmap_mode='r'); P = pd.read_parquet(f'{D}/plots.parquet')
    E = pd.read_parquet(f'{D}/fia_events.parquet').drop_duplicates(['pid', 'kind', 'year'])
    rm = f'{D}/key_remap.parquet'
    if not os.path.exists(rm) and not legacy: sys.exit(f'{rm} missing (table built before G51.2): rebuild the table')
    REMAP = pd.read_parquet(rm) if os.path.exists(rm) else None
    fc = fiadb_con(); FE = G2.fiadb_events(fc, REMAP)
    pidk = dict(zip(map(tuple, P[['st', 'u', 'c', 'p']].astype(int).values), P.pid)); raw = []
    for r in FE.itertuples():
        k_ = (int(r.st), int(r.u), int(r.c), int(r.p))
        if k_ not in pidk: continue
        for cc_, yy_, trt in (('TRTCD1', 'TRTYR1', 1), ('TRTCD2', 'TRTYR2', 1), ('TRTCD3', 'TRTYR3', 1), ('DSTRBCD1', 'DSTRBYR1', 0), ('DSTRBCD2', 'DSTRBYR2', 0), ('DSTRBCD3', 'DSTRBYR3', 0)):
            cd, yr = getattr(r, cc_), getattr(r, yy_)
            if pd.notna(cd) and cd > 0 and pd.notna(yr) and 2000 <= yr <= 2025: raw.append((pidk[k_], ('T' if trt else 'D') + str(int(cd)), int(yr)))
    RAW = pd.DataFrame(raw, columns=['pid', 'code', 'year']).drop_duplicates()
    return META, C.sort_values('cid').reset_index(drop=True), CF, P, E, RAW
def plots_iter(C, CF, pids):
    """per plot of pids: (pid, per [cands, years, NC + 2] candidate evidence, incl_w, inner-zone mask)"""
    ix_ = C.groupby('pid').indices
    for pid in pids:
        if pid not in ix_: continue
        ii = ix_[pid]; cg = C.iloc[ii]
        per = np.asarray(CF[cg.cid.values], dtype=np.float32)[:, 128:128 + NY * (NC + 2)].reshape(len(cg), NY, NC + 2)
        yield pid, per, cg.incl_w.values.astype(float), cg.dist_m.values <= G2.R_IN, cg.dist_m.values
def win_share(X, w, yc, dlo, dhi):
    lo, hi = max(yc + dlo - Y0, 0), min(yc + dhi - Y0, NY - 1)
    return float((w * X[:, lo:hi + 1].any(1)).sum() / w.sum()) if hi >= lo and w.sum() > 0 else np.nan
def measure(C, CF, E, RAW, rows):
    """per row: [Σ f_inner, Σ base_inner, n, Σ f_disk, Σ base_disk] and per (row, single class) the inner sums (report-only)"""
    EV_ALL = {p: set(g.year) for p, g in RAW.groupby('pid')}; acc = {}; cacc = {}; Eg = dict(list(E.groupby('pid')))
    for pid, per, iw, inn, _ in plots_iter(C, CF, sorted(Eg)):
        quiet = [Y0 + j for j in range(NY) if all(abs(Y0 + j - e) >= 4 for e in EV_ALL.get(pid, ()))]
        if len(quiet) < 3 or (iw * inn).sum() <= 0: continue
        for kind, year in zip(Eg[pid].kind, Eg[pid].year):
            if kind not in rows: continue
            cols, (dlo, dhi), _ = rows[kind]
            for key_, cs in [((kind, None), cols)] + [((kind, c), [c]) for c in cols]:
                X = per[:, :, sorted(cs)].sum(-1) > 0; out = []
                for zm in (inn, np.ones(len(iw), bool)):
                    w = iw * zm; f0 = win_share(X, w, int(year), dlo, dhi); bq = np.nanmean([win_share(X, w, yq, dlo, dhi) for yq in quiet])
                    out.append((f0, bq))
                (fi, bi), (fd, bd) = out
                if not (np.isfinite(fi) and np.isfinite(bi)): continue
                tgt = acc if key_[1] is None else cacc
                a_ = tgt.setdefault(key_, [0.0, 0.0, 0, 0.0, 0.0]); a_[0] += fi; a_[1] += bi; a_[2] += 1; a_[3] += fd; a_[4] += bd
    return acc, cacc
def derive(acc, cacc, rows):
    pool = {}
    for (k, _), v in acc.items():
        p_ = pool.setdefault(rows[k][2], [0.0, 0.0, 0]); p_[0] += v[0]; p_[1] += v[1]; p_[2] += v[2]
    out = {}
    for k, (cols, win, det) in rows.items():
        v = acc.get((k, None), [0.0, 0.0, 0, 0.0, 0.0]); lift = lift_of(v[0], v[1], v[2])
        if v[2] >= N_MIN: lr, src = lift, 'measured'
        else:
            pp = pool.get(det, [0.0, 0.0, 0]); lr = lift_of(*pp) if pp[2] else 1.0; src = f'pooled {"detectable" if det else "weak"} rows (n {v[2]} < {N_MIN})'
        if lr < 1: lr, src = 1.0, src + ', floored at 1'
        out[k] = dict(classes=[cname(c) for c in cols], class_idx=[int(c) for c in cols], window=[int(win[0]), int(win[1])], detectable=bool(det),
                      lr=round(float(lr), 3), lr_source=src, n=int(v[2]), lift_inner=None if not np.isfinite(lift) else float(lift),
                      lift_inner_raw=v[0] / v[1] if v[1] > 0 else None, f_inner=v[0] / v[2] if v[2] else None, base_inner=v[1] / v[2] if v[2] else None,
                      lift_disk=lift_of(v[3], v[4], v[2]) if v[2] else None,
                      class_lift_inner={cname(c): (lift_of(*cacc[(k, c)][:3]) if (k, c) in cacc else None) for c in cols})
    return out
def entry_text(region, rows):
    nm = lambda i: "H_" if i == H_ else f"IX['{cname(i)}']"
    body = ',\n'.join(f"    '{k}': ([{', '.join(nm(c) for c in r['class_idx'])}], ({r['window'][0]}, {r['window'][1]}), {r['lr']}, {r['detectable']})" for k, r in rows.items())
    return f'EVK_TABLES[{tuple(region)}] = {{\n{body}}}'
def association(D, OUT, C, CF, P, E, RAW, rules_old, rules_new):
    """the per-code association analysis (2026-10-22): lifts by evidence class x coverage threshold x offset x zone, the measured outcome by code,
    and the share of footprints a rule set lets through in the record windows"""
    V = pd.read_parquet(f'{D}/visits.parquet'); EV_ALL = {p: set(g.year) for p, g in RAW.groupby('pid')}
    EVID = {list(G2.CLS)[i]: i for i in range(NC)}; EVID['hansen'] = H_; EVID['stand_replacing'] = -1
    TH = {'any': 1e-6, '>=3/9': 3 / 9 - 1e-6, '>=5/9': 5 / 9 - 1e-6}
    def kind_of(code): return G2.ev_kind(code[0] == 'T', int(code[1:]))
    RULES = {'table': rules_old, 'derived': rules_new}; racc = {}; acc = {}; Rg = dict(list(RAW.groupby('pid')))
    for pid, per, iw, inn, dist in plots_iter(C, CF, sorted(Rg)):
        g = Rg[pid]; quiet = [j for j, y in enumerate(G2.YEARS) if all(abs(y - e) >= 4 for e in EV_ALL[pid])]
        if len(quiet) < 3: continue
        inn8 = dist <= 800; evm = {}
        for en, ix in EVID.items():
            x = np.maximum(per[:, :, IXc['harvest']], per[:, :, H_]) if ix == -1 else per[:, :, ix]
            for tn, t in TH.items():
                if ix == -1 and tn != '>=5/9': continue
                evm[(en, tn)] = x >= (0.5 - 1e-6 if ix == -1 else t)
        for code, years in g.groupby('code').year:
            kd = kind_of(code)
            for y in set(years):
                for rn, R_ in RULES.items():
                    if kd is None or kd not in R_ or not R_[kd][0]: continue
                    cols, (dlo, dhi) = R_[kd][0], R_[kd][1]; X = per[:, :, sorted(cols)].sum(-1) > 0
                    for zone, zm in (('all', np.ones(len(iw), bool)), ('inner', inn8), ('ring', ~inn8)):
                        w = iw * zm; f0 = win_share(X, w, y, dlo, dhi); bq = np.nanmean([win_share(X, w, Y0 + j, dlo, dhi) for j in quiet])
                        if np.isfinite(f0) and np.isfinite(bq): a_ = racc.setdefault((rn, code, zone), [0.0, 0.0, 0]); a_[0] += f0; a_[1] += bq; a_[2] += 1
                for (en, tn), M in evm.items():
                    for zone, zm in (('all', np.ones(len(iw), bool)), ('inner', inn8), ('ring', ~inn8)):
                        w = iw * zm
                        if w.sum() <= 0: continue
                        sh = (w[:, None] * M).sum(0) / w.sum(); base = sh[quiet].mean()
                        for o in OFF if zone == 'all' else (-1, 0, 1):
                            j = y + o - Y0
                            if 0 <= j < NY: a = acc.setdefault((code, en, tn, o, zone), [0.0, 0.0, 0]); a[0] += sh[j]; a[1] += base; a[2] += 1
    PSEUDO = 0.002      # floor on the mean base share (0.2 % of the disk): rare classes with a ~zero base no longer give absurd lifts
    A = pd.DataFrame([dict(code=k[0], evidence=k[1], threshold=k[2], offset=k[3], zone=k[4], lift=(v[0] / v[2] + PSEUDO) / (v[1] / v[2] + PSEUDO), f=v[0] / v[2],
                           base=v[1] / v[2], n=v[2]) for k, v in acc.items()])
    A.to_csv(f'{OUT}/fia_landfire_association.csv', index=False)
    TK = pd.read_parquet(f'{D}/tokens.parquet'); BA = (TK.assign(b=TK.N * math.pi / 4 * (TK.QMD / 100) ** 2).groupby('vid').b.sum()); PR = pd.read_parquet(f'{D}/pairs.parquet')
    Vy = V.set_index('vid').year; Vp = V.set_index('vid').pid; out = []
    for v1, v2 in zip(PR.v1, PR.v2):
        b1 = BA.get(v1, 0.0)
        if b1 <= 1.0: continue
        r_ = BA.get(v2, 0.0) / b1; p_ = Vp[v1]; y1, y2 = Vy[v1], Vy[v2]
        for cd in set(RAW[(RAW.pid == p_) & (RAW.year + 0.5 >= y1) & (RAW.year + 0.5 <= y2)].code) or {'none'}: out.append((cd, r_))
    O = pd.DataFrame(out, columns=['code', 'ratio']).groupby('code').ratio.agg(n='size', median='median', share_lt_0_3=lambda x: float((x < 0.3).mean()), share_gt_0_6=lambda x: float((x > 0.6).mean()))
    O.to_csv(f'{OUT}/fia_outcome_by_code.csv')
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print('FIA-measured outcome BA(v2)/BA(v1) by FIA code (pairs whose period holds the event):'); print(O[O.n >= 30].sort_values('median').round(3).to_string())
    top = A[(A.zone == 'all') & (A.n >= 40)]
    if len(top): print('\nlift at the FIA year (offset 0), disk level, by code x evidence x coverage threshold:'); print(top[top.offset == 0].pivot_table(index='code', columns=['evidence', 'threshold'], values='lift').round(2).to_string())
    R = pd.DataFrame([dict(rule=k[0], code=k[1], zone=k[2], passing=v[0] / v[2], base=v[1] / v[2], lift=(v[0] / v[2] + PSEUDO) / (v[1] / v[2] + PSEUDO), n=v[2]) for k, v in racc.items()])
    R.to_csv(f'{OUT}/fia_rule_comparison.csv', index=False)
    if len(R): print('\nRULE SETS (the table\'s evk vs the derived rules): share of disk footprints passing in the record window, lift vs quiet years:'); print(R[R.n >= 30].pivot_table(index='code', columns=['rule', 'zone'], values=['passing', 'lift']).round(3).to_string())
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=None); ap.add_argument('--association', action='store_true'); ap.add_argument('--legacy-table', action='store_true')
    A_ = ap.parse_args()
    D = f"{ROOT}/data/proto/{os.environ.get('DATA', 'gafl')}"; OUT = A_.out or D; os.makedirs(OUT, exist_ok=True)
    META, C, CF, P, E, RAW = load(D, A_.legacy_table); REG = region_set(META.get('region', (12, 13)))
    rows = rule_rows(); acc, cacc = measure(C, CF, E, RAW, rows); R_ = derive(acc, cacc, rows)
    pd.set_option('display.width', 250)
    print(pd.DataFrame(R_).T[['detectable', 'n', 'f_inner', 'base_inner', 'lift_inner_raw', 'lift_inner', 'lift_disk', 'lr', 'lr_source']].to_string(), flush=True)
    ej = {k: [r['class_idx'], r['window'], r['lr'], r['detectable']] for k, r in R_.items()}
    prop = dict(region=list(REG), table=os.path.relpath(D, ROOT), table_build_id=META.get('build_id'), table_evk_provisional=bool(META.get('evk_provisional', False)),
                table_evk_hash=META.get('evk_hash'), lr_method=LR_METHOD, n_min=N_MIN, rows=R_, evk_json=ej, evk_hash=G2.evk_hash(ej),
                evk_tables_entry=entry_text(REG, R_), evk_lr_method_entry=f'EVK_LR_METHOD[{tuple(REG)}] = {LR_METHOD!r}')
    G2.write_json_atomic(prop, f'{OUT}/evk_proposal.json'); print(f'proposal: {OUT}/evk_proposal.json\n{prop["evk_tables_entry"]}', flush=True)
    rc = 0
    if REG == (12, 13):         # self-test: the rules and the measurement reproduce the committed GA+FL table
        ref = G2.EVK_TABLES[(12, 13)]
        bad = {k: (rows.get(k), (sorted(v[0]), v[1], v[3])) for k, v in ref.items() if k not in rows or (sorted(rows[k][0]), tuple(rows[k][1]), rows[k][2]) != (sorted(v[0]), tuple(v[1]), v[3])}
        bad.update({k: (rows[k], None) for k in rows if k not in ref})
        print(f"SELFTEST rules = EVK_TABLES[(12, 13)] classes / windows / detectable (incl. 'unknown'): {'PASS' if not bad else 'FAIL ' + str(bad)}")
        dlr = {k: (R_[k]['lr'], float(v[2])) for k, v in ref.items() if k in R_ and abs(R_[k]['lr'] - float(v[2])) > 5e-4}
        print(f"SELFTEST measured lr = the committed GA+FL lr: {'PASS' if not dlr else 'FAIL (measured, committed) ' + str(dlr)}")
        rc = int(bool(bad or dlr))
    if A_.association:
        old = {k: (list(v[0]), tuple(v[1])) for k, v in G2.evk_from_meta(META).items()} if 'evk' in META else {k: (v[0], v[1]) for k, v in rows.items()}
        association(D, OUT, C, CF, P, E, RAW, old, {k: (v[0], v[1]) for k, v in rows.items()})
    sys.exit(rc)
