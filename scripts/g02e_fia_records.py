"""FIA disturbance / treatment records of the training table, for anchoring the dynamics (user, 2026-10-20: FIA's own COND.DSTRB* / COND.TRT* records
anchor FIA data to the disturbances LANDFIRE reports). The same records + rules that built the location prior (g02b.fia_records, shared code):
  fia_events.parquet  pid, kind, year, rmd       typed FIA events (kinds of g02b.EVK) from every forested condition, with the RECORDING visit's date
                                                 (rmd; a record describes the interval ending at that visit — g03b.recs_in, audit r5 I3)
  fia_negs.parquet    pid, y_lo, y_hi            periods without stand-replacing disturbance (FIA protocol; see g02b.fia_records)
  visits.parquet      + fia_blocked              the visit carries a code that could hide stand-replacing evidence (no negative evidence for its period)
                      + plot_converted           the plot turned entirely non-forest at a visit up to 2022 after a forested one (no MIL; r13 G41)
                      + fia_gap                  FIA's previous measurement of the plot (PREV_PLT_CN, any status) is NOT the previous table visit (a
                                                 non-forest / non-sampled visit in between, or unknown): the visit's codes do not cover the whole pair (r6 M1)
Run after every g02b rebuild (pid / vid are row numbers of the current table)."""
import os, sys, numpy as np, pandas as pd, duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g02b_build_gafl as G2
import json as _json; from region import region_check; region_check(_json.load(open(f"{G2.ROOT}/data/proto/{os.environ.get('DATA', 'gafl')}/meta.json")))
from features import ROOT, WS
from common import curated_db, fiadb
D = f"{ROOT}/data/proto/{os.environ.get('DATA', 'gafl')}"; key = ['st', 'u', 'c', 'p']
fc = duckdb.connect(fiadb(), read_only=True)
P = pd.read_parquet(f'{D}/plots.parquet'); V = pd.read_parquet(f'{D}/visits.parquet')
FV = G2.fiadb_visits(fc); FE = G2.fiadb_events(fc)
evs, negs, blocked = G2.fia_records(FE, FV, P, key)
pidk = dict(zip(map(tuple, P[key].values), P.pid)); rows = set()        # the same records as G2.fia_records (typed kinds, years 2000-2025), per recording visit
for r in FE.itertuples():
    k_ = (r.st, r.u, r.c, r.p)
    if k_ not in pidk: continue
    for cc_, yy_, trt in (('TRTCD1', 'TRTYR1', 1), ('TRTCD2', 'TRTYR2', 1), ('TRTCD3', 'TRTYR3', 1), ('DSTRBCD1', 'DSTRBYR1', 0), ('DSTRBCD2', 'DSTRBYR2', 0), ('DSTRBCD3', 'DSTRBYR3', 0)):
        code, yr = getattr(r, cc_), getattr(r, yy_)
        if pd.notna(code) and code > 0 and pd.notna(yr) and 2000 <= yr <= 2025:
            kind = G2.ev_kind(trt, code)
            if kind: rows.add((pidk[k_], kind, int(yr), r.md))
FEV = pd.DataFrame(sorted(rows), columns=['pid', 'kind', 'year', 'rmd'])
assert set(map(tuple, FEV[['pid', 'kind', 'year']].values)) == {(p, k, y) for p, s in evs.items() for k, y in s}   # = the location prior's records
FEV.to_parquet(f'{D}/fia_events.parquet')
pd.DataFrame([(p, a, b) for p, l in negs.items() for a, b in l], columns=['pid', 'y_lo', 'y_hi']).to_parquet(f'{D}/fia_negs.parquet')
if 'md' in V.columns: md = pd.to_datetime(V.md)                                            # the visit date (g02b keeps it since 2026-10-22)
else:                                                                                       # older tables: invert year = year + day-of-year / 366
    yi = np.floor(V.year).astype(int); doy = np.rint((V.year - yi) * 366).astype(int)
    md = pd.to_datetime(yi.astype(str) + '-01-01') + pd.to_timedelta(doy - 1, unit='D')
V['fia_blocked'] = [((int(a), int(b), int(c), int(d)), m) in blocked for a, b, c, d, m in zip(V.st, V.u, V.c, V.p, md)]
PM = fc.execute(f"""select p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, (case when p.MEASMON between 1 and 12 and p.MEASDAY between 1 and 31 then make_date(p.MEASYEAR::int, p.MEASMON::int, p.MEASDAY::int) end) md,
                            (case when q.MEASMON between 1 and 12 and q.MEASDAY between 1 and 31 then make_date(q.MEASYEAR::int, q.MEASMON::int, q.MEASDAY::int) end) pmd   -- (KY 1986-88 periodic plots: MEASDAY 0 -> unknown date = a gap; audit r8 B1)
                     from PLOT p left join PLOT q on q.CN = p.PREV_PLT_CN where p.STATECD in {G2.ST_SQL} and p.MEASYEAR >= 2000""").df()
pmd = {(int(a), int(b), int(c), int(d), pd.Timestamp(m)): (pd.Timestamp(q) if pd.notna(q) else None) for a, b, c, d, m, q in zip(PM.st, PM.u, PM.c, PM.p, PM.md, PM.pmd)}
V['_md'] = md.values; gap = np.zeros(len(V), bool)
for _, g_ in V.sort_values(['pid', '_md']).groupby('pid'):
    ix_, mds_ = g_.index.values, g_._md.values
    for j in range(1, len(ix_)):
        r_ = V.loc[ix_[j]]; prev = pmd.get((int(r_.st), int(r_.u), int(r_.c), int(r_.p), pd.Timestamp(mds_[j])))
        gap[ix_[j]] = prev is None or prev != pd.Timestamp(mds_[j - 1])
V['fia_gap'] = gap
# plots CONVERTED to non-forest before the 2022 forest mask (r13 A-9 / G41): a FIADB visit up to 2022 with no forested condition (and >= 1 non-forest one)
# after a forested visit — the plot's true footprint is likely outside the 2022 mask, so it has no candidate; g03b leaves these plots out of the MIL loss
CV = fc.execute(f"""select p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, p.MEASYEAR my,
                            sum(case when c.COND_STATUS_CD = 1 then 1 else 0 end) nf1, sum(case when c.COND_STATUS_CD = 2 then 1 else 0 end) nf2
                     from PLOT p join COND c on c.PLT_CN = p.CN where p.STATECD in {G2.ST_SQL} and p.MEASYEAR >= 2000 group by all""").df()
conv = set()
for k_, g_ in CV.sort_values('my').groupby(['st', 'u', 'c', 'p']):
    seen_f = False
    for r in g_.itertuples():
        if r.nf1 > 0: seen_f = True
        elif seen_f and r.nf2 > 0 and r.my <= 2022: conv.add(tuple(int(x) for x in k_)); break
V['plot_converted'] = [(int(a), int(b), int(c), int(d)) in conv for a, b, c, d in zip(V.st, V.u, V.c, V.p)]
V = V.drop(columns='_md'); V.to_parquet(f'{D}/visits.parquet')
print(f'plots with typed events {len(evs)} ({sum(map(len, evs.values()))} events); negative periods {sum(map(len, negs.values()))}; '
      f'visits with a hiding code {int(V.fia_blocked.sum())} of {len(V)}; visits whose previous FIA measurement is not the previous table visit {int(V.fia_gap.sum())}; '
      f'plots converted to non-forest before 2022 {len(conv & set(map(tuple, P[key].astype(int).values)))}')
