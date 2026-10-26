"""FIA disturbance / treatment records of the training table, for anchoring the dynamics (user, 2026-10-20: FIA's own COND.DSTRB* / COND.TRT* records
anchor FIA data to the disturbances LANDFIRE reports). The same records + rules that built the location prior (g02b.fia_records, shared code):
  fia_events.parquet  pid, kind, year, rmd       typed FIA events (kinds of g02b.EVK) from every forested condition, with the RECORDING visit's date
                                                 (rmd; a record describes the interval ending at that visit — g03b.recs_in, audit r5 I3)
  fia_negs.parquet    pid, y_lo, y_hi            periods without stand-replacing disturbance (FIA protocol; see g02b.fia_records)
  visits.parquet      + fia_blocked              the visit carries a code that could hide stand-replacing evidence (no negative evidence for its period)
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
PM = fc.execute(f"""select p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, make_date(p.MEASYEAR::int, p.MEASMON::int, p.MEASDAY::int) md,
                            make_date(q.MEASYEAR::int, q.MEASMON::int, q.MEASDAY::int) pmd
                     from PLOT p left join PLOT q on q.CN = p.PREV_PLT_CN where p.STATECD in {G2.ST_SQL} and p.MEASYEAR >= 2000""").df()
pmd = {(int(a), int(b), int(c), int(d), pd.Timestamp(m)): (pd.Timestamp(q) if pd.notna(q) else None) for a, b, c, d, m, q in zip(PM.st, PM.u, PM.c, PM.p, PM.md, PM.pmd)}
V['_md'] = md.values; gap = np.zeros(len(V), bool)
for _, g_ in V.sort_values(['pid', '_md']).groupby('pid'):
    ix_, mds_ = g_.index.values, g_._md.values
    for j in range(1, len(ix_)):
        r_ = V.loc[ix_[j]]; prev = pmd.get((int(r_.st), int(r_.u), int(r_.c), int(r_.p), pd.Timestamp(mds_[j])))
        gap[ix_[j]] = prev is None or prev != pd.Timestamp(mds_[j - 1])
V['fia_gap'] = gap; V = V.drop(columns='_md'); V.to_parquet(f'{D}/visits.parquet')
print(f'plots with typed events {len(evs)} ({sum(map(len, evs.values()))} events); negative periods {sum(map(len, negs.values()))}; '
      f'visits with a hiding code {int(V.fia_blocked.sum())} of {len(V)}; visits whose previous FIA measurement is not the previous table visit {int(V.fia_gap.sum())}')
