"""One-off migration (2026-10-22) of a visits.parquet written before g02b kept the visit date: recover the date from the old encoding
year = year + day-of-year / 366 (exact inverse; a leap-year Dec-31 visit was stored as the next integer year, doy 0 -> Dec 31 of the previous year),
store it as md, and re-encode year with g02b.decimal_year. Row order and vid are unchanged. Idempotent (skips a table that already has md)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g02b_build_gafl as G2
from features import ROOT
fn = f"{ROOT}/data/proto/{os.environ.get('DATA', 'gafl')}/visits.parquet"; V = pd.read_parquet(fn)
if 'md' in V.columns: print('visits.parquet already carries the visit date; nothing to do'); sys.exit(0)
yi = np.floor(V.year).astype(int); doy = np.rint((V.year - yi) * 366).astype(int)
md = pd.to_datetime(yi.astype(str) + '-01-01') + pd.to_timedelta(doy - 1, unit='D')
assert np.allclose(md.dt.year + md.dt.dayofyear / 366, V.year), 'old encoding not inverted exactly'
moved = int((np.floor(V.year) != md.dt.year).sum()); V['md'] = md; V['year'] = G2.decimal_year(md).values
V.to_parquet(fn + '.tmp'); os.replace(fn + '.tmp', fn)
print(f'visits: date stored, year re-encoded; visits that had been placed in the next year: {moved}')
