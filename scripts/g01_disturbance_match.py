"""Generative-vegetation P0 viability: can FIA-reported disturbances localise fuzzed plots via Hansen loss year?
Harvested: GA+FL forested single-condition plots (CONDPROP >= 0.9) with TRTCD1 = 10 (cutting) dated 2002+, stand age at the visit consistent with a
reset (STDAGE <= MEASYEAR - TRTYR1 + 2). Controls: no TRT / DSTRB codes, STDAGE >= 30, measured 2010+.
Per plot, within the public-coordinate fuzz disk (r = 800 m and 1600 m): matching area = pixels with Hansen loss in [year-1, year+1];
'match' = a connected matching patch >= 1 ha; also the public point itself (3x3 window). For controls the year is drawn from the harvest years.
Output: data/derived/g01_match.csv + printed summary."""
import os as _os; _SM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _WS = _os.environ.get('WORKSPACE', _os.path.dirname(_SM))   # relocatable paths
import sys, os, numpy as np, pandas as pd, duckdb, rasterio
from rasterio.windows import Window
from pyproj import Transformer
from scipy import ndimage
ROOT = _SM; rng = np.random.default_rng(1)
c = duckdb.connect(_WS + '/FIA/FIASQLITE2PGSQL/FIADB.duckdb', read_only=True)
q = """select p.CN, p.STATECD, p.LAT, p.LON, p.MEASYEAR, c.STDAGE, c.STDORGCD, c.TRTCD1, c.TRTYR1, c.DSTRBCD1, c.TRTCD2, c.DSTRBCD2, c.CONDPROP_UNADJ
       from COND c join PLOT p on c.PLT_CN = p.CN
       where p.STATECD in (12, 13) and c.COND_STATUS_CD = 1 and c.CONDPROP_UNADJ >= 0.9 and p.MEASYEAR >= 2005 order by p.CN"""
d = c.execute(q).df()
harv = d[(d.TRTCD1 == 10) & (d.TRTYR1 >= 2002) & (d.TRTYR1 <= 2024) & (d.STDAGE <= d.MEASYEAR - d.TRTYR1 + 2)].copy(); harv['grp'] = 'harvest'
ctrl = d[(d.TRTCD1.fillna(0) == 0) & (d.DSTRBCD1.fillna(0) == 0) & (d.STDAGE >= 30) & (d.MEASYEAR >= 2010)].copy(); ctrl['grp'] = 'control'
harv = harv.drop_duplicates('LAT'); ctrl = ctrl.drop_duplicates('LAT')
harv = harv.sample(min(1500, len(harv)), random_state=1); ctrl = ctrl.sample(min(1500, len(ctrl)), random_state=2)
ctrl['TRTYR1'] = rng.choice(harv.TRTYR1.values, len(ctrl))
P = pd.concat([harv, ctrl]); print('plots', P.grp.value_counts().to_dict(), flush=True)
src = rasterio.open(f'{ROOT}/data/hansen/lossyear.tif'); tc = rasterio.open(f'{ROOT}/data/hansen/treecover2000.tif')
tf = Transformer.from_crs(4326, src.crs, always_xy=True)
rows = []
for r in P.itertuples():
    x, y = tf.transform(r.LON, r.LAT); rr, cc = src.index(x, y); R = 54
    w = Window(cc - R, rr - R, 2 * R + 1, 2 * R + 1)
    if rr - R < 0 or cc - R < 0 or rr + R >= src.height or cc + R >= src.width: continue
    ly = src.read(1, window=w).astype(int); t0 = tc.read(1, window=w)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]; dist = np.hypot(yy, xx) * 30
    yr = int(r.TRTYR1) - 2000; m = (ly >= yr - 1) & (ly <= yr + 1) & (ly > 0)
    lab, n = ndimage.label(m); sizes = np.bincount(lab.ravel())[1:] if n else np.array([])
    big = np.isin(lab, np.flatnonzero(sizes >= 11) + 1) if n else np.zeros_like(m)
    out = dict(grp=r.grp, cn=r.CN, year=r.TRTYR1, center=bool(m[R - 1:R + 2, R - 1:R + 2].any()))
    for rad in (800, 1600):
        disk = dist <= rad; forest = disk & (t0 >= 25)
        out[f'match{rad}'] = bool((big & disk).any()); out[f'ha{rad}'] = float((m & disk).sum() * 0.09)
        out[f'frac{rad}'] = float((m & disk).sum() / max(forest.sum(), 1))      # share of the disk's former-forest area that matches
        # controls: candidate area consistent with "no stand-replacing loss since (MEASYEAR - STDAGE)"
        since = int(r.MEASYEAR - r.STDAGE) - 2000
        out[f'consist{rad}'] = float(((ly == 0) | (ly < since))[forest].mean()) if forest.any() else np.nan
    rows.append(out)
R_ = pd.DataFrame(rows); R_.to_csv(f'{ROOT}/data/derived/g01_match.csv', index=False)
for g, x in R_.groupby('grp'):
    print(f'== {g} (n {len(x)}): public point matches {x.center.mean():.2f} | '
          + ' | '.join(f'r{rad}: P(match patch) {x[f"match{rad}"].mean():.2f}, matching area median {x[f"ha{rad}"].median():.1f} ha '
                       f'(= {100*x[f"frac{rad}"].median():.1f} % of the disk forest), age-consistent share {x[f"consist{rad}"].median():.2f}' for rad in (800, 1600)))
h = R_[R_.grp == 'harvest']; k = R_[R_.grp == 'control']
for rad in (800, 1600):
    lr = h[f'match{rad}'].mean() / max(k[f'match{rad}'].mean(), 1e-9)
    print(f'r{rad}: likelihood ratio P(match | harvested) / P(match | control) = {lr:.2f}; candidate-area reduction for matched harvested plots: '
          f'median {1/np.median(h.loc[h[f"match{rad}"], f"frac{rad}"]):.0f}x')
