"""Owner-coalesced parcels (user request 2026-10-23): adjacent parcels of the same owner are one property, so their shared lines are not stand barriers.
Input: <WORKSPACE>/fl_5_co_parcels.geojson (FGDL 2025 parcels of 5 north-Florida counties, with the owner name ONAME).
  owner key = ONAME upper-cased, '&' -> ' AND ', punctuation removed, spaces collapsed; a blank owner never coalesces (each such parcel stays alone).
  An owner's parcels are joined when they touch or nearly touch (gap <= 2 * TOL, digitising slivers): per owner, union of the parcels buffered by TOL,
  split into connected parts; each part = one coalesced parcel, geometry = union(buffer(+TOL)).buffer(-TOL) of its parcels (no internal sliver lines).
Outputs (owner names / addresses are personal data: they are used only as the in-memory grouping key and never written):
  data/parcels/fl5_parcels_owner.gpkg   grp_id, n_parcels, cnty, acres (coalesced parcels, EPSG:5070)
  data/barriers/parcels_owner.tif       their boundaries rasterised on the grid (all_touched; 255 outside the AOI), like data/barriers/parcels.tif (07)
  data/proto/parcels_owner_baker.geojson, data/proto/parcels_all_baker.geojson   Baker County clips (EPSG:4326) for the Forest Explorer layers
  .venv/bin/python -I scripts/07b_parcels_owner.py"""
import os as _os; _SM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _WS = _os.environ.get('WORKSPACE', _os.path.dirname(_SM))   # relocatable paths
import sys; sys.path.insert(0, _SM + '/scripts')
import re, time, numpy as np, pandas as pd, geopandas as gpd, rasterio
from shapely.ops import unary_union
from rasterio.features import rasterize
from rasterio.windows import Window
from common import *
TOL = 1.0                                                                           # m: parcels of one owner closer than 2 m are adjacent
t0 = time.time(); tr, W, H, crs = grid()
g = gpd.read_file(_WS + '/fl_5_co_parcels.geojson', engine='pyogrio', columns=['CNTYNAME', 'ONAME'])
def okey(s):
    if not isinstance(s, str): return None
    s = re.sub(r'\s+', ' ', re.sub(r'[^A-Z0-9 ]', ' ', s.upper().replace('&', ' AND '))).strip()
    return s or None
key = g.ONAME.map(okey); g = g.drop(columns=['ONAME']).to_crs(5070); g['geometry'] = g.geometry.make_valid()
g['k'] = key.values; del key
multi = g.k.notna() & g.duplicated('k', keep=False)
print(f'{len(g)} parcels; blank owner {int(g.k.isna().sum())}; owners with >= 2 parcels {g.k[multi].nunique()} ({int(multi.sum())} parcels)', flush=True)
rows = []
for _, gg in g[multi].groupby('k', sort=False):
    parts = list(getattr(u := unary_union(gg.geometry.buffer(TOL).values), 'geoms', [u]))
    if len(parts) == 1: groups = [gg]
    else:
        pt = gpd.GeoDataFrame(geometry=gg.geometry.representative_point().values, index=gg.index, crs=g.crs)
        pg = gpd.GeoDataFrame({'part': range(len(parts))}, geometry=parts, crs=g.crs)
        j = gpd.sjoin(pt, pg, predicate='within', how='left'); j = j[~j.index.duplicated()]
        groups = [gg.loc[j.index[j.part == q]] for q in range(len(parts)) if (j.part == q).any()]
    for grp in groups:
        geom = grp.geometry.values[0] if len(grp) == 1 else unary_union(grp.geometry.buffer(TOL).values).buffer(-TOL)
        rows.append(dict(n_parcels=len(grp), cnty=grp.CNTYNAME.iloc[0], geometry=geom))
single = g[~multi]
P = pd.concat([gpd.GeoDataFrame(rows, crs=g.crs), gpd.GeoDataFrame(dict(n_parcels=1, cnty=single.CNTYNAME.values), geometry=single.geometry.values, crs=g.crs)], ignore_index=True)
P = P[~P.geometry.is_empty & P.geometry.notna()].reset_index(drop=True)
P['grp_id'] = np.arange(1, len(P) + 1); P['acres'] = (P.geometry.area / 4046.86).round(2)
P = P[['grp_id', 'n_parcels', 'cnty', 'acres', 'geometry']]
P.to_file(f'{ROOT}/data/parcels/fl5_parcels_owner.gpkg', driver='GPKG', engine='pyogrio')
print(f'coalesced: {len(g)} parcels -> {len(P)} owner parcels; {int((P.n_parcels > 1).sum())} of them join 2+ parcels (max {int(P.n_parcels.max())}) ({time.time()-t0:.0f}s)', flush=True)
lines = P.geometry.boundary
a = rasterize(((l, 1) for l in lines if l is not None and not l.is_empty), out_shape=(H, W), transform=tr, fill=0, dtype='uint8', all_touched=True)
prof = dict(PROFILE, width=W, height=H, count=1, dtype='uint8', crs=crs, transform=tr, nodata=255)
with rasterio.open(f'{ROOT}/data/barriers/parcels_owner.tif', 'w', **prof) as d, rasterio.open(aoi_mask()) as m:
    for r in range(0, H, 2048):
        x = a[r:r + 2048].copy(); x[m.read(1, window=Window(0, r, W, x.shape[0])) == 0] = 255
        d.write(x, 1, window=Window(0, r, W, x.shape[0]))
with rasterio.open(f'{ROOT}/data/barriers/parcels.tif') as s, rasterio.open(f'{ROOT}/data/barriers/parcels_owner.tif') as o: b_all, b_own = int((s.read(1) == 1).sum()), int((o.read(1) == 1).sum())
print(f'boundary px inside the AOI: owner-coalesced {b_own} vs all parcels {b_all} ({100 * (1 - b_own / b_all):.1f} % fewer)', flush=True)
# Baker County clips for Forest Explorer (no personal data: ids, counts, areas only)
area = gpd.read_file(f'{ROOT}/data/derived/baker_county.gpkg').to_crs(5070)
os.makedirs(f'{ROOT}/data/proto', exist_ok=True)
for name, df in (('parcels_owner_baker', P), ('parcels_all_baker', gpd.GeoDataFrame(dict(grp_id=np.arange(1, len(g) + 1), n_parcels=1, cnty=g.CNTYNAME.values,
                                                                                         acres=(g.geometry.area / 4046.86).round(2).values), geometry=g.geometry.values, crs=g.crs))):
    c = gpd.clip(df, area); c = c[~c.geometry.is_empty]
    c.assign(geometry=c.geometry.simplify(2.0)).to_crs(4326).to_file(f'{ROOT}/data/proto/{name}.geojson', driver='GeoJSON', engine='pyogrio')
    print(f'{name}: {len(c)} polygons', flush=True)
print('done', round(time.time() - t0))
