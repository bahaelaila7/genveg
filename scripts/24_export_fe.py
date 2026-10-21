"""Export stands to a Forest Explorer prototype layer: stands of out/stands_<name>.gpkg clipped to an area polygon, simplified (5 m, topology kept
by simplifying shared boundaries once via a coverage), EPSG:4326, rounded coordinates, attributes for the popup.
Usage: python scripts/24_export_fe.py <area polygon gpkg> <out geojson> [stands gpkg, default data/derived/runs/<config name>/finalize/stands.gpkg]"""
import sys, os, json, numpy as np, geopandas as gpd, shapely
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import CFG, ROOT
area = gpd.read_file(sys.argv[1]); out = sys.argv[2]
st = gpd.read_file(sys.argv[3] if len(sys.argv) > 3 else f'{ROOT}/data/derived/runs/{CFG["name"]}/finalize/stands.gpkg', layer='stands')
area = area.to_crs(st.crs); poly = area.union_all()
st = st[st.intersects(poly)].copy(); st['geometry'] = st.intersection(poly)
st = st[~st.is_empty & st.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
st['area_ha'] = st.area / 1e4
st = st[st.area_ha >= 0.09]
try:
    cov = shapely.coverage_simplify(st.geometry.values, 5.0)        # shared boundaries simplified once (no gaps / overlaps)
    st['geometry'] = cov
except Exception as e:
    print('coverage_simplify unavailable, per-polygon simplify:', e); st['geometry'] = st.simplify(5.0, preserve_topology=True)
keep = ['stand_id', 'status', 'area_ha', 'loss_year', 'BALIVE', 'STANDHT', 'QMD', 'TPA_LIVE', 'CANOPYPCT', 'SOFTWOOD_FRAC', 'tm_frac', 'tm_main', 'tm_share', 'tm_n',
        'awc030', 'clay030', 'sand030', 'lon', 'lat']
st = st[[c for c in keep if c in st.columns] + ['geometry']].to_crs(4326)
for c in st.columns:
    if c != 'geometry' and st[c].dtype.kind == 'f': st[c] = st[c].round(3)
st['stand_id'] = st.stand_id.astype(int); st['status'] = st.status.astype(int)
gj = json.loads(st.to_json(drop_id=True))
def rnd(c): return [rnd(x) for x in c] if isinstance(c[0], list) else [round(c[0], 6), round(c[1], 6)]
for f in gj['features']: f['geometry']['coordinates'] = rnd(f['geometry']['coordinates'])
json.dump(gj, open(out, 'w'), separators=(',', ':'))
a = st.area_ha.values
print(f'wrote {out}: {len(st)} stands, {a.sum():.0f} ha; area ha 5/25/50/75/95 %: {np.quantile(a, [.05, .25, .5, .75, .95]).round(1)}; status {st.status.value_counts().to_dict()}')
