"""Region 4 grids (docs/GENVEG_V4_R4.md §B): the 8 EPA Region 4 states' counties (TIGER 2024) dissolved and buffered 2 km (1.6 km FIA fuzz disk + margin),
snapped outward to the TreeMap 30 m lattice of data/grid/grid.json (EPSG:5070). Writes data/grid/grid_r4.json (30 m), grid_r4_90.json (90 m, same origin,
a 90 m pixel = 3x3 30 m pixels), aoi_r4.gpkg and aoi_r4_mask90.tif (1 inside the AOI).
  .venv/bin/python -I scripts/r4_grid.py"""
import os as _os; _SM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _WS = _os.environ.get('WORKSPACE', _os.path.dirname(_SM))   # relocatable paths
import sys; sys.path.insert(0, _SM + '/scripts')
import json, math, numpy as np, geopandas as gpd, rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from common import *
STATES = ['01', '12', '13', '21', '28', '37', '45', '47']                    # AL FL GA KY MS NC SC TN
g = json.load(open(f'{ROOT}/data/grid/grid.json')); x0, y0 = g['transform'][2], g['transform'][5]   # TreeMap lattice origin
c = gpd.read_file(_WS + '/TIGER/tl_2024_us_county/tl_2024_us_county.shp')
c = c[c.STATEFP.isin(STATES)].to_crs(5070)
aoi = c.dissolve().geometry.iloc[0].buffer(2000)
gpd.GeoDataFrame({'name': ['EPA_R4']}, geometry=[aoi], crs=5070).to_file(f'{ROOT}/data/grid/aoi_r4.gpkg', driver='GPKG')
l, b, r, t = aoi.bounds
L = x0 + math.floor((l - x0) / 90) * 90; T = y0 - math.floor((y0 - t) / 90) * 90          # origin on the lattice, a multiple of 90 m from the TreeMap origin
W90 = math.ceil((r - L) / 90); H90 = math.ceil((T - b) / 90)
for name, px, w, h in (('grid_r4.json', 30, 3 * W90, 3 * H90), ('grid_r4_90.json', 90, W90, H90)):
    json.dump(dict(transform=[px, 0.0, L, 0.0, -px, T], width=w, height=h, crs='EPSG:5070', pixel=px, bounds=[L, T - px * h, L + px * w, T]),
              open(f'{ROOT}/data/grid/{name}', 'w'), indent=1)
tr90 = Affine(90, 0, L, 0, -90, T)
m = rasterize([(aoi, 1)], out_shape=(H90, W90), transform=tr90, fill=0, dtype='uint8')
with rasterio.open(f'{ROOT}/data/grid/aoi_r4_mask90.tif', 'w', **dict(PROFILE, width=W90, height=H90, count=1, dtype='uint8', crs='EPSG:5070', transform=tr90)) as d:
    d.write(m, 1)
print(f'R4 AOI {aoi.area / 1e6:.0f} km2; 30 m grid {3 * W90} x {3 * H90}; 90 m grid {W90} x {H90} ({m.sum() / 1e6:.1f} M AOI pixels at 90 m); '
      f'origin offset from TreeMap ({(L - x0) / 30:.0f}, {(y0 - T) / 30:.0f}) px')
