import json, numpy as np, rasterio
from rasterio.transform import Affine
import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # the standmap checkout (relocatable)
WS=os.environ.get('WORKSPACE', os.path.dirname(ROOT))                      # sibling projects (FIA databases)
# the FIA databases genveg reads (read-only), relative to WORKSPACE unless absolute; the pipeline sets them from the config `inputs` and fingerprints
# them into the stage hashes (audit r5 I4: a rebuilt curated table must not pass as current)
def curated_db(): return os.path.join(WS, os.environ.get('GV_CURATED_DB', 'landis_region4/data_r4/slim_r4.duckdb'))
def fiadb(): return os.path.join(WS, os.environ.get('GV_FIADB', 'FIA/FIASQLITE2PGSQL/FIADB.duckdb'))
def grid():
    g=json.load(open(f'{ROOT}/data/grid/grid.json'))
    return Affine(*g['transform']), g['width'], g['height'], g['crs']
PROFILE=dict(driver='GTiff',tiled=True,blockxsize=512,blockysize=512,compress='ZSTD',zstd_level=9,BIGTIFF='YES')
def aoi_mask():
    return f'{ROOT}/data/grid/aoi_mask.tif'
