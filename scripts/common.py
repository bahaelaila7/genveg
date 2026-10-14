import json, numpy as np, rasterio
from rasterio.transform import Affine
ROOT='/workspace/standmap'
def grid():
    g=json.load(open(f'{ROOT}/data/grid/grid.json'))
    return Affine(*g['transform']), g['width'], g['height'], g['crs']
PROFILE=dict(driver='GTiff',tiled=True,blockxsize=512,blockysize=512,compress='ZSTD',zstd_level=9,BIGTIFF='YES')
def aoi_mask():
    return f'{ROOT}/data/grid/aoi_mask.tif'
