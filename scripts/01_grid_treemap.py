import sys; sys.path.insert(0,'/workspace/standmap/scripts')
import json, time, numpy as np, geopandas as gpd, rasterio, pandas as pd
from rasterio.windows import Window, from_bounds
from rasterio.features import rasterize
from common import *
t0=time.time()
TM='/workspace/FVSclaude/treemap/Data/TreeMap2022_CONUS.tif'
c=gpd.read_file('/workspace/TIGER/tl_2024_us_county/tl_2024_us_county.shp')
c=c[c.STATEFP.isin(['13','12'])].to_crs(5070)
geom=c.dissolve().geometry.iloc[0].buffer(1000)
aoi=gpd.GeoDataFrame({'name':['GA_FL']},geometry=[geom],crs=5070)
aoi.to_file(f'{ROOT}/data/grid/aoi_gafl.gpkg',driver='GPKG')
src=rasterio.open(TM)
print(src.profile)
t=src.transform
x0,y0,x1,y1=geom.bounds
col0=int(np.floor((x0-t.c)/t.a)); col1=int(np.ceil((x1-t.c)/t.a))
row0=int(np.floor((y1-t.f)/t.e)); row1=int(np.ceil((y0-t.f)/t.e))
col0=max(col0,0); col1=min(col1,src.width); row0=max(row0,0); row1=min(row1,src.height)
win=Window(col0,row0,col1-col0,row1-row0)
tr=src.window_transform(win)
W,H=win.width,win.height
json.dump(dict(transform=list(tr)[:6],width=W,height=H,crs='EPSG:5070',pixel=30.0,bounds=[tr.c,tr.f+tr.e*H,tr.c+tr.a*W,tr.f]),open(f'{ROOT}/data/grid/grid.json','w'),indent=1)
print('grid',tr,W,H)
# AOI mask: rasterize buffered AOI (used for masking all rasters)
m=rasterize([(geom,1)],out_shape=(H,W),transform=tr,fill=0,dtype='uint8')
prof=dict(PROFILE,width=W,height=H,count=1,dtype='uint8',crs='EPSG:5070',transform=tr,nodata=0)
with rasterio.open(aoi_mask(),'w',**prof) as d: d.write(m,1)
print('aoi pixels',m.sum(), 'area km2',m.sum()*900/1e6)
# crop treemap
nd=src.nodata; print('nodata',nd,src.dtypes)
prof=dict(PROFILE,width=W,height=H,count=1,dtype=src.dtypes[0],crs='EPSG:5070',transform=tr,nodata=nd)
with rasterio.open(f'{ROOT}/data/treemap/tm_id.tif','w',**prof) as d:
    for r in range(0,H,2048):
        h=min(2048,H-r)
        w=Window(col0,row0+r,W,h)
        a=src.read(1,window=w)
        mm=m[r:r+h]
        a[mm==0]=nd
        d.write(a,1,window=Window(0,r,W,h))
print('t',time.time()-t0)
