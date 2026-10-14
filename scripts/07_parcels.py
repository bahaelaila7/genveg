import sys; sys.path.insert(0,'/workspace/standmap/scripts')
import time, numpy as np, geopandas as gpd, rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from common import *
t0=time.time()
tr,W,H,crs=grid()
g=gpd.read_file('/workspace/fl_5_co_parcels.geojson',engine='pyogrio')
print(len(g),g.crs); print(g.CNTYNAME.value_counts())
PII=['ONAME','OADDR1','OADDR2','OCITY','OSTATE','OZIPCD','PHYADDR1','PHYCITY','PHYZIP','Owner','GOOGLEMAP']
g=g.drop(columns=[c for c in PII if c in g.columns]).to_crs(5070)
g['geometry']=g.geometry.make_valid()
g.to_file(f'{ROOT}/data/parcels/fl5_parcels.gpkg',driver='GPKG',engine='pyogrio')
print('bounds per county'); print(g.assign(x=g.geometry.centroid.x,y=g.geometry.centroid.y).groupby('CNTYNAME')[['x','y']].agg(['min','max']))
print('total area km2',g.geometry.area.sum()/1e6)
lines=g.geometry.boundary
a=rasterize(((l,1) for l in lines if l is not None and not l.is_empty),out_shape=(H,W),transform=tr,fill=0,dtype='uint8',all_touched=True)
prof=dict(PROFILE,width=W,height=H,count=1,dtype='uint8',crs=crs,transform=tr,nodata=255)
with rasterio.open(f'{ROOT}/data/barriers/parcels.tif','w',**prof) as d, rasterio.open(aoi_mask()) as m:
    for r in range(0,H,2048):
        x=a[r:r+2048].copy(); x[m.read(1,window=Window(0,r,W,x.shape[0]))==0]=255
        d.write(x,1,window=Window(0,r,W,x.shape[0]))
print('boundary px',a.sum(),time.time()-t0)
