import sys; sys.path.insert(0,'/workspace/standmap/scripts')
import glob, time, numpy as np, pandas as pd, geopandas as gpd, rasterio
from pyogrio import read_dataframe
from rasterio.features import rasterize
from multiprocessing import Pool
from common import *
tr,W,H,crs=grid()
ROAD_CLASSES=['S1100','S1200','S1300','S1400','S1630','S1640','S1740']
def rd(f):
    g=read_dataframe(f); g=g[g.geometry.notna()&~g.geometry.is_empty]
    return g.to_crs(5070)
def merge(t):
    fs=sorted(glob.glob(f'{ROOT}/data/tiger/raw/*_{t}.zip'))
    with Pool(5) as p: gs=p.map(rd,fs,chunksize=4)
    g=gpd.GeoDataFrame(pd.concat(gs,ignore_index=True),crs=5070)
    g.to_file(f'{ROOT}/data/tiger/{t}.gpkg',driver='GPKG',engine='pyogrio')
    print(t,len(fs),'files',len(g),'features',list(g.columns),flush=True)
    return g
def save(name,arr):
    prof=dict(PROFILE,width=W,height=H,count=1,dtype='uint8',crs=crs,transform=tr,nodata=255)
    with rasterio.open(f'{ROOT}/data/barriers/{name}.tif','w',**prof) as d, rasterio.open(aoi_mask()) as m:
        for r in range(0,H,2048):
            a=arr[r:r+2048].copy(); a[m.read(1,window=rasterio.windows.Window(0,r,W,a.shape[0]))==0]=255
            d.write(a,1,window=rasterio.windows.Window(0,r,W,a.shape[0]))
if __name__=='__main__':
    t0=time.time()
    r=merge('roads'); print(r.MTFCC.value_counts().to_dict(),flush=True)
    sel=r[r.MTFCC.isin(ROAD_CLASSES)]
    a=rasterize(((g,1) for g in sel.geometry),out_shape=(H,W),transform=tr,fill=0,dtype='uint8',all_touched=True)
    save('roads',a); print('roads raster',a.sum(),time.time()-t0,flush=True); del r,sel
    aw=merge('areawater'); lw=merge('linearwater')
    print(aw.MTFCC.value_counts().to_dict(), lw.MTFCC.value_counts().to_dict())
    a=rasterize(((g,1) for g in aw.geometry),out_shape=(H,W),transform=tr,fill=0,dtype='uint8',all_touched=False)
    b=rasterize(((g,1) for g in lw.geometry),out_shape=(H,W),transform=tr,fill=0,dtype='uint8',all_touched=True)
    save('water',np.maximum(a,b)); print('water',a.sum(),b.sum(),time.time()-t0)
