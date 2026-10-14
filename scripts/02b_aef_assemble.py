import sys; sys.path.insert(0,'/workspace/standmap/scripts')
import glob, re, time, numpy as np, rasterio
from rasterio.windows import Window
from common import *
tr,W,H,crs=grid()
y=int(sys.argv[1]); t0=time.time()
prof=dict(driver='GTiff',width=W,height=H,count=64,dtype='int8',crs=crs,transform=tr,nodata=-128,tiled=True,blockxsize=512,blockysize=512,
          compress='ZSTD',zstd_level=9,predictor=2,INTERLEAVE='PIXEL',BIGTIFF='YES',SPARSE_OK='TRUE')
fs=sorted(glob.glob(f'{ROOT}/data/aef/tiles_{y}/t_*.tif'))
with rasterio.open(f'{ROOT}/data/aef/aef_{y}.tif','w',**prof) as d:
    for k in range(64): d.set_band_description(k+1,f'A{k:02d}')
    d.update_tags(source='AlphaEarth Foundations annual v1 (tge-labs/aef, source.coop)',year=str(y),dequant='sign(v)*(v/127.5)**2',
                  processing='20m overview -> 30m EPSG:5070 average in float domain, renormalized to unit length, requantized int8')
    for f in fs:
        r,c=map(int,re.findall(r't_(\d+)_(\d+)',f)[0])
        with rasterio.open(f) as s:
            a=s.read()
            if (a==-128).all(): continue
            d.write(a,window=Window(c,r,a.shape[2],a.shape[1]))
print(y,len(fs),'tiles',time.time()-t0)
