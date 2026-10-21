import os as _os; _SM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _WS = _os.environ.get('WORKSPACE', _os.path.dirname(_SM))   # relocatable paths
import sys; sys.path.insert(0,_SM + '/scripts')
import glob, time, numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window, transform as wtransform
from multiprocessing import Pool
from common import *
tr,W,H,crs=grid()
T=4096
PRODS={'lossyear':Resampling.mode,'treecover2000':Resampling.average,'datamask':Resampling.nearest}
def work(a):
    prod,r,c=a
    h=min(T,H-r); w=min(T,W-c)
    wt=wtransform(Window(c,r,w,h),tr)
    out=np.full((h,w),255,np.uint8)
    for f in sorted(glob.glob(f'{ROOT}/data/hansen/raw/Hansen_GFC-2024-v1.12_{prod}_*.tif')):
        with rasterio.open(f) as s:
            reproject(rasterio.band(s,1),out,dst_transform=wt,dst_crs='EPSG:5070',resampling=PRODS[prod],init_dest_nodata=False,dst_nodata=255,src_nodata=None)
    with rasterio.open(aoi_mask()) as m: mm=m.read(1,window=Window(c,r,w,h))
    out[mm==0]=255
    return prod,r,c,out
if __name__=='__main__':
    t0=time.time()
    for prod in PRODS:
        prof=dict(PROFILE,width=W,height=H,count=1,dtype='uint8',crs=crs,transform=tr,nodata=255)
        jobs=[(prod,r,c) for r in range(0,H,T) for c in range(0,W,T)]
        with rasterio.open(f'{ROOT}/data/hansen/{prod}.tif','w',**prof) as d, Pool(6) as p:
            for pr,r,c,o in p.imap_unordered(work,jobs):
                d.write(o,1,window=Window(c,r,o.shape[1],o.shape[0]))
        print(prod,time.time()-t0,flush=True)
