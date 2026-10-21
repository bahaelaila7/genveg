"""AlphaEarth annual embeddings -> 30 m EPSG:5070 TreeMap grid (64-band int8, -128 nodata).
Reads the 20 m overview (level 0) of the raw bottom-up COGs via /vsicurl/ (the provided VRTs are warped VRTs without
overviews, so we open the .tiff directly and flip rows ourselves). Averaging is done in de-quantized float domain,
then renormalized to unit length and re-quantized. Per-tile outputs in data/aef/tiles_<year>/ (resumable)."""
import os as _os; _SM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _WS = _os.environ.get('WORKSPACE', _os.path.dirname(_SM))   # relocatable paths
import sys; sys.path.insert(0,_SM + '/scripts')
import os, time, numpy as np, pandas as pd, rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window, transform as wtransform
from rasterio.transform import Affine
from multiprocessing import Pool
from common import *
tr,W,H,crs=grid()
T=1024
IDX=pd.read_csv(f'{ROOT}/data/aef/files_aoi.csv')
ENV=dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tiff',GDAL_HTTP_MULTIPLEX='YES',
         GDAL_HTTP_VERSION='2',GDAL_CACHEMAX=512,CPL_VSIL_CURL_CACHE_SIZE=200000000,GDAL_HTTP_MAX_RETRY=5,GDAL_HTTP_RETRY_DELAY=3,
         GDAL_NUM_THREADS=1)
def url(p): return f'{ROOT}/data/aef/cache/'+p.split('/annual/')[1].replace('/','_')  # local sparse copy from 02a_aef_fetch.py
def tdir(y): return f'{ROOT}/data/aef/tiles_{y}'
def tile_path(y,r,c): return f'{tdir(y)}/t_{r:05d}_{c:05d}.tif'
def dq(a):
    v=a.astype(np.float32); out=np.sign(v)*(v/127.5)**2; out[a==-128]=np.nan; return out
def work(job):
    y,r,c=job
    out=tile_path(y,r,c)
    if os.path.exists(out): return job,'skip',0
    t0=time.time()
    h=min(T,H-r); w=min(T,W-c)
    win=Window(c,r,w,h); wt=wtransform(win,tr)
    with rasterio.open(aoi_mask()) as m: mm=m.read(1,window=win)
    if mm.sum()==0: open(out+'.empty','w').close(); return job,'empty',0
    bx=(wt.c,wt.f+wt.e*h,wt.c+wt.a*w,wt.f)
    dst=np.full((64,h,w),np.nan,np.float32)
    nsrc=0
    with rasterio.Env(**ENV):
        for _,f in IDX[IDX.year==y].iterrows():
            l,b,rr,t=transform_bounds('EPSG:5070',f.crs,*bx,densify_pts=21)
            if rr<f.utm_west or l>f.utm_east or t<f.utm_south or b>f.utm_north: continue
            m=3
            c0=max(int(np.floor((l-f.utm_west)/20))-m,0); c1=min(int(np.ceil((rr-f.utm_west)/20))+m,4096)
            r0=max(int(np.floor((b-f.utm_south)/20))-m,0); r1=min(int(np.ceil((t-f.utm_south)/20))+m,4096)
            if c1<=c0 or r1<=r0: continue
            with rasterio.open(url(f.path),overview_level=0) as s:
                a=s.read(window=Window(c0,r0,c1-c0,r1-r0))
            if (a[0]==-128).all(): continue
            a=a[:,::-1]  # bottom-up -> north-up
            st=Affine(20,0,f.utm_west+c0*20,0,-20,f.utm_south+r1*20)
            for k in range(64):
                reproject(dq(a[k]),dst[k],src_transform=st,src_crs=f.crs,dst_transform=wt,dst_crs='EPSG:5070',
                          resampling=Resampling.average,src_nodata=np.nan,dst_nodata=np.nan,init_dest_nodata=False)
            nsrc+=1
    bad=np.isnan(dst[0])|(mm==0)
    dst[:,bad]=0
    n=np.sqrt((dst**2).sum(0)); bad|=(n==0); n[n==0]=1
    dst/=n
    q=(np.sign(dst)*np.rint(127.5*np.sqrt(np.abs(dst)))).clip(-127,127).astype(np.int8)
    q[:,bad]=-128
    prof=dict(driver='GTiff',width=w,height=h,count=64,dtype='int8',crs=crs,transform=wt,nodata=-128,tiled=True,
              blockxsize=512,blockysize=512,compress='ZSTD',zstd_level=3,INTERLEAVE='PIXEL',BIGTIFF='YES')
    with rasterio.open(out+'.part','w',**prof) as d: d.write(q)
    os.rename(out+'.part',out)
    return job,f'ok n={nsrc}',time.time()-t0
if __name__=='__main__':
    years=[int(x) for x in sys.argv[1].split(',')]
    lim=int(sys.argv[2]) if len(sys.argv)>2 else None
    for y in years: os.makedirs(tdir(y),exist_ok=True)
    jobs=[(y,r,c) for y in years for r in range(0,H,T) for c in range(0,W,T)]
    if lim: jobs=jobs[:lim]
    t0=time.time(); n=0
    with Pool(6) as p:
        for job,st,dt in p.imap_unordered(work,jobs):
            n+=1
            if st.startswith('ok') or n%50==0: print(n,len(jobs),job,st,f'{dt:.0f}s',f'elapsed {time.time()-t0:.0f}',flush=True)
