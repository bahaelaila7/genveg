import os as _os; _SM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _WS = _os.environ.get('WORKSPACE', _os.path.dirname(_SM))   # relocatable paths
import sys; sys.path.insert(0,_SM + '/scripts')
import time, numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window, transform as wtransform, from_bounds
from multiprocessing import Pool
from common import *
tr,W,H,crs=grid()
S=_WS + '/SOLUS/SOLUS100/'
DEP=[(0,5),(5,10),(15,15)]  # layer top, thickness within 0-30 cm
# name: (file pattern, scale)
LAYERED={'clay030':('claytotal_{d}_cm_p',1.0),'sand030':('sandtotal_{d}_cm_p',1.0),'soc030':('soc_{d}_cm_p',1000.0),
         'ph030':('ph1to1h2o_{d}_cm_p',100.0),'awc030':('hydro/derived/awc_{d}_cm',None)}
SINGLE={'restrdepth':'resdept_all_cm_p','lithdepth':'anylithicdpt_cm_p'}
T=4096
def rd(s,win,scale):
    a=s.read(1,window=win,boundless=True).astype('float32')
    nd=s.nodata
    a=np.where(a==nd,np.nan,a) if nd is not None else a
    if scale: a=a/scale
    return a
def work(a):
    r,c=a
    h=min(T,H-r); w=min(T,W-c)
    wt=wtransform(Window(c,r,w,h),tr)
    L,B,R,Tp=wt.c,wt.f+wt.e*h,wt.c+wt.a*w,wt.f
    res={}
    with rasterio.open(S+'claytotal_0_cm_p.tif') as s0:
        win=from_bounds(L-300,B-300,R+300,Tp+300,s0.transform).round_offsets().round_lengths()
        st=s0.window_transform(win)
    with rasterio.open(aoi_mask()) as m: mm=m.read(1,window=Window(c,r,w,h))
    if mm.sum()==0: return r,c,None
    def warp(src):
        d=np.full((h,w),np.nan,np.float32)
        reproject(src,d,src_transform=st,src_crs=crs,dst_transform=wt,dst_crs=crs,resampling=Resampling.bilinear,src_nodata=np.nan,dst_nodata=np.nan)
        d[mm==0]=np.nan
        return d
    for k,(pat,sc) in LAYERED.items():
        acc=0;wsum=0
        for top,th in DEP:
            with rasterio.open(S+pat.format(d=top)+'.tif') as s:
                a=rd(s,win,sc)
            acc=acc+np.nan_to_num(a)*th; wsum=wsum+np.where(np.isnan(a),0,th)
        v=np.where(wsum>0,acc/np.maximum(wsum,1),np.nan).astype('float32')
        res[k]=warp(v)
    for k,f in SINGLE.items():
        with rasterio.open(S+f+'.tif') as s: res[k]=warp(rd(s,win,None))
    return r,c,res
if __name__=='__main__':
    t0=time.time()
    names=list(LAYERED)+list(SINGLE)
    outs={}
    prof=dict(PROFILE,width=W,height=H,count=1,dtype='float32',crs=crs,transform=tr,nodata=-9999.0,predictor=3)
    for n in names: outs[n]=rasterio.open(f'{ROOT}/data/soils/{n}.tif','w',**prof)
    jobs=[(r,c) for r in range(0,H,T) for c in range(0,W,T)]
    with Pool(3) as p:
        for r,c,res in p.imap_unordered(work,jobs):
            if res is None: continue
            for n,v in res.items():
                v=np.where(np.isnan(v),-9999.0,v).astype('float32')
                outs[n].write(v,1,window=Window(c,r,v.shape[1],v.shape[0]))
    for o in outs.values(): o.close()
    print('t',time.time()-t0)
