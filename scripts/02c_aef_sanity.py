import sys; sys.path.insert(0,'/workspace/standmap/scripts')
import numpy as np, rasterio, json
from rasterio.windows import Window, from_bounds
from pyproj import Transformer
from common import *
tr,W,H,crs=grid()
a22=rasterio.open(f'{ROOT}/data/aef/aef_2022.tif'); a25=rasterio.open(f'{ROOT}/data/aef/aef_2025.tif')
ly=rasterio.open(f'{ROOT}/data/hansen/lossyear.tif'); tc=rasterio.open(f'{ROOT}/data/hansen/treecover2000.tif'); tm=rasterio.open(f'{ROOT}/data/treemap/tm_id.tif')
def dq(a): v=a.astype('float32'); return np.sign(v)*(v/127.5)**2
def cos(win):
    x=a22.read(window=win); y=a25.read(window=win)
    ok=(x[0]!=-128)&(y[0]!=-128)
    X=dq(x); Y=dq(y); c=(X*Y).sum(0)
    return c,ok
t=Transformer.from_crs(4326,5070,always_xy=True)
def box(lo0,la0,lo1,la1):
    x0,y0=t.transform(lo0,la0); x1,y1=t.transform(lo1,la1)
    return from_bounds(min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1),tr).round_offsets().round_lengths()
out={}
def stats(name,v):
    if v.size==0: print(name,'no pixels'); return
    print(f'{name}: n={v.size} mean={v.mean():.3f} median={np.median(v):.3f} p10={np.percentile(v,10):.3f} p90={np.percentile(v,90):.3f}')
for name,b in {'Okefenokee NWR':(-82.45,30.6,-82.2,30.85),'Apalachicola NF':(-85.0,30.1,-84.6,30.4),'Osceola NF':(-82.6,30.2,-82.3,30.45),'Chattahoochee NF (N GA)':(-84.1,34.6,-83.7,34.9)}.items():
    win=box(*b); c,ok=cos(win)
    L=ly.read(1,window=win); T=tm.read(1,window=win)!=tm.nodata
    m=ok&T&(L==0)
    stats(f'{name} (forest, no Hansen loss)',c[m])
# whole-AOI strata by Hansen loss year, forest pixels (TreeMap valid) – sampled by row stripes
import collections
acc=collections.defaultdict(list)
rng=np.random.default_rng(0)
for r in range(0,H,1024):
    win=Window(0,r,W,min(1024,H-r))
    L=ly.read(1,window=win); T=tm.read(1,window=win)!=tm.nodata; C=tc.read(1,window=win)
    if not (T|((L>0)&(L<255))).any(): continue
    c,ok=cos(win)
    for k,m in {'no_loss_forest':(L==0)&T&ok,'loss_2023-24':((L==23)|(L==24))&(C>=50)&ok,'loss_2018-21':((L>=18)&(L<=21))&(C>=50)&ok,'loss_2001-10':((L>=1)&(L<=10))&(C>=50)&ok}.items():
        v=c[m]
        if v.size>20000: v=rng.choice(v,20000,replace=False)
        acc[k].append(v)
for k,v in acc.items(): stats('AOI '+k,np.concatenate(v))
