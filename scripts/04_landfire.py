import os as _os; _SM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _WS = _os.environ.get('WORKSPACE', _os.path.dirname(_SM))   # relocatable paths
import sys; sys.path.insert(0,_SM + '/scripts')
import time, numpy as np, pandas as pd, rasterio, os, shutil
from rasterio.windows import Window
from common import *
t0=time.time()
tr,W,H,crs=grid()
R=f'{ROOT}/data/landfire/raw/LF2022_EVT_CONUS/'
d=pd.read_csv(R+'CSV_Data/LF2022_EVT.csv')
def grp(r):
    p=str(r.EVT_PHYS); lf=str(r.EVT_LF); n=str(r.EVT_NAME)
    if p.startswith('Developed') or lf=='Developed': return 'developed'
    if p=='Agricultural' or lf=='Agriculture': return 'agriculture'
    if p=='Open Water' or lf=='Water': return 'water'
    if lf in('Sparse','Barren') or p.startswith('Quarries') or p=='Sparsely Vegetated': return 'barren'
    if lf=='Tree': return 'forest'
    if lf=='Shrub': return 'shrub'
    if lf=='Herb': return 'herb'
    return 'other'
d['group']=d.apply(grp,axis=1)
d=d.rename(columns={'VALUE':'value'})
d.to_csv(f'{ROOT}/data/landfire/evt_classes.csv',index=False)
print(d.group.value_counts())
with rasterio.open(R+'Tif/LF2022_EVT_CONUS.tif') as s:
    col=round((tr.c-s.transform.c)/30); row=round((s.transform.f-tr.f)/30)
    assert abs((tr.c-s.transform.c)/30-col)<1e-6 and abs((s.transform.f-tr.f)/30-row)<1e-6
    prof=dict(PROFILE,width=W,height=H,count=1,dtype='int16',crs=crs,transform=tr,nodata=32767)
    with rasterio.open(f'{ROOT}/data/landfire/evt2022.tif','w',**prof) as o, rasterio.open(aoi_mask()) as m:
        for r in range(0,H,2048):
            h=min(2048,H-r)
            a=s.read(1,window=Window(col,row+r,W,h),boundless=True,fill_value=32767)
            a[m.read(1,window=Window(0,r,W,h))==0]=32767
            o.write(a,1,window=Window(0,r,W,h))
print('t',time.time()-t0)
