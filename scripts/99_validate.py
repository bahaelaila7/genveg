import os as _os; _SM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _WS = _os.environ.get('WORKSPACE', _os.path.dirname(_SM))   # relocatable paths
import sys; sys.path.insert(0,_SM + '/scripts')
import numpy as np, rasterio, os, glob
from rasterio.windows import Window
from common import *
tr,W,H,crs=grid()
tm=rasterio.open(f'{ROOT}/data/treemap/tm_id.tif')
files=sys.argv[1:] or sorted(glob.glob(f'{ROOT}/data/*/*.tif'))
for f in files:
    if 'tiles_' in f or '/grid/' in f or '/cache/' in f: continue
    s=rasterio.open(f)
    ok=(s.transform==tr and s.width==W and s.height==H)
    nv=0; na=0; ntm=0; ntmv=0; mn=None; mx=None
    am=rasterio.open(aoi_mask())
    for r in range(0,H,2048):
        h=min(2048,H-r); win=Window(0,r,W,h)
        m=am.read(1,window=win)==1
        t=tm.read(1,window=win)!=tm.nodata
        a=s.read(1,window=win)
        v=(a!=s.nodata) if s.nodata is not None else np.ones_like(m)
        nv+=(v&m).sum(); na+=m.sum(); ntm+=t.sum(); ntmv+=(v&t).sum()
        x=a[v&m]
        if x.size: mn=x.min() if mn is None else min(mn,x.min()); mx=x.max() if mx is None else max(mx,x.max())
    print(f'{os.path.relpath(f,ROOT)}: grid_match={ok} dtype={s.dtypes[0]} bands={s.count} nodata={s.nodata} size={os.path.getsize(f)/1e6:.0f}MB valid_AOI={100*nv/na:.1f}% valid_on_TMpix={100*ntmv/ntm:.1f}% range=[{mn},{mx}]',flush=True)
