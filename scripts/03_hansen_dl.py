import sys; sys.path.insert(0,'/workspace/standmap/scripts')
import os, requests, math
from common import *
g=json.load(open(f'{ROOT}/data/grid/grid.json')) if False else None
import json
from pyproj import Transformer
g=json.load(open(f'{ROOT}/data/grid/grid.json')); x0,y0,x1,y1=g['bounds']
t=Transformer.from_crs(5070,4326,always_xy=True)
xs=[];ys=[]
for x in (x0,(x0+x1)/2,x1):
    for y in (y0,(y0+y1)/2,y1):
        lo,la=t.transform(x,y); xs.append(lo); ys.append(la)
lon0,lon1,lat0,lat1=min(xs),max(xs),min(ys),max(ys); print(lon0,lon1,lat0,lat1)
os.makedirs(f'{ROOT}/data/hansen/raw',exist_ok=True)
tiles=[]
for nw_lat in range(math.floor(lat0/10)*10+10, math.floor(lat1/10)*10+11,10):
    for w in range(math.floor(lon0/10)*10, math.floor(lon1/10)*10+1,10):
        tiles.append((nw_lat,w))
print(tiles)
for la,lo in tiles:
    nm=f'{la:02d}N_{abs(lo):03d}{"W" if lo<0 else "E"}'
    for p in('lossyear','treecover2000','datamask'):
        n=f'Hansen_GFC-2024-v1.12_{p}_{nm}.tif'; d=f'{ROOT}/data/hansen/raw/{n}'
        if os.path.exists(d): continue
        r=requests.get(f'https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/{n}',stream=True)
        print(n,r.status_code)
        if r.status_code!=200: continue
        with open(d+'.part','wb') as f:
            for ch in r.iter_content(1<<22): f.write(ch)
        os.rename(d+'.part',d)
