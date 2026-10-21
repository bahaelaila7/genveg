import os as _os; _SM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _WS = _os.environ.get('WORKSPACE', _os.path.dirname(_SM))   # relocatable paths
import sys; sys.path.insert(0,_SM + '/scripts')
import os, requests, geopandas as gpd, time
from concurrent.futures import ThreadPoolExecutor
from common import *
c=gpd.read_file(_WS + '/TIGER/tl_2024_us_county/tl_2024_us_county.shp')
fips=sorted((c[c.STATEFP.isin(['13','12'])].GEOID).tolist()); print(len(fips))
os.makedirs(f'{ROOT}/data/tiger/raw',exist_ok=True)
jobs=[(t,f) for f in fips for t in('ROADS','AREAWATER','LINEARWATER')]
def get(j):
    t,f=j; n=f'tl_2024_{f}_{t.lower()}.zip'; p=f'{ROOT}/data/tiger/raw/{n}'
    if os.path.exists(p) and open(p,'rb').read(2)==b'PK': return j,'ok'
    for a in range(4):
        try:
            r=requests.get(f'https://www2.census.gov/geo/tiger/TIGER2024/{t}/{n}',timeout=120)
            if r.status_code==200 and r.content[:2]==b'PK':
                open(p+'.part','wb').write(r.content); os.rename(p+'.part',p); return j,'ok'
            if r.status_code==404: return j,'404'
        except Exception as e: err=str(e)
        time.sleep(5)
    return j,'fail'
with ThreadPoolExecutor(2) as ex: res=list(ex.map(get,jobs))
bad=[r for r in res if r[1]!='ok']; print('failed',bad)
