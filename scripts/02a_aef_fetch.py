"""Fetch only the 20 m overview (level 0) byte span of each AEF COG into a sparse local copy (same offsets)."""
import os as _os; _SM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _WS = _os.environ.get('WORKSPACE', _os.path.dirname(_SM))   # relocatable paths
import sys; sys.path.insert(0,_SM + '/scripts')
import os, io, time, requests, pandas as pd, tifffile
from concurrent.futures import ThreadPoolExecutor
ROOT=_SM
IDX=pd.read_csv(f'{ROOT}/data/aef/files_aoi.csv')
def https(p): return p.replace('s3://us-west-2.opendata.source.coop','https://data.source.coop')
def local(p): return f'{ROOT}/data/aef/cache/'+p.split('/annual/')[1].replace('/','_')
S=requests.Session()
def rng(u,a,b):
    for k in range(6):
        try:
            r=S.get(u,headers={'Range':f'bytes={a}-{b}'},timeout=120)
            if r.status_code in(200,206) and len(r.content)==b-a+1: return r.content
        except Exception: pass
        time.sleep(2)
    raise RuntimeError(f'range fail {u} {a}')
def fetch(p):
    lp=local(p)
    if os.path.exists(lp+'.done'): return p,0
    u=https(p)
    head=rng(u,0,4*1024*1024-1)
    with tifffile.TiffFile(io.BytesIO(head)) as t:
        pg=t.pages[1]; offs=pg.dataoffsets; cnt=pg.databytecounts
    a=min(offs); b=max(o+c for o,c in zip(offs,cnt))
    size=int(S.head(u).headers['content-length'])
    with open(lp,'wb') as f:
        f.truncate(size); f.seek(0); f.write(head)
        CH=8<<20
        parts=[(s,min(s+CH,b)-1) for s in range(a,b,CH)]
        for s,e in parts:
            f.seek(s); f.write(rng(u,s,e))
    open(lp+'.done','w').write(f'{a} {b}')
    return p,b-a
if __name__=='__main__':
    os.makedirs(f'{ROOT}/data/aef/cache',exist_ok=True)
    t0=time.time(); tot=0
    ps=list(IDX.path)
    with ThreadPoolExecutor(6) as ex:
        for i,(p,n) in enumerate(ex.map(fetch,ps)):
            tot+=n; print(i+1,len(ps),f'{tot/1e9:.1f} GB',f'{time.time()-t0:.0f}s',flush=True)
