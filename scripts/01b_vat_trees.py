import sys; sys.path.insert(0,'/workspace/standmap/scripts')
import numpy as np, pandas as pd, rasterio
from common import *
from pyogrio import read_dataframe
D='/workspace/FVSclaude/treemap/Data/'
vat=read_dataframe(D+'TreeMap2022_CONUS.tif.vat.dbf',read_geometry=False)
print(vat.shape, list(vat.columns)[:12])
with rasterio.open(f'{ROOT}/data/treemap/tm_id.tif') as s:
    ids=set()
    for _,w in s.block_windows(1):
        ids.update(np.unique(s.read(1,window=w)).tolist())
ids.discard(4294967295)
print('tm_ids in aoi',len(ids))
tc=[c for c in vat.columns if c.lower()=='tm_id'][0]
vat[tc]=vat[tc].astype('int64')
vat=vat[vat[tc].isin(ids)].copy()
print('vat rows kept',len(vat))
for c in vat.columns:
    if c.upper() in('TM_ID','PLT_CN','CN'): vat[c]=vat[c].astype(np.int64).astype(str) if c.upper()=='TM_ID' else vat[c].astype(str)
vat.to_parquet(f'{ROOT}/data/treemap/vat.parquet',index=False)
tr=pd.read_csv(D+'TreeMap2022_CONUS_Tree_Table.csv',dtype={'PLT_CN':str,'CN':str},low_memory=False)
print(tr.shape,list(tr.columns))
tkey=[c for c in tr.columns if c.lower()=='tm_id'][0]
tr=tr[tr[tkey].isin(ids)]
print('tree rows',len(tr), 'plots',tr.PLT_CN.nunique())
tr.to_parquet(f'{ROOT}/data/treemap/trees.parquet',index=False)
open(f'{ROOT}/data/treemap/_ids.txt','w').write('\n'.join(map(str,sorted(ids))))
