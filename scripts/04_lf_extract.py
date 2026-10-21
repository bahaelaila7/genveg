import os as _os; _SM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _WS = _os.environ.get('WORKSPACE', _os.path.dirname(_SM))   # relocatable paths
import zipfile
z=zipfile.ZipFile(_WS + '/LF2022_EVT_CONUS.zip')
for n in ['CSV_Data/LF2022_EVT.csv','Tif/LF2022_EVT_CONUS.tif','Tif/LF2022_EVT_CONUS.tif.vat.dbf','Tif/LF2022_EVT_CONUS.tif.aux.xml']:
    z.extract('LF2022_EVT_CONUS/'+n,_SM + '/data/landfire/raw')
print('done')
