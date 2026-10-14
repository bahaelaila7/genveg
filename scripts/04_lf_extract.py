import zipfile
z=zipfile.ZipFile('/workspace/LF2022_EVT_CONUS.zip')
for n in ['CSV_Data/LF2022_EVT.csv','Tif/LF2022_EVT_CONUS.tif','Tif/LF2022_EVT_CONUS.tif.vat.dbf','Tif/LF2022_EVT_CONUS.tif.aux.xml']:
    z.extract('LF2022_EVT_CONUS/'+n,'/workspace/standmap/data/landfire/raw')
print('done')
