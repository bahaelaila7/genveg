"""Per-pixel feature stack for segmentation (shared by fit + superpixel stages).
Groups (each scaled to total variance = weight^2):
  aef     : AlphaEarth 2025 PCs (basis fitted on forest-land pixels)
  aef_old : AlphaEarth 2022 projected on the same basis (history / pre-harvest structure)
  change  : 1 - cos(AEF2022, AEF2025)  (stand-replacing change since 2022)
  treemap : TreeMap 2022 stand attributes via the plot VAT (zeroed where the pixel is not TreeMap forest or is harvested)
  soil    : PCs of the soil rasters
"""
import os, numpy as np, pandas as pd, rasterio, yaml
from rasterio.windows import Window
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # the standmap checkout (relocatable)
WS = os.environ.get('WORKSPACE', os.path.dirname(ROOT))                    # sibling projects (FIA databases)
CFG = yaml.safe_load(open(f'{ROOT}/config/pilot.yml'))
SOILS = ['clay030', 'sand030', 'soc030', 'awc030', 'ph030', 'restrdepth']
TM_ATTRS = ['BALIVE', 'STANDHT', 'QMD', 'CANOPYPCT', 'TPA_LIVE', 'SOFTWOOD_FRAC']
TM_LOG = {'BALIVE', 'QMD', 'TPA_LIVE'}

def dequant(q):
    """AEF int8 -> float (sign(v) (v/127.5)^2); -128 = nodata -> nan."""
    q = q.astype(np.float32); bad = q == -128
    x = np.sign(q) * (q / 127.5) ** 2; x[bad] = np.nan
    return x

def vat_table():
    v = pd.read_parquet(f'{ROOT}/data/treemap/vat.parquet'); v['TM_ID'] = v.TM_ID.astype(np.int64)   # VAT stores TM_ID as text, trees as int
    t = pd.read_parquet(f'{ROOT}/data/treemap/trees.parquet')
    t = t[t.STATUSCD == 1].copy()
    t['ba'] = 0.005454 * t.DIA ** 2 * t.TPA_UNADJ
    t['soft'] = t.SPCD < 300
    sw = t.groupby('TM_ID').apply(lambda g: (g.ba * g.soft).sum() / max(g.ba.sum(), 1e-9), include_groups=False)
    v['SOFTWOOD_FRAC'] = v.TM_ID.map(sw).fillna(0.0)
    return v

def open_sources():
    s = dict(land=f'{ROOT}/data/derived/land_clean.tif', tm=f'{ROOT}/data/treemap/tm_id.tif',
             a25=f'{ROOT}/data/aef/aef_2025.tif', a22=f'{ROOT}/data/aef/aef_2022.tif')
    for k in SOILS: s[k] = f'{ROOT}/data/soils/{k}.tif'
    return {k: rasterio.open(p) for k, p in s.items()}

class Featurizer:
    def __init__(self, fit=None):
        self.v = vat_table()
        self.tm_lut_idx = None
        self.fit = fit
        ids = self.v.TM_ID.values.astype(np.int64)
        self.lut = np.full(int(ids.max()) + 1, -1, np.int64); self.lut[ids] = np.arange(ids.size)
        A = np.stack([self.v[a].astype(float).values for a in TM_ATTRS], 1)
        for j, a in enumerate(TM_ATTRS):
            if a in TM_LOG: A[:, j] = np.log1p(np.clip(A[:, j], 0, None))
        self.tm_attr = np.nan_to_num(A).astype(np.float32)

    def raw(self, src, win):
        """raw per-pixel inputs for a window: dict of arrays (H, W, k)."""
        land = src['land'].read(1, window=win)
        tm = src['tm'].read(1, window=win).astype(np.int64)
        a25 = np.moveaxis(dequant(src['a25'].read(window=win)), 0, -1)
        a22 = np.moveaxis(dequant(src['a22'].read(window=win)), 0, -1)
        soil = np.stack([src[k].read(1, window=win).astype(np.float32) for k in SOILS], -1)
        for k, s in zip(SOILS, range(len(SOILS))):
            nd = src[k].nodata
            if nd is not None: soil[..., s][soil[..., s] == nd] = np.nan
        ok_tm = (tm >= 0) & (tm < self.lut.size)
        idx = np.where(ok_tm, self.lut[np.clip(tm, 0, self.lut.size - 1)], -1)
        tma = np.where((idx >= 0)[..., None], self.tm_attr[np.clip(idx, 0, None)], np.nan)
        tma[(land != 1)] = np.nan          # TreeMap attributes only describe un-harvested TreeMap forest
        return dict(land=land, tm=tm, a25=a25, a22=a22, soil=soil, tma=tma)

    def features(self, r):
        f = self.fit; w = CFG['segmentation']['weights']
        out = []
        def grp(x, mean, comp, scale, weight):
            y = (np.nan_to_num(x - mean)) @ comp.T
            return y * (weight / scale)
        out.append(grp(r['a25'], f['aef_mean'], f['aef_comp'], f['aef_scale'], w['aef']))
        out.append(grp(r['a22'], f['aef_mean'], f['aef_comp'][:8], f['aef_scale8'], w.get('aef_old', 0.4)))
        cos = np.nansum(r['a25'] * r['a22'], -1) / np.sqrt(np.nansum(r['a25'] ** 2, -1) * np.nansum(r['a22'] ** 2, -1) + 1e-12)
        ch = (1 - np.nan_to_num(cos, nan=1.0))[..., None]
        out.append(ch * (w['aef_change'] / f['change_sd']))
        tz = (r['tma'] - f['tm_mean']) / f['tm_sd']
        out.append(np.nan_to_num(tz) * (w['treemap'] / np.sqrt(tz.shape[-1])))
        sz = (r['soil'] - f['soil_mean']) / f['soil_sd']
        out.append(grp(sz, 0, f['soil_comp'], f['soil_scale'], w['soil']))
        return np.concatenate(out, -1).astype(np.float32)
