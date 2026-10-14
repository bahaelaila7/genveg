"""Remove status speckle: connected patches (4-conn) of one forest-land class smaller than the MMU take the class of the
nearest larger forest-land patch (within 15 px), so e.g. 3-pixel Hansen loss specks inside a stand do not become
'harvested' islands, and small retention clumps inside a clear-cut join the harvest. Isolated forest-land fragments
(nothing larger nearby) are kept as they are. data/derived/land.tif -> land_clean.tif"""
import sys, os, numpy as np, rasterio, yaml
from rasterio.windows import Window
from scipy import ndimage
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__)); from common import ROOT, PROFILE
cfg = yaml.safe_load(open(f'{ROOT}/config/pilot.yml'))['segmentation']
MMU = int(round(cfg['mmu_ha'] * 1e4 / 900)); T, HALO = 2048, 128
SRC = f'{ROOT}/data/derived/land.tif'

def work(rc):
    r, c = rc
    with rasterio.open(SRC) as s:
        H, W = s.shape
        r0, c0 = max(r - HALO, 0), max(c - HALO, 0); r1, c1 = min(r + T + HALO, H), min(c + T + HALO, W)
        land = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
    fl = (land >= 1) & (land <= 4)
    if not fl.any(): return r, c, None
    small = np.zeros(land.shape, bool)
    for cls in (1, 2, 3, 4):
        lab, n = ndimage.label(land == cls)
        if n == 0: continue
        sz = np.bincount(lab.ravel()); sz[0] = 1 << 30
        small |= sz[lab] < MMU
    target = fl & ~small
    out = land.copy()
    if small.any() and target.any():
        d, (iy, ix) = ndimage.distance_transform_edt(~target, return_indices=True)
        re = small & (d <= 15)
        out[re] = land[iy[re], ix[re]]
    h, w = min(T, H - r), min(T, W - c)
    return r, c, out[r - r0:r - r0 + h, c - c0:c - c0 + w]

if __name__ == '__main__':
    with rasterio.open(SRC) as s: prof = s.profile; H, W = s.shape
    jobs = [(r, c) for r in range(0, H, T) for c in range(0, W, T)]
    with rasterio.open(f'{ROOT}/data/derived/land_clean.tif', 'w', **dict(prof, **PROFILE)) as d, Pool(6) as p:
        for i, (r, c, a) in enumerate(p.imap_unordered(work, jobs)):
            if a is not None: d.write(a, 1, window=Window(c, r, a.shape[1], a.shape[0]))
            if i % 50 == 0: print(i, len(jobs), flush=True)
    print('done')
