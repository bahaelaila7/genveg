"""Stand-delineation quality metrics (pipeline stage `metrics`; the objective of scripts/stands_tune.py; docs/STANDS_PIPELINE.md).
On the run's window:
  size      stands, area quantiles, area-weighted median, area share in stands < MMU
  held-out  for every layer of config metrics.heldout that the segmentation does NOT use (Hansen 2000 canopy, LANDFIRE terrain, EVC tree cover ...):
            R² = 1 − within-stand SS / total SS over forest-land pixels, and the same for an EQUAL-COUNT square-block partition (blocks × land status, so
            the baseline also respects status). gain = R²(stands) − R²(blocks): how much more of an independent layer's spatial structure the stands
            capture than arbitrary units of the same number. Also R² of the superpixels (the finer building blocks: an upper reference).
Writes <run>/metrics.json; prints one summary line."""
import os, sys, json, numpy as np, rasterio
from rasterio.windows import Window
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stands_cfg import CFG, IN, WINDOW, ROOT, OUT, sp_extent, mmu_px
PXHA = 0.09; SEG = CFG['segmentation']; MMU_PX = mmu_px(CFG)
F = IN('finalize'); srs = rasterio.open(f'{F}/stands.tif'); H, W = srs.shape
r0, c0, r1, c1 = sp_extent(CFG, WINDOW); win = Window(c0, r0, c1 - c0, r1 - r0)   # the superpixel extent (finalize's)
A = srs.read(1, window=win).astype(np.int64); L = rasterio.open(f"{IN('clean')}/land_clean.tif").read(1, window=win)
SP = rasterio.open(f"{IN('sp')}/sp.tif").read(1, window=win).astype(np.int64)
edge_ids = np.unique(np.concatenate([A[0], A[-1], A[:, 0], A[:, -1]]))                    # stands cut by the data extent: excluded
fl = (L >= 1) & (L <= 4) & (A > 0) & ~np.isin(A, edge_ids)
ids, size = np.unique(A[fl], return_counts=True); K = ids.size; a = np.sort(size * PXHA)
out = dict(stands=int(K), area_ha=float(a.sum()), q05_25_50_75_95=np.quantile(a, [.05, .25, .5, .75, .95]).round(2).tolist(),
           wmedian_ha=float(a[np.searchsorted(np.cumsum(a), a.sum() / 2)]), small_area_share=float(size[size < MMU_PX].sum() / size.sum()), edge_stands_excluded=int((edge_ids > 0).sum()))
rr, cc = np.mgrid[0:A.shape[0], 0:A.shape[1]]
def blocks(side): return ((rr // side) * (A.shape[1] // side + 1) + cc // side) * 8 + L.astype(np.int64)
def nunits(side): return np.unique(blocks(side)[fl]).size
lo_, hi_ = 1, int(np.sqrt(fl.sum())) + 2                                   # EQUAL COUNT: the block side whose block x status units number K
while hi_ - lo_ > 1:
    mid = (lo_ + hi_) // 2; (lo_, hi_) = (mid, hi_) if nunits(mid) > K else (lo_, mid)
side = hi_ if abs(nunits(hi_) - K) <= abs(nunits(lo_) - K) else lo_
B = blocks(side); out['block_units'] = int(nunits(side))
def r2(lab, x, m):
    l = lab[m]; v = x[m]; _, inv = np.unique(l, return_inverse=True)
    n = np.bincount(inv); s = np.bincount(inv, v); s2 = np.bincount(inv, v * v)
    within = (s2 - s * s / n).sum(); tot = ((v - v.mean()) ** 2).sum()
    return float(1 - within / tot) if tot > 0 else float('nan')
out['block_side_px'] = side; out['heldout'] = {}
for rel in CFG['metrics']['heldout']:
    p = f'{ROOT}/data/{rel}'
    if not os.path.exists(p): out['heldout'][rel] = 'missing'; continue
    with rasterio.open(p) as s:
        assert s.transform == srs.transform and s.shape == srs.shape, f'{rel} not on the grid'
        x = s.read(1, window=win).astype(np.float64); nd = s.nodata
    ok = fl & np.isfinite(x) & ((x != nd) if nd is not None and not np.isnan(nd) else True)
    if ok.sum() < 1000: out['heldout'][rel] = 'too few pixels'; continue
    rs, rb, rp = r2(A, x, ok), r2(B, x, ok), r2(SP, x, ok & (SP > 0))
    out['heldout'][rel] = dict(r2_stands=round(rs, 4), r2_blocks=round(rb, 4), gain=round(rs - rb, 4), r2_superpixels=round(rp, 4))
g = [v['gain'] for v in out['heldout'].values() if isinstance(v, dict)]
out['mean_gain'] = float(np.mean(g)) if g else float('nan')
json.dump(out, open(f'{OUT}/metrics.json', 'w'), indent=1)
print(json.dumps(out['heldout']))
print(f"stands {K} (block units {out['block_units']}, side {side} px); area-weighted median {out['wmedian_ha']:.1f} ha; area < MMU {100*out['small_area_share']:.2f} %; held-out R² gain over equal-count blocks: "
      + ', '.join(f"{os.path.basename(k).split('.')[0]} {v['gain']:+.3f}" for k, v in out['heldout'].items() if isinstance(v, dict)) + f" | mean {out['mean_gain']:+.3f}")
