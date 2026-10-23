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
import heldout as HO
PXHA = 0.09; SEG = CFG['segmentation']; MMU_PX = mmu_px(CFG)
F = IN('finalize'); srs = rasterio.open(f'{F}/stands.tif'); H, W = srs.shape
r0, c0, r1, c1 = sp_extent(CFG, WINDOW); win = Window(c0, r0, c1 - c0, r1 - r0)   # the superpixel extent (finalize's)
A = srs.read(1, window=win).astype(np.int64); L = rasterio.open(f"{IN('clean')}/land_clean.tif").read(1, window=win)
SP = rasterio.open(f"{IN('sp')}/sp.tif").read(1, window=win).astype(np.int64)
edge_ids = np.unique(np.concatenate([A[0], A[-1], A[:, 0], A[:, -1]]))                    # stands cut by the data extent: excluded
fl_all = (L >= 1) & (L <= 4) & (A > 0) & ~np.isin(A, edge_ids)                           # interior stands (size statistics)
ids, size = np.unique(A[fl_all], return_counts=True); a = np.sort(size * PXHA)
fl, K, n_iso, _ = HO.eval_mask(A, L, MMU_PX)                                             # held-out evaluation: interior, without isolated sub-MMU pieces
out = dict(stands=int(ids.size), stands_evaluated=int(K), isolated_small_excluded=n_iso, area_ha=float(a.sum()), q05_25_50_75_95=np.quantile(a, [.05, .25, .5, .75, .95]).round(2).tolist(),
           wmedian_ha=float(a[np.searchsorted(np.cumsum(a), a.sum() / 2)]), small_area_share=float(size[size < MMU_PX].sum() / size.sum()), edge_stands_excluded=int((edge_ids > 0).sum()))
BL = HO.Baseline(L, fl, K); out['block_units'] = [BL.n_lo, BL.n_hi]; out['block_interp_t'] = round(BL.t, 4)
def r2(lab, x, m):
    l = lab[m]; v = x[m]; _, inv = np.unique(l, return_inverse=True)
    n = np.bincount(inv); s = np.bincount(inv, v); s2 = np.bincount(inv, v * v)
    within = (s2 - s * s / n).sum(); tot = ((v - v.mean()) ** 2).sum()
    return float(1 - within / tot) if tot > 0 else float('nan')
out['block_side_px'] = [int(BL.side[0]), int(BL.side[1])]; out['heldout'] = {}
def oracle_r2(x, ok):
    """R² ceiling at the same unit count: the run's superpixels merged by the same Ward / RAG code on the layer itself (status hard, no size cap / shape)."""
    from rag_merge import merge
    sp = np.where(ok & (SP > 0), SP, 0); idu, inv = np.unique(sp[sp > 0], return_inverse=True); loc = np.zeros(sp.max() + 1, np.int64); loc[idu] = np.arange(idu.size)
    n = np.bincount(inv).astype(float); st_ = np.zeros(idu.size, np.int64); st_[inv] = L[sp > 0]; mu = np.bincount(inv, x[sp > 0]) / n
    pa, pb = [], []
    for a_, b_ in ((sp[:, :-1], sp[:, 1:]), (sp[:-1], sp[1:])):
        k = (a_ > 0) & (b_ > 0) & (a_ != b_); pa.append(loc[a_[k]]); pb.append(loc[b_[k]])
    ea, eb = np.concatenate(pa), np.concatenate(pb); S_ = (mu * n)[:, None]
    lo, hi = 1e-6, 1e9; lab = None
    for _ in range(26):                                                     # bisection on the threshold to K units
        mid = np.sqrt(lo * hi); l_ = merge(n, S_, st_, ea, eb, np.ones(ea.size), 4 * n, mid, 1e12, shape_w=0.0, min_n=0, status_hard=True)
        if l_.max() + 1 > K: lo = mid
        else: hi = mid; lab = l_
    O = np.zeros_like(sp); O[sp > 0] = lab[inv] + 1
    return r2(O, x, ok & (SP > 0))
for rel in CFG['metrics']['heldout']:
    p = f'{ROOT}/data/{rel}'
    if not os.path.exists(p): out['heldout'][rel] = 'missing'; continue
    with rasterio.open(p) as s:
        assert s.transform == srs.transform and s.shape == srs.shape, f'{rel} not on the grid'
        x = s.read(1, window=win).astype(np.float64); nd = s.nodata
    ok = fl & np.isfinite(x) & ((x != nd) if nd is not None and not np.isnan(nd) else True)
    if ok.sum() < 1000: out['heldout'][rel] = 'too few pixels'; continue
    rs, rp = r2(A, x, ok), r2(SP, x, ok & (SP > 0)); rb = BL.r2(x, ok)                  # equal-count baseline (interpolated, heldout.Baseline)
    out['heldout'][rel] = dict(r2_stands=round(rs, 4), r2_blocks=round(rb, 4), gain=round(rs - rb, 4), share=round((rs - rb) / (1 - rb), 4) if 1 - rb > 1e-9 else None,
                               headroom=round(1 - rb, 4), r2_superpixels=round(rp, 4))
    if CFG['metrics'].get('oracle'):
        ro = oracle_r2(x, ok); out['heldout'][rel].update(r2_oracle=round(ro, 4), oracle_share=round((rs - rb) / (ro - rb), 4) if ro > rb else None)
g = [v['gain'] for v in out['heldout'].values() if isinstance(v, dict)]
out['mean_gain'] = float(np.mean(g)) if g else float('nan')
sh = [v['share'] for v in out['heldout'].values() if isinstance(v, dict) and v['headroom'] >= CFG['metrics']['headroom_min']]
out['mean_share'] = float(np.mean(sh)) if sh else float('nan')        # THE ranking metric: share of the blocks' unexplained variance captured
# tile-seam artifact: share of forest pixel pairs ACROSS a superpixel-tile seam that are stand boundaries vs the share among all pairs (1 = none)
Tt = int(SEG['sp_tile']); colg = np.arange(c0, c1 - 1) + 1; rowg = np.arange(r0, r1 - 1) + 1
pv = (A[:, :-1] > 0) & (A[:, 1:] > 0); ph = (A[:-1] > 0) & (A[1:] > 0); bv = pv & (A[:, :-1] != A[:, 1:]); bh = ph & (A[:-1] != A[1:])
sv, sh = colg % Tt == 0, rowg % Tt == 0
seam_pairs = pv[:, sv].sum() + ph[sh, :].sum()
if seam_pairs > 0:
    seam_rate = (bv[:, sv].sum() + bh[sh, :].sum()) / seam_pairs; all_rate = (bv.sum() + bh.sum()) / (pv.sum() + ph.sum())
    out['seam_boundary_rate'] = float(seam_rate); out['boundary_rate'] = float(all_rate); out['seam_excess'] = float(seam_rate / all_rate)
json.dump(out, open(f'{OUT}/metrics.json', 'w'), indent=1)
print(json.dumps(out['heldout']))
print(f"stands {ids.size} ({K} evaluated; {n_iso} isolated pieces < MMU excluded) (block units {out['block_units']} at sides {out['block_side_px']} px, interpolated); area-weighted median {out['wmedian_ha']:.1f} ha; area < MMU {100*out['small_area_share']:.2f} %; held-out R² gain over equal-count blocks: "
      + ', '.join(f"{os.path.basename(k).split('.')[0]} {v['gain']:+.3f}" for k, v in out['heldout'].items() if isinstance(v, dict)) + f" | mean gain {out['mean_gain']:+.3f} | mean SHARE of the blocks' unexplained variance {out['mean_share']:.3f}" + (f" | tile-seam excess {out['seam_excess']:.2f}x" if 'seam_excess' in out else ''))
