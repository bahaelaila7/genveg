"""Stages B + C: region-adjacency graph of superpixels (streamed), then Ward/mutual-best merging into stands.
Writes <OUT>/sp2stand.npy (superpixel id -> stand id, 0 = none) and <OUT>/stand_feat.npz
(stand id, n px, status, mean features). Threshold: config segmentation.merge_threshold, or 'auto' = bisection so the
area-weighted median stand size matches segmentation.target_median_ha on a sample of graph components.
segmentation.merge_method: ward (above) | graph_slic (hierarchical SLIC level 2 on the superpixel graph, graph_slic.py; size from graph_slic.ha)."""
import sys, os, time, numpy as np, rasterio
from rasterio.windows import Window
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, os.path.dirname(__file__))
from features import CFG, ROOT
from stands_cfg import OUT, IN
from rag_merge import merge
from graph_slic import graph_slic, aggregate
SEG = CFG['segmentation']; PXHA = 900 / 1e4
t0 = time.time()
st = np.load(f"{IN('sp')}/sp_stats.npz"); N = st['n'].size
n = st['n'].astype(np.float64); S = st['S'].astype(np.float64); status = st['status'].astype(np.int64)
# --- B: edges + perimeter ---
src = rasterio.open(f"{IN('sp')}/sp.tif"); H, W = src.shape
internal = np.zeros(N + 1, np.int64); keys = []; prev = None; sy = np.zeros(N + 1); sx = np.zeros(N + 1)
def pairs(a, b):
    m = (a > 0) & (b > 0)
    a, b = a[m].astype(np.int64), b[m].astype(np.int64)
    same = a == b
    np.add.at(internal, a[same], 1)
    a, b = a[~same], b[~same]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    u, c = np.unique(lo * (1 << 32) + hi, return_counts=True); return u, c
for r in range(0, H, 2048):
    x = src.read(1, window=Window(0, r, W, min(2048, H - r)))
    xi = x.astype(np.int64).ravel(); rows_, cols_ = np.divmod(np.arange(xi.size), W)          # superpixel centroids (graph_slic)
    sy += np.bincount(xi, rows_ + r, minlength=N + 1); sx += np.bincount(xi, cols_, minlength=N + 1)
    parts = [pairs(x[:, :-1], x[:, 1:]), pairs(x[:-1], x[1:])]
    if prev is not None: parts.append(pairs(prev, x[0]))
    prev = x[-1].copy()
    keys += parts
    print(f'edges row {r}/{H} {time.time()-t0:.0f}s', flush=True)
u = np.concatenate([k[0] for k in keys]); c = np.concatenate([k[1] for k in keys])
u, inv = np.unique(u, return_inverse=True); cnt = np.zeros(u.size, np.int64); np.add.at(cnt, inv, c)
ea = (u >> 32).astype(np.int64) - 1; eb = (u & 0xFFFFFFFF).astype(np.int64) - 1; eln = cnt.astype(np.float64)
perim = 4 * n - 2 * internal[1:]
cy = sy[1:] / np.maximum(n, 1); cx = sx[1:] / np.maximum(n, 1)
print(f'nodes {N} edges {u.size} ({time.time()-t0:.0f}s)', flush=True)
# --- graph components (barriers disconnect the graph) ---
G = coo_matrix((np.ones(ea.size), (ea, eb)), shape=(N, N))
ncomp, comp = connected_components(G, directed=False)
print('components', ncomp, flush=True)
from stands_cfg import mmu_px
max_n = SEG['max_ha'] / PXHA; min_n = mmu_px(CFG)            # regions < min_n px are absorbed (the MMU of every stage)
METHOD = SEG['merge_method']; GS = CFG['graph_slic']
def run_graph_slic(nodes_mask):
    """Hierarchical SLIC level 2 (graph_slic.py) on the superpixels of nodes_mask, then the MMU absorption of rag_merge.merge (no Ward merges)."""
    idx = np.flatnonzero(nodes_mask); loc = np.full(N, -1); loc[idx] = np.arange(idx.size)
    em = nodes_mask[ea] & nodes_mask[eb]; a_, b_, l_ = loc[ea[em]], loc[eb[em]], eln[em]
    lab0 = graph_slic(n[idx], S[idx], status[idx], cy[idx], cx[idx], a_, b_, np.sqrt(GS['ha'] / PXHA), GS['compactness'], GS['iters'])
    rn, rS, rst, ra, rb, rl, rp = aggregate(lab0, n[idx], S[idx], status[idx], a_, b_, l_, perim[idx])
    lab1 = merge(rn, rS, rst, ra, rb, rl, rp, 0.0, 1e18, shape_w=SEG['shape_weight'], min_n=min_n, status_hard=SEG['status_is_hard'])
    return idx, lab1[lab0]
def run(nodes_mask, thr):
    idx = np.flatnonzero(nodes_mask); loc = np.full(N, -1); loc[idx] = np.arange(idx.size)
    em = nodes_mask[ea] & nodes_mask[eb]
    lab = merge(n[idx], S[idx], status[idx], loc[ea[em]], loc[eb[em]], eln[em], perim[idx], thr, max_n,
                shape_w=SEG['shape_weight'], min_n=min_n, status_hard=SEG['status_is_hard'])
    return idx, lab
def wmedian_ha(idx, lab):
    sz = np.bincount(lab, weights=n[idx]) * PXHA; o = np.argsort(sz); cw = np.cumsum(sz[o]); return sz[o][np.searchsorted(cw, cw[-1] / 2)]
thr = SEG.get('merge_threshold', 'auto') if METHOD == 'ward' else float('nan')
if METHOD == 'ward' and thr == 'auto':
    rng = np.random.default_rng(0)
    csize = np.bincount(comp, weights=n)
    big = np.flatnonzero(csize * PXHA > 4 * SEG['target_median_ha'])
    if big.size == 0: big = np.arange(ncomp); print('  calib: no component > 4 x target; sampling all components', flush=True)
    pick = rng.choice(big, min(big.size, CFG['merge']['calib_components']), replace=False)
    sm = np.isin(comp, pick)
    lo, hi = 1e-2, 1e4; target = SEG['target_median_ha']
    for it in range(CFG['merge']['calib_iters']):
        mid = np.sqrt(lo * hi); idx, lab = run(sm, mid); med = wmedian_ha(idx, lab)
        print(f'  calib thr {mid:.4g} -> area-weighted median {med:.1f} ha', flush=True)
        if med < target: lo = mid
        else: hi = mid
    thr = np.sqrt(lo * hi)
    if lo <= 1e-2 * 1.0001 or hi >= 1e4 / 1.0001: raise SystemExit(f'auto threshold did not bracket the target median (ended at the bound {thr:.3g}): widen the range or change target_median_ha')
print('merge threshold', thr, flush=True)
# --- C: merge in batches of whole components ---
order = np.argsort(comp, kind='stable'); csz = np.bincount(comp)
sp2stand = np.zeros(N + 1, np.int64); next_id = 1
bounds = np.cumsum(csz); start = 0; batch_lim = 1_500_000
cstart = 0
while cstart < ncomp:
    cend = cstart; tot = 0
    while cend < ncomp and (tot == 0 or tot + csz[cend] <= batch_lim): tot += csz[cend]; cend += 1
    mask = (comp >= cstart) & (comp < cend)
    idx, lab = run(mask, thr) if METHOD == 'ward' else run_graph_slic(mask)
    sp2stand[idx + 1] = lab + next_id; next_id += lab.max() + 1
    print(f'components {cend}/{ncomp} stands {next_id-1} ({time.time()-t0:.0f}s)', flush=True)
    cstart = cend
K = next_id - 1
sid = sp2stand[1:]
sn = np.bincount(sid, weights=n, minlength=K + 1)
sS = np.zeros((K + 1, S.shape[1])); np.add.at(sS, sid, S)
# status = status of the largest share
W_ = np.stack([np.bincount(sid, weights=n * (status == s_), minlength=K + 1) for s_ in (1, 2, 3, 4)], 1)
sst = W_.argmax(1) + 1
os.makedirs(OUT, exist_ok=True)
np.save(f'{OUT}/sp2stand.npy', sp2stand)
np.savez(f'{OUT}/stand_feat.npz', n=sn[1:], status=sst[1:], mean=(sS[1:] / np.maximum(sn[1:, None], 1)).astype(np.float32),
         threshold=thr)
a = sn[1:] * PXHA
print(f'stands {K}; area ha quantiles 5/25/50/75/95%: {np.quantile(a, [.05, .25, .5, .75, .95]).round(1)}; '
      f'area-weighted median {np.sort(a)[np.searchsorted(np.cumsum(np.sort(a)), a.sum()/2)]:.1f}; < MMU: {(a < SEG["mmu_ha"]).mean():.3f}; '
      f'{time.time()-t0:.0f}s')
