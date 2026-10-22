"""Stand-delineation verification (pipeline stage `verify`; docs/STANDS_PIPELINE.md). Invariants of one run, on its window; PASS / FAIL per check,
exit code = number of FAILs. Inputs: the run's stage dirs (STANDS_STAGE_JSON from the driver)."""
import os, sys, json, subprocess, numpy as np, pandas as pd, rasterio, geopandas as gpd
from rasterio.windows import Window
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stands_cfg import CFG, IN, WINDOW, ROOT, sp_extent, mmu_px, barrier_files
from features import TM_ATTRS, vat_table
NF = [0]
def check(cid, ok, msg): print(f"{'PASS' if ok else 'FAIL'} {cid:4s} {msg}", flush=True); NF[0] += (not ok)
PXHA = 0.09; SEG = CFG['segmentation']; MMU_PX = mmu_px(CFG); MAX_PX = SEG['max_ha'] / PXHA
F = IN('finalize'); srs = rasterio.open(f'{F}/stands.tif'); H, W = srs.shape
r0, c0, r1, c1 = sp_extent(CFG, WINDOW); win = Window(c0, r0, c1 - c0, r1 - r0)   # the superpixel extent (finalize's)
A = srs.read(1, window=win).astype(np.int64)
L = rasterio.open(f"{IN('clean')}/land_clean.tif").read(1, window=win)
SP = rasterio.open(f"{IN('sp')}/sp.tif").read(1, window=win).astype(np.int64)
fl = (L >= 1) & (L <= 4)
edge_ids = set(np.unique(np.concatenate([A[0], A[-1], A[:, 0], A[:, -1]])).tolist()) - {0}      # stands cut by the data extent
print(f'extent rows {r0}:{r1} cols {c0}:{c1}; forest-land px {fl.sum()}; stands {len(np.unique(A[A > 0]))} ({len(edge_ids)} touch the extent border: excluded from stand-level checks)')
# V1 coverage
check('V1', bool((A[fl] > 0).all()) and bool((A[~fl] == 0).all()), f'every forest-land pixel in a stand ({(A[fl] == 0).sum()} missing), none outside forest land ({(A[~fl] > 0).sum()})')
# V2 connectivity
ids, size = np.unique(A[A > 0], return_counts=True); K = ids.size; objs = ndimage.find_objects(A)
disc = []; edge = np.zeros(A.shape, bool); edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
for sid in ids:                                                   # stands whose pixels form > 1 4-connected piece; minor pieces = all but the largest
    sl = objs[sid - 1]; m = A[sl] == sid; lab, n = ndimage.label(m)
    if n > 1:
        sz = np.bincount(lab.ravel())[1:]; disc.append((sid, n, int(sz.sum() - sz.max()), bool((edge[sl] & m).any())))
inner = [d for d in disc if not d[3]]; mpx = sum(d[2] for d in inner)
check('V2', mpx <= 1e-4 * fl.sum(), f'stands are single 4-connected pieces: disconnected {len(inner)} stands inside the window (minor pieces {mpx} px = {100*mpx/max(fl.sum(),1):.4f} % of the area); '
      f'{len(disc) - len(inner)} more touch the window edge (they may connect outside it)')
# V3 status purity
cnt = np.zeros((A.max() + 1, 5), np.int64); np.add.at(cnt, (A[A > 0], np.clip(L[A > 0], 0, 4)), 1)
maj = cnt[:, 1:].argmax(1) + 1; minority = cnt[ids, 1:].sum(1) - cnt[ids, 1:].max(1)
mfrac = minority.sum() / max(fl.sum(), 1)
check('V3', (not SEG['status_is_hard']) or mfrac < 0.005, f'status purity (status_is_hard={SEG["status_is_hard"]}): pixels outside their stand\'s majority status {100*mfrac:.3f} % '
      f'(barrier / sliver reassignment); stands with mixed status {(minority > 0).sum()}')
# V4 superpixel -> stand: a superpixel never split between stands
m = SP > 0; pairs = np.unique(np.stack([SP[m], A[m]], 1), axis=0); split = np.unique(pairs[:, 0], return_counts=True)[1] > 1
check('V4', not split.any(), f'every superpixel inside exactly one stand ({split.sum()} split of {len(split)})')
# V5 MMU after generalisation: no small stand with an admissible (same-status if hard) neighbour
small = set(ids[size < MMU_PX].tolist()) - edge_ids; adm = 0
bar = np.zeros(A.shape, bool)
if not CFG['finalize']['mmu_across_barriers']:
    for b in barrier_files(CFG):
        with rasterio.open(b) as bs: bar |= bs.read(1, window=win) > 0
if small:
    pr = []
    for a, b, ba, bb in ((A[:, :-1], A[:, 1:], bar[:, :-1], bar[:, 1:]), (A[:-1], A[1:], bar[:-1], bar[1:])):
        mm = (a != b) & (a > 0) & (b > 0) & ~ba & ~bb; pr.append(np.stack([a[mm], b[mm]], 1)); pr.append(np.stack([b[mm], a[mm]], 1))
    P = np.unique(np.concatenate(pr), axis=0); P = P[np.isin(P[:, 0], list(small))]
    if SEG['status_is_hard']: P = P[maj[P[:, 0]] == maj[P[:, 1]]]
    adm = len(np.unique(P[:, 0])) if len(P) else 0
gen = json.load(open(f'{F}/generalisation.json'))
across = CFG['finalize']['mmu_across_barriers']
check('V5', adm == 0 and not gen.get('skipped', False), f'interior stands < MMU ({SEG["mmu_ha"]} ha): {len(small)} ({100*size[size < MMU_PX].sum()/size.sum():.2f} % of the area); with an admissible neighbour {adm} '
      f'(generalisation passes {gen["passes"]}, merged {gen["merged"]}, skipped {gen.get("skipped", False)}; across barriers={across})')
# V6 max size
big = size > MAX_PX * 1.05
check('V6', big.sum() <= max(1, 0.001 * K), f'stands > max_ha ({SEG["max_ha"]} ha, +5 % for reassigned barrier pixels): {big.sum()} (largest {size.max()*PXHA:.0f} ha)')
# V7 polygons <-> raster
g = gpd.read_file(f'{F}/stands.gpkg', layer='stands')
ok_ids = set(g.stand_id.astype(int)) == set(ids.tolist())
pa = dict(zip(ids.tolist(), (size * PXHA).tolist())); ga = g.set_index('stand_id')
err_a = max(abs(ga.area_ha[s] - pa[s]) for s in ids.tolist())
err_g = float(np.max(np.abs(ga.geometry.area.values / 1e4 - ga.area_ha.values)))
err_s = int((ga.status.astype(int).values != maj[ga.index.values.astype(int)]).sum())
check('V7', ok_ids and err_a < 1e-6 and err_g < 1e-6 and err_s == 0, f'gpkg: same stand ids ({ok_ids}), area_ha = pixel count × 0.09 (max err {err_a:.2g}), polygon area = area_ha (max err {err_g:.2g} ha), status = majority ({err_s} differ)')
# V8 attributes recomputed from pixels (TreeMap means) on a sample of stands
v = vat_table(); vid = v.TM_ID.values.astype(np.int64); lut = np.full(int(vid.max()) + 1, -1, np.int64); lut[vid] = np.arange(vid.size)
TM = rasterio.open(f'{ROOT}/data/treemap/tm_id.tif').read(1, window=win).astype(np.int64)
vi = np.where((TM > 0) & (TM < lut.size), lut[np.clip(TM, 0, lut.size - 1)], -1)
rng = np.random.default_rng(0); samp = rng.choice(ids, min(CFG['verify']['attr_sample'], K), replace=False); bad = 0
for s in samp:
    mm = (A == s) & (vi >= 0)
    for a in ('BALIVE', 'QMD'):
        x = v[a].astype(float).values[vi[mm]]; want = x.mean() if x.size else np.nan; got = ga.loc[s, a]
        bad += not ((np.isnan(want) and pd.isna(got)) or abs(want - got) <= 1e-6 * max(1, abs(want)))
check('V8', bad == 0, f'TreeMap stand means (BALIVE, QMD) recomputed from pixels for {len(samp)} stands: {bad} mismatches')
# V9 merge threshold calibration
sf = np.load(f"{IN('merge')}/stand_feat.npz"); thr = float(sf['threshold'])
a = np.sort(size * PXHA); wmed = a[np.searchsorted(np.cumsum(a), a.sum() / 2)]
auto = SEG['merge_threshold'] == 'auto'; tgt = SEG['target_median_ha']
check('V9', (not auto) or abs(wmed / tgt - 1) < 0.35, f'merge threshold {thr:.4g} ({"auto" if auto else "fixed"}); final area-weighted median {wmed:.1f} ha (target {tgt} ha before generalisation)')
# V10 unit test of the merge
r = subprocess.run([sys.executable, '-I', os.path.join(ROOT, 'scripts', 'test_rag_merge.py')], capture_output=True, text=True)
check('V10', r.returncode == 0 and r.stdout.strip().endswith('OK'), 'rag_merge unit tests')
print('ALL CHECKS PASS' if NF[0] == 0 else f'{NF[0]} CHECK(S) FAILED'); sys.exit(NF[0])
