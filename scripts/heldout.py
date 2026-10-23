"""Held-out evaluation shared by stands_metrics.py (stands) and 25_levels.py (fine units, stands, compartments): one definition of the evaluated
pixels, the equal-count block baseline and R².
  eval_mask   forest-land pixels of the units that are interior (not cut by the extent border) and not ISOLATED SUB-MMU pieces: a unit below the MMU
              with no 4-adjacent unit (forest pixels boxed in by barriers / non-forest; 15 % of the stands, 0.5 % of the area in Baker 2026-10-23).
              They count as units but carry almost no area, so including them inflated the unit count the block baseline is matched to.
  baseline    square blocks x land status. The block side is an integer, so no side gives exactly K units: the two sides bracketing K are found
              (lo: more units than K, hi: at most K) and the block R² is interpolated in log(unit count), t clamped to [0, 1].
  share       (R²_units − R²_blocks) / (1 − R²_blocks)."""
import numpy as np

def isolated_small(U, mmu_px):
    """ids of units below mmu_px pixels without any 4-adjacent unit"""
    ids, n = np.unique(U[U > 0], return_counts=True); nb = np.zeros(int(U.max()) + 1, bool)
    for a, b in ((U[:, 1:], U[:, :-1]), (U[1:], U[:-1])):
        k = (a > 0) & (b > 0) & (a != b); nb[a[k]] = True; nb[b[k]] = True
    return ids[(n < mmu_px) & ~nb[ids]]

def eval_mask(U, L, mmu_px):
    """(mask, K, n_isolated_small, n_edge): evaluated pixels and their unit count"""
    edge = np.unique(np.concatenate([U[0], U[-1], U[:, 0], U[:, -1]])); iso = isolated_small(U, mmu_px)
    fl = (L >= 1) & (L <= 4) & (U > 0) & ~np.isin(U, edge) & ~np.isin(U, iso)
    return fl, int(np.unique(U[fl]).size), int(iso.size), int((edge > 0).sum())

class Baseline:
    """equal-count square blocks x status on the evaluated pixels, interpolated between the two bracketing integer sides"""
    def __init__(self, L, fl, K):
        rr, cc = np.mgrid[0:L.shape[0], 0:L.shape[1]]; self._b = lambda s: ((rr // s) * (L.shape[1] // s + 1) + cc // s) * 8 + L.astype(np.int64)
        nunits = lambda s: np.unique(self._b(s)[fl]).size
        lo, hi = 1, int(np.sqrt(fl.sum())) + 2
        while hi - lo > 1:
            mid = (lo + hi) // 2; (lo, hi) = (mid, hi) if nunits(mid) > K else (lo, mid)
        self.n_lo, self.n_hi = nunits(lo), nunits(hi); self.side = (lo, hi)
        t = (np.log(K) - np.log(self.n_hi)) / (np.log(self.n_lo) - np.log(self.n_hi)) if self.n_lo != self.n_hi else 0.0
        self.t = float(min(max(t, 0.0), 1.0)); self.B_lo, self.B_hi = self._b(lo), self._b(hi)
    def r2(self, x, mk): return r2(self.B_hi, x, mk) + self.t * (r2(self.B_lo, x, mk) - r2(self.B_hi, x, mk))

def r2(lab, x, mk):
    l = lab[mk]; v = x[mk]; _, inv = np.unique(l, return_inverse=True)
    n = np.bincount(inv); s = np.bincount(inv, v); s2 = np.bincount(inv, v * v)
    within = (s2 - s * s / n).sum(); tot = ((v - v.mean()) ** 2).sum()
    return float(1 - within / tot) if tot > 0 else float('nan')
