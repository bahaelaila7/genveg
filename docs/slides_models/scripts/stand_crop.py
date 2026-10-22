"""Shared crop loader for the stand-map figures (reads existing outputs only)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sstyle import *
import numpy as np, rasterio
from rasterio.windows import Window
RUN = WS / 'standmap' / 'data' / 'derived' / 'runs' / 'gafl_pilot'
R0, C0, H, W = 16243, 16603, 100, 133     # Baker window, ~4.0 x 3.0 km; forest + clear-cuts + regeneration
K = 6                                      # boundary up-sampling
def rd(p, bands=None):
    with rasterio.open(p) as r:
        w = Window(C0, R0, W, H)
        return r.read(bands, window=w) if bands else r.read(1, window=w)
def aef_rgb():
    q = rd(WS / 'standmap' / 'data' / 'aef' / 'aef_2025.tif', list(range(1, 65))).astype(np.float32)
    bad = (q == -128).any(0); x = np.sign(q) * (q / 127.5) ** 2
    X = x.reshape(64, -1).T; ok = ~bad.ravel(); Z = X[ok] - X[ok].mean(0)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    pc = Z @ vt[:3].T; out = np.ones((H * W, 3), np.float32)
    lo, hi = np.percentile(pc, 2, 0), np.percentile(pc, 98, 0)
    out[ok] = np.clip((pc - lo) / (hi - lo), 0, 1); return out.reshape(H, W, 3)
def up(a): return np.kron(a, np.ones((K, K), a.dtype))
def edges(lab):
    u = up(lab); b = np.zeros(u.shape, bool)
    b[:, :-1] |= u[:, :-1] != u[:, 1:]; b[:-1, :] |= u[:-1, :] != u[1:, :]
    d = b.copy(); d[1:, :] |= b[:-1, :]; d[:, 1:] |= b[:, :-1]; d[1:, 1:] |= b[:-1, :-1]; return d
def draw_edges(ax, lab, color='white', alpha=1.0, base=None):
    b = edges(lab); o = np.zeros(b.shape + (4,), np.float32); o[b] = (*matplotlib.colors.to_rgb(color), alpha)
    ax.imshow(o, extent=(0, W * 0.03, 0, H * 0.03), interpolation='antialiased')
def setup(ax, rgb_or_none=None):
    ax.set_xlim(0, W * .03); ax.set_ylim(0, H * .03); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
def scalebar(ax, km=1.0, color='white'):
    x0, y0 = 0.2, 0.2
    ax.plot([x0, x0 + km], [y0, y0], color=color, lw=4, solid_capstyle='butt')
    ax.text(x0 + km / 2, y0 + .12, f'{km:g} km', color=color, ha='center', fontsize=14, fontweight='bold')
def show_rgb(ax, rgb): ax.imshow(rgb, extent=(0, W * .03, 0, H * .03), interpolation='nearest')
