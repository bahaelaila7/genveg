import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sstyle import *
import pandas as pd, numpy as np
T = WS / 'standmap' / 'data' / 'derived' / 'tune'
def get(sweep, v):
    r = pd.read_csv(T / sweep / 'results.csv'); b = float(r[r.variant == v].mean_share.iloc[0])
    try:
        w = pd.read_csv(T / sweep / 'report_window.csv'); w = w[w.variant == v]
        c = float(w.mean_share.iloc[0]) if len(w) and 'mean_share' in w else np.nan
    except Exception: c = np.nan
    return b, c
V = [('34-d stack, compactness 0.3 (old default)', 'tune_embed', 0),
     ('34-d stack, fixed scaling, c 0.15', 'tune_slic', 6),
     ('Raw AlphaEarth 22+25, c 0.6', 'tune_embed', 4),
     ('Raw AlphaEarth 22+25, c 0.3', 'tune_embed', 3),
     ('Raw AlphaEarth 22+25, c 0.05', 'tune_embed2', 3),
     ('IWAE latent 24-d, c 0.08', 'tune_embed2', 6),
     ('IWAE latent 12-d, c 0.08', 'tune_embed2', 4),
     ('Raw AlphaEarth 22+25, c 0.08', 'tune_embed2', 2),
     ('Raw AlphaEarth 22+25, c 0.15 (default)', 'tune_embed', 2),
     ('... + superpixels 0.25 ha', 'tune_hier_a', 1)]
rows = [(l, *get(s, v)) for l, s, v in V]
for r in rows: print(r)
lev = [('fine\n~5 ha', .2276, SKY), ('stands\n~15 ha', .1971, GREEN), ('compart.\n~50 ha', -.02, RUST)]  # levels/log.txt of runs/gafl_pilot
fig, (ax, ax2) = plt.subplots(1, 2, figsize=SLIDE_FULL, gridspec_kw=dict(width_ratios=[3, 1], wspace=.35))
y = np.arange(len(rows)); h = .38
for i, (l, b, c) in enumerate(rows):
    d = 'default' in l
    ax.barh(i + h / 2, b, h, color=GREEN if d else MOSS, alpha=1 if d else .8)
    if not np.isnan(c): ax.barh(i - h / 2, c, h, color=AMBER if d else '#E3B766')
    ax.text(b + .002, i + h / 2, f'{b:.3f}', va='center', fontsize=11)
    if not np.isnan(c): ax.text(c + .002, i - h / 2, f'{c:.3f}', va='center', fontsize=11)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=12); ax.set_xlim(.15, .215)
for t in ax.get_yticklabels():
    if 'default' in t.get_text(): t.set_fontweight('bold')
ax.set_xlabel('mean share vs. equal-count blocks', fontsize=14); ax.tick_params(axis='x', labelsize=12)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=MOSS, label='Baker (ranking)'), Patch(color='#E3B766', label='Waycross (report)')], loc='lower right', fontsize=12, frameon=False)
ax2.bar(range(3), [v for _, v, _ in lev], color=[c for *_, c in lev])
for i, (n, v, c) in enumerate(lev): ax2.text(i, v + (.006 if v > 0 else -.02), f'{v:.2f}' if v < 0 else f'{v:.3f}', ha='center', fontsize=12)
ax2.axhline(0, color=INK, lw=1); ax2.set_xticks(range(3)); ax2.set_xticklabels([n for n, *_ in lev], fontsize=12)
ax2.set_ylabel('held-out mean share', fontsize=14); ax2.set_title('Nested levels', fontsize=15); ax2.set_ylim(-.05, .26)
ax2.tick_params(axis='y', labelsize=12)
fig.subplots_adjust(left=.29, right=.99, bottom=.14, top=.92); save(fig, 'stands_tuning')
