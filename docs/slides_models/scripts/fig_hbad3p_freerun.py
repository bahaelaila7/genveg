import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sstyle import *
import pandas as pd, numpy as np
B = WS / 'landis_hbad3' / 'results' / 'hbad3'
regs = [('GA+FL', 'fit3p_P3_GAFL'), ('Appalachians', 'fit3p_P3_APP')]
vars_ = [('BA', 'basal area (m²/ha)'), ('N12', 'stems ≥12.7 cm (/ha)'), ('QMD', 'QMD (cm)'), ('LH', 'Lorey height (m)')]
fig, axs = plt.subplots(2, 4, figsize=SLIDE_FULL)
for i, (rn, d) in enumerate(regs):
    df = pd.read_csv(B / d / 'freerun_val.csv')
    for j, (v, vl) in enumerate(vars_):
        s = df[df['var'] == v].dropna(subset=['obsL', 'sim']); ax = axs[i, j]
        ax.plot(s.obsL, s.sim, '.', ms=3, alpha=.4, color=MODEL_COLORS['HBAD3P'], mew=0)
        hi = max(s.obsL.max(), s.sim.max()) * 1.03; ax.plot([0, hi], [0, hi], color=INK, lw=.8); ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        bt = (s.sim.sum() / s.obsL.sum() - 1) * 100; bm = (s.sim / s.obsL.replace(0, np.nan)).mean() * 100 - 100
        r2 = 1 - ((s.obsL - s.sim) ** 2).sum() / ((s.obsL - s.obsL.mean()) ** 2).sum()
        print(rn, v, 'n=%d' % len(s), 'bias_totals=%+.1f%% mean_ratio=%+.1f%% R2=%.2f' % (bt, bm, r2))
        ax.text(.04, .96, 'n=%d\nbias %+.1f%%\nR²=%.2f' % (len(s), bt, r2), transform=ax.transAxes, va='top', fontsize=8)
        ax.set_xlabel('observed, last visit: ' + vl, fontsize=7.5); ax.tick_params(labelsize=8)
        if j == 0: ax.set_ylabel(rn + '\nsimulated', fontsize=10)
        if i == 0: ax.set_title(v, fontsize=12)
fig.tight_layout(); save(fig, 'fig_hbad3p_freerun')
