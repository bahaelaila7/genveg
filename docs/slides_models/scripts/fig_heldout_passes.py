import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sstyle import *
from sdata import *
fig, (ax, ax2) = plt.subplots(1, 2, figsize=SLIDE_FULL, gridspec_kw={'width_ratios': [3, 1]})
w = .26; shades = {5: .45, 10: .7, 20: 1.0}
for i, k in enumerate(MODELS):
    lab, col = MODELS[k][:2]; fd = folds(k); pc = pass_counts(fd)
    print(lab, {t: (np.mean(v), v) for t, v in pc.items()}, [len(d) for d in fd.values()])
    for j, t in enumerate((5, 10, 20)):
        v = np.array(pc[t]); m = v.mean()
        ax.bar(i + (j - 1) * w, m, w * .92, color=col, alpha=shades[t], yerr=[[m - v.min()], [v.max() - m]], capsize=3, error_kw=dict(lw=1))
        ax.text(i + (j - 1) * w, v.max() + 1, '%.0f' % m, ha='center', fontsize=10)
ax.set_xticks(range(4)); ax.set_xticklabels(['Biomass\nSuccession', 'PnET', 'HBAD v1', 'BS Sim B'])
ax.set_ylabel('held-out groups passing (of ~47)'); ax.set_ylim(0, 52)
from matplotlib.patches import Patch
ax.legend([Patch(color=GREEN, alpha=a) for a in shades.values()], ['±5%', '±10%', '±20%'], title='TOST margin', loc='upper right', fontsize=9, frameon=False)
ax.set_title('mean over 3 folds, bars = min–max', fontsize=11)
for i, k in enumerate(MODELS):
    fd = folds(k); v = [np.median(np.abs(d.pct_bias)) for d in fd.values()]
    ax2.bar(i, np.mean(v), .6, color=MODELS[k][1]); ax2.text(i, np.mean(v) + .5, '%.1f' % np.mean(v), ha='center', fontsize=10)
    print(k, 'median|bias| per fold', np.round(v, 1), 'mean', round(np.mean(v), 1))
ax2.set_xticks(range(4)); ax2.set_xticklabels(['BS', 'PnET', 'HBAD', 'Sim B'], fontsize=10)
ax2.set_ylabel('median held-out |bias| (%)'); ax2.set_title('median over groups', fontsize=11)
fig.tight_layout(); save(fig, 'fig_heldout_passes')
