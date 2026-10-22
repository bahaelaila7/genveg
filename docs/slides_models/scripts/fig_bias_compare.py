import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sstyle import *
from sdata import *
ks = ['bs', 'pnet', 'hbad']; fds = {k: folds(k) for k in ks}
allf = {k: pd.concat(fds[k].values()) for k in ks}
n = pd.concat(allf.values()).groupby('group').n.sum().sort_values(ascending=False)
order = list(n.index); half = (len(order) + 1) // 2
fig, axs = plt.subplots(1, 2, figsize=SLIDE_FULL, sharex=True)
for ax, grp in zip(axs, [order[:half], order[half:]]):
    ys = np.arange(len(grp))[::-1]
    ax.axvspan(-20, 20, color=TINT, zorder=0); ax.axvspan(-10, 10, color='#cfe0c8', zorder=0); ax.axvline(0, color=INK, lw=.8)
    for j, k in enumerate(ks):
        m = allf[k].groupby('group').pct_bias.mean().reindex(grp)
        ax.plot(m.values, ys + (j - 1) * .2, 'o', ms=4.5, color=MODELS[k][1], label=MODELS[k][0])
    ax.set_yticks(ys); ax.set_yticklabels(grp, fontsize=7); ax.set_ylim(-.7, len(grp) - .3)
    ax.set_xlim(-60, 60); ax.set_xlabel('fold-mean held-out AGB bias (%)')
axs[0].set_ylabel('group (sorted by n, largest on top)', fontsize=9); axs[0].legend(fontsize=8, loc='lower left')
fig.tight_layout(); save(fig, 'fig_bias_compare')
