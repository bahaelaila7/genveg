import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sstyle import *
from sdata import *
KEY = 'hbad'
lab, col = MODELS[KEY][:2]; fd = folds(KEY); order = list(pooled_n(fd).index)
half = (len(order) + 1) // 2
fig, axs = plt.subplots(1, 2, figsize=SLIDE_FULL, sharex=True)
fcol = {'train': GREEN, 'val': AMBER, 'test': SKY}
allv = pd.concat(fd.values()); lim = float(np.clip(max(abs(allv.ci_lo).max(), abs(allv.ci_hi).max()) * 1.05, 25, 80))
for ax, grp in zip(axs, [order[:half], order[half:]]):
    ys = np.arange(len(grp))[::-1]
    ax.axvspan(-20, 20, color=TINT, zorder=0); ax.axvspan(-10, 10, color='#cfe0c8', zorder=0); ax.axvline(0, color=INK, lw=.8)
    for j, f in enumerate(FOLDS):
        d = fd[f].set_index('group').reindex(grp); yy = ys + (j - 1) * .22
        ax.errorbar(d.pct_bias, yy, xerr=[(d.pct_bias - d.ci_lo).clip(lower=0), (d.ci_hi - d.pct_bias).clip(lower=0)], fmt='o', ms=3.5, lw=.8, color=fcol[f], label='held-out ' + f, elinewidth=.8)
    ax.set_yticks(ys); ax.set_yticklabels(grp, fontsize=7); ax.set_ylim(-.7, len(grp) - .3)
    ax.set_xlim(-lim, lim); ax.set_xlabel('mean AGB bias (%, 90% CI)'); ax.tick_params(axis='x', labelsize=9)
axs[0].set_ylabel('group (sorted by n, largest on top)', fontsize=9)
axs[0].legend(loc='lower left', fontsize=8, frameon=True); axs[1].set_title('shading: ±10% / ±20%', fontsize=9, color=MUTED)
fig.suptitle(lab + ': held-out per-group bias', fontsize=14); fig.tight_layout(); save(fig, 'fig_heldout_bias_' + KEY)
