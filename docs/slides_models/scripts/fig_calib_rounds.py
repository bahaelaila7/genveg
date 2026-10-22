import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sstyle import *
from sdata import *
fig, axs = plt.subplots(1, 4, figsize=SLIDE_FULL, sharey=False)
for ax, k in zip(axs, MODELS):
    lab, col = MODELS[k][:2]; r = rounds(k)
    x = r['round'].values
    ax.plot(x, r.n10, '-o', color=col, ms=4, label='groups within ±10%')
    ax.plot(x, r.n20, '--s', color=col, ms=4, alpha=.7, label='groups within ±20%')
    ax.set_ylim(0, 50); ax.set_title(lab, fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(r.param, rotation=90, fontsize=9)
    ax.set_xlabel('calibration round (parameter adjusted)', fontsize=10)
    if k == 'bs': ax.set_ylabel('groups passing (of ~47)')
    a2 = ax.twinx(); a2.plot(x, r.median_abs_bias, ':', color=MUTED, marker='^', ms=4)
    a2.set_yscale('symlog', linthresh=1); a2.set_ylim(0, 100); a2.grid(False)
    a2.set_yticks([0, 1, 10, 100]); a2.set_yticklabels(['0', '1', '10', '100'])
    if k == 'simb': a2.set_ylabel('median |bias| (%, dotted)')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower right', ncol=2, fontsize=9, frameon=False)
fig.text(.3, .01, 'all-data fit (in-sample)', ha='center', fontsize=10, color=MUTED, style='italic')
fig.tight_layout(rect=(0, .03, 1, 1)); save(fig, 'fig_calib_rounds')
