"""Generative vegetation, held-out mapping path 2022-2025: median |log error| of TPH / BA / QMD (calibrated mapping path vs kNN vs null)
and dominant-species accuracy, read from the genveg pipeline's own evaluation (data/proto/gafl_pipe/bayes_eval.json)."""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sstyle import *
import numpy as np
E = json.load(open(WS / 'standmap' / 'data' / 'proto' / 'gafl_pipe' / 'bayes_eval.json'))
M = E['mapping_path_heldout_2022_2025']; A = E['A_prior_vs_knn_all_heldout']
vars_ = ['TPH', 'BA', 'QMD']
rows = [('generative (mapping path)', [M[v]['map_path'] for v in vars_], GREEN),
        ('kNN on landscape features', [M[v]['knn'] for v in vars_], SKY),
        ('null (median)', [M[v]['null'] for v in vars_], MUTED)]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=SLIDE_FULL, gridspec_kw={'width_ratios': [3, 1.2]})
x = np.arange(len(vars_)); w = 0.26
for i, (lab, vals, c) in enumerate(rows):
    b = ax.bar(x + (i - 1) * w, vals, w, color=c, label=lab)
    for xi, v in zip(x + (i - 1) * w, vals): ax.text(xi, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
ax.set_xticks(x, ['trees / ha', 'basal area', 'QMD']); ax.set_ylabel('median |log error| (lower is better)')
ax.set_title(f"held-out visits 2022–2025 (n = {M['n']})"); ax.legend(loc='upper right', frameon=False)
ax.set_ylim(0, max(max(r[1]) for r in rows) * 1.25)
acc = [('generative', A['dominant_acc_A'], GREEN), ('kNN', A['dominant_acc_knn'], SKY)]
ax2.bar([a[0] for a in acc], [a[1] for a in acc], color=[a[2] for a in acc])
for i, a in enumerate(acc): ax2.text(i, a[1] + 0.01, f'{a[1]:.2f}', ha='center', va='bottom', fontsize=10)
ax2.set_ylim(0, 0.6); ax2.set_ylabel('dominant-species accuracy'); ax2.set_title(f"all held-out (n = {A['n']})")
fig.tight_layout(); save(fig, 'fig_genveg_results')
