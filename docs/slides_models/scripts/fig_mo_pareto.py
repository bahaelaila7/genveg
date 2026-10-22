import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sstyle import *
from sdata import *
L = WS / 'landis_mogd' / 'runs_local'
def parse(p):
    t = open(p).read(); rows = re.findall(r'\[(\d)/6\] cand \d+\s+pass±5%=(\d+)/(\d+)\s+pass±10%=(\d+)/\d+\s+pass±20%=(\d+)/\d+\s+median\|bias\|=([\d.]+)%', t)
    return [(int(p5), int(p10), int(p20), float(m), int(ng)) for _, p5, ng, p10, p20, m in rows]
fig, axs = plt.subplots(2, 2, figsize=SLIDE_FULL, gridspec_kw={'width_ratios': [1, 1.1]})
rc = {'min_A': GREEN, 'knee': AMBER, 'min_W': RUST}
for r, (k, lab, ref) in enumerate([('bs', 'Biomass Succession', 'bs'), ('hbad', 'HBAD v1', 'hbad')]):
    a = pd.read_csv(L / ('mogd_%s_legacyW' % k) / 'mo_archive.csv'); ax = axs[r, 0]
    ax.plot(a.sumA, a.sumW, 'o', color=MUTED, ms=4, alpha=.6, label='archive')
    for role, c in rc.items():
        s = a[a.role.fillna('').str.contains(role)]
        ax.plot(s.sumA, s.sumW, 'o', ms=10, color=c, mec=INK, label=role.replace('_', ' '))
    ax.set_xlabel('Σ mean |log AGB error| (A)'); ax.set_ylabel('Σ Wasserstein (W)'); ax.set_title(lab, fontsize=12); ax.legend(fontsize=7)
    print(k, a[a.role.notna()][['sumA', 'sumW', 'role']].to_string())
    ax = axs[r, 1]; v = parse(L / ('mogd_%s_legacyW_val.log' % k)); print(k, 'val', v)
    x = np.arange(len(v)); w = .27
    for j, (t, c) in enumerate(zip((0, 1, 2), (.45, .7, 1.))):
        ax.bar(x + (j - 1) * w, [q[t] for q in v], w * .95, color=MODEL_COLORS[lab], alpha=c, label=['±5%', '±10%', '±20%'][j])
    fd = folds(ref)['val']; pc = [int(fd['pass_%d' % t].sum()) for t in (5, 10, 20)]
    for j, t in enumerate(pc): ax.hlines(t, -.5 + 0, len(v) - .5, colors=INK, linestyles=['dotted', 'dashed', 'solid'][j], lw=1.2, label='TOST-calibrated (val): %d' % t)
    print(k, 'TOST val fold passes 5/10/20', pc, 'of', len(fd))
    ax.set_xticks(x); ax.set_xticklabels(['cand %d' % (i + 1) for i in x], fontsize=8); ax.set_ylabel('val groups passing'); ax.set_ylim(0, 62)
    ax.legend(fontsize=6.5, ncol=2, loc='upper right')
fig.tight_layout(); save(fig, 'fig_mo_pareto')
