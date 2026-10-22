import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sstyle import *
from stand_crop import *
rgb = 0.35 * aef_rgb() + 0.65   # light background
L = RUN / 'levels'
fig, ax = plt.subplots(1, 3, figsize=SLIDE_FULL)
for a, (t, p, c) in zip(ax, [('Fine units', L / 'fine.tif', GREEN), ('Stands', RUN / 'finalize' / 'stands.tif', RUST), ('Compartments', L / 'compartments.tif', INK)]):
    lab = rd(p); show_rgb(a, rgb); draw_edges(a, lab, c); setup(a); scalebar(a, color=INK)
    n = len(np.unique(lab[lab > 0])); a.set_title(f'{t}\n({n} units in crop)', fontsize=17)
fig.tight_layout(); save(fig, 'stands_levels')
