import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sstyle import *
from stand_crop import *
from matplotlib.patches import Patch
rgb = aef_rgb(); st = rd(RUN / 'finalize' / 'stands.tif'); land = rd(RUN / 'clean' / 'land_clean.tif')
cols = {1: GREEN, 2: RUST, 3: AMBER, 4: SKY, 0: '#D9D9D6', 9: '#D9D9D6'}
img = np.zeros((H, W, 3)); 
for k, c in cols.items(): img[land == k] = matplotlib.colors.to_rgb(c)
fig, ax = plt.subplots(1, 2, figsize=SLIDE_FULL)
show_rgb(ax[0], rgb); draw_edges(ax[0], st, 'white'); setup(ax[0]); scalebar(ax[0])
ax[0].set_title('AlphaEarth 2025 (PC 1-3)', fontsize=18)
show_rgb(ax[1], img); draw_edges(ax[1], st, 'black', .8); setup(ax[1]); scalebar(ax[1], color='white')
ax[1].set_title('Stands by land status', fontsize=18)
ax[1].legend(handles=[Patch(color=cols[k], label=l) for k, l in [(1, 'forest'), (2, 'harvested'), (3, 'regenerating'), (0, 'other')]],
             loc='upper center', bbox_to_anchor=(.5, -0.02), ncol=4, fontsize=13, frameon=False, handlelength=1, columnspacing=1)
fig.tight_layout(); save(fig, 'stands_map')
