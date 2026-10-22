"""Shared figure style for the slides_models deck: `from sstyle import *`. Reuses the PERSEUS deck's matplotlib style and fonts.
Paths are derived from this file (WORKSPACE env overrides): scripts -> slides_models -> docs -> standmap -> <workspace>."""
import os, matplotlib
from pathlib import Path
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
DECK = HERE.parent
WS = Path(os.environ.get('WORKSPACE') or HERE.parents[3])
PSCRIPTS = WS / 'PERSEUS_PRESENTATION' / 'scripts'
for f in os.listdir(PSCRIPTS / 'fonts'):
    font_manager.fontManager.addfont(str(PSCRIPTS / 'fonts' / f))
plt.style.use(str(PSCRIPTS / 'perseus.mplstyle'))
NEW = DECK / 'figures' / 'new'
GREEN, AMBER, SKY, RUST, MOSS, BARK, MUTED, INK, TINT = '#1F4E3D', '#D08A12', '#2F6F9F', '#B5462B', '#5E8C31', '#7A5C3E', '#6B726E', '#1E2421', '#E9EFE6'
MODEL_COLORS = {'Biomass Succession': GREEN, 'PnET': SKY, 'HBAD v1': AMBER, 'Sim B': RUST, 'HBAD3P': BARK}
SLIDE_FULL = (11.0, 5.2)    # inches, full-width figure on a 16:9 slide
SLIDE_HALF = (5.4, 4.6)     # half-width
def save(fig, name):
    NEW.mkdir(parents=True, exist_ok=True)
    p = NEW / (name + '.pdf'); fig.savefig(p); fig.savefig(NEW / (name + '.png'), dpi=110)
    plt.close(fig); print('wrote', p)
