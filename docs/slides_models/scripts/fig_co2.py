"""Atmospheric CO2 used by PnET: NOAA Mauna Loa annual means (observed, the simulator's co2_annual.csv) + the SSP2-4.5 / SSP5-8.5 global-mean
concentration pathways of RCMIP v5.1.0 (Meinshausen et al. 2020), decadal values linearly interpolated (pan_climate README; ClimateScenario.co2_series)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sstyle import *
import pandas as pd, numpy as np
obs = pd.read_csv(WS / 'landis_region4' / 'data_r4' / 'co2_annual.csv')
yrs = [2015, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100, 2110, 2120, 2130, 2140, 2150]
s245 = [399.9, 414.4, 444.1, 475.3, 506.9, 537.1, 563.9, 585.1, 597.7, 602.8, 605.7, 610.1, 615.2, 620.6, 626.3]
s585 = [399.9, 414.9, 451.5, 500.0, 562.8, 643.0, 743.7, 864.4, 998.3, 1135.2, 1269.3, 1397.6, 1519.0, 1632.5, 1737.3]
L = int(obs.year.max()); fy = np.arange(L + 1, 2151)                       # observed years win (co2_series); SSP from the year after
fig, ax = plt.subplots(figsize=SLIDE_FULL)
ax.plot(obs.year, obs.co2_ppm, color=INK, lw=2.2, label=f'NOAA Mauna Loa, observed {int(obs.year.min())}–{L}')
ax.plot(fy, np.interp(fy, yrs, s245), color=SKY, lw=2.2, label='SSP2-4.5 (RCMIP)')
ax.plot(fy, np.interp(fy, yrs, s585), color=RUST, lw=2.2, label='SSP5-8.5 (RCMIP)')
ax.plot([L + 1, 2150], [obs.co2_ppm.iloc[-1]] * 2, '--', color=MUTED, lw=1.5, label=f'constant at {L} ({obs.co2_ppm.iloc[-1]:.0f} ppm)')
ax.plot(yrs, s245, 'o', color=SKY, ms=4); ax.plot(yrs, s585, 'o', color=RUST, ms=4)
for y_, v_, c_ in ((2100, 602.8, SKY), (2100, 1135.2, RUST), (2150, 626.3, SKY), (2150, 1737.3, RUST)):
    ax.annotate(f'{v_:,.0f}', (y_, v_), textcoords='offset points', xytext=(-4, 8), ha='right', fontsize=10, color=c_)
ax.set_xlim(1955, 2155); ax.set_ylim(280, 1850); ax.set_ylabel('atmospheric CO$_2$ (ppm)'); ax.legend(loc='upper left', frameon=False)
fig.tight_layout(); save(fig, 'fig_co2')
