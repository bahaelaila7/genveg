"""One Region-4 EPA L4 ecoregion (8.3.5.65m, the pan_climate test fixture): annual means of tmin, tmax and precipitation — PRISM observed 1895-2024,
the baseline forecast 2025-2100 of pan_climate.duckdb (AutoETS tmin/tmax, 1995-2024 climatology for ppt), and an example user scenario applied with the
ClimateScenario rules (tmean +2 degC ramp 2025->2075 then held: shifts tmin and tmax; ppt -10 % ramp over the same years). Nothing is re-fitted."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sstyle import *
import duckdb, numpy as np
L4 = '8.3.5.65m'
c = duckdb.connect(str(WS / 'pan_climate' / 'pan_climate.duckdb'), read_only=True)
ob = c.sql(f"select var, year, avg(value) v from observed_monthly where l4='{L4}' and var in ('tmin','tmax','ppt') and year<=2024 group by 1,2 order by 1,2").df()
fc = c.sql(f"select var, year, avg(value) v from forecast_monthly where l4='{L4}' and var in ('tmin','tmax','ppt') and year between 2025 and 2100 group by 1,2 order by 1,2").df()
ramp = lambda y: np.clip((y - 2025) / (2075 - 2025), 0, 1)                    # ClimateScenario 'ramp': 0 at start_year, 1 at end_year, held after
fig, axes = plt.subplots(1, 2, figsize=SLIDE_FULL)
ax = axes[0]
for var, col in (('tmax', RUST), ('tmin', SKY)):
    o = ob[ob['var'] == var]; f = fc[fc['var'] == var]
    ax.plot(o.year, o.v, color=col, lw=1.2, label=f'{var} observed (PRISM)')
    ax.plot(f.year, f.v, color=col, lw=2.2, ls='--', label=f'{var} baseline forecast')
    ax.plot(f.year, f.v + 2.0 * ramp(f.year.values), color=col, lw=2.2, alpha=0.55, label=f'{var} scenario +2 °C ramp')
ax.set_ylabel('annual mean (°C)'); ax.legend(frameon=False, fontsize=8.5, ncol=2, loc='center', bbox_to_anchor=(0.5, 0.55)); ax.set_title(f'temperature, L4 {L4}')
ax = axes[1]; o = ob[ob['var'] == 'ppt']; f = fc[fc['var'] == 'ppt']
ax.plot(o.year, o.v * 12, color=GREEN, lw=1.2, label='observed (PRISM)')
ax.plot(f.year, f.v * 12, color=GREEN, lw=2.2, ls='--', label='baseline: 1995–2024 climatology')
ax.plot(f.year, f.v * 12 * (1 - 0.10 * ramp(f.year.values)), color=AMBER, lw=2.2, label='scenario −10 % ramp')
ax.set_ylabel('annual precipitation (mm)'); ax.legend(frameon=False, fontsize=9, loc='lower left'); ax.set_title('precipitation')
for a in axes: a.axvline(2024.5, color=MUTED, lw=0.8)
fig.tight_layout(); save(fig, 'fig_climate_l4')
