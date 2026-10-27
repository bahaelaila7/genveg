"""Recompute the location prior of the KEPT candidates under the current matching rules (g02b.EVK / prior_weights) without rebuilding the table
(2026-10-22: strict data-derived rules). The stratified candidate sample and its inclusion weights incl_w are design-based and stay valid for any
prior: kept candidate c carries prior ∝ w_c · incl_w_c (w = g02b.prior_weights, unnormalised, baseline 1). Full-disk numbers are Horvitz-Thompson
estimates: n_hi = Σ incl_w over hi (w >= 4), mass_hi = their prior mass, conc = prior mass on the top 10 % of footprints, n_fp = Σ incl_w.
Rewrites cands.parquet (prior, hi) and plots.parquet (conc, n_hi, mass_hi, n_fp); backups *.bak_prior_<tag>. Run g02e first (records), then g02g.
LEGACY, REFUSES TO RUN (G51.6 / A15-7): it read g02b's module EVK instead of the table's, dropped the radial location prior and rewrote the table
without a new build id; rebuild the table with scripts/g02b_build_gafl.py instead."""
import os, sys, shutil, numpy as np, pandas as pd
sys.exit('g02i_reprior.py is legacy and refuses to run (docs/DECISIONS.md G51.6): rebuild the table with scripts/g02b_build_gafl.py')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g02b_build_gafl as G2
from features import ROOT
D = f"{ROOT}/data/proto/{os.environ.get('DATA', 'gafl')}"; NY, NC = len(G2.YEARS), G2.NC; tag = sys.argv[1] if len(sys.argv) > 1 else 'prev'
C = pd.read_parquet(f'{D}/cands.parquet'); P = pd.read_parquet(f'{D}/plots.parquet'); CF = np.load(f'{D}/cand_feat.npy', mmap_mode='r')
E = pd.read_parquet(f'{D}/fia_events.parquet'); N = pd.read_parquet(f'{D}/fia_negs.parquet')
ev = {p: list(zip(g.kind, g.year.astype(int))) for p, g in E.groupby('pid')}; ng = {p: list(zip(g.y_lo.astype(int), g.y_hi.astype(int))) for p, g in N.groupby('pid')}
for f in ('cands', 'plots'):
    if not os.path.exists(f'{D}/{f}.parquet.bak_prior_{tag}'): shutil.copy2(f'{D}/{f}.parquet', f'{D}/{f}.parquet.bak_prior_{tag}')
C = C.sort_values('cid').reset_index(drop=True); prior = np.zeros(len(C)); hi = np.zeros(len(C), bool); stats = {}
for pid, ix in C.groupby('pid').indices.items():
    g = C.iloc[ix]; per = np.asarray(CF[g.cid.values], dtype=np.float32)[:, 128:128 + NY * (NC + 2)].reshape(len(g), NY, NC + 2); iw = g.incl_w.values.astype(float)
    wr = G2.prior_weights(per, ev.get(pid, []), ng.get(pid, [])); m = wr * iw; p = m / m.sum(); prior[ix] = p; h = wr >= 4.0; hi[ix] = h
    o = np.argsort(-wr); cum = np.cumsum(iw[o]); top = o[:max(1, int(np.searchsorted(cum, 0.1 * iw.sum())) + 1)]
    stats[pid] = (float(p[top].sum()), float(iw[h].sum()), float(p[h].sum()), float(iw.sum()))
C['prior'] = prior.astype(np.float32); C['hi'] = hi; C.to_parquet(f'{D}/cands.parquet')
for j, nm in enumerate(['conc', 'n_hi', 'mass_hi', 'n_fp']): P[nm] = P.pid.map({k: v[j] for k, v in stats.items()})
P.to_parquet(f'{D}/plots.parquet')
print(f'prior recomputed for {len(stats)} plots under the current rules; plots with a high-prior footprint {sum(1 for v in stats.values() if v[1] > 0)}; '
      f'concentration quantiles {np.quantile([v[0] for v in stats.values()], [.1, .5, .9]).round(3)}')
