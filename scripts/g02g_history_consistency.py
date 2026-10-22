"""History-consistent candidates per plot (user, 2026-10-21: what coinciding FIA / LANDFIRE records establish holds for ALL of the plot's visits).
A candidate footprint is consistent with the plot's WHOLE FIA history (g02e) when
  - every detectable FIA event cluster (g02b.clusters, likelihood ratio >= 4) has compatible LANDFIRE / Hansen evidence in its window, and
  - no FIA-undisturbed period (g02e negs) shows stand-replacing evidence (harvest class or Hansen loss on >= half the footprint);
applied sequentially in time order, skipping a constraint that no remaining footprint satisfies (2026-10-22).
g03b (LOC=history, default) draws every visit of the plot — MIL AlphaEarth matches, remeasurement pairs, Q targets — from the prior restricted to this
set; if it is empty (FIA / LANDFIRE disagree: missed detections, timing, swapped plots) the plot keeps its prior.
NOT a localisation claim: g02f's held-out test (2026-10-21) found ~0.003 nats / plot of footprint-level location information beyond a permutation
placebo — the set constrains which disturbance HISTORIES the plot can have, which is what the dynamics need.
Writes data/proto/gafl/cand_hist.parquet (cid, pid, cons) and localisation.parquet (n_hi = consistent footprints, mass_hi = their prior mass)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g02b_build_gafl as G2
from features import ROOT
D = f"{ROOT}/data/proto/{os.environ.get('DATA', 'gafl')}"; NY, NC = len(G2.YEARS), G2.NC
C = pd.read_parquet(f'{D}/cands.parquet'); CF = np.load(f'{D}/cand_feat.npy', mmap_mode='r'); P = pd.read_parquet(f'{D}/plots.parquet').set_index('pid')
E = pd.read_parquet(f'{D}/fia_events.parquet'); N = pd.read_parquet(f'{D}/fia_negs.parquet')
ev = {p: list(zip(g.kind, g.year.astype(int))) for p, g in E.groupby('pid')}; ng = {p: list(zip(g.y_lo.astype(int), g.y_hi.astype(int))) for p, g in N.groupby('pid')}
def consistent(per, evs, negs):
    """sequential narrowing in time order (detectable event clusters and undisturbed periods); a constraint that would leave no footprint is
    skipped (a missed detection / FIA-LANDFIRE disagreement must not erase the plot's whole history) — the same rule as g03b's pairs"""
    m = np.ones(len(per), bool); ncl = 0; sr = np.maximum(per[:, :, G2.IX['harvest']], per[:, :, G2.H_]) >= 0.5; cons_ = []
    for c in G2.clusters(evs):
        if c['lr'] < 4: continue
        lo, hi = max(c['lo'] - G2.YEARS[0], 0), min(c['hi'] - G2.YEARS[0], NY - 1) + 1
        if hi > lo: cons_.append((c['lo'], (per[:, lo:hi][:, :, sorted(c['cols'])].sum(-1) > 0).any(1))); ncl += 1
    for a, b in negs:
        a2, b2 = max(a - G2.YEARS[0], 0), min(b - G2.YEARS[0], NY - 1)
        if b2 >= a2: cons_.append((a, ~sr[:, a2:b2 + 1].any(1)))
    for _, x in sorted(cons_, key=lambda t: t[0]):
        if (m & x).any(): m = m & x
    return m, ncl
if __name__ == '__main__':
    out, loc = [], []
    for pid, g in C.sort_values('cid').groupby('pid'):
        per = np.asarray(CF[g.cid.values], dtype=np.float32)[:, 128:128 + NY * (NC + 2)].reshape(len(g), NY, NC + 2); iw = g.incl_w.values
        m, ncl = consistent(per, ev.get(pid, []), ng.get(pid, []))
        out.append(pd.DataFrame({'cid': g.cid.values, 'pid': pid, 'cons': m}))           # pid with cid: consumers check the file belongs to THIS table
        loc.append((pid, float(iw[m].sum()), float(g.prior.values[m].sum()), float(iw.sum()), ncl))
    pd.concat(out, ignore_index=True).to_parquet(f'{D}/cand_hist.parquet')
    L = pd.DataFrame(loc, columns=['pid', 'n_hi', 'mass_hi', 'n', 'n_clusters']).set_index('pid').join(P[['st', 'u', 'c', 'p']])
    L['mass_hi'] = np.where(L.n_hi > 0, 1.0, 0.0)            # all mass on the consistent set when it exists (g03b LOC=history)
    L.to_parquet(f'{D}/localisation.parquet')
    for nm, s_ in (('all', L), ('>=1 detectable cluster', L[L.n_clusters > 0]), ('>=2', L[L.n_clusters > 1])):
        print(f'{nm:24s} n {len(s_):5d}: consistent footprints median {s_.n_hi.median():5.0f} of {s_.n.median():.0f} | empty {np.mean(s_.n_hi == 0):.3f} | <=50 {np.mean((s_.n_hi > 0) & (s_.n_hi <= 50)):.3f}')
