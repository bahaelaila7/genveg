"""Plot-level location posterior from the plot's WHOLE FIA history (user, 2026-10-21: a location matched by coinciding FIA / LANDFIRE disturbance
records holds for ALL of the plot's visits). One posterior per plot, shared by every visit (g03b: MIL draws, remeasurement pairs, Q target; g04 / g05).

Model (per plot; footprints c of the 1.6 km disk, each kept candidate standing for incl_w footprints — Horvitz-Thompson, valid for any prior):
  base_c   FIA fuzz: a share pi of plots lies uniformly within 800 m of the public point, the rest uniformly in the 800-1600 m ring
  LR_c     Π over the plot's FIA event clusters (g02b.clusters of g02e's events, all visits) of  q_g / b  if c shows compatible evidence in the
           window, (1 - q_g) / (1 - b) if not   (q_g = detection probability of group g = strong / medium / weak by g02b.EVK likelihood ratio;
           b = share of footprints with such evidence in the footprint's distance class, inner disk / ring)
           × Π over FIA-undisturbed periods (g02e negs) of  f / b0  if c shows stand-replacing evidence (harvest class or Hansen loss on >= half the
           footprint), (1 - f) / (1 - b0) if not
  outside  a share s of plots is not in its disk (swapped plots, conversions): records explain nothing there, LR = 1
  P(c | records) = (1 - r) δ_in + r · uniform-over-disk, r = posterior probability of 'outside'; δ_in ∝ base_c LR_c.
q_g, f, pi, s fitted by EM over all plots (location latent). Control: a permutation placebo (records of another plot of the same state).
Writes data/proto/gafl/{cand_prior.parquet (cid, prior_loc, hi_loc), location_em.json, localisation_em.parquet (credible-set size; g02g's localisation.parquet is left alone)}.
hi_loc = the candidates of the 80 % credible set; localisation.n_hi = its footprint count, mass_hi = posterior mass on it."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g02b_build_gafl as G2
from features import ROOT
D = f"{ROOT}/data/proto/{os.environ.get('DATA', 'gafl')}"; NY, NC = len(G2.YEARS), G2.NC; GRP = {8.0: 'strong', 4.0: 'medium', 3.0: 'weak'}; R_IN = 800.0
C = pd.read_parquet(f'{D}/cands.parquet'); CF = np.load(f'{D}/cand_feat.npy', mmap_mode='r'); P = pd.read_parquet(f'{D}/plots.parquet').set_index('pid')
E = pd.read_parquet(f'{D}/fia_events.parquet'); N = pd.read_parquet(f'{D}/fia_negs.parquet')
ev = {p: list(zip(g.kind, g.year.astype(int))) for p, g in E.groupby('pid')}; ng = {p: list(zip(g.y_lo.astype(int), g.y_hi.astype(int))) for p, g in N.groupby('pid')}
order = C.sort_values('cid'); groups = list(order.groupby('pid'))
print(f'plots {len(groups)}; reading candidate evidence', flush=True)
PL = []                               # per plot: pid, cid, iw, inner flag, per (evidence) array
for pid, g in groups:
    per = np.asarray(CF[g.cid.values], dtype=np.float32)[:, 128:128 + NY * (NC + 2)].reshape(len(g), NY, NC + 2)
    PL.append((pid, g.cid.values, g.incl_w.values.astype(float), g.dist_m.values <= R_IN, per))
def bg(m, iw, inn):
    """background rate of the evidence PER DISTANCE CLASS (inner disk / ring) as a per-footprint array: disturbance is more common in the ring than near
    the public point, and a disk-wide rate lets the inner / ring split itself fit records unrelated to the location (permutation placebo)."""
    out = np.zeros(len(m)); tot = float((iw * m).sum() / iw.sum())
    if not 0 < tot < 1: return None
    for cls in (inn, ~inn):
        if cls.any(): out[cls] = min(max((iw[cls] * m[cls]).sum() / iw[cls].sum(), 0.5 / iw[cls].sum()), 1 - 1e-3)
    return out
def records(shift):
    out = []
    for pid, cid, iw, inn, per in PL:
        evl, ngl = [], []
        for c in G2.clusters([(k, y + shift) for k, y in ev.get(pid, [])]):
            lo, hi = max(c['lo'] - G2.YEARS[0], 0), min(c['hi'] - G2.YEARS[0], NY - 1) + 1
            if hi <= lo: continue
            m = (per[:, lo:hi][:, :, sorted(c['cols'])].sum(-1) > 0).any(1); b = bg(m, iw, inn)
            if b is not None: evl.append((GRP[c['lr']], m, b))
        sr = np.maximum(per[:, :, G2.IX['harvest']], per[:, :, G2.H_]) >= 0.5
        for a, bb in ng.get(pid, []):
            a, bb = max(a + shift - G2.YEARS[0], 0), min(bb + shift - G2.YEARS[0], NY - 1)
            if bb < a: continue
            m = sr[:, a:bb + 1].any(1); b0 = bg(m, iw, inn)
            if b0 is not None: ngl.append((m, b0))
        out.append((evl, ngl))
    return out
def estep(RC, th):
    q, f, pi, s = th['q'], th['f'], th['pi'], th['s']; res = []; ll = 0.0
    for (pid, cid, iw, inn, per), (evl, ngl) in zip(PL, RC):
        A_in, A_out = iw[inn].sum(), iw[~inn].sum()
        base = np.where(inn, pi / max(A_in, 1e-9), (1 - pi) / max(A_out, 1e-9)) * iw if A_out > 0 and A_in > 0 else iw / iw.sum()
        lw = np.log(np.maximum(base, 1e-300))
        for grp, m, b in evl: lw = lw + np.where(m, np.log(q[grp] / b), np.log((1 - q[grp]) / (1 - b)))
        for m, b0 in ngl: lw = lw + np.where(m, np.log(f / b0), np.log((1 - f) / (1 - b0)))
        mx = lw.max(); lin = mx + np.log(np.exp(lw - mx).sum()); post = np.exp(lw - lin)          # δ_in over candidates (sums to 1)
        lmix = np.logaddexp(np.log(1 - s) + lin, np.log(s)); r_out = float(np.exp(np.log(s) - lmix)); ll += lmix
        res.append((post, 1 - r_out, evl, ngl, inn, iw))
    return res, ll
def mstep(res):
    num = {k: 0.0 for k in GRP.values()}; den = {k: 0.0 for k in GRP.values()}; fn = fd = 0.0; pin = pw = 0.0; sout = 0.0
    for post, rin, evl, ngl, inn, iw in res:
        for grp, m, b in evl: num[grp] += rin * post[m].sum(); den[grp] += rin
        for m, b0 in ngl: fn += rin * post[m].sum(); fd += rin
        if inn.any() and (~inn).any(): pin += rin * post[inn].sum(); pw += rin
        sout += 1 - rin
    cl = lambda x: min(max(x, 1e-4), 1 - 1e-4)
    return dict(q={k: cl(num[k] / max(den[k], 1e-9)) for k in num}, f=cl(fn / max(fd, 1e-9)), pi=cl(pin / max(pw, 1e-9)), s=cl(sout / len(res)))
def fit(RC, iters=int(os.environ.get('EM_ITERS', 150)), tag=''):
    th = dict(q={k: 0.3 for k in GRP.values()}, f=0.1, pi=0.5, s=0.2)
    for it in range(iters):
        res, ll = estep(RC, th); th = mstep(res)
        if it % 25 == 0 or it == iters - 1: print(f'{tag} iter {it + 1}: log-lik vs no-information {ll:.1f} | q {({k: round(v, 3) for k, v in th["q"].items()})} f {th["f"]:.4f} pi {th["pi"]:.3f} s {th["s"]:.3f}', flush=True)
    res, ll = estep(RC, th); return th, res, ll
RC = records(0); th, res, ll = fit(RC, tag='real')
# control: PERMUTATION placebo — each disk gets the records of another plot (same state), which removes any relation to the location (a year shift
# is confounded: shifted records contradict the true footprint's real events and push mass AWAY from it). Reported: the real parameters scored on
# the permuted records, and a refit on them (the gain the model's flexibility alone can produce).
def records_perm(seed=7):
    rg = np.random.default_rng(seed); st_ = np.array([int(P.loc[pid].st) for pid, *_ in PL]); perm = np.arange(len(PL))
    for sv in np.unique(st_):
        ix_ = np.flatnonzero(st_ == sv); perm[ix_] = rg.permutation(ix_)
    global ev, ng
    ev0, ng0 = ev, ng; pids_ = [pl[0] for pl in PL]
    ev = {pids_[i]: ev0.get(pids_[perm[i]], []) for i in range(len(PL))}; ng = {pids_[i]: ng0.get(pids_[perm[i]], []) for i in range(len(PL))}
    out_ = records(0); ev, ng = ev0, ng0; return out_
RP_ = records_perm(); _, ll_perm_fixed = estep(RP_, th); _, _, ll_pl = fit(RP_, tag='permutation placebo')
print(f'permutation placebo: real parameters {ll_perm_fixed:.1f}, refit {ll_pl:.1f}  (real records {ll:.1f})', flush=True)
out = []; loc = []
for (pid, cid, iw, inn, per), (post, rin, evl, ngl, _, _) in zip(PL, res):
    u = iw / iw.sum(); p = rin * post + (1 - rin) * u                                   # candidate-level posterior (each candidate = iw footprints)
    o = np.argsort(-p / iw); cs = np.cumsum(p[o]); k = int(np.searchsorted(cs, 0.8) + 1); hi = np.zeros(len(p), bool); hi[o[:k]] = True
    out.append(pd.DataFrame({'cid': cid, 'prior_loc': p.astype(np.float32), 'hi_loc': hi}))
    ph = p / iw; neff = float(np.exp(-(p * np.log(np.maximum(ph, 1e-300))).sum()))
    loc.append((pid, float(iw[hi].sum()), float(p[hi].sum()), float(iw.sum()), rin, neff, len(evl), len(ngl)))
pd.concat(out, ignore_index=True).to_parquet(f'{D}/cand_prior.parquet')
L = pd.DataFrame(loc, columns=['pid', 'n_hi', 'mass_hi', 'n', 'p_in_disk', 'n_eff', 'n_clusters', 'n_negs']).set_index('pid').join(P[['st', 'u', 'c', 'p']])
L.to_parquet(f'{D}/localisation_em.parquet')          # NOT localisation.parquet (g02g's, read by g03b / g04 / g08)
json.dump(dict(theta=th, loglik_vs_noinfo=ll, loglik_permutation_placebo_refit=ll_pl, loglik_permutation_placebo_real_params=ll_perm_fixed, n_plots=len(PL), credible=0.8, inner_radius_m=R_IN), open(f'{D}/location_em.json', 'w'), indent=1)
print(f'real {ll:.1f} vs permutation placebo refit {ll_pl:.1f} / with real parameters {ll_perm_fixed:.1f}; theta {th}')
for nm, s_ in (('all', L), ('>=1 cluster', L[L.n_clusters > 0]), ('>=2', L[L.n_clusters > 1])):
    print(f'{nm:12s} n {len(s_):5d}: 80% credible set footprints median {s_.n_hi.median():5.0f} (of {s_.n.median():.0f}), effective {s_.n_eff.median():5.0f}, <=50: {np.mean(s_.n_hi <= 50):.3f}, P(in disk) median {s_.p_in_disk.median():.2f}')
