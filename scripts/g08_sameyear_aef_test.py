"""Does SAME-YEAR AlphaEarth at the candidate footprints carry location information about an FIA visit? (2026-10-22; the FIA records carry
~0.003 nats / plot, g02f; the forward likelihood B through long flows is at chance.)
Visits of the g02g-localised plots (<= 50 history-consistent footprints) measured 2017-2024: the AlphaEarth embedding of the VISIT YEAR at each
consistent candidate (3x3 mean, de-quantised) — 2022 from the mosaic (candidate features), other years from the targeted windows of g02c.
Model: frozen encoder z(visit) (g03b model.pt) → small head → AlphaEarth(visit year); MIL over the plot's consistent candidates,
loss = −T logsumexp_c(−d_c / T + log p_c), trained on TRAINING plots only (g03b spatial blocks + buffer).
Held-out tests (corrected 2026-10-22 after review: the earlier 'gain' and 'overlap vs permuted' did NOT test location):
  (1) plot <-> landscape fit: log Σ_c p_c exp(−d_c(z_real)/T) − the same with another plot's latent. Invariant to which candidate is true -> landscape
      information only (bootstrap by plot).
  (2) FOOTPRINT test: overlap (cosine similarity, insensitive to sharpness) of the candidate posteriors of two visits of a plot with its OWN latents minus the overlap with ANOTHER plot's two
      latents on the same candidates / AlphaEarth / years / prior (bootstrap by plot). > 0 = footprint selection specific to the plot's vegetation. Writes data/proto/gafl/sameyear_aef_test.json."""
import os, sys, json, math, numpy as np, pandas as pd, torch, torch.nn as nn
os.environ.setdefault('DATA', 'gafl'); os.environ['LOAD'] = '1'; os.environ['EPOCHS'] = '0'
_argv = sys.argv; sys.argv = ['g03b']
_HERE = os.path.dirname(os.path.abspath(__file__)); SRC = open(f'{_HERE}/g03b_train_gafl.py').read(); SRC = SRC[:SRC.index('# ---------------- evaluation on held-out plots')]
exec(compile(SRC, 'g03b_train_gafl.py', 'exec')); sys.argv = _argv
torch.manual_seed(0); np.random.seed(0); RP = int(1600 // 30) + 2
LS = pd.read_parquet(f'{D}/localisation.parquet'); locp = set(LS.index[(LS.n_hi > 0) & (LS.n_hi <= 50)])
Pk = P.set_index('pid'); land = __import__('rasterio').open(f'{ROOT}/data/derived/land_clean.tif')
def dq(a): x = a.astype(np.float32); o = np.sign(x) * (x / 127.5) ** 2; o[a == -128] = np.nan; return o
rows = []                                    # (vid, pid, cand ids, prior, AEF [n, 64])
for v in range(nv):
    pid = vpid[v]; y = int(math.floor(float(vyear[v])))
    if pid not in locp or pid not in cand_of or not 2017 <= y <= 2024: continue
    g = C[C.pid == pid]; g = g[g.cons] if 'cons' in g.columns and g.cons.any() else g
    if y == 2022: A = AEF[g.cid.values][:, :64].numpy()
    else:
        p = Pk.loc[pid]; fn = f'{D}/visit_aef/{int(p.st)}_{int(p.u)}_{int(p.c)}_{int(p.p)}_{y}.npy'
        if not os.path.exists(fn): continue
        W = dq(np.load(fn)); r0, c0 = land.index(p.x, p.y); rr, cc = g.row.values - (r0 - RP), g.col.values - (c0 - RP)
        A = np.stack([np.nanmean(W[:, r - 1:r + 2, c - 1:c + 2].reshape(64, -1), 1) for r, c in zip(rr, cc)]); A = np.nan_to_num(A)
    ok = np.abs(A).sum(1) > 0
    if ok.sum() < 2: continue
    pr = g.prior.values[ok]; rows.append((v, pid, g.cid.values[ok], pr / pr.sum(), torch.tensor(A[ok], dtype=torch.float32), y))
# georeferencing of the windows is checked at pixel precision by gv_verify T13 (the agreement with the 2022 mosaic peaks at offset 0 among ±2 px)
tr_p = set(vpid[tr_v]); te_p = test_pid
TR = [r for r in rows if r[1] in tr_p]; TE = [r for r in rows if r[1] in te_p]
print(f'visits with same-year AlphaEarth: train {len(TR)}, held-out {len(TE)}', flush=True)
with torch.no_grad(): ZV = {r[0]: enc(torch.tensor([r[0]]), vyear[[r[0]]])[0] for r in rows}
def new_head(): return nn.Sequential(nn.Linear(D_Z + 1, 64), nn.GELU(), nn.Linear(64, 64))
def hz(head, z, y): return head(torch.cat([z, torch.tensor([(y - 2020) / 4.0])]))
def dists(head, z, r): return ((hz(head, z, r[5]).unsqueeze(0) - r[4]) ** 2).sum(-1)
def lse(head, z, r): return torch.logsumexp(-dists(head, z, r) / TEMP + torch.log(torch.tensor(r[3], dtype=torch.float32)), 0)
def train_head(R_):
    head = new_head(); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(int(os.environ.get('EP8', 20))):
        for i in np.random.permutation(len(R_)):
            r = R_[i]; l = -lse(head, ZV[r[0]], r) * TEMP; opt.zero_grad(); l.backward(); opt.step()
    return head
head = train_head(TR)
def boot_by_plot(vals, pids, n=2000):         # visits of one plot are not independent: resample PLOTS
    byp = {}
    for v_, p_ in zip(vals, pids): byp.setdefault(p_, []).append(v_)
    ks = list(byp); bs = []
    for _ in range(n):
        pick = np.random.choice(len(ks), len(ks)); bs.append(np.mean([x for k in pick for x in byp[ks[k]]]))
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
out = {'n_train_visits': len(TR), 'n_heldout_visits': len(TE)}
with torch.no_grad():
    # (1) PLOT <-> LANDSCAPE fit (NOT location): real latent vs another held-out plot's latent (same year) over all of this plot's candidates.
    #     The mixture is invariant to which candidate is true, so this measures only how well the plot's vegetation fits its landscape's AlphaEarth.
    byy = {}
    for r in TE: byy.setdefault(r[5], []).append(r)
    g1, p1 = [], []
    for r in TE:
        others = [o for o in byy[r[5]] if o[1] != r[1]]
        if not others: continue
        zp = ZV[others[np.random.randint(len(others))][0]]; g1.append(float(lse(head, ZV[r[0]], r) - lse(head, zp, r))); p1.append(r[1])
    out['landscape_fit_gain_nats_per_visit'] = float(np.mean(g1)) if g1 else None; out['landscape_fit_gain_ci95_by_plot'] = boot_by_plot(g1, p1) if g1 else None
    # (2) FOOTPRINT test: two visits of a held-out plot A; overlap of their candidate posteriors with A's OWN latents vs with the two latents of another
    #     held-out plot B (same candidates, same AlphaEarth, same years, same prior; B's latents stable across its visits too). Only footprint
    #     selection specific to A's vegetation separates them.
    byp = {}
    for r in TE: byp.setdefault(r[1], []).append(r)
    multi = [p for p, rs in byp.items() if len(rs) >= 2]; d2, pp2 = [], []
    def post(z, r): return dict(zip(r[2].tolist(), torch.softmax(-dists(head, z, r) / TEMP + torch.log(torch.tensor(r[3], dtype=torch.float32)), 0).numpy()))
    def overlap(pa, pb):
        com = sorted(set(pa) & set(pb))
        if len(com) < 5: return None
        a1 = np.array([pa[c] for c in com]); a2 = np.array([pb[c] for c in com]); return float((a1 * a2).sum() / np.sqrt((a1 ** 2).sum() * (a2 ** 2).sum()))   # cosine: insensitive to sharpness
    for pA in multi:
        rs = byp[pA]; others = [q for q in multi if q != pA]
        if not others: continue
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                pB = others[np.random.randint(len(others))]; zB = [ZV[x[0]] for x in byp[pB][:2]]
                o_r = overlap(post(ZV[rs[i][0]], rs[i]), post(ZV[rs[j][0]], rs[j])); o_p = overlap(post(zB[0], rs[i]), post(zB[1], rs[j]))
                if o_r is None or o_p is None: continue
                d2.append(o_r - o_p); pp2.append(pA)
    out['footprint_pairs'] = len(d2); out['footprint_plots'] = len(set(pp2))
    out['footprint_overlap_real_minus_placebo'] = float(np.mean(d2)) if d2 else None; out['footprint_ci95_by_plot'] = boot_by_plot(d2, pp2) if d2 else None
print(json.dumps(out, indent=1)); json.dump(out, open(f'{D}/sameyear_aef_test.json', 'w'), indent=1)
# ---- joint per-plot location posterior (TRAINING plots only; CROSS-FITTED: the head that scores a plot never saw it), over the candidates COMMON to
# all of the plot's visits, log w_c = log p_c + (1 / n_visits) Σ_v (−d_cv / T)   (tempered: AlphaEarth is autocorrelated across years)
if os.environ.get('WRITE_LOC') == '1':
    trp = sorted({r[1] for r in TR}); rg = np.random.default_rng(5); fold = dict(zip(trp, rg.integers(0, 2, len(trp))))
    heads = {f: train_head([r for r in TR if fold[r[1]] != f]) for f in (0, 1)}; recs = []; ess = []
    with torch.no_grad():
        byp = {}
        for r in TR: byp.setdefault(r[1], []).append(r)
        for pid, rs in byp.items():
            hd = heads[fold[pid]]; com = set(rs[0][2].tolist())
            for r in rs[1:]: com &= set(r[2].tolist())
            if len(com) < 2: continue
            cs = np.array(sorted(com)); g = C[C.pid == pid].set_index('cid').loc[cs]; lw = np.log(np.maximum(g.prior.values / g.prior.values.sum(), 1e-300))
            acc = np.zeros(len(cs))
            for r in rs:
                ix = {c: k for k, c in enumerate(r[2].tolist())}; d = dists(hd, ZV[r[0]], r).numpy() / TEMP; acc -= np.array([d[ix[c]] for c in cs])
            lw = lw + acc / len(rs); w = np.exp(lw - lw.max()); w /= w.sum(); ess.append(1 / (w ** 2).sum())
            recs.append(pd.DataFrame({'pid': pid, 'cid': cs, 'prior_aef': w}))
    L8 = pd.concat(recs, ignore_index=True); L8.to_parquet(f'{D}/cand_loc_aef.parquet')
    print(f'joint AlphaEarth location posterior (cross-fitted, common candidates) for {L8.pid.nunique()} training plots; effective candidates median {np.median(ess):.1f}', flush=True)
