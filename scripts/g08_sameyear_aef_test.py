"""Does SAME-YEAR AlphaEarth at the candidate footprints carry location information about an FIA visit? (2026-10-22; the FIA records carry
~0.003 nats / plot, g02f; the forward likelihood B through long flows is at chance.)
Visits of the g02g-localised plots (<= 50 history-consistent footprints) measured 2017-2024: the AlphaEarth embedding of the VISIT YEAR at each
consistent candidate (3x3 mean, de-quantised) — 2022 from the mosaic (candidate features), other years from the targeted windows of g02c.
Model: frozen encoder z(visit) (g03b model.pt) → small head → AlphaEarth(visit year); MIL over the plot's consistent candidates,
loss = −T logsumexp_c(−d_c / T + log p_c), trained on TRAINING plots only (g03b spatial blocks + buffer).
Test on held-out plots: per visit, log-likelihood gain  log Σ_c p_c exp(−d_c(z_real)/T) − log Σ_c p_c exp(−d_c(z_placebo)/T), z_placebo = the
latent of a random other held-out visit of the same year (permutation placebo). Mean gain > 0 (with a bootstrap CI) = location information.
Also: posterior concentration (effective footprints) real vs placebo. Writes data/proto/gafl/sameyear_aef_test.json."""
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
# georeferencing check of the targeted windows: AlphaEarth is stable year to year in undisturbed forest -> window (2021 / 2023) vs the 2022 mosaic at the
# same candidates should agree far better than at candidates shifted by 3 footprints
cs_ok, cs_sh = [], []
for r in rows:
    if r[5] not in (2021, 2023): continue
    a22 = AEF[r[2]][:, :64].numpy(); A = r[4].numpy(); n_ = len(A)
    if n_ < 8: continue
    nrm = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    cs_ok.append(float(np.median((nrm(A) * nrm(a22)).sum(1)))); cs_sh.append(float(np.median((nrm(A) * nrm(np.roll(a22, 3, axis=0))).sum(1))))
print(f'window georeferencing: median cosine to the 2022 mosaic, same candidates {np.median(cs_ok):.3f} vs shuffled candidates {np.median(cs_sh):.3f} ({len(cs_ok)} visits)', flush=True)
tr_p = set(vpid[tr_v]); te_p = test_pid
TR = [r for r in rows if r[1] in tr_p]; TE = [r for r in rows if r[1] in te_p]
print(f'visits with same-year AlphaEarth: train {len(TR)}, held-out {len(TE)}', flush=True)
with torch.no_grad(): ZV = {r[0]: enc(torch.tensor([r[0]]), vyear[[r[0]]])[0] for r in rows}
head = nn.Sequential(nn.Linear(D_Z + 1, 64), nn.GELU(), nn.Linear(64, 64)); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
def hz(z, y): return head(torch.cat([z, torch.tensor([(y - 2020) / 4.0])]))
def lse(z, r): d = ((hz(z, r[5]).unsqueeze(0) - r[4]) ** 2).sum(-1); return torch.logsumexp(-d / TEMP + torch.log(torch.tensor(r[3], dtype=torch.float32)), 0)
for ep in range(int(os.environ.get('EP8', 20))):
    tot = 0.0
    for i in np.random.permutation(len(TR)):
        r = TR[i]; l = -lse(ZV[r[0]], r) * TEMP; opt.zero_grad(); l.backward(); opt.step(); tot += float(l)
    if ep % 5 == 4: print(f'epoch {ep + 1} loss {tot / max(len(TR), 1):.4f}', flush=True)
gains, ne_r, ne_p = [], [], []
with torch.no_grad():
    byy = {}
    for r in TE: byy.setdefault(r[5], []).append(r)
    for r in TE:
        others = [o for o in byy[r[5]] if o[1] != r[1]]
        if not others: continue
        zp = ZV[others[np.random.randint(len(others))][0]]
        lr_, lp_ = lse(ZV[r[0]], r), lse(zp, r); gains.append(float(lr_ - lp_))
        for z_, acc in ((ZV[r[0]], ne_r), (zp, ne_p)):
            d = ((hz(z_, r[5]).unsqueeze(0) - r[4]) ** 2).sum(-1); w = torch.softmax(-d / TEMP + torch.log(torch.tensor(r[3], dtype=torch.float32)), 0); acc.append(float(1 / (w ** 2).sum()))
# FOOTPRINT-level test: two visits (different years) of the same held-out plot must put their posterior on the SAME footprints more than chance
ov_r, ov_p = [], []
with torch.no_grad():
    byp = {}
    for r in TE: byp.setdefault(r[1], []).append(r)
    for pid, rs in byp.items():
        if len(rs) < 2: continue
        post = []
        for r in rs:
            d = ((hz(ZV[r[0]], r[5]).unsqueeze(0) - r[4]) ** 2).sum(-1); w = torch.softmax(-d / TEMP + torch.log(torch.tensor(r[3], dtype=torch.float32)), 0).numpy()
            post.append(dict(zip(r[2].tolist(), w)))
        for i in range(len(post)):
            for j in range(i + 1, len(post)):
                com = sorted(set(post[i]) & set(post[j]))
                if len(com) < 5: continue
                a1 = np.array([post[i][c] for c in com]); a2 = np.array([post[j][c] for c in com]); a1, a2 = a1 / a1.sum(), a2 / a2.sum()
                ov_r.append(float((a1 * a2).sum())); ov_p.append(float(np.mean([(a1 * np.random.permutation(a2)).sum() for _ in range(50)])))
g_ = np.array(gains); bs = [np.mean(np.random.choice(g_, len(g_))) for _ in range(2000)] if len(g_) else [np.nan]
out = dict(window_cosine_same=float(np.median(cs_ok)) if cs_ok else None, window_cosine_shuffled=float(np.median(cs_sh)) if cs_sh else None, n_train=len(TR), n_heldout=len(g_), mean_gain_nats=float(g_.mean()) if len(g_) else None, ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
           share_positive=float((g_ > 0).mean()) if len(g_) else None, eff_candidates_real=float(np.median(ne_r)) if ne_r else None, eff_candidates_placebo=float(np.median(ne_p)) if ne_p else None,
           cross_visit_pairs=len(ov_r), cross_visit_overlap_real=float(np.mean(ov_r)) if ov_r else None, cross_visit_overlap_permuted=float(np.mean(ov_p)) if ov_p else None,
           cross_visit_share_real_gt_perm=float(np.mean(np.array(ov_r) > np.array(ov_p))) if ov_r else None)
print(json.dumps(out, indent=1)); json.dump(out, open(f'{D}/sameyear_aef_test.json', 'w'), indent=1)
# ---- joint per-plot location posterior from ALL of the plot's same-year matches (TRAINING plots only: localising a held-out plot with its own
# vegetation would leak into the evaluation): log w_c = log p_c + Σ_visits (−d_cv / T), candidates common to the visits; written for g03b LOC=aef
if os.environ.get('WRITE_LOC') == '1':
    recs = []; ess = []
    with torch.no_grad():
        byp = {}
        for r in TR: byp.setdefault(r[1], []).append(r)
        for pid, rs in byp.items():
            lw = {}
            for r in rs:
                d = ((hz(ZV[r[0]], r[5]).unsqueeze(0) - r[4]) ** 2).sum(-1) / TEMP
                for c, dd, pp in zip(r[2].tolist(), d.tolist(), r[3].tolist()): lw[c] = lw.get(c, math.log(max(pp, 1e-300))) - dd
            cs = np.array(list(lw)); l_ = np.array([lw[c] for c in cs]); w = np.exp(l_ - l_.max()); w /= w.sum(); ess.append(1 / (w ** 2).sum())
            recs.append(pd.DataFrame({'pid': pid, 'cid': cs, 'prior_aef': w}))
    L8 = pd.concat(recs, ignore_index=True); L8.to_parquet(f'{D}/cand_loc_aef.parquet')
    print(f'joint AlphaEarth location posterior written for {L8.pid.nunique()} training plots; effective candidates median {np.median(ess):.1f}', flush=True)
