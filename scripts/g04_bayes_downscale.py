"""Generative vegetation v3 — Bayesian downscaling (user choice 2026-10-20: options 2 + 3; true FIA coordinates not available).
  A  landscape prior p(z | L, year): the plot is a random draw from the forest of its 1.6 km landscape (the fuzz disk), described as a whole (mean AlphaEarth
     2022/2025, 32-cluster histogram of AlphaEarth 2025 footprints, disturbance-class fractions by period, Hansen loss fractions, soils) — robust to the
     coordinate fuzzing. z = the v2 vegetation latent (encoder of g03b, frozen); p(z | ·) = mixture-density network (8 diagonal Gaussians).
  B  forward likelihood p(AlphaEarth | z, disturbance history): the v2 flow + jump + heads, fine-tuned ONLY on well-localised plots (FIA-dated events pin
     ≥ 50 % of the prior to ≤ 50 footprints); MIL restricted to those footprints (nearly supervised).
  C  mapping: g05_map_stands.py (stand landscapes on a 1 km grid, consistent with the plots' disks).
Evaluation (spatial blocks of g03b):
  A alone vs kNN on X_L (all held-out plots); A×B at the high-prior footprints of held-out localised plots vs A alone vs kNN; localisation skill of B.
kNN baselines use training plots' last visits within ±3 years of the query visit (A is conditioned on the year too).
Writes data/proto/gafl/{bayes_eval.json, bayes_models.pt, bayes_kmeans.pkl}."""
import os, sys, json, math, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
os.environ.setdefault('DATA', 'gafl'); os.environ['LOAD'] = '1'; os.environ['EPOCHS'] = '0'
_argv = sys.argv; sys.argv = ['g03b']
SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'g03b_train_gafl.py')).read()
SRC = SRC[:SRC.index('# ---------------- evaluation on held-out plots')]          # reuse data, model classes, loaded v2 weights; skip its evaluation
exec(compile(SRC, 'g03b_train_gafl.py', 'exec'))
sys.argv = _argv; torch.manual_seed(1); np.random.seed(1)
# decoded summaries (from g03b's evaluation section)
def summaries(lp, val):
    """slots → plot summaries: TPH, BA (m²/ha), QMD (cm), share of the dominant key — EXPECTED values (slot density x calibrated presence probability)."""
    pr = torch.sigmoid(calibrated(lp)); N = torch.exp(val[..., 0].clamp(max=12)) * pr; Dq = torch.exp(val[..., 1].clamp(max=math.log(150.0)))   # QMD cap 150 cm
    tph = N.sum(1); ba = (N * math.pi / 4 * (Dq / 100) ** 2).sum(1); qmd = torch.sqrt((N * Dq ** 2).sum(1) / tph.clamp(min=1e-6))
    bk = (N * Dq ** 2).view(-1, NK, NA).sum(-1); return tph, ba, qmd, bk.argmax(1), bk / bk.sum(1, keepdim=True).clamp(min=1e-9)
def obs_summ(v):
    m = slot_p[v]; N = torch.exp(slot_v[v][..., 0]) * m; Dq = torch.exp(slot_v[v][..., 1])
    tph = N.sum(1); ba = (N * math.pi / 4 * (Dq / 100) ** 2).sum(1); qmd = torch.sqrt((N * Dq ** 2).sum(1) / tph.clamp(min=1e-6))
    bk = (N * Dq ** 2).view(-1, NK, NA).sum(-1); return tph, ba, qmd, bk.argmax(1), bk / bk.sum(1, keepdim=True).clamp(min=1e-9)

t0 = time.time()
# ---------------- landscape descriptors
from sklearn.cluster import MiniBatchKMeans
samp = np.random.choice(len(AEF), 200000, replace=False); km = MiniBatchKMeans(32, random_state=0, n_init=3).fit(AEF[samp][:, 64:].numpy())
BINS = [(2000, 2009), (2010, 2016), (2017, 2021), (2022, 2025)]
MAP_Y0 = float(os.environ.get('MAP_Y0', 2021.0))       # the REQUESTED start year of the mapping path (g04 gates it; g05 maps with bayes_eval.json map_y0)
def landscape(F_aef, F_dist, F_soil, wt=None):
    """X_L from a set of footprints, weighted by inverse inclusion probability wt (None = uniform sample): mean AEF22/25, AEF25 cluster histogram,
    disturbance class + Hansen fractions per period, mean soils — unbiased whole-disk statistics."""
    w_ = torch.ones(len(F_aef)) if wt is None else torch.as_tensor(wt, dtype=torch.float32); w_ = w_ / w_.sum()
    lab = km.predict(F_aef[:, 64:].numpy()); hist = np.bincount(lab, weights=w_.numpy(), minlength=32)
    ev = []
    for a, b in BINS:
        sl = slice(a - YEARS[0], b - YEARS[0] + 1); d = F_dist[:, sl]                      # [n, years, NC+2]
        ev += list(((d[:, :, :NC].amax(1) > 0).float() * w_.unsqueeze(-1)).sum(0).numpy()) + [float(((d[:, :, NC + 1].amax(1) > 0).float() * w_).sum())]
    return np.r_[(F_aef * w_.unsqueeze(-1)).sum(0).numpy(), hist, ev, (F_soil * w_.unsqueeze(-1)).sum(0).numpy()].astype(np.float32)
XL = {}
IW = dict(zip(C.cid.values, C.incl_w.values)) if 'incl_w' in C.columns else None
for pid, (cid, pr) in cand_of.items(): XL[pid] = landscape(AEF[cid], DIST[cid], SOIL[cid], None if IW is None else [IW[c_] for c_ in cid])
XLm = np.stack(list(XL.values())); xmu, xsd = XLm.mean(0), XLm.std(0) + 1e-6
nx = lambda x: torch.tensor((x - xmu) / xsd, dtype=torch.float32)
print(f'landscape descriptors {XLm.shape} ({time.time()-t0:.0f}s)', flush=True)
# ---------------- A: MDN p(z | X_L, year) on frozen v2 latents
with torch.no_grad(): ZV = torch.cat([enc(torch.tensor(v), vyear[torch.tensor(v)]) for v in np.array_split(np.arange(nv), 50)])
GM = 8
class MDN(nn.Module):
    def __init__(s): super().__init__(); s.f = nn.Sequential(nn.Linear(XLm.shape[1] + 1, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, GM * (1 + 2 * D_Z)))
    def forward(s, x, yr):
        o = s.f(torch.cat([x, ((yr - 2012) / 10).unsqueeze(-1)], -1)); lw = o[:, :GM].log_softmax(-1)
        mu = o[:, GM:GM + GM * D_Z].view(-1, GM, D_Z); ls = o[:, GM + GM * D_Z:].view(-1, GM, D_Z).clamp(-4, 1); return lw, mu, ls
    def nll(s, x, yr, z):
        lw, mu, ls = s(x, yr); lp = (-0.5 * ((z.unsqueeze(1) - mu) / torch.exp(ls)) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1)
        return -(torch.logsumexp(lw + lp, -1)).mean()
    def sample(s, x, yr, m):
        lw, mu, ls = s(x, yr); k = torch.multinomial(lw.exp(), m, replacement=True)               # [n, m]
        i = torch.arange(len(x)).unsqueeze(-1); return (mu[i, k] + torch.exp(ls[i, k]) * torch.randn(len(x), m, D_Z)).clamp(-3, 3)   # encoder range (3 tanh)
mdn = MDN(); om = torch.optim.AdamW(mdn.parameters(), lr=1e-3, weight_decay=1e-4)
Xv = torch.stack([nx(XL[vpid[v]]) for v in range(nv) if vpid[v] in XL]); vv = np.array([v for v in range(nv) if vpid[v] in XL]); vi = {v: i for i, v in enumerate(vv)}
# calibration plots: with VAL_SPLIT, the VALIDATION plots — never trained on by g03b (flow, decoder) nor by the MDN — so the calibration factors and the
# start-year gate are out of sample for the whole model (audit r8 I2); else (older models) 15 % of the TRAINING blocks, held out of the MDN fit only
if VAL_SPLIT and len(va_v): cal_pid = set(vpid[va_v]) & set(XL)
else:
    _trb = sorted(set(P.blk[P.pid.isin(set(vpid[tr_v]))])); cal_blk = set(np.random.default_rng(2).choice(_trb, max(1, int(0.15 * len(_trb))), replace=False))
    cal_pid = set(P.pid[P.blk.isin(cal_blk)]) & set(vpid[tr_v])
_ca = P[P.pid.isin(cal_pid)]; _dc, _ = _KD(_ca[['x', 'y']].values).query(P[['x', 'y']].values)
calbuf_pid = set(P.pid[(_dc < 3200) & ~P.pid.isin(cal_pid)])          # MDN-fit plots whose disk could overlap a calibration plot's: not fitted (r6 M7)
trA = np.array([vi[v] for v in tr_v if v in vi and vpid[v] not in cal_pid and vpid[v] not in calbuf_pid]); teA = np.array([vi[v] for v in te_v if v in vi])
print(f'MDN fit visits {len(trA)}, calibration plots {len(cal_pid)}', flush=True)
for ep in range(int(os.environ.get('MDN_EP', 60))):
    perm = np.random.permutation(trA)
    for b in range(0, len(perm), 256):
        i = torch.tensor(perm[b:b + 256]); l = mdn.nll(Xv[i], vyear[vv[i]], ZV[vv[i]]); om.zero_grad(); l.backward(); om.step()
with torch.no_grad(): print(f'MDN nll train {float(mdn.nll(Xv[trA], vyear[vv[trA]], ZV[vv[trA]])):.3f} test {float(mdn.nll(Xv[teA], vyear[vv[teA]], ZV[vv[teA]])):.3f} ({time.time()-t0:.0f}s)', flush=True)
# ---------------- B: forward likelihood fine-tuned on well-localised plots (MIL over the high-prior footprints only)
LS = pd.read_parquet(f'{D}/localisation.parquet'); loc_p = set(LS.index[(LS.n_hi > 0) & (LS.n_hi <= 50) & (LS.mass_hi >= 0.5)])
def hi_cands(pid):     # kept candidates inside the full disk's high-prior set (g02b), prior renormalised over them
    cid, pr = cand_of[pid]; m = cand_iw[pid][1]; return cid[m], pr[m] / pr[m].sum()
trB = np.array([v for v in tr_v if vpid[v] in loc_p]); teB = np.array([v for v in te_v if vpid[v] in loc_p]); print(f'localised visits train {len(trB)} test {len(teB)}', flush=True)
for p_ in enc.parameters(): p_.requires_grad_(False)
# B = a COPY of the v2 flow + heads, fine-tuned for the AlphaEarth likelihood only; the v2 flow stays the vegetation dynamics the decoder was trained on
import copy; flowB, h22B, h25B = copy.deepcopy(flow), copy.deepcopy(h22), copy.deepcopy(h25)
fparams = [*flowB.parameters(), *h22B.parameters(), *h25B.parameters()]; of = torch.optim.AdamW(fparams, lr=5e-4, weight_decay=1e-4)
def fwd_dist(z, start, cid, rs=None, ra=None):
    """squared AlphaEarth distance of the B-flowed latent at 2022 / 2025 for footprints cid (z broadcast to each), mean over the targets after the visit."""
    st = flowB(z, start, [2022, 2025], DIST[cid], SOIL[cid], rs=rs, ra=ra); d = torch.zeros(len(cid)); n_ = torch.zeros(len(cid))   # (rs: the visit's recorded own-year event, as MIL; r6 M6)
    for yr, head, sl in ((2022, h22B, A22), (2025, h25B, A25)):
        ok = (start <= yr + 1).float(); d = d + ok * ((head(st[yr]) - AEF[cid][:, sl]) ** 2).sum(-1); n_ = n_ + ok
    return d / n_.clamp(min=1), st
for ep in range(int(os.environ.get('EPB', 8))):
    tl = 0
    for v in np.random.permutation(trB):
        cid, pr = hi_cands(vpid[v]); z = ZV[[v]].repeat(len(cid), 1); d, _ = fwd_dist(z, vyear[[v]].repeat(len(cid)), torch.tensor(cid), rs=VREC[[v]].repeat(len(cid)), ra=VRA[[v]].repeat(len(cid)))
        l = -(torch.logsumexp(-d / TEMP + torch.log(torch.tensor(pr)), 0)) * TEMP; of.zero_grad(); l.backward(); of.step(); tl += float(l)
    print(f'B epoch {ep+1} loss {tl/len(trB):.4f} ({time.time()-t0:.0f}s)', flush=True)
torch.save(dict(mdn=mdn.state_dict(), flowB=flowB.state_dict(), h22B=h22B.state_dict(), h25B=h25B.state_dict(), xmu=xmu, xsd=xsd), f'{D}/bayes_models.pt')
import pickle; pickle.dump(km, open(f'{D}/bayes_kmeans.pkl', 'wb'))
# ---------------- evaluation
def summ(z):
    lp, val = dec(z); return summaries(lp, val)
ev = {}
with torch.no_grad():
    # localisation skill of B on held-out localised plots: posterior mass (uniform over ALL candidates) on the high-prior footprints vs their share
    loc = []
    for v in teB[:500]:
        cid, pr = cand_of[vpid[v]]; iw_, hi = cand_iw[vpid[v]]
        d, _ = fwd_dist(ZV[[v]].repeat(len(cid), 1), vyear[[v]].repeat(len(cid)), torch.tensor(cid), rs=VREC[[v]].repeat(len(cid)), ra=VRA[[v]].repeat(len(cid)))
        post = torch.softmax(-d / TEMP + torch.log(torch.tensor(iw_, dtype=torch.float32)), 0).numpy()     # uniform over the disk's footprints
        loc.append((post[hi].sum(), iw_[hi].sum() / iw_.sum()))
    a = np.array(loc); ev['B_localisation'] = {'visits': len(a), 'posterior_mass_on_pinned_footprints': float(a[:, 0].mean()), 'their_share': float(a[:, 1].mean())}
    # A alone vs kNN on X_L (last visit of every held-out plot)
    lastv = V[fs_ok]; lastv = lastv.loc[lastv.groupby('pid').year.idxmax()]       # last visit passing the forest-share filter
    trp_ = set(vpid[tr_v])
    te_l = [v for v in lastv[lastv.pid.isin(test_pid)].vid.values if v in vi]; tr_l = [v for v in lastv[lastv.pid.isin(trp_)].vid.values if v in vi]   # kNN pool: training plots
    o_te = obs_summ(torch.tensor(te_l)); o_tr = obs_summ(torch.tensor(tr_l))
    X_te = Xv[[vi[v] for v in te_l]]; X_tr = Xv[[vi[v] for v in tr_l]]
    zs = mdn.sample(X_te, vyear[te_l], 32); S_ = summ(zs.reshape(-1, D_Z)); A_ = [S_[i].view(len(te_l), 32) for i in range(3)]
    predA = [torch.exp(torch.log(x + 1e-3).median(1).values) for x in A_]
    # post-hoc calibration estimated on training plots only (the decoder predicts log-medians; sums of N and N·D² are biased low)
    trs_v = [v for v in lastv[lastv.pid.isin(cal_pid)].vid.values if v in vi]           # calibration plots' last visits (not used by the MDN fit)
    zt = mdn.sample(Xv[[vi[v] for v in trs_v]], vyear[trs_v], 32); St = summ(zt.reshape(-1, D_Z)); o_ts = obs_summ(torch.tensor(trs_v))
    CAL = [float(torch.median(torch.log((o_ts[i] + 1e-3) / (torch.exp(torch.log(St[i].view(len(trs_v), 32) + 1e-3).median(1).values))))) for i in range(3)]
    print(f'calibration log-factors (TPH, BA, QMD) from {len(trs_v)} calibration plots:', [round(c, 3) for c in CAL], flush=True)
    predA_raw = predA; predA = [predA[i] * math.exp(CAL[i]) for i in range(3)]
    compA = S_[4].view(len(te_l), 32, NK).mean(1)
    y_tr = vyear[tr_l]
    def knn_idx(Xq, yq):     # 5 nearest training plots in X_L among those whose last visit is within ±3 years of the query visit
        dd = torch.cdist(Xq, X_tr); dd = dd + 1e9 * ((yq.unsqueeze(1) - y_tr.unsqueeze(0)).abs() > 3).float(); return dd.topk(5, largest=False).indices
    nn5 = knn_idx(X_te, vyear[te_l]); knn = [o_tr[i][nn5].float().median(1).values for i in range(3)]; compK = o_tr[4][nn5].mean(1)
    err = lambda p, o: float(torch.median(torch.abs(torch.log((p + 1e-3) / (o + 1e-3)))))
    bias = lambda p, o: float(torch.median(torch.log((p + 1e-3) / (o + 1e-3))))
    bray = lambda p, o: float((torch.abs(p - o).sum(1) / 2).mean())
    ev['calibration_log_factors_TPH_BA_QMD'] = CAL
    sk = o_te[0] > 0                                                                 # stocked visits (bare visits: no composition; reported separately)
    p_trl = np.array([vpid[v] for v in tr_l])
    def null_pred(yq, excl=None):       # NULL baseline: median of the training plots' last visits within ±3 years (no satellite / landscape information at all)
        m_ = ((yq.unsqueeze(1) - y_tr.unsqueeze(0)).abs() <= 3).float(); out_ = []         # excl: plots left out of the pool (the ones being judged; r6 M7)
        if excl is not None: m_ = m_ * torch.tensor(~np.isin(p_trl, list(excl)), dtype=torch.float32).unsqueeze(0)
        for i in range(3):
            v_ = o_tr[i].float().unsqueeze(0).expand(len(yq), -1).clone(); v_[m_ == 0] = float('nan'); out_.append(torch.nanmedian(v_, 1).values)
        return out_
    nul = null_pred(vyear[te_l]); ev['null_composition_bray'] = bray(o_tr[4][o_tr[0] > 0].mean(0, keepdim=True).expand(int(sk.sum()), -1), o_te[4][sk])
    ev['A_prior_vs_knn_all_heldout'] = {'n': len(te_l), 'n_stocked': int(sk.sum()), 'bare_pred_BA_median_A': float(predA[1][~sk].median()) if (~sk).any() else None,
                                        'bare_pred_BA_median_knn': float(knn[1][~sk].median()) if (~sk).any() else None,
                                        **{nm: {'A_calibrated': err(predA[i][sk], o_te[i][sk]), 'A_bias': bias(predA[i][sk], o_te[i][sk]), 'A_uncalibrated': err(predA_raw[i][sk], o_te[i][sk]),
                                                'knn': err(knn[i][sk], o_te[i][sk]), 'knn_bias': bias(knn[i][sk], o_te[i][sk]), 'null': err(nul[i][sk], o_te[i][sk])} for i, nm in enumerate(['TPH', 'BA', 'QMD'])},
                                        'bray_A': bray(compA[sk], o_te[4][sk]), 'bray_knn': bray(compK[sk], o_te[4][sk]),
                                        'dominant_acc_A': float((compA[sk].argmax(1) == o_te[3][sk]).float().mean()), 'dominant_acc_knn': float((compK[sk].argmax(1) == o_te[3][sk]).float().mean())}
    # A x B at the pinned footprints of held-out localised plots (location nearly known): posterior-resampled prior draws vs A alone
    te_loc = [v for v in te_l if vpid[v] in loc_p]
    if te_loc:
        # at each held-out localised visit: A = prior samples AT THE VISIT YEAR, decoded at the visit; AB = the same samples resampled by the AlphaEarth match
        # of their forward flow (visit year -> 2022 / 2025, through each pinned footprint's history; targets after the visit only)
        MS = 64; res = {'A': [], 'AB': []}; used = []; ess = []
        for v in te_loc:
            yv = float(vyear[v])
            if yv > 2025.5: continue
            cid, pr = hi_cands(vpid[v]); zs = mdn.sample(Xv[[vi[v]]], torch.tensor([yv]), MS)[0]
            Dm = torch.stack([fwd_dist(zs, torch.full((MS,), yv), torch.tensor([c_] * MS), rs=VREC[[v]].repeat(MS), ra=VRA[[v]].repeat(MS))[0] for c_ in cid])     # [footprints, MS]
            lw = torch.logsumexp(-Dm / TEMP + torch.log(torch.tensor(pr, dtype=torch.float32)).unsqueeze(1), 0)    # log Σ_c p_c exp(-d_c(z)/T): location mixture
            w = torch.softmax(lw, 0); idx = torch.multinomial(w, 32, replacement=True); ess.append(float(1 / (w ** 2).sum()))
            sA = summ(zs); sB = summ(zs[idx]); used.append(v)
            for nm, s_ in (('A', sA), ('AB', sB)): res[nm].append([torch.exp(torch.log(s_[i] + 1e-3).median()) * math.exp(CAL[i]) for i in range(3)] + [s_[4].mean(0)])
        te_loc = used
        o_l = obs_summ(torch.tensor(te_loc)); nn5 = knn_idx(Xv[[vi[v] for v in te_loc]], vyear[te_loc])
        kn = [o_tr[i][nn5].float().median(1).values for i in range(3)]; kc = o_tr[4][nn5].mean(1)
        sl_ = o_l[0] > 0; o_l = [x[sl_] for x in o_l]; kn = [x[sl_] for x in kn]; kc = kc[sl_]; res = {k_: [r for r, m_ in zip(rs, sl_.tolist()) if m_] for k_, rs in res.items()}
        out = {'n': len(te_loc), 'n_stocked': int(sl_.sum()), 'ESS_of_64_median': float(np.median(ess)), 'ESS_q10': float(np.quantile(ess, 0.1))}
        for nm in ('A', 'AB'):
            P_ = [torch.stack([r[i] for r in res[nm]]) for i in range(3)]; Cp = torch.stack([r[3] for r in res[nm]])
            out[nm] = {k: err(P_[i], o_l[i]) for i, k in enumerate(['TPH', 'BA', 'QMD'])} | {'bray': bray(Cp, o_l[4]), 'dominant_acc': float((Cp.argmax(1) == o_l[3]).float().mean())}
        nl_ = [x[sl_] for x in null_pred(vyear[te_loc])]; out['null'] = {k: err(nl_[i], o_l[i]) for i, k in enumerate(['TPH', 'BA', 'QMD'])}
        out['knn'] = {k: err(kn[i], o_l[i]) for i, k in enumerate(['TPH', 'BA', 'QMD'])} | {'bray': bray(kc, o_l[4]), 'dominant_acc': float((kc.argmax(1) == o_l[3]).float().mean())}
        ev['AxB_localised_heldout'] = out
    # MAP_Y0: the year the mapping path starts (z drawn from the landscape prior at MAP_Y0, then flowed through the footprint's own history; 2000 lets
    # the stand's whole 2000-2025 history act — audit r3 B1: from 2021, a stand cut in 2016-20 was mapped as mature)
    # ---- the MAPPING path of g05, validated: z ~ p(z | X_L, MAP_Y0) -> v2 flow (decoder-consistent) through a prior-drawn candidate's history to the visit
    # date -> decode. Calibration factors for THIS path from the calibration plots (visits 2022-2025), errors on held-out plots' visits 2022-2025.
    def map_path(vs, y0, ms=32, X=None):
        out_ = []
        for q_, v in enumerate(vs):
            cid, pr = cand_of[vpid[v]]; ci = torch.tensor(np.random.choice(cid, ms, replace=True, p=pr)); yv = vyear[v]
            xv = Xv[[vi[v]]] if X is None else X[q_:q_ + 1]
            zs = mdn.sample(xv, torch.tensor([y0]), ms)[0]; yy = min(int(math.floor(float(yv))), YEARS[-1])
            st = flow(zs, torch.full((ms,), y0), [yy], DIST[ci], SOIL[ci], end=yv.repeat(ms), re=VREC[[v]].repeat(ms)); s_ = summ(st[yy])
            out_.append([torch.exp(torch.log(s_[i] + 1e-3).median()) for i in range(3)] + [s_[4].mean(0)])
        return out_
    rec = lambda vs: [v for v in vs if 2022.0 <= float(vyear[v]) < 2026.0]
    cal_m = rec(trs_v); te_m = rec(te_l)
    if cal_m and te_m:
        oc = obs_summ(torch.tensor(cal_m)); skc = oc[0] > 0; ot = obs_summ(torch.tensor(te_m)); skt = ot[0] > 0; nm_ = null_pred(vyear[te_m])
        def evaluate(y0, cal=None, te=None):
            cal = cal_m if cal is None else cal; te = te_m if te is None else te
            occ = obs_summ(torch.tensor(cal)); skc_ = occ[0] > 0; ott = obs_summ(torch.tensor(te)); skt_ = ott[0] > 0
            pc = map_path(cal, y0); CF = [float(torch.median(torch.log((occ[i][skc_] + 1e-3) / torch.stack([r[i] for r in pc])[skc_]))) for i in range(3)]
            pt = map_path(te, y0); P_ = [torch.stack([r[i] for r in pt]) * math.exp(CF[i]) for i in range(3)]
            P_[2] = torch.where(P_[0] > 0, 100 * torch.sqrt(P_[1] / (P_[0] * math.pi / 4).clamp(min=1e-9)), torch.zeros_like(P_[0]))   # QMD as g05 maps it:
            # from the calibrated TPH / BA (GEN_QMD), not the samples' median QMD x its own factor (audit r8 M6)
            return CF, pt, P_, {nm: err(P_[i][skt_], ott[i][skt_]) for i, nm in enumerate(['TPH', 'BA', 'QMD'])}
        # PRE-REGISTERED gate of the mapping start year (audit r4 B1): MAP_Y0 (e.g. 2000) is used only if its TPH and BA errors are no worse than the 2021
        # path's and than the null's, AND the prediction still depends on the stand's landscape (BA predictions with the landscapes shuffled correlate
        # < 0.5 with the real ones); else 2021. The gate is decided on the CALIBRATION plots (cross-fitted: factors from one half of the plots, errors on
        # the other), never on the held-out plots it then reports (audit r5 M1). The chosen year is recorded; g05 maps with it.
        res = {2021.0: evaluate(2021.0)}; gate = {'requested': MAP_Y0, 'decided_on': 'calibration plots, 2-fold by plot'}
        if MAP_Y0 != 2021.0:
            res[MAP_Y0] = evaluate(MAP_Y0)
            hA = [v for v in cal_m if vpid[v] % 2 == 0]; hB = [v for v in cal_m if vpid[v] % 2 == 1]
            e0, e1, cs_ = {k: 0.0 for k in ('TPH', 'BA')}, {k: 0.0 for k in ('TPH', 'BA')}, []
            nulle = {k: 0.0 for k in ('TPH', 'BA')}
            for ca, cb in ((hA, hB), (hB, hA)):              # each half judged with factors from the other
                r0, r1 = evaluate(2021.0, ca, cb), evaluate(MAP_Y0, ca, cb); ob_ = obs_summ(torch.tensor(cb)); sk_ = ob_[0] > 0; nmb = null_pred(vyear[cb], excl=set(vpid[cb]))
                Xs = Xv[[vi[cb[i]] for i in np.random.default_rng(7).permutation(len(cb))]]
                ps_ = map_path(cb, MAP_Y0, X=Xs); ba_r = torch.log(torch.stack([r[1] for r in r1[1]]) + 1e-3); ba_s = torch.log(torch.stack([r[1] for r in ps_]) + 1e-3)
                cs_.append(float(np.corrcoef(ba_r.numpy(), ba_s.numpy())[0, 1]))
                for i, k in ((0, 'TPH'), (1, 'BA')):
                    e0[k] += r0[3][k] / 2; e1[k] += r1[3][k] / 2; nulle[k] += err(nmb[i][sk_], ob_[i][sk_]) / 2
            cs = float(np.mean(cs_))
            ok = all(e1[k] <= e0[k] and e1[k] <= nulle[k] for k in ('TPH', 'BA')) and cs < 0.5
            gate.update({'err_2021': e0, 'err_requested': e1, 'err_null': nulle, 'shuffled_landscape_corr_BA': cs, 'passed': bool(ok),
                         'heldout_err_2021': res[2021.0][3], 'heldout_err_requested': res[MAP_Y0][3]})     # (held-out: reported, not used)
        y0u = MAP_Y0 if gate.get('passed') else 2021.0; CALF, pt, Pm, _ = res[y0u]; Cm = torch.stack([r[3] for r in pt])
        ev['map_y0'] = y0u; ev['map_y0_gate'] = gate
        # VALIDATION metrics of the chosen mapping path (cross-fitted on the calibration = validation plots: factors from one half, errors on the other) —
        # model choices (v3 vs v4, seeds) are made on these; the held-out block below is the test report (r9)
        hA = [v for v in cal_m if vpid[v] % 2 == 0]; hB = [v for v in cal_m if vpid[v] % 2 == 1]; xv_ = {k: 0.0 for k in ('TPH', 'BA', 'QMD')}
        for ca, cb in ((hA, hB), (hB, hA)):
            r_ = evaluate(y0u, ca, cb)
            for k in xv_: xv_[k] += r_[3][k] / 2
        ev['mapping_path_validation_crossfit'] = {'map_y0': y0u, 'n': len(cal_m), 'median_abs_log_err': xv_, 'set': 'validation plots' if VAL_SPLIT else 'calibration blocks (training)'}
        nnm = knn_idx(Xv[[vi[v] for v in te_m]], vyear[te_m]); km_ = [o_tr[i][nnm].float().median(1).values for i in range(3)]
        ev['calibration_log_factors_mapping_path'] = CALF
        ev['mapping_path_heldout_2022_2025'] = {'n': len(te_m), 'n_stocked': int(skt.sum()), 'n_calibration_visits': len(cal_m), 'map_y0': y0u,
            **{nm: {'map_path': err(Pm[i][skt], ot[i][skt]), 'map_bias': bias(Pm[i][skt], ot[i][skt]), 'knn': err(km_[i][skt], ot[i][skt]), 'null': err(nm_[i][skt], ot[i][skt])} for i, nm in enumerate(['TPH', 'BA', 'QMD'])},
            'bray_map': bray(Cm[skt], ot[4][skt]), 'bray_knn': bray(o_tr[4][nnm].mean(1)[skt], ot[4][skt])}
print(json.dumps(ev, indent=1)); json.dump(ev, open(f'{D}/bayes_eval.json', 'w'), indent=1)
