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
Mapping path (the estimator g05 maps): start-year gate and calibration factors as means of R_REP replicates (G46.11); the map is ACCEPTED only if, on
validation, its calibrated TPH / BA errors and medoid Bray are below the null's (G46.12, bayes_eval.json map_accept; g05 refuses otherwise).
Writes data/proto/gafl/{bayes_eval.json, bayes_models.pt, bayes_kmeans.pkl}."""
import os, sys, json, math, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
os.environ.setdefault('DATA', 'gafl'); os.environ['LOAD'] = '1'; os.environ['EPOCHS'] = '0'
_argv = sys.argv; sys.argv = ['g03b']
SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'g03b_train_gafl.py')).read()
SRC = SRC[:SRC.index('# ---------------- evaluation on held-out plots')]          # reuse data, model classes, loaded v2 weights; skip its evaluation
exec(compile(SRC, 'g03b_train_gafl.py', 'exec'))
sys.argv = _argv; torch.manual_seed(1); np.random.seed(1)
# decoded summaries (from g03b's evaluation section)
def lmed(x):     # TRUE median in log space (torch.median returns the LOWER middle value: 2-5 % low at 16 vs 32 samples; r11 M-1)
    return torch.exp(torch.quantile(torch.log(x + 1e-3), 0.5, dim=-1))
def tmed(x, dim=None):     # TRUE median (G46.15; dim None: over all elements)
    return torch.quantile(x.float(), 0.5) if dim is None else torch.quantile(x.float(), 0.5, dim=dim)
MS = 32      # (G46.13) iid posterior draws per mapped stand / validated visit: g05 maps and g04 validates the same estimator
def calibrated_median(s_i, cf):
    """THE calibrated summary of g05 and g04's mapping-path validation (G46.13): e^CF · lmed(s) (the calibration factor applied to the true median of
    the draws, not the median of calibrated draws)"""
    return lmed(s_i) * math.exp(cf)
def medoid_draw(lp_, val_, s_, seed):
    """THE stand estimator of g05, shared with g04's validation (r13 C-1 / G32): the posterior draw closest to the median summary in log TPH / BA /
    QMD-from-TPH-BA (r13 F#12), its slots present by Bernoulli(calibrated p) seeded by `seed` (else its largest slot). Returns the draw index, the present
    slots, the slot arrays and the medoid's BA composition by key (shares are invariant to g05's uniform TPH / diameter scaling) + dominant key."""
    tph_m, ba_m = float(lmed(s_[0])), float(lmed(s_[1]))
    qd = 100 * torch.sqrt(s_[1].clamp(min=1e-9) / (s_[0].clamp(min=1e-9) * math.pi / 4)); qd_m = 100 * math.sqrt(ba_m / (tph_m * math.pi / 4)) if tph_m > 0 else 1e-6
    dist_ = (torch.log(s_[0].clamp(min=1e-6)) - math.log(max(tph_m, 1e-6))) ** 2 + (torch.log(s_[1].clamp(min=1e-6)) - math.log(max(ba_m, 1e-6))) ** 2 \
            + (torch.log(qd.clamp(min=1e-6)) - math.log(max(qd_m, 1e-6))) ** 2
    i_ = int(dist_.argmin())
    pr_ = torch.sigmoid(calibrated(lp_)); Nr = torch.exp(val_[..., 0].clamp(max=12)); Dd = torch.exp(val_[..., 1].clamp(max=math.log(150.0)))
    u_ = torch.rand(pr_.shape[1], generator=torch.Generator().manual_seed(int(seed)))
    pres = torch.nonzero(u_ < pr_[i_]).flatten().tolist() or [int((pr_[i_] * Nr[i_]).argmax())]
    comp = torch.zeros(NK)
    for j_ in pres: comp[j_ // NA] += float(Nr[i_, j_] * Dd[i_, j_] ** 2)
    comp = comp / comp.sum().clamp(min=1e-12)
    return dict(i=i_, pres=pres, pr=pr_, Nr=Nr, Dd=Dd, tph_m=tph_m, ba_m=ba_m, comp=comp, dom=int(comp.argmax()))
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
        sl = slice(a - YEARS[0], b - YEARS[0] + 1); d = F_dist[:, sl]                      # shape (n, years, NC+2)
        ev += list(((d[:, :, :NC].amax(1) > 0).float() * w_.unsqueeze(-1)).sum(0).numpy()) + [float(((d[:, :, NC + 1].amax(1) > 0).float() * w_).sum())]
    return np.r_[(F_aef * w_.unsqueeze(-1)).sum(0).numpy(), hist, ev, (F_soil * w_.unsqueeze(-1)).sum(0).numpy()].astype(np.float32)
XL = {}
IW = dict(zip(C.cid.values, C.incl_w.values)) if 'incl_w' in C.columns else None
for pid, (cid, pr) in cand_of.items(): XL[pid] = landscape(AEF[cid], DIST[cid], SOIL[cid], None if IW is None else [IW[c_] for c_ in cid])
XLm = np.stack(list(XL.values())); xmu, xsd = XLm.mean(0), XLm.std(0) + 1e-6   # (over every plot, held-out ones too: unsupervised, no labels — C9 / G46.20)
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
        lw, mu, ls = s(x, yr); k = torch.multinomial(lw.exp(), m, replacement=True)               # shape (n, m)
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
trB = np.array([v for v in tr_v if vpid[v] in loc_p and not bool(VCONV[v])]); teB = np.array([v for v in te_v if vpid[v] in loc_p])   # (G46.7: B is an AlphaEarth likelihood)
print(f'localised visits train {len(trB)} (converted plots left out: {sum(1 for v in tr_v if vpid[v] in loc_p and bool(VCONV[v]))} visits) test {len(teB)}', flush=True)
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
ev = {}; R_REP = 8                                    # (G46.11) replicate draws of the gate statistics and of the mapping calibration factors
with torch.no_grad():
    # localisation skill of B on held-out localised plots: posterior mass (uniform over ALL candidates) on the high-prior footprints vs their share
    loc = []
    for v in np.random.default_rng(32).choice(teB, min(500, len(teB)), replace=False):   # random, not key order (r11 M-2)
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
    zs = mdn.sample(X_te, vyear[te_l], MS); S_ = summ(zs.reshape(-1, D_Z)); A_ = [S_[i].view(len(te_l), MS) for i in range(3)]
    predA = [lmed(x) for x in A_]                                                     # (true median: G46.15)
    # post-hoc calibration estimated on training plots only (the decoder predicts log-medians; sums of N and N·D² are biased low)
    trs_v = [v for v in lastv[lastv.pid.isin(cal_pid)].vid.values if v in vi]           # calibration plots' last visits (not used by the MDN fit)
    zt = mdn.sample(Xv[[vi[v] for v in trs_v]], vyear[trs_v], MS); St = summ(zt.reshape(-1, D_Z)); o_ts = obs_summ(torch.tensor(trs_v))
    CAL = [float(tmed(torch.log((o_ts[i] + 1e-3) / lmed(St[i].view(len(trs_v), MS))))) for i in range(3)]
    print(f'calibration log-factors (TPH, BA, QMD) from {len(trs_v)} calibration plots:', [round(c, 3) for c in CAL], flush=True)
    predA_raw = predA; predA = [predA[i] * math.exp(CAL[i]) for i in range(3)]
    compA = S_[4].view(len(te_l), MS, NK).mean(1)
    y_tr = vyear[tr_l]
    # (r13 C-7 / G46.18) the baseline fallbacks are counted per QUERY SET (FB_SET names the set being judged), not silent and not mixed
    FALLBACK = {'knn_queries_with_lt5_in_window': {}, 'null_queries_without_pool': {}}; FB_SET = 'test_all_heldout'
    def knn_idx(Xq, yq):     # 5 nearest training plots in X_L among those whose last visit is within ±3 years of the query visit
        far = ((yq.unsqueeze(1) - y_tr.unsqueeze(0)).abs() > 3); c_ = FALLBACK['knn_queries_with_lt5_in_window']; c_[FB_SET] = c_.get(FB_SET, 0) + int(((~far).sum(1) < 5).sum())
        dd = torch.cdist(Xq, X_tr) + 1e9 * far.float(); return dd.topk(5, largest=False).indices
    nn5 = knn_idx(X_te, vyear[te_l]); knn = [o_tr[i][nn5].float().median(1).values for i in range(3)]; compK = o_tr[4][nn5].mean(1)   # (5 neighbours: the median is the middle one)
    err = lambda p, o: float(tmed(torch.abs(torch.log((p + 1e-3) / (o + 1e-3)))))
    bias = lambda p, o: float(tmed(torch.log((p + 1e-3) / (o + 1e-3))))
    bray = lambda p, o: float((torch.abs(p - o).sum(1) / 2).mean())
    ev['calibration_log_factors_TPH_BA_QMD'] = CAL
    sk = o_te[0] > 0                                                                 # stocked visits (bare visits: no composition; reported separately)
    p_trl = np.array([vpid[v] for v in tr_l])
    def null_pred(yq, excl=None):       # NULL baseline: median of the training plots' last visits within ±3 years (no satellite / landscape information at all)
        m_ = ((yq.unsqueeze(1) - y_tr.unsqueeze(0)).abs() <= 3).float(); out_ = []         # excl: plots left out of the pool (the ones being judged; r6 M7)
        if excl is not None: m_ = m_ * torch.tensor(~np.isin(p_trl, list(excl)), dtype=torch.float32).unsqueeze(0)
        c_ = FALLBACK['null_queries_without_pool']; c_[FB_SET] = c_.get(FB_SET, 0) + int((m_.sum(1) == 0).sum())
        for i in range(3):
            v_ = o_tr[i].float().unsqueeze(0).expand(len(yq), -1).clone(); v_[m_ == 0] = float('nan'); out_.append(torch.nanquantile(v_, 0.5, dim=1))   # (true median)
        return out_
    NULL_COMP = o_tr[4][o_tr[0] > 0].mean(0, keepdim=True)                            # NULL composition: mean BA shares of the stocked training visits
    nul = null_pred(vyear[te_l]); ev['null_composition_bray'] = bray(NULL_COMP.expand(int(sk.sum()), -1), o_te[4][sk])
    ev['A_prior_vs_knn_all_heldout'] = {'n': len(te_l), 'n_stocked': int(sk.sum()), 'bare_pred_BA_median_A': float(tmed(predA[1][~sk])) if (~sk).any() else None,
                                        'bare_pred_BA_median_knn': float(tmed(knn[1][~sk])) if (~sk).any() else None,
                                        **{nm: {'A_calibrated': err(predA[i][sk], o_te[i][sk]), 'A_bias': bias(predA[i][sk], o_te[i][sk]), 'A_uncalibrated': err(predA_raw[i][sk], o_te[i][sk]),
                                                'knn': err(knn[i][sk], o_te[i][sk]), 'knn_bias': bias(knn[i][sk], o_te[i][sk]), 'null': err(nul[i][sk], o_te[i][sk])} for i, nm in enumerate(['TPH', 'BA', 'QMD'])},
                                        'bray_A': bray(compA[sk], o_te[4][sk]), 'bray_knn': bray(compK[sk], o_te[4][sk]),
                                        'dominant_acc_A': float((compA[sk].argmax(1) == o_te[3][sk]).float().mean()), 'dominant_acc_knn': float((compK[sk].argmax(1) == o_te[3][sk]).float().mean())}
    # A x B at the pinned footprints of held-out localised plots (location nearly known): posterior-resampled prior draws vs A alone
    te_loc = [v for v in te_l if vpid[v] in loc_p]
    if te_loc:
        # at each held-out localised visit: A = prior samples AT THE VISIT YEAR, decoded at the visit; AB = the same samples resampled by the AlphaEarth match
        # of their forward flow (visit year -> 2022 / 2025, through each pinned footprint's history; targets after the visit only)
        MSAB = 64; res = {'A': [], 'AB': []}; used = []; ess = []
        for v in te_loc:
            yv = float(vyear[v])
            if yv > 2025.5: continue
            cid, pr = hi_cands(vpid[v]); zs = mdn.sample(Xv[[vi[v]]], torch.tensor([yv]), MSAB)[0]
            Dm = torch.stack([fwd_dist(zs, torch.full((MSAB,), yv), torch.tensor([c_] * MSAB), rs=VREC[[v]].repeat(MSAB), ra=VRA[[v]].repeat(MSAB))[0] for c_ in cid])     # shape (footprints, MSAB)
            lw = torch.logsumexp(-Dm / TEMP + torch.log(torch.tensor(pr, dtype=torch.float32)).unsqueeze(1), 0)    # log Σ_c p_c exp(-d_c(z)/T): location mixture
            w = torch.softmax(lw, 0); idx = torch.multinomial(w, MS, replacement=True); ess.append(float(1 / (w ** 2).sum()))
            sA = summ(zs); sB = summ(zs[idx]); used.append(v)
            for nm, s_ in (('A', sA), ('AB', sB)): res[nm].append([calibrated_median(s_[i], CAL[i]) for i in range(3)] + [s_[4].mean(0)])
        te_loc = used
        FB_SET = 'test_localised'
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
    # date -> decode -> the medoid draw (medoid_draw) and the calibrated median (calibrated_median), MS iid draws: the estimator g05 maps (G46.13).
    # Calibration factors for THIS path from the calibration plots (visits 2022-2025), errors on held-out plots' visits 2022-2025.
    def map_path(vs, y0, ms=MS, X=None, to_end=False):
        """per visit: [true-median TPH, BA, QMD of the ms draws (uncalibrated: calibrated_median with CF 0), mean BA shares, medoid BA shares, medoid
        dominant key, medoid age-class distribution by TPH, by BA]. to_end: flow to the end of 2025 as g05 maps (G46.14), not to the visit date"""
        out_ = []
        for q_, v in enumerate(vs):
            cid, pr = cand_of[vpid[v]]; ci = torch.tensor(np.random.choice(cid, ms, replace=True, p=pr)); yv = vyear[v]
            xv = Xv[[vi[v]]] if X is None else X[q_:q_ + 1]
            zs = mdn.sample(xv, torch.tensor([y0]), ms)[0]
            if to_end: yy = YEARS[-1]; st = flow(zs, torch.full((ms,), y0), [yy], DIST[ci], SOIL[ci])
            else:
                yy = min(int(math.floor(float(yv))), YEARS[-1])
                st = flow(zs, torch.full((ms,), y0), [yy], DIST[ci], SOIL[ci], end=yv.repeat(ms), re=VREC[[v]].repeat(ms))
            lp_m, val_m = dec(st[yy]); s_ = summaries(lp_m, val_m)
            md_ = medoid_draw(lp_m, val_m, s_, int(v))                       # the composition / dominant key g05 maps (G32)
            aN, aB = torch.zeros(NA), torch.zeros(NA)                        # the medoid's cohorts by age class (g05's cohort table; its scaling keeps the shares)
            for j_ in md_['pres']: aN[j_ % NA] += float(md_['Nr'][md_['i'], j_]); aB[j_ % NA] += float(md_['Nr'][md_['i'], j_] * md_['Dd'][md_['i'], j_] ** 2)
            out_.append([calibrated_median(s_[i], 0.0) for i in range(3)] + [s_[4].mean(0), md_['comp'], md_['dom'], aN / aN.sum().clamp(min=1e-12), aB / aB.sum().clamp(min=1e-12)])
        return out_
    def obs_age(vs):    # observed age-class distribution by TPH, by BA
        m = slot_p[vs]; N = torch.exp(slot_v[vs][..., 0]) * m; Dq = torch.exp(slot_v[vs][..., 1])
        aN = N.view(-1, NK, NA).sum(1); aB = (N * Dq ** 2).view(-1, NK, NA).sum(1)
        return aN / aN.sum(1, keepdim=True).clamp(min=1e-12), aB / aB.sum(1, keepdim=True).clamp(min=1e-12)
    def w1_age(p, o):   # Wasserstein-1 distance between age-class distributions (10-yr classes), yr
        return 10 * (torch.cumsum(p, -1) - torch.cumsum(o, -1)).abs().sum(-1)
    rec = lambda vs: [v for v in vs if 2022.0 <= float(vyear[v]) < 2026.0]
    cal_m = rec(trs_v); te_m = rec(te_l)
    if cal_m and te_m:
        ot = obs_summ(torch.tensor(te_m)); skt = ot[0] > 0; FB_SET = 'test_2022_2025'; nm_ = null_pred(vyear[te_m])
        def fit_cf(y0, cal):
            """the mapping-path calibration log-factors (TPH, BA, QMD) fitted on the visits cal: true median of log(observed / median draw), stocked visits"""
            occ = obs_summ(torch.tensor(cal)); skc_ = occ[0] > 0; pc = map_path(cal, y0)
            return [float(tmed(torch.log((occ[i][skc_] + 1e-3) / torch.stack([r[i] for r in pc])[skc_]))) for i in range(3)]
        def score(y0, CF, te):
            """the mapped estimator at the visits te with factors CF: per-visit draws, calibrated predictions, and its metrics (stocked: median |log err|,
            medoid Bray / dominant key; report-only: medoid age-class Wasserstein (G46.19); bare visits: the predicted BA)"""
            ott = obs_summ(torch.tensor(te)); skt_ = ott[0] > 0; pt = map_path(te, y0)
            P_ = [torch.stack([r[i] for r in pt]) * math.exp(CF[i]) for i in range(3)]   # = calibrated_median of each visit's draws
            P_[2] = torch.where(P_[0] > 0, 100 * torch.sqrt(P_[1] / (P_[0] * math.pi / 4).clamp(min=1e-9)), torch.zeros_like(P_[0]))   # QMD as g05 maps it:
            # from the calibrated TPH / BA (GEN_QMD), not the samples' median QMD x its own factor (audit r8 M6)
            Cmd = torch.stack([r[4] for r in pt]); dmd = torch.tensor([r[5] for r in pt]); oaN, oaB = obs_age(torch.tensor(te))
            aN_, aB_ = torch.stack([r[6] for r in pt]), torch.stack([r[7] for r in pt])
            met = {**{nm: err(P_[i][skt_], ott[i][skt_]) for i, nm in enumerate(['TPH', 'BA', 'QMD'])},
                   'bray_medoid': bray(Cmd[skt_], ott[4][skt_]), 'dominant_acc_medoid': float((dmd[skt_] == ott[3][skt_]).float().mean()),
                   'age_w1_tph_yr': float(tmed(w1_age(aN_[skt_], oaN[skt_]))), 'age_w1_ba_yr': float(tmed(w1_age(aB_[skt_], oaB[skt_]))),
                   'n_bare': int((~skt_).sum()), 'bare_pred_BA_median': float(tmed(P_[1][~skt_])) if (~skt_).any() else None}
            return pt, P_, met
        def evaluate(y0, cal, te):
            CF = fit_cf(y0, cal); pt, P_, met = score(y0, CF, te); return CF, pt, P_, met
        _cp = sorted(set(vpid[cal_m])); _hp = set(np.random.default_rng(34).permutation(_cp)[:len(_cp) // 2])     # random plot halves (r13 C-8: not pid parity)
        hA = [v for v in cal_m if vpid[v] in _hp]; hB = [v for v in cal_m if vpid[v] not in _hp]
        HALVES = ((hA, hB), (hB, hA))                                                     # each half judged with factors from the other
        XF = {}                                                                           # (y0, replicate) -> the two halves' cross-fitted (CF, draws, P, metrics)
        def crossfit(y0, r_):
            if (y0, r_) not in XF: XF[(y0, r_)] = [evaluate(y0, ca, cb) for ca, cb in HALVES]
            return XF[(y0, r_)]
        def derange(n, rng_):
            """a uniformly random derangement of range(n) (no visit keeps its own landscape; rejection sampling); n < 2: the identity"""
            while n >= 2:
                p_ = rng_.permutation(n)
                if not (p_ == np.arange(n)).any(): return p_
            return np.arange(n)
        def mean_sd(xs):
            return float(np.mean(xs)), float(np.std(xs))
        # PRE-REGISTERED gate of the mapping start year (audit r4 B1): MAP_Y0 (e.g. 2000) is used only if its TPH and BA errors are no worse than the 2021
        # path's and than the null's, AND the prediction still depends on the stand's landscape (at least half the 2021 path's landscape dependence, r10); else 2021. The gate is decided on the CALIBRATION plots (cross-fitted: factors from one half of the plots, errors on
        # the other), never on the held-out plots it then reports (audit r5 M1). Every statistic is the mean of R_REP replicates (fresh draws, a fresh
        # derangement of the landscapes; G46.11); a NaN statistic fails the gate with its reason (G46.18). The chosen year is recorded; g05 maps with it.
        gate = {'requested': MAP_Y0, 'decided_on': 'calibration plots, 2-fold by plot (seeded random halves)', 'replicates': R_REP}
        if MAP_Y0 != 2021.0:
            FB_SET = 'gate_halves'; nulle = {k: 0.0 for k in ('TPH', 'BA')}
            for ca, cb in HALVES:                            # the null is deterministic: once per half
                ob_ = obs_summ(torch.tensor(cb)); sk_ = ob_[0] > 0; nmb = null_pred(vyear[cb], excl=set(vpid[cb]))
                for i, k in ((0, 'TPH'), (1, 'BA')): nulle[k] += err(nmb[i][sk_], ob_[i][sk_]) / 2
            reps = []
            for r_ in range(R_REP):
                e0, e1, cs_, cs0_, cr_, cr0_ = {k: 0.0 for k in ('TPH', 'BA')}, {k: 0.0 for k in ('TPH', 'BA')}, [], [], [], []
                rng_d = np.random.default_rng(7 + r_)           # a fresh derangement per replicate (C13)
                for h_, (ca, cb) in enumerate(HALVES):
                    r0, r1 = crossfit(2021.0, r_)[h_], crossfit(MAP_Y0, r_)[h_]
                    Xs = Xv[[vi[cb[i]] for i in derange(len(cb), rng_d)]]
                    ps_ = map_path(cb, MAP_Y0, X=Xs); ba_r = torch.log(torch.stack([r[1] for r in r1[1]]) + 1e-3); ba_s = torch.log(torch.stack([r[1] for r in ps_]) + 1e-3)
                    cs_.append(float(np.corrcoef(ba_r.numpy(), ba_s.numpy())[0, 1]))
                    ps0 = map_path(cb, 2021.0, X=Xs); b0r = torch.log(torch.stack([r[1] for r in r0[1]]) + 1e-3); b0s = torch.log(torch.stack([r[1] for r in ps0]) + 1e-3)
                    cs0_.append(float(np.corrcoef(b0r.numpy(), b0s.numpy())[0, 1]))   # the same statistic for the 2021 path (the reference: r10)
                    # the Monte-Carlo noise floor: the same landscapes REDRAWN (new candidate / MDN draws) — dependence = corr(redraw) − corr(shuffled) (r12 S-1)
                    pr1 = map_path(cb, MAP_Y0); pr0 = map_path(cb, 2021.0)
                    cr_.append(float(np.corrcoef(ba_r.numpy(), torch.log(torch.stack([r[1] for r in pr1]) + 1e-3).numpy())[0, 1]))
                    cr0_.append(float(np.corrcoef(b0r.numpy(), torch.log(torch.stack([r[1] for r in pr0]) + 1e-3).numpy())[0, 1]))
                    for k in ('TPH', 'BA'): e0[k] += r0[3][k] / 2; e1[k] += r1[3][k] / 2
                cs, cs0, cr, cr0 = float(np.mean(cs_)), float(np.mean(cs0_)), float(np.mean(cr_)), float(np.mean(cr0_))
                reps.append({'err_2021': e0, 'err_requested': e1, 'shuffled_landscape_corr_BA': cs, 'shuffled_landscape_corr_BA_2021': cs0, 'redraw_corr_BA': cr,
                             'redraw_corr_BA_2021': cr0, 'landscape_dependence': cr - cs, 'landscape_dependence_2021': cr0 - cs0})
            # the gate statistics = replicate means (sd recorded); dependence = mean redraw corr − mean shuffled corr (the same as the mean of the differences)
            gm = {k: mean_sd([r[k] for r in reps]) for k in ('shuffled_landscape_corr_BA', 'shuffled_landscape_corr_BA_2021', 'redraw_corr_BA', 'redraw_corr_BA_2021', 'landscape_dependence', 'landscape_dependence_2021')}
            e0 = {k: mean_sd([r['err_2021'][k] for r in reps]) for k in ('TPH', 'BA')}; e1 = {k: mean_sd([r['err_requested'][k] for r in reps]) for k in ('TPH', 'BA')}
            cs, cs0, cr, cr0 = gm['shuffled_landscape_corr_BA'][0], gm['shuffled_landscape_corr_BA_2021'][0], gm['redraw_corr_BA'][0], gm['redraw_corr_BA_2021'][0]
            dep, dep0 = cr - cs, cr0 - cs0                    # landscape dependence above the redraw noise floor (G24)
            stats_ = {'err_2021_TPH': e0['TPH'][0], 'err_2021_BA': e0['BA'][0], 'err_requested_TPH': e1['TPH'][0], 'err_requested_BA': e1['BA'][0], 'err_null_TPH': nulle['TPH'],
                      'err_null_BA': nulle['BA'], 'shuffled_landscape_corr_BA': cs, 'shuffled_landscape_corr_BA_2021': cs0, 'redraw_corr_BA': cr, 'redraw_corr_BA_2021': cr0}
            why = [f'NaN statistic {k}' for k, x in stats_.items() if not math.isfinite(x)]     # (G46.18: an explicit FAIL, not a silent NaN comparison)
            for k in ('TPH', 'BA'):
                if not e1[k][0] <= e0[k][0]: why.append(f'{k} error {e1[k][0]:.4f} > 2021 path {e0[k][0]:.4f}')
                if not e1[k][0] <= nulle[k]: why.append(f'{k} error {e1[k][0]:.4f} > null {nulle[k]:.4f}')
            if not dep0 > 0: why.append(f'2021 path landscape dependence {dep0:.4f} <= 0 (G33)')
            if not dep >= 0.5 * dep0: why.append(f'landscape dependence {dep:.4f} < half the 2021 path {dep0:.4f}')
            ok = not why
            gate.update({'err_2021': {k: e0[k][0] for k in e0}, 'err_requested': {k: e1[k][0] for k in e1}, 'err_null': nulle, 'shuffled_landscape_corr_BA': cs,
                         'shuffled_landscape_corr_BA_2021': cs0, 'redraw_corr_BA': cr, 'redraw_corr_BA_2021': cr0, 'landscape_dependence': dep, 'landscape_dependence_2021': dep0,
                         'sd': {**{f'err_2021_{k}': e0[k][1] for k in e0}, **{f'err_requested_{k}': e1[k][1] for k in e1}, **{k: v_[1] for k, v_ in gm.items()}},
                         'per_replicate': reps, 'passed': bool(ok), 'fail_reasons': why})
        y0u = MAP_Y0 if gate.get('passed') else 2021.0
        # the mapping calibration factors g05 applies (CALF): the mean of R_REP replicate fits on all calibration plots (G46.11); per path, for the report
        CFr = {y0: [fit_cf(y0, cal_m) for _ in range(R_REP)] for y0 in sorted({2021.0, MAP_Y0})}
        CFm = {y0: [float(np.mean([c[i] for c in CFr[y0]])) for i in range(3)] for y0 in CFr}
        res = {y0: score(y0, CFm[y0], te_m) for y0 in CFr}                               # held-out (test) scores: reported, never used for a decision
        if MAP_Y0 != 2021.0: gate.update({'heldout_err_2021': res[2021.0][2], 'heldout_err_requested': res[MAP_Y0][2]})
        CALF = CFm[y0u]; pt, Pm, _ = res[y0u]; Cm = torch.stack([r[3] for r in pt])
        ev['map_y0'] = y0u; ev['map_y0_gate'] = gate
        # VALIDATION metrics of the chosen mapping path (cross-fitted on the calibration = validation plots: factors from one half, errors on the other;
        # mean of R_REP replicates) with the null and kNN on the same visits (G46.12) — model choices (v3 vs v4, seeds) and the map acceptance are made on
        # these; the held-out block below is the test report (r9)
        mk_ = ('TPH', 'BA', 'QMD', 'bray_medoid', 'dominant_acc_medoid', 'age_w1_tph_yr', 'age_w1_ba_yr', 'bare_pred_BA_median')
        xr_ = [{k: float(np.mean([crossfit(y0u, r_)[h_][3][k] for h_ in range(2) if crossfit(y0u, r_)[h_][3][k] is not None] or [float('nan')])) for k in mk_} for r_ in range(R_REP)]
        xv_ = {k: mean_sd([x[k] for x in xr_]) for k in mk_}
        FB_SET = 'validation_halves'; nv_ = {k: 0.0 for k in ('TPH', 'BA', 'QMD', 'bray')}; kv_ = {k: 0.0 for k in ('TPH', 'BA', 'QMD', 'bray', 'dominant_acc')}
        for ca, cb in HALVES:
            ob_ = obs_summ(torch.tensor(cb)); sk_ = ob_[0] > 0; nmb = null_pred(vyear[cb], excl=set(vpid[cb])); nnv = knn_idx(Xv[[vi[v] for v in cb]], vyear[cb])
            knv = [o_tr[i][nnv].float().median(1).values for i in range(3)]; kcv = o_tr[4][nnv].mean(1)
            for i, k in enumerate(('TPH', 'BA', 'QMD')): nv_[k] += err(nmb[i][sk_], ob_[i][sk_]) / 2; kv_[k] += err(knv[i][sk_], ob_[i][sk_]) / 2
            nv_['bray'] += bray(NULL_COMP.expand(int(sk_.sum()), -1), ob_[4][sk_]) / 2; kv_['bray'] += bray(kcv[sk_], ob_[4][sk_]) / 2
            kv_['dominant_acc'] += float((kcv[sk_].argmax(1) == ob_[3][sk_]).float().mean()) / 2
        ev['mapping_path_validation_crossfit'] = {'map_y0': y0u, 'n': len(cal_m), 'replicates': R_REP, 'set': 'validation plots' if VAL_SPLIT else 'calibration blocks (training)',
                                                  'median_abs_log_err': {k: xv_[k][0] for k in ('TPH', 'BA', 'QMD')}, 'composition': {k: xv_[k][0] for k in ('bray_medoid', 'dominant_acc_medoid')},
                                                  'sd': {k: xv_[k][1] for k in mk_}, 'null': nv_, 'knn': kv_,
                                                  'report_only': {k: xv_[k][0] for k in ('age_w1_tph_yr', 'age_w1_ba_yr', 'bare_pred_BA_median')}}
        # PRE-REGISTERED map acceptance (G46.12): on validation, the chosen path's calibrated median |log err| of BA and TPH and its medoid Bray each below
        # the null's (kNN reported, not gated); g05 refuses to map otherwise
        mwhy = [f'NaN {k}' for k in ('TPH', 'BA', 'bray_medoid') if not math.isfinite(xv_[k][0])] + [f'NaN null {k}' for k in ('TPH', 'BA', 'bray') if not math.isfinite(nv_[k])]
        mwhy += [f'{k} error {xv_[k][0]:.4f} >= null {nv_[k]:.4f}' for k in ('TPH', 'BA') if not xv_[k][0] < nv_[k]]
        if not xv_['bray_medoid'][0] < nv_['bray']: mwhy.append(f"medoid Bray {xv_['bray_medoid'][0]:.4f} >= null {nv_['bray']:.4f}")
        ev['map_accept'] = not mwhy; ev['map_accept_reasons'] = mwhy
        # visit date vs end-2025 (G46.14, a documented residual): g04 validates the state at the visit date, g05 maps the end-2025 state; the median shift
        # of the calibrated medians between the two on the validation visits, with common random numbers
        _rs = (torch.get_rng_state(), np.random.get_state()); pv_ = map_path(cal_m, y0u); torch.set_rng_state(_rs[0]); np.random.set_state(_rs[1]); pe_ = map_path(cal_m, y0u, to_end=True)
        ev['visit_date_vs_end2025_validation'] = {'n': len(cal_m), 'median_years_to_end2025': float(np.median([2026.0 - float(vyear[v]) for v in cal_m])),
                                                  **{f'median_log_shift_{k}': float(tmed(torch.log(torch.stack([r[i] for r in pe_]) / torch.stack([r[i] for r in pv_])))) for i, k in ((0, 'TPH'), (1, 'BA'))}}
        FB_SET = 'test_2022_2025'; nnm = knn_idx(Xv[[vi[v] for v in te_m]], vyear[te_m]); km_ = [o_tr[i][nnm].float().median(1).values for i in range(3)]
        ev['calibration_log_factors_mapping_path'] = CALF
        ev['calibration_log_factors_mapping_path_replicates'] = {'per_replicate': CFr[y0u], 'sd': [float(np.std([c[i] for c in CFr[y0u]])) for i in range(3)]}
        ev['mapping_path_heldout_2022_2025'] = {'n': len(te_m), 'n_stocked': int(skt.sum()), 'n_calibration_visits': len(cal_m), 'map_y0': y0u,
            **{nm: {'map_path': err(Pm[i][skt], ot[i][skt]), 'map_bias': bias(Pm[i][skt], ot[i][skt]), 'knn': err(km_[i][skt], ot[i][skt]), 'null': err(nm_[i][skt], ot[i][skt])} for i, nm in enumerate(['TPH', 'BA', 'QMD'])},
            'bray_map_meanshare': bray(Cm[skt], ot[4][skt]), 'bray_knn': bray(o_tr[4][nnm].mean(1)[skt], ot[4][skt]),
            'bray_map_medoid': res[y0u][2]['bray_medoid'], 'dominant_acc_map_medoid': res[y0u][2]['dominant_acc_medoid'],    # the composition g05 maps (G32)
            'dominant_acc_knn': float((o_tr[4][nnm].mean(1).argmax(1)[skt] == ot[3][skt]).float().mean()),
            'report_only': {k: res[y0u][2][k] for k in ('age_w1_tph_yr', 'age_w1_ba_yr', 'n_bare', 'bare_pred_BA_median')}}
    else: ev['map_accept'] = False; ev['map_accept_reasons'] = ['no calibration or held-out visits 2022-2025: the mapping path cannot be validated']
ev['baseline_fallbacks'] = FALLBACK
print(json.dumps(ev, indent=1)); json.dump(ev, open(f'{D}/bayes_eval.json', 'w'), indent=1)
