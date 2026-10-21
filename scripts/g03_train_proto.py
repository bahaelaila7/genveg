"""Generative-vegetation prototype (Baker County FL), step 2: temporal JEPA model (liquid flow + disturbance jumps + SIGReg) — train, evaluate, map.
  E_v      masked set encoder over cohort tokens (species, age class, log N, log QMD, H, CR) → z (D_Z), SIGReg-regularised (LeJEPA: Epps-Pulley test of
           random 1-D projections against N(0,1))
  flow     liquid time-constant cell, yearly steps: dz = −(1/τ + g)⊙z + g⊙A, g = σ(W [z, s] + b) (s = static site covariates), from the visit year on
  jump     at a candidate's disturbance years: z ← z + J([z, class one-hot, severity, Hansen loss])
  heads    z(2022) → AlphaEarth 2022, z(2025) → AlphaEarth 2025 (64-d)
  MIL      the plot location is latent: K candidates sampled ∝ the disturbance prior; loss = −logsumexp_c(−d_c / T) (+ log K)
  pairs    FIA remeasurement: flow(z(v1)) along the candidate's history to v2's year ≈ z(v2) (posterior-weighted, stop-gradient on the target)
  decoder  z → cohort slots (species x age class): presence (BCE) + log N, log QMD, H, CR (masked MSE)
  Q        inverse encoder: candidate features (AlphaEarth 2022 + 2025, disturbance history, soils) → z(2025) (Gaussian) — the mapping model
Held-out: plots of 20 % of 10-km blocks. Baseline: kNN on the prior-weighted mean candidate features. Writes data/proto/baker/{model.pt, eval.json, stand_veg.parquet}."""
import os as _os; _SM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _WS = _os.environ.get('WORKSPACE', _os.path.dirname(_SM))   # relocatable paths
import os, json, math, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(int(os.environ.get('NT', 6)))
D = _SM + '/data/proto/baker'; M = json.load(open(f'{D}/meta.json'))
V = pd.read_parquet(f'{D}/visits.parquet'); TK = pd.read_parquet(f'{D}/tokens.parquet'); PR = pd.read_parquet(f'{D}/pairs.parquet')
C = pd.read_parquet(f'{D}/cands.parquet'); CF = np.load(f'{D}/cand_feat.npy').astype(np.float32)
NK, NA = len(M['keys']), 16; YEARS = M['years']; NY = len(YEARS); NC = len(M['classes']); D_Z = int(os.environ.get('DZ', 16))
# ---- feature slices
A22, A25 = slice(0, 64), slice(64, 128); DS = slice(128, 128 + NY * (NC + 2)); SO = slice(128 + NY * (NC + 2), CF.shape[1])
smu, ssd = CF[:, SO].mean(0), CF[:, SO].std(0) + 1e-6; CF[:, SO] = (CF[:, SO] - smu) / ssd
DIST = torch.tensor(CF[:, DS].reshape(-1, NY, NC + 2)); AEF = torch.tensor(CF[:, :128]); SOIL = torch.tensor(CF[:, SO])
# ---- tokens → padded tensors per visit
TK = TK[TK.N > 0].copy(); TK[['key', 'ac', 'vid']] = TK[['key', 'ac', 'vid']].astype(int); TK['lN'] = np.log(TK.N); TK['lQ'] = np.log(TK.QMD); TK['h'] = TK.H / 30; TK['crm'] = TK.CR.notna().astype(float); TK['cr'] = TK.CR.fillna(0)
nv = len(V); maxT = int(TK.groupby('vid').size().max())
tok_k = torch.zeros(nv, maxT, dtype=torch.long); tok_a = torch.zeros(nv, maxT, dtype=torch.long); tok_x = torch.zeros(nv, maxT, 5); tok_m = torch.zeros(nv, maxT)
for vid, g in TK.groupby('vid'):
    n = len(g); tok_k[vid, :n] = torch.tensor(g.key.values); tok_a[vid, :n] = torch.tensor(g.ac.values)
    tok_x[vid, :n] = torch.tensor(g[['lN', 'lQ', 'h', 'cr', 'crm']].values, dtype=torch.float32); tok_m[vid, :n] = 1
# decoder targets: slots key x age class
slot_p = torch.zeros(nv, NK * NA); slot_v = torch.zeros(nv, NK * NA, 4); slot_vm = torch.zeros(nv, NK * NA, 4)
for r in TK.itertuples():
    j = r.key * NA + r.ac; slot_p[r.vid, j] = 1; slot_v[r.vid, j] = torch.tensor([r.lN, r.lQ, r.h, r.cr]); slot_vm[r.vid, j] = torch.tensor([1, 1, 1, r.crm])
vyear = torch.tensor(V.year.values, dtype=torch.float32); vpid = V.pid.values
# ---- candidates per plot (index arrays) and spatial blocks
cand_of = {pid: (g.cid.values, g.prior.values) for pid, g in C.groupby('pid')}
P = pd.read_parquet(f'{D}/plots.parquet'); P['blk'] = (P.x // 10000).astype(int) * 1000 + (P.y // 10000).astype(int)
rng = np.random.default_rng(1); blks = P.blk.unique(); test_blk = set(rng.choice(blks, max(1, int(0.2 * len(blks))), replace=False))
test_pid = set(P.pid[P.blk.isin(test_blk)]); have = set(cand_of)
tr_v = np.array([v for v in range(nv) if vpid[v] in have and vpid[v] not in test_pid]); te_v = np.array([v for v in range(nv) if vpid[v] in have and vpid[v] in test_pid])
print(f'visits train {len(tr_v)} test {len(te_v)} | plots test {len(test_pid & have)} | tokens/visit max {maxT}', flush=True)
# ---- model
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.ek = nn.Embedding(NK, 8); s.ea = nn.Embedding(NA, 4)
        s.phi = nn.Sequential(nn.Linear(17, 64), nn.GELU(), nn.Linear(64, 64)); s.rho = nn.Sequential(nn.Linear(129, 64), nn.GELU(), nn.Linear(64, D_Z))
    def forward(s, v, yr):
        h = s.phi(torch.cat([s.ek(tok_k[v]), s.ea(tok_a[v]), tok_x[v]], -1)); m = tok_m[v].unsqueeze(-1)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1); mx = (h + (m - 1) * 1e4).max(1).values
        mx = torch.where(m.sum(1) > 0, mx, torch.zeros_like(mx))           # visits without trees (fresh clear-cuts): defined empty encoding
        r = s.rho(torch.cat([mean, mx, ((yr - 2012) / 10).unsqueeze(-1)], -1))
        return 3 * torch.tanh(r / 3)                                       # soft bound: SIGReg's ECF test saturates (no restoring gradient) for large z
class Flow(nn.Module):        # liquid time-constant cell (Hasani et al.): dz = −(1/τ + g) z + g A, Euler with dt = 1 yr, + disturbance jumps
    def __init__(s):
        super().__init__(); ns = SOIL.shape[1]
        s.g = nn.Sequential(nn.Linear(D_Z + ns, 64), nn.Tanh(), nn.Linear(64, D_Z)); s.A = nn.Parameter(torch.zeros(D_Z)); s.ltau = nn.Parameter(torch.full((D_Z,), 2.0))
        s.J = nn.Sequential(nn.Linear(D_Z + NC + 2, 64), nn.GELU(), nn.Linear(64, D_Z))
    def forward(s, z, start, stop, dist, soil):
        """z [n, D_Z] at year `start` [n] → states at the end of each year, recorded at `stop` years; dist [n, NY, NC+2]; returns dict stop->state."""
        y0 = int(math.floor(start.min().item())); out = {st: z for st in stop if st < max(y0, YEARS[0])}   # stops before the run: initial state (masked by the caller)
        for j, y in enumerate(range(max(y0, YEARS[0]), YEARS[-1] + 1)):
            act = (start <= y + 1).float().unsqueeze(-1)              # the visit happened before the end of year y
            jj = y - YEARS[0]
            g = torch.sigmoid(s.g(torch.cat([z, soil], -1)))
            lam = 1 / torch.exp(s.ltau) + g; zs = g * 3 * torch.tanh(s.A) / lam   # exact (exponential) integration over the 1-yr step; attractor bounded (|A| ≤ 3)
            z = act * (zs + (z - zs) * torch.exp(-lam)) + (1 - act) * z
            ev = dist[:, jj]; has = (ev[:, :NC].sum(-1, keepdim=True) + ev[:, NC + 1:NC + 2] > 0).float() * act
            z = z + has * torch.tanh(s.J(torch.cat([z, ev], -1)))          # bounded jump
            for st in stop:
                if y == st: out[st] = z
        return out
class Dec(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Linear(D_Z, 128), nn.GELU(), nn.Linear(128, NK * NA * 5))
    def forward(s, z): o = s.f(z).view(-1, NK * NA, 5); return o[..., 0], o[..., 1:]
class Q(nn.Module):           # satellite + disturbance history + soils → z(2025)
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Linear(128 + NY * (NC + 2) + SOIL.shape[1], 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 2 * D_Z))
    def forward(s, x): o = s.f(x); return 3 * torch.tanh(o[:, :D_Z] / 3), o[:, D_Z:].clamp(-4, 1)
enc, flow, dec, qn = Enc(), Flow(), Dec(), Q(); h22, h25 = nn.Linear(D_Z, 64), nn.Linear(D_Z, 64)
params = [*enc.parameters(), *flow.parameters(), *dec.parameters(), *qn.parameters(), *h22.parameters(), *h25.parameters()]
opt = torch.optim.AdamW(params, lr=float(os.environ.get('LR', 1e-3)), weight_decay=1e-4)
def sigreg(z, nproj=32, t=torch.linspace(-3, 3, 17)):
    """LeJEPA SIGReg: Epps-Pulley distance between the empirical characteristic function of random 1-D projections of z and that of N(0, 1)."""
    u = Fn.normalize(torch.randn(z.shape[1], nproj), dim=0); p = (z - z.mean(0)) @ u                       # [n, nproj]
    arg = p.unsqueeze(-1) * t                                                                            # [n, nproj, T]
    ecf_re, ecf_im = torch.cos(arg).mean(0), torch.sin(arg).mean(0); tgt = torch.exp(-t ** 2 / 2)
    w = torch.exp(-t ** 2 / 2); return (((ecf_re - tgt) ** 2 + ecf_im ** 2) * w).mean()
def sample_cands(pids, K):
    idx = []
    for p in pids:
        cid, pr = cand_of[p]; k = min(K, len(cid)); idx.append(np.random.choice(cid, k, replace=len(cid) < K, p=pr) if len(cid) >= 1 else None)
    return np.stack([np.resize(i, K) for i in idx])
def dec_loss(z, v):
    lp, val = dec(z); bce = Fn.binary_cross_entropy_with_logits(lp, slot_p[v])
    m = slot_vm[v] * slot_p[v].unsqueeze(-1); mse = ((val - slot_v[v]) ** 2 * m).sum() / m.sum().clamp(min=1); return bce + mse
TEMP = 0.05; K = int(os.environ.get('KC', 48)); EP = int(os.environ.get('EPOCHS', 30)); B = 32; t0 = time.time()
vpair = dict(zip(PR.v1.values, PR.v2.values))
for ep in range(EP):
    perm = np.random.permutation(tr_v); tot = {}
    for b in range(0, len(perm), B):
        v = torch.tensor(perm[b:b + B]); pids = vpid[v.numpy()]
        z = enc(v, vyear[v]); ci = sample_cands(pids, K)                                                  # [B, K]
        cz = z.repeat_interleave(K, 0); cst = vyear[v].repeat_interleave(K); cidx = torch.tensor(ci.ravel())
        st = flow(cz, cst, [2022, 2025], DIST[cidx], SOIL[cidx])
        d = torch.zeros(len(cidx)); n_t = torch.zeros(len(cidx))
        for yr, head, sl in ((2022, h22, A22), (2025, h25, A25)):
            ok = (cst <= yr + 1).float(); pred = head(st[yr]); d = d + ok * ((pred - AEF[cidx][:, sl]) ** 2).sum(-1); n_t = n_t + ok
        d = (d / n_t.clamp(min=1)).view(-1, K)
        mil = -(torch.logsumexp(-d / TEMP, 1) - math.log(K)).mean() * TEMP
        post = torch.softmax(-d.detach() / TEMP, 1)                                                        # posterior over the sampled candidates
        # remeasurement pairs: flow z(v1) along each candidate's history to v2's year
        has2 = [i for i, vv in enumerate(v.tolist()) if vv in vpair]
        lpair = torch.tensor(0.0); ldecf = torch.tensor(0.0)
        if has2:
            v2 = torch.tensor([vpair[v[i].item()] for i in has2]); y2 = vyear[v2]; z2 = enc(v2, y2).detach()
            sel = torch.tensor(has2); zz = z[sel].repeat_interleave(K, 0); c2 = torch.tensor(ci[has2].ravel())
            yy = torch.floor(y2).long().clamp(YEARS[0], YEARS[-1]); outs = []
            stp = sorted(set(yy.tolist())); st2 = flow(zz, vyear[v[sel]].repeat_interleave(K), stp, DIST[c2], SOIL[c2])
            fz = torch.stack([st2[int(yy[i // K])][i] for i in range(len(c2))]).view(len(has2), K, D_Z)
            lpair = ((fz - z2.unsqueeze(1)) ** 2).mean(-1).mul(post[sel]).sum(1).mean()
            # the decoder must read FLOWED states too: decode the flowed z(v1→v2) against v2's observed cohorts (posterior-weighted)
            lp2, val2 = dec(fz.reshape(-1, D_Z)); tp = slot_p[v2].repeat_interleave(K, 0); tv = slot_v[v2].repeat_interleave(K, 0); tm = slot_vm[v2].repeat_interleave(K, 0) * tp.unsqueeze(-1)
            bce2 = Fn.binary_cross_entropy_with_logits(lp2, tp, reduction='none').mean(-1); mse2 = (((val2 - tv) ** 2 * tm).sum((-1, -2)) / tm.sum((-1, -2)).clamp(min=1))
            ldecf = ((bce2 + mse2).view(len(has2), K) * post[sel]).sum(1).mean()
        else:
            ldecf = torch.tensor(0.0)
        # inverse encoder Q: candidate features → z(2025) (posterior-weighted, stop-gradient)
        tgt = st[2025].detach().view(-1, K, D_Z); qx = torch.cat([AEF[cidx], DIST[cidx].flatten(1), SOIL[cidx]], -1)
        qm, qls = qn(qx); nll = (0.5 * ((tgt.reshape(-1, D_Z) - qm) ** 2 / torch.exp(2 * qls)) + qls).sum(-1).view(-1, K)
        lq = (nll * post).sum(1).mean() / D_Z
        lpq, valq = dec(qm); tpq = slot_p[v].repeat_interleave(K, 0); tvq = slot_v[v].repeat_interleave(K, 0); tmq = slot_vm[v].repeat_interleave(K, 0) * tpq.unsqueeze(-1)
        dq = Fn.binary_cross_entropy_with_logits(lpq, tpq, reduction='none').mean(-1) + (((valq - tvq) ** 2 * tmq).sum((-1, -2)) / tmq.sum((-1, -2)).clamp(min=1))
        wy = torch.exp(-torch.abs(vyear[v] - 2023.5) / 4)                    # visits near the AlphaEarth years count most
        lqd = ((dq.view(-1, K) * post).sum(1) * wy).sum() / wy.sum()
        loss = mil + 0.5 * lpair + dec_loss(z, v) + 0.5 * ldecf + 1.0 * sigreg(z) * z.shape[0] ** 0.5 + 0.2 * lq + 0.5 * lqd
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        for k_, val in (('mil', mil), ('pair', lpair), ('dec', dec_loss(z, v)), ('decflow', ldecf), ('q', lq), ('qdec', lqd)): tot[k_] = tot.get(k_, 0) + float(val) * len(v)
    print(f'epoch {ep+1} ' + ' '.join(f'{k_} {val/len(perm):.4f}' for k_, val in tot.items()) + f' ({time.time()-t0:.0f}s)', flush=True)
torch.save(dict(enc=enc.state_dict(), flow=flow.state_dict(), dec=dec.state_dict(), q=qn.state_dict(), h22=h22.state_dict(), h25=h25.state_dict()), f'{D}/model.pt')
# ---------------- evaluation on held-out plots (last visit of each plot): vegetation summaries from satellite features
def summaries(lp, val):
    """slots → plot summaries: TPH, BA (m²/ha), QMD (cm), share of the dominant key; presence by probability > 0.5 (expected values)."""
    pr = torch.sigmoid(lp); N = torch.exp(val[..., 0]) * pr; Dq = torch.exp(val[..., 1])
    tph = N.sum(1); ba = (N * math.pi / 4 * (Dq / 100) ** 2).sum(1); qmd = torch.sqrt((N * Dq ** 2).sum(1) / tph.clamp(min=1e-6))
    bk = (N * Dq ** 2).view(-1, NK, NA).sum(-1); return tph, ba, qmd, bk.argmax(1)
def obs_summ(v):
    m = slot_p[v]; N = torch.exp(slot_v[v][..., 0]) * m; Dq = torch.exp(slot_v[v][..., 1])
    tph = N.sum(1); ba = (N * math.pi / 4 * (Dq / 100) ** 2).sum(1); qmd = torch.sqrt((N * Dq ** 2).sum(1) / tph.clamp(min=1e-6))
    bk = (N * Dq ** 2).view(-1, NK, NA).sum(-1); return tph, ba, qmd, bk.argmax(1)
last = V.loc[V.groupby('pid').year.idxmax()]; te_last = last[last.pid.isin(test_pid & have)].vid.values; tr_last = last[last.pid.isin(have - test_pid)].vid.values
def plot_feat(pid):    # prior-weighted mean candidate features (the plot's satellite description under location uncertainty)
    cid, pr = cand_of[pid]; X = torch.cat([AEF[cid], DIST[cid].flatten(1), SOIL[cid]], -1); return (X * torch.tensor(pr, dtype=torch.float32).unsqueeze(-1)).sum(0)
with torch.no_grad():
    Xte = torch.stack([plot_feat(vpid[v]) for v in te_last]); Xtr = torch.stack([plot_feat(vpid[v]) for v in tr_last])
    def pred_plot(pid):   # per-candidate prediction, prior-weighted in log space (TPH, BA, QMD) and by vote (dominant key)
        cid, pr = cand_of[pid]; X = torch.cat([AEF[cid], DIST[cid].flatten(1), SOIL[cid]], -1); qm_, _ = qn(X); s_ = summaries(*dec(qm_)); w_ = torch.tensor(pr, dtype=torch.float32)
        return [torch.exp((torch.log(s_[i] + 1e-3) * w_).sum()) for i in range(3)] + [torch.bincount(s_[3], weights=w_, minlength=NK).argmax()]
    pp = [pred_plot(vpid[v]) for v in te_last]; m_t = [torch.stack([p_[i] for p_ in pp]) for i in range(4)]; o_t = obs_summ(torch.tensor(te_last))
    # kNN baseline (k = 5) in standardised feature space on the training plots' last visits
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; dd = torch.cdist((Xte - mu) / sd, (Xtr - mu) / sd); nn5 = dd.topk(5, largest=False).indices
    o_tr = obs_summ(torch.tensor(tr_last)); k_t = [o_tr[i][nn5].float().mean(1) if i < 3 else torch.mode(o_tr[3][nn5], 1).values for i in range(4)]
    def err(p, o): return float(torch.median(torch.abs(torch.log((p + 1e-3) / (o + 1e-3)))))
    ev = {'n_test_plots': len(te_last)}
    for i, nm in enumerate(['TPH', 'BA', 'QMD']):
        ev[nm] = {'model_median_abs_log_err': err(m_t[i], o_t[i]), 'knn_median_abs_log_err': err(k_t[i], o_t[i]),
                  'model_bias_log': float(torch.median(torch.log((m_t[i] + 1e-3) / (o_t[i] + 1e-3)))), 'knn_bias_log': float(torch.median(torch.log((k_t[i] + 1e-3) / (o_t[i] + 1e-3))))}
    ev['dominant_key_acc'] = {'model': float((m_t[3] == o_t[3]).float().mean()), 'knn': float((k_t[3] == o_t[3]).float().mean()),
                              'majority_class': float((o_t[3] == torch.mode(o_tr[3]).values).float().mean())}
    # localisation check: plots with dated FIA events — posterior mass (uniform prior) on prior-favoured (event-matching) candidates vs their share
    loc = []
    for v in te_v[:400]:
        p = vpid[v]; cid, pr = cand_of[p]
        if pr.max() / pr.min() < 4: continue                                    # no event information for this plot
        z = enc(torch.tensor([v]), vyear[[v]]).repeat(len(cid), 1); st = flow(z, vyear[[v]].repeat(len(cid)), [2022, 2025], DIST[cid], SOIL[cid])
        dd_ = sum(((hh(st[y]) - AEF[cid][:, sl]) ** 2).sum(-1) * float(vyear[v] <= y + 1) for y, hh, sl in ((2022, h22, A22), (2025, h25, A25)))
        post = torch.softmax(-dd_ / TEMP, 0).numpy(); fav = pr >= np.quantile(pr, 0.9)
        loc.append((post[fav].sum(), fav.mean()))
    if loc:
        a = np.array(loc); ev['localisation'] = {'plots': len(a), 'posterior_mass_on_event_matching_candidates': float(a[:, 0].mean()), 'their_share_of_candidates': float(a[:, 1].mean())}
print(json.dumps(ev, indent=1)); json.dump(ev, open(f'{D}/eval.json', 'w'), indent=1)
# ---------------- map Baker stands: K_S samples per footprint → cohort slots → stand summaries (mean of footprints)
S = pd.read_parquet(f'{D}/stands.parquet'); SF = np.load(f'{D}/stand_feat.npy').astype(np.float32); SF[:, SO] = (SF[:, SO] - smu) / ssd
with torch.no_grad():
    X = torch.tensor(np.concatenate([SF[:, :128], SF[:, DS], SF[:, SO]], 1)); qm, qls = qn(X)
    res = []
    for k_ in range(8):
        z = qm + torch.exp(qls) * torch.randn_like(qm); lp, val = dec(z); tph, ba, qmd, dk = summaries(lp, val); res.append(torch.stack([tph, ba, qmd, dk.float()], 1))
    R_ = torch.stack(res)                                                     # [samples, footprints, 4]
S['tph'] = R_[:, :, 0].mean(0).numpy(); S['ba'] = R_[:, :, 1].mean(0).numpy(); S['qmd'] = R_[:, :, 2].mean(0).numpy()
S['ba_sd'] = R_[:, :, 1].std(0).numpy(); S['dom'] = torch.mode(R_[:, :, 3].long(), 0).values.numpy()
out = S.groupby('stand_id').agg(tph=('tph', 'mean'), ba=('ba', 'mean'), qmd=('qmd', 'mean'), ba_sd=('ba_sd', 'mean'), dom=('dom', lambda x: x.mode().iloc[0])).reset_index()
out['dom_key'] = [M['keys'][int(i)] for i in out.dom]; out.to_parquet(f'{D}/stand_veg.parquet')
print('stands mapped', len(out), out[['tph', 'ba', 'qmd', 'ba_sd']].describe().loc[['mean', '50%']].round(2).to_dict(), out.dom_key.value_counts().head(6).to_dict())
