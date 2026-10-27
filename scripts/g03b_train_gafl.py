"""Generative-vegetation prototype v2 (DATA=gafl: all GA+FL plots; curriculum on well-localised plots; class-weighted species presence; composition
metrics). Derived from g03_train_proto.py, step 2: temporal JEPA model (liquid flow + disturbance jumps + SIGReg) — train, evaluate, map.
  E_v      masked set encoder over cohort tokens (species, age class, log N, log QMD, H, CR) → z (D_Z), SIGReg-regularised (LeJEPA: Epps-Pulley test of
           random 1-D projections against N(0,1))
  flow     liquid time-constant cell, yearly steps: dz = −(1/τ + g)⊙z + g⊙A, g = σ(W [z, s] + b) (s = static site covariates), from the visit year on
  jump     at a candidate's disturbance years: z ← z + J([z, class one-hot, severity, Hansen loss])
  heads    z(2022) → AlphaEarth 2022, z(2025) → AlphaEarth 2025 (64-d)
  MIL      the plot location is latent: K candidates sampled ∝ the disturbance prior; loss = −logsumexp_c(−d_c / T) (+ log K)
  pairs    FIA remeasurement: flow(z(v1)) along candidate histories CONSISTENT WITH THE PLOT'S FIA RECORDS (TRT / DSTRB between the visits) to v2's
           date ≈ z(v2) (stop-gradient on the target), and decoded ≈ v2's cohorts — the FIA records anchor the disturbance jumps
  decoder  z → cohort slots (species x age class): presence (BCE) + log N, log QMD, H, CR (masked MSE)
  Q        inverse encoder: candidate features (AlphaEarth 2022 + 2025, disturbance history, soils) → z(2025) (Gaussian) — a direct predictor evaluated here; the stand mapping is g04 (landscape prior × likelihood) + g05
Held-out: plots of 20 % of 10-km blocks. Baseline: kNN on the prior-weighted mean candidate features. Writes data/proto/baker/{model.pt, eval.json, stand_veg.parquet}."""
import os, sys, json, math, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
SEED = int(os.environ.get('SEED', 0)); torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(int(os.environ.get('NT', 6)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # also when exec'd by g04 / g05 / gv_verify (same scripts directory)
D = f'{ROOT}/data/proto/' + os.environ.get('DATA', 'gafl'); M = json.load(open(f'{D}/meta.json'))
sys.path.insert(0, os.path.join(ROOT, 'scripts')); from region import region_check; region_check(M)
V = pd.read_parquet(f'{D}/visits.parquet'); TK = pd.read_parquet(f'{D}/tokens.parquet'); PR = pd.read_parquet(f'{D}/pairs.parquet')
CF16 = os.environ.get('CF16', '0') == '1'      # keep the candidate features float16 in memory (Region 4: ~4 M candidates), cast to float32 when indexed
C = pd.read_parquet(f'{D}/cands.parquet'); CF = np.load(f'{D}/cand_feat.npy', mmap_mode='r') if CF16 else np.load(f'{D}/cand_feat.npy').astype(np.float32)
NK, NA = len(M['keys']), 16; YEARS = M['years']; NY = len(YEARS); NC = len(M['classes']); D_Z = int(os.environ.get('DZ', 16))
# ---- feature slices
A22, A25 = slice(0, 64), slice(64, 128); DS = slice(128, 128 + NY * (NC + 2)); SO = slice(128 + NY * (NC + 2), CF.shape[1])
if CF16:
    class F32View:                                 # float16 storage; indexing returns float32 (exact: every float16 is a float32), as the float32 tensors do
        def __init__(self, t): self.t = t
        def __getitem__(self, i): return self.t[i].float()
        def __len__(self): return len(self.t)
    _so = np.asarray(CF[:, SO], dtype=np.float32); smu, ssd = _so.mean(0), _so.std(0) + 1e-6; SOIL = torch.tensor((_so - smu) / ssd); del _so
    DIST = F32View(torch.from_numpy(np.ascontiguousarray(CF[:, DS]).reshape(-1, NY, NC + 2))); AEF = F32View(torch.from_numpy(np.ascontiguousarray(CF[:, :128]))); del CF
else:
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
# AGE_CH = 1 (audit r4 I4: the contracting LTC flow cannot age a stand — decoded mean age +0.1 yr / yr on undisturbed pairs vs +0.87 observed): latent
# dimension 0 is the stand's AGE CLOCK, z0 = (BA-weighted mean age − AGE_C) / AGE_S, supervised from the visit's cohorts (W_AGE), advanced by dt / AGE_S
# per year in the flow (not mean-reverting), reset only by the disturbance jumps; excluded from SIGReg. The gate (and so the other dimensions'
# dynamics) and the decoder read it.
AGE_CH = os.environ.get('AGE_CH', '0') == '1'; AGE_C, AGE_S = 40.0, 40.0; W_AGE = float(os.environ.get('W_AGE', 1.0))
# observed stand age per visit (the AGE_CH target): BA-weighted mean cohort age (N QMD²); a visit without live trees has age 0
_w = TK.N * TK.QMD ** 2; _ag = (_w * TK.A).groupby(TK.vid).sum() / _w.groupby(TK.vid).sum()
AGE_OBS = torch.zeros(nv); AGE_OBS[torch.tensor(_ag.index.values)] = torch.tensor(_ag.values, dtype=torch.float32); AGE_Z = (AGE_OBS - AGE_C) / AGE_S
def bce(lp, tp, reduction='mean'): return Fn.binary_cross_entropy_with_logits(lp, tp, pos_weight=POSW, reduction=reduction)
# ---- candidates per plot (index arrays) and spatial blocks
LOC = os.environ.get('LOC', 'history')
def _attach(C, fn, cols):
    """merge a per-candidate file on (cid, pid) and require it to cover THIS table (cid / pid are row numbers that shift between rebuilds)"""
    X = pd.read_parquet(fn); on = ['cid', 'pid'] if 'pid' in X.columns else ['cid']
    if 'pid' not in X.columns: print(f'WARNING {os.path.basename(fn)} has no pid column: cannot check it belongs to this table (rerun its builder)', flush=True)
    M_ = C.merge(X[on + cols], on=on, how='left', indicator=True)
    return M_.drop(columns='_merge'), float((M_._merge == 'both').mean())
if LOC in ('history', 'aef') and not os.path.exists(f'{D}/cand_hist.parquet'):
    raise SystemExit(f'LOC={LOC} needs {D}/cand_hist.parquet (run scripts/g02g_history_consistency.py after g02b / g02e), or set LOC=prior')
if LOC == 'history' and os.path.exists(f'{D}/cand_hist.parquet'):   # g02g: every visit of a plot draws from the candidates consistent with its WHOLE FIA history
    C, cov_ = _attach(C, f'{D}/cand_hist.parquet', ['cons'])
    if cov_ < 1.0: raise SystemExit(f'cand_hist.parquet covers {cov_:.3f} of the candidates: stale (rerun g02g)')
    C['cons'] = C.cons.fillna(False).astype(bool)
    has_ = C.groupby('pid').cons.transform('any'); keep_ = ~has_ | C.cons                  # plots with an empty consistent set keep their prior
    C['prior0'] = C.prior; C['prior'] = np.where(keep_, C.prior, 0.0); C['prior'] = C.prior / C.groupby('pid').prior.transform('sum'); C['hi'] = np.where(has_, C.cons, C.hi)
    print(f"location: FIA-history-consistent candidates (g02g) for {int(has_.groupby(C.pid).first().sum())} of {C.pid.nunique()} plots", flush=True)
if LOC == 'aef' and os.path.exists(f'{D}/cand_hist.parquet'):      # history-consistent set (as LOC=history) × the joint same-year AlphaEarth posterior (g08, training plots)
    C, cov_ = _attach(C, f'{D}/cand_hist.parquet', ['cons'])
    if cov_ < 1.0: raise SystemExit(f'cand_hist.parquet covers {cov_:.3f} of the candidates: stale (rerun g02g)')
    C['cons'] = C.cons.fillna(False).astype(bool)
    has_ = C.groupby('pid').cons.transform('any'); keep_ = ~has_ | C.cons
    C['prior0'] = C.prior; C['prior'] = np.where(keep_, C.prior, 0.0); C['hi'] = np.where(has_, C.cons, C.hi)
    if os.path.exists(f'{D}/cand_loc_aef.parquet'):
        C, _ = _attach(C, f'{D}/cand_loc_aef.parquet', ['prior_aef']); ha = C.groupby('pid').prior_aef.transform(lambda x: x.notna().any())
        C['prior'] = np.where(ha, C.prior_aef.fillna(0.0), C.prior); print(f'location: + same-year AlphaEarth posterior for {int(ha.groupby(C.pid).first().sum())} training plots', flush=True)
    C['prior'] = C.prior / C.groupby('pid').prior.transform('sum')
elif LOC == 'plot' and os.path.exists(f'{D}/cand_prior.parquet'):   # g02f EM posterior (not used by default: held-out ~0.003 nats / plot of location information)
    C = C.merge(pd.read_parquet(f'{D}/cand_prior.parquet'), on='cid', how='left'); C['prior'] = C.prior_loc.fillna(0.0); C['hi'] = C.hi_loc.fillna(False).astype(bool)
    C['prior'] = C.prior / C.groupby('pid').prior.transform('sum'); print('location: plot-level EM posterior (g02f)', flush=True)
cand_of = {pid: (g.cid.values, g.prior.values) for pid, g in C.groupby('pid')}
cand_hist = {pid: (g.prior0.values, g.cons.values) for pid, g in C.groupby('pid')} if 'cons' in C.columns else {}   # g02b prior + history-consistency flag
cand_iw = {pid: (g.incl_w.values, g.hi.values) for pid, g in C.groupby('pid')}     # inverse inclusion weights (footprints per candidate), high-posterior (credible) flag
P = pd.read_parquet(f'{D}/plots.parquet'); P['blk'] = (P.x // 10000).astype(int) * 1000 + (P.y // 10000).astype(int)
rng = np.random.default_rng(1); blks = P.blk.unique(); test_blk = set(rng.choice(blks, max(1, int(0.2 * len(blks))), replace=False))
test_pid = set(P.pid[P.blk.isin(test_blk)]); have = set(cand_of)
from scipy.spatial import cKDTree as _KD
_te = P[P.pid.isin(test_pid)]; _dn, _ = _KD(_te[['x', 'y']].values).query(P[['x', 'y']].values)
buffer_pid = set(P.pid[(_dn < 3200) & ~P.pid.isin(test_pid)])   # training plots whose 1.6 km disk could overlap a test plot's disk: excluded from training
# VAL_SPLIT = 1 (audit r6 I2): 10 % of the remaining blocks are a VALIDATION set (+ its own 3.2 km buffer), excluded from training — the pre-registered
# dynamics acceptance (g06) and other selection decisions use it, so the test blocks stay a pure held-out report. Default 0 keeps the older splits.
VAL_SPLIT = os.environ.get('VAL_SPLIT', '0') == '1'; val_pid, vbuf_pid = set(), set()
if VAL_SPLIT:
    _rest = sorted(set(blks) - test_blk); val_blk = set(np.random.default_rng(3).choice(_rest, max(1, int(0.1 * len(_rest))), replace=False))
    val_pid = set(P.pid[P.blk.isin(val_blk)]) - buffer_pid
    _va = P[P.pid.isin(val_pid)]; _dv, _ = _KD(_va[['x', 'y']].values).query(P[['x', 'y']].values)
    vbuf_pid = set(P.pid[(_dv < 3200) & ~P.pid.isin(val_pid) & ~P.pid.isin(test_pid)])
MIN_FS = float(os.environ.get('MIN_FSHARE', 0.25))     # data vetting 2026-10-22: per-forested-area densities from < 1/4 of the plot are inflated / noisy (9 % of
fs_ok = (V.fsub.values >= MIN_FS) if 'fsub' in V.columns else np.ones(nv, bool)   # visits with forest share 0.10-0.25 have BA > 60 m2/ha vs 0.5 % when fully forested)
tr_v = np.array([v for v in range(nv) if vpid[v] in have and vpid[v] not in test_pid and vpid[v] not in buffer_pid and vpid[v] not in val_pid and vpid[v] not in vbuf_pid and fs_ok[v]]); te_v = np.array([v for v in range(nv) if vpid[v] in have and vpid[v] in test_pid and fs_ok[v]])
va_v = np.array([v for v in range(nv) if vpid[v] in have and vpid[v] in val_pid and fs_ok[v]], dtype=int)
print(f'visits train {len(tr_v)} test {len(te_v)} val {len(va_v)} | plots test {len(test_pid & have)} val {len(val_pid & have)} | buffer plots excluded {len((buffer_pid | vbuf_pid) & have)} | tokens/visit max {maxT}', flush=True)
pos = slot_p[tr_v].mean(0).clamp(min=1e-4); POSW = ((1 - pos) / pos).sqrt().clamp(max=30)   # per-slot positive weight from TRAINING visits (rare species × age classes count more)
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
# dynamics switches (audit r3 I1: the undisturbed flow reverted to ONE global attractor and shrank stands ~2.4 %/yr over long horizons, FIA +2.3 %/yr):
# SITE_ATTR = 1: the attractor depends on the site (soils) — A(soil) = A + MLP(soil), zero-initialised; PAIR_ALL = 1: every ordered visit pair of a plot
# (multi-horizon, up to PAIR_HMAX yr) trains the flow, not only consecutive visits. (PAIR_SYN — injecting FIA-recorded but undetected events — was
# removed after audit r4 I1: the injected footprints were impossible full clear-cuts.) Defaults 0 keep the older models (and their checkpoints) unchanged.
SITE_ATTR = os.environ.get('SITE_ATTR', '0') == '1'; PAIR_ALL = os.environ.get('PAIR_ALL', '0') == '1'
PAIR_HMAX = float(os.environ.get('PAIR_HMAX', 25.0))
NJ = D_Z + (1 if AGE_CH else 0)                                            # jump-network outputs (AGE_CH: + the clock-reset logit)
class Flow(nn.Module):        # liquid time-constant cell (Hasani et al.): dz = −(1/τ + g) z + g A, exact exponential step over each year's part after the visit, + disturbance jumps
    sparse_j = False
    def __init__(s):
        super().__init__(); ns = SOIL.shape[1]
        s.g = nn.Sequential(nn.Linear(D_Z + ns, 64), nn.Tanh(), nn.Linear(64, D_Z)); s.A = nn.Parameter(torch.zeros(D_Z)); s.ltau = nn.Parameter(torch.full((D_Z,), 2.0))
        s.J = nn.Sequential(nn.Linear(D_Z + NC + 2, 64), nn.GELU(), nn.Linear(64, NJ))
        if AGE_CH: s.lrho = nn.Parameter(torch.zeros(1))                  # (AGE_CH) log clock rate: the BA-weighted mean age advances < 1 yr / yr (ingrowth, mortality)
        if SITE_ATTR:
            s.Am = nn.Sequential(nn.Linear(ns, 64), nn.Tanh(), nn.Linear(64, D_Z)); nn.init.zeros_(s.Am[2].weight); nn.init.zeros_(s.Am[2].bias)
    def jump(s, z, ev):
        """bounded disturbance jump of the state; with AGE_CH, dimension 0 (the age clock) is RESET toward age 0 by a learned fraction r (a clear-cut of a
        100-yr stand needs −2.5 on the clock, beyond any bounded additive jump: audit r5 I1) plus a small bounded shift (thinning from below raises the age)"""
        o = s.J(torch.cat([z, ev], -1)); jd = torch.tanh(o[:, :D_Z])
        if not AGE_CH: return jd
        r = torch.sigmoid(o[:, D_Z:D_Z + 1]); return torch.cat([r * (-AGE_C / AGE_S - z[:, :1]) + 0.25 * jd[:, :1], jd[:, 1:]], 1)
    def forward(s, z, start, stop, dist, soil, end=None, rs=None, re=None, ra=None):
        """z [n, D_Z] at time `start` [n] (decimal year) → state at the end of each year (or at `end` [n], after which it is frozen), recorded at `stop`
        years; dist [n, NY, NC+2]. An event of year y applies iff mid-year y lies in [start, end] — except in a VISIT's own calendar year when that visit
        recorded an FIA event of that year (rs / re [n] bool: the start / end visit did): the year's evidence then lies BEFORE that visit (audit r5 I3);
        and ra [n] bool: a LATER visit recorded a new event dated in the start visit's year, which therefore lies AFTER the start (audit r7 I2) —
        unless the start visit itself recorded an own-year event (rs): then the year's evidence (e.g. the cut) predates the start and is not re-applied
        (r8 M5; the later record's own evidence of that year, e.g. site preparation, is lost: 0.2 % of pairs).
        Returns dict stop->state."""
        end = torch.full_like(start, 1e9) if end is None else end
        rs = torch.zeros_like(start, dtype=torch.bool) if rs is None else rs; re = torch.zeros_like(start, dtype=torch.bool) if re is None else re
        ra = torch.zeros_like(start, dtype=torch.bool) if ra is None else ra
        fs_, fe_ = torch.floor(start), torch.floor(end)
        y0 = int(math.floor(start.min().item())); out = {st: z for st in stop if st < max(y0, YEARS[0])}   # stops before the run: initial state (masked by the caller)
        Aa = s.A + s.Am(soil) if SITE_ATTR else s.A                          # (SITE_ATTR: the site's attractor; soil is constant along the run)
        for j, y in enumerate(range(max(y0, YEARS[0]), YEARS[-1] + 1)):
            dt = (torch.clamp(end, max=y + 1) - torch.clamp(start, min=y)).clamp(0, 1).unsqueeze(-1)   # part of year y inside [start, end]
            jj = y - YEARS[0]
            g = torch.sigmoid(s.g(torch.cat([z, soil], -1)))
            lam = 1 / torch.exp(s.ltau) + g; zs = g * 3 * torch.tanh(Aa) / lam   # exact (exponential) integration over dt; attractor bounded (|A| ≤ 3)
            zn = zs + (z - zs) * torch.exp(-lam * dt)
            z = torch.cat([z[:, :1] + dt * torch.exp(s.lrho) / AGE_S, zn[:, 1:]], 1) if AGE_CH else zn    # (AGE_CH: dimension 0 is the age clock)
            act = ((((start <= y + 0.5) & ~(rs & (fs_ == y))) | (ra & ~rs & (fs_ == y))) & ((end >= y + 0.5) | (re & (fe_ == y)))).float().unsqueeze(-1)   # visit-year rule
            ev = dist[:, jj]; has = (ev[:, :NC].sum(-1, keepdim=True) + ev[:, NC + 1:NC + 2] > 0).float() * act
            if s.sparse_j:                                                  # (v4) the jump network only on the rows with an event: same result, ~1/20 of the cost
                ix = has.squeeze(-1).nonzero(as_tuple=True)[0]
                if len(ix): z = z.index_add(0, ix, s.jump(z[ix], ev[ix]))
            else: z = z + has * s.jump(z, ev)                               # bounded jump
            for st in stop:
                if y == st: out[st] = z
        return out
class Dec(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Linear(D_Z, 128), nn.GELU(), nn.Linear(128, NK * NA * 5))
    def forward(s, z): o = s.f(z).view(-1, NK * NA, 5); return o[..., 0], o[..., 1:]
def calibrated(lp): return lp - torch.log(POSW)                      # undo the class weighting for probabilities (training used pos_weight)
class Q(nn.Module):           # satellite + disturbance history + soils → z(2025)
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Linear(128 + NY * (NC + 2) + SOIL.shape[1], 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 2 * D_Z))
    def forward(s, x): o = s.f(x); return 3 * torch.tanh(o[:, :D_Z] / 3), o[:, D_Z:].clamp(-4, 1)
# ---- ARCH=v4 (docs/GENVEG_V4_R4.md §A, user 2026-10-25: deeper encoder / jump / decoder, non-linear satellite heads). Same interfaces as above, so every
# stage that execs this file (g04, g05, g06, gv_verify) gets the architecture of the train env. Default ARCH=v3 = the classes above, unchanged.
ARCH = os.environ.get('ARCH', 'v3')
class ResBlock(nn.Module):    # pre-LayerNorm residual MLP block
    def __init__(s, d, h=None):
        super().__init__(); h = h or 2 * d; s.ln = nn.LayerNorm(d); s.f = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
    def forward(s, x): return x + s.f(s.ln(x))
class SAB(nn.Module):         # pre-LN self-attention block over the cohort tokens (key padding mask)
    def __init__(s, d=128, nh=4):
        super().__init__(); s.l1 = nn.LayerNorm(d); s.at = nn.MultiheadAttention(d, nh, batch_first=True); s.ff = ResBlock(d)
    def forward(s, x, pad):
        h = s.l1(x); x = x + s.at(h, h, h, key_padding_mask=pad, need_weights=False)[0]; return s.ff(x)
class EncV4(nn.Module):       # Set Transformer: token embedding -> 4 SAB -> PMA (1 learned seed) + year -> MLP -> z
    def __init__(s, d=128, L=4):
        super().__init__(); s.ek = nn.Embedding(NK, 16); s.ea = nn.Embedding(NA, 8); s.inp = nn.Linear(16 + 8 + 5, d)
        s.cls = nn.Parameter(torch.zeros(1, 1, d))               # always-valid token: visits without trees (fresh clear-cuts) get a defined encoding
        s.blocks = nn.ModuleList([SAB(d) for _ in range(L)]); s.seed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        s.pma = nn.MultiheadAttention(d, 4, batch_first=True); s.lo = nn.LayerNorm(d)
        s.out = nn.Sequential(nn.Linear(d + 1, d), nn.GELU(), ResBlock(d), nn.Linear(d, D_Z))
    def forward(s, v, yr):
        x = s.inp(torch.cat([s.ek(tok_k[v]), s.ea(tok_a[v]), tok_x[v]], -1)); n = x.shape[0]
        x = torch.cat([s.cls.expand(n, -1, -1), x], 1); pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool), tok_m[v] == 0], 1)
        for b in s.blocks: x = b(x, pad)
        p = s.pma(s.seed.expand(n, -1, -1), s.lo(x), s.lo(x), key_padding_mask=pad, need_weights=False)[0].squeeze(1)
        r = s.out(torch.cat([p, ((yr - 2012) / 10).unsqueeze(-1)], -1))
        return 3 * torch.tanh(r / 3)
class FlowV4(Flow):           # same liquid time-constant cell; deeper gate; residual jump network ending in a Linear (gv_verify T-checks use J[-1])
    sparse_j = True
    def __init__(s):
        super().__init__(); ns = SOIL.shape[1]
        s.g = nn.Sequential(nn.Linear(D_Z + ns, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, D_Z))
        s.J = nn.Sequential(nn.Linear(D_Z + NC + 2, 128), *[ResBlock(128) for _ in range(4)], nn.LayerNorm(128), nn.Linear(128, NJ))   # class one-hot -> embedding via the input layer
class DecV4(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Linear(D_Z, 256), *[ResBlock(256) for _ in range(4)], nn.LayerNorm(256), nn.Linear(256, NK * NA * 5))
    def forward(s, z): o = s.f(z).view(-1, NK * NA, 5); return o[..., 0], o[..., 1:]
class HeadV4(nn.Module):      # satellite head: linear skip (keeps the latent tied to AlphaEarth space for the MIL distance) + 2-layer MLP
    def __init__(s):
        super().__init__(); s.lin = nn.Linear(D_Z, 64); s.mlp = nn.Sequential(nn.Linear(D_Z, 128), nn.GELU(), nn.Linear(128, 64))
    def forward(s, z): return s.lin(z) + s.mlp(z)
if ARCH == 'v4':
    enc, flow, dec, qn = EncV4(), FlowV4(), DecV4(), Q(); h22, h25 = HeadV4(), HeadV4()
elif ARCH == 'v3':
    enc, flow, dec, qn = Enc(), Flow(), Dec(), Q(); h22, h25 = nn.Linear(D_Z, 64), nn.Linear(D_Z, 64)
else: raise ValueError(f'ARCH must be v3 or v4, got {ARCH}')
if AGE_CH:
    with torch.no_grad(): flow.J[-1].bias[D_Z:].fill_(-2.0)                 # clock reset fraction starts small (sigmoid(−2) = 0.12)
print(f'ARCH {ARCH}: D_Z {D_Z}; parameters enc {sum(p.numel() for p in enc.parameters()):,} flow {sum(p.numel() for p in flow.parameters()):,} '
      f'dec {sum(p.numel() for p in dec.parameters()):,} heads {sum(p.numel() for p in [*h22.parameters(), *h25.parameters()]):,}', flush=True)
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
        cid, pr = cand_of[p]; idx.append(np.random.choice(cid, K, replace=True, p=pr))      # iid draws from the prior: the MIL / posterior estimators assume it
    return np.stack([np.resize(i, K) for i in idx])
# observed plot summaries per visit — the quantities the simulation engines / harvest planning use. The slot-wise objectives alone reconstructed them
# poorly (log BA slope 0.24, corr 0.48, even for the encoded visit itself; 2026-10-21), compressing every downstream change ~4x
_N = torch.exp(slot_v[..., 0]) * slot_p; _D = torch.exp(slot_v[..., 1])
OBS_T = _N.sum(1); OBS_B = (_N * math.pi / 4 * (_D / 100) ** 2).sum(1); OBS_Q = torch.sqrt((_N * _D ** 2).sum(1) / OBS_T.clamp(min=1e-6))
_bko = (_N * _D ** 2).view(-1, NK, NA).sum(-1); OBS_S = _bko / _bko.sum(-1, keepdim=True).clamp(min=1e-9)      # observed BA shares by species key
T_PAIR = float(os.environ.get('T_PAIR', 0.1))      # temperature of the mixture over a pair's candidate histories (per-dimension squared latent distance)
SMOKE_B = int(os.environ.get('SMOKE_BATCHES', 0)) if int(os.environ.get('EPOCHS', 30)) > 0 and os.environ.get('LOAD') != '1' else 0   # training only
QD_WIN = float(os.environ.get('QD_WIN', 1.5))     # Q-decode loss: visits within this many years of 2025 (the state Q predicts)
W_SUM = float(os.environ.get('W_SUM', 1.0)); W_COMP = float(os.environ.get('W_COMP', 1.0)); LQ_MAX = math.log(150.0)                # QMD of a slot <= 150 cm (physical cap, also in g04.summaries)
# values of ABSENT slots are otherwise untrained, yet enter the expected summaries through their (small) presence probability: pull them toward the
# slot's training mean (log N, log QMD, H; CR excluded)
_cnt = slot_p[tr_v].sum(0).clamp(min=1); SLOT_MEAN = (slot_v[tr_v] * slot_p[tr_v].unsqueeze(-1)).sum(0) / _cnt.unsqueeze(-1)
_glob = (slot_v[tr_v] * slot_p[tr_v].unsqueeze(-1)).sum((0, 1)) / slot_p[tr_v].sum().clamp(min=1); SLOT_MEAN = torch.where(slot_p[tr_v].sum(0).unsqueeze(-1) > 0, SLOT_MEAN, _glob)
def absent_reg(val, v):
    w = (1 - slot_p[v]).unsqueeze(-1) * torch.tensor([1.0, 1.0, 1.0, 0.0]); return (((val - SLOT_MEAN) ** 2) * w).sum((-1, -2)) / w.sum((-1, -2)).clamp(min=1)
def summ_err(lp, val, v):
    """per-sample mean squared log error of the decoded summaries (TPH, BA, QMD; differentiable expected values with calibrated presence)"""
    pr = torch.sigmoid(calibrated(lp)); N = torch.exp(val[..., 0].clamp(max=12)) * pr; Dq = torch.exp(val[..., 1].clamp(max=LQ_MAX))
    tph = N.sum(-1); ba = (N * math.pi / 4 * (Dq / 100) ** 2).sum(-1); qmd = torch.sqrt((N * Dq ** 2).sum(-1) / tph.clamp(min=1e-6) + 1e-9)
    # QMD is undefined for a bare visit (OBS_T = 0, OBS_Q = 0 by construction): its QMD term is masked (audit 2026-10-26 M3), as composition is below
    e = ((torch.log1p(tph) - torch.log1p(OBS_T[v])) ** 2 + (torch.log(ba + 0.1) - torch.log(OBS_B[v] + 0.1)) ** 2 + (torch.log1p(qmd) - torch.log1p(OBS_Q[v])) ** 2 * (OBS_T[v] > 0).float()) / 3
    if W_COMP > 0:      # species composition: Bray-Curtis of the BA shares by species key (stocked visits); the totals loss alone degraded composition
        bk = (N * Dq ** 2).view(*N.shape[:-1], NK, NA).sum(-1); bk = bk / bk.sum(-1, keepdim=True).clamp(min=1e-9)
        e = e + W_COMP * (OBS_S[v] - bk).abs().sum(-1) / 2 * (OBS_B[v] > 0).float()
    return e
def dec_loss(z, v):
    lp, val = dec(z); bce_ = bce(lp, slot_p[v])
    m = slot_vm[v] * slot_p[v].unsqueeze(-1); mse = ((val - slot_v[v]) ** 2 * m).sum() / m.sum().clamp(min=1)
    return bce_ + mse + W_SUM * summ_err(lp, val, v).mean() + 0.1 * absent_reg(val, v).mean()
TEMP = 0.05; K = int(os.environ.get('KC', 48)); EP = int(os.environ.get('EPOCHS', 30)); B = int(os.environ.get('BATCH', 64)); t0 = time.time()
E1 = int(os.environ.get('CURRIC', 4))                                                   # curriculum: first E1 epochs on the best-localised plots only
if os.path.exists(f'{D}/localisation.parquet') and 'n_clusters' in pd.read_parquet(f'{D}/localisation.parquet').columns:   # g02g: <= 50 consistent footprints
    _L = pd.read_parquet(f'{D}/localisation.parquet'); cq = 50.0; good = set(_L.index[(_L.n_hi > 0) & (_L.n_hi <= 50)])
else:
    cq = P.conc.quantile(0.7) if 'conc' in P.columns else 0; good = set(P.pid[P.conc >= cq]) if 'conc' in P.columns else set(P.pid)
tr_good = np.array([v for v in tr_v if vpid[v] in good]); print(f'curriculum: {E1} epochs on {len(tr_good)} visits of the best-anchored plots (criterion {cq:.3f})', flush=True)
vpair = {a: b for a, b in zip(PR.v1.values, PR.v2.values) if fs_ok[a] and fs_ok[b]}       # both visits must pass the forest-share filter
# ---- FIA-anchored remeasurement pairs: the plot's own COND.TRT* / COND.DSTRB* records (true location, g02e) select the candidate histories the flow may
# take between v1 and v2. Every detectable FIA event cluster in the period (likelihood ratio >= 4, g02b.EVK / clusters) must be matched by compatible
# LANDFIRE / Hansen evidence in its window; if v2 carries no code that could hide a stand-replacing event, the history must have none in between
# (harvest class or Hansen loss on >= half the footprint). Candidates are then drawn from the location prior restricted to that set.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import g02b_build_gafl as G2
FEV = pd.read_parquet(f'{D}/fia_events.parquet')
if 'rmd' not in FEV.columns: sys.exit('fia_events.parquet has no recording visit (rmd): rerun g02e_fia_records.py (audit r5 I3)')
_rmd = pd.to_datetime(FEV.rmd); FEV['ry'] = _rmd.dt.year + (_rmd.dt.dayofyear - 0.5) / np.where(_rmd.dt.is_leap_year, 366.0, 365.0)   # = visits.year of that date
ev_of = {p: list(zip(g.kind, g.year.astype(int), g.ry)) for p, g in FEV.groupby('pid')}      # (kind, event year, decimal year of the RECORDING visit)
vblk = V.fia_blocked.values if 'fia_blocked' in V.columns else np.ones(nv, bool); vyr = V.year.values
if 'fia_gap' in V.columns: vblk = vblk | V.fia_gap.values    # FIA's previous measurement is not the previous table visit: the codes do not cover the pair (r6 M1)
# a visit that recorded an FIA event of its OWN calendar year: that year's evidence lies before the visit (Flow rs / re; audit r5 I3)
_day = lambda x: pd.to_datetime(x).values.astype('datetime64[D]').astype(np.int64)                 # date -> day number (unit-safe key)
_vk = dict(zip(zip(V.pid.values, _day(V.md)), range(nv))) if 'md' in V.columns else {}
VREC = torch.zeros(nv, dtype=torch.bool)
for _p, _k, _y, _d in zip(FEV.pid.values, FEV.kind.values, FEV.year.values, _day(_rmd)):
    _v = _vk.get((_p, _d))
    if _v is not None and int(_y) == int(math.floor(vyr[_v])): VREC[_v] = True
def recs_in(p_, y1, y2):
    """typed FIA records of plot p_ that describe the interval (y1, y2]: recorded at a visit in it (FIA codes refer to the period since the previous
    visit); a re-report of an event already recorded at a visit <= y1 is dropped (audit r5 I3: the mid-year rule misassigned 3 % of records)."""
    ev = ev_of.get(p_, []); old = {(k, y) for k, y, ry in ev if ry <= y1 + 1e-9}
    return sorted({(k, y) for k, y, ry in ev if y1 + 1e-9 < ry <= y2 + 1e-9 and (k, y) not in old}, key=lambda e: (e[1], e[0]))
def act_span(y1, y2, rs, re, ra=False):
    """first / last calendar year whose events the flow applies between visits at y1 and y2 (Flow's visit-year rule; b < a: none)"""
    f1, f2 = int(math.floor(y1)), int(math.floor(y2))
    return (f1 if (not rs and (y1 <= f1 + 0.5 or ra)) else f1 + 1), (f2 if (y2 >= f2 + 0.5 or re) else f2 - 1)
def pair_ra(v1, v2):
    """a record of the interval (v1, v2] dated in v1's calendar year: that event lies after v1 (Flow ra; audit r7 I2)"""
    f1 = int(math.floor(vyr[v1])); return any(y == f1 for _, y in recs_in(vpid[v1], float(vyr[v1]), float(vyr[v2])))
# the same for a flow from a visit with no end visit (MIL, Q, B): any later visit's new record dated in the visit's year
VRA = torch.tensor([any(y == int(math.floor(vyr[v])) for _, y in recs_in(vpid[v], float(vyr[v]), 1e9)) for v in range(nv)], dtype=torch.bool)
def pair_clusters(p_, y1, y2, a, b):
    """event clusters of the interval with their windows clipped to the years the flow applies ([a, b]); a record whose window falls outside them (FIA
    year before the previous visit) is clamped to the nearest applied year — FIA says it happened in the interval"""
    out = []
    if b < a: return out
    for c in G2.clusters(recs_in(p_, y1, y2)):
        lo, hi = max(c['lo'], a), min(c['hi'], b)
        if hi < lo: lo = hi = (a if c['hi'] < a else b)
        out.append(dict(c, lo=lo, hi=hi))
    return out
pair_cand = {}; nanch = {'with_event': 0, 'event_matched': 0, 'neg_constrained': 0, 'fallback_all': 0}
SEVER = os.environ.get('SEVERITY_ANCHOR', '1') in ('1', '2') and os.environ.get('LOAD') != '1'; SEVER2 = os.environ.get('SEVERITY_ANCHOR', '1') == '2' and SEVER   # outcome-informed candidate selection (training only, never in evaluation)
_tb = TK.assign(ba=TK.N * math.pi / 4 * (TK.QMD / 100) ** 2).groupby('vid').ba.sum(); BAv = dict(zip(_tb.index.values, _tb.values))
PAIR = os.environ.get('PAIR_ANCHOR', 'both')    # 'both': plot-history set ∩ this period's consistent set, else the period set, else the prior; 'plot' | 'period'
def _narrow(m, new, tag):
    """apply a constraint only if candidates remain (a constraint that would empty the set is skipped and counted, the earlier ones are kept)"""
    if (m & new).any(): return m & new
    nanch[tag + '_skipped'] = nanch.get(tag + '_skipped', 0) + 1; return m
def pair_set(v1, v2, blk=None, span=None):
    """(candidates, weights) of the history between visits v1 < v2 of a plot (None: plot without candidates). blk: a visit in (v1, v2] carries an FIA
    code that could hide a stand-replacing event (default: v2's own, the consecutive case); span: the visits in (v1, v2] (default [v2]) — the severity
    rule reads the MINIMUM BA ratio over them (over 15-25 yr a clear-cut regrows above 0.6 of v1: audit r4 I2). Records are those RECORDED in (v1, v2],
    windows clipped to the years the flow applies (audit r5 I3 / M2)."""
    blk = vblk[v2] if blk is None else blk; span = [v2] if span is None else span
    p_ = vpid[v1]
    if p_ not in cand_of: return None
    if PAIR == 'plot': return cand_of[p_]
    cid, pr = cand_of[p_]; y1, y2 = float(vyr[v1]), float(vyr[v2]); per = DIST[cid].numpy(); m = np.ones(len(cid), bool)
    ya, yb = act_span(y1, y2, bool(VREC[v1]), bool(VREC[v2]), pair_ra(v1, v2)); a, b = max(ya - YEARS[0], 0), min(yb - YEARS[0], NY - 1)    # applied years (index, inclusive)
    pcl = pair_clusters(p_, y1, y2, ya, yb); cl_ = [c for c in pcl if c['lr'] >= 4]; sk0 = nanch.get('event_skipped', 0)
    for c in cl_:                                       # each detectable FIA event cluster of the period: compatible evidence in its window
        lo, hi = max(c['lo'] - YEARS[0], 0), min(c['hi'] - YEARS[0], NY - 1) + 1
        if hi > lo:
            ok_ = (per[:, lo:hi][:, :, sorted(c['cols'])].sum(-1) > 0).any(1)
            m = _narrow(m, ok_, 'event')
    nanch['with_event'] += bool(cl_); nanch['event_matched'] += bool(cl_) and nanch.get('event_skipped', 0) == sk0
    r_ = min(BAv.get(u, 0.0) for u in span) / max(BAv.get(v1, 0.0), 1e-6) if BAv.get(v1, 0.0) > 1.0 else None; loss_applied = False
    if cl_ and SEVER and r_ is not None:   # training only: FIA's measured outcome says how severe the recorded event was
        srw = np.zeros(len(cid), bool)
        for c in cl_:
            lo, hi = max(c['lo'] - YEARS[0], 0), min(c['hi'] - YEARS[0], NY - 1) + 1
            if hi > lo: srw |= (np.maximum(per[:, lo:hi, G2.IX['harvest']], per[:, lo:hi, G2.H_]) >= 0.5).any(1)
        if r_ < 0.3: m = _narrow(m, srw, 'replacing'); loss_applied = True; nanch['replacing'] = nanch.get('replacing', 0) + 1
        elif r_ > 0.6: m = _narrow(m, ~srw, 'partial'); nanch['partial'] = nanch.get('partial', 0) + 1
    if SEVER2 and not cl_ and r_ is not None and b >= a:   # training only, pairs without a detectable record: the measured outcome alone
        srw = (np.maximum(per[:, a:b + 1, G2.IX['harvest']], per[:, a:b + 1, G2.H_]) >= 0.5).any(1)
        if r_ < 0.3: m = _narrow(m, srw, 'outcome_loss'); loss_applied = True; nanch['outcome_loss'] = nanch.get('outcome_loss', 0) + 1
        elif r_ > 0.6: m = _narrow(m, ~srw, 'outcome_kept'); nanch['outcome_kept'] = nanch.get('outcome_kept', 0) + 1
    collapse = SEVER and r_ is not None and r_ < 0.3     # training only: a measured major loss (min BA ratio < 0.3) is no 'undisturbed' period (r8 M3)
    if not blk and not loss_applied and not collapse and b >= a:   # FIA-undisturbed period (a measured major loss overrides the 'no code' reading)
        m = _narrow(m, ~(np.maximum(per[:, a:b + 1, G2.IX['harvest']], per[:, a:b + 1, G2.H_]) >= 0.5).any(1), 'neg'); nanch['neg_constrained'] += 1
        if not cl_:     # FIA-undisturbed AND no detectable record: the histories must be event-free where any are (audit r4 I3: else the mixture credits
                        # spurious LANDFIRE / Hansen jumps of mislocated footprints, not the undisturbed drift) — EXCEPT the evidence of the period's
                        # recorded weak events (ground fire, insects, ...) inside their windows, which FIA confirms (audit r5 I2)
            ex = np.zeros((NY, NC + 2), bool)
            for c in pcl:
                lo, hi = max(c['lo'] - YEARS[0], 0), min(c['hi'] - YEARS[0], NY - 1) + 1
                if hi > lo: ex[lo:hi, sorted(c['cols'])] = True
            pe = np.where(ex[None], 0.0, per[:, :, :])[:, a:b + 1]
            m = _narrow(m, ~((pe[:, :, :NC].sum(-1) + pe[:, :, NC + 1]) > 0).any(1), 'eventfree')
    pr0 = pr
    if PAIR == 'both' and p_ in cand_hist:
        pr0, cons = cand_hist[p_]
        if (m & cons).any(): m = m & cons; nanch['history_and_period'] = nanch.get('history_and_period', 0) + 1
        else: nanch['period_only'] = nanch.get('period_only', 0) + 1
    w_ = pr[m] if pr[m].sum() > 0 else pr0[m]            # the plot's location prior (as for MIL); outside it (period-only) the g02b prior
    return (cid[m], w_ / w_.sum() if w_.sum() > 0 else np.full(int(m.sum()), 1.0 / m.sum()))
pair_cand = {}
for v1, v2 in vpair.items():
    r_ = pair_set(v1, v2)
    if r_ is not None: pair_cand[v1] = r_
_n_consec = dict(nanch)                                                              # (counters of the consecutive pairs, as logged before PAIR_ALL)
# multi-horizon pairs (PAIR_ALL): every later visit of the plot within PAIR_HMAX yr; a training visit samples one of its pairs uniformly
pairs_of = {v1: [v2] for v1, v2 in vpair.items() if v1 in pair_cand}; pair_cand2 = {(v1, v2): pair_cand[v1] for v1, v2 in vpair.items() if v1 in pair_cand}
if PAIR_ALL:
    for p_, g_ in V.assign(vid=np.arange(nv)).groupby('pid'):
        vs_ = g_.sort_values('year').vid.values
        for i_, a_ in enumerate(vs_):
            for j_ in range(i_ + 2, len(vs_)):
                b_ = vs_[j_]
                if not (fs_ok[a_] and fs_ok[b_]) or vyr[b_] - vyr[a_] > PAIR_HMAX: continue
                r_ = pair_set(a_, b_, bool(vblk[vs_[i_ + 1:j_ + 1]].any()), list(vs_[i_ + 1:j_ + 1]))   # undisturbed only if no visit in between has a hiding code
                if r_ is not None: pairs_of.setdefault(a_, []).append(b_); pair_cand2[(a_, b_)] = r_
print(f'FIA-anchored pairs {len(pair_cand)} (mode {PAIR}): {_n_consec} | multi-horizon pairs {len(pair_cand2)} (PAIR_ALL {PAIR_ALL})', flush=True)
if os.environ.get('LOAD') == '1':
    sd = torch.load(f"{D}/{os.environ.get('MODEL', 'model.pt')}"); enc.load_state_dict(sd['enc']); flow.load_state_dict(sd['flow']); dec.load_state_dict(sd['dec']); qn.load_state_dict(sd['q'])
    h22.load_state_dict(sd['h22']); h25.load_state_dict(sd['h25']); EP = 0; print('loaded', f"{D}/{os.environ.get('MODEL', 'model.pt')}", flush=True)
for ep in range(EP):
    perm = np.random.permutation(tr_good if ep < E1 else tr_v); tot = {}; nseen = 0
    for b in range(0, len(perm), B):
        if SMOKE_B and b // B >= SMOKE_B: break                               # (SMOKE_BATCHES: a code-path check of the training step only)
        v = torch.tensor(perm[b:b + B]); pids = vpid[v.numpy()]; nseen += len(v)
        z = enc(v, vyear[v]); ci = sample_cands(pids, K)                                                  # [B, K]
        cz = z.repeat_interleave(K, 0); cst = vyear[v].repeat_interleave(K); cidx = torch.tensor(ci.ravel())
        crs = VREC[v].repeat_interleave(K); cra = VRA[v].repeat_interleave(K); st = flow(cz, cst, [2022, 2025], DIST[cidx], SOIL[cidx], rs=crs, ra=cra)
        d = torch.zeros(len(cidx)); n_t = torch.zeros(len(cidx))
        for yr, head, sl in ((2022, h22, A22), (2025, h25, A25)):
            ok = (cst <= yr + 1).float(); pred = head(st[yr]); d = d + ok * ((pred - AEF[cidx][:, sl]) ** 2).sum(-1); n_t = n_t + ok
        d = (d / n_t.clamp(min=1)).view(-1, K)
        mil = -(torch.logsumexp(-d / TEMP, 1) - math.log(K)).mean() * TEMP
        post = torch.softmax(-d.detach() / TEMP, 1)                                                        # posterior over the sampled candidates
        # remeasurement pairs: flow z(v1) along each candidate's history to v2's year
        has2 = [i for i, vv in enumerate(v.tolist()) if vv in pairs_of]; tgt2 = [pairs_of[v[i].item()][np.random.randint(len(pairs_of[v[i].item()]))] for i in has2]
        lpair = torch.tensor(0.0); ldecf = torch.tensor(0.0)
        if has2:
            v2 = torch.tensor(tgt2); y2 = vyear[v2]; z2 = enc(v2, y2).detach()
            sel = torch.tensor(has2); zz = z[sel].repeat_interleave(K, 0)
            c2 = torch.tensor(np.concatenate([np.random.choice(pair_cand2[(v[i].item(), t_)][0], K, replace=True, p=pair_cand2[(v[i].item(), t_)][1]) for i, t_ in zip(has2, tgt2)]))
            d2 = DIST[c2]   # FIA-consistent histories
            yy = torch.floor(y2).long().clamp(YEARS[0], YEARS[-1]); outs = []
            stp = sorted(set(yy.tolist())); st2 = flow(zz, vyear[v[sel]].repeat_interleave(K), stp, d2, SOIL[c2], end=y2.repeat_interleave(K),
                                                rs=VREC[v[sel]].repeat_interleave(K), re=VREC[v2].repeat_interleave(K),
                                                ra=torch.tensor([pair_ra(v[i].item(), t_) for i, t_ in zip(has2, tgt2)]).repeat_interleave(K))   # state AT v2's date
            fz = torch.stack([st2[int(yy[i // K])][i] for i in range(len(c2))]).view(len(has2), K, D_Z)
            # one of the K candidate histories is the plot's: a MIXTURE over them (soft-min, as the MIL), not their average — averaging trained the flow toward
            # the history-mean outcome (22 % of the pairs mix stand-replacing and other histories; audit 2026-10-26 F6); the decoding of the flowed state is
            # weighted by the same (detached) posterior over the histories
            epr = ((fz - z2.unsqueeze(1)) ** 2).mean(-1)                                    # [pairs, K]  (not `ep`: the epoch counter)
            lpair = -(torch.logsumexp(-epr / T_PAIR, 1) - math.log(K)).mean() * T_PAIR; wph = torch.softmax(-epr.detach() / T_PAIR, 1)
            # the decoder must read FLOWED states too: decode the flowed z(v1→v2) against v2's observed cohorts (bare v2 after a cut: no presence)
            lp2, val2 = dec(fz.reshape(-1, D_Z)); tp = slot_p[v2].repeat_interleave(K, 0); tv = slot_v[v2].repeat_interleave(K, 0); tm = slot_vm[v2].repeat_interleave(K, 0) * tp.unsqueeze(-1)
            bce2 = bce(lp2, tp, reduction='none').mean(-1); mse2 = (((val2 - tv) ** 2 * tm).sum((-1, -2)) / tm.sum((-1, -2)).clamp(min=1))
            v2r = v2.repeat_interleave(K); ldecf = ((bce2 + mse2 + W_SUM * summ_err(lp2, val2, v2r) + 0.1 * absent_reg(val2, v2r)).view(len(has2), K) * wph).sum(1).mean()
        else:
            ldecf = torch.tensor(0.0)
        # inverse encoder Q: candidate features → z(2025) (posterior-weighted, stop-gradient)
        tgt = st[2025].detach().view(-1, K, D_Z); qx = torch.cat([AEF[cidx], DIST[cidx].flatten(1), SOIL[cidx]], -1)
        qm, qls = qn(qx); nll = (0.5 * ((tgt.reshape(-1, D_Z) - qm) ** 2 / torch.exp(2 * qls)) + qls).sum(-1).view(-1, K)
        lq = (nll * post).sum(1).mean() / D_Z
        # Q's decode: the decoder is FROZEN here (only Q learns from this term) and only visits within QD_WIN yr of 2025 count — Q predicts z(2025), so an
        # older visit's cohorts are out of date and would teach the decoder 'flowed-2025 latent -> younger stand' (audit 2026-10-26 M2)
        lpq, valq = torch.func.functional_call(dec, {k_: p_.detach() for k_, p_ in dec.named_parameters()}, (qm,)); tpq = slot_p[v].repeat_interleave(K, 0); tvq = slot_v[v].repeat_interleave(K, 0); tmq = slot_vm[v].repeat_interleave(K, 0) * tpq.unsqueeze(-1)
        dq = bce(lpq, tpq, reduction='none').mean(-1) + (((valq - tvq) ** 2 * tmq).sum((-1, -2)) / tmq.sum((-1, -2)).clamp(min=1)) + W_SUM * summ_err(lpq, valq, v.repeat_interleave(K)) + 0.1 * absent_reg(valq, v.repeat_interleave(K))
        wy = (torch.abs(vyear[v] - 2026.0) <= QD_WIN).float()                # visits contemporaneous with z(2025) = the END-2025 state (2026.0)
        dc_ = DIST[cidx]; evy = (dc_[:, :, :NC].sum(-1) + dc_[:, :, NC + 1]) > 0                       # [B*K, years] any candidate evidence
        yrs_ = (torch.arange(NY) + YEARS[0]).float().unsqueeze(0)
        fcs_ = torch.floor(cst).unsqueeze(-1) == yrs_
        after = (evy & ((((cst.unsqueeze(-1) <= yrs_ + 0.5) & ~(crs.unsqueeze(-1) & fcs_))) | ((cra & ~crs).unsqueeze(-1) & fcs_))).any(1).view(-1, K)   # = the flow's rule
        pq = post * (~after).float(); okv = pq.sum(1) > 0; pq = pq / pq.sum(1, keepdim=True).clamp(min=1e-12)   # the visit's cohorts describe 2025 only without a later event
        wy = wy * okv.float(); lqd = ((dq.view(-1, K) * pq).sum(1) * wy).sum() / wy.sum().clamp(min=1e-6)
        ldec = dec_loss(z, v); zr = z[:, 1:] if AGE_CH else z                          # (the age clock is supervised, not regularised toward N(0, 1))
        lage = ((z[:, 0] - AGE_Z[v]) ** 2).mean() if AGE_CH else torch.tensor(0.0)
        loss = mil + 0.5 * lpair + ldec + 0.5 * ldecf + 1.0 * sigreg(zr) * z.shape[0] ** 0.5 + 0.2 * lq + 0.5 * lqd + W_AGE * lage
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        for k_, val in (('mil', mil), ('pair', lpair), ('dec', ldec), ('decflow', ldecf), ('q', lq), ('qdec', lqd)) + ((('age', lage),) if AGE_CH else ()): tot[k_] = tot.get(k_, 0) + float(val.detach()) * len(v)
    print(f'epoch {ep+1} ' + ' '.join(f'{k_} {val/max(nseen, 1):.4f}' for k_, val in tot.items()) + f' ({time.time()-t0:.0f}s)', flush=True)
if SMOKE_B: print('SMOKE_BATCHES: training step ran; no checkpoint, no evaluation', flush=True); sys.exit(0)
os.environ.get('LOAD') == '1' or torch.save(dict(enc=enc.state_dict(), flow=flow.state_dict(), dec=dec.state_dict(), q=qn.state_dict(), h22=h22.state_dict(), h25=h25.state_dict()), f"{D}/{os.environ.get('MODEL', 'model.pt')}")
# ---------------- evaluation on held-out plots (last visit of each plot): vegetation summaries from satellite features
def summaries(lp, val):
    """slots → plot summaries: TPH, BA (m²/ha), QMD (cm), the dominant key, BA shares by key — EXPECTED values (slot density × calibrated presence probability)."""
    pr = torch.sigmoid(calibrated(lp)); N = torch.exp(val[..., 0].clamp(max=12)) * pr; Dq = torch.exp(val[..., 1].clamp(max=math.log(150.0)))
    tph = N.sum(1); ba = (N * math.pi / 4 * (Dq / 100) ** 2).sum(1); qmd = torch.sqrt((N * Dq ** 2).sum(1) / tph.clamp(min=1e-6))
    bk = (N * Dq ** 2).view(-1, NK, NA).sum(-1); return tph, ba, qmd, bk.argmax(1), bk / bk.sum(1, keepdim=True).clamp(min=1e-9)
def obs_summ(v):
    m = slot_p[v]; N = torch.exp(slot_v[v][..., 0]) * m; Dq = torch.exp(slot_v[v][..., 1])
    tph = N.sum(1); ba = (N * math.pi / 4 * (Dq / 100) ** 2).sum(1); qmd = torch.sqrt((N * Dq ** 2).sum(1) / tph.clamp(min=1e-6))
    bk = (N * Dq ** 2).view(-1, NK, NA).sum(-1); return tph, ba, qmd, bk.argmax(1), bk / bk.sum(1, keepdim=True).clamp(min=1e-9)
last = V[fs_ok]; last = last.loc[last.groupby('pid').year.idxmax()]; last = last[last.year >= 2019]       # Q predicts the 2025 state: compare with recent visits only (both model and kNN)
te_last = last[last.pid.isin(test_pid & have)].vid.values; tr_last = last[last.pid.isin(set(vpid[tr_v]))].vid.values   # kNN / null pool: TRAINING plots only (no validation self-matches: r10 I-1)
def plot_feat(pid):    # prior-weighted mean candidate features (the plot's satellite description under location uncertainty)
    cid, pr = cand_of[pid]; X = torch.cat([AEF[cid], DIST[cid].flatten(1), SOIL[cid]], -1); return (X * torch.tensor(pr, dtype=torch.float32).unsqueeze(-1)).sum(0)
with torch.no_grad():
    Xte = torch.stack([plot_feat(vpid[v]) for v in te_last]); Xtr = torch.stack([plot_feat(vpid[v]) for v in tr_last])
    def wmedian(x, w):    # weighted median (the geometric mean of the per-candidate predictions was biased low: audit F5b)
        o = torch.argsort(x); cw = torch.cumsum(w[o], 0); return x[o][torch.searchsorted(cw, cw[-1] * 0.5).clamp(max=len(x) - 1)]
    def pred_plot(pid):   # per-candidate prediction, prior-weighted median (TPH, BA, QMD) and by vote (dominant key)
        cid, pr = cand_of[pid]; X = torch.cat([AEF[cid], DIST[cid].flatten(1), SOIL[cid]], -1); qm_, _ = qn(X); s_ = summaries(*dec(qm_)); w_ = torch.tensor(pr, dtype=torch.float32)
        comp = (s_[4] * w_.unsqueeze(-1)).sum(0)
        return [wmedian(s_[i], w_) for i in range(3)] + [comp.argmax(), comp]
    pp = [pred_plot(vpid[v]) for v in te_last]; m_t = [torch.stack([p_[i] for p_ in pp]) for i in range(5)]; o_t = obs_summ(torch.tensor(te_last))
    # kNN baseline (k = 5) in standardised feature space on the training plots' last visits
    # feature groups (AlphaEarth, disturbance history, soils) each scaled to unit total variance: per-dimension z-scores would blow up rare disturbance dims
    gs = [slice(0, 128), slice(128, 128 + NY * (NC + 2)), slice(128 + NY * (NC + 2), Xtr.shape[1])]; mu = Xtr.mean(0); sc = torch.ones(Xtr.shape[1])
    for g_ in gs: sc[g_] = Xtr[:, g_].var(0).sum().sqrt() + 1e-9
    dd = torch.cdist((Xte - mu) / sc, (Xtr - mu) / sc); nn5 = dd.topk(5, largest=False).indices
    o_tr = obs_summ(torch.tensor(tr_last)); k_t = [o_tr[i][nn5].float().median(1).values for i in range(3)] + [None, o_tr[4][nn5].mean(1)]; k_t[3] = k_t[4].argmax(1)
    def metrics(mk):
        bare = o_t[0] <= 0; ev = {'n_test_plots': int(mk.sum()), 'n_bare': int((mk & bare).sum()),
                                  'bare_pred_BA_median_model': float(m_t[1][mk & bare].median()) if (mk & bare).any() else None}
        mk = mk & ~bare                                                         # log errors and composition on stocked visits (bare visits: no dominant species)
        m_ = [x[mk] for x in m_t]; o_ = [x[mk] for x in o_t]; k_ = [x[mk] for x in k_t]
        def err(p, o): return float(torch.median(torch.abs(torch.log((p + 1e-3) / (o + 1e-3)))))
        ev['n_stocked'] = int(mk.sum())
        for i, nm in enumerate(['TPH', 'BA', 'QMD']):
            nul_ = torch.full_like(o_[i], float(o_tr[i][o_tr[0] > 0].float().median()))       # NULL: median of the stocked training visits
            ev[nm] = {'model_median_abs_log_err': err(m_[i], o_[i]), 'knn_median_abs_log_err': err(k_[i], o_[i]), 'null_median_abs_log_err': err(nul_, o_[i]),
                      'model_bias_log': float(torch.median(torch.log((m_[i] + 1e-3) / (o_[i] + 1e-3)))), 'knn_bias_log': float(torch.median(torch.log((k_[i] + 1e-3) / (o_[i] + 1e-3))))}
        def bal_acc(p, o): return float(np.mean([float((p[o == k] == k).float().mean()) for k in torch.unique(o)]))
        def bray(p, o): return float((torch.abs(p - o).sum(1) / 2).mean())
        ev['composition'] = {'bray_curtis_BA_by_species_model': bray(m_[4], o_[4]), 'bray_curtis_knn': bray(k_[4], o_[4]),
                             'dominant_balanced_acc_model': bal_acc(m_[3], o_[3]), 'dominant_balanced_acc_knn': bal_acc(k_[3], o_[3])}
        ev['dominant_key_acc'] = {'model': float((m_[3] == o_[3]).float().mean()), 'knn': float((k_[3] == o_[3]).float().mean()),
                                  'majority_class': float((o_[3] == torch.mode(o_tr[3][o_tr[0] > 0]).values).float().mean())}
        return ev
    bk_set = set(map(tuple, pd.read_parquet(f'{ROOT}/data/proto/baker/plots.parquet')[['st', 'u', 'c', 'p']].values))
    Pk = P.set_index('pid'); mk_b = torch.tensor([tuple(Pk.loc[vpid[v], ['st', 'u', 'c', 'p']]) in bk_set for v in te_last])
    ev = {'all_heldout': metrics(torch.ones(len(te_last), dtype=torch.bool)), 'baker_50km_heldout': metrics(mk_b) if mk_b.sum() > 5 else None}
    # the same metrics on the VALIDATION plots (VAL_SPLIT): model choices (v3 vs v4, seeds) are made here, the test block above is the report (r9)
    va_last = last[last.pid.isin(val_pid & have)].vid.values if VAL_SPLIT else np.array([], dtype=int)
    if len(va_last) > 5:
        Xte = torch.stack([plot_feat(vpid[v]) for v in va_last]); pp = [pred_plot(vpid[v]) for v in va_last]
        m_t = [torch.stack([p_[i] for p_ in pp]) for i in range(5)]; o_t = obs_summ(torch.tensor(va_last))
        nn5 = torch.cdist((Xte - mu) / sc, (Xtr - mu) / sc).topk(5, largest=False).indices
        k_t = [o_tr[i][nn5].float().median(1).values for i in range(3)] + [None, o_tr[4][nn5].mean(1)]; k_t[3] = k_t[4].argmax(1)
        ev['validation'] = metrics(torch.ones(len(va_last), dtype=torch.bool))
    # localisation check: plots with dated FIA events — posterior mass (uniform prior) on prior-favoured (event-matching) candidates vs their share
    loc = []
    for v in np.random.default_rng(31).choice(te_v, min(400, len(te_v)), replace=False):   # a random sample, not the first plots in key order (one state; r11 M-2)
        p = vpid[v]; cid, pr = cand_of[p]; iw_, hi_ = cand_iw[p]
        if not hi_.any() or hi_.all(): continue                                 # no event information for this plot
        z = enc(torch.tensor([v]), vyear[[v]]).repeat(len(cid), 1); st = flow(z, vyear[[v]].repeat(len(cid)), [2022, 2025], DIST[cid], SOIL[cid], rs=VREC[[v]].repeat(len(cid)), ra=VRA[[v]].repeat(len(cid)))
        nh_ = sum(float(vyear[v] <= y + 1) for y in (2022, 2025))
        dd_ = sum(((hh(st[y]) - AEF[cid][:, sl]) ** 2).sum(-1) * float(vyear[v] <= y + 1) for y, hh, sl in ((2022, h22, A22), (2025, h25, A25))) / max(nh_, 1)   # mean over heads, as in training
        post = torch.softmax(-dd_ / TEMP + torch.log(torch.tensor(iw_, dtype=torch.float32)), 0).numpy()   # uniform over the DISK (candidates x footprints they stand for)
        loc.append((post[hi_].sum(), iw_[hi_].sum() / iw_.sum()))
    if loc:
        a = np.array(loc); ev['localisation'] = {'visits': len(a), 'posterior_mass_on_event_matching_candidates': float(a[:, 0].mean()), 'their_share_of_candidates': float(a[:, 1].mean())}
_mstem = os.path.splitext(os.environ.get('MODEL', 'model.pt'))[0]                      # per-model outputs (every seed wrote eval.json: audit F5a)
print(json.dumps(ev, indent=1)); json.dump(ev, open(f'{D}/eval_{_mstem}.json', 'w'), indent=1)
# ---------------- map Baker stands: K_S samples per footprint → cohort slots → stand summaries (mean of footprints)
S = pd.read_parquet(f'{D}/stands.parquet'); SF = np.load(f'{D}/stand_feat.npy').astype(np.float32); SF[:, SO] = (SF[:, SO] - smu) / ssd
with torch.no_grad():
    X = torch.tensor(np.concatenate([SF[:, :128], SF[:, DS], SF[:, SO]], 1)); qm, qls = qn(X)
    res = []
    for k_ in range(8):
        z = qm + torch.exp(qls) * torch.randn_like(qm); lp, val = dec(z); tph, ba, qmd, dk, _ = summaries(lp, val); res.append(torch.stack([tph, ba, qmd, dk.float()], 1))
    R_ = torch.stack(res)                                                     # [samples, footprints, 4]
S['tph'] = R_[:, :, 0].mean(0).numpy(); S['ba'] = R_[:, :, 1].mean(0).numpy(); S['qmd'] = R_[:, :, 2].mean(0).numpy()
S['ba_sd'] = R_[:, :, 1].std(0).numpy(); S['dom'] = torch.mode(R_[:, :, 3].long(), 0).values.numpy()
out = S.groupby('stand_id').agg(tph=('tph', 'mean'), ba=('ba', 'mean'), qmd=('qmd', 'mean'), ba_sd=('ba_sd', 'mean'), dom=('dom', lambda x: x.mode().iloc[0])).reset_index()
out['dom_key'] = [M['keys'][int(i)] for i in out.dom]; out.to_parquet(f'{D}/stand_veg_q_{_mstem}.parquet')     # (the direct Q predictor; the stand map is g05)
print('stands mapped', len(out), out[['tph', 'ba', 'qmd', 'ba_sd']].describe().loc[['mean', '50%']].round(2).to_dict(), out.dom_key.value_counts().head(6).to_dict())
