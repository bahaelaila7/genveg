"""Generative vegetation v3 — stand mapping with CONSISTENT landscape descriptors (fix of g04's stand mapping, whose landscapes came from 4 footprints per stand
and were not comparable with the plots' disks). Landscape descriptors on a 1 km grid over the area: all forest footprints (3x3, stride 3) within 1.6 km, the same
uniform 256-subsample as the plots (g02b), the same descriptor (g04.landscape). Per stand: X_L of the nearest grid node; z ~ p(z | X_L, 2021) (MDN, saved by g04);
optional likelihood weighting (USE_B=1, B's flow + heads saved by g04 — likelihood only; joint evidence of the stand's footprints = mean AlphaEarth distance); flow to 2025
through the footprints' disturbance histories with the v2 flow (decoder-consistent; samples spread over the footprints); decode;
the mapping-path calibration factors of g04 (validated there on held-out plots). Writes data/proto/gafl/stand_veg_v3.parquet and prints diagnostics (by status, correlation with TreeMap)."""
import os, sys, json, math, pickle, numpy as np, pandas as pd, torch, rasterio, geopandas as gpd
os.environ.setdefault('DATA', 'gafl'); os.environ['LOAD'] = '1'; os.environ['EPOCHS'] = '0'
_argv = sys.argv; sys.argv = ['g03b']
_HERE = os.path.dirname(os.path.abspath(__file__)); SRC = open(f'{_HERE}/g03b_train_gafl.py').read(); SRC = SRC[:SRC.index('# ---------------- evaluation on held-out plots')]
exec(compile(SRC, 'g03b_train_gafl.py', 'exec')); sys.argv = _argv
G4 = open(f'{_HERE}/g04_bayes_downscale.py').read()
exec(compile(G4[G4.index('# decoded summaries'):G4.index('t0 = time.time()')], 'g04', 'exec'))            # summaries / obs_summ
exec(compile(G4[G4.index('BINS = '):G4.index('XL = {}')], 'g04', 'exec'))                               # landscape()
exec(compile(G4[G4.index('GM = 8'):G4.index('mdn = MDN()')], 'g04', 'exec'))                            # MDN class
sys.path.insert(0, _HERE); import g02b_build_gafl as G2
from stand_runs import stands_run
BM = torch.load(f'{D}/bayes_models.pt', weights_only=False); km = pickle.load(open(f'{D}/bayes_kmeans.pkl', 'rb'))
XLm = np.zeros((1, len(BM['xmu'])))                                                 # descriptor width for the MDN constructor
import copy; flowB, h22B, h25B = copy.deepcopy(flow), copy.deepcopy(h22), copy.deepcopy(h25)       # flow / h22 / h25 = v2 (model.pt): vegetation dynamics
mdn = MDN(); mdn.load_state_dict(BM['mdn']); flowB.load_state_dict(BM['flowB']); h22B.load_state_dict(BM['h22B']); h25B.load_state_dict(BM['h25B'])   # B: likelihood only
xmu, xsd = BM['xmu'], BM['xsd']; nx = lambda x: torch.tensor((x - xmu) / xsd, dtype=torch.float32)
CAL = json.load(open(f'{D}/bayes_eval.json'))['calibration_log_factors_mapping_path']      # calibrated and validated for THIS path in g04
G2.init(); land = G2._S['land']
S = pd.read_parquet(f'{D}/stands.parquet'); SF = np.load(f'{D}/stand_feat.npy').astype(np.float32); SF[:, SO] = (SF[:, SO] - smu) / ssd
SA = torch.tensor(SF[:, :128]); SD = torch.tensor(SF[:, DS].reshape(-1, NY, NC + 2)); SS = torch.tensor(SF[:, SO])
# ---- landscape grid (1 km) over the stands' extent, descriptors exactly as for the plots
r0, r1, c0, c1 = S.row.min(), S.row.max(), S.col.min(), S.col.max(); step = 33; RP = int(1600 // 30) + 2
nodes = [(r, c) for r in range(r0, r1 + step, step) for c in range(c0, c1 + step, step)]
XLg = []; rng = np.random.default_rng(0)
for (r, c) in nodes:
    rr, cc, F = G2.feats(r - RP, c - RP, 2 * RP + 1, 2 * RP + 1); d = np.hypot(rr - RP, cc - RP) * 30; F = F[d <= 1600]
    if len(F) > 256: F = F[rng.choice(len(F), 256, replace=False)]
    if len(F) == 0: XLg.append(None); continue
    F[:, SO] = (F[:, SO] - smu) / ssd; F = torch.tensor(F)
    XLg.append(landscape(F[:, :128], F[:, DS].reshape(-1, NY, NC + 2), F[:, SO]))
ok = [i for i, x in enumerate(XLg) if x is not None]; from scipy.spatial import cKDTree
kt = cKDTree(np.array([nodes[i] for i in ok])); print('landscape nodes', len(ok), flush=True)
np.savez(f'{D}/stand_nodes.npz', nodes=np.array([nodes[i] for i in ok]), XL=np.stack([XLg[i] for i in ok]))      # for gv_verify_upstream (U8)
USE_B = os.environ.get('USE_B') == '1'; Mz = 64; rows = []; ESS = []
with torch.no_grad():
    for sid, g in S.groupby('stand_id'):
        fp = g.index.values; _, j = kt.query([g.row.mean(), g.col.mean()]); xl = nx(XLg[ok[j]]).unsqueeze(0)
        zs = mdn.sample(xl, torch.tensor([2021.0]), Mz)[0]
        w = torch.ones(Mz)
        if USE_B:     # every footprint of the stand observes the SAME vegetation: tempered joint evidence = mean distance over the footprints
            dm = torch.zeros(Mz)
            for f_ in fp:
                st = flowB(zs, torch.full((Mz,), 2021.0), [2022, 2025], SD[[f_] * Mz], SS[[f_] * Mz])
                dm = dm + sum(((h(st[y]) - SA[[f_] * Mz][:, sl]) ** 2).sum(-1) for y, h, sl in ((2022, h22B, A22), (2025, h25B, A25))) / 2 / len(fp)
            w = torch.exp(-(dm - dm.min()) / TEMP); ESS.append(float(w.sum() ** 2 / (w ** 2).sum()))
        idx = torch.multinomial(w / w.sum(), 16, replacement=True); fpi = [fp[i % len(fp)] for i in range(16)]   # samples spread over the stand's footprints
        st = flow(zs[idx], torch.full((16,), 2021.0), [2025], SD[fpi], SS[fpi]); s_ = list(summaries(*dec(st[2025])))
        s_[:3] = [s_[i] * math.exp(CAL[i]) for i in range(3)]; comp = s_[4].mean(0)
        rows.append(dict(stand_id=int(sid), node=int(j), tph=float(s_[0].median()), ba=float(s_[1].median()), qmd=float(s_[2].median()), ba_sd=float(s_[1].std()), dom=int(comp.argmax())))
out = pd.DataFrame(rows); keys = json.load(open(f'{D}/meta.json'))['keys']; out['dom_key'] = [keys[i] for i in out.dom]; out.to_parquet(f'{D}/stand_veg_v3.parquet')
st_ = gpd.read_file(stands_run()['gpkg'], layer='stands', ignore_geometry=True).set_index('stand_id'); d = st_.join(out.set_index('stand_id'), how='inner')
d['gen_ba_ft'] = d.ba * 4.356; d['gen_tpa'] = d.tph / 2.471
print('USE_B', USE_B, '| likelihood ESS of', Mz, 'median / q10:', (round(float(np.median(ESS)), 1), round(float(np.quantile(ESS, 0.1)), 1)) if ESS else None); print(d.groupby('status')[['BALIVE', 'gen_ba_ft', 'TPA_LIVE', 'gen_tpa']].median().round(1))
f = d[d.status == 1]; print('forest stands: corr(TreeMap BA, gen BA)', round(float(np.corrcoef(f.BALIVE.fillna(0), f.gen_ba_ft)[0, 1]), 3), '| gen BA spread (sd / median)', round(float(f.gen_ba_ft.std() / f.gen_ba_ft.median()), 3))
print(out.dom_key.value_counts().head(6).to_dict())
