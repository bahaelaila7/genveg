"""Generative vegetation v3 — stand mapping with CONSISTENT landscape descriptors (fix of g04's stand mapping, whose landscapes came from 4 footprints per stand
and were not comparable with the plots' disks). Per stand, its OWN landscape: the forest footprints (3x3, stride 3) within 1.6 km of the stand's centroid, the
a uniform 256-subsample (seed = stand id; the plots' descriptors are inclusion-weighted estimates from their NCAND stratified candidates —
both estimate the full-disk descriptor, with different noise), the same descriptor (g04.landscape). (Until 2026-10-22 every stand took the descriptor of the
nearest node of a 1 km grid: the node explained 41-65 % of the stand-to-stand variance and drew 1 km squares on the map.) z ~ p(z | X_L, map_y0) (bayes_eval.json map_y0: the start year g04 gated) (MDN, saved by g04);
optional likelihood weighting (USE_B=1, B's flow + heads saved by g04 — likelihood only; joint evidence of the stand's footprints = mean AlphaEarth distance); flow to 2025
through the footprints' disturbance histories with the v2 flow (decoder-consistent; samples spread over the footprints); decode;
the estimator g04 validated (G46.13): MS iid draws, the calibrated median e^CF · lmed (calibrated_median) and the medoid draw (medoid_draw), with g04's
mapping-path calibration factors. Refuses unless g04 accepted the mapped estimator on validation (bayes_eval.json map_accept, G46.12).
USE_B=1 is NOT validated (non-production, audit F4 / C11): it reweights the draws by the stand likelihood but applies the B0 path's calibration factors,
and B was fine-tuned on encoder latents at the visit year while it scores MDN prior draws from map_y0 here. Writes data/proto/<data>/stand_veg_v3_B<USE_B>.parquet + stand_cohorts_v3_B<USE_B>.parquet (genveg_pipeline promotes the production one to
stand_veg_v3.parquet / stand_cohorts_v3.parquet) and prints diagnostics (by status, correlation with TreeMap)."""
import os, sys, json, math, pickle, numpy as np, pandas as pd, torch, rasterio, geopandas as gpd
os.environ.setdefault('DATA', 'gafl'); os.environ['LOAD'] = '1'; os.environ['EPOCHS'] = '0'
_argv = sys.argv; sys.argv = ['g03b']
_HERE = os.path.dirname(os.path.abspath(__file__)); SRC = open(f'{_HERE}/g03b_train_gafl.py').read(); SRC = SRC[:SRC.index('# ---------------- evaluation on held-out plots')]
exec(compile(SRC, 'g03b_train_gafl.py', 'exec')); sys.argv = _argv
G4 = open(f'{_HERE}/g04_bayes_downscale.py').read()
exec(compile(G4[G4.index('# decoded summaries'):G4.index('t0 = time.time()')], 'g04', 'exec'))            # summaries, obs_summ, medoid_draw, calibrated_median, MS
exec(compile(G4[G4.index('BINS = '):G4.index('XL = {}')], 'g04', 'exec'))                               # the landscape function
exec(compile(G4[G4.index('GM = 8'):G4.index('mdn = MDN()')], 'g04', 'exec'))                            # MDN class
sys.path.insert(0, _HERE); import g02b_build_gafl as G2
from stand_runs import stands_run
BM = torch.load(f'{D}/bayes_models.pt', weights_only=False); km = pickle.load(open(f'{D}/bayes_kmeans.pkl', 'rb'))
XLm = np.zeros((1, len(BM['xmu'])))                                                 # descriptor width for the MDN constructor
import copy; flowB, h22B, h25B = copy.deepcopy(flow), copy.deepcopy(h22), copy.deepcopy(h25)       # flow, h22, h25: the v2 model (model.pt), vegetation dynamics
mdn = MDN(); mdn.load_state_dict(BM['mdn']); flowB.load_state_dict(BM['flowB']); h22B.load_state_dict(BM['h22B']); h25B.load_state_dict(BM['h25B'])   # B: likelihood only
xmu, xsd = BM['xmu'], BM['xsd']; nx = lambda x: torch.tensor((x - xmu) / xsd, dtype=torch.float32)
BE = json.load(open(f'{D}/bayes_eval.json'))
if BE.get('map_accept') is not True:     # (G46.12) the pre-registered acceptance of the mapped estimator on validation
    raise SystemExit(f"g04 did not accept the mapped estimator on validation (bayes_eval.json map_accept {BE.get('map_accept')}: {BE.get('map_accept_reasons')}): not mapping")
CAL = BE['calibration_log_factors_mapping_path']                                    # calibrated and validated for THIS path in g04
MAP_Y0 = float(BE.get('map_y0', 2021.0))   # the start year g04 calibrated and gated (not the env's request)
G2.init(); land = G2._S['land']
S = pd.read_parquet(f'{D}/stands.parquet'); SF = np.load(f'{D}/stand_feat.npy').astype(np.float32); SF[:, SO] = (SF[:, SO] - smu) / ssd
SA = torch.tensor(SF[:, :128]); SD = torch.tensor(SF[:, DS].reshape(-1, NY, NC + 2)); SS = torch.tensor(SF[:, SO])
# ---- per-stand landscapes: footprint features once over the extent (global stride-3 grid, row bands), then each stand's own 1.6 km disk
RP = int(1600 // 30) + 2; cen = S.groupby('stand_id')[['row', 'col']].mean()
r0, r1, c0, c1 = int(cen.row.min()) - RP, int(cen.row.max()) + RP + 1, int(cen.col.min()) - RP, int(cen.col.max()) + RP + 1
dR, dC = G2.stand_offset(rasterio.open(stands_run()['tif']))                        # (AEF90: feature grid grid_r4; the extent starts on a 90 m block)
if G2.AEF90: r0, c0 = 3 * (r0 // 3), 3 * (c0 // 3)
FR, FC, FF = [], [], []; BAND = 600                                                 # multiple of 3: the stride-3 grid continues across bands
for rb in range(r0, r1, BAND):
    rr, cc, F = G2.feats(rb + dR, c0 + dC, min(BAND, r1 - rb) + 2, c1 - c0 + 2)
    keep = rr <= min(BAND, r1 - rb)                                                   # rows of this band only (feats' grid starts at row 1 of the window)
    FR.append(rr[keep] + rb); FC.append(cc[keep] + c0); FF.append(F[keep])
FR, FC = np.concatenate(FR), np.concatenate(FC); FF = np.concatenate(FF); FF[:, SO] = (FF[:, SO] - smu) / ssd; FF = torch.tensor(FF)
from scipy.spatial import cKDTree
kf = cKDTree(np.stack([FR, FC], 1)); sids = cen.index.values; ctr = np.rint(cen.values).astype(int); XLs = {}
for sid, (cr, cc_) in zip(sids, ctr):
    ix = np.array(kf.query_ball_point([cr, cc_], 1600 / 30), dtype=int)
    if len(ix) == 0: continue
    if len(ix) > 256: ix = np.sort(np.random.default_rng(int(sid)).choice(ix, 256, replace=False))
    F = FF[ix]; XLs[int(sid)] = landscape(F[:, :128], F[:, DS].reshape(-1, NY, NC + 2), F[:, SO])
print(f'stand landscapes {len(XLs)} of {len(sids)} stands ({len(FR)} footprints in the extent)', flush=True)
order = [int(x) for x in sids if int(x) in XLs]; node_of = {sid: k for k, sid in enumerate(order)}
np.savez(f'{D}/stand_nodes.npz', nodes=np.array([ctr[list(sids).index(sid)] for sid in order]), XL=np.stack([XLs[sid] for sid in order]))   # for gv_verify_upstream (U8)
USE_B = os.environ.get('USE_B') == '1'; Mz = 64; rows = []; ESS = []; crows = []; fND = []    # (USE_B: Mz prior draws reweighted, MS resampled)
keys_ = M['keys']
with torch.no_grad():
    for sid, g in S.groupby('stand_id'):
        fp = g.index.values
        if int(sid) not in XLs: continue
        j = node_of[int(sid)]; xl = nx(XLs[int(sid)]).unsqueeze(0); torch.manual_seed(int(sid))     # a stand's draws depend on its id only (not on the stand set / order)
        zs = mdn.sample(xl, torch.tensor([MAP_Y0]), Mz if USE_B else MS)[0]                         # (B0, production: MS iid draws as g04 validated)
        if USE_B:     # every footprint of the stand observes the SAME vegetation: tempered joint evidence = mean distance over the footprints
            dm = torch.zeros(Mz)
            for f_ in fp:
                st = flowB(zs, torch.full((Mz,), MAP_Y0), [2022, 2025], SD[[f_] * Mz], SS[[f_] * Mz])
                dm = dm + sum(((h(st[y]) - SA[[f_] * Mz][:, sl]) ** 2).sum(-1) for y, h, sl in ((2022, h22B, A22), (2025, h25B, A25))) / 2 / len(fp)
            w = torch.exp(-(dm - dm.min()) / TEMP); ESS.append(float(w.sum() ** 2 / (w ** 2).sum())); zs = zs[torch.multinomial(w / w.sum(), MS, replacement=True)]
        fpi = [fp[i % len(fp)] for i in range(MS)]                                    # draws spread over the stand's footprints
        st = flow(zs, torch.full((MS,), MAP_Y0), [2025], SD[fpi], SS[fpi]); lp_, val_ = dec(st[2025]); s_ = summaries(lp_, val_)   # (uncalibrated, as g04)
        tph_m, ba_m = (float(calibrated_median(s_[i], CAL[i])) for i in range(2))      # the calibrated stand summary e^CF · lmed (G46.13)
        # stand cohorts (TreeMap-independent initial state for the simulation engines): ONE plausible stand — the medoid draw (closest to the median summary
        # in log TPH / BA / QMD-from-TPH-BA, F#12; calibration does not move it) with its present slots (presence sampled per slot by Bernoulli(calibrated
        # p) seeded by the stand id, inside medoid_draw; else its largest slot) — not the ensemble mean of the draws, which spread every stand over ~13 keys
        # and ~48 cohorts (audit F2) and made the mean-share dominant collapse onto the regional mode (F3). Its totals are scaled to the calibrated median
        # TPH / BA (one density and one diameter factor; ages, heights, crowns kept). The SHARED estimator g04 validates (medoid_draw, G32)
        md_ = medoid_draw(lp_, val_, s_, int(sid)); i_, pres, Nr, Dd = md_['i'], md_['pres'], md_['Nr'], md_['Dd']
        Hh = (val_[..., 2] * 30).clamp(min=1.37); Cr = val_[..., 3].clamp(0.05, 0.95)
        c0_ = len(crows)
        for j_ in pres:
            crows.append(dict(stand_id=int(sid), key=keys_[j_ // NA], age_class=j_ % NA, age=float(j_ % NA * 10 + 5), tph=float(Nr[i_, j_]),
                              qmd_cm=float(Dd[i_, j_]), ht_m=float(Hh[i_, j_]), cr=float(Cr[i_, j_])))
        ct = sum(c['tph'] for c in crows[c0_:]); cb = sum(c['tph'] * math.pi / 4 * (c['qmd_cm'] / 100) ** 2 for c in crows[c0_:]); fND.append((math.nan, math.nan))
        if ct > 0 and cb > 0 and tph_m > 0 and ba_m > 0:
            fN = tph_m / ct; fD = math.sqrt((ba_m / tph_m) / (cb / ct)); fND[-1] = (fN, fD)
            for c in crows[c0_:]: c['tph'] *= fN; c['qmd_cm'] *= fD
        kb = {}
        for c in crows[c0_:]: kb[c['key']] = kb.get(c['key'], 0.0) + c['tph'] * c['qmd_cm'] ** 2
        dom_ = keys_.index(max(kb, key=kb.get)) if kb else int(s_[4].mean(0).argmax())
        # GEN_QMD = the QMD of the calibrated TPH / BA (consistent with the cohort table; the median of the samples' QMDs was not: audit F10)
        rows.append(dict(stand_id=int(sid), node=int(j), tph=tph_m, ba=ba_m, qmd=100 * math.sqrt(ba_m / (tph_m * math.pi / 4)) if tph_m > 0 else 0.0,
                         ba_sd=float(s_[1].std()) * math.exp(CAL[1]), dom=dom_, draw=i_))   # (the calibrated draws' sd)
out = pd.DataFrame(rows); keys = json.load(open(f'{D}/meta.json'))['keys']; out['dom_key'] = [keys[i] for i in out.dom]; out['map_y0'] = MAP_Y0; out['map_cal_tph'], out['map_cal_ba'] = CAL[0], CAL[1]; out.to_parquet(f'{D}/stand_veg_v3_B{int(USE_B)}.parquet')   # (the pipeline promotes the production use_b to stand_veg_v3.parquet: r6 M2)
C_ = pd.DataFrame(crows); C_.to_parquet(f'{D}/stand_cohorts_v3_B{int(USE_B)}.parquet')
_t = C_.groupby('stand_id').tph.sum(); _b = (C_.tph * math.pi / 4 * (C_.qmd_cm / 100) ** 2).groupby(C_.stand_id).sum()
_fnd = np.array(fND); print(f'medoid-draw factors fN q05/q50/q95 {np.nanquantile(_fnd[:, 0], [.05, .5, .95]).round(3)}, fD {np.nanquantile(_fnd[:, 1], [.05, .5, .95]).round(3)}', flush=True)
_o = out.set_index('stand_id'); print(f'stand cohorts: {len(C_)} rows for {C_.stand_id.nunique()} stands (median {int(C_.groupby("stand_id").size().median())} per stand); '
      f'cohort-sum / stand summary median ratio TPH {float((_t / _o.tph.reindex(_t.index)).median()):.3f}, BA {float((_b / _o.ba.reindex(_b.index)).median()):.3f}', flush=True)
st_ = gpd.read_file(stands_run()['gpkg'], layer='stands', ignore_geometry=True).set_index('stand_id'); d = st_.join(out.set_index('stand_id'), how='inner')
d['gen_ba_ft'] = d.ba * 4.356; d['gen_tpa'] = d.tph / 2.47105
print('USE_B', USE_B, '| likelihood ESS of', Mz, 'median / q10:', (round(float(np.median(ESS)), 1), round(float(np.quantile(ESS, 0.1)), 1)) if ESS else None); print(d.groupby('status')[['BALIVE', 'gen_ba_ft', 'TPA_LIVE', 'gen_tpa']].median().round(1))
f = d[d.status == 1]; print('forest stands: corr(TreeMap BA, gen BA)', round(float(np.corrcoef(f.BALIVE.fillna(0), f.gen_ba_ft)[0, 1]), 3), '| gen BA spread (sd / median)', round(float(f.gen_ba_ft.std() / f.gen_ba_ft.median()), 3))
print(out.dom_key.value_counts().head(6).to_dict())
