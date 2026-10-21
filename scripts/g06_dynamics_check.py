"""Held-out test of the learned DYNAMICS (disturbance response) of the v2 flow + decoder (g03b model.pt), anchored on FIA's own records.
  1  synthetic: stocked visits of 2019-20 flowed to 2025 with no event / a 2023 clear-cut (harvest class + Hansen loss on the whole footprint, high
     severity) / a 2023 fire → decoded BA (ratio to no event).
  2  held-out remeasurement pairs (test plots, spatial blocks of g03b): z(v1) flowed to v2's date through candidate histories consistent with the
     plot's FIA records between the visits (g03b.pair_cand) → decoded BA and TPH vs v2's observed; grouped by the FIA record of the period:
     detectable cut / regeneration (T10-T40 clusters), other detectable events, none. Baselines: persistence (v1's observed values).
Writes data/proto/gafl/dynamics_check.json; tag with TAG=… (e.g. before / after the FIA-anchored training)."""
import os, sys, json, math, numpy as np, pandas as pd, torch
os.environ.setdefault('DATA', 'gafl'); os.environ['LOAD'] = '1'; os.environ['EPOCHS'] = '0'
_argv = sys.argv; sys.argv = ['g03b']
_HERE = os.path.dirname(os.path.abspath(__file__)); SRC = open(f'{_HERE}/g03b_train_gafl.py').read(); SRC = SRC[:SRC.index('# ---------------- evaluation on held-out plots')]
exec(compile(SRC, 'g03b_train_gafl.py', 'exec')); sys.argv = _argv
G4 = open(f'{_HERE}/g04_bayes_downscale.py').read(); exec(compile(G4[G4.index('# decoded summaries'):G4.index('t0 = time.time()')], 'g04', 'exec'))
torch.manual_seed(0); np.random.seed(0); out = {'tag': os.environ.get('TAG', '')}
EVAL_LOC = os.environ.get('EVAL_LOC', 'history'); out['eval_loc'] = EVAL_LOC; C0 = pd.read_parquet(f'{D}/cands.parquet')   # history | public | uniform
with torch.no_grad():
    o_ = obs_summ; vv_ = V[(V.year >= 2019) & (V.year < 2021)].vid.values; vv_ = vv_[o_(torch.tensor(vv_))[1].numpy() > 10]; vv_ = np.array([v for v in vv_ if vpid[v] in cand_of])[:800]; vt = torch.tensor(vv_)
    z = enc(vt, vyear[vt]); cid0 = np.array([cand_of[vpid[v]][0][0] for v in vv_]); soil = SOIL[cid0]; res = {}
    for kind in ('none', 'clearcut', 'fire'):
        d = torch.zeros(len(vv_), NY, NC + 2); j = 2023 - YEARS[0]
        if kind == 'clearcut': d[:, j, G2.IX['harvest']] = 1; d[:, j, G2.H_] = 1; d[:, j, NC] = 3
        if kind == 'fire': d[:, j, G2.IX['fire']] = 1; d[:, j, NC] = 1
        s = summaries(*dec(flow(z, vyear[vt], [2025], d, soil)[2025])); res[kind] = float(s[1].median())
    vt_ = torch.tensor(te_v); o_ = obs_summ(vt_); s_ = summaries(*dec(torch.cat([enc(c, vyear[c]) for c in torch.split(vt_, 2000)]))); rec = {}
    for i, k in enumerate(('TPH', 'BA', 'QMD')):       # autoencoder fidelity on held-out visits (decode(encode(visit)) vs the visit)
        m = o_[i] > 0; x = torch.log(o_[i][m]); y = torch.log(s_[i][m] + 1e-3)
        rec[k] = {'slope': float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()), 'corr': float(np.corrcoef(x, y)[0, 1]), 'median_log_bias': float((y - x).median())}
    out['reconstruction_heldout'] = rec
    out['synthetic_BA_2025_median'] = res; out['synthetic_ratio_to_none'] = {k: v / res['none'] for k, v in res.items()}
    # held-out pairs
    cutk = {'T10', 'T20', 'T30', 'T40'}; rows = []
    for v1, v2 in vpair.items():
        if (vpid[v1] not in test_pid if os.environ.get('G06_SET', 'test') == 'test' else vpid[v1] not in set(vpid[tr_v])) or v1 not in pair_cand: continue
        y1, y2 = float(vyear[v1]), float(vyear[v2]); evp = [(k, y) for k, y in ev_of.get(vpid[v1], []) if y1 <= y + 0.5 <= y2]
        cl_ = [c for c in G2.clusters(evp) if c['lr'] >= 4]; grp = 'cut' if any(k in cutk for k, _ in evp) and cl_ else ('other_event' if cl_ else 'none')
        rows.append((v1, v2, grp))
    pred = {}
    for v1, v2, grp in rows:
        cid, pr = pair_cand[v1]
        if EVAL_LOC != 'history':           # ablation of WHERE the plot is assumed to be (same model, same pairs)
            g_ = C0[C0.pid == vpid[v1]]
            if EVAL_LOC == 'public': cid, pr = g_.cid.values[[int(np.argmin(g_.dist_m.values))]], np.ones(1)       # pinned at the public coordinate
            else: cid, pr = g_.cid.values, g_.incl_w.values / g_.incl_w.values.sum()                                # uniform over the disk's footprints
        ci = torch.tensor(np.random.choice(cid, 16, replace=True, p=pr)); y2 = vyear[v2]; yy = int(math.floor(float(y2)))
        st = flow(enc(torch.tensor([v1]), vyear[[v1]]).repeat(16, 1), vyear[[v1]].repeat(16), [yy], DIST[ci], SOIL[ci], end=y2.repeat(16))
        s = summaries(*dec(st[yy])); s0 = summaries(*dec(enc(torch.tensor([v1]), vyear[[v1]])))           # s0: v1 decoded without flow (reconstruction)
        pred[(v1, v2)] = (float(torch.exp(torch.log(s[1] + 1e-3).median())), float(torch.exp(torch.log(s[0] + 1e-3).median())), float(s0[1][0]))
    o1 = obs_summ(torch.tensor([r[0] for r in rows])); o2 = obs_summ(torch.tensor([r[1] for r in rows]))
    LSg = pd.read_parquet(f'{D}/localisation.parquet') if os.path.exists(f'{D}/localisation.parquet') else None
    nh_ = (lambda v: float(LSg.n_hi.get(vpid[v], np.inf))) if LSg is not None and 'n_clusters' in LSg.columns else (lambda v: np.inf)
    grp_sets = [(g_, None) for g_ in ('cut', 'other_event', 'none')] + [('cut', (0, 10)), ('cut', (10, 50)), ('cut', (50, 1e9))]
    for grp, rng_ in grp_sets:
        ix = [i for i, r in enumerate(rows) if r[2] == grp and (rng_ is None or rng_[0] < nh_(r[0]) <= rng_[1])]
        if rng_ is not None: grp = f'{grp}_consistent_{int(rng_[0])}_{int(min(rng_[1], 9999))}'
        if not ix: continue
        pb = np.array([pred[(rows[i][0], rows[i][1])][0] for i in ix]); ob2 = o2[1].numpy()[ix]; ob1 = o1[1].numpy()[ix]
        e = lambda p, o: float(np.median(np.abs(np.log((p + 0.5) / (o + 0.5)))))    # +0.5 m2/ha: bare v2 visits are common after cuts
        p0 = np.array([pred[(rows[i][0], rows[i][1])][2] for i in ix]); dch = np.log((pb + 0.5) / (p0 + 0.5)); och = np.log((ob2 + 0.5) / (ob1 + 0.5))
        out[f'pairs_{grp}_change'] = {'median_pred_logchange': float(np.median(dch)), 'median_obs_logchange': float(np.median(och)),
                                      'err_change_model': float(np.median(np.abs(dch - och))), 'err_change_zero': float(np.median(np.abs(och))),
                                      'corr': float(np.corrcoef(dch, och)[0, 1]) if len(ix) > 3 else None}     # dynamics isolated from reconstruction bias
        out[f'pairs_{grp}'] = {'n': len(ix), 'obs_v1_BA_median': float(np.median(ob1)), 'obs_v2_BA_median': float(np.median(ob2)), 'pred_v2_BA_median': float(np.median(pb)),
                               'share_v2_bare': float(np.mean(ob2 <= 0)), 'err_model': e(pb, ob2), 'err_persistence': e(ob1, ob2),
                               'corr_logchange_model_obs': float(np.corrcoef(np.log((pb + 0.5) / (ob1 + 0.5)), np.log((ob2 + 0.5) / (ob1 + 0.5)))[0, 1]) if len(ix) > 3 else None}
print(json.dumps(out, indent=1)); json.dump(out, open(f'{D}/dynamics_check_{out["tag"] or "run"}.json', 'w'), indent=1)
