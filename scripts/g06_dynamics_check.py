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
    o_ = obs_summ; vv_ = V[(V.year >= 2019) & (V.year < 2021) & V.pid.isin(test_pid | val_pid)].vid.values; vv_ = vv_[fs_ok[vv_]]; vv_ = vv_[o_(torch.tensor(vv_))[1].numpy() > 10]; vv_ = np.array([v for v in vv_ if vpid[v] in cand_of]); vv_ = np.random.default_rng(33).choice(vv_, min(800, len(vv_)), replace=False); vt = torch.tensor(vv_)   # (held-out visits only: r8 M8)   # (fs_ok + max-prior soil: r13 C-5)
    z = enc(vt, vyear[vt]); cid0 = np.array([cand_of[vpid[v]][0][int(np.argmax(cand_of[vpid[v]][1]))] for v in vv_]); soil = SOIL[cid0]; res = {}
    for kind in ('none', 'clearcut', 'fire'):
        d = torch.zeros(len(vv_), NY, NC + 2); j = 2023 - YEARS[0]
        if kind == 'clearcut': d[:, j, G2.IX['harvest']] = 1; d[:, j, G2.H_] = 1; d[:, j, NC] = 3
        if kind == 'fire': d[:, j, G2.IX['fire']] = 1; d[:, j, NC] = 1
        s = summaries(*dec(flow(z, vyear[vt], [2025], d, soil, rs=VREC[vt])[2025])); res[kind] = float(torch.quantile(s[1], 0.5))
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
        y1, y2 = float(vyear[v1]), float(vyear[v2]); evp = recs_in(vpid[v1], y1, y2)        # records of the interval (recorded at v2: audit r5 I3)
        cl_ = [c for c in G2.clusters(evp) if c['lr'] >= 4]; grp = 'cut' if any(k in cutk for k, _ in evp) and cl_ else ('other_event' if cl_ else 'none')
        rows.append((v1, v2, grp))
    pred = {}
    for v1, v2, grp in rows:
        cid, pr = pair_cand[v1]
        if EVAL_LOC != 'history':           # ablation of WHERE the plot is assumed to be (same model, same pairs)
            g_ = C0[C0.pid == vpid[v1]]
            if EVAL_LOC == 'public': cid, pr = g_.cid.values[[int(np.argmin(g_.dist_m.values))]], np.ones(1)       # pinned at the public coordinate
            else: cid, pr = g_.cid.values, g_.incl_w.values / g_.incl_w.values.sum()                                # uniform over the disk's footprints
        ci = torch.tensor(np.random.choice(cid, 16, replace=True, p=pr)); y2 = vyear[v2]; yy = min(int(math.floor(float(y2))), YEARS[-1])   # (a 2026 visit: the state is the end-2025 one, as g03b; r7 B2)
        st = flow(enc(torch.tensor([v1]), vyear[[v1]]).repeat(16, 1), vyear[[v1]].repeat(16), [yy], DIST[ci], SOIL[ci], end=y2.repeat(16), rs=VREC[[v1]].repeat(16), re=VREC[[v2]].repeat(16), ra=torch.tensor([pair_ra(v1, v2)]).repeat(16))
        s = summaries(*dec(st[yy])); s0 = summaries(*dec(enc(torch.tensor([v1]), vyear[[v1]])))           # s0: v1 decoded without flow (reconstruction)
        pred[(v1, v2)] = (float(lmed(s[1])), float(lmed(s[0])), float(s0[1][0]))
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
    # 3 LONG-HORIZON undisturbed growth (audit r3 I1: the no-event flow shrank stands ~2.4 %/yr over long horizons; FIA undisturbed pairs grow):
    #   held-out pairs of every horizon (g03b.pair_cand2: PAIR_ALL) with no detectable FIA event in the span and no hiding code at any visit in between —
    #   predicted vs observed BA log change per year, by horizon
    blk_ = vblk                                                                            # (g03b: hiding codes | gaps in FIA's measurement chain)
    vs_of = V.assign(vid=np.arange(nv)).sort_values('year').groupby('pid').vid.apply(list).to_dict()   # (audit r5 B1: this assignment sat inside the comment)
    def long_horizon(pset, prefix):
        lh = []
        for (v1, v2), (cid, pr) in pair_cand2.items():
            if vpid[v1] not in pset: continue
            y1, y2 = float(vyear[v1]), float(vyear[v2]); seq = vs_of[vpid[v1]]; mid = seq[seq.index(v1) + 1:seq.index(v2) + 1]
            if blk_[mid].any() or [c for c in G2.clusters(recs_in(vpid[v1], y1, y2)) if c['lr'] >= 4]: continue
            ob1, ob2 = float(obs_summ(torch.tensor([v1]))[1][0]), float(obs_summ(torch.tensor([v2]))[1][0])
            if ob1 < 1.0: continue
            ci = torch.tensor(np.random.choice(cid, 16, replace=True, p=pr)); yy = min(int(math.floor(y2)), YEARS[-1]); z0 = enc(torch.tensor([v1]), vyear[[v1]])
            st = flow(z0.repeat(16, 1), vyear[[v1]].repeat(16), [yy], DIST[ci], SOIL[ci], end=vyear[[v2]].repeat(16), rs=VREC[[v1]].repeat(16), re=VREC[[v2]].repeat(16), ra=torch.tensor([pair_ra(v1, v2)]).repeat(16)); pb = float(lmed(summaries(*dec(st[yy]))[1]))   # (true median: r13 C-2)
            p0 = float(summaries(*dec(z0))[1][0])
            def dec_age(zz):    # decoded BA-weighted mean age of the expected slots (age-class midpoints)
                lp_, val_ = dec(zz); pr_ = torch.sigmoid(calibrated(lp_)); w_ = torch.exp(val_[..., 0].clamp(max=12)) * pr_ * torch.exp(val_[..., 1].clamp(max=math.log(150.0))) ** 2
                mid = (torch.arange(NK * NA) % NA).float() * 10 + 5; return float(torch.quantile((w_ * mid).sum(-1) / w_.sum(-1).clamp(min=1e-9), 0.5))   # (true median: r13 C-2)
            lh.append((y2 - y1, math.log((pb + 0.5) / (p0 + 0.5)) / (y2 - y1), math.log((ob2 + 0.5) / (ob1 + 0.5)) / (y2 - y1),
                       (dec_age(st[yy]) - dec_age(z0)) / (y2 - y1), float(AGE_OBS[v2] - AGE_OBS[v1]) / (y2 - y1)))
        for lo_, hi_ in ((0, 8), (8, 15), (15, 99)):
            r_ = [x for x in lh if lo_ <= x[0] < hi_]
            if r_: out[f'{prefix}h{lo_}_{hi_}'] = {'n': len(r_), 'median_pred_logchange_per_yr': float(np.median([x[1] for x in r_])), 'median_obs_logchange_per_yr': float(np.median([x[2] for x in r_])),
                                                       'median_pred_age_per_yr': float(np.median([x[3] for x in r_])), 'median_obs_age_per_yr': float(np.median([x[4] for x in r_]))}
    # acceptance on the VALIDATION plots when the model has them (VAL_SPLIT: audit r6 I2); the test plots are reported only
    ACC_SET = 'val' if VAL_SPLIT else 'test'
    long_horizon(val_pid if VAL_SPLIT else test_pid, 'undisturbed_')
    if VAL_SPLIT: long_horizon(test_pid, 'test_undisturbed_')
    # PRE-REGISTERED acceptance of the undisturbed dynamics (audit r4): per horizon bin, BA log change within ±0.01 / yr of the observed, and the decoded age
    # advancing 0.7–1.3 x the observed age progression
    acc = {}
    for k_, d_ in out.items():
        if k_.startswith('undisturbed_h') and d_['n'] >= 20:
            acc[k_] = {'BA_change_ok': abs(d_['median_pred_logchange_per_yr'] - d_['median_obs_logchange_per_yr']) <= 0.01,
                       'age_ok': d_['median_obs_age_per_yr'] > 0 and 0.7 <= d_['median_pred_age_per_yr'] / d_['median_obs_age_per_yr'] <= 1.3}
    out['latent'] = latent_stats(np.asarray(va_v if VAL_SPLIT else te_v))     # (G29) latent health on the acceptance set's encodings
    out['undisturbed_acceptance'] = {'set': ACC_SET, 'bins': acc, 'latent_ok': out['latent']['ok'],
                                     'passed': bool(acc) and all(all(v.values()) for v in acc.values()) and out['latent']['ok']}
print(json.dumps(out, indent=1)); json.dump(out, open(f'{D}/dynamics_check_{out["tag"] or "run"}.json', 'w'), indent=1)
# the acceptance is ENFORCED (audit r5 I5): a model whose undisturbed dynamics fail it is not mapped (the pipeline's downscale / map depend on this stage)
# enforced only for the production seed's history evaluation (G06_ENFORCE=1, set by the pipeline) and only on a validation set (never on test)
if os.environ.get('G06_ENFORCE', '0') == '1' and EVAL_LOC == 'history' and not VAL_SPLIT:   # (pass or fail: never decided on the test plots; r8 M1)
    print('ACCEPTANCE cannot be decided: no validation set (VAL_SPLIT=0) and the test plots are never used for it', flush=True); sys.exit(4)
if os.environ.get('G06_ENFORCE', '0') == '1' and EVAL_LOC == 'history' and not out['undisturbed_acceptance']['passed']:
    print(f"ACCEPTANCE FAILED: {out['undisturbed_acceptance']}", flush=True); sys.exit(3)
