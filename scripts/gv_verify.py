"""Independent verification of the generative-vegetation pipeline (user: "triple check again. I don't trust it yet").
Every check recomputes a stage from the raw sources by a DIFFERENT code path (SQL instead of pandas, direct raster reads at stored coordinates, FIADB COND
instead of curated_trees for events) and compares. Output: PASS / FAIL / SKIP / INFO lines + numbers; exit code 1 if anything fails.
  python scripts/gv_verify.py [n_sample=300]
The paths it reads: VERIFY_INPUTS (data, the exact list the driver fingerprints: G45.7) and VERIFY_CODE (the pipeline code it execs / imports / scans),
module constants the driver evaluates in a subprocess; importing this module runs no check (main() does). A test harness may call
main(argv, force_pairs=[v1, ...], stop_after='T8') to put chosen pairs into every T8 sample and stop early (no environment switch).
Every FAIL line ends with a provenance label (r14 G47.9): `stale` (a producing script's code hash recorded with the table — meta.json code_hash, the
parquet provenance record of g02e / g02g / g02j, G46.6 — differs from the current script's, or none was recorded: the table predates G46.6) or
`divergent` (the table was built by the current code, or the check reads no table: the code disagrees with the spec).

PRE-REGISTERED THRESHOLDS (one line each: check — threshold — rationale). [r13] / [r14] = first set in audit round 13 / 14 (before any run of it).
  T0    0 hits; list unavailable = FAIL   a swallowed statement silently changes a stage (5 historical instances); list = the driver's pipeline_scripts() [r14: no fallback]
  T1a/c exact equality                 the visit set / order is a deterministic function of FIADB + the curated table (G40 bare / all-bare rules [r13]; G46.1 incomplete visits out [r14])
  T1r   0 table visits [r14]           G46.1: no table visit with FIADB live tallied trees >= 3 and curated ones < half of them (meta.json incomplete_visits: INFO)
  T1b   exact                          species keys are a deterministic ranking (ties by symbol)
  T1d-g rtol 1e-6 / 1e-9               the same arithmetic on the same records: only float noise may differ
  T1o   max |Δ| 1e-6 [r13]             token A / CR: the same weighted means, float noise only
  T1f   exact                          bare visits carry no tokens by construction
  T1h/i exact                          pairs / planted flag are deterministic
  T1j   < 2 % missing                  coarse (by MEASYEAR) sanity check of absent forested visits; allows sub-0.1-share visits
  T1k   <= 1 % of bare visits          pre-registered r7: a bare table visit with live FIADB trees is a false clear-cut target
  T1l   median 1 ± 0.02, < 2 % < 0.9   G14 (pre-registered r9): expansion factors identical by construction, residual should be ~0
  T1m   <= 2 % absent [r13]            A-8: FIADB live-tree plots in the curated L3 scope missing from the table bias the training population
  T1n   exact [r13; rule r14 G46.3 / G48.2]   plot_converted: the last SAMPLED FIADB visit dated <= 2022 is non-forest (non-forest land / water, no forest) after a forested one
  T1q   exact [r13]                    E-F8: the table must be built with the configured TOPK / NCAND / RADIAL_P
  T2a0  > 99.9 % keys within 0.2 km    annual-inventory coordinates of one plot key are one location
  T2a/b allclose (1e-3 m)              deterministic transforms
  T2c   > 99.5 % (R4, 2 km) / 99 %     FIA coordinate fuzz is < 1.6 km; state polygons simplified
  T3a   1e-3 m, all <= 1600 m          deterministic geometry; the fuzz disk radius
  T3b   all classes 1-4                candidates are forest land by construction
  T3c   2e-3                           float16 storage of AlphaEarth values in [-1, 1]
  T3c2  median max|Δ| < 0.02, cos q05 > 0.98, n > 50 (GA+FL mosaic), 2022 and 2025 [2022: r14 G47.8]; T3c2z [r13] every R4 UTM zone (15-18), >= 4 footprints, min cos > 0.999 (source COGs)  re-quantisation noise; a 1-block shift gives 0.77-0.996
  T3d   err < 1e-3 relative            float16 storage of soils
  T3e   |Δ| <= 2e-3                    float16 storage of fractions in ninths / severities in thirds
  T3f   ratio spread < 1 + 2e-3        the prior is a product of the spec's factors; float16-free (float64) storage
  T3g   rel err < 1e-4                 the inclusion weights sum to the disk's footprint count exactly
  T3j   exact [r13]                    g02e fia_negs = the FIA-protocol periods of the spec (G36 evidence end, G37 NONBLOCK, G39 lookback)
  T3h   all monotone, >= 10 plots [r13]  hi is a threshold; fewer than 10 plots with both classes is vacuous
  T3i   per state, year cells n >= 30 [r13]; post max < pre min and < clean − 0.05   time-axis alignment; pooled states hid KY / TN (G34)
        [r14 G47.2 / G48.1, set after seeing data] per detectable row (lr >= 4) of the REGION's table on its POOLED compatible classes; every class INFO
  T3k   exact [r14]                    G47.2: the verifier has its own evidence table for the region (else FAIL) and it equals meta.json evk (classes, window, lr per kind)
  T4a-c rtol 1e-4                      float32 model tensors
  T4d   >= 3200 m; VAL_SPLIT -> val non-empty [r13]   two 1.6 km disks cannot overlap
  T4e-m 1e-5 / 1e-7                    float32 closed forms of the flow
  T4n/o/p/q 1e-5 / 1e-4; SIGReg ratio > 20, |Δ| <= 25 % [r13]   G29-G31 closed forms; the shifted-cloud ratio is ~100x when the location is tested
        [r14] T4n bounded reset (G46.10: clamp(z_post + shift) target, states stay in ±3, clock >= age 0); T4q exact G46.8 verdicts
  T4r   1e-6                           the loss vanishes at a perfect prediction
  T4v   exact [r13]                    MIN_FSHARE vetting is deterministic
  T4w   1e-6 [r14]                     G46.20 / A14-8: a 2026 visit's flow state = the end-2025 state (the YEARS[-1] clamp), 2026 events dropped
  T4x   exact 0 [r14]                  G46.7: converted plots carry no MIL / Q / Q-decode weight, no g04 B-likelihood (trB) visit, no pair
  T4y   exact [r14]                    G47.8 / G48.5: tensor names / shapes = the configured model; the checkpoint's `switches` = the configuration (T4Y_COMPARED;
                                       unset = None; every recorded name known; SEED = the MODEL name's _s<seed>)
  T4g   SIGReg(N) < 0.2 SIGReg(3N)     the test statistic must separate a scaled cloud
  T4i   max |freq − prior| < 0.02      19 200 draws: binomial sd <= 0.0036
  T4l   1e-3 yr                        float32 ages
  T5    < 5 % of dims |t| > 3.5        an unbiased estimator; 3.5 sd keeps the family-wise false alarm low over ~200 dims
  T5s   [r13] aggregate: weighted sum t^2 < unweighted when the latter > df + 3 sqrt(2 df) (decides); per dim (weighted |bias| <= 0.5 x unweighted where
        unweighted |t| > 3.5) report-only [r14 G47.7]; plots with a NON-CONSTANT incl_w only [r14] (constant weights: weighted = unweighted)   E-F5 / E14-8; INFO when none
  T6a-d exact / 2e-3 / err < 1 [r13 T6c, T6d]   as T3c-e (float16 storage); the stand offset is exact (500 stands)
  T7a-c exact                          deterministic joins
  T7e   rtol 1e-9 [r13]                the production LOC=history prior is a deterministic restriction + renormalisation
  T8*   exact sets, weights rel 1e-6 (T8, T8c) / 1e-5 (T8h) [r13 weights, T8h, T8p]   deterministic pair rules; float32 / float64 renormalisation noise only
        [r14 G47.1] records = every FIADB record made at a visit dated in (v1, v2], table visit or not; mismatching pairs printed (T8, T8f, T8c, T8h)
        [r14 G46.5] undisturbed constraints clipped to the evidence end; [r14 G46.7] no pair of a converted plot (T8p)
  T8b   (a) exact on 3000 + 1000 random pairs under the configured PAIR_ANCHOR + LOAD=1 [r13]; (b) 0 violations, >= 1 pair   the evaluation must never see outcome-informed sets
        [r14 G47.6] PAIR_ANCHOR / SEVERITY_ANCHOR read from the environment; supported both / period and 0 / 1, any other value FAILs
  T8c   PAIR_ALL=1 -> multi-horizon pairs exist [r13]   E-F6: an empty family is vacuous
  T8h   [r14 G47.6] >= 300 consecutive pairs, >= 200 multi-horizon pairs (PAIR_ALL=1), >= 20 with a detectable cluster, >= 20 with a severity rule applied
        (training config)   the sample must exercise the event and severity rules (the r13 size conditions were tautological)
  T9    rel err < 1e-3                 float32 EM posterior
  T10   exact, >= 1 constrained plot [r13]   deterministic; an all-unconstrained sample is vacuous
  T11   max |Δ| < 0.11                 rounding to 0.1 / 0.01 in the FE layer (g07 uses 2.47105 since r13: E-F10)
  T11b  1e-6                           the cohorts are scaled to the summary exactly
  T11c/d exact                         G5 / G19 / G24 / G33 gate rule re-derived; missing keys FAIL [r13 F#10]
  T11e/f/g exact, 1e-6 [r13]           g05 QMD / dominant key / coverage and g07 cohorts JSON are deterministic
  T11h  exact [r14]                    G46.13: g05 maps with g04's estimator (the same `calibrated` helper and medoid draw, MS = 32 iid draws)
  T11i  rel 1e-6 [r14]                 G46.11 / G46.12 / G48.5: CAL, CALF, the validation errors / medoid Bray, the null and map_accept recomputed from
                                       g04_plots.parquet; every applied factor = the other half's fit; dumped nulls = an independent null (rel 1e-4)
  T12   exact set equalities [r13]     E-F3: g04 calibration = validation plots, MDN fit outside them + 3.2 km, training-only pools, random halves
        [r14 G47.5] every anchored statement assigned exactly once in g04 (else FAIL); behavioural null exclusion; halves not sorted / first-half / parity
  T13   argmax at offset (0, 0)        pixel-exact windows peak at no shift; N/A (reason printed) in Region 4 [r14]
  T14   no non-training plot, non-empty [r13 non-empty]   leakage; N/A in Region 4 [r14]
  T15   [r14 G47.4 / G46.8 / G46.17] GV_DYN_JSON only (unset / missing: FAIL); latent key required: non-clock max |mean| <= 2, min sd >= 0.3, share
        |z| > 2.9 <= 1 %; G7 per pre-registered horizon bin recomputed (true medians) from g06's per-pair dump: |Δ BA log change| <= 0.01 / yr, age
        ratio 0.7-1.3, every bin n >= 20 by the verifier's own recount (recorded n must equal it); set = val checked against the split (dump pairs =
        validation plots)"""
import sys, os, json, math, ast, hashlib, numpy as np, pandas as pd, duckdb, rasterio
from rasterio.windows import Window
from pyproj import Transformer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from stand_runs import stands_run; from common import curated_db, fiadb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); WS = os.environ.get('WORKSPACE', os.path.dirname(ROOT)); D = f"{ROOT}/data/proto/{os.environ.get('DATA', 'gafl')}"
AEF90 = os.environ.get('AEF90', '0') == '1'
def _verify_inputs():
    """every path main() reads under the current environment (G45.7: the driver fingerprints exactly this list): table files, stage outputs, rasters,
    the dynamics record GV_DYN_JSON and the checkpoint MODEL; the exec'd pipeline code is VERIFY_CODE"""
    try: SR_ = stands_run()
    except FileNotFoundError: SR_ = {}
    try: yrs, soils = (lambda m_: (m_['years'], m_['soils']))(json.load(open(f'{D}/meta.json')))
    except Exception: yrs, soils = list(range(2000, 2026)), ['clay030', 'sand030', 'soc030', 'awc030', 'ph030', 'restrdepth']
    MAPC = os.environ.get('GV_REQUIRE_MAP') != '0'                  # (vtab, GV_REQUIRE_MAP=0, reads no map-stage output)
    dd = [f'{D}/{f}' for f in ('meta.json', 'visits.parquet', 'tokens.parquet', 'pairs.parquet', 'plots.parquet', 'cands.parquet', 'cand_feat.npy', 'cand_hist.parquet',
                               'localisation.parquet', 'fia_events.parquet', 'fia_negs.parquet', 'stands.parquet', 'stand_feat.npy', 'cand_prior.parquet', 'location_em.json',
                               'cand_loc_aef.parquet', 'visit_aef')]
    if MAPC:
        dd += [f'{D}/{f}' for f in ('stand_veg_v3.parquet', 'stand_cohorts_v3.parquet', 'bayes_eval.json') + tuple(G04_DUMPS)]
        if os.environ.get('GV_DYN_JSON'): dd += [os.environ['GV_DYN_JSON'], g06_dump_path(os.environ['GV_DYN_JSON'])]
        if os.environ.get('MODEL'): dd.append(f"{D}/{os.environ['MODEL']}")
    fe = os.environ.get('FE_GEOJSON', os.path.join(os.path.dirname(ROOT), 'fe_pan/apps/forest-explorer/public/stands/baker_fl.geojson'))
    rs = [f'{ROOT}/data/aef/aef_2022.tif', f'{ROOT}/data/landfire/dist/dist_codes.csv']
    if AEF90:
        rs += [f'{ROOT}/data/r4/land.tif', f'{ROOT}/data/r4/hansen_lossyear.tif', f'{ROOT}/data/r4/soils90/fill.json', f'{ROOT}/data/aef/aef_2025.tif', f'{ROOT}/data/aef/aef_index.parquet',
               f'{WS}/TIGER/tl_2024_us_county/tl_2024_us_county.shp'] + [f'{ROOT}/data/aef90/aef90_{y}.tif' for y in (2022, 2025)] + \
              [f'{ROOT}/data/r4/soils90/{k}.tif' for k in soils] + [f'{ROOT}/data/r4/dist/dist_{y}.tif' for y in yrs]
    else:
        rs += [f'{ROOT}/data/aef/aef_2025.tif', f'{ROOT}/data/hansen/lossyear.tif'] + [f'{ROOT}/data/soils/{k}.tif' for k in soils] + [f'{ROOT}/data/landfire/dist/dist_{y}.tif' for y in yrs]
    fe2 = [fe, fe[:-len('.geojson')] + '_cohorts.json' if fe.endswith('.geojson') else fe + '_cohorts.json'] if MAPC else []
    return sorted(set([curated_db(), fiadb()] + dd + fe2 + rs + [SR_[k] for k in (('tif',) if AEF90 else ('land_clean', 'tif')) if k in SR_]))
def _verify_code():
    """the pipeline code main() execs or imports, plus every script T0 scans (the driver's pipeline_scripts(): T0's verdict depends on all of them)"""
    out = [f'{ROOT}/scripts/{f}' for f in ('g03b_train_gafl.py', 'g04_bayes_downscale.py', 'g05_map_stands.py', 'g02b_build_gafl.py', 'g02e_fia_records.py',
                                           'g02g_history_consistency.py', 'g02j_stand_footprints.py', 'check_comments.py', 'region.py', 'stand_runs.py', 'common.py',
                                           'features.py', 'stands_cfg.py')]
    try:
        import check_comments as _cc; out += list(_cc.pipeline_scripts(ROOT))
    except Exception: pass                                       # (T0 then FAILs in main: the list is unavailable)
    return sorted(set(out))
# ---------------------------------------------------------------- interfaces of the r14 pipeline outputs (G46): file names the checks read
G04_DUMPS = ('g04_plots.parquet',)                                                    # g04's per-plot dump (G48.5; bayes_eval.json plot_dump; T11i)
def g06_dump_path(dyn_json):
    """g06's per-pair dump next to its acceptance record (G46.17; T15): the record's `pair_dump` (a file name in the same directory), else by name
    (dynamics_check_<TAG>.json -> dynamics_pairs_<TAG>.parquet)"""
    try: pf = json.load(open(dyn_json)).get('pair_dump')
    except Exception: pf = None
    b_ = os.path.basename(dyn_json)
    if not pf: pf = ('dynamics_pairs_' + b_[len('dynamics_check_'):] if b_.startswith('dynamics_check_') else b_)[:-len('.json')] + '.parquet' if b_.endswith('.json') else b_ + '.parquet'
    return os.path.join(os.path.dirname(dyn_json), pf)
# ---------------------------------------------------------------- T15: the enforced dynamics acceptance, re-derived (G7 / G46.8 / G46.17 / G47.4)
T15_BINS = ((0, 8), (8, 15), (15, 99))        # pre-registered horizon bins (yr, lo <= horizon < hi; G7 r4, kept by G46.17); every bin must be populated
T15_NMIN = 20                                  # pairs per bin
def t15_judge(dj, dump, val_pids):
    """(G47.4) the pre-registered acceptance of g06's record dj re-derived: G46.8 latent criteria on the recorded non-clock statistics (max |mean| <= 2,
    min sd >= 0.3, share |z| > 2.9 <= 1 %; the `latent` key required); per pre-registered horizon bin the TRUE medians recomputed from the per-pair dump
    (rows of the acceptance set): n >= 20, |median predicted − observed BA log change| <= 0.01 / yr, observed age gain > 0 and the age-gain ratio 0.7-1.3;
    set = 'val' and every dump pair of that set on a validation plot (val_pids, from the split). Returns (ok, the rule's verdict, problems, per-bin numbers)"""
    why = []; ua = dj.get('undisturbed_acceptance', {}) if isinstance(dj, dict) else {}; lat = dj.get('latent') if isinstance(dj, dict) else None; lok = False
    if not isinstance(lat, dict): why.append('no latent key (G46.8)')
    else:
        miss = [k for k in ('max_abs_mean', 'min_sd', 'share_abs_gt_2p9') if lat.get(k) is None]
        if miss: why.append(f'latent lacks {miss} (G46.8)')
        else: lok = float(lat['max_abs_mean']) <= 2.0 and float(lat['min_sd']) >= 0.3 and float(lat['share_abs_gt_2p9']) <= 0.01
        if 'ok' in lat and bool(lat['ok']) != lok: why.append(f"g06's latent ok {lat['ok']} != the rule {lok}")
    if ua.get('set') != 'val': why.append(f"acceptance set {ua.get('set')!r}, not 'val' (G7)")
    if [tuple(b_) for b_ in ua.get('bins_required', [])] != list(T15_BINS): why.append(f"recorded bins {ua.get('bins_required')} != pre-registered {list(T15_BINS)}")
    bins = {}
    if dump is None: why.append('input missing: the per-pair dump (G46.17)')
    else:
        need = {'set', 'pid', 'horizon', 'pred_BA_logchange_per_yr', 'obs_BA_logchange_per_yr', 'pred_age_per_yr', 'obs_age_per_yr'}
        if not need <= set(dump.columns): why.append(f'dump lacks columns {sorted(need - set(dump.columns))}')
        else:
            d_ = dump[dump.set == ua.get('set')]; off = sorted(set(int(p_) for p_ in d_.pid) - set(int(p_) for p_ in val_pids))
            if off: why.append(f'{len(off)} acceptance-set plots are not validation plots, e.g. {off[:5]}')
            for lo, hi in T15_BINS:
                r_ = d_[(d_.horizon >= lo) & (d_.horizon < hi)]; n_ = len(r_); rec_ = dj.get(f'undisturbed_h{lo}_{hi}', {})
                if rec_ and 'n' in rec_ and int(rec_['n']) != n_: why.append(f'bin {lo}-{hi}: recorded n {rec_["n"]} != recount {n_}')
                if n_ < T15_NMIN: bins[(lo, hi)] = (n_, None); continue
                mp, mo = float(np.median(r_.pred_BA_logchange_per_yr)), float(np.median(r_.obs_BA_logchange_per_yr))
                ap, ao = float(np.median(r_.pred_age_per_yr)), float(np.median(r_.obs_age_per_yr))
                ok_ = abs(mp - mo) <= 0.01 and ao > 0 and 0.7 <= ap / ao <= 1.3; bins[(lo, hi)] = (n_, ok_)
                if rec_ and any(abs(float(rec_.get(k_, np.nan)) - v_) > 1e-9 for k_, v_ in (('median_pred_logchange_per_yr', mp), ('median_obs_logchange_per_yr', mo),
                                                                                         ('median_pred_age_per_yr', ap), ('median_obs_age_per_yr', ao))):
                    why.append(f'bin {lo}-{hi}: recorded medians != recomputed')
    verdict = lok and len(bins) == len(T15_BINS) and all(v_[1] is True for v_ in bins.values())
    if 'passed' in ua and bool(ua['passed']) != verdict: why.append(f"g06 recorded passed {ua['passed']} != the rule {verdict}")
    return (not why) and verdict, verdict, why, bins
# ---------------------------------------------------------------- the verifier's own FIA-record -> evidence tables, per region set (G47.2)
def evidence_spec(region):
    """{(record type 'T' / 'D', code): (evidence classes, (window lo, hi) around the record year, likelihood ratio)} for a region set (FIA state codes, any
    order), written out from the specification independently of g02b's EVK_TABLES; None = the verifier has no table for that region (checks FAIL).
    GA+FL: the STRICT data-derived rules (g09; 2026-10-22). 'H' = Hansen loss."""
    reg = tuple(sorted({int(x) for x in region}))
    if reg != (12, 13): return None
    S = {('T', 10): (['harvest', 'thin', 'unknown', 'H'], (-1, 1), 4), ('T', 20): (['harvest', 'chem', 'fire', 'unknown', 'H'], (-1, 1), 4),     # cutting / site prep / regeneration
         ('T', 30): (['harvest', 'chem', 'unknown', 'H'], (-2, 1), 4), ('T', 40): (['harvest', 'fire', 'unknown', 'H'], (-2, 1), 4),           # accept LANDFIRE 'unknown' (2026-10-22)
         ('T', 50): (['thin', 'chem'], (-1, 1), 2)}
    for cd in range(10, 23): S[('D', cd)] = (['insdis'], (-1, 1), 2)
    for cd in (30, 31): S[('D', cd)] = (['fire'], (-1, 1), 2)
    S[('D', 32)] = (['fire', 'H'], (-1, 1), 4)
    for cd in range(50, 55): S[('D', cd)] = (['weather'], (-1, 1), 2)
    S[('D', 80)] = (['harvest', 'dev'], (-1, 1), 2)          # DSTRBCD 70 (unknown) is not matched
    return S
def spec_kind(t_, cd):
    """the record kind (the name the table's evidence table uses) of an FIA code: TRTCD 10-50 -> T10-T50; DSTRBCD insects / disease 10-22, ground fire
    30-31, crown fire 32, weather 50-54, human 80; other codes carry no kind"""
    cd = int(cd)
    if t_ == 'T': return f'T{cd}' if cd in (10, 20, 30, 40, 50) else None
    return 'Dins' if 10 <= cd <= 22 else 'Dfire' if cd in (30, 31) else 'Dcrown' if cd == 32 else 'Dwx' if 50 <= cd <= 54 else 'Dhum' if cd == 80 else None
def evk_divergence(spec, evk_meta, classes):
    """(G47.2) the kinds whose evidence row differs between the verifier's spec and a table's meta.json evk (kind -> [class indices, [lo, hi], lr];
    index len(classes) + 1 = Hansen loss, len(classes) = severity): [(kind, the spec's row(s), the table's row)], empty when they agree"""
    nc_ = len(classes); nm = lambda i: 'H' if int(i) == nc_ + 1 else (classes[int(i)] if 0 <= int(i) < nc_ else f'?{int(i)}')
    mine = {}
    for (t_, cd), (cols, w, lr) in spec.items(): mine.setdefault(spec_kind(t_, cd), set()).add((frozenset(cols), (int(w[0]), int(w[1])), float(lr)))
    theirs = {k: (frozenset(nm(c) for c in v[0]), (int(v[1][0]), int(v[1][1])), float(v[2])) for k, v in evk_meta.items()}
    return [(k, sorted((sorted(a), w, l) for a, w, l in mine.get(k, ())), (sorted(theirs[k][0]), theirs[k][1], theirs[k][2]) if k in theirs else None)
            for k in sorted(set(mine) | set(theirs)) if len(mine.get(k, ())) != 1 or k not in theirs or next(iter(mine[k])) != theirs[k]]
def _assign_lines(tree, name):
    """line numbers of every statement of a parsed script that binds `name` (assignment / augmented / annotated / walrus / for / with targets, def / class,
    import alias)"""
    out = set()
    for n in ast.walk(tree):
        tg = []
        if isinstance(n, ast.Assign): tg = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor)): tg = [n.target]
        elif isinstance(n, ast.NamedExpr): tg = [n.target]
        elif isinstance(n, (ast.With, ast.AsyncWith)): tg = [i.optional_vars for i in n.items if i.optional_vars is not None]
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name: out.add(n.lineno)
        elif isinstance(n, (ast.Import, ast.ImportFrom)) and any((a_.asname or a_.name).split('.')[0] == name for a_ in n.names): out.add(n.lineno)
        for t in tg:
            if any(isinstance(m, ast.Name) and m.id == name for m in ast.walk(t)): out.add(n.lineno)
    return out
def t12_check(G4s, ns_, V, P, FSOK, VS12):
    """T12 (r13 E-F3, r14 G47.5) on g04's source G4s and the exec'd g03b namespace ns_: (ok, detail). FSOK: the independent MIN_FSHARE vetting per visit;
    VS12: VAL_SPLIT. Module level so a test harness can call it on a planted g04."""
    import textwrap as _tw12, torch
    ns12 = dict(ns_); _why12 = []; T4G = ast.parse(G4s); _L12 = G4s.splitlines()
    _v12, _vp12 = np.arange(len(V)), ns_['vpid']; ns12['XL'] = {p_: None for p_ in ns_['cand_of']}; ns12['vv'] = np.array([v for v in _v12 if _vp12[v] in ns12['XL']]); ns12['vi'] = {v: i for i, v in enumerate(ns12['vv'])}
    try:
        exec(compile(G4s[G4s.index('# calibration plots:'):G4s.index("print(f'MDN fit visits")], 'g04', 'exec'), ns12)
    except Exception as e_: _why12.append(f'set construction not executable ({type(e_).__name__}: {e_})')
    ANCH12 = {'lastv = V[fs_ok]': ('lastv',), 'trp_ = set(': ('trp_',), 'te_l = [': ('te_l', 'tr_l'), 'trs_v = [': ('trs_v',), 'p_trl = np.array(': ('p_trl',),
              'rec = lambda': ('rec',), 'cal_m = rec(': ('cal_m', 'te_m'), '_cp = sorted(': ('_cp', '_hp'), 'hA = [': ('hA', 'hB')}
    for pre, names in ANCH12.items():
        hit = [i + 1 for i, ln in enumerate(_L12) if ln.strip().startswith(pre)]
        if len(hit) != 1: _why12.append(f'anchor {pre!r}: {len(hit)} lines in g04 (exactly one required)'); continue
        for nm_ in names:
            al = _assign_lines(T4G, nm_)
            if al != {hit[0]}: _why12.append(f'{nm_!r} assigned on g04 lines {sorted(al)}, not only on its anchored line {hit[0]} (G47.5)')
        try: exec(compile(_tw12.dedent(_L12[hit[0] - 1]).strip(), 'g04', 'exec'), ns12)
        except Exception as e_: _why12.append(f'{pre!r}: {type(e_).__name__}: {e_}')
    # g04's own null_pred (its definition executed here) on a pool PLANTED with the judged half's visits at absurd values: wherever g04 calls null_pred on a
    # gate / validation half `cb`, its own excl expression must remove them (the result equals the unplanted pool's), and without excl they must show
    _nul12 = None
    if not _why12:
        try:
            fdef = next(n for n in ast.walk(T4G) if isinstance(n, ast.FunctionDef) and n.name == 'null_pred')
            ns12['o_tr'] = ns_['obs_summ'](torch.tensor(np.asarray(ns12['tr_l'], int))); ns12['y_tr'] = ns_['vyear'][np.asarray(ns12['tr_l'], int)]
            ns12['FB_SET'] = 'gv_verify'                                   # (g04's own FALLBACK counters: its assignment statement executed here)
            exec(compile(ast.Module(body=[next(n for n in ast.walk(T4G) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FALLBACK' for t in n.targets))], type_ignores=[]), 'g04', 'exec'), ns12)
            exec(compile(_tw12.dedent(ast.get_source_segment(G4s, fdef)), 'g04', 'exec'), ns12)
            calls = [n for n in ast.walk(T4G) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'null_pred' and any(isinstance(m, ast.Name) and m.id == 'cb' for a_ in n.args for m in ast.walk(a_))]
            hb_ = list(ns12['hB']); jud = set(int(x) for x in _vp12[hb_]); yq = ns_['vyear'][hb_]
            base = {k_: ns12[k_] for k_ in ('p_trl', 'o_tr', 'y_tr')}
            nul0 = ns12['null_pred'](yq, excl=jud)
            ns12['p_trl'] = np.concatenate([base['p_trl'], _vp12[hb_]]); ns12['y_tr'] = torch.cat([base['y_tr'], yq]); ns12['o_tr'] = [torch.cat([o_.float(), torch.full((len(hb_),) + tuple(o_.shape[1:]), 1e6)]) for o_ in base['o_tr']]
            res_c = []
            for cl in calls:
                kw = next((k_.value for k_ in cl.keywords if k_.arg == 'excl'), None)
                if kw is None: res_c.append((cl.lineno, 'no excl')); continue
                ns12['cb'] = hb_; ns12['ca'] = list(ns12['hA']); ex_ = set(int(x) for x in eval(compile(ast.Expression(kw), 'g04', 'eval'), ns12))
                got = ns12['null_pred'](yq, excl=ex_); res_c.append((cl.lineno, all(torch.allclose(g_.float(), n_.float(), equal_nan=True) for g_, n_ in zip(got[:3], nul0[:3]))))
            leak = ns12['null_pred'](yq, excl=None); shows = not all(torch.allclose(l_.float(), n_.float(), equal_nan=True) for l_, n_ in zip(leak[:3], nul0[:3]))   # (the planted pool moves the medians)
            for k_, v_ in base.items(): ns12[k_] = v_
            _nul12 = (len(calls) > 0 and all(r_ is True for _, r_ in res_c) and shows, f'{len(calls)} gate / validation null calls {res_c}, planted pool shows without excl {shows}')
        except Exception as e_: _nul12 = (False, f'null_pred not testable ({type(e_).__name__}: {e_})')
    if _why12: return False, '; '.join(_why12)
    trP, vaP, teP = set(_vp12[ns_['tr_v']]), set(_vp12[ns_['va_v']]), set(_vp12[ns_['te_v']]); have12 = set(ns_['cand_of'])
    calE = (vaP & have12) if VS12 else set(ns12['cal_pid'])
    _Pxy = P.set_index('pid')[['x', 'y']]; _cx = _Pxy.loc[sorted(calE)].values if calE else np.zeros((0, 2)); _dmin = np.full(len(_Pxy), np.inf)
    for i_ in range(0, len(_cx), 500): _dmin = np.minimum(_dmin, np.sqrt(((_Pxy.values[:, None, :] - _cx[None, i_:i_ + 500, :]) ** 2).sum(-1)).min(1))
    dmin12 = dict(zip(_Pxy.index.values, _dmin))
    fitE = {int(v) for v in ns_['tr_v'] if _vp12[v] not in calE and dmin12[_vp12[v]] >= 3200 and _vp12[v] in ns12['XL']}; fitG = {int(ns12['vv'][i]) for i in ns12['trA']}
    lastE = {}                                                    # last visit passing MIN_FSHARE of every plot (independent share)
    for v in np.flatnonzero(FSOK):
        p_ = _vp12[v]
        if p_ not in lastE or V.year.values[v] > V.year.values[lastE[p_]]: lastE[p_] = v
    gateE = {lastE[p_] for p_ in calE if p_ in lastE and 2022.0 <= V.year.values[lastE[p_]] < 2026.0}; gateG = set(ns12['cal_m'])
    hA_, hB_ = {_vp12[v] for v in ns12['hA']}, {_vp12[v] for v in ns12['hB']}; hall = hA_ | hB_
    srt = sorted(hall); seq = np.array([p_ in hA_ for p_ in srt]); nA, nB = int(seq.sum()), int((~seq).sum()); runs = 1 + int((seq[1:] != seq[:-1]).sum()) if len(seq) else 0
    mu_r = 1 + 2 * nA * nB / max(nA + nB, 1); var_r = 2 * nA * nB * (2 * nA * nB - nA - nB) / max((nA + nB) ** 2 * (nA + nB - 1), 1); zr = (runs - mu_r) / math.sqrt(var_r) if var_r > 0 else np.inf
    par = np.mean([p_ % 2 == 0 for p_ in hA_]) if hA_ else np.nan
    poolP = set(int(x) for x in ns12['p_trl']); knnE = {lastE[p_] for p_ in trP if p_ in lastE and lastE[p_] in ns12['vi']}
    res12 = {'cal = validation plots': set(ns12['cal_pid']) == calE and (not VS12 or len(calE) > 0),
             'MDN fit visits = training visits outside the calibration plots + 3.2 km': fitG == fitE and (min(dmin12[_vp12[v]] for v in fitG) >= 3200 if fitG else False),
             'gate visits = last vetted visits 2022-25 of the calibration plots': gateG == gateE and len(gateG) > 0,
             'gate halves disjoint, cover the gate set, balanced, random (runs test |z| < 4, both pid parities in half A: not parity / sorted / first-half)': not (hA_ & hB_) and hall == {_vp12[v] for v in gateG}
                 and len(hall) >= 20 and abs(nA - nB) <= 1 and abs(zr) < 4 and 0.0 < par < 1.0,
             'kNN / null pool = last vetted visits of training plots only': set(ns12['tr_l']) == knnE and poolP <= trP and not (poolP & (teP | vaP)),
             'gate / validation null excludes the judged plots (behavioural)': bool(_nul12 and _nul12[0])}
    return all(res12.values()), (
          f"failing {[k for k, v in res12.items() if not v]}; calibration plots {len(calE)}, MDN fit visits {len(fitG)} (expected {len(fitE)}), gate visits {len(gateG)} (plot halves {nA} / {nB}, "
          f"runs z {zr:.2f}, even-pid share of half A {par:.2f}), pool plots {len(poolP)}; null: {_nul12[1] if _nul12 else 'not run'}")
def t11h_check(G4h, G5h, ns_):
    """T11h (G46.13) on g04 / g05 sources and the exec'd g03b namespace: (ok, detail); module level for the planted-defect harness"""
    import torch
    T5h, T4h = ast.parse(G5h), ast.parse(G4h)
    nsh = dict(ns_); exec(compile(G4h[G4h.index('# decoded summaries'):G4h.index('t0 = time.time()')], 'g04', 'exec'), nsh)       # (the slice g05 executes)
    why_h = [f'g05 executes no {nm_}' for nm_ in ('MS', 'calibrated_median', 'medoid_draw') if nm_ not in nsh]
    MSh = nsh.get('MS')
    if MSh != 32: why_h.append(f'MS {MSh} != 32 (G46.13)')
    own_h = [nm_ for nm_ in ('MS', 'calibrated_median', 'medoid_draw', 'lmed', 'summaries') if _assign_lines(T5h, nm_)]
    if own_h: why_h.append(f'g05 defines its own {own_h}')
    for nm_ in ('calibrated_median', 'medoid_draw'):
        if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == nm_ for n in ast.walk(T5h)): why_h.append(f'g05 never calls {nm_}')
    smp = [n for n in ast.walk(T5h) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'sample' and isinstance(n.func.value, ast.Name) and n.func.value.id == 'mdn']
    dr_ = [eval(compile(ast.Expression(n.args[-1]), 'g05', 'eval'), {'USE_B': False, 'Mz': -1, 'MS': MSh}) for n in smp]
    if not smp or any(x != MSh for x in dr_): why_h.append(f'g05 draws {dr_} per production stand, not MS {MSh}')
    mp_ = next((n for n in ast.walk(T4h) if isinstance(n, ast.FunctionDef) and n.name == 'map_path'), None)
    if mp_ is None: why_h.append('g04 has no map_path')
    else:
        dfl = {a_.arg: d_ for a_, d_ in zip(mp_.args.args[-len(mp_.args.defaults):], mp_.args.defaults)} if mp_.args.defaults else {}
        if not (isinstance(dfl.get('ms'), ast.Name) and dfl['ms'].id == 'MS'): why_h.append('g04 map_path does not draw MS by default')
        ovr = [n.lineno for n in ast.walk(T4h) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'map_path' and (len(n.args) >= 3 or any(k_.arg == 'ms' for k_ in n.keywords))]
        if ovr: why_h.append(f'g04 calls map_path with its own draw count on lines {ovr}')
    torch.manual_seed(46); sh_ = torch.rand(7, 32) * 100 + 1; cfh = 0.37
    got_h = nsh['calibrated_median'](sh_, cfh) if 'calibrated_median' in nsh else None
    exp_h = torch.exp(torch.quantile(torch.log(sh_ + 1e-3), 0.5, dim=-1)) * math.exp(cfh) if got_h is not None else None
    if got_h is None or not torch.allclose(torch.as_tensor(got_h).float(), exp_h.float(), rtol=1e-4, atol=1e-3): why_h.append('calibrated_median(s, cf) != e^cf · true median of s')
    return not why_h, f'MS {MSh}; problems {why_h}'
def t4x_check(ns_, SRC4, G4x, V, BAtk):
    """T4x (G46.7) on g03b / g04 sources and the exec'd g03b namespace (pairs built): (ok, detail); updates the model (run it last)"""
    import torch
    LOOP4 = SRC4[SRC4.index('for ep in range(EP):'):SRC4.index('if SMOKE_B: print(')]
    exec(compile(next(ln for ln in SRC4.splitlines() if ln.startswith('TEMP = 0.05')), 'g03b', 'exec'), ns_)     # (the loop's constants: TEMP, K, EP, B, t0)
    _cand4 = np.array([v for v in ns_['tr_v'] if int(ns_['vpid'][v]) in ns_['cand_of'] and float(V.year.values[v]) >= 2024.5 and float(BAtk.get(int(v), 0.0)) > 5.0])
    b4 = np.random.default_rng(44).choice(_cand4, min(16, len(_cand4)), replace=False) if len(_cand4) else np.zeros(0, int); res4 = {}
    sv4 = {k: ns_.get(k) for k in ('tr_v', 'tr_good', 'E1', 'EP', 'SMOKE_B', 'B')}; vc4 = ns_['VCONV'].clone()
    for tag4, flag in (('unflagged', False), ('flagged', True)):
        ns_['VCONV'] = vc4.clone(); ns_['VCONV'][torch.tensor(b4)] = flag
        ns_.update(tr_v=b4, tr_good=b4, E1=0, EP=1, SMOKE_B=1, B=len(b4)); torch.manual_seed(45); np.random.seed(45)
        exec(compile(LOOP4, 'g03b', 'exec'), ns_); res4[tag4] = {k: float(ns_['tot'].get(k, np.nan)) for k in ('mil', 'q', 'qdec')}
    ns_.update(sv4); ns_['VCONV'] = vc4
    T4x_ = ast.parse(G4x)
    stB = next(n for n in ast.walk(T4x_) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'trB' for t in n.targets))   # g04's trB statement
    nsB = dict(ns_); nsB['loc_p'] = set(int(x) for x in ns_['vpid']); vcB = ns_['VCONV'].clone(); plB = np.asarray(ns_['tr_v'])[::7]; vcB[torch.tensor(plB)] = True; nsB['VCONV'] = vcB
    exec(compile(ast.Module(body=[stB], type_ignores=[]), 'g04', 'exec'), nsB); trB4 = set(int(x) for x in nsB['trB'])
    okB4 = len(trB4) > 0 and not (trB4 & set(int(x) for x in plB)) and trB4 == set(int(x) for x in ns_['tr_v']) - set(int(x) for x in plB) - set(np.flatnonzero(vc4.numpy()).tolist())
    ok4x = len(b4) >= 8 and all(res4['flagged'][k] == 0.0 for k in ('mil', 'q', 'qdec')) and all(abs(res4['unflagged'][k]) > 0 for k in ('mil', 'q', 'qdec')) and okB4
    return ok4x, (f'{len(b4)} training visits 2024.5+ with BA > 5: terms unflagged {res4.get("unflagged")}, flagged {res4.get("flagged")}; g04 trB: {len(trB4)} visits, planted converted inside {len(trB4 & set(int(x) for x in plB))}')
# ---------------------------------------------------------------- T4y / T11i (G48.5 interfaces: checkpoint `switches`, g04_plots.parquet)
T4Y_COMPARED = ('AGE_CH', 'AGE_DELTA', 'AGE_FLOW', 'ARCH', 'BATCH', 'CF16', 'CURRIC', 'DZ', 'EPOCHS', 'JUMP_RESET', 'KC', 'LOC', 'LR', 'MIN_FSHARE', 'PAIR_ALL',
                'PAIR_ANCHOR', 'PAIR_HMAX', 'QD_WIN', 'SEVERITY_ANCHOR', 'SIGREG_FLOW', 'SIGREG_MEAN', 'SITE_ATTR', 'STATE_ATTR', 'T_PAIR', 'VAL_SPLIT',
                'W_AGE', 'W_COMP', 'W_SUM')   # training switches: recorded value = configured value (unset = None); a new g03b switch (as G49 AGE_DELTA was) is
#   an unknown recorded name until it is added here (and to the driver's STAGE_SWITCHES), so T4y FAILs rather than ignore it
T4Y_CONTEXT = ('DATA', 'MODEL', 'NT', 'LOAD', 'SMOKE_BATCHES', 'STAND_DIAG', 'SEED')     # run context / performance: not compared (SEED: from the MODEL name)
def env_snapshot():
    """the configured training switches as this process received them (main() later sets EPOCHS / LOAD for its exec of g03b): {name: value or None}"""
    return {k: os.environ.get(k) for k in T4Y_COMPARED}
def t4y_switches(rec, env0, model_name):
    """(G47.8 / G48.5) the checkpoint's recorded `switches` against the configured ones: every compared switch equal (numbers numerically, unset = None),
    every recorded name known to the verifier (a new g03b switch must be added to T4Y_COMPARED), the recorded SEED = the MODEL name's _s<seed>.
    Returns (ok, detail)"""
    import re as _re
    if not isinstance(rec, dict): return False, 'input missing: the checkpoint records no `switches` (G48.5)'
    def same(a, b):
        if a in (None, '') or b in (None, ''): return a in (None, '') and b in (None, '')
        try: return abs(float(a) - float(b)) < 1e-12
        except (TypeError, ValueError): return str(a) == str(b)
    dif = {k: (rec.get(k), env0.get(k)) for k in T4Y_COMPARED if (k in rec or env0.get(k) not in (None, '')) and not same(rec.get(k), env0.get(k))}
    unk = sorted(set(rec) - set(T4Y_COMPARED) - set(T4Y_CONTEXT)); m_ = _re.search(r'_s(\d+)\.pt$', model_name or '')
    seed_ok = m_ is None or same(rec.get('SEED'), m_.group(1))
    return not dif and not unk and seed_ok, (f'{len(rec)} recorded switches; differing (recorded, configured) {dif}; recorded names the verifier does not know {unk}; '
                                            f'SEED recorded {rec.get("SEED")} vs MODEL name {m_.group(1) if m_ else "no _s<seed>"}')
def t11i_check(be, dump, null_ref=None, tol=1e-6):
    """(G46.11 / G46.12 / G48.5) g04's calibration and map acceptance recomputed from its per-plot dump (g04_plots.parquet), independently of g04's code:
      CAL  = true median over the A_cal visits of log(obs + 1e-3) − log(median draw);
      a factor fit (cf_fit) = the same over its STOCKED visits; CALF = the mean of the R_REP 'all' fits at map_y0; every score row's applied factors =
      the fit of the other half (same y0 and replicate);
      validation errors per replicate = mean over the two judged halves of the true median, over stocked visits, of |log((pred·e^cf + 1e-3) / (obs + 1e-3))|
      (QMD from the calibrated TPH / BA), medoid Bray = mean of the per-visit Bray; the decision statistics = replicate means;
      null = mean over the halves of the same median |log err| of the null prediction, null Bray = mean per-visit Bray;
      map_accept = TPH and BA errors and the medoid Bray each below the null's.
    null_ref: {vid: (TPH, BA, QMD)} independently recomputed null predictions (from the table) to compare with the dump's. Returns (ok, detail, numbers)"""
    why = []; d = dump
    def tmed_(x): x = np.asarray(x, float); return float(np.median(x)) if len(x) else float('nan')
    def close(a, b): return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    a_ = d[d.kind == 'A_cal']; cal = [tmed_(a_[f'obs_log_{k}'] - a_[f'pred_log_{k}']) for k in ('TPH', 'BA', 'QMD')]
    rc = be.get('calibration_log_factors_TPH_BA_QMD')
    if rc is None or not all(close(x, y) for x, y in zip(cal, rc)): why.append(f'CAL {cal} != recorded {rc}')
    y0u = float(be.get('map_y0', 2021.0)); cf = d[d.kind == 'cf_fit']
    def fit(rows): r_ = rows[rows.stocked]; return [tmed_(r_[f'obs_log_{k}'] - r_[f'pred_log_{k}']) for k in ('TPH', 'BA', 'QMD')]
    allf = [fit(g_) for _, g_ in cf[(cf.half == 'all') & np.isclose(cf.y0, y0u)].groupby('rep')]
    calf = [float(np.mean([f_[i] for f_ in allf])) for i in range(3)] if allf else [float('nan')] * 3
    rf = be.get('calibration_log_factors_mapping_path')
    if rf is None or not all(close(x, y) for x, y in zip(calf, rf)): why.append(f'CALF {calf} != recorded {rf}')
    sc = d[(d.kind == 'score') & (d.set == 'validation') & np.isclose(d.y0, y0u)]; reps = {}; badcf = 0
    for (rep_, half), g_ in sc.groupby(['rep', 'half']):
        other = {'A': 'B', 'B': 'A'}[half]; f_ = fit(cf[(cf.half == other) & (cf.rep == rep_) & np.isclose(cf.y0, y0u)])
        badcf += int(sum((~np.isclose(g_[f'cf_{k}'].values, f_[i], rtol=tol, atol=tol)) for i, k in enumerate(('TPH', 'BA', 'QMD'))).astype(bool).sum())   # (rows)
        s_ = g_[g_.stocked]; P = {k: s_[f'pred_{k}'].values * np.exp(f_[i]) for i, k in enumerate(('TPH', 'BA', 'QMD'))}   # (the factors re-fitted here, not the dumped ones)
        P['QMD'] = np.where(P['TPH'] > 0, 100 * np.sqrt(P['BA'] / np.maximum(P['TPH'] * math.pi / 4, 1e-9)), 0.0)
        e_ = {k: tmed_(np.abs(np.log((P[k] + 1e-3) / (s_[f'obs_{k}'].values + 1e-3)))) for k in ('TPH', 'BA', 'QMD')}
        e_['bray_medoid'] = float(s_.bray_medoid.mean()) if len(s_) else float('nan'); reps.setdefault(rep_, []).append(e_)
    if badcf: why.append(f'{badcf} score rows apply factors that are not the other half\'s fit')
    if not reps or any(len(v_) != 2 for v_ in reps.values()): why.append(f'validation score rows: replicates {sorted(reps)} with {[len(v_) for v_ in reps.values()]} halves (2 each required)')
    xv = {k: float(np.mean([np.mean([h_[k] for h_ in v_]) for v_ in reps.values()])) if reps else float('nan') for k in ('TPH', 'BA', 'QMD', 'bray_medoid')}
    bl = d[(d.kind == 'baseline') & (d.set == 'validation')]; nv = {k: 0.0 for k in ('TPH', 'BA', 'QMD', 'bray')}; nh = 0
    for half, g_ in bl.groupby('half'):
        s_ = g_[g_.stocked]; nh += 1
        for k in ('TPH', 'BA', 'QMD'): nv[k] += tmed_(np.abs(np.log((s_[f'null_{k}'].values + 1e-3) / (s_[f'obs_{k}'].values + 1e-3))))
        nv['bray'] += float(s_.bray_null.mean())
    nv = {k: v_ / nh if nh else float('nan') for k, v_ in nv.items()}
    if nh != 2: why.append(f'validation baseline halves {nh} (2 required)')
    xr = be.get('mapping_path_validation_crossfit', {}); rme, rco, rnu = xr.get('median_abs_log_err', {}), xr.get('composition', {}), xr.get('null', {})
    for k in ('TPH', 'BA', 'QMD'):
        if not close(xv[k], float(rme.get(k, np.nan))): why.append(f'validation {k} error {xv[k]:.6f} != recorded {rme.get(k)}')
        if not close(nv[k], float(rnu.get(k, np.nan))): why.append(f'null {k} error {nv[k]:.6f} != recorded {rnu.get(k)}')
    if not close(xv['bray_medoid'], float(rco.get('bray_medoid', np.nan))): why.append(f"medoid Bray {xv['bray_medoid']:.6f} != recorded {rco.get('bray_medoid')}")
    if not close(nv['bray'], float(rnu.get('bray', np.nan))): why.append(f"null Bray {nv['bray']:.6f} != recorded {rnu.get('bray')}")
    nnull = 0
    if null_ref is not None:
        for r_ in bl.itertuples():
            ref = null_ref.get(int(r_.vid))
            if ref is None or not all(abs(float(getattr(r_, f'null_{k}')) - ref[i]) <= 1e-4 * max(1.0, abs(ref[i])) for i, k in enumerate(('TPH', 'BA', 'QMD'))): nnull += 1
        if nnull: why.append(f'{nnull} of {len(bl)} dumped null predictions differ from the independent null (training pool, ±3 yr, judged plots excluded)')
    acc = all(math.isfinite(xv[k]) and math.isfinite(nv[k]) and xv[k] < nv[k] for k in ('TPH', 'BA')) and math.isfinite(xv['bray_medoid']) and xv['bray_medoid'] < nv['bray']
    if be.get('map_accept') is None: why.append('input missing: bayes_eval.json map_accept (G46.12)')
    elif bool(be['map_accept']) != acc: why.append(f"map_accept {be['map_accept']} != the rule {acc}")
    if xr.get('set') != 'validation plots': why.append(f"decided on {xr.get('set')!r}, not the validation plots")
    return not why and acc, f'CAL {np.round(cal, 4).tolist()}, CALF {np.round(calf, 4).tolist()}; validation {dict((k, round(v_, 4)) for k, v_ in xv.items())} vs null {dict((k, round(v_, 4)) for k, v_ in nv.items())}; rule {acc}, map_accept {be.get("map_accept")}; problems {why}', (cal, calf, xv, nv, acc)
def t11i_null_ref(ns_, V, FSOK, dump):
    """{vid: (TPH, BA, QMD)} the NULL prediction of every dumped validation baseline visit recomputed from the table: the true median of obs_summ over the
    pool = the last visit passing MIN_FSHARE of every training plot with candidates, within ±3 yr of the query visit, the judged half's plots excluded"""
    import torch
    vp = ns_['vpid']; trP = set(int(x) for x in vp[ns_['tr_v']]); last = {}
    for v in np.flatnonzero(FSOK):
        p_ = int(vp[v])
        if p_ in trP and p_ in ns_['cand_of'] and (p_ not in last or V.year.values[v] > V.year.values[last[p_]]): last[p_] = v
    pool = np.array(sorted(last.values())); ob = ns_['obs_summ'](torch.tensor(pool)); oy = V.year.values[pool]; opl = vp[pool]
    bl = dump[(dump.kind == 'baseline') & (dump.set == 'validation')]; out = {}
    for half, g_ in bl.groupby('half'):
        jud = set(int(x) for x in g_.pid)
        for vid in g_.vid.values:
            m_ = (np.abs(oy - V.year.values[int(vid)]) <= 3) & ~np.isin(opl, list(jud))
            out[int(vid)] = tuple(float(np.median(ob[i].numpy()[m_])) if m_.any() else float('nan') for i in range(3))
    return out
# ---------------------------------------------------------------- LANDFIRE code tables, decoded independently of g02b (shared with gv_verify_upstream U8)
def landfire_codes(path, classes):
    """({(file year, code): (class index or None, event calendar year)}, {(file year, code): severity 1-3}) of the LANDFIRE annual-disturbance code tables:
    type -> class (burned 'Unknown' -> fire; 'Increased Green', water and unmapped pixels are no disturbance); event year = CALENDAR_YR of the
    fiscal-year products, else the file year; severity: Low 1 .. High 3 (Medium = Moderate in the 2013-14 / 2024-25 products; no rating = neutral 2)"""
    codes = pd.read_csv(path, low_memory=False)
    cmap = {'harvest': {'Clearcut', 'Harvest'}, 'thin': {'Thinning', 'Mastication', 'Other Mechanical', 'Mechanical Remove', 'Mechanical Add', 'Mechanical Unknown'},
            'fire': {'Wildfire', 'Fire', 'Prescribed Fire', 'Wildland Fire Use', 'Wildland Fire', 'Unknown/Possibly Fire'}, 'insdis': {'Insects/Disease', 'Insects', 'Disease'},
            'weather': {'Weather'}, 'chem': {'Herbicide', 'Chemical', 'Insecticide', 'Biological'}, 'dev': {'Development'}, 'unknown': {'Unknown', 'Unknown/Fire Doubtful'}}
    codes['desc'] = (codes.DESCRIPTION.fillna('') + ' ' + codes.DESCRIPTIO.fillna('')).str.replace(r'\s+', ' ', regex=True)
    codes['ey'] = codes.CALENDAR_YR.fillna(codes.year).astype(int)
    def cls_of(r):
        t = str(r.DIST_TYPE).strip()
        if 'Green' in str(r.SEVERITY) or str(r.SEVERITY).strip() in ('Water', 'Nodata/Non-processing mask'): return None
        if t == 'Unknown' and any(p in r.desc for p in ('indicates burned', 'burn probability is 95% or greater', 'most likely fire')): t = 'Wildfire'
        return next((i for i, k in enumerate(classes) if t in cmap[k]), None)
    SEVS = {'Low': 1, 'Unburned/Low': 1, 'Low-Medium': 1.5, 'Medium-Low': 1.5, 'Low-Moderate': 1.5, 'Moderate-Low': 1.5, 'Medium': 2, 'Moderate': 2, 'No Severity': 2,
            'Medium-High': 2.5, 'High-Medium': 2.5, 'Moderate-High': 2.5, 'High-Moderate': 2.5, 'High': 3}
    _sv = lambda r: SEVS['No Severity'] if pd.isna(r.SEVERITY) else SEVS[str(r.SEVERITY).strip()]
    lk = {(int(r.year), int(r.VALUE)): (cls_of(r), int(r.ey)) for r in codes.itertuples()}
    return lk, {(int(r.year), int(r.VALUE)): _sv(r) for r in codes.itertuples() if cls_of(r) is not None}
# ---------------------------------------------------------------- provenance of the table (G46.6 / G47.9): stale vs divergent
PROV_KEY = b'genveg_provenance'
SCRIPT_OF = {'g02b': 'g02b_build_gafl.py', 'g02e': 'g02e_fia_records.py', 'g02g': 'g02g_history_consistency.py', 'g02j': 'g02j_stand_footprints.py'}
PROV_FILE = {'g02e': 'fia_events.parquet', 'g02g': 'cand_hist.parquet', 'g02j': 'stands.parquet'}        # (g02b: meta.json code_hash)
def code_hash(path):
    """a script's provenance hash as G46.6 defines it: sha256 (hex) of ast.dump of the parsed module after removing the docstring of the module and of
    every def / class (an emptied body becomes `pass`): comments, docstrings and formatting do not count"""
    tree = ast.parse(open(path).read())
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr) \
                and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()
def recorded_provenance(d_):
    """{script: (code hash of that script recorded with the table, the g02b hash recorded with it)} from meta.json code_hash (g02b) and the parquet
    provenance records of g02e / g02g / g02j (None: nothing recorded)"""
    out = {}
    try: out['g02b'] = (json.load(open(f'{d_}/meta.json')).get('code_hash', {}).get('g02b'), None)
    except Exception: out['g02b'] = (None, None)
    import pyarrow.parquet as pq_
    for s_, f_ in PROV_FILE.items():
        try:
            md_ = pq_.read_schema(f'{d_}/{f_}').metadata or {}; pr_ = json.loads(md_[PROV_KEY]) if PROV_KEY in md_ else {}
            out[s_] = (pr_.get('code_hash', {}).get(s_), pr_.get('code_hash', {}).get('g02b'))
        except Exception: out[s_] = (None, None)
    return out
def provenance_label(scripts, rec, cur):
    """the stale / divergent label of a failing check whose inputs the given table scripts produced (rec: recorded_provenance, cur: current hashes)"""
    if not scripts: return 'divergent (the check reads current pipeline code / outputs without a table code hash: the code disagrees with the spec)'
    miss = [s_ for s_ in scripts if rec.get(s_, (None, None))[0] is None]
    if miss: return f'stale (no code hash recorded for {", ".join(miss)}: the table predates G46.6; rebuild before triage)'
    old = [s_ for s_ in scripts if rec[s_][0] != cur.get(s_)]
    if old: return f'stale ({", ".join(old)} changed since the table was built)'
    lag = [s_ for s_ in scripts if s_ != 'g02b' and rec[s_][1] is not None and rec[s_][1] != rec['g02b'][0]]
    if lag: return f'stale ({", ".join(lag)} ran on an older g02b table)'
    return f'divergent (built by the current {", ".join(scripts)}: the code disagrees with the spec)'
# which table scripts produced the inputs of each check (by check id); checks absent here read no table output
PRODUCERS = {**{k: ('g02b',) for k in ('T1a', 'T1b', 'T1c', 'T1d', 'T1e', 'T1f', 'T1g', 'T1h', 'T1i', 'T1j', 'T1k', 'T1l', 'T1m', 'T1o', 'T1q', 'T1r',
                                       'T2a0', 'T2a', 'T2b', 'T2c', 'T3a', 'T3b', 'T3c', 'T3c0', 'T3c2', 'T3c2z', 'T3d', 'T3e', 'T3f', 'T3g', 'T3h', 'T3i', 'T3k',
                                       'T4a', 'T4b', 'T4c', 'T4d', 'T4i', 'T4l', 'T4r', 'T4v', 'T5', 'T5s', 'T7a')},
             'T1n': ('g02b', 'g02e'), 'T3j': ('g02b', 'g02e'), 'T6a': ('g02b', 'g02j'), 'T6b': ('g02b', 'g02j'), 'T6c': ('g02b', 'g02j'),
             'T6d': ('g02b', 'g02j'), 'T7b': ('g02b', 'g02e', 'g02g'), 'T7e': ('g02b', 'g02e', 'g02g'), 'T10': ('g02b', 'g02e', 'g02g'),
             **{k: ('g02b', 'g02e') for k in ('T8p', 'T8', 'T8e', 'T8f', 'T8g', 'T8c', 'T8b', 'T8d')}, 'T8h': ('g02b', 'g02e', 'g02g')}
VERIFY_INPUTS = _verify_inputs()
VERIFY_CODE = _verify_code()
def main(argv=None, force_pairs=(), stop_after=None):
    argv = sys.argv if argv is None else argv; NS = int(argv[1]) if len(argv) > 1 else 300; FORCE = [int(x) for x in force_pairs]
    rng = np.random.default_rng(42); RES = []; _rec0 = recorded_provenance(D); ENV0 = env_snapshot()
    _cur0 = {s_: (code_hash(f'{ROOT}/scripts/{f_}') if os.path.exists(f'{ROOT}/scripts/{f_}') else None) for s_, f_ in SCRIPT_OF.items()}
    def check(name, ok, detail=''):
        RES.append((name, bool(ok))); lab = '' if ok else f'  [{provenance_label(PRODUCERS.get(name.split()[0], ()), _rec0, _cur0)}]'
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}{lab}", flush=True)
    def done(tag):                                                        # (test harness: stop after the named block)
        if stop_after == tag:
            nf_ = sum(1 for _, ok in RES if not ok); print(f'\n(stopped after {tag}) {len(RES) - nf_} / {len(RES)} checks PASS'); return True
        return False
    import check_comments as _cc0
    try: _f0 = list(_cc0.pipeline_scripts(ROOT)); _e0 = ''
    except Exception as e_: _f0 = []; _e0 = f'; the driver\'s list is unavailable ({type(e_).__name__}: {e_})'
    _h0 = [(os.path.basename(f), ln) for f in _f0 for ln, _ in _cc0.swallowed(f)]
    check('T0 no code swallowed by an inline comment in the pipeline scripts (static scan of the driver\'s pipeline_scripts(), G47.8)', len(_f0) > 0 and not _h0, f'{len(_f0)} files; hits {_h0}{_e0}')
    M = json.load(open(f'{D}/meta.json')); from region import region_check; region_check(M); KEYS = M['keys']; YEARS = M['years']; NC = len(M['classes']); NY = len(YEARS)
    # T1q (r13 E-F8): the table's recorded build switches = the configured ones (the driver passes the table config keys TOPK / NCAND / RADIAL_P; absent -> skipped)
    _m8 = {'TOPK': len(KEYS) - 2, 'NCAND': M.get('ncand'), 'RADIAL_P': M.get('radial_p')}; _e8 = {k: os.environ[k] for k in _m8 if os.environ.get(k) not in (None, '')}
    if _e8: check('T1q meta.json topk / ncand / radial_p = the configured table switches', all(_m8[k] is not None and abs(float(_m8[k]) - float(v)) < 1e-9 for k, v in _e8.items()),
                  f'configured {_e8}, table {_m8}')
    else: print('SKIP  T1q no TOPK / NCAND / RADIAL_P in the environment (the driver passes them)')
    # region switches (as g02b): REGION = FIA STATECDs; AEF90 = Region 4 rasters (AlphaEarth / soils at 90 m: a footprint = one 90 m pixel, centres = 1 mod 3)
    REGION = tuple(int(x) for x in os.environ.get('REGION', '12,13').split(',')); ST_SQL = f"({','.join(map(str, REGION))})"; AEF90 = os.environ.get('AEF90', '0') == '1'
    R4 = tuple(sorted(set(REGION))) != (12, 13)                    # Region 4 (any region set but GA+FL)
    # T3k (G47.2) the verifier's own evidence table for this region set, and the table's meta.json evk against it. With no table for the region every
    # spec-dependent check FAILs (it is still computed with the GA+FL table and its verdict shown, for diagnosis only)
    SPEC = evidence_spec(REGION); NOSPEC = SPEC is None
    if NOSPEC:
        SPEC = evidence_spec((12, 13))
        check('T3k the verifier has an evidence table for the region, = meta.json evk', False, f'no verifier evidence table for region {sorted(set(REGION))} (G47.2); spec checks FAIL')
    elif 'evk' not in M: check('T3k the verifier has an evidence table for the region, = meta.json evk', False, 'input missing: meta.json has no evk (G46.4)')
    else:
        _dv = evk_divergence(SPEC, M['evk'], M['classes'])
        check('T3k the verifier has an evidence table for the region, = meta.json evk (classes / window / likelihood ratio per record kind); evidence end 2024 (G36)',
              not _dv and M.get('evid_end') == 2024, f'{len(M["evk"])} kinds in the table; diverging (kind, spec, table): {_dv}; meta.json evid_end {M.get("evid_end", "absent")}')
    SPEC_CHECKS = {'T3f', 'T3j', 'T3i', 'T8', 'T8e', 'T8f', 'T8g', 'T8c', 'T8b', 'T8d', 'T8h', 'T9', 'T10'}
    if NOSPEC:
        _check0 = check
        def check(name, ok, detail=''):
            if name.split()[0] in SPEC_CHECKS: return _check0(name, False, f'no verifier evidence table for region {sorted(set(REGION))} (G47.2); with the GA+FL table: {"PASS" if ok else "FAIL"} {detail}')
            return _check0(name, ok, detail)
    LAND = f'{ROOT}/data/r4/land.tif' if AEF90 else stands_run()['land_clean']
    AEFP = {y: f'{ROOT}/data/aef90/aef90_{y}.tif' if AEF90 else f'{ROOT}/data/aef/aef_{y}.tif' for y in (2022, 2025)}
    SOILP = (lambda k: f'{ROOT}/data/r4/soils90/{k}.tif') if AEF90 else (lambda k: f'{ROOT}/data/soils/{k}.tif')
    DISTP = (lambda y: f'{ROOT}/data/r4/dist/dist_{y}.tif') if AEF90 else (lambda y: f'{ROOT}/data/landfire/dist/dist_{y}.tif')
    HANSP = f'{ROOT}/data/r4/hansen_lossyear.tif' if AEF90 else f'{ROOT}/data/hansen/lossyear.tif'
    key = ['st', 'u', 'c', 'p']; V = pd.read_parquet(f'{D}/visits.parquet'); TK = pd.read_parquet(f'{D}/tokens.parquet'); PR = pd.read_parquet(f'{D}/pairs.parquet')
    P = pd.read_parquet(f'{D}/plots.parquet'); C = pd.read_parquet(f'{D}/cands.parquet'); CF = np.load(f'{D}/cand_feat.npy', mmap_mode='r')
    con = duckdb.connect(curated_db(), read_only=True)
    # ======================= T1 visits and cohort tokens (SQL aggregation, independent of the pandas builder)
    # (r11 M-4: ranked over live trees on FORESTED conditions only — the cohort table describes forested conditions — FIADB COND_STATUS_CD by plot / date / CONDID)
    _lt = con.execute(f"""select r.SPECIES_SYMBOL sym, t.STATECD st, t.UNITCD u, t.COUNTYCD c, t.PLOT p, t.MEASDATE md, t.CONDID cond from curated_trees t join REF_SPECIES r on r.SPCD=t.SPCD
                           where t.STATECD in {ST_SQL} and t.MEASDATE>='2000-01-01' and t.STATUSCD=1""").df()
    _cs = duckdb.connect(fiadb(), read_only=True).execute(f"""select p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p,
                (case when p.MEASMON between 1 and 12 and p.MEASDAY between 1 and 31 then make_date(p.MEASYEAR::int, p.MEASMON::int, p.MEASDAY::int) end) md, c.CONDID cond, c.COND_STATUS_CD cs
                from PLOT p join COND c on c.PLT_CN = p.CN where p.STATECD in {ST_SQL} and p.MEASYEAR >= 2000""").df()
    _lt['md'] = pd.to_datetime(_lt.md); _cs['md'] = pd.to_datetime(_cs.md); _lt = _lt.merge(_cs, on=['st', 'u', 'c', 'p', 'md', 'cond'], how='left')
    _vc = _lt[_lt.cs == 1].groupby('sym').size().reset_index(name='n').sort_values(['n', 'sym'], ascending=[False, True]); top = list(_vc.sym[:len(KEYS) - 2])
    check(f'T1b species keys = top-{len(KEYS) - 2} live species on forested conditions (SQL + FIADB COND) + _H/_S', sorted(top) + ['_H', '_S'] == KEYS, f'{sorted(top)[:5]}…')
    samp = V.sample(min(NS, len(V)), random_state=1)
    reg = duckdb.connect(); reg.register('samp', samp[key + ['vid']].assign(yr=samp.year))
    q = f"""select t.STATECD st, t.UNITCD u, t.COUNTYCD c, t.PLOT p, t.MEASDATE md, r.SPECIES_SYMBOL sym, r.SFTWD_HRDWD grp, t.DIA dia, t.HT ht, t.CR cr, t.TPA_UNADJ tpa,
                  t.estimated_age age, t.STATUSCD status, t.cond_stdorgcd org, t.CONDID cond, t.SUBP subp
           from curated_trees t join REF_SPECIES r on r.SPCD = t.SPCD where t.STATECD in {ST_SQL} and t.MEASDATE >= '2000-01-01'"""
    T = con.execute(q).df(); T['md'] = pd.to_datetime(T.md)
    # ---- the visit set, rebuilt independently from FIADB (pandas aggregation of per-condition rows; string dates):
    #      visits with curated trees on a forested condition  ∪  forested FIADB visits of the same plots with no live tree in FIADB TREE;
    #      forested share of the SAMPLED area (non-sampled status 5 excluded) >= 0.1
    _fx = duckdb.connect(fiadb(), read_only=True)
    _cc = _fx.execute(f"""select p.CN pcn, p.MEASYEAR my, p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, printf('%04d-%02d-%02d', p.MEASYEAR::int, p.MEASMON::int, p.MEASDAY::int) ds,
            c.CONDID cond, c.COND_STATUS_CD cs, c.SUBPPROP_UNADJ sp, c.MICRPROP_UNADJ mp, c.STDORGCD org from PLOT p join COND c on c.PLT_CN = p.CN
            where p.STATECD in {ST_SQL} and p.MEASYEAR >= 2000""").df()
    # (G40) a bare visit has no live TALLIED tree on a forested condition (STATUSCD 1, TPA_UNADJ > 0, SUBP <= 4, COND_STATUS_CD 1): a correlated subquery here
    _nl = _fx.execute(f"""select t.PLT_CN pcn, count(*) nl from TREE t where t.STATECD in {ST_SQL} and t.STATUSCD = 1 and t.TPA_UNADJ > 0 and t.SUBP <= 4
                          and exists (select 1 from COND c where c.PLT_CN = t.PLT_CN and c.CONDID = t.CONDID and c.COND_STATUS_CD = 1) group by 1""").df()
    _f = _cc.assign(sp1=np.where(_cc.cs == 1, _cc.sp, 0), sp5=np.where(_cc.cs == 5, _cc.sp, 0), mp1=np.where(_cc.cs == 1, _cc.mp, 0), mp5=np.where(_cc.cs == 5, _cc.mp, 0),
                    org1=np.where(_cc.cs == 1, _cc.org, np.nan)).groupby(['pcn'] + key + ['ds']).agg(sp1=('sp1', 'sum'), sp5=('sp5', 'sum'), mp1=('mp1', 'sum'), mp5=('mp5', 'sum'), org=('org1', 'max')).reset_index()
    _f = _f.merge(_nl, on='pcn', how='left').fillna({'nl': 0}); _f = _f[_f.sp1 > 0]
    _f['fs'] = np.minimum(_f.sp1 / (1 - _f.sp5), 1.0); _f['fm'] = np.where(_f.mp1 > 0, np.minimum(_f.mp1 / (1 - _f.mp5), 1.0), _f.fs)      # shares of the SAMPLED area (visit filter)
    _f['fsw'] = np.minimum(_f.sp1, 1.0); _f['fmw'] = np.where(_f.mp1 > 0, np.minimum(_f.mp1, 1.0), _f.fsw)          # shares of the WHOLE plot: TPA_UNADJ expands to it (densities)
    T['ds'] = T.md.dt.strftime('%Y-%m-%d'); T = T.merge(_cc[key + ['ds', 'cond', 'cs']], on=key + ['ds', 'cond'], how='left'); nT = len(T); T = T[T.cs == 1]
    _tv = T[key + ['ds']].drop_duplicates()
    # (G46.1) INCOMPLETE curated visits: FIADB has >= 3 live tallied trees (STATUSCD 1, TPA_UNADJ > 0, SUBP <= 4) on forested conditions and the curated
    # table fewer than half as many such trees (on the forested conditions) -> not a table visit (no tokens, no pair endpoint, no bare visit)
    NCUR = T[(T.status == 1) & (T.tpa > 0) & (T.subp <= 4)].groupby(key + ['ds']).size().rename('n_cur').reset_index()
    _iv = _tv.merge(NCUR, on=key + ['ds'], how='left').merge(_f[key + ['ds', 'nl', 'fs']], on=key + ['ds'], how='inner'); _iv['n_cur'] = _iv.n_cur.fillna(0)
    INC = _iv[(_iv.nl >= 3) & (_iv.n_cur < 0.5 * _iv.nl)][key + ['ds', 'n_cur', 'nl', 'fs']]
    _tvk = _tv.merge(INC[key + ['ds']].assign(_x=1), on=key + ['ds'], how='left'); _tvk = _tvk[_tvk._x.isna()][key + ['ds']]
    # (G40) + FIADB plots bare at EVERY forested visit whose EPA L3 ecoregion (FIADB data_plot_eco) is in the curated table's L3 scope (the L3s of its plots)
    _eco = _fx.execute(f"select statecd st, unitcd u, countycd c, plot p, epa_l3 l3 from data_plot_eco where statecd in {ST_SQL}").df()
    L3S = set(_tv[key].drop_duplicates().merge(_eco, on=key).l3.dropna()); _ab = _f.groupby(key).nl.max(); _ab = _ab[_ab == 0].reset_index()[key].merge(_eco, on=key)
    _ab = _ab[_ab.l3.isin(L3S)][key]
    _bare = _f[_f.nl == 0].merge(pd.concat([_tv[key].drop_duplicates(), _ab]).drop_duplicates(), on=key).merge(_tv.assign(_i=1), on=key + ['ds'], how='left')
    _bare = _bare[_bare._i.isna()][key + ['ds']]
    EX = pd.concat([_tvk, _bare]).merge(_f[key + ['ds', 'fs', 'fm', 'fsw', 'fmw', 'org']], on=key + ['ds'], how='inner'); EX = EX[EX.fs >= 0.1]
    EX['md'] = pd.to_datetime(EX.ds); EX = EX.sort_values(key + ['md']).reset_index(drop=True); EX['vid'] = np.arange(len(EX)); EX['year'] = EX.md.dt.year + (EX.md.dt.dayofyear - 0.5) / np.where(EX.md.dt.is_leap_year, 366.0, 365.0)
    check('T1a visit set = complete curated forested-condition visits ∪ bare FIADB visits of their plots and of all-bare plots in the L3 scope, share >= 0.1 (rebuilt from FIADB)', len(EX) == len(V),
          f'independent {len(EX)} (bare {len(_bare)}; all-bare plots in the L3 scope {len(_ab)}; incomplete curated visits left out {int((INC.fs >= 0.1).sum())}) table {len(V)}; only independent '
          f'{len(set(map(tuple, EX[key + ["ds"]].values)) - set(map(tuple, V.assign(ds=pd.to_datetime(V.md).dt.strftime("%Y-%m-%d"))[key + ["ds"]].values)))}, only table '
          f'{len(set(map(tuple, V.assign(ds=pd.to_datetime(V.md).dt.strftime("%Y-%m-%d"))[key + ["ds"]].values)) - set(map(tuple, EX[key + ["ds"]].values)))}; non-forest-condition tree records {nT - len(T)}')
    # T1m (r13 A-8) FIADB plots of the region with a live tallied tree on a forested visit (share >= 0.1) inside the curated L3 scope that are ABSENT from the table
    _lp1 = _f[(_f.nl > 0) & (_f.fs >= 0.1)][key].drop_duplicates().merge(_eco, on=key); _lp1 = _lp1[_lp1.l3.isin(L3S)][key]
    _ab1 = _lp1.merge(V[key].drop_duplicates().assign(_t=1), on=key, how='left'); _ab1 = _ab1[_ab1._t.isna()]
    check('T1m FIADB plots with live trees (region, curated L3 scope) absent from the table entirely <= 2 %', len(_lp1) > 0 and len(_ab1) / len(_lp1) <= 0.02,
          f'{len(_ab1)} of {len(_lp1)} plots absent ({len(_ab1) / max(len(_lp1), 1):.3f}); by state {_ab1.st.value_counts().to_dict()}')
    # T1n (G41, refined G46.3 / G48.2) visits.plot_converted = the plot's LAST SAMPLED FIADB visit (a condition of status 1-4) dated <= 2022 (the mask year) is non-forest — no forested
    # condition and >= 1 condition of non-forest land or water (COND_STATUS_CD 2 / 3 / 4) — and an earlier visit was forested (independent)
    _cv = _cc.assign(f1=(_cc.cs == 1).astype(int), f2=_cc.cs.isin([2, 3, 4]).astype(int)).groupby(key + ['ds']).agg(f1=('f1', 'max'), f2=('f2', 'max')).reset_index().sort_values(key + ['ds'])
    _cv = _cv[(_cv.ds <= '2022-12-31') & ((_cv.f1 == 1) | (_cv.f2 == 1))]; _cv['seen'] = _cv.groupby(key).f1.transform(lambda x: x.cummax().shift(1).fillna(0))      # a forested visit EARLIER
    _cvl = _cv.groupby(key).tail(1); CONVP = set(map(tuple, _cvl[(_cvl.f1 == 0) & (_cvl.f2 == 1) & (_cvl.seen > 0)][key].astype(int).values))   # converted plot keys
    if 'plot_converted' in V.columns:
        _cvp = CONVP
        _cve = np.array([tuple(int(x) for x in r_) in _cvp for r_ in V[key].values]); _cvd = int((_cve != V.plot_converted.values.astype(bool)).sum())
        check('T1n visits.plot_converted (g02e, G41 / G46.3) = independent FIADB recomputation (last sampled visit up to 2022 non-forest after a forested one, G48.2)', _cvd == 0,
              f'{V[V.plot_converted.astype(bool)].pid.nunique()} converted plots in the table, expected {len(set(V[_cve].pid))}; mismatching visits {_cvd}')
    else: check('T1n visits.plot_converted present (g02e, G41)', False, 'input missing: visits.parquet has no plot_converted column (G46.3: g03b refuses it); rerun g02e')
    Vs = V.sort_values('vid')
    check('T1c visit ids reproduce (sorted plot, date order)', len(EX) == len(V) and (EX[key].values == Vs[key].values).all() and (np.abs(EX.year.values - Vs.year.values) < 1e-9).all())
    # the independent visits carry the TABLE's visit ids from here on (matched by plot + date): a T1a / T1c failure must not cascade into every later check (r13 E-F7)
    _vd = V.assign(ds=pd.to_datetime(V.md).dt.strftime('%Y-%m-%d'))[key + ['ds', 'vid']]; EX = EX.drop(columns='vid').merge(_vd, on=key + ['ds'], how='inner')
    if len(EX) != len(V): print(f'   (T1a mismatch: {len(EX)} visits matched by plot + date; checks below compare on those)')
    T = T.merge(EX[key + ['ds', 'vid', 'fs', 'fm', 'fsw', 'fmw']], on=key + ['ds'])
    _ns0 = len(samp); samp = samp[samp.vid.isin(set(EX.vid))]                      # (sampled table visits the independent visit set also has: no T1a cascade, E-F7)
    if len(samp) < _ns0: print(f'   ({_ns0 - len(samp)} sampled table visits are not independent visits (T1a): left out of T1d-T1g / T4a-c / T4r)')
    Ts = T[T.vid.isin(samp.vid)]
    live = Ts[(Ts.status == 1) & (Ts.dia > 0) & (Ts.tpa > 0) & (Ts.age > 0) & (Ts.ht > 0)].copy()
    _KS = set(k for k in KEYS if not k.startswith('_'))          # the TABLE's keys (T1b checks them against FIADB; a T1b failure must not crash the rest: r13 E-F7)
    live['key'] = [KEYS.index(s) if s in _KS else KEYS.index('_' + (g if g in ('S', 'H') else 'H')) for s, g in zip(live.sym, live.grp)]
    live['share'] = np.where(live.dia < 5, live.fmw, live.fsw)
    live['ac'] = np.clip((live.age // 10).astype(int), 0, 15); live['w'] = live.tpa * 2.47105 / live.share; live['D'] = live.dia * 2.54; live['H'] = live.ht * 0.3048
    print(f'   (sampled visits with forested share < 0.95: {(live.groupby("vid").fs.first() < 0.95).mean():.3f})')
    def _acr(x):            # (r13 coverage: token A = density-weighted mean tree age; CR = density-weighted mean crown ratio over trees with a recorded CR > 0, else NaN)
        ok_ = x.cr.notna() & (x.cr > 0); wc_ = x.w[ok_]
        return pd.Series({'N': x.w.sum(), 'QMD': math.sqrt((x.w * x.D ** 2).sum() / x.w.sum()), 'H': (x.w * x.H).sum() / x.w.sum(), 'A': (x.w * x.age).sum() / x.w.sum(),
                          'CR': float((wc_ * x.cr[ok_] / 100).sum() / wc_.sum()) if ok_.any() else np.nan})
    agg = live.groupby(['vid', 'key', 'ac']).apply(_acr, include_groups=False).reset_index()
    tk = TK[TK.vid.isin(samp.vid)].merge(agg, on=['vid', 'key', 'ac'], how='outer', suffixes=('', '_ind'), indicator=True)
    check('T1d token set identical to independent aggregation (vid, key, age class)', (tk._merge == 'both').all(), f"{tk._merge.value_counts().to_dict()}")
    b = tk[tk._merge == 'both']
    check('T1e token N / QMD / H equal (rtol 1e-6)', np.allclose(b.N, b.N_ind, rtol=1e-6) and np.allclose(b.QMD, b.QMD_ind, rtol=1e-6) and np.allclose(b.H, b.H_ind, rtol=1e-6),
          f'max rel N {np.max(np.abs(b.N / b.N_ind - 1)):.2e} QMD {np.max(np.abs(b.QMD / b.QMD_ind - 1)):.2e}')
    _crn = b.CR.isna().values == b.CR_ind.isna().values; _crm = b.CR.notna().values & b.CR_ind.notna().values
    _eA = float(np.max(np.abs(b.A.values - b.A_ind.values))) if len(b) else np.nan; _eC = float(np.max(np.abs(b.CR.values[_crm] - b.CR_ind.values[_crm]))) if _crm.any() else 0.0
    check(f'T1o token A (mean tree age) / CR (mean crown ratio of trees with a CR) = independent aggregation ({samp.vid.nunique()} random visits)', len(b) > 0 and _eA < 1e-6 and _crn.all() and _eC < 1e-6,
          f'{len(b)} tokens; max |ΔA| {_eA:.1e} yr, CR missing-pattern agrees {bool(_crn.all())}, max |ΔCR| {_eC:.1e}')
    emp = set(samp.vid) - set(live.vid); tkv = set(TK[TK.vid.isin(samp.vid)].vid)
    check('T1f visits without qualifying live trees have no tokens (and are kept)', not (emp & tkv) and emp <= set(V.vid), f'empty in sample {len(emp)}')
    # T1k (audit r7 M2: T1f is circular): every table visit WITHOUT cohort tokens has no live FIADB tree (STATUSCD 1, TPA_UNADJ > 0, SUBP <= 4) on a forested
    # condition — else the curated table lost live trees and the "bare" visit is a false clear-cut target
    _lv = _fx.execute(f"""select p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, p.MEASYEAR * 10000 + p.MEASMON * 100 + p.MEASDAY vk, count(*) n
                          from TREE t join PLOT p on p.CN = t.PLT_CN join COND c on c.PLT_CN = t.PLT_CN and c.CONDID = t.CONDID
                          where p.STATECD in {ST_SQL} and p.MEASYEAR >= 2000 and t.STATUSCD = 1 and t.TPA_UNADJ > 0 and t.SUBP <= 4 and c.COND_STATUS_CD = 1 group by all""").df()
    LIVE = {(int(a), int(b), int(c_), int(d), int(k)): int(n) for a, b, c_, d, k, n in zip(_lv.st, _lv.u, _lv.c, _lv.p, _lv.vk, _lv.n) if pd.notna(k)}
    _bare = V[~V.vid.isin(set(TK[TK.N > 0].vid.astype(int)))]; _md = pd.to_datetime(_bare.md)
    _nl = np.array([LIVE.get((int(a), int(b), int(c_), int(d), int(m.year * 10000 + m.month * 100 + m.day)), 0) for a, b, c_, d, m in zip(_bare.st, _bare.u, _bare.c, _bare.p, _md)])
    check('T1k bare table visits (no cohort tokens) have no live FIADB trees on their forested conditions (<= 1 %)', len(_nl) == 0 or (_nl > 0).mean() <= 0.01,
          f'{len(_nl)} bare visits, with FIADB live trees {int((_nl > 0).sum())} ({(_nl > 0).mean() if len(_nl) else 0:.3f}), median live trees there {int(np.median(_nl[_nl > 0])) if (_nl > 0).any() else 0}')
    # T1r (G46.1, A14-1) no TABLE visit is incomplete: FIADB live tallied trees on forested conditions (LIVE, T1k) >= 3 and the curated table's (NCUR)
    # fewer than half of them; meta.json incomplete_visits shown for information
    _nc1 = {(int(a), int(b), int(c_), int(d), int(str(ds_).replace('-', ''))): int(n) for a, b, c_, d, ds_, n in zip(NCUR.st, NCUR.u, NCUR.c, NCUR.p, NCUR.ds, NCUR.n_cur)}
    _vk1 = [(int(a), int(b), int(c_), int(d), int(m.year * 10000 + m.month * 100 + m.day)) for a, b, c_, d, m in zip(V.st, V.u, V.c, V.p, pd.to_datetime(V.md))]
    _bad1 = [k for k in _vk1 if LIVE.get(k, 0) >= 3 and _nc1.get(k, 0) < 0.5 * LIVE.get(k, 0)]
    check('T1r no table visit is incomplete (FIADB live tallied trees >= 3, curated ones < half of them; G46.1)', not _bad1,
          f'{len(V)} table visits, incomplete {len(_bad1)} (curated without a live tree {sum(1 for k in _bad1 if _nc1.get(k, 0) == 0)}) e.g. {_bad1[:3]}; independent incomplete '
          f'curated visits (share >= 0.1) {int((INC.fs >= 0.1).sum())}, meta.json incomplete_visits {M.get("incomplete_visits", "absent")}')
    # T1l (audit r9 I-1: the T1 / T4 token checks recompute from curated_trees, i.e. the table against its own source): STOCKED visits' token TPH and BA
    # = FIADB live trees (STATUSCD 1, TPA_UNADJ > 0, SUBP <= 4) on forested conditions per forested area (shares from FIADB COND: SUBPPROP_UNADJ for trees
    # >= 5 in, MICRPROP_UNADJ for saplings, of COND_STATUS_CD 1) — catches live trees lost by the curation on visits that still have some
    _ft = _fx.execute(f"""with sh as (select PLT_CN, sum(case when COND_STATUS_CD = 1 then SUBPPROP_UNADJ else 0 end) fs, sum(case when COND_STATUS_CD = 1 then MICRPROP_UNADJ else 0 end) fm
                                     from COND group by PLT_CN)
                          select p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, p.MEASYEAR * 10000 + p.MEASMON * 100 + p.MEASDAY vk,
                                 sum(t.TPA_UNADJ * 2.47105 / case when t.DIA < 5 then nullif(sh.fm, 0) else nullif(sh.fs, 0) end) tph,
                                 sum(t.TPA_UNADJ * 2.47105 * pi() / 4 * power(t.DIA * 0.0254, 2) / case when t.DIA < 5 then nullif(sh.fm, 0) else nullif(sh.fs, 0) end) ba
                          from TREE t join PLOT p on p.CN = t.PLT_CN join COND c on c.PLT_CN = t.PLT_CN and c.CONDID = t.CONDID join sh on sh.PLT_CN = p.CN
                          where p.STATECD in {ST_SQL} and p.MEASYEAR >= 2000 and t.STATUSCD = 1 and t.TPA_UNADJ > 0 and t.SUBP <= 4 and c.COND_STATUS_CD = 1 group by all""").df()
    FT = {(int(a), int(b), int(c_), int(d), int(k)): (x, y) for a, b, c_, d, k, x, y in zip(_ft.st, _ft.u, _ft.c, _ft.p, _ft.vk, _ft.tph, _ft.ba) if pd.notna(k)}
    _tk = TK[TK.N > 0].assign(ba=lambda x: x.N * math.pi / 4 * (x.QMD / 100) ** 2).groupby('vid').agg(tph=('N', 'sum'), ba=('ba', 'sum'))
    _sv = V[V.vid.isin(_tk.index)].sample(min(1500, len(_tk)), random_state=12); _md = pd.to_datetime(_sv.md); rt, rb = [], []
    for (a, b, c_, d, m, vid) in zip(_sv.st, _sv.u, _sv.c, _sv.p, _md, _sv.vid):
        f_ = FT.get((int(a), int(b), int(c_), int(d), int(m.year * 10000 + m.month * 100 + m.day)))
        if f_ and f_[0] and f_[0] > 0: rt.append(_tk.tph[vid] / f_[0]); rb.append(_tk.ba[vid] / f_[1])
    rt, rb = np.array(rt), np.array(rb)
    check('T1l stocked visits: token TPH / BA = FIADB live trees per forested area (median ratio within 1 ± 0.02, < 2 % of visits below 0.9)',
          len(rt) > 100 and abs(np.median(rt) - 1) <= 0.02 and abs(np.median(rb) - 1) <= 0.02 and (rt < 0.9).mean() < 0.02 and (rb < 0.9).mean() < 0.02,
          f'{len(rt)} visits; TPH ratio q10 / q50 {np.quantile(rt, 0.1):.3f} / {np.median(rt):.3f}, < 0.9: {(rt < 0.9).mean():.3f}; BA q10 / q50 {np.quantile(rb, 0.1):.3f} / {np.median(rb):.3f}, < 0.9: {(rb < 0.9).mean():.3f}')
    ba_ind = live.groupby('vid').apply(lambda x: (x.tpa / x.share * 0.005454154 * x.dia ** 2).sum() * 0.229568, include_groups=False)
    tt = TK[TK.vid.isin(ba_ind.index)]; ba_tok = tt.groupby('vid').apply(lambda x: (x.N * math.pi / 4 * (x.QMD / 100) ** 2).sum(), include_groups=False)
    check('T1g plot BA per forested area (FIA formula ft2/ac -> m2/ha) = BA from tokens', np.allclose(ba_ind.sort_index(), ba_tok.reindex(ba_ind.index).values, rtol=1e-6), f'n {len(ba_ind)}')
    pv = V.sort_values(['pid', 'year']); exp_pairs = set(zip(pv.vid.values[:-1], pv.vid.values[1:])) - {(a, b_) for a, b_, p1, p2 in zip(pv.vid.values[:-1], pv.vid.values[1:], pv.pid.values[:-1], pv.pid.values[1:]) if p1 != p2}
    check('T1h pairs = consecutive visits of each plot', set(zip(PR.v1, PR.v2)) == exp_pairs, f'{len(PR)} pairs')
    pl_ind = EX.set_index('vid').org.fillna(0) == 1
    check('T1i planted flag = COND STDORGCD 1 (forested conditions) of the visit', (V.set_index('vid').planted.reindex(pl_ind.index) == pl_ind).all(), f'planted share {pl_ind.mean():.3f}')
    fv = duckdb.connect(fiadb(), read_only=True).execute(f"""select distinct p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, p.MEASYEAR my
          from PLOT p join COND c on c.PLT_CN = p.CN where p.STATECD in {ST_SQL} and p.MEASYEAR >= 2000 and c.COND_STATUS_CD = 1""").df()
    ours = V[key].assign(my=V.year.astype(int)).drop_duplicates(); inplots = fv.merge(V[key].drop_duplicates(), on=key)
    miss = inplots.merge(ours, on=key + ['my'], how='left', indicator=True); nm = (miss._merge == 'left_only').sum()
    nplots_f = len(fv[key].drop_duplicates()); nplots_o = len(V[key].drop_duplicates())
    check('T1j FIADB forested visits of our plots that are absent from the table (bare visits with no tree record?) < 2 %', nm / len(inplots) < 0.02,
          f'missing {nm} of {len(inplots)}; forested FIADB plots {nplots_f}, in table {nplots_o} (curated product subset)')
    # ======================= T2 plot coordinates
    fc = duckdb.connect(fiadb(), read_only=True)
    pl = fc.execute(f"select STATECD st, UNITCD u, COUNTYCD c, PLOT p, MEASYEAR my, LAT lat, LON lon from PLOT where STATECD in {ST_SQL} and MEASYEAR >= 2000").df()
    _sp = pl.groupby(key).agg(la0=('lat', 'min'), la1=('lat', 'max'), lo0=('lon', 'min'), lo1=('lon', 'max')).reset_index()
    check('T2a0 a plot key\'s annual-inventory rows (MEASYEAR >= 2000) agree on the coordinates (< 0.2 km)', ((_sp.la1 - _sp.la0) * 111 < 0.2).mean() > 0.999 and ((_sp.lo1 - _sp.lo0) * 92 < 0.2).mean() > 0.999,
          f'{len(_sp)} keys; spread > 0.2 km: {int((((_sp.la1 - _sp.la0) * 111 >= 0.2) | ((_sp.lo1 - _sp.lo0) * 92 >= 0.2)).sum())}')
    pl = pl.groupby(key).agg(lat=('lat', 'mean'), lon=('lon', 'mean')).reset_index()
    m = P.merge(pl, on=key, suffixes=('', '_f'))
    check('T2a plot lat/lon = FIADB PLOT mean of the annual-inventory rows', np.allclose(m.lat, m.lat_f) and np.allclose(m.lon, m.lon_f), f'n {len(m)}')
    land = rasterio.open(LAND); tf = Transformer.from_crs(4326, land.crs, always_xy=True); x, y = tf.transform(m.lon.values, m.lat.values)
    check('T2b projected x/y (EPSG:5070) recomputed', np.allclose(x, m.x, atol=1e-3) and np.allclose(y, m.y, atol=1e-3))
    if AEF90:          # point in the state's polygon (TIGER counties dissolved), 2 km tolerance (FIA coordinate fuzz)
        import geopandas as gpd
        stp = gpd.read_file(f'{WS}/TIGER/tl_2024_us_county/tl_2024_us_county.shp'); stp = stp[stp.STATEFP.astype(int).isin(REGION)].to_crs(land.crs).dissolve('STATEFP').geometry
        pts = gpd.GeoSeries(gpd.points_from_xy(x, y), crs=land.crs); okst = np.mean([stp[f'{st_:02d}'].distance(pt) <= 2000 for st_, pt in zip(m.st.astype(int), pts)])
        check('T2c plots lie in their state (state polygon, 2 km tolerance)', okst > 0.995, f'{okst:.4f}')
    else:
        okst = ((m.st == 12) & (m.lat < 31.1) | (m.st == 13) & (m.lat > 30.3)).mean()
        check('T2c plots lie in their state (latitude band check)', okst > 0.99, f'{okst:.4f}')
    # ======================= T3 candidates: direct raster reads at stored coordinates
    pids = rng.choice(C.pid.unique(), min(60, C.pid.nunique()), replace=False); Cs = C[C.pid.isin(pids)]
    Pi = P.set_index('pid')
    rows_ok = []; dist_err = []
    for pid, g in Cs.groupby('pid'):
        r0, c0 = land.index(Pi.loc[pid].x, Pi.loc[pid].y); dist_err.append(np.max(np.abs(np.hypot(g.row - r0, g.col - c0) * 30 - g.dist_m)))
    check('T3a stored dist_m = distance(plot pixel, candidate pixel); all <= 1600 m', max(dist_err) < 1e-3 and (Cs.dist_m <= 1600).all(), f'max err {max(dist_err):.2e}')
    lc = np.array([land.read(1, window=Window(c, r, 1, 1))[0, 0] for r, c in zip(Cs.row, Cs.col)])
    check('T3b every candidate centre is forest land (classes 1-4)', ((lc >= 1) & (lc <= 4)).all(), f'classes {np.unique(lc)}')
    sub = Cs.sample(min(400, len(Cs)), random_state=3); X = np.asarray(CF[sub.cid.values], dtype=np.float32)
    a22 = rasterio.open(AEFP[2022]); a25 = rasterio.open(AEFP[2025])
    def dq(a): v = a.astype(np.float32); o = np.sign(v) * (v / 127.5) ** 2; o[a == -128] = np.nan; return o
    def aef_fp(src, r, c):             # the footprint's AlphaEarth: 3x3 mean of the 30 m mosaic, or (AEF90) the 90 m pixel of its block
        if AEF90: return np.nan_to_num(dq(src.read(window=Window(c // 3, r // 3, 1, 1))).reshape(64))
        return np.nan_to_num(np.nanmean(dq(src.read(window=Window(c - 1, r - 1, 3, 3))).reshape(64, -1), 1))
    if AEF90: check('T3c0 footprint centres are 90 m block centres (row, col = 1 mod 3)', ((C.row % 3 == 1) & (C.col % 3 == 1)).all())
    e22 = np.stack([aef_fp(a22, r, c) for r, c in zip(sub.row, sub.col)]); e25 = np.stack([aef_fp(a25, r, c) for r, c in zip(sub.row, sub.col)])
    err = max(np.abs(e22 - X[:, :64]).max(), np.abs(e25 - X[:, 64:128]).max())
    check('T3c AlphaEarth 2022 / 2025 = 3x3 mean of the mosaics at the stored centre' + (' (AEF90: the 90 m pixel)' if AEF90 else ''), err < 2e-3, f'max abs err {err:.2e} (float16 storage)')
    if AEF90:          # the 90 m route against the GA+FL 30 m mosaics (independent: 3x3 means of 02_aef.py's 30 m product), candidates inside that extent
        g30 = {y: rasterio.open(f'{ROOT}/data/aef/aef_{y}.tif') for y in (2022, 2025)}; tl = land.transform
        ci = C[C.st.isin([12, 13])] if 'st' in C.columns else C.merge(P[['pid', 'st']], on='pid').query('st in (12, 13)')
        ci = ci.sample(min(300, len(ci)), random_state=4); res3c = {}
        for y3, sl3 in ((2022, slice(0, 64)), (2025, slice(64, 128))):          # (r14 G47.8: 2022 too)
            e3, e9 = [], []
            for r, c, cid_ in zip(ci.row, ci.col, ci.cid):
                xx, yy = tl * (c + 0.5, r + 0.5); r3, c3 = g30[y3].index(xx, yy)
                if not (1 <= r3 < g30[y3].height - 1 and 1 <= c3 < g30[y3].width - 1): continue
                m30 = np.nanmean(dq(g30[y3].read(window=Window(c3 - 1, r3 - 1, 3, 3))).reshape(64, -1), 1)
                if np.isnan(m30).any(): continue
                e3.append(m30); e9.append(np.asarray(CF[cid_, sl3], dtype=np.float32))
            e3, e9 = np.array(e3).reshape(-1, 64), np.array(e9).reshape(-1, 64); cs = (e3 * e9).sum(1) / (np.linalg.norm(e3, axis=1) * np.linalg.norm(e9, axis=1) + 1e-9)
            res3c[y3] = (len(e3), float(np.median(np.abs(e3 - e9).max(1))) if len(e3) else np.nan, float(np.quantile(cs, 0.05)) if len(e3) else np.nan)
        check('T3c2 AEF90 footprint (2022 and 2025) = 3x3 mean of the GA+FL 30 m mosaics (re-quantisation tolerance)', all(n_ > 50 and md_ < 0.02 and q_ > 0.98 for n_, md_, q_ in res3c.values()),
              f'{{year: (footprints, median max|Δ|, cosine q05)}} {res3c}')
        # T3c2z (r13 E-F9) every UTM zone of Region 4 (15-18; the GA+FL mosaic above covers 16-17 only): the 90 m footprint against an independent route from
        # the SOURCE AlphaEarth COGs (20 m level-0 overview, remote): each 30 m pixel = mean of 3x3 sub-points at 10 m spacing looked up in the source tile
        # (bottom-up rows), unit-normalised and int8-requantised as 02_aef, then the block mean. A 1-block shift drops the cosine to 0.94-0.99 (measured), so
        # every footprint must have cosine > 0.999 (correct: >= 0.9998; shifted: 0.77-0.996) and the median max|Δ| < 0.02; >= 4 footprints per zone, footprints covered by exactly one source tile
        IDX3 = pd.read_parquet(f'{ROOT}/data/aef/aef_index.parquet', columns=['crs', 'path', 'year', 'utm_west', 'utm_south', 'wgs84_west', 'wgs84_south', 'wgs84_east', 'wgs84_north'])
        IDX3 = IDX3[IDX3.year == 2025]; tl4 = land.transform; _ll3 = Transformer.from_crs(5070, 4326, always_xy=True)
        ENV3 = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tiff', GDAL_HTTP_MAX_RETRY=5, GDAL_HTTP_RETRY_DELAY=3)
        cz = C.sample(min(20000, len(C)), random_state=35); xz, yz = tl4 * (cz.col.values + 0.5, cz.row.values + 0.5); loz, laz = _ll3.transform(xz, yz)
        cz = cz.assign(lon=loz, lat=laz, zone=np.floor((np.asarray(loz) + 180) / 6).astype(int) + 1); res3z = {}
        def _tile3(lo, la):
            h_ = IDX3[(IDX3.wgs84_west <= lo) & (IDX3.wgs84_east >= lo) & (IDX3.wgs84_south <= la) & (IDX3.wgs84_north >= la)]; return h_.iloc[0] if len(h_) == 1 else None
        for z3 in (15, 16, 17, 18):
            g3 = cz[cz.zone == z3]; cs3, md3 = [], []
            if not len(g3): res3z[z3] = (0, None, None); continue
            try:
                t3 = None
                for lo, la in zip(g3.lon.values, g3.lat.values):          # a random footprint's single source tile, then up to 6 footprints of that tile
                    t3 = _tile3(lo, la)
                    if t3 is not None: break
                fz = [i for i in range(len(g3)) if (lambda t_: t_ is not None and t_.path == t3.path)(_tile3(g3.lon.values[i], g3.lat.values[i]))][:6] if t3 is not None else []
                T3 = Transformer.from_crs(5070, t3.crs, always_xy=True) if t3 is not None else None
                with rasterio.Env(**ENV3), rasterio.open('/vsicurl/' + t3.path.replace('s3://us-west-2.opendata.source.coop', 'https://data.source.coop'), overview_level=0) as src3:
                    for i in fz:
                        r_, c_ = int(g3.row.values[i]), int(g3.col.values[i])
                        sp = np.array([(rr + fr, cc + fc) for rr in (r_ - 1, r_, r_ + 1) for cc in (c_ - 1, c_, c_ + 1) for fr in (1 / 6, .5, 5 / 6) for fc in (1 / 6, .5, 5 / 6)])
                        X_, Y_ = tl4 * (sp[:, 1], sp[:, 0]); U_, W_ = T3.transform(X_, Y_); ci_ = np.floor((U_ - t3.utm_west) / 20).astype(int); ri_ = np.floor((W_ - t3.utm_south) / 20).astype(int)
                        if ci_.min() < 0 or ri_.min() < 0 or ci_.max() >= src3.width or ri_.max() >= src3.height: continue
                        A_ = src3.read(window=Window(ci_.min(), ri_.min(), ci_.max() - ci_.min() + 1, ri_.max() - ri_.min() + 1))
                        v_ = dq(A_)[:, ri_ - ri_.min(), ci_ - ci_.min()].reshape(64, 9, 9); p30 = np.nanmean(v_, 2); n_ = np.sqrt(np.nansum(p30 ** 2, 0)); p30 = p30 / np.where(n_ > 0, n_, 1)
                        q_ = np.sign(p30) * np.rint(127.5 * np.sqrt(np.abs(p30))); e_ = np.nanmean(np.sign(q_) * (q_ / 127.5) ** 2, 1)
                        if np.isnan(e_).any(): continue
                        e9_ = np.asarray(CF[g3.cid.values[i], 64:128], dtype=np.float32); cs3.append(float((e_ * e9_).sum() / (np.linalg.norm(e_) * np.linalg.norm(e9_) + 1e-9))); md3.append(float(np.abs(e_ - e9_).max()))
                res3z[z3] = (len(cs3), min(cs3) if cs3 else None, float(np.median(md3)) if md3 else None)
            except Exception as e_:
                res3z[z3] = (0, None, f'{type(e_).__name__}: {str(e_)[:80]}')
        _zin = sorted(z for z in res3z if (cz.zone == z).any())
        check('T3c2z AEF90 footprints in EVERY UTM zone of the region = an independent route from the source AlphaEarth COGs (remote, 2025)',
              len(_zin) > 0 and all(res3z[z][0] >= 4 and res3z[z][1] > 0.999 and res3z[z][2] < 0.02 for z in _zin),
              f'{{zone: (footprints, min cosine, median max|Δ|)}} {res3z}; zones with candidates {_zin}')
    SOILS = M['soils']; FILL = json.load(open(f'{ROOT}/data/r4/soils90/fill.json')) if AEF90 else {'clay030': 7.58, 'sand030': 83.2, 'soc030': 0.964, 'awc030': 0.052, 'ph030': 5.18, 'restrdepth': 200.0}
    SDIV = 3 if AEF90 else 1               # (AEF90: soils on the 90 m grid)
    so = np.array([[(lambda v: FILL[k] if v <= -9000 else v)(float(rasterio.open(SOILP(k)).read(1, window=Window(c // SDIV, r // SDIV, 1, 1))[0, 0])) for k in SOILS]
                   for r, c in zip(sub.row[:120], sub.col[:120])])
    Xs = X[:120, -len(SOILS):]; se = np.abs(so - Xs) / (np.abs(so) * 1e-3 + 1e-3)
    check('T3d soils = centre pixel, nodata -> regional median (float16 storage)', se.max() < 1 and (np.asarray(CF[::97, -len(SOILS):]) > -9000).all(), f'max err / tol {se.max():.2f}; no -9999 left')
    # --- LANDFIRE: independent decoding of the code tables (type; burned 'Unknown' -> fire; 'Increased Green' is no disturbance; event year = CALENDAR_YR
    #     of the fiscal-year products, else the file year), evidence = fraction of the footprint's 3x3 pixels
    CLS = M['classes']; lk, lks = landfire_codes(f'{ROOT}/data/landfire/dist/dist_codes.csv', CLS)
    print(f"   (fiscal-year codes placed in the previous calendar year: {sum(1 for (fy, _), (c_, ey) in lk.items() if c_ is not None and ey != fy)})")
    hl = rasterio.open(HANSP); LF = {yr: rasterio.open(DISTP(yr)) for yr in YEARS}; bad = 0; nev = 0; nh = 0; maxd = 0
    def fp_dist(r, c):
        """the spec of a footprint's disturbance features (shared by T3e candidates and T6c stand footprints): its 3x3 pixels (AEF90: the 90 m block holding
        (r, c); for a block centre = the 3x3 window around it), a pixel counted once per class and event calendar year (whichever product codes it), severity =
        max label, Hansen loss fraction per year"""
        r0_, c0_ = (3 * (r // 3), 3 * (c // 3)) if AEF90 else (r - 1, c - 1)
        hw = hl.read(1, window=Window(c0_, r0_, 3, 3)); exp = np.zeros((NY, NC)); esv = np.zeros(NY); hit = set()
        for yr in YEARS:
            win = LF[yr].read(1, window=Window(c0_, r0_, 3, 3))
            for pix, code in enumerate(win.ravel()):
                if code > 0:
                    c_, ey = lk.get((yr, int(code)), (None, yr))
                    if c_ is not None and YEARS[0] <= ey <= YEARS[-1]: hit.add((ey, c_, pix)); esv[ey - YEARS[0]] = max(esv[ey - YEARS[0]], lks[(yr, int(code))])
        for ey, c_, pix in hit: exp[ey - YEARS[0], c_] += 1 / 9
        he = np.array([((hw == yr - 2000) & (hw > 0)).sum() / 9 for yr in YEARS]); return exp, esv, he
    def dist_err(xrow, r, c):
        per = xrow[128:128 + NY * (NC + 2)].reshape(NY, NC + 2); exp, esv, he = fp_dist(r, c)
        return max(np.abs(per[:, :NC] - exp).max(), np.abs(per[:, NC + 1] - he).max(), np.abs(per[:, NC] - esv).max() / 3), int((exp > 0).sum()), int((he > 0).sum())
    for (r, c), xrow in zip(zip(sub.row[:150], sub.col[:150]), X[:150]):
        dd_, a_, b_ = dist_err(xrow, r, c); maxd = max(maxd, dd_); bad += int(dd_ > 2e-3); nev += a_; nh += b_
    check('T3e disturbance fractions (3x3) and severity (max) at the event calendar year (LANDFIRE) and Hansen fractions = direct raster lookups', bad == 0,
          f'150 candidates x {NY} years, class-years with evidence {nev}, Hansen-years {nh}, max |Δ| {maxd:.1e}, candidates off {bad}')
    # --- prior: re-implemented from the SPECIFICATION (not the code): FIADB COND events by type, records of one physical event (years chaining within 2 yr)
    #     merged into one cluster (union of evidence and windows, max likelihood ratio, ×2 for a cut matched by a harvest class); negative periods per FIA
    #     protocol (since the previous measurement, PREV_PLT_CN, or the lookback yr0-4..yr0-1 of a new plot / after a non-sampled one: G39; up to EVID_END 2024: G36) for visits without a code that could hide stand-replacing
    #     evidence; stand-replacing = harvest class or Hansen loss on >= half the footprint -> ×0.25. Stored prior must be ∝ w x incl_w.
    ix = {k: i for i, k in enumerate(CLS)}; HH = NC + 1; EVID_END = 2024     # (G36) the last year with stand-replacing evidence coverage
    # SPEC: the region's evidence table (evidence_spec, G47.2; module level, written out independently of g02b's EVK_TABLES)
    HARMLESS = {11, 21, 31, 40, 42, 43, 44, 45, 46}   # (G37: 41 beaver / flooding and 60 vegetation block) FIADB v9.4 DSTRBCD: 11 / 21 insect / disease damage to UNDERSTORY vegetation (12 / 22 = to trees),
    # 31 ground fire, 40 and 42-46 animal — the EVALIDATOR variable library's labels (audit r6 I1: the old set had 12 / 22); 41 beaver / flooding and 60 vegetation are NOT harmless (G37)
    evr = _fx.execute(f"""select p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, p.MEASYEAR my, p.MEASMON mm, p.MEASDAY dd, c.COND_STATUS_CD cs, q.MEASYEAR pmy, q.PLOT_STATUS_CD pst,
                        c.TRTCD1, c.TRTYR1, c.TRTCD2, c.TRTYR2, c.TRTCD3, c.TRTYR3, c.DSTRBCD1, c.DSTRBYR1, c.DSTRBCD2, c.DSTRBYR2, c.DSTRBCD3, c.DSTRBYR3
                        from COND c join PLOT p on c.PLT_CN = p.CN left join PLOT q on q.CN = p.PREV_PLT_CN where p.STATECD in {ST_SQL} and p.MEASYEAR >= 2000""").df()
    evr = evr[evr.cs == 1]; evr['vk'] = evr.my * 10000 + evr.mm * 100 + evr.dd
    # FIA's previous measurement of every visit (PREV_PLT_CN, any plot status): the period a visit's codes describe starts there
    # (+ G46.2) its plot status and its forested share of the whole plot (COND 1 SUBPPROP_UNADJ; unknown when none of its conditions carries one)
    _pm = _fx.execute(f"""select p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, p.MEASYEAR * 10000 + p.MEASMON * 100 + p.MEASDAY vk, q.MEASYEAR * 10000 + q.MEASMON * 100 + q.MEASDAY pvk,
                                 q.PLOT_STATUS_CD qps, (select sum(case when c2.COND_STATUS_CD = 1 then c2.SUBPPROP_UNADJ else 0 end) from COND c2 where c2.PLT_CN = q.CN) qf1,
                                 (select count(c2.SUBPPROP_UNADJ) from COND c2 where c2.PLT_CN = q.CN) qn
                          from PLOT p left join PLOT q on q.CN = p.PREV_PLT_CN where p.STATECD in {ST_SQL} and p.MEASYEAR >= 2000""").df()
    PREVK = {(int(a), int(b), int(c), int(d), int(k)): (int(q) if pd.notna(q) else None) for a, b, c, d, k, q in zip(_pm.st, _pm.u, _pm.c, _pm.p, _pm.vk, _pm.pvk)}
    # forest -> non-forest conversion: the forested share of the plot (COND_STATUS_CD 1 SUBPPROP_UNADJ) changed by >= 0.25 since the previous forested visit —
    # no TRT / DSTRB code can record it (codes are on forested conditions), so the visit hides possible clearing. (G46.2) also against FIA's PREVIOUS
    # measurement (PREV_PLT_CN): a pre-2000 one (the first visit since 2000), and an entirely non-forest one (PLOT_STATUS_CD 2) counts as share 0; a
    # non-sampled one (3) or one whose conditions carry no share says nothing
    _fs = _fx.execute(f"""select p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, p.MEASYEAR * 10000 + p.MEASMON * 100 + p.MEASDAY vk,
                                 sum(case when c.COND_STATUS_CD = 1 then c.SUBPPROP_UNADJ else 0 end) f1
                          from PLOT p join COND c on c.PLT_CN = p.CN where p.STATECD in {ST_SQL} and p.MEASYEAR >= 2000 group by all""").df()
    _fs = _fs[_fs.f1 > 0].sort_values(['st', 'u', 'c', 'p', 'vk']); CONV = set()
    for k4, g4 in _fs.groupby(['st', 'u', 'c', 'p']):
        f_, v_ = g4.f1.values, g4.vk.values
        for j in range(1, len(f_)):
            if abs(f_[j - 1] - f_[j]) >= 0.25: CONV.add(tuple(int(x) for x in k4) + (int(v_[j]),))
    _f1v = {tuple(int(x) for x in r_[:5]): float(r_[5]) for r_ in _fs[['st', 'u', 'c', 'p', 'vk', 'f1']].values}; nconv2 = 0
    for a, b, c_, d, k, qps, qf1, qn in zip(_pm.st, _pm.u, _pm.c, _pm.p, _pm.vk, _pm.qps, _pm.qf1, _pm.qn):
        if pd.isna(k) or (int(a), int(b), int(c_), int(d), int(k)) not in _f1v: continue
        q_ = 0.0 if pd.notna(qps) and int(qps) == 2 else (float(qf1) if pd.notna(qps) and int(qps) == 1 and pd.notna(qn) and qn > 0 else None)
        k5 = (int(a), int(b), int(c_), int(d), int(k))
        if q_ is not None and abs(q_ - _f1v[k5]) >= 0.25 and k5 not in CONV: CONV.add(k5); nconv2 += 1
    print(f'   (forest-share blocks: {len(CONV)} visits, {nconv2} of them only against the previous FIA measurement, G46.2)')
    ratios, nneg, ncl = [], 0, 0; REC = {}
    for pid, g in Cs.groupby('pid'):
        k_ = tuple(int(v) for v in Pi.loc[pid][key]); e = evr[(evr.st == k_[0]) & (evr.u == k_[1]) & (evr.c == k_[2]) & (evr.p == k_[3])]
        Xg = np.asarray(CF[g.cid.values], dtype=np.float32)[:, 128:128 + NY * (NC + 2)].reshape(len(g), NY, NC + 2); w = np.ones(len(g))
        recs = set(); hiding = set()
        for r in e.itertuples():
            for t_, a_, b_ in (('T', 'TRTCD1', 'TRTYR1'), ('T', 'TRTCD2', 'TRTYR2'), ('T', 'TRTCD3', 'TRTYR3'), ('D', 'DSTRBCD1', 'DSTRBYR1'), ('D', 'DSTRBCD2', 'DSTRBYR2'), ('D', 'DSTRBCD3', 'DSTRBYR3')):
                cd, yr = getattr(r, a_), getattr(r, b_)
                if pd.isna(cd) or cd <= 0: continue
                if t_ == 'T' or int(cd) not in HARMLESS: hiding.add(r.vk)
                if (t_, int(cd)) in SPEC and pd.notna(yr) and 2000 <= yr <= 2025: recs.add((int(yr), t_, int(cd)))
            if pd.notna(r.vk) and k_ + (int(r.vk),) in CONV: hiding.add(r.vk)
        cl = []                                       # chain records by year (gap <= 2)
        for yr, t_, cd in sorted(recs):
            cols_, (lo, hi), lr = SPEC[(t_, cd)]; cs_ = {HH if k == 'H' else ix[k] for k in cols_}
            if cl and yr - cl[-1][4] <= 2: a = cl[-1]; cl[-1] = (a[0] | cs_, min(a[1], yr + lo), max(a[2], yr + hi), max(a[3], lr), yr, a[5] or (t_ == 'T' and cd == 10))
            else: cl.append((cs_, yr + lo, yr + hi, lr, yr, t_ == 'T' and cd == 10))
        for cs_, lo, hi, lr, _, cut in cl:
            a, b = max(lo - 2000, 0), min(hi - 2000, NY - 1)
            if b < a: continue
            w = w * np.where(Xg[:, a:b + 1][:, :, sorted(cs_)].sum(-1).sum(-1) > 0, lr, 1.0)
            if cut: w = w * np.where(Xg[:, a:b + 1, ix['harvest']].sum(-1) > 0, 2.0, 1.0)
        ncl += len(cl)
        vv = e.groupby('vk').agg(my=('my', 'first'), pmy=('pmy', 'first'), pst=('pst', 'first')).sort_index(); srv = np.maximum(Xg[:, :, ix['harvest']], Xg[:, :, HH]) >= 0.5; per_ = []
        for j in range(len(vv)):
            if vv.index[j] in hiding: continue
            # spec: a visit's codes cover the period since FIA's previous measurement (PREV_PLT_CN), or since the previous forested visit if that is later
            # (G39) a previous measurement that was NOT SAMPLED (PLOT_STATUS_CD 3) does not start the period: the lookback rule applies, as for a new plot
            #       (bounded by the previous forested visit); the first visit's lookback is the calendar years yr0 - 4 .. yr0 - 1
            my_, ns3 = int(vv.my.iloc[j]), pd.notna(vv.pst.iloc[j]) and int(vv.pst.iloc[j]) == 3
            if pd.notna(vv.pmy.iloc[j]) and not ns3: per_.append((max(int(vv.pmy.iloc[j]), int(vv.my.iloc[j - 1]) if j > 0 else -9999) + 1, my_ - 1))
            elif j > 0: per_.append(((max(my_ - 4, int(vv.my.iloc[j - 1]) + 1) if ns3 else int(vv.my.iloc[j - 1]) + 1), my_ - 1))
            else: per_.append((my_ - 4, my_ - 1))
        per_ = [(a, min(b, EVID_END)) for a, b in per_]                                  # (G36) negative evidence only up to the evidence end
        for a, b in per_:
            a, b = max(a - 2000, 0), min(b - 2000, NY - 1)
            if b >= a: w = w * np.where(srv[:, a:b + 1].any(1), 0.25, 1.0); nneg += 1
        # radial location prior (FIA perturbation: RADIAL_P of the mass within 0.5 mile, uniform per zone; written out from the spec, not imported)
        rp_ = float(M.get('radial_p', 0.9)); rad = np.where(g.dist_m.values <= 804.672, rp_ / (np.pi * 804.672 ** 2), (1 - rp_) / (np.pi * (1600.0 ** 2 - 804.672 ** 2)))
        q = g.prior.values / (w * rad * g.incl_w.values); ratios.append(q.max() / q.min()); REC[pid] = (cl, per_)
    check('T3f prior ∝ (clustered typed FIA events x FIA-protocol negative evidence) x radial location prior x incl_w — re-implemented from the spec on FIADB COND', max(ratios) < 1 + 2e-3,
          f'{len(ratios)} plots ({ncl} event clusters, {nneg} negative periods), max ratio spread {max(ratios):.5f}')
    # T3j (r13: the G36 / G37 / G39 rules on g02e's own output) the negative-evidence periods of fia_negs.parquet (g02e) = the spec's periods (T3f's, b >= a)
    if os.path.exists(f'{D}/fia_negs.parquet'):
        _ng = pd.read_parquet(f'{D}/fia_negs.parquet'); _ng = {p_: sorted(zip(g_.y_lo.astype(int), g_.y_hi.astype(int))) for p_, g_ in _ng.groupby('pid')}
        _bj = [pid for pid, (_, pe_) in REC.items() if sorted((a, b) for a, b in pe_ if b >= a) != _ng.get(pid, [])]
        check('T3j negative-evidence periods (g02e fia_negs) = the spec (FIA protocol, lookback yr0-4..yr0-1, non-sampled predecessor -> lookback, <= 2024, NONBLOCK)', not _bj,
              f'{len(REC)} plots, {sum(len(v) for v in _ng.values())} periods in the file; mismatching plots {len(_bj)} {_bj[:5]}')
    # --- semantic alignment of the time axes: AlphaEarth change 2022->2025 must concentrate on candidates whose evidence falls in 2023-2025
    # (r13 E-F11: PER STATE — pooling hid a state whose evidence does not move AlphaEarth; G34). (r14 G47.2 / G48.1, set after seeing data) per state, every
    # DETECTABLE row of the region's table (lr >= 4) is judged on the POOLED cells of its compatible classes (a footprint counts in year y when any of the
    # row's classes covers >= half of it: the row claims its record kind is detectable, not that each class is); every class is also reported on its own
    # (INFO; in GA+FL prescribed 'fire' never lowers the cosine). A year cell counts with >= 30 single-event candidates; a (state, row) is judged when it
    # has >= 1 pre (2017-21) and post (2023-24) cell, else INFO. LANDFIRE classes 2017-2025, Hansen 2017-2024
    _rows3 = {}
    for (t_, cd_), (cols_, _, lr_) in SPEC.items():
        if lr_ >= 4: _rows3[spec_kind(t_, cd_)] = tuple(sorted(cols_))
    _cls3 = sorted({c_ for cols_, _, _ in SPEC.values() for c_ in cols_})
    _st3 = C.pid.map(P.set_index('pid').st).values; _r3 = np.random.default_rng(33); res3i = {}; _w3 = 128 + NY * (NC + 2)
    for st3 in sorted(set(int(x) for x in np.unique(_st3))):
        idx3 = np.flatnonzero(_st3 == st3); smp = np.sort(_r3.choice(idx3, min(120000, len(idx3)), replace=False)); XX = np.asarray(CF[smp, :_w3], dtype=np.float32)
        cosv = (XX[:, :64] * XX[:, 64:128]).sum(1) / (np.linalg.norm(XX[:, :64], axis=1) * np.linalg.norm(XX[:, 64:128], axis=1) + 1e-9)
        pp = XX[:, 128:_w3].reshape(-1, NY, NC + 2); anyd = (pp[:, :, :NC].sum(-1) + pp[:, :, HH]) > 0; single = anyd.sum(1) == 1      # one disturbed year only
        c_clean = float(np.median(cosv[~anyd[:, 2015 - 2000:].any(1)])) if (~anyd[:, 2015 - 2000:].any(1)).any() else np.nan
        cell = lambda m_: (int(m_.sum()), round(float(np.median(cosv[m_])), 3) if m_.sum() >= 30 else None)
        cm3 = {c3: (pp[:, :, HH] >= 0.5 if c3 == 'H' else pp[:, :, ix[c3]] >= 0.5) for c3 in _cls3}
        def judge3(m3, yr_hi):
            tab = {yr: cell(m3[:, yr - 2000] & single) for yr in range(2017, yr_hi) if yr - 2000 < NY}
            pre = [v[1] for y, v in tab.items() if y <= 2021 and v[1] is not None]; post = [v[1] for y, v in tab.items() if y in (2023, 2024) and v[1] is not None]
            judged = bool(pre and post); return judged, judged and max(post) < min(pre) and max(post) < c_clean - 0.05, tab
        for k3, cols3 in sorted(_rows3.items()):
            j_, o_, tab = judge3(np.logical_or.reduce([cm3[c_] for c_ in cols3]), 2026); res3i[(st3, k3)] = (j_, o_)
            print(f"   T3i state {st3} row {k3} {list(cols3)} pooled: {'judged' if j_ else 'too few qualifying samples'} {'ok' if o_ else ('VIOLATED' if j_ else '')}; "
                  f"no disturbance since 2015 {c_clean:.3f}; {{yr: (n, cos)}} {tab}", flush=True)
        for c3 in _cls3:
            j_, o_, tab = judge3(cm3[c3], 2025 if c3 == 'H' else 2026)
            print(f"   T3i state {st3} class {c3} (INFO): {'would pass' if o_ else ('would be violated' if j_ else 'too few qualifying samples')}; {{yr: (n, cos)}} {tab}", flush=True)
        del XX, pp
    check('T3i time axes PER STATE and detectable evidence row of the region\'s table (pooled compatible classes): events of 2023-24 lower the AlphaEarth 2022->2025 cosine, events <= 2021 do not',
          any(j for j, _ in res3i.values()) and all(o for j, o in res3i.values() if j),
          f"rows {_rows3}; judged {[k for k, (j, _) in res3i.items() if j]}, violating {[k for k, (j, o) in res3i.items() if j and not o]}, INFO {[k for k, (j, _) in res3i.items() if not j]}")
    def _ev(g):    # the evidence part of the prior: prior / (incl_w x radial location weight) — hi is a threshold of the evidence alone (audit r3 B3)
        rp_ = float(M.get('radial_p', 0.9)); rad = np.where(g.dist_m.values <= 804.672, rp_ / (np.pi * 804.672 ** 2), (1 - rp_) / (np.pi * (1600.0 ** 2 - 804.672 ** 2)))
        return pd.Series(g.prior.values / (g.incl_w.values * rad), index=g.index)
    _p3h = np.random.default_rng(31).choice(C.pid.unique(), min(400, C.pid.nunique()), replace=False)     # (a larger random plot sample than T3a-g: r13 E-F6)
    hok = [(_ev(g)[g.hi].min() >= _ev(g)[~g.hi].max() * (1 - 1e-4)) for _, g in C[C.pid.isin(_p3h)].groupby('pid') if g.hi.any() and (~g.hi).any()]
    check('T3h hi flag = a threshold of the prior\'s evidence part (prior / (incl_w x radial weight), monotone); >= 10 plots with both classes', len(hok) >= 10 and all(hok),
          f'{len(_p3h)} random plots, {len(hok)} with both classes, violations {len(hok) - sum(hok)}')
    sys.path.insert(0, f'{ROOT}/scripts'); import g02b_build_gafl as G2; G2.init()
    cnt_err = []
    for pid, g in Cs.groupby('pid'):
        r0, c0 = land.index(Pi.loc[pid].x, Pi.loc[pid].y); RP = int(1600 // 30) + 2
        L = land.read(1, window=Window(c0 - RP - 3, r0 - RP - 3, 2 * RP + 7, 2 * RP + 7), boundless=True, fill_value=0)       # (window origin r0 - RP - 3)
        if AEF90: ar_ = np.arange(r0 - RP - 3, r0 + RP + 4); ac_ = np.arange(c0 - RP - 3, c0 + RP + 4); ar_, ac_ = ar_[ar_ % 3 == 1], ac_[ac_ % 3 == 1]   # 90 m block centres
        else: ar_, ac_ = np.arange(r0 - RP + 1, r0 + RP, 3), np.arange(c0 - RP + 1, c0 + RP, 3)                                    # the plot window's own stride-3 lattice
        AR, AC = np.meshgrid(ar_, ac_, indexing='ij'); AR, AC = AR.ravel(), AC.ravel(); Lc = L[AR - (r0 - RP - 3), AC - (c0 - RP - 3)]
        fl = (Lc >= 1) & (Lc <= 4) & (np.hypot(AR - r0, AC - c0) * 30 <= 1600)
        cnt_err.append(abs(g.incl_w.sum() - fl.sum()) / fl.sum())
    check('T3g inclusion weights sum to the full forest-footprint count of the disk', max(cnt_err) < 1e-4, f'max rel err {max(cnt_err):.2e}')
    if done('T3'): return RES
    # ======================= T4 model inputs (g03b data tensors) and unit tests of the model pieces
    os.environ.update(DATA=os.environ.get('DATA', 'gafl'), EPOCHS='0', LOAD='0', NT=os.environ.get('NT', '4')); _a = sys.argv; sys.argv = ['g03b']
    SRC = open(f'{ROOT}/scripts/g03b_train_gafl.py').read(); SRC = SRC[:SRC.index('TEMP = 0.05')]; ns_ = {'__file__': f'{ROOT}/scripts/g03b_train_gafl.py'}
    exec(compile(SRC, 'g03b', 'exec'), ns_); sys.argv = _a; import torch
    tv = samp.vid.values[:200]
    G4 = open(f'{ROOT}/scripts/g04_bayes_downscale.py').read(); exec(compile(G4[G4.index('# decoded summaries'):G4.index('t0 = time.time()')], 'g04', 'exec'), ns_)
    o = ns_['obs_summ'](torch.tensor(tv)); ba_t = pd.Series(o[1].numpy(), index=tv)
    ba_i = ba_ind.reindex(tv).fillna(0.0)
    check('T4a obs_summ BA (model targets) = FIA-formula BA (independent)', np.allclose(ba_t.values, ba_i.values, rtol=1e-4, atol=1e-6), f'max abs diff {np.max(np.abs(ba_t.values - ba_i.values)):.2e}')
    tph_i = live.groupby('vid').w.sum().reindex(tv).fillna(0.0); check('T4b obs_summ TPH = sum of TPA_UNADJ x 2.47105 / forested share (independent)', np.allclose(o[0].numpy(), tph_i.values, rtol=1e-4, atol=1e-6))
    NK, NA = len(KEYS), 16; v_ = torch.tensor(tv[:50]); lp_perfect = torch.where(ns_['slot_p'][v_] > 0, torch.tensor(60.0), torch.tensor(-60.0)) + torch.log(ns_['POSW'])
    s_pred = ns_['summaries'](lp_perfect, ns_['slot_v'][v_]); o2 = ns_['obs_summ'](v_)
    check('T4c decoder summaries on perfect slots (with calibration) = observed summaries', all(np.allclose(s_pred[i].numpy(), o2[i].numpy(), rtol=1e-4, atol=1e-5) for i in range(3)))
    tr_p = set(ns_['vpid'][ns_['tr_v']]); te_p = set(ns_['vpid'][ns_['te_v']])
    from scipy.spatial import cKDTree as _KDv
    _xy = Pi[['x', 'y']] if 'x' in Pi.columns else None
    va_p = set(ns_['vpid'][ns_['va_v']]) if len(ns_['va_v']) else set()
    if _xy is not None:
        _d_te = _KDv(_xy.loc[sorted(te_p)].values).query(_xy.loc[sorted(tr_p)].values)[0].min()
        _d_va = _KDv(_xy.loc[sorted(va_p)].values).query(_xy.loc[sorted(tr_p)].values)[0].min() if va_p else np.inf
    else: _d_te = _d_va = np.nan
    _VS4 = os.environ.get('VAL_SPLIT', '0') == '1'
    check('T4d train / test / validation plots disjoint and every training plot >= 3.2 km from every test and validation plot (no overlapping 1.6 km disks); VAL_SPLIT=1 -> a non-empty validation set',
          not (tr_p & te_p) and not (tr_p & va_p) and not (va_p & te_p) and _d_te >= 3200 and _d_va >= 3200 and (len(va_p) > 0 or not _VS4),
          f'train {len(tr_p)} test {len(te_p)} val {len(va_p)} plots; min train-test distance {_d_te:.0f} m, train-val {_d_va:.0f} m')
    # T4v the forest-share vetting (MIN_FSHARE, r13 coverage): every train / test / validation visit and both visits of every model pair have a forested share of
    # the WHOLE plot (independent, FIADB COND 1 SUBPPROP: T1a's fsw = g02b fsub) >= MIN_FSHARE, and every table visit passing it whose plot has candidates is in exactly one set or a buffer
    MINF = float(os.environ.get('MIN_FSHARE', 0.25)); _fsi = V.fsub.values.astype(float).copy() if 'fsub' in V.columns else np.ones(len(V)); _fsi[EX.vid.values.astype(int)] = EX.fsw.values; _fok = _fsi >= MINF   # (FIADB COND 1 share of the WHOLE plot, as visits.fsub; unmatched visits: the table's)
    _sets4 = np.concatenate([ns_['tr_v'], ns_['te_v'], ns_['va_v']]).astype(int)
    _fsd = float(np.max(np.abs(V.fsub.values - _fsi))) if 'fsub' in V.columns else np.nan
    check(f'T4v forest-share vetting (MIN_FSHARE {MINF}): visits in train / test / val all pass it (independent FIADB share; model pairs: T8p), no passing visit of a plot with candidates is dropped outside the buffers; visits.fsub = FIADB share',
          _fok[_sets4].all() and len(set(_sets4)) == len(_sets4) and (_fsd < 1e-6) and
          not ({v for v in np.flatnonzero(_fok) if ns_['vpid'][v] in ns_['cand_of'] and ns_['vpid'][v] not in (ns_['buffer_pid'] | ns_['vbuf_pid'])} - set(_sets4)),
          f'{len(_sets4)} set visits, failing {int((~_fok[_sets4]).sum())}; table visits failing the share {int((~_fok).sum())}; max |fsub - FIADB share| {_fsd:.1e}')
    _SWK = ('SITE_ATTR', 'AGE_CH', 'NJ', 'JUMP_RESET', 'STATE_ATTR'); _sw0 = {k: ns_.get(k, False) for k in _SWK}
    def sw(site, age, jr=False, sa=False):        # the flow's module switches of the exec'd g03b (Flow reads them at construction / call time)
        ns_.update(SITE_ATTR=site, AGE_CH=age, JUMP_RESET=jr, STATE_ATTR=sa, NJ=2 * ns_['D_Z'] if jr else ns_['D_Z'] + (1 if age else 0))
    PROD_SW = (_sw0['SITE_ATTR'], _sw0['AGE_CH'], _sw0['JUMP_RESET'], _sw0['STATE_ATTR'])          # the configured (production) switches
    sw(False, False)                                                                                # T4e-j: the base flow
    flow = type(ns_['flow'])(); Dz = ns_['D_Z']; z0 = torch.zeros(3, Dz); dist = torch.zeros(3, NY, NC + 2); dist[1, 2010 - 2000, 0] = 1; soil = torch.zeros(3, ns_['SOIL'].shape[1])
    with torch.no_grad():
        for p_ in flow.J.parameters(): p_.fill_(0.0)
        flow.J[-1].bias.fill_(1.0); flow.A.fill_(0.5)                                    # jump = tanh(1) on every dim when an event fires; non-zero attractor
        st = flow(z0, torch.tensor([2005.0, 2005.0, 2015.0]), [2009, 2010, 2011], dist, soil)
        j10 = (st[2010][1] - st[2010][0]).abs().mean().item(); j09 = (st[2009][1] - st[2009][0]).abs().max().item()
        moved = st[2011][0].abs().min().item(); frozen = st[2011][2].abs().max().item()
    check('T4e flow: event jump = tanh(1) exactly at its year, none before; a visit-2015 state is untouched in 2011 while a visit-2005 state has flowed',
          j09 < 1e-7 and abs(j10 - math.tanh(1.0)) < 1e-5 and moved > 0 and frozen == 0, f'diff@2009 {j09:.1e}, jump@2010 {j10:.5f} (tanh 1 = 0.76159), flowed {moved:.3f}, frozen {frozen}')
    with torch.no_grad():
        st2 = flow(torch.zeros(2, Dz), torch.tensor([2005.0, 2015.0]), [2011], torch.zeros(2, NY, NC + 2), torch.zeros(2, ns_['SOIL'].shape[1]))
    check('T4f flow: a sample whose visit is after the stop year keeps its start state', (st2[2011][1] == 0).all().item())
    with torch.no_grad():
        dist3 = torch.zeros(3, NY, NC + 2); dist3[:, 2010 - 2000, 0] = 1
        st3 = flow(torch.zeros(3, Dz), torch.tensor([2010.25, 2010.75, 2009.5]), [2010], dist3, torch.zeros(3, ns_['SOIL'].shape[1]))
        nojump = flow(torch.zeros(1, Dz), torch.tensor([2010.25]), [2010], torch.zeros(1, NY, NC + 2), torch.zeros(1, ns_['SOIL'].shape[1]))[2010]
        j_early = (st3[2010][0] - nojump[0]).abs().mean().item(); j_late = (st3[2010][1] - flow(torch.zeros(1, Dz), torch.tensor([2010.75]), [2010], torch.zeros(1, NY, NC + 2), torch.zeros(1, ns_['SOIL'].shape[1]))[2010][0]).abs().max().item()
        m_a = flow(torch.zeros(1, Dz), torch.tensor([2010.0]), [2010], torch.zeros(1, NY, NC + 2), torch.zeros(1, ns_['SOIL'].shape[1]))[2010].abs().mean().item()
        m_b = nojump.abs().mean().item()
    check('T4h flow timing: a visit-year event counts for a first-half visit only; the first year integrates only the part after the visit',
          abs(j_early - math.tanh(1.0)) < 1e-5 and j_late < 1e-7 and 0 < m_b < m_a, f'jump(2010.25) {j_early:.5f}, jump(2010.75) {j_late:.1e}, drift full year {m_a:.3f} > 0.75 yr {m_b:.3f}')
    pid_t = max((k for k in ns_['cand_of'] if len(ns_['cand_of'][k][0]) >= 3 and ns_['cand_of'][k][1].max() < 0.9), key=lambda k: ns_['cand_of'][k][1].max()); cid_t, pr_t = ns_['cand_of'][pid_t]; np.random.seed(0)   # (a non-degenerate draw: r8 M7)
    draws = np.concatenate([ns_['sample_cands']([pid_t], 48)[0] for _ in range(400)]); fr = pd.Series(draws).value_counts().reindex(cid_t).fillna(0).values / len(draws)
    check('T4i MIL candidates are iid draws from the prior (empirical frequency of the most concentrated plot)', np.abs(fr - pr_t).max() < 0.02, f'max prior {pr_t.max():.3f}, empirical {fr[np.argmax(pr_t)]:.3f}, max |Δ| {np.abs(fr - pr_t).max():.4f}')
    with torch.no_grad():
        zA = torch.zeros(2, Dz); stE = flow(zA, torch.tensor([2010.2, 2010.2]), [2014], torch.zeros(2, NY, NC + 2), torch.zeros(2, ns_['SOIL'].shape[1]), end=torch.tensor([2014.3, 2014.9]))
        stF = flow(zA[:1], torch.tensor([2010.2]), [2014, 2015], torch.zeros(1, NY, NC + 2), torch.zeros(1, ns_['SOIL'].shape[1]))
        dE = torch.zeros(1, NY, NC + 2); dE[0, 2014 - 2000, 0] = 1
        e1 = flow(zA[:1], torch.tensor([2010.2]), [2014], dE, torch.zeros(1, ns_['SOIL'].shape[1]), end=torch.tensor([2014.3]))[2014]
        e2 = flow(zA[:1], torch.tensor([2010.2]), [2014], dE, torch.zeros(1, ns_['SOIL'].shape[1]), end=torch.tensor([2014.9]))[2014]
        e2n = flow(zA[:1], torch.tensor([2010.2]), [2014], torch.zeros(1, NY, NC + 2), torch.zeros(1, ns_['SOIL'].shape[1]), end=torch.tensor([2014.9]))[2014]
    check('T4j flow end time (pair targets): the state is taken AT the end date (between end-of-2013 and end-of-2014 states), end-year events only if before the end',
          (stE[2014][0] - stE[2014][1]).abs().max() > 0 and (e1 - stE[2014][0]).abs().max() < 1e-6 and abs((e2 - e2n).abs().mean().item() - math.tanh(1.0)) < 1e-5,
          f'Δ(end 2014.3 vs 2014.9) {(stE[2014][0] - stE[2014][1]).abs().max():.3f}; event of 2014 with end 2014.3 ignored, with end 2014.9 applied')
    # T4w (G46.20 / A14-8) the 2026 clamp: a visit after the last year (2026) is flowed to the END-2025 state (stop year YEARS[-1], end in 2026): its
    # 2026 part and 2026 events are dropped, while an end inside 2025 still changes the state (the test can tell the end time apart)
    with torch.no_grad():
        torch.manual_seed(14); zw = torch.randn(2, Dz); dw = torch.zeros(2, NY, NC + 2); dw[:, YEARS[-1] - YEARS[0], 0] = 1; sw_ = torch.zeros(2, ns_['SOIL'].shape[1])
        w26 = flow(zw, torch.tensor([2019.4, 2019.4]), [YEARS[-1]], dw, sw_, end=torch.tensor([YEARS[-1] + 1.3, YEARS[-1] + 1.9]))[YEARS[-1]]
        w25 = flow(zw, torch.tensor([2019.4, 2019.4]), [YEARS[-1]], dw, sw_)[YEARS[-1]]
        w25m = flow(zw, torch.tensor([2019.4, 2019.4]), [YEARS[-1]], dw, sw_, end=torch.tensor([YEARS[-1] + 0.4, YEARS[-1] + 0.4]))[YEARS[-1]]
    _n26 = int((V.year.values >= YEARS[-1] + 1).sum())
    check(f'T4w 2026 clamp: a visit in {YEARS[-1] + 1} is flowed to the end-{YEARS[-1]} state (its own-year part and events dropped); an end inside {YEARS[-1]} differs',
          (w26 - w25).abs().max().item() < 1e-6 and (w25m - w25).abs().max().item() > 1e-4, f'|Δ| end {YEARS[-1] + 1}.3 / .9 vs end of {YEARS[-1]}: {(w26 - w25).abs().max().item():.1e}; '
          f'end {YEARS[-1]}.4 vs end of {YEARS[-1]}: {(w25m - w25).abs().max().item():.3f}; table visits in {YEARS[-1] + 1}+: {_n26}')
    # T4m visit-year rule (audit r5 I3): an event in the END visit's own year is applied before a first-half end only if that visit recorded it (re); an event
    # in the START visit's own year is not applied after a first-half start if the start visit recorded it (rs)
    with torch.no_grad():
        dm = torch.zeros(1, NY, NC + 2); dm[0, 2014 - 2000, 0] = 1; zm = torch.zeros(1, Dz); sm = torch.zeros(1, ns_['SOIL'].shape[1]); T_, F_ = torch.tensor([True]), torch.tensor([False])
        e_no = flow(zm, torch.tensor([2010.2]), [2014], dm, sm, end=torch.tensor([2014.3]))[2014]; e_re = flow(zm, torch.tensor([2010.2]), [2014], dm, sm, end=torch.tensor([2014.3]), re=T_)[2014]
        s_no = flow(zm, torch.tensor([2014.2]), [2014], dm, sm)[2014]; s_rs = flow(zm, torch.tensor([2014.2]), [2014], dm, sm, rs=T_)[2014]
        s_rsra = flow(zm, torch.tensor([2014.2]), [2014], dm, sm, rs=T_, ra=T_)[2014]       # rs wins: the start's own-year evidence predates it (r8 M5)
        s_ra = flow(zm, torch.tensor([2014.7]), [2014], dm, sm, ra=T_)[2014]; s_late = flow(zm, torch.tensor([2014.7]), [2014], dm, sm)[2014]
        s_0b = flow(zm, torch.tensor([2014.7]), [2014], torch.zeros_like(dm), sm)[2014]
        s_0 = flow(zm, torch.tensor([2014.2]), [2014], torch.zeros_like(dm), sm)[2014]; e_0 = flow(zm, torch.tensor([2010.2]), [2014], torch.zeros_like(dm), sm, end=torch.tensor([2014.3]))[2014]
    sw(True, True, PROD_SW[2], PROD_SW[3])                                                         # the same rule with the production switches (the clock dimension excluded)
    fP = type(ns_['flow'])()
    with torch.no_grad():
        for p_ in fP.J.parameters(): p_.fill_(0.0)
        fP.J[-1].bias.fill_(1.0)
        q_no = fP(zm, torch.tensor([2010.2]), [2014], dm, sm, end=torch.tensor([2014.3]))[2014][:, 1:]; q_re = fP(zm, torch.tensor([2010.2]), [2014], dm, sm, end=torch.tensor([2014.3]), re=T_)[2014][:, 1:]
        q_0 = fP(zm, torch.tensor([2010.2]), [2014], torch.zeros_like(dm), sm, end=torch.tensor([2014.3]))[2014][:, 1:]
    sw(False, False)
    prod_ok = (q_no - q_0).abs().max() < 1e-7 and ((q_re - q_0).abs().mean().item() > 1e-3 if PROD_SW[2] else abs((q_re - q_0).abs().mean().item() - math.tanh(1.0)) < 1e-5)   # (JUMP_RESET: a reset, not tanh(1); T4n)
    check('T4m flow visit-year rule (base and production switches): re applies the end visit\'s own-year event before a first-half end; rs suppresses the start visit\'s recorded own-year event',
          prod_ok and (s_rsra - s_0).abs().max() < 1e-7 and (s_late - s_0b).abs().max() < 1e-7 and abs((s_ra - s_0b).abs().mean().item() - math.tanh(1.0)) < 1e-5 and
          (e_no - e_0).abs().max() < 1e-7 and abs((e_re - e_0).abs().mean().item() - math.tanh(1.0)) < 1e-5 and abs((s_no - s_0).abs().mean().item() - math.tanh(1.0)) < 1e-5 and (s_rs - s_0).abs().max() < 1e-7)
    # T4k site attractor (SITE_ATTR) and age clock (AGE_CH): an independently-built flow with both switches on (module globals of the exec'd g03b)
    sw(True, True)                                                                                   # T4k: SITE_ATTR + AGE_CH (the r4-r12 flow)
    try:
        fk = type(ns_['flow'])(); nso = ns_['SOIL'].shape[1]; torch.manual_seed(4); AS = ns_['AGE_S']; AC = ns_['AGE_C']
        with torch.no_grad():
            for p_ in [*fk.g.parameters(), *fk.J.parameters()]: p_.fill_(0.0)
            fk.g[-1].bias.fill_(2.0); fk.ltau.fill_(0.0); fk.A.fill_(0.3); fk.J[-1].bias.fill_(1.0); fk.lrho.fill_(math.log(0.8))
            for p_ in fk.Am.parameters(): p_.copy_(torch.randn_like(p_))
            sk = torch.randn(2, nso); dk = torch.zeros(2, NY, NC + 2); dk[1, 2015 - 2000, 0] = 1
            sK = fk(torch.zeros(2, Dz), torch.tensor([2000.0, 2000.0]), [2012, 2015, 2025], dk, sk)
            gk = torch.sigmoid(torch.tensor(2.0)); lk4 = 1.0 + gk; zfix = gk * 3 * torch.tanh(fk.A + fk.Am(sk)) / lk4     # the fixed point, computed independently
            e_fix = (sK[2012][0, 1:] - zfix[0, 1:]).abs().max().item(); e_soil = (zfix[0, 1:] - zfix[1, 1:]).abs().max().item()
            ck12 = sK[2012][0, 0].item(); ck25 = sK[2025][0, 0].item()
            pre = 16 * 0.8 / AS; r_ = 1 / (1 + math.exp(-1.0)); post = pre + r_ * (-AC / AS - pre) + 0.25 * math.tanh(1.0)      # the event of 2015 at end-of-year 2015
            ck15e = sK[2015][1, 0].item()
            fk.J[-1].bias[Dz:].fill_(30.0)                                                    # full reset: an old stand cut to age ~0
            zo = torch.zeros(1, Dz); zo[0, 0] = (100 - AC) / AS
            sR = fk(zo, torch.tensor([2014.9]), [2015], dk[1:], sk[1:])[2015][0, 0].item()
        check('T4k flow SITE_ATTR + AGE_CH: dims 1.. converge to the soil-specific attractor; dim 0 advances by dt·rho / AGE_S; an event RESETS the clock by the learned fraction (a cut 100-yr stand -> ~0)',
              e_fix < 1e-4 and e_soil > 1e-2 and abs(ck12 - 13 * 0.8 / AS) < 1e-5 and abs(ck25 - 26 * 0.8 / AS) < 1e-5 and abs(ck15e - post) < 1e-5 and abs(sR - (-AC / AS + 0.25 * math.tanh(1.0))) < 1e-5,
              f'|z - fixed point| {e_fix:.1e}, soil spread {e_soil:.3f}, clock 2012 {ck12:.4f} (13·0.8/40 = 0.26), after the 2015 event {ck15e:.4f} (expected {post:.4f}), cut 100-yr stand -> age {sR * AS + AC:.1f} yr')
    finally: ns_.update(_sw0)
    # T4n-q the r13 model switches (G29-G31), each unit-tested on an independently-built flow / statistic of the exec'd g03b (whatever the configured switches)
    nso = ns_['SOIL'].shape[1]
    if 'JUMP_RESET' in open(f'{ROOT}/scripts/g03b_train_gafl.py').read():
        try:
            for age_ in (False, True):          # T4n JUMP_RESET (G30, bounded by G46.10): the jump is r ⊙ (zt − z) with target zt = z_post + sc ⊙ tanh(shift) clamped to ±3, z_post the P network's 3·tanh output
                                                # (AGE_CH: clock z_post = −AGE_C / AGE_S, its shift scale 0.25 (G30), its target never below age 0); a convex step stays in [−3, 3]
                sw(True, age_, True, False); torch.manual_seed(5); fj = type(ns_['flow'])()
                with torch.no_grad():
                    for p_ in fj.J.parameters(): p_.fill_(0.0)
                    bj = torch.randn(2 * Dz); fj.J[-1].bias.copy_(bj); z_ = torch.randn(4, Dz) * 1.5; ev_ = torch.rand(4, NC + 2); so_ = torch.randn(4, nso)
                    W1, b1, W2, b2 = fj.P[0].weight, fj.P[0].bias, fj.P[2].weight, fj.P[2].bias; zp_ = 3 * torch.tanh(torch.tanh(torch.cat([ev_, so_], 1) @ W1.T + b1) @ W2.T + b2)
                    if age_: zp_[:, 0] = -ns_['AGE_C'] / ns_['AGE_S']
                    sc_ = torch.full((Dz,), 0.5); sc_[0] = 0.25 if age_ else 0.5
                    zt_ = (zp_ + sc_ * torch.tanh(bj[:Dz])).clamp(-3, 3)
                    if age_: zt_[:, 0] = zt_[:, 0].clamp(min=-ns_['AGE_C'] / ns_['AGE_S'])
                    ej = (fj.jump(z_, ev_, so_) - torch.sigmoid(bj[Dz:]) * (zt_ - z_)).abs().max().item()
                    zb_ = (torch.rand(256, Dz) * 6 - 3); eb_, sb_ = torch.rand(256, NC + 2), torch.randn(256, nso)
                    fj.J[-1].bias.copy_(torch.randn(2 * Dz) * 3); nb_ = zb_ + fj.jump(zb_, eb_, sb_); fj.J[-1].bias.copy_(bj)
                    eb = max(0.0, (nb_.abs().max().item() - 3.0))                  # (G46.10: bounded)
                    ec = max(0.0, (-ns_['AGE_C'] / ns_['AGE_S'] - nb_[:, 0]).max().item()) if age_ and (zb_[:, 0] >= -ns_['AGE_C'] / ns_['AGE_S']).all() else 0.0
                    fj.J[-1].bias[:Dz].fill_(0.0); fj.J[-1].bias[Dz:].fill_(40.0)          # r -> 1, no shift: a stand at +2.9 lands AT z_post
                    dj = torch.zeros(1, NY, NC + 2); dj[0, 2010 - 2000, 0] = 1; sj = torch.randn(1, nso)
                    land_ = fj(torch.full((1, Dz), 2.9), torch.tensor([2009.2]), [2010], dj, sj)[2010]
                    zq = 3 * torch.tanh(fj.P(torch.cat([dj[:, 2010 - 2000], sj], 1)))
                    if age_: zq[:, 0] = -ns_['AGE_C'] / ns_['AGE_S']
                    el = (land_ - zq).abs().max().item()
                check(f'T4n JUMP_RESET{" + AGE_CH" if age_ else ""}: jump = sigmoid(r) (clamp(z_post + sc tanh(shift)) − z) exactly; with r -> 1 a stand at +2.9 lands at z_post; jumped states stay in [−3, 3] (G46.10)',
                      ej < 1e-5 and el < 1e-5 and eb < 1e-6 and ec < 1e-6, f'max |jump − spec| {ej:.1e}; |landed − z_post| {el:.1e}; beyond ±3 by {eb:.1e}' + (f'; clock below age 0 by {ec:.1e}' if age_ else ''))
        finally: ns_.update(_sw0)
    else: print('SKIP  T4n g03b has no JUMP_RESET switch')
    if 'STATE_ATTR' in open(f'{ROOT}/scripts/g03b_train_gafl.py').read():
        try:                                     # T4o STATE_ATTR (G31): attractor A + Am([soil, z]) frozen within each year (exact exponential step), ltau clamped to [−5, 10]
            sw(True, False, False, True); torch.manual_seed(6); fo = type(ns_['flow'])()
            with torch.no_grad():
                for p_ in fo.Am.parameters(): p_.copy_(torch.randn_like(p_) * 0.5)
                for p_ in fo.J.parameters(): p_.fill_(0.0)
                z0 = torch.randn(3, Dz); so_ = torch.randn(3, nso); d0 = torch.zeros(3, NY, NC + 2)
                got = fo(z0, torch.tensor([2010.0] * 3), [2011], d0, so_)[2011]
                zz = z0.clone()
                for _ in range(2):                  # the spec: per year g, λ and the attractor from the state at the START of the year, exact step over dt = 1
                    g_ = torch.sigmoid(fo.g(torch.cat([zz, so_], 1))); lam_ = 1 / torch.exp(fo.ltau.clamp(-5, 10)) + g_; Aa_ = fo.A + fo.Am(torch.cat([so_, zz], 1))
                    zs_ = g_ * 3 * torch.tanh(Aa_) / lam_; zz = zs_ + (zz - zs_) * torch.exp(-lam_)
                e1 = (got - zz).abs().max().item()
                for p_ in [*fo.g.parameters(), *fo.Am.parameters()]: p_.fill_(0.0)
                fo.g[-1].bias.fill_(2.0); fo.ltau.fill_(0.0); fo.A.fill_(1.0); fo.Am[-1].bias.fill_(-3.0)     # soil-only box [0, 3 tanh(1)]; the state network pulls to A − 3 = −2
                zf = fo(torch.full((1, Dz), 2.0), torch.tensor([2000.0]), [2025], d0[:1], so_[:1])[2025]; gk = torch.sigmoid(torch.tensor(2.0)); zstar = float(gk * 3 * torch.tanh(torch.tensor(-2.0)) / (1 + gk))
                e2 = (zf - zstar).abs().max().item()
                fo.ltau.fill_(-50.0); a_ = fo(z0[:1], torch.tensor([2010.0]), [2010], d0[:1], so_[:1])[2010]; fo.ltau.fill_(-5.0); b_ = fo(z0[:1], torch.tensor([2010.0]), [2010], d0[:1], so_[:1])[2010]
                fo.ltau.fill_(50.0); c_ = fo(z0[:1], torch.tensor([2010.0]), [2010], d0[:1], so_[:1])[2010]; fo.ltau.fill_(10.0); d_ = fo(z0[:1], torch.tensor([2010.0]), [2010], d0[:1], so_[:1])[2010]
                e3 = max((a_ - b_).abs().max().item(), (c_ - d_).abs().max().item())
            check('T4o STATE_ATTR: the yearly step = exact exponential step to A + Am([soil, z]) frozen within the year; a positive state converges to a negative attractor outside the soil-only box [0, a(soil)]; ltau clamped to [−5, 10]',
                  e1 < 1e-5 and e2 < 1e-4 and zstar < 0 and e3 < 1e-6, f'|2-yr flow − spec| {e1:.1e}; 25 yr from +2.0: |z − z*| {e2:.1e} (z* = {zstar:.3f}); clamp |Δ| {e3:.1e}')
        finally: ns_.update(_sw0)
    else: print('SKIP  T4o g03b has no STATE_ATTR switch')
    if 'SIGREG_MEAN' in ns_:                    # T4p SIGREG_MEAN (G29): the location is tested too (uncentred); centred, a shifted cloud is invisible
        _sm0 = ns_['SIGREG_MEAN']; torch.manual_seed(7); zn_ = torch.randn(4096, Dz)
        try:
            ns_['SIGREG_MEAN'] = True; on0, on3 = ns_['sigreg'](zn_).item(), ns_['sigreg'](zn_ + 3).item(); ns_['SIGREG_MEAN'] = False; of0, of3 = ns_['sigreg'](zn_).item(), ns_['sigreg'](zn_ + 3).item()
        finally: ns_['SIGREG_MEAN'] = _sm0
        check('T4p SIGREG_MEAN: sigreg(N(0, I) + 3) >> sigreg(N(0, I)) with SIGREG_MEAN = 1; about equal with 0', on3 > 20 * max(on0, 1e-6) and abs(of3 - of0) <= 0.25 * max(of0, of3) + 1e-4,
              f'SIGREG_MEAN 1: {on0:.5f} vs shifted {on3:.5f}; 0: {of0:.5f} vs {of3:.5f}')
    else: print('SKIP  T4p g03b has no SIGREG_MEAN switch')
    if 'latent_stats' in ns_:                   # T4q latent_stats (G46.8, superseding G29's rank rule): its verdict = the rule re-derived here on each test cloud
        _enc0 = ns_['enc']; _ac = 1 if ns_.get('AGE_CH') else 0; _nq = min(4096, len(V)); torch.manual_seed(8)
        def _lat(Zn):
            Zf = torch.cat([torch.randn(len(Zn), _ac), Zn], 1) if _ac else Zn; ns_['enc'] = lambda v, yr: Zf[v]
            try: return ns_['latent_stats'](np.arange(len(Zf)))
            finally: ns_['enc'] = _enc0
        def _rule(Zn): return bool(Zn.mean(0).abs().max() <= 2.0 and Zn.std(0).min() >= 0.3 and (Zn.abs() > 2.9).float().mean() <= 0.01)   # (non-clock dims)
        sat_ = torch.randn(_nq, 8); sat_[:int(0.03 * _nq), :] = 2.95 * torch.sign(torch.randn(int(0.03 * _nq), 8))
        clouds = {'N(0, I) 8': torch.randn(_nq, 8), 'N(0, I) 16': torch.randn(_nq, 16), '3 dims at mean 2.9': torch.randn(_nq, 8) + torch.tensor([2.9, 2.9, 2.9, 0, 0, 0, 0, 0]),
                  'one dim at mean 2.1': torch.randn(_nq, 8) + torch.tensor([2.1, 0, 0, 0, 0, 0, 0, 0]), 'one dim sd 0.1': torch.randn(_nq, 8) * torch.tensor([0.1] + [1.0] * 7),
                  '3 % of the values at ±2.95': sat_, 'rank 2, sd >= 0.5 (rank is report-only)': torch.randn(_nq, 2) @ (torch.ones(2, 16) * 0.6)}
        res_q = {k: (bool(_lat(z)['ok']), _rule(z)) for k, z in clouds.items()}
        exp_q = {'N(0, I) 8': True, 'N(0, I) 16': True, '3 dims at mean 2.9': False, 'one dim at mean 2.1': False, 'one dim sd 0.1': False, '3 % of the values at ±2.95': False}
        check('T4q latent_stats (G46.8): verdict = max |mean| <= 2, min sd >= 0.3, share |z| > 2.9 <= 1 % re-derived on every test cloud; healthy clouds pass, shifted / shrunk / saturated fail',
              all(g_ == r_ for g_, r_ in res_q.values()) and all(res_q[k][0] == v for k, v in exp_q.items()), f'{{cloud: (latent_stats ok, rule)}} {res_q}')
    else: print('SKIP  T4q g03b has no latent_stats')
    TKp = TK[TK.N > 0]; ao = ns_['AGE_OBS']; _vs4 = np.random.default_rng(41).choice(TKp.vid.unique(), 300, replace=False); _e4 = 0.0
    for vv in _vs4:
        tv_ = TKp[TKp.vid == vv]; w_ = tv_.N * tv_.QMD ** 2; _e4 = max(_e4, abs(float(ao[int(vv)]) - float((w_ * tv_.A).sum() / w_.sum())))
    check('T4l age-clock target AGE_OBS = BA-weighted mean cohort age of the visit (independent, 300 visits); bare visits 0',
          _e4 < 1e-3 and float(ao[torch.tensor(sorted(set(range(len(V))) - set(TKp.vid.astype(int).unique())))].abs().max() if len(set(range(len(V))) - set(TKp.vid.astype(int).unique())) else 0) == 0)
    zn = torch.randn(4096, Dz); zs = zn * 3
    check('T4g SIGReg: ~0 for N(0, I), large for a scaled cloud', ns_['sigreg'](zn).item() < 0.2 * ns_['sigreg'](zs).item(), f'{ns_["sigreg"](zn).item():.4f} vs {ns_["sigreg"](zs).item():.4f}')
    # ======================= T5 landscape descriptors: inclusion-weighted subsample vs the full disk
    exec(compile(G4[G4.index('from sklearn.cluster import'):G4.index('XL = {}')], 'g04', 'exec'), ns_)
    AEF, DIST, SOIL = ns_['AEF'], ns_['DIST'], ns_['SOIL']; smu, ssd = ns_['smu'], ns_['ssd']; land_ = ns_['landscape']; dW, dU = [], []
    _iw5 = C.groupby('pid').incl_w.agg(['min', 'max']); _sub5 = _iw5.index.values[(_iw5['max'].values > 1) & (_iw5['max'].values > _iw5['min'].values * (1 + 1e-6))]   # disks subsampled
    # with a NON-CONSTANT inclusion weight (r14 G47.7 / E14-8: with a constant one the weighted and unweighted estimates coincide, which dilutes T5s' power)
    for pid in np.random.default_rng(51).choice(_sub5, min(60, len(_sub5)), replace=False) if len(_sub5) else []:   # (random subsampled plots: r13 E-F5)
        g = C[C.pid == pid]; cid = g.cid.values
        xs = land_(AEF[cid], DIST[cid], SOIL[cid], g.incl_w.values); xu = land_(AEF[cid], DIST[cid], SOIL[cid])
        r0, c0 = land.index(Pi.loc[pid].x, Pi.loc[pid].y); RP = int(1600 // 30) + 2
        if AEF90: wr0, wc0, hw = 3 * ((r0 - RP) // 3), 3 * ((c0 - RP) // 3), 2 * RP + 7
        else: wr0, wc0, hw = r0 - RP, c0 - RP, 2 * RP + 1
        rr, cc, F = G2.feats(wr0, wc0, hw, hw); F = F[np.hypot(rr + wr0 - r0, cc + wc0 - c0) * 30 <= 1600]; F[:, -len(SOILS):] = (F[:, -len(SOILS):] - smu) / ssd
        F = torch.tensor(F); xf = land_(F[:, :128], F[:, 128:128 + NY * (NC + 2)].reshape(-1, NY, NC + 2), F[:, -len(SOILS):])
        dW.append(xs - xf); dU.append(xu - xf)
    dW, dU = np.array(dW), np.array(dU)
    def frac_sig(dd_):                                         # share of descriptor dims whose mean signed error across plots is significant (|t| > 3.5)
        sd_ = dd_.std(0, ddof=1); ok_ = sd_ > 1e-9; t_ = np.abs(dd_.mean(0)[ok_]) / (sd_[ok_] / np.sqrt(len(dd_))); return float((t_ > 3.5).mean())
    fw, fu = frac_sig(dW), frac_sig(dU)
    check('T5 inclusion-weighted landscape is an UNBIASED estimate of the full-disk landscape (signed errors across subsampled plots; unweighted shown for power)',
          len(dW) >= 10 and fw < 0.05, f'{len(dW)} subsampled plots: dims with significant bias — weighted {fw:.3f}, unweighted {fu:.3f}; mean |Δ| weighted {np.abs(dW).mean():.4f}, unweighted {np.abs(dU).mean():.4f}')
    # T5s self-test of T5's power (r13 E-F5): the same statistic with the inclusion weights DROPPED. Where the unweighted estimate deviates materially from the
    # full disk, the weighted one must be closer. The aggregate rule decides (r14 G47.7); the per-dim rule (|mean signed error| <= 0.5 x the unweighted one on
    # dims with unweighted |t| > 3.5) is reported only (its false-alarm rate is uncalibrated: the weighted estimate has a larger sd). With no material
    # unweighted deviation the weights cannot be told apart here (INFO, not a failure: the subsample is then near-uniform in that table).
    if len(dW) >= 10:
        def _tst(dd_):                     # per-dim |t| of the mean signed error across plots
            sd_ = dd_.std(0, ddof=1); t_ = np.zeros(dd_.shape[1]); ok_ = sd_ > 1e-9; t_[ok_] = np.abs(dd_.mean(0)[ok_]) / (sd_[ok_] / np.sqrt(len(dd_))); return t_, ok_
        _tU, _okU = _tst(dU); _tW, _ = _tst(dW); _mat = _tU > 3.5
        _bw, _bu = np.abs(dW.mean(0)), np.abs(dU.mean(0)); _viol = _mat & (_bw > 0.5 * _bu)
        # aggregate form (more power when the bias is spread thinly): the unweighted bias is material when sum t^2 exceeds its no-bias mean by 3 sd
        # (df + 3 sqrt(2 df)); then the weighted sum t^2 must be smaller
        _df = int(_okU.sum()); _cU, _cW = float((_tU[_okU] ** 2).sum()), float((_tW[_okU] ** 2).sum()); _matA = _cU > _df + 3 * math.sqrt(2 * max(_df, 1))
        if _matA:
            check('T5s T5 power self-test: where the UNWEIGHTED landscape is materially biased, the weighted one is closer to the full disk (aggregate: smaller sum t^2; per dim reported)',
                  not _matA or _cW < _cU,
                  f'{len(dU)} plots with non-constant incl_w; sum t^2 unweighted {_cU:.0f} / weighted {_cW:.0f} (df {_df}, material above {_df + 3 * math.sqrt(2 * max(_df, 1)):.0f}); '
                  f'per dim (report only): {int(_mat.sum())} materially biased dims (unweighted |t| > 3.5), weighted not within 0.5x on {int(_viol.sum())}')
        else: print(f'INFO  T5s no material bias without the weights ({len(dU)} plots, max unweighted |t| {_tU.max():.2f}, sum t^2 {_cU:.0f} for df {_df}): T5 cannot distinguish missing weights in this table')
    # ======================= T6 stand mapping inputs
    S = pd.read_parquet(f'{D}/stands.parquet'); SF = np.load(f'{D}/stand_feat.npy', mmap_mode='r'); ssub = S.sample(200, random_state=5); Xs = np.asarray(SF[ssub.index.values], dtype=np.float32)
    sraster = rasterio.open(stands_run()['tif'])
    sid_chk = np.array([sraster.read(1, window=Window(c, r, 1, 1))[0, 0] for r, c in zip(ssub.row, ssub.col)])
    check('T6a stand footprints lie in their stand (stands.tif)', (sid_chk == ssub.stand_id.values).all())
    dR6, dC6 = G2.stand_offset(sraster)          # (AEF90: the feature grid is grid_r4)
    e25s = np.stack([aef_fp(a25, r + dR6, c + dC6) for r, c in zip(ssub.row, ssub.col)])
    check('T6b stand footprint AlphaEarth 2025 = mosaic 3x3 mean' + (' (AEF90: the 90 m pixel)' if AEF90 else ''), np.abs(e25s - Xs[:, 64:128]).max() < 2e-3, f'max abs err {np.abs(e25s - Xs[:, 64:128]).max():.2e}')
    # T6c (r13 coverage, g02j): the stand footprints' AlphaEarth 2022, disturbance history (T3e's spec) and soils (T3d's spec) — T6b checked 2025 only
    e22s = np.stack([aef_fp(a22, r + dR6, c + dC6) for r, c in zip(ssub.row, ssub.col)]); _e6 = np.abs(e22s - Xs[:, :64]).max()
    _d6 = [dist_err(xr, r + dR6, c + dC6)[0] for (r, c), xr in zip(zip(ssub.row.values[:100], ssub.col.values[:100]), Xs[:100])]
    _so6 = np.array([[(lambda v: FILL[k] if v <= -9000 else v)(float(rasterio.open(SOILP(k)).read(1, window=Window((c + dC6) // SDIV, (r + dR6) // SDIV, 1, 1))[0, 0])) for k in SOILS]
                     for r, c in zip(ssub.row.values[:100], ssub.col.values[:100])]); _se6 = (np.abs(_so6 - Xs[:100, -len(SOILS):]) / (np.abs(_so6) * 1e-3 + 1e-3)).max()
    check('T6c stand footprint AlphaEarth 2022, disturbance fractions / severity / Hansen (T3e spec) and soils (T3d spec) = direct raster lookups', _e6 < 2e-3 and max(_d6) <= 2e-3 and _se6 < 1,
          f'{len(ssub)} footprints: AEF22 max |Δ| {_e6:.1e}; {len(_d6)} disturbance histories, max |Δ| {max(_d6):.1e}, off {sum(d > 2e-3 for d in _d6)}; soils max err / tol {_se6:.2f}')
    # T6d (r13 E-F9) the stand-raster -> feature-grid offset of g02b.stand_offset re-derived by COORDINATES: a stand pixel's centre (stands.tif transform) indexed in
    # the feature land raster must be the offset pixel, on 500 random stand footprints
    _s6d = S.sample(min(500, len(S)), random_state=61); _xy6 = [sraster.xy(r, c) for r, c in zip(_s6d.row.values, _s6d.col.values)]
    _rc6 = np.array([land.index(x_, y_) for x_, y_ in _xy6]); _ok6 = (_rc6[:, 0] == _s6d.row.values + dR6) & (_rc6[:, 1] == _s6d.col.values + dC6)
    _g6 = (_s6d.kind == 'grid').values if 'kind' in _s6d.columns else np.ones(len(_s6d), bool)
    _c6 = (((_s6d.row.values + dR6) % 3 == 1) & ((_s6d.col.values + dC6) % 3 == 1))[_g6].mean() if AEF90 and _g6.any() else 1.0
    check('T6d stand offset (g02b.stand_offset) = the coordinate route on 500 stand footprints' + ('; grid footprints on 90 m block centres' if AEF90 else ''), _ok6.all() and _c6 == 1.0,
          f'offset ({dR6}, {dC6}); agreement {_ok6.mean():.4f}' + (f'; grid footprints at block centres {_c6:.4f}' if AEF90 else ''))
    # ======================= T7 downstream keys (pid is a row number of THIS table)
    LS = pd.read_parquet(f'{D}/localisation.parquet'); jl = LS.join(P.set_index('pid')[key], rsuffix='_t')
    check('T7a localisation rows carry the same FIA plot identity as plots.parquet', all((jl[k] == jl[k + '_t']).all() for k in key) and len(LS) == C.pid.nunique(), f'{len(LS)} plots')
    if 'n_clusters' in LS.columns:     # g02g localisation: n_hi = footprints consistent with the plot's whole FIA history
        CHh = pd.read_parquet(f'{D}/cand_hist.parquet'); kh = C.merge(CHh, on=['cid', 'pid']).query('cons').groupby('pid').incl_w.sum().reindex(LS.index).fillna(0)
        check('T7b localisation n_hi = consistent footprints (Σ incl_w over cand_hist.cons); mass_hi = 1 iff non-empty', np.allclose(kh, LS.n_hi) and ((LS.mass_hi == 1) == (LS.n_hi > 0)).all(),
              f'{len(LS)} plots, non-empty {int((LS.n_hi > 0).sum())}')
    else:
        Pq = P.set_index('pid').reindex(LS.index); check('T7b localisation n_hi / mass_hi = the full-disk numbers of g02b (plots)', np.allclose(LS.n_hi, Pq.n_hi) and np.allclose(LS.mass_hi, Pq.mass_hi))
        kh = C[C.hi].groupby('pid').incl_w.sum().reindex(LS.index).fillna(0); full = C.groupby('pid').incl_w.max().reindex(LS.index) == 1
        check('T7c kept high-prior footprints = n_hi where the disk was not subsampled', np.allclose(kh[full], LS.n_hi[full]), f'{int(full.sum())} plots unsampled')
    import glob; fns = [os.path.basename(f)[:-4] for f in glob.glob(f'{D}/visit_aef/*.npy') if '.part' not in f]
    kn = set(tuple(map(int, f.split('_'))) for f in fns if f.count('_') == 4)
    lp = LS[(LS.n_hi > 0) & (LS.n_hi <= 50) & (LS.mass_hi >= 0.5)]; vl = V[V.pid.isin(lp.index)].assign(yi=lambda x: x.year.astype(int))
    need = {(int(r.st), int(r.u), int(r.c), int(r.p), int(r.yi)) for r in vl.itertuples() if 2017 <= r.yi <= 2025 and r.yi not in (2022, 2025)}
    print(f'INFO  T7d visit_aef (not consumed by any model, r13 E-F6): {len(fns)} files, named by FIA plot identity {len(kn) == len(fns)} (of current table plots '
          f'{sum(1 for k in kn if k[:4] in set(map(tuple, P[key].astype(int).values)))}); needed {len(need)}, missing {len(need - kn)}')
    # T7e the production location prior of the model (LOC; r13 coverage): LOC=history -> g02b's prior zeroed outside the FIA-history-consistent set (cand_hist cons)
    # and renormalised per plot, hi = cons (plots with an empty consistent set keep g02b's prior and hi) — recomputed here, compared with g03b's cand_of / cand_iw
    LOC7 = os.environ.get('LOC', 'history')
    if LOC7 in ('history', 'prior'):
        _c7 = C[['pid', 'cid', 'prior', 'hi']].copy()
        if LOC7 == 'history':
            _h7 = pd.read_parquet(f'{D}/cand_hist.parquet')[['cid', 'pid', 'cons']]; _c7 = _c7.merge(_h7, on=['cid', 'pid'], how='left'); _c7['cons'] = _c7.cons.fillna(False).astype(bool)
            _has7 = _c7.groupby('pid').cons.transform('max').astype(bool); _c7['pe'] = np.where(_has7 & ~_c7.cons, 0.0, _c7.prior); _c7['he'] = np.where(_has7, _c7.cons, _c7.hi)
        else: _c7['pe'] = _c7.prior; _c7['he'] = _c7.hi
        _c7['pe'] = _c7.pe / _c7.groupby('pid').pe.transform('sum')
        _m7 = pd.DataFrame({'cid': np.concatenate([v_[0] for v_ in ns_['cand_of'].values()]), 'pm': np.concatenate([v_[1] for v_ in ns_['cand_of'].values()]),
                            'hm': np.concatenate([ns_['cand_iw'][k_][1] for k_ in ns_['cand_of']])}).merge(_c7, on='cid', how='outer', indicator=True)
        _ok7 = (_m7._merge == 'both').all() and np.allclose(_m7.pm, _m7.pe, rtol=1e-9, atol=1e-12) and (_m7.hm.astype(bool) == _m7.he.astype(bool)).all()
        check(f'T7e production location prior (LOC={LOC7}) = g02b prior restricted to the FIA-history-consistent set, renormalised; hi = cons (independent)', _ok7,
              f"{C.pid.nunique()} plots, {len(_m7)} candidates ({_m7._merge.value_counts().to_dict()}); max |Δ prior| {np.nanmax(np.abs(_m7.pm - _m7.pe)):.1e}; hi mismatches {int((_m7.hm.astype(bool) != _m7.he.astype(bool)).sum())}"
              + (f"; plots with a non-empty consistent set {int(_c7.groupby('pid').cons.any().sum())}" if LOC7 == 'history' else ''))
    else: print(f'SKIP  T7e LOC={LOC7}: only the history / prior location priors are re-derived here')
    CAND_OF_PROD = ns_['cand_of']; CAND_HIST_PROD = ns_.get('cand_hist', {})      # (T8h: the production prior; T8 below swaps in g02b's)
    # ======================= T8 FIA-anchored remeasurement pairs (g03b.pair_cand) vs an independent recomputation from FIADB + the spec
    src8 = open(f'{ROOT}/scripts/g03b_train_gafl.py').read(); ns_['V'] = pd.read_parquet(f'{D}/visits.parquet'); _ENV8 = {k: os.environ.get(k) for k in ('PAIR_ANCHOR', 'SEVERITY_ANCHOR', 'LOAD')}
    def run_pairs(anchor, sev, load, cand_of_):
        """exec g03b's pair block (vpair .. pair_cand / pair_cand2) under the given switches and location prior; returns (pair_cand, pair_cand2)"""
        os.environ.update(PAIR_ANCHOR=anchor, SEVERITY_ANCHOR=sev, LOAD=load); ns_['cand_of'] = cand_of_; ns_['cand_hist'] = CAND_HIST_PROD
        exec(compile(src8[src8.index('vpair = '):src8.index("print(f'FIA-anchored pairs")], 'g03b', 'exec'), ns_)      # (vpair applies g03b's forest-share filter fs_ok)
        return dict(ns_['pair_cand']), dict(ns_.get('pair_cand2', {}))
    COF_G2 = {pid: (g.cid.values, g.prior.values) for pid, g in C.groupby('pid')}             # g02b's prior (T8 / T8c test the period rule on g02b's candidates)
    PC, PC2 = run_pairs('period', '0', '0', COF_G2)
    EXv = EX.set_index('vid'); rng8 = np.random.default_rng(8); bad8 = 0; n8 = 0; nev8 = 0
    KIND8 = spec_kind                                                    # (a re-report = the same record kind and event year)
    _VMD = pd.to_datetime(V.md); _vkey = lambda u: (lambda d_: d_.year * 10000 + d_.month * 100 + d_.day)(_VMD.iloc[int(u)])      # (the visit's FIA date)
    FS1 = {tuple(int(x) for x in k_): min(float(f_), 1.0) for k_, f_ in zip(_fs[['st', 'u', 'c', 'p', 'vk']].values, _fs.f1.values)}      # FIADB forested share (COND 1 SUBPPROP) per visit
    FSOK = _fok              # (T4v) independent MIN_FSHARE vetting per visit
    BAtk = TK[TK.N > 0].assign(ba=lambda x: x.N * math.pi / 4 * (x.QMD / 100) ** 2).groupby('vid').ba.sum()
    def pair_spec(v1, v2, span):
        """The spec of a remeasurement history v1 -> v2 (span = the visits in (v1, v2]), from FIADB COND records — independent of g02b / g03b code:
        a record describes the interval ending at the visit that RECORDED it: every FIADB record made at a visit dated in (v1, v2], a table visit or not
        (G47.1: curation-subset, low-share and incomplete visits record events too), except a re-report of an event of the same kind and year recorded at a
        visit <= v1; events of calendar year y act between the visits iff mid-year y lies in [v1, v2], except in a visit's own year when that visit recorded an
        event of that year (then it lies before the visit); records chaining within 2 yr form one cluster (union of evidence classes / windows, max lift),
        whose window is clipped to the acting years (clamped to the nearest one if it falls outside). Blocked ('hid') for the undisturbed rules: a hiding code
        (not NONBLOCK, G37) or a forest-share change >= 0.25 at any visit of the span (G46.2: also against FIA's previous measurement), a gap in FIA's
        measurement chain, an end-to-end forest-share change >= 0.25 (G38). The undisturbed rules apply up to the evidence end only (G36 as a clip, G46.5)."""
        pid = int(V.pid.values[v1]); k_ = tuple(int(x) for x in Pi.loc[pid][key]); y1, y2 = float(V.year.values[v1]), float(V.year.values[v2])
        e = evr[(evr.st == k_[0]) & (evr.u == k_[1]) & (evr.c == k_[2]) & (evr.p == k_[3])]; vk1, vk2 = _vkey(v1), _vkey(v2); vks = {_vkey(u) for u in span}
        typed, hid = [], False
        seq_ = [v1] + list(span)                       # a visit whose FIA previous measurement is not the previous table visit: its codes do not cover the pair
        for u_prev, u in zip(seq_[:-1], seq_[1:]):
            if PREVK.get(k_ + (_vkey(u),)) != _vkey(u_prev) or k_ + (_vkey(u),) in CONV: hid = True
        if abs(FS1.get(k_ + (vk1,), 0.0) - FS1.get(k_ + (vk2,), 0.0)) >= 0.25: hid = True        # (G38) end-to-end forest-share change
        for r in e.itertuples():
            for t_, a_, b_ in (('T', 'TRTCD1', 'TRTYR1'), ('T', 'TRTCD2', 'TRTYR2'), ('T', 'TRTCD3', 'TRTYR3'), ('D', 'DSTRBCD1', 'DSTRBYR1'), ('D', 'DSTRBCD2', 'DSTRBYR2'), ('D', 'DSTRBCD3', 'DSTRBYR3')):
                cd, yr = getattr(r, a_), getattr(r, b_)
                if pd.isna(cd) or cd <= 0: continue
                if r.vk in vks and (t_ == 'T' or int(cd) not in HARMLESS): hid = True
                if (t_, int(cd)) in SPEC and pd.notna(yr) and 2000 <= yr <= 2025: typed.append((r.vk, t_, int(cd), int(yr)))
        old = {(KIND8(t_, cd), yr) for vk, t_, cd, yr in typed if vk <= vk1}
        recs = sorted({(yr, t_, cd) for vk, t_, cd, yr in typed if vk1 < vk <= vk2 and (KIND8(t_, cd), yr) not in old})
        rs = any(vk == vk1 and yr == math.floor(y1) for vk, _, _, yr in typed); re_ = any(vk == vk2 and yr == math.floor(y2) for vk, _, _, yr in typed)
        ra = any(yr == math.floor(y1) for yr, _, _ in recs)       # a new record of the interval dated in v1's year: the event lies after v1
        f1, f2 = math.floor(y1), math.floor(y2); a = f1 if (not rs and (y1 <= f1 + 0.5 or ra)) else f1 + 1; b = f2 if (y2 >= f2 + 0.5 or re_) else f2 - 1
        cl = []
        for yr, t_, cd in recs:
            cols_, (lo, hi), lr = SPEC[(t_, cd)]; cs_ = {HH if kk == 'H' else ix[kk] for kk in cols_}
            if cl and yr - cl[-1][4] <= 2: a0 = cl[-1]; cl[-1] = (a0[0] | cs_, min(a0[1], yr + lo), max(a0[2], yr + hi), max(a0[3], lr), yr)
            else: cl.append((cs_, yr + lo, yr + hi, lr, yr))
        clc = []
        if b >= a:
            for cs_, lo, hi, lr, _ in cl:
                lo2, hi2 = max(lo, a), min(hi, b)
                if hi2 < lo2: lo2 = hi2 = (a if hi < a else b)
                clc.append((cs_, lo2, hi2, lr))
        g = C[C.pid == pid]; Xg = np.asarray(CF[g.cid.values], dtype=np.float32)[:, 128:128 + NY * (NC + 2)].reshape(len(g), NY, NC + 2)
        return dict(g=g, Xg=Xg, cl=clc, det=any(c[3] >= 4 for c in clc), hid=hid, a=a, b=b, pid=pid)
    def _srw(Xg, lo, hi):
        """stand-replacing evidence (harvest class or Hansen loss on >= half the footprint) in calendar years [lo, hi] clipped to the evidence end (G36)"""
        a0, b0 = max(lo - 2000, 0), min(min(hi, EVID_END) - 2000, NY - 1)
        return (np.maximum(Xg[:, a0:b0 + 1, ix['harvest']], Xg[:, a0:b0 + 1, HH]) >= 0.5).any(1) if b0 >= a0 else np.zeros(len(Xg), bool)
    _g2p = dict(zip(C.cid.values, C.prior.values))                     # g02b's prior (pr0)
    _lp8 = dict(zip(_c7.cid.values, _c7.pe.values)) if LOC7 in ('history', 'prior') else None      # T7e's independent production location prior
    _cons8 = dict(zip(_c7.cid.values, _c7.cons.values)) if LOC7 == 'history' else {}
    def spec_set(v1, v2, span, sev=False, both=False, prior=None):
        """candidate set + weights of the history v1 -> v2 by the spec (pair_spec):
          - every detectable cluster (lift >= 4) needs compatible evidence in its window;
          - (sev: training) the span's MIN BA ratio to v1 over the visits passing MIN_FSHARE (G35; v2 always), if BA(v1) > 1 m2/ha: < 0.3 -> only candidates
            with stand-replacing evidence in a detectable cluster's window; > 0.6 -> only candidates without; a min ratio < 0.3 also blocks the undisturbed rules;
          - unblocked (pair_spec hid) and no measured collapse: no stand-replacing evidence in the acting years up to the evidence end (G46.5: the part after
            it is unconstrained), and with no detectable record no evidence at all there except the period's weak events inside their windows;
          - (both) ∩ the plot's FIA-history-consistent set (cand_hist cons) if that leaves a candidate, else the period set;
          a constraint that would leave no candidate is skipped. Weights: the location prior restricted to the set (else g02b's prior there, else uniform)."""
        q = pair_spec(v1, v2, span); g, Xg, m = q['g'], q['Xg'], np.ones(len(q['g']), bool)
        nar = lambda m_, new: (m_ & new) if (m_ & new).any() else m_
        dcl = [c for c in q['cl'] if c[3] >= 4]
        for cs_, lo, hi, lr in dcl:
            a0, b0 = max(lo - 2000, 0), min(hi - 2000, NY - 1)
            if b0 >= a0: m = nar(m, Xg[:, a0:b0 + 1][:, :, sorted(cs_)].sum(-1).sum(-1) > 0)
        r_ = (min(float(BAtk.get(u, 0.0)) for u in span if FSOK[u] or u == v2) / float(BAtk[v1])) if float(BAtk.get(v1, 0.0)) > 1.0 else None
        loss = False
        if sev and dcl and r_ is not None:
            srw = np.zeros(len(g), bool)
            for cs_, lo, hi, lr in dcl: srw |= _srw(Xg, lo, hi)
            if r_ < 0.3: m = nar(m, srw); loss = True
            elif r_ > 0.6: m = nar(m, ~srw)
        collapse = sev and r_ is not None and r_ < 0.3
        if not q['hid'] and not loss and not collapse and q['b'] >= q['a']:          # (G46.5: windows clipped to EVID_END by _srw / below)
            m = nar(m, ~_srw(Xg, q['a'], q['b']))
            if not q['det']:
                Xe = Xg.copy()
                for cs_, lo, hi, lr in q['cl']: Xe[:, max(lo - 2000, 0):min(hi - 2000, NY - 1) + 1, sorted(cs_)] = 0
                a0, b0 = max(q['a'] - 2000, 0), min(min(q['b'], EVID_END) - 2000, NY - 1)
                m = nar(m, ~((Xe[:, a0:b0 + 1, :NC].sum(-1) + Xe[:, a0:b0 + 1, HH]) > 0).any(1))
        if both:
            cons = np.array([bool(_cons8.get(c_, False)) for c_ in g.cid.values])
            if (m & cons).any(): m = m & cons
        cids = g.cid.values[m]; pm = prior if prior is not None else _g2p
        w_ = np.array([pm.get(c_, 0.0) for c_ in cids]);
        if w_.sum() <= 0: w_ = np.array([_g2p[c_] for c_ in cids])
        w_ = w_ / w_.sum() if w_.sum() > 0 else np.full(len(cids), 1.0 / len(cids))
        return dict(zip(cids.tolist(), w_.tolist())), q['det'], r_
    def same(dexp, got, wtol=1e-6):
        """set equality and weight agreement (relative, max) of a spec dict {cid: w} and g03b's (cid, w)"""
        dg = dict(zip(np.asarray(got[0]).tolist(), np.asarray(got[1], dtype=float).tolist()))
        return set(dexp) == set(dg), (max(abs(dexp[c_] - dg[c_]) / max(dexp[c_], 1e-12) for c_ in dexp) if set(dexp) == set(dg) and dexp else 0.0)
    vs_of8 = V.assign(vid=np.arange(len(V))).sort_values('year').groupby('pid').vid.apply(list).to_dict()
    _span = lambda v1, v2: (lambda sq: sq[sq.index(v1) + 1:sq.index(v2) + 1])(vs_of8[int(V.pid.values[v1])])
    # T8p the model pairs are the consecutive table pairs whose both visits pass MIN_FSHARE (independent share, T4v); (G46.7) the anchored pairs are those
    # of plots with candidates that are NOT converted (independent T1n rule: the pair code has no FIA-only path, so converted plots' pairs are dropped)
    _vp8 = set(ns_['vpair'].items()); _pe8 = {(int(a), int(b_)) for a, b_ in zip(PR.v1.values, PR.v2.values) if FSOK[a] and FSOK[b_]}
    _cv8 = {pid for pid in COF_G2 if tuple(int(x) for x in Pi.loc[pid][key]) in CONVP}; _pc8e = {a for a, _ in _pe8 if int(V.pid.values[a]) in COF_G2 and int(V.pid.values[a]) not in _cv8}
    _pc8c = {a for a in PC if int(V.pid.values[a]) in _cv8} | {k_[0] for k_ in PC2 if int(V.pid.values[k_[0]]) in _cv8}   # (consecutive or multi-horizon)
    check('T8p model pairs (g03b.vpair) = consecutive table pairs with both visits passing MIN_FSHARE (independent FIADB share, T4v); anchored pairs = those of non-converted plots with candidates (G46.7)',
          _vp8 == _pe8 and set(PC) == _pc8e and not _pc8c,
          f'{len(_vp8)} pairs, expected {len(_pe8)}, differing {len(_vp8 ^ _pe8)}; anchored {len(PC)}, expected {len(_pc8e)}, of converted plots {len(_pc8c)} '
          f'({len(_cv8)} converted plots with candidates), missing {len(_pc8e - set(PC))}')
    def _mis8(tag, v1, v2, es, got, extra=''):
        if len(_ex8.setdefault(tag, [])) < 12: _ex8[tag].append((int(v1), int(v2), len(es), len(got[0]), extra))
        print(f'   {tag} mismatch: pair {int(v1)} -> {int(v2)}: spec {len(es)} candidates, g03b {len(got[0])}{extra}', flush=True)
    _ex8 = {}
    _s8 = [x for x in FORCE if x in PC] + [x for x in rng8.choice(list(PC), min(300, len(PC)), replace=False) if x not in FORCE]
    for v1 in _s8:
        es, det, _ = spec_set(v1, ns_['vpair'][v1], [ns_['vpair'][v1]]); nev8 += det
        n8 += 1; ok_, we_ = same(es, PC[v1]); bad8 += (not ok_) or we_ > 1e-6
        if (not ok_) or we_ > 1e-6: _mis8('T8', v1, ns_['vpair'][v1], es, PC[v1], f', sets equal {ok_}, max rel weight diff {we_:.1e}')
        elif v1 in FORCE: print(f'   T8 forced pair {int(v1)} -> {int(ns_["vpair"][v1])}: agrees ({len(es)} candidates)', flush=True)
    check('T8 FIA-anchored pair histories (g03b.pair_cand, PAIR_ANCHOR=period, records only) = independent recomputation from FIADB records + spec (sets and weights)', n8 > 0 and bad8 == 0,
          f'{n8} pairs ({nev8} with detectable events; forced {sum(1 for x in FORCE if x in PC)}), mismatches {bad8}: {_ex8.get("T8", [])}')
    # T8e the record assignment itself: records made AT v2 whose year is v2's own year belong to (v1, v2] (the mid-year rule dropped first-half visits'
    # records: audit r5 I3), and the flow applies their year's evidence before v2 (Flow re) — counted on the sample, non-vacuous
    n8e = 0
    for v1 in np.random.default_rng(84).permutation(list(PC))[:4000]:   # (random, not key order: r12 M-3)
        v2 = ns_['vpair'][v1]; y2 = float(V.year.values[v2])
        if y2 - math.floor(y2) < 0.5 and bool(ns_['VREC'][v2]):
            q = pair_spec(v1, v2, [v2]); n8e += 1
            if not (q['b'] == math.floor(y2) and q['cl']): n8e = -10 ** 9; break
    check('T8e first-half visits that recorded an event of their own year: the interval (v1, v2] holds the record and acts through v2\'s year', n8e > 0, f'{max(n8e, 0)} such pairs among 4000 random pairs')
    # T8f g03b's record assignment and acting years (recs_in / act_span / pair_clusters, used by every stage) = the spec (pair_spec), pair by pair
    bad8f = 0; n8f = 0; ex8f = []
    for v1 in [x for x in FORCE if x in PC] + [x for x in rng8.choice(list(PC), min(400, len(PC)), replace=False) if x not in FORCE]:
        v2 = ns_['vpair'][v1]; q = pair_spec(v1, v2, [v2]); y1, y2 = float(V.year.values[v1]), float(V.year.values[v2])
        ya, yb = ns_['act_span'](y1, y2, bool(ns_['VREC'][v1]), bool(ns_['VREC'][v2]), ns_['pair_ra'](v1, v2)); pc_ = ns_['pair_clusters'](int(V.pid.values[v1]), y1, y2, ya, yb)
        mine = sorted((c[1], c[2], c[3], tuple(sorted(c[0]))) for c in q['cl']); theirs = sorted((c['lo'], c['hi'], int(c['lr']), tuple(sorted(c['cols']))) for c in pc_)
        n8f += 1; bad_ = (ya, yb) != (q['a'], q['b']) or mine != theirs; bad8f += bad_
        if bad_ or v1 in FORCE:
            print(f"   T8f {'mismatch' if bad_ else 'forced pair agrees'}: pair {int(v1)} -> {int(v2)}: acting years spec {(q['a'], q['b'])} g03b {(ya, yb)}; clusters spec {mine} g03b {theirs}", flush=True)
            if bad_ and len(ex8f) < 12: ex8f.append((int(v1), int(v2)))
    check('T8f g03b act_span / recs_in / pair_clusters (acting years, clipped event clusters of the interval) = the independent spec', n8f > 0 and bad8f == 0, f'{n8f} pairs, mismatches {bad8f}: {ex8f}')
    # T8g VRA (MIL / Q-decode / B flows from a visit): a LATER visit recorded a new (not re-reported) typed event dated in the visit's own calendar year —
    # recomputed from FIADB COND records (r11 M-5)
    _vs8 = np.random.default_rng(81).choice(len(V), min(3000, len(V)), replace=False); bad8g = 0; npos8g = 0
    for v in _vs8:
        pid = int(V.pid.values[v]); k_ = tuple(int(x) for x in Pi.loc[pid][key]); yv = float(V.year.values[v]); vkv = _vkey(int(v))
        e = evr[(evr.st == k_[0]) & (evr.u == k_[1]) & (evr.c == k_[2]) & (evr.p == k_[3])]; typed = []
        for r in e.itertuples():
            for t_, a_, b_ in (('T', 'TRTCD1', 'TRTYR1'), ('T', 'TRTCD2', 'TRTYR2'), ('T', 'TRTCD3', 'TRTYR3'), ('D', 'DSTRBCD1', 'DSTRBYR1'), ('D', 'DSTRBCD2', 'DSTRBYR2'), ('D', 'DSTRBCD3', 'DSTRBYR3')):
                cd, yr = getattr(r, a_), getattr(r, b_)
                if pd.notna(cd) and cd > 0 and (t_, int(cd)) in SPEC and pd.notna(yr) and 2000 <= yr <= 2025: typed.append((r.vk, KIND8(t_, int(cd)), int(yr)))
        old = {(kd, yr) for vk, kd, yr in typed if vk <= vkv}
        exp_ = any(vk > vkv and (kd, yr) not in old and yr == math.floor(yv) for vk, kd, yr in typed)
        npos8g += exp_; bad8g += exp_ != bool(ns_['VRA'][int(v)])
    check('T8g VRA = a later visit\'s new record dated in the visit\'s year (independent, FIADB COND)', bad8g == 0 and npos8g > 0, f'{len(_vs8)} visits, positives {npos8g}, mismatches {bad8g}')
    # T8c multi-horizon pairs (PAIR_ALL): the same independent recomputation over the span (a hiding code at ANY visit in between blocks the undisturbed rules)
    PAIR_ALL8 = os.environ.get('PAIR_ALL', '0') == '1'
    multi = [k for k in PC2 if k[0] in ns_['vpair'] and ns_['vpair'][k[0]] != k[1]]
    if multi:
        bad8c = 0; n8c = 0
        for i8 in rng8.choice(len(multi), min(200, len(multi)), replace=False):
            v1, v2 = multi[i8]; es, _, _ = spec_set(v1, v2, _span(v1, v2)); n8c += 1; ok_, we_ = same(es, PC2[(v1, v2)]); bad8c += (not ok_) or we_ > 1e-6
            if (not ok_) or we_ > 1e-6: _mis8('T8c', v1, v2, es, PC2[(v1, v2)], f', sets equal {ok_}, max rel weight diff {we_:.1e}')
        check('T8c multi-horizon pair histories (g03b.pair_cand2, period, records only) = independent recomputation over the span (sets and weights)', bad8c == 0,
              f'{n8c} pairs, mismatches {bad8c}: {_ex8.get("T8c", [])}')
    elif PAIR_ALL8: check('T8c multi-horizon pair histories exist with PAIR_ALL=1', False, 'PAIR_ALL=1 but g03b built no multi-horizon pair')
    else: print('SKIP  T8c no multi-horizon pairs (PAIR_ALL off)')
    # T8b (a) evaluation safety, under the EVALUATION config (the configured PAIR_ANCHOR and SEVERITY_ANCHOR, LOAD=1, the production location prior): the
    # severity rule must be OFF, i.e. the sets and weights = the records-only sets (SEVERITY_ANCHOR 0) of the same config; compared on a seeded random sample
    # of 3000 pairs (+ 1000 multi-horizon). (G47.6) the switches are read from the environment; values the spec does not cover FAIL T8b / T8h
    ANC8, SEV8 = _ENV8['PAIR_ANCHOR'] or 'both', _ENV8['SEVERITY_ANCHOR'] or '1'; OK8 = ANC8 in ('both', 'period') and SEV8 in ('0', '1')
    MSG8 = '' if OK8 else f'unsupported configuration PAIR_ANCHOR={ANC8!r} SEVERITY_ANCHOR={SEV8!r} (the spec covers both / period and 0 / 1; G47.6)'
    if not OK8: ANC8, SEV8 = 'both', '1'                                 # (computed for diagnosis only; T8b / T8h FAIL)
    PCr, PC2r = run_pairs(ANC8, '0', '0', CAND_OF_PROD); PCl, PC2l = run_pairs(ANC8, SEV8, '1', CAND_OF_PROD)
    _k8a = list(PCr); _k8a = [_k8a[i] for i in np.random.default_rng(87).permutation(len(_k8a))[:3000]]; _k8m = list(PC2r); _k8m = [_k8m[i] for i in np.random.default_rng(88).permutation(len(_k8m))[:1000]]
    def _eq8(a_, b_): return set(np.asarray(a_[0]).tolist()) == set(np.asarray(b_[0]).tolist()) and np.allclose(np.asarray(a_[1])[np.argsort(a_[0])], np.asarray(b_[1])[np.argsort(b_[0])])
    off_ok = set(PCr) == set(PCl) and set(PC2r) == set(PC2l) and all(_eq8(PCr[k], PCl[k]) for k in _k8a) and all(_eq8(PC2r[k], PC2l[k]) for k in _k8m)
    # (b) training (LOAD=0, the configured switches, the production location prior): with SEVERITY_ANCHOR 1 every stand-replacing pair (detectable FIA event,
    # min BA ratio < 0.3) whose consistent set CONTAINS a candidate with stand-replacing evidence in the window uses only such candidates (non-vacuous:
    # pairs without any are not counted); with SEVERITY_ANCHOR 0 part (b) does not apply
    PCs, PC2s = run_pairs(ANC8, SEV8, '0', CAND_OF_PROD)
    for k_, v_ in _ENV8.items():
        if v_ is None: os.environ.pop(k_, None)
        else: os.environ[k_] = v_
    def strong_set(v1, v2, span):
        """event-consistent candidates with stand-replacing evidence (harvest class or Hansen loss on >= half the footprint, <= the evidence end) in a detectable cluster's window"""
        q = pair_spec(v1, v2, span); cl_ = [c for c in q['cl'] if c[3] >= 4]
        if not cl_: return None
        Xp, allc = q['Xg'], q['g'].cid.values; w_ = np.zeros(len(allc), bool); m_ev = np.ones(len(allc), bool)
        for cs_, lo, hi, lr in cl_:
            lo_, hi_ = max(lo - 2000, 0), min(hi - 2000, NY - 1) + 1
            if hi_ > lo_:
                w_ |= _srw(Xp, lo, hi)
                nm_ = m_ev & (Xp[:, lo_:hi_][:, :, sorted(cs_)].sum(-1) > 0).any(1)
                if nm_.any(): m_ev = nm_
        return set(allc[w_ & m_ev])
    okr = 0; nrep = 0
    _vp_ = list(ns_['vpair'].items())
    for v1, v2 in [_vp_[i] for i in np.random.default_rng(85).permutation(len(_vp_))]:   # (random order: r12 M-3)
        if v1 not in PCs or BAtk.get(v1, 0) <= 1.0 or BAtk.get(v2, 0) / BAtk.get(v1, 1) >= 0.3: continue
        strong = strong_set(v1, v2, [v2])
        if not strong: continue
        nrep += 1; okr += set(PCs[v1][0].tolist()) <= strong       # selected ⊆ event-consistent candidates with stand-replacing evidence
        if nrep >= 300: break
    check(f'T8b severity anchoring (PAIR_ANCHOR {ANC8}, SEVERITY_ANCHOR {SEV8}): OFF under the evaluation config (LOAD=1: sets and weights = the records-only ones); ON in training: stand-replacing pairs use only candidates with stand-replacing evidence',
          OK8 and off_ok and (SEV8 == '0' or (okr == nrep and nrep > 0)),
          f'evaluation sets unchanged: {off_ok} ({len(_k8a)} + {len(_k8m)} random pairs compared); ' + (f'{nrep} stand-replacing pairs with strong candidates checked, violations {nrep - okr}' if SEV8 == '1' else '(b) n/a: SEVERITY_ANCHOR 0') + (f'; {MSG8}' if MSG8 else ''))
    # T8d multi-horizon severity (audit r4 I2): a span whose BA fell below 0.3 of v1 at ANY visit in between that passes MIN_FSHARE (G35; even if it regrew by
    # the end) is stand-replacing
    okd = 0; nd = 0
    if PAIR_ALL8 and SEV8 == '1':
        _pc2_ = list(PC2s.items())
        for (v1, v2), (cs2, _) in [_pc2_[i] for i in np.random.default_rng(86).permutation(len(_pc2_))]:   # (random order: r12 M-3)
            if ns_['vpair'].get(v1) == v2 or BAtk.get(v1, 0) <= 1.0: continue
            span = _span(v1, v2)
            if min(BAtk.get(u, 0.0) for u in span if FSOK[u]) / BAtk.get(v1) >= 0.3 or BAtk.get(v2, 0.0) / BAtk.get(v1) < 0.3: continue   # only regrown spans: end ratio misleads
            strong = strong_set(v1, v2, span)
            if not strong: continue
            nd += 1; okd += set(np.asarray(cs2).tolist()) <= strong
            if nd >= 300: break
        check('T8d multi-horizon severity: spans whose BA fell below 0.3 of v1 at an intermediate visit passing MIN_FSHARE (regrown by the end) use only stand-replacing candidates',
              okd == nd and nd > 0, f'{nd} regrown stand-replacing spans checked, violations {nd - okd}' + ('' if nd else ' (none in this table: vacuous)'))
    else: print(f'SKIP  T8d PAIR_ALL {int(PAIR_ALL8)}, SEVERITY_ANCHOR {SEV8} (needs multi-horizon pairs and the severity rule)')
    # T8h (r13 E-F2) the PRODUCTION pair configuration (the configured PAIR_ANCHOR / SEVERITY_ANCHOR, training, the production location prior) against the
    # full spec (spec_set: event rule, severity on the min BA ratio over the span's MIN_FSHARE visits, collapse block, undisturbed rules to the evidence end,
    # ∩ cand_hist cons with its fallback when both, weights = the location prior restricted to the set with g02b's prior as fallback) — sets AND weights.
    # (G47.6) real minima: >= 300 consecutive pairs, >= 200 multi-horizon pairs (PAIR_ALL=1), >= 20 with a detectable cluster, >= 20 with a severity rule (SEVERITY_ANCHOR 1)
    if _lp8 is None: check('T8h PRODUCTION pair sets and weights = the independent spec', False, f'LOC={LOC7}: the location prior of this LOC is not re-derived (T7e)')
    else:
        _kc = list(PCs); _kc = [x for x in FORCE if x in PCs] + [x for x in (_kc[i] for i in np.random.default_rng(89).permutation(len(_kc))[:300]) if x not in FORCE]
        _km = [k for k in PC2s if ns_['vpair'].get(k[0]) != k[1]]; _km = [_km[i] for i in np.random.default_rng(90).permutation(len(_km))[:200]]
        bh = {'c_set': 0, 'c_w': 0, 'm_set': 0, 'm_w': 0}; wmax = 0.0; nsev = 0; ndet = 0; ex8h = []; sv8, bo8 = SEV8 == '1', ANC8 == 'both'
        for v1 in _kc:
            v2 = ns_['vpair'][v1]; es, det, r_ = spec_set(v1, v2, [v2], sev=sv8, both=bo8, prior=_lp8); ok_, we_ = same(es, PCs[v1]); nsev += sv8 and det and r_ is not None and (r_ < 0.3 or r_ > 0.6); ndet += det
            bh['c_set'] += not ok_; bh['c_w'] += ok_ and we_ > 1e-5; wmax = max(wmax, we_)
            if not ok_ or we_ > 1e-5: _mis8('T8h', v1, v2, es, PCs[v1], f', sets equal {ok_}, max rel weight diff {we_:.1e}')
            elif v1 in FORCE: print(f'   T8h forced pair {int(v1)} -> {int(v2)}: agrees ({len(es)} candidates)', flush=True)
        for v1, v2 in _km:
            es, det, r_ = spec_set(v1, v2, _span(v1, v2), sev=sv8, both=bo8, prior=_lp8); ok_, we_ = same(es, PC2s[(v1, v2)]); nsev += sv8 and det and r_ is not None and (r_ < 0.3 or r_ > 0.6); ndet += det
            bh['m_set'] += not ok_; bh['m_w'] += ok_ and we_ > 1e-5; wmax = max(wmax, we_)
            if not ok_ or we_ > 1e-5: _mis8('T8h', v1, v2, es, PC2s[(v1, v2)], f', multi-horizon, sets equal {ok_}, max rel weight diff {we_:.1e}')
        _min8 = {'consecutive >= 300': len(_kc) >= 300, 'multi-horizon >= 200 (PAIR_ALL=1)': len(_km) >= 200 or not PAIR_ALL8, 'detectable >= 20': ndet >= 20,
                 'severity rule >= 20': nsev >= 20 or not sv8}
        check(f'T8h PRODUCTION pair sets and weights (PAIR_ANCHOR {ANC8}, SEVERITY_ANCHOR {SEV8}, training; g03b pair_cand / pair_cand2) = the independent spec',
              OK8 and all(_min8.values()) and not any(bh.values()),
              f'{len(_kc)} consecutive + {len(_km)} multi-horizon pairs ({ndet} with a detectable cluster, {nsev} with a severity rule applied); minima unmet {[k for k, v in _min8.items() if not v]}; '
              f'mismatches {bh}; max rel weight diff {wmax:.1e}; mismatching pairs {_ex8.get("T8h", [])}' + (f'; {MSG8}' if MSG8 else ''))
    if done('T8'): return RES
    # ======================= T9 plot-level location posterior (g02f) recomputed from the specification with the fitted parameters
    if os.path.exists(f'{D}/cand_prior.parquet'):
        def bcls(m, iw, inn):      # background per distance class (inner 800 m disk / ring), clipped as specified
            out = np.zeros(len(m))
            for cl_ in (inn, ~inn):
                if cl_.any(): out[cl_] = np.clip((iw[cl_] * m[cl_]).sum() / iw[cl_].sum(), 0.5 / iw[cl_].sum(), 1 - 1e-3)
            return out
        th = json.load(open(f'{D}/location_em.json'))['theta']; CP = pd.read_parquet(f'{D}/cand_prior.parquet').set_index('cid'); GRPV = {8: 'strong', 4: 'medium', 3: 'weak'}; err9 = []
        for pid, g in Cs.groupby('pid'):
            Xg = np.asarray(CF[g.cid.values], dtype=np.float32)[:, 128:128 + NY * (NC + 2)].reshape(len(g), NY, NC + 2); iw = g.incl_w.values.astype(float); inn = g.dist_m.values <= 800
            lw = np.log(np.where(inn, th['pi'] / iw[inn].sum(), (1 - th['pi']) / iw[~inn].sum()) * iw) if inn.any() and (~inn).any() else np.log(iw / iw.sum())
            cl, per_ = REC[pid]
            for c_ in cl:
                cs_, lo, hi, lr = c_[0], c_[1], c_[2], c_[3]; a0, b0 = max(lo - 2000, 0), min(hi - 2000, NY - 1)
                if b0 < a0: continue
                m = Xg[:, a0:b0 + 1][:, :, sorted(cs_)].sum(-1).sum(-1) > 0; qq = th['q'][GRPV[int(lr)]]
                if 0 < (iw * m).sum() / iw.sum() < 1: bf = bcls(m, iw, inn); lw = lw + np.where(m, np.log(qq / bf), np.log((1 - qq) / (1 - bf)))
            srv = np.maximum(Xg[:, :, ix['harvest']], Xg[:, :, HH]) >= 0.5
            for a0, b0 in per_:
                a0, b0 = max(a0 - 2000, 0), min(b0 - 2000, NY - 1)
                if b0 < a0: continue
                m = srv[:, a0:b0 + 1].any(1)
                if 0 < (iw * m).sum() / iw.sum() < 1: bf = bcls(m, iw, inn); lw = lw + np.where(m, np.log(th['f'] / bf), np.log((1 - th['f']) / (1 - bf)))
            lin = np.logaddexp.reduce(lw); post = np.exp(lw - lin); rin = 1 - np.exp(np.log(th['s']) - np.logaddexp(np.log(1 - th['s']) + lin, np.log(th['s'])))
            p = rin * post + (1 - rin) * iw / iw.sum(); err9.append(np.abs(p - CP.loc[g.cid.values].prior_loc.values).max() / p.max())
        check('T9 plot-level location posterior (g02f) = recomputation from the spec + fitted EM parameters', max(err9) < 1e-3, f'{len(err9)} plots, max rel err {max(err9):.1e}; theta q {th["q"]} f {th["f"]:.3f} pi {th["pi"]:.3f} s {th["s"]:.3f}')
    else: print('SKIP  T9 no cand_prior.parquet (g02f location EM not run for this table; LOC=plot unused)')
    # ======================= T10 FIA-history-consistent candidates (g02g) recomputed from the spec (T3f's independent records)
    if os.path.exists(f'{D}/cand_hist.parquet'):
        CH = pd.read_parquet(f'{D}/cand_hist.parquet').set_index('cid'); bad10 = 0; ne10 = 0; nc10 = 0
        for pid, g in Cs.groupby('pid'):
            Xg = np.asarray(CF[g.cid.values], dtype=np.float32)[:, 128:128 + NY * (NC + 2)].reshape(len(g), NY, NC + 2); cl, per_ = REC[pid]; m = np.ones(len(g), bool)
            srv = np.maximum(Xg[:, :, ix['harvest']], Xg[:, :, HH]) >= 0.5; cons10 = []
            for c_ in cl:
                if c_[3] < 4: continue
                a0, b0 = max(c_[1] - 2000, 0), min(c_[2] - 2000, NY - 1)
                if b0 >= a0: cons10.append((c_[1], Xg[:, a0:b0 + 1][:, :, sorted(c_[0])].sum(-1).sum(-1) > 0))
            for a0, b0 in per_:
                a1, b1 = max(a0 - 2000, 0), min(b0 - 2000, NY - 1)
                if b1 >= a1: cons10.append((a0, ~srv[:, a1:b1 + 1].any(1)))
            for _, x in sorted(cons10, key=lambda t: t[0]):          # time order; a constraint no remaining footprint satisfies is skipped (spec)
                if (m & x).any(): m = m & x
            bad10 += int((CH.loc[g.cid.values].cons.values != m).sum() > 0); ne10 += int(not m.any()); nc10 += int(m.any() and not m.all())
        check('T10 FIA-history-consistent candidates (g02g) = independent recomputation (detectable clusters + undisturbed periods, time order, unsatisfiable skipped); some plots constrained',
              bad10 == 0 and nc10 > 0, f'{Cs.pid.nunique()} plots, constrained (a strict non-empty subset) {nc10}, mismatching plots {bad10}, empty sets {ne10}')
    else: print('SKIP  T10 no cand_hist.parquet (g02g not run for this table)')
    # ======================= T11 Forest Explorer export (g07): GEN_* = stand map x unit conversions
    FEJ = os.environ.get('FE_GEOJSON', os.path.join(os.path.dirname(ROOT), 'fe_pan/apps/forest-explorer/public/stands/baker_fl.geojson'))   # genveg_pipeline: config fe.geojson
    if os.environ.get('GV_REQUIRE_MAP') == '1':          # after the map / fe stages the map checks must RUN, not skip (r11 M-5)
        _need = [FEJ, f'{D}/stand_veg_v3.parquet', f'{D}/stand_cohorts_v3.parquet', f'{D}/bayes_eval.json']
        check('T11z map-stage outputs present (T11 / T11b / T11c run)', all(os.path.exists(f) for f in _need), f'missing {[os.path.basename(f) for f in _need if not os.path.exists(f)]}')
    _MAPCHK = os.environ.get('GV_REQUIRE_MAP') != '0'       # vtab (before any map exists) skips the map checks entirely (r12 M-1)
    if _MAPCHK and os.path.exists(FEJ) and os.path.exists(f'{D}/stand_veg_v3.parquet'):
        SV = pd.read_parquet(f'{D}/stand_veg_v3.parquet').set_index('stand_id'); fe = json.load(open(FEJ))['features']; e11 = []; n11 = 0
        for f in fe:
            p = f['properties']
            if 'GEN_BA' not in p: continue
            r = SV.loc[int(p['stand_id'])]; n11 += 1
            e11.append(max(abs(p['GEN_BA'] - r.ba / 0.229568), abs(p['GEN_TPA'] - r.tph / 2.47105), abs(p['GEN_QMD'] - r.qmd / 2.54) * 10, abs(p['GEN_BA_SD'] - r.ba_sd / 0.229568)))
            if p['GEN_DOM'] != r.dom_key: e11.append(99)
        check('T11 Forest Explorer stand layer: GEN_BA / TPA / QMD / BA_SD / DOM = stand map (g05) x conversions (m2/ha->ft2/ac, /ha->/ac, cm->in)', n11 > 0 and max(e11) < 0.11,
              f'{n11} stands, max abs diff {max(e11) if e11 else float("nan"):.3f} (rounding 0.05 / 0.005)')
        # T11g (r13 coverage, g07): the FE layer carries GEN_* exactly for its stands that are mapped; <layer>_cohorts.json = stand_cohorts_v3.parquet of those stands
        _fid = {int(f['properties']['stand_id']) for f in fe}; _fg = {int(f['properties']['stand_id']) for f in fe if 'GEN_BA' in f['properties']}; _sv11 = set(int(x) for x in SV.index)
        _cj = FEJ[:-len('.geojson')] + '_cohorts.json' if FEJ.endswith('.geojson') else FEJ + '_cohorts.json'; e11g = None; n11g = 0
        if os.path.exists(_cj) and os.path.exists(f'{D}/stand_cohorts_v3.parquet'):
            CJ = json.load(open(_cj))['stands']; CC = pd.read_parquet(f'{D}/stand_cohorts_v3.parquet'); CC = CC[CC.stand_id.isin(_fid)]; e11g = 0.0
            _exp11 = {str(int(k)): sorted((str(r.key), round(float(r.age), 1), round(float(r.tph), 2), round(float(r.qmd_cm), 2), round(float(r.ht_m), 2), round(float(r.cr), 3)) for r in grp.itertuples())
                      for k, grp in CC.groupby('stand_id')}
            e11g = len(set(_exp11) ^ set(CJ)) + sum(sorted(tuple(x) for x in CJ[k]) != v for k, v in _exp11.items() if k in CJ); n11g = len(_exp11)
        check('T11g FE layer: GEN_* exactly on its mapped stands; the layer\'s cohorts JSON = the stand cohort table (g07)', _fg == (_fid & _sv11) and e11g == 0,
              f'{len(_fid)} layer stands, with GEN_* {len(_fg)}, mapped {len(_fid & _sv11)}; cohorts JSON {"missing" if e11g is None else f"{n11g} stands, mismatching {e11g}"}')

    # T11c g05 maps with the start year g04 GATED (bayes_eval.json map_y0), and a non-2021 year only with a passed pre-registered gate
    if _MAPCHK and os.path.exists(f'{D}/bayes_eval.json'):
        be = json.load(open(f'{D}/bayes_eval.json')); y0b = be.get('map_y0', 2021.0); gt = be.get('map_y0_gate', {})
        _svp = f'{D}/stand_veg_v3.parquet'; _sy = pd.read_parquet(_svp, columns=['map_y0']).map_y0.unique().tolist() if os.path.exists(_svp) and 'map_y0' in pd.read_parquet(_svp).columns else None
        _sc = pd.read_parquet(_svp)[['map_cal_tph', 'map_cal_ba']].drop_duplicates().values.tolist() if _sy is not None and 'map_cal_tph' in pd.read_parquet(_svp).columns else None
        _bc = be.get('calibration_log_factors_mapping_path'); _cal_ok = _sc is not None and _bc is not None and len(_sc) == 1 and np.allclose(_sc[0], _bc[:2], atol=1e-9)
        check('T11c mapping start year: the production stand map (g05) was mapped from the start year g04 gated; a year other than 2021 only if the gate passed',
              _sy == [y0b] and _cal_ok and (y0b == 2021.0 or gt.get('passed') is True), f'map_y0 {y0b}, stand map {_sy}, gate passed {gt.get("passed")}; calibration factors map {_sc} vs g04 {_bc[:2] if _bc else None}')
        if 'err_requested' in gt or float(gt.get('requested', 2021.0)) != 2021.0:      # T11d the gate decision re-derived from its recorded numbers by the pre-registered rule (G5 / G19 / G24 / G33), not trusted (r12 M-5)
            _nk = ('err_requested', 'err_2021', 'err_null', 'redraw_corr_BA', 'shuffled_landscape_corr_BA', 'redraw_corr_BA_2021', 'shuffled_landscape_corr_BA_2021')
            _miss = [k for k in _nk if k not in gt]
            if _miss: check('T11d start-year gate decision = the pre-registered rule applied to its recorded numbers', False, f'bayes_eval.json map_y0_gate lacks {_miss} (older g04: rerun downscale)')   # (r13 F#10)
            else:
                _dep, _dep0 = gt['redraw_corr_BA'] - gt['shuffled_landscape_corr_BA'], gt['redraw_corr_BA_2021'] - gt['shuffled_landscape_corr_BA_2021']
                _exp = all(gt['err_requested'][k] <= gt['err_2021'][k] and gt['err_requested'][k] <= gt['err_null'][k] for k in ('TPH', 'BA')) and _dep0 > 0 and _dep >= 0.5 * _dep0   # (G33: dep0 > 0)
                check('T11d start-year gate decision = the pre-registered rule applied to its recorded errors / correlations', _exp == bool(gt.get('passed')) and y0b == (gt.get('requested') if _exp else 2021.0),
                      f'rule says {_exp} (dependence {_dep:.3f} vs 2021 {_dep0:.3f}), g04 recorded {gt.get("passed")}, map_y0 {y0b}')
    # ======================= T11h (r14 G46.13) g05 maps with g04's validated estimator: the g04 slice g05 executes defines MS (= 32), calibrated_median and medoid_draw;
    # g05 defines none of them itself, draws MS samples per production stand (USE_B = 0) and calls calibrated_median / medoid_draw; g04's map_path draws MS
    # by default and no caller overrides it; calibrated_median(s, cf) = e^cf · (true median of s) (behavioural)
    if _MAPCHK:
        try:
            ok_h, det_h = t11h_check(open(f'{ROOT}/scripts/g04_bayes_downscale.py').read(), open(f'{ROOT}/scripts/g05_map_stands.py').read(), ns_)
            check('T11h g05 maps with g04\'s validated estimator (MS = 32 draws, calibrated_median, medoid_draw; G46.13)', ok_h, det_h)
        except Exception as e_: check('T11h g05 maps with g04\'s validated estimator (G46.13)', False, f'not testable ({type(e_).__name__}: {str(e_)[:200]})')
    # ======================= T11i (r14 G46.11 / G46.12 / G48.5) g04's calibration factors (CAL, CALF), validation errors / medoid Bray, null baselines and
    # map_accept recomputed from its per-plot dump (t11i_check); the dumped null predictions against an independent null from the table (the last visit
    # passing MIN_FSHARE of every training plot with candidates, within ±3 yr of the query visit, judged plots excluded; true medians of obs_summ)
    if _MAPCHK and os.path.exists(f'{D}/bayes_eval.json'):
        be_i = json.load(open(f'{D}/bayes_eval.json')); pf_i = be_i.get('plot_dump', G04_DUMPS[0])
        if pf_i not in G04_DUMPS or not os.path.exists(f'{D}/{pf_i}'):
            check('T11i g04 calibration / map acceptance re-derived from its per-plot dump', False, f'input missing: {pf_i} (bayes_eval.json plot_dump; G48.5)')
        else:
            try: ok_i, det_i, _ = t11i_check(be_i, pd.read_parquet(f'{D}/{pf_i}'), t11i_null_ref(ns_, V, FSOK, pd.read_parquet(f'{D}/{pf_i}')))
            except Exception as e_: ok_i, det_i = False, f'not testable ({type(e_).__name__}: {str(e_)[:200]})'
            check('T11i g04 calibration (CAL, CALF), validation errors / Bray, null baselines and map_accept re-derived from its per-plot dump; nulls = an independent recomputation',
                  ok_i, det_i)
    # ======================= T11b stand cohort tables (g05, the TreeMap-independent initial state of the simulation engines) carry the CALIBRATED stand summary
    if _MAPCHK and os.path.exists(f'{D}/stand_cohorts_v3.parquet') and os.path.exists(f'{D}/stand_veg_v3.parquet'):
        Cc = pd.read_parquet(f'{D}/stand_cohorts_v3.parquet'); Sv = pd.read_parquet(f'{D}/stand_veg_v3.parquet').set_index('stand_id')
        ct_ = Cc.groupby('stand_id').tph.sum(); cb_ = (Cc.tph * math.pi / 4 * (Cc.qmd_cm / 100) ** 2).groupby(Cc.stand_id).sum(); jj = Sv[['tph', 'ba']].join(pd.DataFrame({'ct': ct_, 'cb': cb_}), how='inner')
        jj = jj[(jj.tph > 0) & (jj.ba > 0)]; rt = (jj.ct / jj.tph - 1).abs().max(); rb = (jj.cb / jj.ba - 1).abs().max()
        check('T11b stand cohort tables sum to the calibrated stand TPH and BA (g05)', rt < 1e-6 and rb < 1e-6, f'{len(jj)} stands; max |ratio - 1| TPH {rt:.1e} BA {rb:.1e}')
        # T11e (r13 coverage, g05): GEN_QMD's source = the QMD of the calibrated stand TPH / BA; dom_key = the BA-dominant key of the stand's cohort table
        _q = 100 * np.sqrt(Sv.ba / (Sv.tph * math.pi / 4)).where(Sv.tph > 0, 0.0); eq_ = float(np.nanmax(np.abs(_q - Sv.qmd)))
        _kb = (Cc.tph * math.pi / 4 * (Cc.qmd_cm / 100) ** 2).groupby([Cc.stand_id, Cc.key]).sum().reset_index(name='b'); _dk = _kb.loc[_kb.groupby('stand_id').b.idxmax()].set_index('stand_id').key
        _jd = Sv.dom_key.reindex(_dk.index); nd_ = int((_jd != _dk).sum())
        check('T11e stand map QMD = QMD of the calibrated TPH / BA; dom_key = the BA-dominant key of the stand cohort table (g05)', eq_ < 1e-6 and nd_ == 0,
              f'{len(Sv)} stands: max |QMD - QMD(TPH, BA)| {eq_:.1e} cm; dominant-key mismatches {nd_} of {len(_dk)} stands with cohorts')
        # T11f map coverage: every stand with a footprint (stands.parquet, g02j) is mapped with finite TPH / BA
        _sp = set(int(x) for x in pd.read_parquet(f'{D}/stands.parquet').stand_id.unique()); _mp = set(int(x) for x in Sv.index[np.isfinite(Sv.tph) & np.isfinite(Sv.ba)])
        check('T11f map coverage: every stand footprint stand (g02j) has a mapped TPH / BA (g05)', not (_sp - _mp), f'{len(_sp)} footprint stands, unmapped {len(_sp - _mp)}, mapped stands without a footprint {len(_mp - _sp)}')

    # ======================= T4r (was T12 before r13) summary-reconstruction targets (g03b OBS_*) and the loss at a perfect prediction
    vv12 = torch.tensor(samp.vid.values[:300]); o12 = ns_['obs_summ'](vv12)
    okt = all(np.allclose(ns_[k][vv12].numpy(), o12[i].numpy(), rtol=1e-5, atol=1e-6) for i, k in enumerate(['OBS_T', 'OBS_B', 'OBS_Q']))
    lp_perf = torch.where(ns_['slot_p'][vv12] > 0, torch.tensor(60.0), torch.tensor(-60.0)) + torch.log(ns_['POSW'])
    se12 = ns_['summ_err'](lp_perf, ns_['slot_v'][vv12], vv12)
    check('T4r summary-loss targets = obs_summ (TPH, BA, QMD; BA independently = FIA formula by T4a) and the loss vanishes at a perfect prediction (incl. the Bray term)',
          okt and float(se12.max()) < 1e-6, f'max loss at perfect prediction {float(se12.max()):.2e}')
    # ======================= T13 pixel-exact georeferencing of the targeted AlphaEarth windows (g02c): their 3x3 means must match the 2022 mosaic best at offset 0
    import glob as _gl
    LSg = pd.read_parquet(f'{D}/localisation.parquet'); RPw = int(1600 // 30) + 2; acc13 = {}; nv13 = 0
    _f13 = sorted(_gl.glob(f'{D}/visit_aef/*_2021.npy') + _gl.glob(f'{D}/visit_aef/*_2023.npy'))
    _f13 = [_f13[i] for i in sorted(np.random.default_rng(13).choice(len(_f13), min(60, len(_f13)), replace=False))] if _f13 else []      # (seeded random: r13 E-F4)
    # the g02c windows are on the 30 m TreeMap lattice: the reference is the 30 m 2022 mosaic (AEF90: data/aef/aef_2022.tif by coordinates, not the 90 m mosaic: r13 F#4)
    _a13 = rasterio.open(f'{ROOT}/data/aef/aef_2022.tif') if AEF90 else a22
    def _m13(r, c):
        if AEF90: xx_, yy_ = land.transform * (c + 0.5, r + 0.5); r, c = _a13.index(xx_, yy_)
        return np.nan_to_num(np.nanmean(dq(_a13.read(window=Window(c - 1, r - 1, 3, 3), boundless=True, fill_value=-128)).reshape(64, -1), 1))
    for fnw in _f13:
        st_, u_, c_, p_, yr_ = map(int, os.path.basename(fnw)[:-4].split('_')); pp = P[(P.st == st_) & (P.u == u_) & (P.c == c_) & (P.p == p_)]
        if not len(pp): continue
        pid = int(pp.pid.iloc[0]); g = C[C.pid == pid].sample(min(40, (C.pid == pid).sum()), random_state=0); Wn = dq(np.load(fnw)); r0, c0 = land.index(pp.x.iloc[0], pp.y.iloc[0])
        M22 = np.stack([_m13(r, c) for r, c in zip(g.row, g.col)])
        M22 /= np.linalg.norm(M22, axis=1, keepdims=True) + 1e-9; nv13 += 1
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr = g.row.values - (r0 - RPw) + dr; cc = g.col.values - (c0 - RPw) + dc; ok_ = (rr >= 1) & (rr < Wn.shape[1] - 1) & (cc >= 1) & (cc < Wn.shape[2] - 1)
                Wv = np.stack([np.nan_to_num(np.nanmean(Wn[:, r - 1:r + 2, c - 1:c + 2].reshape(64, -1), 1)) for r, c in zip(rr[ok_], cc[ok_])]); Wv /= np.linalg.norm(Wv, axis=1, keepdims=True) + 1e-9
                acc13.setdefault((dr, dc), []).append(float(np.median((Wv * M22[ok_]).sum(1))))
    med13 = {k: np.median(v) for k, v in acc13.items()}; best13 = max(med13, key=med13.get) if med13 else None
    if best13 is None and R4: print('N/A   T13 Region 4: no targeted AlphaEarth windows (g02c is no driver stage and was not run for this table; the R4 footprints are checked by T3c2 / T3c2z)')
    elif best13 is None: print('SKIP  T13 no targeted AlphaEarth windows (g02c) in this table')
    else: check('T13 targeted AlphaEarth windows are pixel-exact: agreement with the 2022 mosaic peaks at offset (0, 0) among ±2 px shifts', best13 == (0, 0),
          f'{nv13} windows; median cosine at (0,0) {med13[(0, 0)]:.4f}, best {best13} {med13[best13]:.4f}, at (+1,0) {med13[(1, 0)]:.4f}, (0,+1) {med13[(0, 1)]:.4f}')
    # ======================= T14 no leakage: the joint AlphaEarth location posterior (g08) exists only for TRAINING plots (spatial blocks + buffer of g03b)
    if os.path.exists(f'{D}/cand_loc_aef.parquet'):
        LA = pd.read_parquet(f'{D}/cand_loc_aef.parquet'); trp = set(ns_['vpid'][ns_['tr_v']]); bad14 = set(LA.pid) - trp
        check('T14 same-year AlphaEarth location posterior only for training plots (no held-out / buffer plot); file non-empty', len(LA) > 0 and not bad14, f'{LA.pid.nunique()} plots, outside training {len(bad14)}')
    elif R4: print('N/A   T14 Region 4: no same-year AlphaEarth location posterior (g08 is no driver stage and was not run for this table; LOC=aef unused)')
    else: print('SKIP  T14 no cand_loc_aef.parquet (g08 not run; LOC=aef unused)')
    # ======================= T12 g04 set hygiene (r13 E-F3, r14 G47.5): g04's own set construction (exec'd up to, not including, the MDN fit; its evaluation-set
    # statements executed by their anchors) against sets recomputed here from g03b's tr_v / va_v / te_v and the plot coordinates (brute-force distances, not a
    # KD-tree). Every anchored name must be assigned in g04 on its anchored line only (a later reassignment would go unseen: E14-6); the null pool's exclusion
    # of the judged plots is tested BEHAVIOURALLY (g04's own null_pred and its own excl expressions, on a pool planted with the judged plots); the plot halves
    # must not be a parity, sorted or first-half split (runs test on the sorted calibration plots)
    G4s = open(f'{ROOT}/scripts/g04_bayes_downscale.py').read(); ok12, det12 = t12_check(G4s, ns_, V, P, FSOK, os.environ.get('VAL_SPLIT', '0') == '1')
    check('T12 g04 set hygiene (calibration / buffer / MDN fit / gate halves / kNN-null pools) = sets recomputed from g03b splits + coordinates; behavioural null exclusion', ok12, det12)
    # ======================= T15 the enforced dynamics acceptance (g06, G7 / G46.8) re-derived for the production seed (r13 E-F3, r14 G47.4): the driver passes the
    # exact record as GV_DYN_JSON (G45.8); unset or missing FAILs; the bins and medians are recomputed from g06's per-pair dump (t15_judge)
    if not _MAPCHK: print('SKIP  T15 vtab mode (GV_REQUIRE_MAP=0): the dynamics stage runs after training')
    else:
        _f15 = os.environ.get('GV_DYN_JSON', '')
        if not _f15 or not os.path.exists(_f15): check('T15 dynamics acceptance (production seed) re-derived and passed', False, f'input missing: GV_DYN_JSON {_f15!r} (unset or no such file; G45.8 / G47.4)')
        else:
            dj = json.load(open(_f15)); _p15 = g06_dump_path(_f15); dmp = pd.read_parquet(_p15) if os.path.exists(_p15) else None
            ok15, vd15, why15, bins15 = t15_judge(dj, dmp, set(int(p_) for p_ in ns_['vpid'][ns_['va_v']]) if len(ns_['va_v']) else set())
            lat15 = dj.get('latent', {}) if isinstance(dj.get('latent'), dict) else {}
            check('T15 dynamics acceptance (production seed) = the pre-registered rule re-derived from the per-pair dump (G7 bins, true medians) and the G46.8 latent criteria, on the validation set, and passed',
                  ok15, f"{os.path.basename(_f15)} (dump {os.path.basename(_p15)}{'' if dmp is not None else ' MISSING'}): set {dj.get('undisturbed_acceptance', {}).get('set')}; "
                  f"bins {{(lo, hi): (n, ok)}} {bins15}; latent max |mean| {lat15.get('max_abs_mean')}, min sd {lat15.get('min_sd')}, share |z| > 2.9 {lat15.get('share_abs_gt_2p9')}; "
                  f"rule says {vd15}, g06 recorded {dj.get('undisturbed_acceptance', {}).get('passed')}; problems {why15}")
    # ======================= T4y (r14 G47.8) the checkpoint's training switches = the configured ones: every tensor name / shape of the checkpoint = the model g03b
    # builds under this environment (ARCH, DZ, TOPK, SITE_ATTR, AGE_CH, JUMP_RESET, STATE_ATTR change them), and the switches it records (`switches`) = the
    # environment's; the loss-only switches leave no trace in the weights, so a checkpoint without that record FAILs
    _mdl = os.environ.get('MODEL', '')
    if not _MAPCHK or not _mdl: print('SKIP  T4y vtab mode / MODEL unset: no checkpoint before training')
    elif not os.path.exists(f'{D}/{_mdl}'): check('T4y checkpoint switches = the configured ones', False, f'input missing: checkpoint {D}/{_mdl}')
    else:
        sd4 = torch.load(f'{D}/{_mdl}', weights_only=False, map_location='cpu'); shp4 = {}
        for nm4, mod4 in (('enc', ns_['enc']), ('flow', ns_['flow']), ('dec', ns_['dec']), ('q', ns_['qn']), ('h22', ns_['h22']), ('h25', ns_['h25'])):
            ex4 = {k: tuple(v.shape) for k, v in mod4.state_dict().items()}; got4 = {k: tuple(v.shape) for k, v in sd4.get(nm4, {}).items()}
            if ex4 != got4: shp4[nm4] = {'names only one side': sorted(set(ex4) ^ set(got4))[:4], 'shapes': [(k, ex4[k], got4[k]) for k in sorted(set(ex4) & set(got4)) if ex4[k] != got4[k]][:3]}
        ok4y, det4y = t4y_switches(sd4.get('switches'), ENV0, _mdl)
        check('T4y checkpoint (MODEL) built under the configured switches: tensor names / shapes = the configured model; recorded `switches` = the configuration (G48.5)',
              not shp4 and ok4y, f'{_mdl}: structure differs {shp4 or "no"}; {det4y}')
    # ======================= T4x (r14 G46.7) converted plots carry no AlphaEarth term: g03b's own training step (exec'd for one batch, last: it updates the model)
    # gives MIL = Q = Q-decode = 0 on a batch whose visits are flagged converted, and non-zero terms on the same batch unflagged; g04's B-likelihood set (trB)
    # leaves converted visits out; their pairs are dropped (T8p)
    try:
        ok4x, det4x = t4x_check(ns_, open(f'{ROOT}/scripts/g03b_train_gafl.py').read(), open(f'{ROOT}/scripts/g04_bayes_downscale.py').read(), V, BAtk)
        check('T4x converted plots (G46.7): g03b\'s training step gives MIL = Q = Q-decode = 0 on a batch flagged converted (non-zero unflagged); g04 trB leaves converted visits out; pairs dropped (T8p)', ok4x, det4x)
    except Exception as e_:
        check('T4x converted plots carry no AlphaEarth term (G46.7)', False, f'not testable on this code ({type(e_).__name__}: {str(e_)[:200]})')
    nf = sum(1 for _, ok in RES if not ok); print(f'\n{len(RES) - nf} / {len(RES)} checks PASS'); sys.exit(1 if nf else 0)
if __name__ == '__main__': main()
