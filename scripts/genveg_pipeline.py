"""Generative-vegetation pipeline driver — every stage in order, one configuration (config/genveg.yml; docs/GENVEG_PIPELINE.md).
  .venv/bin/python -I scripts/genveg_pipeline.py [--config config/genveg.yml] [--set key=value ...] [--force stage,...] [--adopt stage,...] [stage ...]
Stages (default all, in order):
  table         g02b_build_gafl.py       visits, cohort tokens, footprint candidates + location prior (forest mask = the stand run's land_clean)
  localisation  g02d_localisation.py     per-plot localisation numbers keyed by FIA plot identity
  records       g02e_fia_records.py      typed FIA events / negative-evidence periods (fia_events, fia_negs, visits.fia_blocked)
  history       g02g_history_consistency.py  candidates consistent with each plot's whole FIA history (cand_hist; overwrites localisation)
  stands_fp     g02j_stand_footprints.py stand footprints of the mapping set from the stand run (stands.parquet, stand_feat.npy)
  vup           gv_verify_upstream.py    upstream logic checks (grid, forest-land rules, stands, FE display; U8 later) [check: exit 0 required]
  train         g03b_train_gafl.py       temporal JEPA model, one per seed (<model_prefix>_s<seed>.pt)
  dynamics      g06_dynamics_check.py    held-out disturbance-response test of every seed and eval_loc
  downscale     g04_bayes_downscale.py   landscape prior (MDN) + likelihood B + mapping-path calibration (model = model_seed)
  map           g05_map_stands.py        stand vegetation per use_b (stand_veg_v3_B<b>.parquet; the last = stand_veg_v3.parquet) + stand cohorts (stand_cohorts_v3[_B<b>].parquet)
  fe            24_export_fe.py + g07_fe_attach.py   Forest Explorer stands layer of the stand run + GEN_* attributes (the host serves it)
  verify        gv_verify.py + gv_verify_upstream.py   independent recomputation of every stage (+ U8 stand mapping) [check: exit 0 required]
Outputs are the scripts' own files under data/proto/<data>/. The driver keeps data/proto/<data>/pipeline_state.json: per stage the hash of its config
section, the code of its scripts and local modules, the upstream stage hashes and pixel-content fingerprints of the stand-run rasters it reads
(land_clean / land / stands), so a re-run of the stand pipeline with identical outputs does not invalidate genveg. A stage whose hash is unchanged is
skipped; --force re-runs it; --adopt records existing outputs as current WITHOUT running (only for artifacts known to come from this code + config).
Logs: data/proto/<data>/pipeline_logs/<stage>.log."""
import os, sys, json, time, argparse, hashlib, subprocess, shutil, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stands_stages import apply_overrides
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); SCR = os.path.join(ROOT, 'scripts'); WS = os.environ.get('WORKSPACE', os.path.dirname(ROOT))
LIB = ['common.py', 'features.py', 'stand_runs.py', 'region.py']   # (stands_cfg.py: the stand pipeline's stage context, unused by genveg scripts)
STAGES = {   # stage: scripts (code hash), config sections, upstream stages, stand-run stages it depends on
    'table':        dict(code=['g02b_build_gafl.py'], cfg=['table', 'data'], up=[], stands=['clean']),
    'localisation': dict(code=['g02d_localisation.py'], cfg=[], up=['table'], stands=[]),
    'records':      dict(code=['g02e_fia_records.py', 'g02b_build_gafl.py'], cfg=[], up=['table'], stands=[]),
    'history':      dict(code=['g02g_history_consistency.py', 'g02b_build_gafl.py'], cfg=[], up=['records', 'localisation'], stands=[]),
    'stands_fp':    dict(code=['g02j_stand_footprints.py', 'g02b_build_gafl.py'], cfg=[], up=['table'], stands=['finalize']),
    'vup':          dict(code=['gv_verify_upstream.py'], cfg=['verify'], up=['stands_fp'], stands=['land', 'clean', 'finalize']),
    'train':        dict(code=['g03b_train_gafl.py'], cfg=['train'], up=['history'], stands=[]),
    'dynamics':     dict(code=['g06_dynamics_check.py', 'g03b_train_gafl.py', 'g04_bayes_downscale.py'], cfg=['dynamics', 'train'], up=['train'], stands=[]),
    'downscale':    dict(code=['g04_bayes_downscale.py', 'g03b_train_gafl.py'], cfg=['downscale', 'train', 'model_seed'], up=['train'], stands=[]),
    'map':          dict(code=['g05_map_stands.py', 'g04_bayes_downscale.py', 'g03b_train_gafl.py'], cfg=['map', 'train', 'model_seed'], up=['downscale', 'stands_fp'], stands=['finalize']),
    'fe':           dict(code=['24_export_fe.py', 'g07_fe_attach.py'], cfg=['fe'], up=['map'], stands=['finalize']),
    'verify':       dict(code=['gv_verify.py', 'gv_verify_upstream.py', 'g03b_train_gafl.py', 'g02b_build_gafl.py'], cfg=['verify', 'train'], up=['fe', 'history'], stands=['finalize', 'land', 'clean']),
}
ORDER = list(STAGES)

STAND_FILES = dict(clean='clean/land_clean.tif', land='land/land.tif', finalize='finalize/stands.tif')
# every switch a stage script reads from its environment: never inherited from the shell (an exported AEF90 / REGION / ARCH … would otherwise build or
# train another region / architecture under this config's hash; audit 2026-10-26 P1) — only the config (top-level env, stage env) sets them
STAGE_SWITCHES = {'AEF90', 'ARCH', 'RADIAL_P', 'T_PAIR', 'SITE_ATTR', 'PAIR_ALL', 'PAIR_HMAX', 'MAP_Y0', 'AGE_CH', 'W_AGE', 'SMOKE_BATCHES', 'BATCH', 'BUF_KM', 'CF16', 'CURRIC', 'DATA', 'DZ', 'EM_ITERS', 'EP8', 'EPB', 'EPOCHS', 'EVAL_LOC', 'FE_GEOJSON', 'G06_SET',
                  'GV_SKIP_U8', 'KC', 'LOAD', 'LOC', 'LR', 'MDN_EP', 'MIN_FSHARE', 'MODEL', 'NCAND', 'NPROC', 'NT', 'PAIR_ANCHOR', 'QD_WIN', 'RAD_M', 'REGION',
                  'SEED', 'SEVERITY_ANCHOR', 'STANDS_RUN', 'TAG', 'TOPK', 'USE_B', 'WRITE_LOC', 'W_COMP', 'W_SUM'}
# Region 4 (AEF90) inputs the table / stand-footprint stages read instead of the stand run's rasters (audit P2): fingerprinted by size + mtime (the
# 90 m mosaics are GBs; a pixel hash would take minutes) together with the code that built them
R4_INPUTS = ['data/r4/land.tif', 'data/r4/hansen_lossyear.tif', 'data/r4/dist/dist_*.tif', 'data/r4/soils90/*.tif', 'data/r4/soils90/fill.json',
             'data/aef90/aef90_2022.tif', 'data/aef90/aef90_2025.tif', 'data/grid/grid_r4.json', 'data/grid/grid_r4_90.json', 'data/grid/aoi_r4_mask90.tif']
R4_CODE = ['r4_grid.py', 'r4_aef90.py', 'r4_rasters.py']
def r4_fingerprint():
    import glob
    h = hashlib.sha256()
    for pat in R4_INPUTS:
        fs = sorted(glob.glob(os.path.join(ROOT, pat)))
        if not fs: sys.exit(f'AEF90 input missing: {pat}')
        for f in fs: st = os.stat(f); h.update(f'{os.path.relpath(f, ROOT)}:{st.st_size}:{st.st_mtime_ns}'.encode())
    for f in R4_CODE: h.update(open(os.path.join(SCR, f), 'rb').read())
    return h.hexdigest()[:16]
def pixel_hash(path):
    """Content fingerprint of a raster: its decoded pixels + shape / dtype / transform (byte hashes differ between identical rasters written with
    different tiling / compression). Cached next to the file by (size, mtime)."""
    import rasterio
    from rasterio.windows import Window
    real = os.path.realpath(path); st = os.stat(real); cf = real + '.pixhash.json'
    if os.path.exists(cf):
        c = json.load(open(cf))
        if c.get('size') == st.st_size and c.get('mtime') == st.st_mtime_ns: return c['hash']
    h = hashlib.sha256(); s = rasterio.open(real); H, W = s.shape; h.update(str((s.shape, s.dtypes, s.transform)).encode())
    for r in range(0, H, 4096): h.update(s.read(1, window=Window(0, r, W, min(4096, H - r))).tobytes())
    out = h.hexdigest()[:16]; json.dump(dict(size=st.st_size, mtime=st.st_mtime_ns, hash=out), open(cf, 'w')); return out

def code_hash(files):
    h = hashlib.sha256()
    for f in files + LIB: h.update(f.encode()); h.update(open(os.path.join(SCR, f), 'rb').read())
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=os.path.join(ROOT, 'config', 'genveg.yml')); ap.add_argument('--set', action='append', default=[], dest='sets')
    ap.add_argument('--force', default=''); ap.add_argument('--adopt', default=''); ap.add_argument('stages', nargs='*')
    a = ap.parse_args(); force = set(filter(None, a.force.split(','))); adopt = set(filter(None, a.adopt.split(',')))
    cfg = apply_overrides(yaml.safe_load(open(a.config)), a.sets)
    D = os.path.join(ROOT, 'data', 'proto', cfg['data']); LOGD = os.path.join(D, 'pipeline_logs'); os.makedirs(LOGD, exist_ok=True)
    SR = os.path.join(ROOT, 'data', 'derived', 'runs', cfg['stands_run'])
    if not os.path.exists(f'{SR}/summary.json'): sys.exit(f'stand run {SR} missing: run scripts/stands_pipeline.py --name {cfg["stands_run"]} first')
    srh = {k: pixel_hash(os.path.join(SR, f)) for k, f in STAND_FILES.items()}          # CONTENT of the stand run's outputs genveg reads
    stf = f'{D}/pipeline_state.json'; state = json.load(open(stf)) if os.path.exists(stf) else {}
    r4h = r4_fingerprint() if str(cfg.get('env', {}).get('AEF90', '0')) == '1' else None
    # stage hash = its config sections, code, region switches and inputs, and per upstream stage its hash AND the fingerprint of the outputs that run wrote
    # (state[u]['out']): a re-run or --force of a stage (e.g. a non-deterministic retrain) makes every downstream stage stale (audit 2026-10-26 F11)
    H = {}
    def rehash():
        for st in ORDER:
            d = STAGES[st]
            H[st] = hashlib.sha256(json.dumps(dict(stage=st, cfg={k: cfg.get(k) for k in d['cfg']}, data=cfg['data'], code=code_hash(d['code']),
                                                   up={u: (H[u], state.get(u, {}).get('out')) for u in d['up']}, stands={s: srh[s] for s in d['stands']},
                                                   **({'env': cfg['env']} if 'env' in cfg else {}), **({'r4': r4h} if r4h and st in ('table', 'stands_fp') else {})),
                                              sort_keys=True, default=str).encode()).hexdigest()[:12]
    rehash()
    def outputs_since(t0):
        """Fingerprint of the files a stage wrote in the data dir (name, size, mtime; logs and the pipeline state excluded)."""
        h = hashlib.sha256()
        for root, _, fs in os.walk(D):
            if os.path.abspath(root).startswith(os.path.abspath(LOGD)): continue
            for f in sorted(fs):
                fp = os.path.join(root, f); st_ = os.stat(fp)
                if f != 'pipeline_state.json' and st_.st_mtime >= t0 - 1: h.update(f'{os.path.relpath(fp, D)}:{st_.st_size}:{st_.st_mtime_ns}'.encode())
        return h.hexdigest()[:12]
    py = [sys.executable, '-I']
    tenv = {k: str(v) for k, v in cfg['train']['env'].items()}
    base_env = dict({k: v for k, v in os.environ.items() if k not in STAGE_SWITCHES}, STANDS_RUN=SR, DATA=cfg['data'], WORKSPACE=WS, FE_GEOJSON=os.path.join(WS, cfg['fe']['geojson']), **{k: str(v) for k, v in cfg.get('env', {}).items()})
    model = f"{cfg['train']['model_prefix']}_s{cfg['model_seed']}.pt"
    def sh(stage, cmds):
        with open(f'{LOGD}/{stage}.log', 'w') as lf:
            for args, env in cmds:
                lf.write(f'$ {" ".join(args)}  [{", ".join(f"{k}={v}" for k, v in env.items())}]\n'); lf.flush()
                r = subprocess.run(args, env=dict(base_env, **env), stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
                if r.returncode != 0: return r.returncode
        return 0
    def cmds_of(st):
        S = lambda f, *x: py + [os.path.join(SCR, f), *map(str, x)]
        if st == 'table': return [(S('g02b_build_gafl.py'), {k: str(v) for k, v in cfg['table'].items()})]
        if st == 'localisation': return [(S('g02d_localisation.py'), {})]
        if st == 'records': return [(S('g02e_fia_records.py'), {})]
        if st == 'history': return [(S('g02g_history_consistency.py'), {})]
        if st == 'stands_fp': return [(S('g02j_stand_footprints.py'), {})]
        if st == 'vup': return [(S('gv_verify_upstream.py', cfg['verify']['n_upstream']), dict(GV_SKIP_U8='1'))]   # U8 (stand mapping) runs in verify
        if st == 'train': return [(S('g03b_train_gafl.py'), dict(tenv, SEED=str(s), MODEL=f"{cfg['train']['model_prefix']}_s{s}.pt")) for s in cfg['train']['seeds']]
        if st == 'dynamics':
            return [(S('g06_dynamics_check.py'), dict(tenv, **{k: str(v) for k, v in cfg['dynamics']['env'].items()}, MODEL=f"{cfg['train']['model_prefix']}_s{s}.pt",
                                                       EVAL_LOC=L, TAG=f"{cfg['name']}_{L}_s{s}")) for s in cfg['train']['seeds'] for L in cfg['dynamics']['eval_loc']]
        if st == 'downscale': return [(S('g04_bayes_downscale.py'), dict(tenv, **{k: str(v) for k, v in cfg['downscale']['env'].items()}, MODEL=model))]
        if st == 'map':
            out = []
            for b in cfg['map']['use_b']:
                out.append((S('g05_map_stands.py'), dict(tenv, **{k: str(v) for k, v in cfg['map']['env'].items()}, MODEL=model, USE_B=str(b))))
                out.append((['cp', f'{D}/stand_veg_v3.parquet', f'{D}/stand_veg_v3_B{b}.parquet'], {}))
                out.append((['cp', f'{D}/stand_cohorts_v3.parquet', f'{D}/stand_cohorts_v3_B{b}.parquet'], {}))
            return out
        if st == 'fe':
            gj = os.path.join(WS, cfg['fe']['geojson'])
            return [(S('24_export_fe.py', os.path.join(ROOT, cfg['fe']['area']), gj, f'{SR}/finalize/stands.gpkg'), {}), (S('g07_fe_attach.py', gj, cfg['fe']['tag']), {})]
        if st == 'verify': return [(S('gv_verify.py', cfg['verify']['n']), dict(tenv, MODEL=model)), (S('gv_verify_upstream.py', cfg['verify']['n_upstream']), {})]
    for st in (a.stages or ORDER):
        if st not in STAGES: sys.exit(f'unknown stage {st}')
        if st in adopt:
            state[st] = dict(hash=H[st], at=time.strftime('%F %T'), how='adopted', out=state.get(st, {}).get('out')); rehash()
            print(f'== {st}: ADOPTED existing outputs as current ({H[st]})'); continue
        if state.get(st, {}).get('hash') == H[st] and st not in force:
            print(f'== {st}: current ({state[st]["how"]} {state[st]["at"]})'); continue
        stale = [u for u in STAGES[st]['up'] if state.get(u, {}).get('hash') != H[u]]
        if stale: sys.exit(f'!! {st}: upstream stage(s) {stale} not current — run them first')
        if st in ('downscale', 'map', 'verify'): shutil.copy(f'{D}/{model}', f'{D}/model.pt')      # tools that read the default model see the chosen one
        state.pop(st, None); json.dump(state, open(stf, 'w'), indent=1)      # running = not current: a failed (forced) re-run leaves no stale 'current' mark (audit r3 I4)
        t0 = time.time(); print(f'== {st}: running (log {os.path.relpath(LOGD, ROOT)}/{st}.log)', flush=True)
        rc = sh(st, cmds_of(st))
        tail = open(f'{LOGD}/{st}.log').read().strip().splitlines()[-1:] or ['']
        if rc != 0:
            print(f'!! {st} FAILED (exit {rc}): {tail[0][:300]}'); json.dump(state, open(stf, 'w'), indent=1); sys.exit(1)
        state[st] = dict(hash=H[st], at=time.strftime('%F %T'), how=f'ran {time.time()-t0:.0f}s', out=outputs_since(t0)); json.dump(state, open(stf, 'w'), indent=1)
        rehash()                                                                            # downstream hashes now see this run's outputs
        print(f'   {tail[0][:300]}', flush=True)
    json.dump(state, open(stf, 'w'), indent=1)

if __name__ == '__main__':
    main()
