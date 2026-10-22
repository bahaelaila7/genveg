"""Generative-vegetation pipeline driver — every stage in order, one configuration (config/genveg.yml; docs/GENVEG_PIPELINE.md).
  .venv/bin/python -I scripts/genveg_pipeline.py [--config config/genveg.yml] [--set key=value ...] [--force stage,...] [--adopt stage,...] [stage ...]
Stages (default all, in order):
  table         g02b_build_gafl.py       visits, cohort tokens, footprint candidates + location prior (forest mask = the stand run's land_clean)
  localisation  g02d_localisation.py     per-plot localisation numbers keyed by FIA plot identity
  records       g02e_fia_records.py      typed FIA events / negative-evidence periods (fia_events, fia_negs, visits.fia_blocked)
  history       g02g_history_consistency.py  candidates consistent with each plot's whole FIA history (cand_hist; overwrites localisation)
  stands_fp     g02j_stand_footprints.py stand footprints of the mapping set from the stand run (stands.parquet, stand_feat.npy)
  vup           gv_verify_upstream.py    upstream logic checks (grid, forest-land rules, stands, FE display)          [check: exit 0 required]
  train         g03b_train_gafl.py       temporal JEPA model, one per seed (<model_prefix>_s<seed>.pt)
  dynamics      g06_dynamics_check.py    held-out disturbance-response test of every seed and eval_loc
  downscale     g04_bayes_downscale.py   landscape prior (MDN) + likelihood B + mapping-path calibration (model = model_seed)
  map           g05_map_stands.py        stand vegetation per use_b (stand_veg_v3_B<b>.parquet; the last = stand_veg_v3.parquet)
  fe            24_export_fe.py + g07_fe_attach.py   Forest Explorer stands layer of the stand run + GEN_* attributes (the host serves it)
  verify        gv_verify.py             independent recomputation of every stage                                   [check: exit 0 required]
Outputs are the scripts' own files under data/proto/<data>/. The driver keeps data/proto/<data>/pipeline_state.json: per stage the hash of its config
section, the code of its scripts and local modules, the upstream stage hashes and the stand run's stage hashes. A stage whose hash is unchanged is
skipped; --force re-runs it; --adopt records existing outputs as current WITHOUT running (only for artifacts known to come from this code + config).
Logs: data/proto/<data>/pipeline_logs/<stage>.log."""
import os, sys, json, time, argparse, hashlib, subprocess, shutil, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stands_stages import apply_overrides
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); SCR = os.path.join(ROOT, 'scripts'); WS = os.environ.get('WORKSPACE', os.path.dirname(ROOT))
LIB = ['common.py', 'features.py', 'stands_cfg.py', 'stand_runs.py']
STAGES = {   # stage: scripts (code hash), config sections, upstream stages, stand-run stages it depends on
    'table':        dict(code=['g02b_build_gafl.py'], cfg=['table', 'data'], up=[], stands=['clean']),
    'localisation': dict(code=['g02d_localisation.py'], cfg=[], up=['table'], stands=[]),
    'records':      dict(code=['g02e_fia_records.py', 'g02b_build_gafl.py'], cfg=[], up=['table'], stands=[]),
    'history':      dict(code=['g02g_history_consistency.py', 'g02b_build_gafl.py'], cfg=[], up=['records', 'localisation'], stands=[]),
    'stands_fp':    dict(code=['g02j_stand_footprints.py', 'g02b_build_gafl.py'], cfg=[], up=['table'], stands=['finalize']),
    'vup':          dict(code=['gv_verify_upstream.py'], cfg=['verify'], up=['stands_fp'], stands=['land', 'clean', 'finalize']),
    'train':        dict(code=['g03b_train_gafl.py'], cfg=['train'], up=['history'], stands=[]),
    'dynamics':     dict(code=['g06_dynamics_check.py', 'g03b_train_gafl.py'], cfg=['dynamics', 'train'], up=['train'], stands=[]),
    'downscale':    dict(code=['g04_bayes_downscale.py', 'g03b_train_gafl.py'], cfg=['downscale', 'train', 'model_seed'], up=['train'], stands=[]),
    'map':          dict(code=['g05_map_stands.py', 'g04_bayes_downscale.py', 'g03b_train_gafl.py'], cfg=['map', 'train', 'model_seed'], up=['downscale', 'stands_fp'], stands=['finalize']),
    'fe':           dict(code=['24_export_fe.py', 'g07_fe_attach.py'], cfg=['fe'], up=['map'], stands=['finalize']),
    'verify':       dict(code=['gv_verify.py', 'g03b_train_gafl.py', 'g02b_build_gafl.py'], cfg=['verify', 'train'], up=['fe', 'history'], stands=['finalize']),
}
ORDER = list(STAGES)

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
    srh = json.load(open(f'{SR}/summary.json'))['hashes']
    stf = f'{D}/pipeline_state.json'; state = json.load(open(stf)) if os.path.exists(stf) else {}
    H = {}
    for st in ORDER:
        d = STAGES[st]
        H[st] = hashlib.sha256(json.dumps(dict(stage=st, cfg={k: cfg.get(k) for k in d['cfg']}, data=cfg['data'], code=code_hash(d['code']),
                                               up={u: H[u] for u in d['up']}, stands={s: srh[s] for s in d['stands']}), sort_keys=True, default=str).encode()).hexdigest()[:12]
    py = [sys.executable, '-I']
    tenv = {k: str(v) for k, v in cfg['train']['env'].items()}
    base_env = dict(os.environ, STANDS_RUN=SR, DATA=cfg['data'], WORKSPACE=WS, FE_GEOJSON=os.path.join(WS, cfg['fe']['geojson']))
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
        if st == 'vup': return [(S('gv_verify_upstream.py', cfg['verify']['n_upstream']), {})]
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
            return out
        if st == 'fe':
            gj = os.path.join(WS, cfg['fe']['geojson'])
            return [(S('24_export_fe.py', os.path.join(ROOT, cfg['fe']['area']), gj, f'{SR}/finalize/stands.gpkg'), {}), (S('g07_fe_attach.py', gj, cfg['fe']['tag']), {})]
        if st == 'verify': return [(S('gv_verify.py', cfg['verify']['n']), dict(tenv, MODEL=model))]
    for st in (a.stages or ORDER):
        if st not in STAGES: sys.exit(f'unknown stage {st}')
        if st in adopt:
            state[st] = dict(hash=H[st], at=time.strftime('%F %T'), how='adopted'); print(f'== {st}: ADOPTED existing outputs as current ({H[st]})'); continue
        if state.get(st, {}).get('hash') == H[st] and st not in force:
            print(f'== {st}: current ({state[st]["how"]} {state[st]["at"]})'); continue
        stale = [u for u in STAGES[st]['up'] if state.get(u, {}).get('hash') != H[u]]
        if stale: sys.exit(f'!! {st}: upstream stage(s) {stale} not current — run them first')
        if st in ('downscale', 'map', 'verify'): shutil.copy(f'{D}/{model}', f'{D}/model.pt')      # tools that read the default model see the chosen one
        t0 = time.time(); print(f'== {st}: running (log {os.path.relpath(LOGD, ROOT)}/{st}.log)', flush=True)
        rc = sh(st, cmds_of(st))
        tail = open(f'{LOGD}/{st}.log').read().strip().splitlines()[-1:] or ['']
        if rc != 0:
            print(f'!! {st} FAILED (exit {rc}): {tail[0][:300]}'); json.dump(state, open(stf, 'w'), indent=1); sys.exit(1)
        state[st] = dict(hash=H[st], at=time.strftime('%F %T'), how=f'ran {time.time()-t0:.0f}s'); json.dump(state, open(stf, 'w'), indent=1)
        print(f'   {tail[0][:300]}', flush=True)
    json.dump(state, open(stf, 'w'), indent=1)

if __name__ == '__main__':
    main()
