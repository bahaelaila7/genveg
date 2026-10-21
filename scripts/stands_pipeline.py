"""Stand-delineation pipeline driver — every stage in order, one configuration (docs/STANDS_PIPELINE.md).
  .venv/bin/python -I scripts/stands_pipeline.py [--config config/pilot.yml] [--set key=value ...] [--window r0 c0 r1 c1] [--name NAME]
                                                 [--force stage,...] [stage ...]
Stages (default all, in order): land clean fit sp merge finalize verify metrics
  land      10_forest_land.py   forest land + status (forest / harvested / regenerating / forest_unmapped) and loss year      [full extent]
  clean     11_land_clean.py    status speckle < MMU removed                                                                  [full extent]
  fit       20_fit_features.py  feature transforms (AEF PCA, TreeMap / soil scaling) on sampled forest-land pixels           [full extent]
  sp        21_superpixels.py   SLIC superpixels per tile and status, inside forest land minus barriers                       [window tiles]
  merge     22_merge.py         region-adjacency graph + Ward / mutual-best merging into stands (auto threshold)
  finalize  23_finalize.py      stands raster (barrier / sliver pixels assigned), MMU generalisation, polygons + attributes   [window]
  verify    stands_verify.py    invariants of the result (run dir verify.txt; exit code = number of FAILs)
  metrics   stands_metrics.py   size + held-out-layer homogeneity vs equal-count square blocks (run dir metrics.json)
Stage outputs live in content-hashed cache dirs data/derived/cache/<stage>-<hash> (hash = the stage's config keys, window, code, upstream hashes):
a variant re-runs only the stages its changes affect. data/derived/runs/<NAME>/ links the stages of one run + config_resolved.yml + summary.json.
NAME defaults to the config name (no --set) or <config name>_<finalize hash>."""
import os, sys, json, time, argparse, subprocess, shutil, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stands_cfg as SC

def resolve(config, sets, window):
    cfg = SC.apply_overrides(yaml.safe_load(open(config)), sets)
    win = tuple(window) if window else (tuple(cfg['window']) if cfg.get('window') else None)
    return cfg, win, SC.stage_hashes(cfg, win)

def run_dir(cfg, sets, hashes, name=None):
    return os.path.join(SC.ROOT, 'data', 'derived', 'runs', name or (cfg['name'] if not sets else f"{cfg['name']}_{hashes['finalize']}"))

def run(config=SC.DEFAULT_CONFIG, sets=(), window=None, name=None, stages=None, force=(), py=sys.executable, quiet=False):
    cfg, win, H = resolve(config, list(sets), window)
    RD = run_dir(cfg, sets, H, name); os.makedirs(RD, exist_ok=True)
    yaml.safe_dump(dict(cfg, window=list(win) if win else None), open(f'{RD}/config_resolved.yml', 'w'), sort_keys=False, allow_unicode=True)
    dirs = {st: SC.stage_dir(st, H[st]) for st in SC.ORDER}
    stages = stages or SC.ORDER + ['verify', 'metrics']
    summ = dict(name=os.path.basename(RD), hashes=H, window=win, sets=list(sets), stages={})
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    for st in stages:
        t0 = time.time()
        if st in SC.STAGES:
            d = dirs[st]
            if os.path.exists(f'{d}/DONE') and st not in force:
                say(f'== {st}: cached {os.path.relpath(d, SC.ROOT)}'); summ['stages'][st] = 'cached'
            else:
                tmp = d + '.partial'; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
                yaml.safe_dump(dict(cfg, window=list(win) if win else None), open(f'{tmp}/config_resolved.yml', 'w'), sort_keys=False, allow_unicode=True)
                ctx = dict(cfg=f'{tmp}/config_resolved.yml', out=tmp, **{'in': dirs}, window=list(win) if win else None)
                say(f'== {st}: {SC.STAGES[st]["script"]} -> {os.path.relpath(d, SC.ROOT)}')
                with open(f'{tmp}/log.txt', 'w') as lf:
                    r = subprocess.run([py, '-I', os.path.join(SC.SCRIPTS, SC.STAGES[st]['script'])], env=dict(os.environ, STANDS_STAGE_JSON=json.dumps(ctx)),
                                       stdout=lf, stderr=subprocess.STDOUT, cwd=SC.ROOT)
                if r.returncode != 0:
                    print(f'!! {st} FAILED; log {tmp}/log.txt:'); print(open(f'{tmp}/log.txt').read()[-3000:]); sys.exit(1)
                shutil.rmtree(d, ignore_errors=True); os.rename(tmp, d); open(f'{d}/DONE', 'w').write(time.strftime('%F %T'))
                say('   ' + open(f'{d}/log.txt').read().strip().splitlines()[-1][:300]); summ['stages'][st] = f'ran {time.time()-t0:.0f}s'
            ln = f'{RD}/{st}'
            if os.path.islink(ln) or os.path.exists(ln): os.remove(ln)
            os.symlink(os.path.relpath(dirs[st], RD), ln)
        elif st in ('verify', 'metrics'):
            script = 'stands_verify.py' if st == 'verify' else 'stands_metrics.py'
            ctx = dict(cfg=f'{RD}/config_resolved.yml', out=RD, **{'in': dirs}, window=list(win) if win else None)
            r = subprocess.run([py, '-I', os.path.join(SC.SCRIPTS, script)], env=dict(os.environ, STANDS_STAGE_JSON=json.dumps(ctx)), cwd=SC.ROOT,
                               capture_output=True, text=True)
            open(f'{RD}/{st}.txt', 'w').write(r.stdout + r.stderr)
            say(f'== {st}: ' + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:]))
            summ['stages'][st] = r.returncode
            if st == 'verify' and r.returncode != 0 and not quiet: print(r.stdout)
        else:
            raise SystemExit(f'unknown stage {st}')
    json.dump(summ, open(f'{RD}/summary.json', 'w'), indent=1, default=str)
    return RD, summ

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=SC.DEFAULT_CONFIG); ap.add_argument('--set', action='append', default=[], dest='sets')
    ap.add_argument('--window', nargs=4, type=int); ap.add_argument('--name'); ap.add_argument('--force', default='')
    ap.add_argument('stages', nargs='*')
    a = ap.parse_args()
    RD, summ = run(a.config, a.sets, a.window, a.name, a.stages or None, [s for s in a.force.split(',') if s])
    print('run dir', os.path.relpath(RD, SC.ROOT))
    if summ['stages'].get('verify', 0): sys.exit(1)
