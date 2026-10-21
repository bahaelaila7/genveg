"""Tune the stand delineation (docs/STANDS_PIPELINE.md §tuning): run variants of the configuration through the cached pipeline on the pilot window and
rank them by the held-out metrics of scripts/stands_metrics.py.
  .venv/bin/python -I scripts/stands_tune.py config/tune_baker.yml
Spec (YAML): base (config file), window (optional), stages (default sp merge finalize verify metrics), and either
  grid:     {key: [values, ...], ...}         every combination
  variants: [[key=value, ...], ...]           explicit override lists
  random:   {n: N, seed: S}                   with grid: N random combinations instead of all
Each variant is a pipeline run named <spec name>__<i> (data/derived/runs/); stages unaffected by a variant's keys are reused from the cache.
Output data/derived/tune/<spec name>/results.csv + results.md: overrides, verify FAIL count, stands, area-weighted median, area share < MMU, held-out R²
gain per layer and their mean (the ranking key; size constraints of the spec — wmedian_ha [lo, hi], max small_area_share — are flagged, not optimised)."""
import os, sys, json, itertools, random, time, yaml, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stands_cfg as SC
from stands_pipeline import run
spec_path = sys.argv[1]; spec = yaml.safe_load(open(spec_path)); NAME = os.path.splitext(os.path.basename(spec_path))[0]
base = os.path.join(SC.ROOT, spec.get('base', 'config/pilot.yml')); stages = spec.get('stages', ['sp', 'merge', 'finalize', 'verify', 'metrics'])
if 'variants' in spec:
    V = [list(v) for v in spec['variants']]
else:
    keys = list(spec['grid']); combos = list(itertools.product(*[spec['grid'][k] for k in keys]))
    if 'random' in spec: random.Random(spec['random'].get('seed', 0)).shuffle(combos); combos = combos[:spec['random']['n']]
    V = [[f'{k}={json.dumps(v)}' for k, v in zip(keys, c)] for c in combos]
V = [[]] + [v for v in V if v]                                     # variant 0 = the base configuration
OUTD = os.path.join(SC.ROOT, 'data', 'derived', 'tune', NAME); os.makedirs(OUTD, exist_ok=True)
cons = spec.get('constraints', {}); rows = []
for i, sets in enumerate(V):
    t0 = time.time(); print(f'[{i}/{len(V)-1}] {sets or "(base)"}', flush=True)
    full = (['land', 'clean', 'fit'] if i == 0 else []) + stages
    RD, summ = run(base, sets, spec.get('window'), f'{NAME}__{i}', full, quiet=True)
    m = json.load(open(f'{RD}/metrics.json')); vf = summ['stages'].get('verify', 0)
    r = dict(variant=i, sets='; '.join(sets) or '(base)', verify_fails=vf, stands=m['stands'], wmedian_ha=round(m['wmedian_ha'], 1),
             small_area_pct=round(100 * m['small_area_share'], 2), mean_gain=round(m['mean_gain'], 4))
    for k, v in m['heldout'].items():
        if isinstance(v, dict): r['gain:' + os.path.basename(k).split('.')[0]] = v['gain']
    lo, hi = cons.get('wmedian_ha', [0, 1e9]); r['ok_size'] = lo <= m['wmedian_ha'] <= hi and 100 * m['small_area_share'] <= cons.get('small_area_pct_max', 100)
    rows.append(r); print(f'    {r} ({time.time()-t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows).sort_values(['ok_size', 'mean_gain'], ascending=[False, False])
    df.to_csv(f'{OUTD}/results.csv', index=False)
    open(f'{OUTD}/results.md', 'w').write(f'# Stand-delineation tuning {NAME} (base {os.path.relpath(base, SC.ROOT)})\n\n' + df.to_markdown(index=False) + '\n')
print(open(f'{OUTD}/results.md').read())
