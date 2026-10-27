"""Generative-vegetation pipeline driver — every stage in order, one configuration (default config/genveg_v3b.yml; docs/GENVEG_PIPELINE.md).
  .venv/bin/python -I scripts/genveg_pipeline.py [--config config/genveg_v3b.yml] [--set key=value ...] [--force stage,...] [--adopt stage,...] [--log FILE] [stage ...]
Stages (default all, in order):
  table         g02b_build_gafl.py       visits, cohort tokens, footprint candidates + location prior (forest mask = the stand run's land_clean; AEF90: data/r4/land.tif)
  localisation  g02d_localisation.py     per-plot localisation numbers keyed by FIA plot identity
  records       g02e_fia_records.py      typed FIA events / negative-evidence periods (fia_events, fia_negs, visits.fia_blocked)
  history       g02g_history_consistency.py  candidates consistent with each plot's whole FIA history (cand_hist; overwrites localisation)
  stands_fp     g02j_stand_footprints.py stand footprints of the mapping set from the stand run (stands.parquet, stand_feat.npy)
  vup           gv_verify_upstream.py    upstream logic checks (grid, forest-land rules, stands, FE display; without U8) [check: exit 0 required]
  vtab          gv_verify.py             the table-level data checks before any training (T0-T10; the map checks are skipped) [check: exit 0 required]
  train         g03b_train_gafl.py       temporal JEPA model, one per seed (<model_prefix>_s<seed>.pt)
  dynamics      g06_dynamics_check.py    held-out disturbance-response test of every seed and eval_loc [check: the production seed's exit status]
  downscale     g04_bayes_downscale.py   landscape prior (MDN) + likelihood B + mapping-path calibration (model = model_seed)
  map           g05_map_stands.py        stand vegetation per use_b (stand_veg_v3_B<b>.parquet; the last = stand_veg_v3.parquet) + stand cohorts (stand_cohorts_v3[_B<b>].parquet)
  fe            24_export_fe.py + g07_fe_attach.py   Forest Explorer stands layer of the stand run + GEN_* attributes (the host serves it)
  verify        gv_verify.py + gv_verify_upstream.py   independent recomputation of every stage (gv_verify with the map checks + gv_verify_upstream with U8) [check: exit 0 required]
Outputs are the scripts' own files under data/proto/<data>/. The driver keeps data/proto/<data>/pipeline_state.json: per stage the hash of its config
section (performance knobs excluded), its code (scripts + the genveg shared modules + the definitions and module-level statements of the stand
pipeline's modules it imports), its upstream stages (hash + output fingerprint), the inputs it reads (stage_inputs; for the check stages the
verifiers' own VERIFY_INPUTS) and pixel-content fingerprints of the stand-run rasters it reads.
A stage is current if its hash matches, the files its run wrote are unchanged and its upstream stages are current; a current stage is skipped;
--force re-runs it; --adopt records the stage's existing files (its last recorded file list, as they are now) as current WITHOUT running it (only for
artifacts known to come from this code + config; never a check stage). --log FILE tees the driver's output to FILE (any name, also in the data dir).
Logs: data/proto/<data>/pipeline_logs/<stage>.log. One driver per data dir (a lock, held by the stages it starts too); no other process may write
into the data dir during a run."""
import os, sys, json, time, argparse, hashlib, subprocess, shutil, glob, ast, re, copy, fnmatch, signal, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stands_stages import apply_overrides
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); SCR = os.path.join(ROOT, 'scripts'); WS = os.environ.get('WORKSPACE', os.path.dirname(ROOT))
# shared modules: the genveg ones are hashed whole; features.py / common.py are STAND-pipeline modules, so only their module-level statements and the
# top-level definitions a stage's scripts import from them (and what those reference) enter its hash (r13 I-1: an edit to the stand featurizer must
# not retrain genveg; G45.3: importing a module runs all its module-level statements)
LIB_WHOLE = ['stand_runs.py', 'region.py']
LIB_DEFS = ['features.py', 'common.py']
# scanned by check_comments, not hashed: stands_cfg (imported by features), stands_stages (imported by this driver; G45.10), the R4 raster builders
SCAN_ONLY = ['stands_cfg.py', 'stands_stages.py', 'r4_grid.py', 'r4_aef90.py', 'r4_rasters.py', 'genveg_pipeline.py', 'check_comments.py']
# stage: scripts (code hash), config sections, up = hash dependencies (their hash + output fingerprint enter this stage's hash), gate = stages that must
# be current (so they ran and passed) before this one runs WITHOUT entering its hash (G21; r13 S-1 / S-2), stands = stand-run rasters it reads
# (stands_gafl: read only outside AEF90, where data/r4/land.tif replaces the stand run's land_clean; r13 I-2). The check stages vup / vtab / verify
# fingerprint what their verifiers list (VERIFY_INPUTS, stand-run rasters included; G45.7)
STAGES = {
    'table':        dict(code=['g02b_build_gafl.py'], cfg=['table', 'data'], up=[], stands=[], stands_gafl=['clean']),
    'localisation': dict(code=['g02d_localisation.py'], cfg=[], up=['table'], stands=[]),
    'records':      dict(code=['g02e_fia_records.py', 'g02b_build_gafl.py'], cfg=[], up=['table'], stands=[]),
    # (localisation: g02g OVERWRITES localisation.parquet, it does not read it — an ordering gate, r13 S-1)
    'history':      dict(code=['g02g_history_consistency.py', 'g02b_build_gafl.py'], cfg=[], up=['records'], gate=['localisation'], stands=[]),
    'stands_fp':    dict(code=['g02j_stand_footprints.py', 'g02b_build_gafl.py'], cfg=[], up=['table'], stands=['finalize'], stands_gafl=['clean']),   # (clean: g02b.feats' forest filter, r13 A-11)
    'vup':          dict(code=['gv_verify_upstream.py'], cfg=['verify.n_upstream'], up=['stands_fp'], stands=[]),
    # vtab: gv_verify BEFORE training — the table-level data checks (T0-T10, incl. the pre-registered T1k / T1l) must pass before any seed is trained (r11 M-3)
    'vtab':         dict(code=['gv_verify.py', 'g03b_train_gafl.py', 'g02b_build_gafl.py', 'g04_bayes_downscale.py', 'check_comments.py'], cfg=['verify.n', 'train', 'table.TOPK', 'table.NCAND', 'table.RADIAL_P'], up=['history', 'stands_fp'], stands=[]),
    # train: up = history only (g03b reads the table, the FIA records and the history-consistent candidates; its stand map is the opt-in STAND_DIAG
    # diagnostic, so no stands_fp: G27); gated on the upstream checks (vup, r10 M-1) and the table checks (vtab, r11 M-3)
    'train':        dict(code=['g03b_train_gafl.py'], cfg=['train'], up=['history'], gate=['vup', 'vtab'], stands=[]),
    # (model_seed: whose acceptance is enforced, r8 I1; name: the TAG of its acceptance records, G45.8)
    'dynamics':     dict(code=['g06_dynamics_check.py', 'g03b_train_gafl.py', 'g04_bayes_downscale.py'], cfg=['dynamics', 'train', 'model_seed', 'name'], up=['train'], stands=[]),
    # (dynamics: its acceptance gates mapping, audit r5 I5; g04 reads none of its files — a gate, r13 S-2)
    'downscale':    dict(code=['g04_bayes_downscale.py', 'g03b_train_gafl.py'], cfg=['downscale', 'train', 'model_seed'], up=['train'], gate=['dynamics'], stands=[]),
    'map':          dict(code=['g05_map_stands.py', 'g04_bayes_downscale.py', 'g03b_train_gafl.py'], cfg=['map', 'train', 'model_seed'], up=['downscale', 'stands_fp'], stands=['finalize'], stands_gafl=['clean']),
    'fe':           dict(code=['24_export_fe.py', 'g07_fe_attach.py'], cfg=['fe'], up=['map'], stands=[]),
    # (dynamics: T15 re-derives the production seed's acceptance from the dynamics_check file the driver passes as GV_DYN_JSON; name: its file name)
    'verify':       dict(code=['gv_verify.py', 'gv_verify_upstream.py', 'g03b_train_gafl.py', 'g02b_build_gafl.py', 'check_comments.py'], cfg=['verify', 'train', 'table.TOPK', 'table.NCAND', 'table.RADIAL_P', 'name'], up=['fe', 'history', 'dynamics'], stands=[]),
}
ORDER = list(STAGES)
CHECKS = ('vup', 'vtab', 'dynamics', 'verify')                 # their result is their exit status: never adopted (G45.1)
VERIFIERS = ('gv_verify.py', 'gv_verify_upstream.py')            # module constants VERIFY_INPUTS / VERIFY_CODE, evaluated under the stage's env (G45.7)
def scanned_scripts():
    """every script check_comments scans (check_comments.pipeline_scripts; gv_verify T0): stage code, shared modules, stands_cfg, stands_stages, the R4 builders, driver"""
    return sorted({c for d in STAGES.values() for c in d['code']} | set(LIB_WHOLE) | set(LIB_DEFS) | set(SCAN_ONLY))
def hashed_scripts():
    """the scripts whose environment reads are checked: stage code, shared modules and stands_cfg (imported by features)"""
    return sorted({c for d in STAGES.values() for c in d['code']} | set(LIB_WHOLE) | set(LIB_DEFS) | {'stands_cfg.py'})

STAND_FILES = dict(clean='clean/land_clean.tif', land='land/land.tif', finalize='finalize/stands.tif')
# every switch a stage script reads from its environment: never inherited from the shell (an exported AEF90 / REGION / ARCH … would otherwise build or
# train another region / architecture under this config's hash; audit 2026-10-26 P1) — only the config (top-level env, stage env) sets them. The driver
# refuses to run a stage script that reads a switch missing here.
STAGE_SWITCHES = {'STAND_DIAG', 'GV_REQUIRE_MAP', 'GV_CURATED_DB', 'GV_FIADB', 'VAL_SPLIT', 'G06_ENFORCE', 'AEF90', 'ARCH', 'RADIAL_P', 'T_PAIR', 'SITE_ATTR', 'PAIR_ALL', 'PAIR_HMAX', 'MAP_Y0', 'AGE_CH', 'W_AGE', 'SMOKE_BATCHES', 'BATCH', 'CF16', 'CURRIC', 'DATA', 'DZ', 'EPB', 'EPOCHS', 'EVAL_LOC', 'FE_GEOJSON', 'G06_SET',
                  'GV_SKIP_U8', 'KC', 'LOAD', 'LOC', 'LR', 'MDN_EP', 'MIN_FSHARE', 'MODEL', 'NCAND', 'NPROC', 'NT', 'PAIR_ANCHOR', 'QD_WIN', 'REGION',
                  'SEED', 'SEVERITY_ANCHOR', 'STANDS_RUN', 'TAG', 'TOPK', 'USE_B', 'W_COMP', 'W_SUM',
                  'STANDS_STAGE_JSON',                       # stands_cfg (imported by features): a stand-pipeline stage context must not leak in (r13 I-1)
                  'WORKSPACE', 'SIGREG_MEAN', 'JUMP_RESET', 'STATE_ATTR',   # (G29-G31)
                  'SIGREG_FLOW', 'AGE_FLOW',                 # (G46.9)
                  'GV_DYN_JSON'}                             # (gv_verify T15: the production acceptance record, passed by the driver; G45.8)
# switches only the driver sets (per stage / from other config keys): a config env section setting one is refused (r13 D-6)
DRIVER_SET = {'GV_SKIP_U8', 'LOAD', 'SMOKE_BATCHES', 'STAND_DIAG', 'G06_ENFORCE', 'GV_REQUIRE_MAP', 'MODEL', 'SEED', 'USE_B', 'TAG', 'EVAL_LOC', 'DATA',
              'STANDS_RUN', 'WORKSPACE', 'FE_GEOJSON', 'GV_CURATED_DB', 'GV_FIADB', 'STANDS_STAGE_JSON', 'GV_DYN_JSON'}
REGION_SWITCHES = {'REGION', 'AEF90'}                        # accepted in the top-level env only (G45.9)
# performance knobs: passed to the stages but kept out of every hash (r13 S-4). Thread counts can change the order of float reductions, so a re-run
# with another NT / NPROC need not be bit-identical — it is not a different model. Library environment (PROJ_*, GDAL_*, OMP / MKL thread counts,
# CUDA_VISIBLE_DEVICES) is inherited and not hashed either (G45.12).
PERF_TABLE = {'NPROC'}; PERF_ENV = {'NT'}
# env sections passed to stage scripts: section -> the stages that receive it (a switch none of their scripts reads is refused: G45.9)
SECTION_STAGES = {'env': ORDER, 'table': ['table'], 'train.env': ['train', 'dynamics', 'downscale', 'map', 'vtab', 'verify'], 'dynamics.env': ['dynamics'],
                  'downscale.env': ['downscale'], 'map.env': ['map']}
def env_sections(cfg):
    """[(name, section)]: the config sections passed to stage scripts as environment variables"""
    out = []
    for nm in SECTION_STAGES:
        sec = cfg.get(nm) if '.' not in nm else cfg[nm.split('.')[0]].get('env')
        if sec is not None: out.append((nm, sec))
    return out
def envstr(v):                                               # YAML booleans as the scripts test them ('1' / '0'; r13 D-7)
    return '1' if v is True else '0' if v is False else str(v)

# ---------------------------------------------------------------- inputs each stage READS (r12 I-1, G25), fingerprinted into that stage's hash only
# rasters / tables: path + size + mtime (the mosaics are GBs); vector files: their CONTENT. The check stages: their verifiers' VERIFY_INPUTS (G45.7)
DIST_CODES = 'data/landfire/dist/dist_codes.csv'
GAFL_RASTERS = ['data/aef/aef_2022.tif', 'data/aef/aef_2025.tif', 'data/landfire/dist/dist_*.tif', 'data/hansen/lossyear.tif', 'data/soils/*.tif']
# Region 4 (AEF90) rasters the table / footprint stages read instead (audit P2); the code that built them is scanned, not hashed (r13 S-3)
R4_INPUTS = ['data/r4/land.tif', 'data/r4/hansen_lossyear.tif', 'data/r4/dist/dist_*.tif', 'data/r4/soils90/*.tif', 'data/r4/soils90/fill.json',
             'data/aef90/aef90_2022.tif', 'data/aef90/aef90_2025.tif', 'data/grid/grid_r4.json', 'data/grid/grid_r4_90.json', 'data/grid/aoi_r4_mask90.tif']
def stage_inputs(st, cfg, r4):
    """{kind: [paths]}: db (config keys), req (ROOT globs, must exist), data (data-dir globs, optional), vec (vector files: STANDS_GPKG, cfg:<key> or a ROOT path)"""
    I = dict(db=(), req=[], data=[], vec=[])
    feat = R4_INPUTS if r4 else GAFL_RASTERS
    if st == 'table': I.update(db=('curated_db', 'fiadb'), req=[DIST_CODES, *feat])
    elif st == 'records': I.update(db=('fiadb',))
    elif st == 'stands_fp': I.update(req=[DIST_CODES, *feat], vec=['STANDS_GPKG', 'data/derived/baker_county.gpkg'])
    elif st == 'train':
        loc = str(cfg['train']['env'].get('LOC', 'history'))         # g03b's optional location posteriors (LOC=aef / plot)
        I.update(req=['data/proto/baker/plots.parquet'], data={'aef': ['cand_loc_aef.parquet'], 'plot': ['cand_prior.parquet']}.get(loc, []))
    elif st == 'map': I.update(vec=['STANDS_GPKG'])
    elif st == 'fe': I.update(vec=['STANDS_GPKG', 'cfg:fe.area'])                    # (24_export_fe's area polygon = fe.area; r13 M-1)
    return I
def _cache_get(cf, st, v):
    """a size / mtime keyed hash cache next to a file: its hash, or None when absent, stale or unreadable (recomputed, never fatal: G45.6)"""
    try:
        c = json.load(open(cf))
        if isinstance(c, dict) and c.get('v') == v and c.get('size') == st.st_size and c.get('mtime') == st.st_mtime_ns and isinstance(c.get('hash'), str): return c['hash']
    except (OSError, ValueError): pass
    return None
def _cache_put(cf, st, v, h):
    """written tmp + fsync + replace (a concurrent reader or a kill never sees a partial file); an unwritable location is skipped"""
    tmp = f'{cf}.{os.getpid()}.tmp'
    try:
        with open(tmp, 'w') as f: json.dump(dict(v=v, size=st.st_size, mtime=st.st_mtime_ns, hash=h), f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, cf)
    except OSError:
        try: os.unlink(tmp)
        except OSError: pass
def vector_hash(path):
    """content fingerprint of a vector file (per layer: CRS, column names + dtypes, attributes, geometries as WKB; row order kept), cached next to it by (size, mtime)"""
    import geopandas as gpd
    real = os.path.realpath(path); st = os.stat(real); cf = real + '.vechash.json'
    out = _cache_get(cf, st, 2)
    if out is not None: return out
    h = hashlib.sha256()
    for layer in sorted(__import__('pyogrio').list_layers(real)[:, 0]):
        g = gpd.read_file(real, layer=layer); h.update(layer.encode()); h.update(str(g.crs).encode())
        h.update(json.dumps([(str(c), str(t)) for c, t in g.dtypes.items()]).encode())          # (column names + dtypes: r13 M-8)
        h.update(pd_bytes(g.drop(columns='geometry'))); h.update(b''.join(g.geometry.to_wkb()))
    out = h.hexdigest()[:16]; _cache_put(cf, st, 2, out); return out
def pd_bytes(df):
    import pandas as pd
    return pd.util.hash_pandas_object(df, index=True).values.tobytes()
def pixel_hash(path):
    """Content fingerprint of a raster: its decoded pixels (every band) + shape / dtypes / transform / CRS / nodata (byte hashes differ between identical
    rasters written with different tiling / compression; G45.6). Cached next to the file by (size, mtime)."""
    import rasterio
    from rasterio.windows import Window
    real = os.path.realpath(path); st = os.stat(real); cf = real + '.pixhash.json'
    out = _cache_get(cf, st, 2)
    if out is not None: return out
    h = hashlib.sha256(); s = rasterio.open(real); H, W = s.shape
    h.update(str((s.count, s.shape, s.dtypes, s.transform, s.crs.to_wkt() if s.crs else None, s.nodatavals)).encode())
    for r in range(0, H, 4096): h.update(s.read(window=Window(0, r, W, min(4096, H - r))).tobytes())
    out = h.hexdigest()[:16]; _cache_put(cf, st, 2, out); return out
def _under(p, base):
    base = os.path.abspath(base); return p == base or p.startswith(base.rstrip(os.sep) + os.sep)       # (a path-component compare: G45.10)
def inputs_fingerprint(cfg, r4, sr, D, stages):
    """per-stage input fingerprints {stage: hash} of the given stages (only those being run or checked: G45.10) and the configured database paths"""
    inp = cfg.get('inputs', {})
    dbs = {'curated_db': inp.get('curated_db', 'landis_region4/data_r4/slim_r4.duckdb'), 'fiadb': inp.get('fiadb', 'FIA/FIASQLITE2PGSQL/FIADB.duckdb')}
    def db(k):                                       # the CONFIGURED path (relative to WORKSPACE) + size / mtime: a moved checkout keeps its hashes (r13 I-3)
        f = os.path.join(WS, dbs[k])
        if not os.path.exists(f): sys.exit(f'input database missing: {k} = {f}')
        st = os.stat(os.path.realpath(f)); return f'{k}:{dbs[k]}:{st.st_size}:{st.st_mtime_ns}'
    def stamp(base, pats, need):
        out = []
        for pat in pats:
            fs = sorted(glob.glob(os.path.join(base, pat)))
            if not fs:
                if need: sys.exit(f'input missing: {pat}')
                out.append(f'{pat}:missing'); continue
            out += [f'{os.path.relpath(f, base)}:{os.stat(f).st_size}:{os.stat(f).st_mtime_ns}' for f in fs]
        return out
    vh = {}
    def vec(v):
        if v == 'STANDS_GPKG': p_ = f'{sr}/finalize/stands.gpkg'
        elif v.startswith('cfg:'):
            x = cfg
            for part in v[4:].split('.'): x = x[part]
            p_ = os.path.join(ROOT, x)
        else: p_ = os.path.join(ROOT, v)
        if not os.path.exists(p_): sys.exit(f'vector input missing: {v} = {p_}')
        if p_ not in vh: vh[p_] = vector_hash(p_)
        return f'{v}:{vh[p_]}'
    out = {}
    for st_ in ORDER:
        if st_ not in stages: continue
        I = stage_inputs(st_, cfg, r4)
        parts = [db(k) for k in I['db']] + stamp(ROOT, I['req'], True) + [f'D:{x}' for x in stamp(D, I['data'], False)] + [vec(v) for v in I['vec']]
        if parts: out[st_] = hashlib.sha256('|'.join(parts).encode()).hexdigest()[:16]
    return out, dbs
def path_fp(p, D, SR, srh):
    """fingerprint of one path a verifier reads (G45.7): the stand-run rasters by pixel content, its gpkg / small text files by content, everything else
    by size + mtime, named relative to the stand run / data dir / checkout / WORKSPACE (a moved checkout keeps its hashes); a directory by its files"""
    p = os.path.abspath(p)
    for k, f in STAND_FILES.items():
        if p == os.path.abspath(os.path.join(SR, f)): return f'SR:{f}:' + (srh(k) if os.path.exists(p) else 'missing')
    for base, tag in ((SR, 'SR:'), (D, 'D:'), (ROOT, ''), (WS, 'ws:')):
        if _under(p, base): rel = tag + os.path.relpath(p, base); break
    else: rel = p
    if not os.path.exists(p): return f'{rel}:missing'
    if os.path.isdir(p):
        fs = sorted(os.path.join(r, f) for r, _, ff in os.walk(p) for f in ff)
        return '|'.join(f'{rel}/{os.path.relpath(f, p)}:{os.stat(f).st_size}:{os.stat(f).st_mtime_ns}' for f in fs) or f'{rel}:empty'
    if rel.startswith('SR:') and p.endswith('.gpkg'): return f'{rel}:{vector_hash(p)}'
    if rel.startswith('SR:') and os.path.getsize(p) < 1 << 20: return f'{rel}:' + hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]   # (r13 S-5: rewritten every run)
    side = [p[:-4] + x for x in ('.shx', '.dbf', '.prj', '.cpg')] if p.endswith('.shp') else []        # a shapefile is read with its sidecars (r13 M-9)
    return '|'.join(f'{rel[:-4] + f[-4:] if side else rel}:{os.stat(f).st_size}:{os.stat(f).st_mtime_ns}' for f in [p] + [f for f in side if os.path.exists(f)])
_VI_MARK = '@@GENVEG_VERIFY_INPUTS@@'
_VI = f"""import importlib.util, json, sys
s = importlib.util.spec_from_file_location('_genveg_verify_inputs', sys.argv[1]); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
print('\\n{_VI_MARK}' + json.dumps(dict(inputs=sorted(map(str, m.VERIFY_INPUTS)), code=sorted(map(str, getattr(m, 'VERIFY_CODE', []))))))
"""

# ---------------------------------------------------------------- code
def _bound(s):
    """the module-level names a top-level statement binds"""
    if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): return {s.name}
    if isinstance(s, (ast.Import, ast.ImportFrom)): return {(a.asname or a.name).split('.')[0] for a in s.names}
    return {n.id for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
def _isdoc(s):
    return isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)
def _nodoc(s):
    """a statement without its docstrings (G45.3: like comments, they do not count)"""
    s = copy.deepcopy(s)
    for n in ast.walk(s):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body and _isdoc(n.body[0]): n.body = n.body[1:] or [ast.Pass()]
    return s
def defs_hash(mod, names):
    """hash of module `mod`'s module-level statements (all but def / class and the module docstring: importing runs them; G45.3), the top-level
    definitions of `names` and, transitively, the module-level names these reference (ast dump without docstrings: comments, docstrings and formatting
    do not count); '*' = the whole file"""
    src = open(os.path.join(SCR, mod)).read()
    if '*' in names: return hashlib.sha256(src.encode()).hexdigest()
    body = ast.parse(src).body; binds = {}
    for i, s in enumerate(body):
        for nm in _bound(s): binds.setdefault(nm, []).append(i)
    keep = {i for i, s in enumerate(body) if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not (i == 0 and _isdoc(s))}
    todo, seen = list(names) + [n.id for i in keep for n in ast.walk(body[i]) if isinstance(n, ast.Name)], set()
    while todo:
        nm = todo.pop()
        if nm in seen: continue
        seen.add(nm)
        if nm not in binds:
            if nm in names: sys.exit(f'{mod}: {nm} (imported by a genveg stage) is not defined at module level')
            continue
        for i in binds[nm]:
            if i not in keep: keep.add(i); todo += [n.id for n in ast.walk(body[i]) if isinstance(n, ast.Name)]
    return hashlib.sha256('\n'.join(f'{mod}:{ast.dump(_nodoc(body[i]))}' for i in sorted(keep)).encode()).hexdigest()
def lib_imports(files):
    """{module: names} the given scripts import from the stand-pipeline modules (LIB_DEFS)"""
    want = {m: set() for m in LIB_DEFS}
    for f in files:
        for n in ast.walk(ast.parse(open(os.path.join(SCR, f)).read())):
            if isinstance(n, ast.ImportFrom) and n.module and f'{n.module}.py' in want: want[f'{n.module}.py'] |= {a.name for a in n.names}
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if f'{a.name}.py' in want: want[f'{a.name}.py'].add('*')         # a plain `import features`: the whole module
    return want
_CH = {}
def code_hash(files):
    k = tuple(files)
    if k not in _CH:
        h = hashlib.sha256()
        for f in files + LIB_WHOLE: h.update(f.encode()); h.update(open(os.path.join(SCR, f), 'rb').read())
        for m, names in sorted(lib_imports(files).items()):
            if names: h.update(f'{m}:{sorted(names)}:{defs_hash(m, names)}'.encode())
        _CH[k] = h.hexdigest()
    return _CH[k]
def env_reads(path):
    """(reads, bad) of a script (G45.10, AST-based): reads = {switch: True if some read of it is float(...)-parsed}, from os.environ.get / setdefault /
    pop / [...] / `in os.environ` and os.getenv (aliases included); bad = [line] of reads whose name is not a constant — a name looped over a literal
    (directly or through a name bound to one in the same scope) is resolved"""
    tree = ast.parse(open(path).read()); par = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n): par[c] = n
    osn, envn, getn = {'os'}, set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import): osn |= {a.asname or a.name for a in n.names if a.name == 'os'}
        elif isinstance(n, ast.ImportFrom) and n.module == 'os':
            envn |= {a.asname or a.name for a in n.names if a.name == 'environ'}; getn |= {a.asname or a.name for a in n.names if a.name == 'getenv'}
    def is_env(x): return (isinstance(x, ast.Attribute) and x.attr == 'environ' and isinstance(x.value, ast.Name) and x.value.id in osn) or (isinstance(x, ast.Name) and x.id in envn)
    for n in ast.walk(tree):                                                    # (aliases of os.environ)
        if isinstance(n, ast.Assign) and is_env(n.value): envn |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Module)
    COMPS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    def gen_binds(g, nm): return (isinstance(g.target, ast.Name) and g.target.id == nm) or (isinstance(g.target, ast.Tuple) and g.target.elts and isinstance(g.target.elts[0], ast.Name) and g.target.elts[0].id == nm)
    def strs(v, at, depth=0):
        """the constant strings an expression evaluates to / iterates over, or None"""
        if depth > 8: return None
        if isinstance(v, ast.Constant): return {v.value} if isinstance(v.value, str) else None
        if isinstance(v, (ast.Tuple, ast.List, ast.Set)): ss = [strs(e, at, depth + 1) for e in v.elts]; return None if any(s is None for s in ss) else set().union(*ss)
        if isinstance(v, ast.Dict): ss = [strs(k, at, depth + 1) if k is not None else None for k in v.keys]; return None if any(s is None for s in ss) else set().union(*ss)
        if isinstance(v, (ast.DictComp, ast.SetComp, ast.ListComp)) and len(v.generators) == 1:
            k = v.key if isinstance(v, ast.DictComp) else v.elt
            return strs(v.generators[0].iter, v, depth + 1) if isinstance(k, ast.Name) and gen_binds(v.generators[0], k.id) else None
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ('sorted', 'list', 'tuple', 'set') and len(v.args) == 1: return strs(v.args[0], at, depth + 1)
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr in ('items', 'keys') and not v.args: return strs(v.func.value, at, depth + 1)
        if isinstance(v, ast.Name): return name_strs(v.id, at, depth + 1)
        return None
    def name_strs(nm, at, depth):
        x, prev = at, at
        while x in par:                                                         # an enclosing comprehension / loop that binds it, else its scope
            prev, x = x, par[x]
            if isinstance(x, COMPS):
                for g in x.generators:
                    if gen_binds(g, nm): return strs(g.iter, x, depth + 1)
            if isinstance(x, (ast.For, ast.AsyncFor)) and prev is not x.iter and gen_binds(x, nm):     # (the loop target, not re-bound in the body)
                if not any(isinstance(t, ast.Name) and t.id == nm and isinstance(t.ctx, ast.Store) for b in x.body for t in ast.walk(b)): return strs(x.iter, x, depth + 1)
            if isinstance(x, SCOPES): break
        out, found = set(), False
        def walk(n):
            nonlocal found
            for c in ast.iter_child_nodes(n):
                if isinstance(c, SCOPES + COMPS): continue
                if isinstance(c, ast.Assign) and any(isinstance(t, ast.Name) and t.id == nm for t in c.targets): found = True; out.update(strs(c.value, c, depth) or {None})
                elif isinstance(c, (ast.For, ast.AsyncFor)) and (isinstance(c.target, ast.Name) and c.target.id == nm or isinstance(c.target, ast.Tuple) and c.target.elts and isinstance(c.target.elts[0], ast.Name) and c.target.elts[0].id == nm):
                    found = True; out.update(strs(c.iter, c, depth) or {None})
                elif isinstance(c, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr, ast.With, ast.arguments)) and any(isinstance(t, ast.Name) and t.id == nm for t in ast.walk(c)): found = True; out.add(None)
                walk(c)
        walk(x)
        if not found and not isinstance(x, ast.Module): return name_strs(nm, x, depth)     # a free name: the enclosing scope
        return None if (not found or None in out) else out
    reads, bad = {}, []
    def add(arg, node):
        ss = strs(arg, node)
        if ss is None: bad.append(node.lineno); return
        p = par.get(node); fl = isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.func.id == 'float' and p.args and p.args[0] is node
        for s in ss: reads[s] = reads.get(s, False) or fl
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and n.args:
            f = n.func
            if isinstance(f, ast.Attribute) and is_env(f.value) and f.attr in ('get', 'setdefault', 'pop', '__getitem__', '__contains__'): add(n.args[0], n)
            elif (isinstance(f, ast.Attribute) and f.attr == 'getenv' and isinstance(f.value, ast.Name) and f.value.id in osn) or (isinstance(f, ast.Name) and f.id in getn): add(n.args[0], n)
        elif isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Load) and is_env(n.value): add(n.slice, n)
        elif isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn)) and is_env(n.comparators[0]): add(n.left, n)
    return reads, bad
_ER = {}
def env_reads_of(files):
    """{switch: float-parsed somewhere} over scripts (SCR-relative); refuses a non-constant name read"""
    out = {}
    for f in files:
        if f not in _ER: _ER[f] = env_reads(os.path.join(SCR, f))
        r, bad = _ER[f]
        if bad: sys.exit(f'{f}: environment read(s) with a non-constant name at line(s) {bad}: name the switch (G45.10)')
        for k, v in r.items(): out[k] = out.get(k, False) or v
    return out

# ---------------------------------------------------------------- report-only outputs (r13 S-6)
# Rule: an output counts for currency only if a later stage or the product reads it. Report-only files — evaluations, diagnostics, non-production
# stand maps / acceptance records, logs — are listed in the state but neither enter the output fingerprint nor make a stage stale when deleted or
# changed. The production seed's history acceptance record (dynamics_check_<name>_history_s<model_seed>.json) is NOT report-only: verify T15 reads it.
REPORT_ONLY = ['eval_*.json', 'dynamics_check_*.json', 'stand_veg_q_*.parquet', '*.log', '*.out']
def report_only(rel, prod_b, keep=()):
    b = os.path.basename(rel)
    if b in keep: return False
    if any(fnmatch.fnmatch(b, p) for p in REPORT_ONLY): return True
    m = re.fullmatch(r'stand_(?:veg|cohorts)_v3_B(\d+)\.parquet', b)
    return bool(m) and int(m.group(1)) != prod_b

def check_config(cfg):
    """G45.2 / G45.9: list-typed keys are non-empty lists of the right type, env sections hold scalar switches their stages read (region switches in
    the top-level env only), no empty values, no float for a switch no script parses as a float; normalises null sections and use_b (0 / 1)"""
    if 'env' in cfg and cfg['env'] is None: cfg['env'] = {}
    if 'table' in cfg and cfg['table'] is None: cfg['table'] = {}
    for s in ('train', 'dynamics', 'downscale', 'map'):
        if not isinstance(cfg.get(s), dict): sys.exit(f'config section {s} missing or not a mapping')
        if cfg[s].get('env') is None: cfg[s]['env'] = {}
    ms, seeds = cfg.get('model_seed'), cfg['train'].get('seeds')
    if not isinstance(seeds, list) or not seeds or not all(isinstance(x, int) and not isinstance(x, bool) for x in seeds):
        sys.exit(f'train.seeds must be a non-empty YAML list of integer seeds, got {seeds!r}')
    if not isinstance(ms, int) or isinstance(ms, bool): sys.exit(f'model_seed must be an integer seed (one of train.seeds), got {ms!r} ({type(ms).__name__})')
    if ms not in seeds: sys.exit(f"model_seed {ms} is not in train.seeds {seeds}")          # the production seed must be one this config trains (r11 I-1)
    el = cfg['dynamics'].get('eval_loc')
    if not isinstance(el, list) or not el or not all(isinstance(x, str) and x for x in el):
        sys.exit(f'dynamics.eval_loc must be a non-empty YAML list of location models (e.g. [history]), got {el!r}')
    if 'history' not in el: sys.exit(f"dynamics.eval_loc {el} lacks 'history'")              # the production location model's acceptance must be tested (r13 D-5)
    ub = cfg['map'].get('use_b')
    if not isinstance(ub, list) or not ub or not all(isinstance(x, bool) or (isinstance(x, int) and x in (0, 1)) for x in ub):
        sys.exit(f'map.use_b must be a non-empty YAML list of booleans (true / false or 1 / 0), got {ub!r}')
    cfg['map']['use_b'] = [int(x) for x in ub]
    lib = set(LIB_WHOLE) | set(LIB_DEFS) | {'stands_cfg.py'}
    for nm, sec in env_sections(cfg):
        if not isinstance(sec, dict): sys.exit(f'config env section {nm} must be a mapping, got {sec!r}')
        reads = env_reads_of(sorted({c for st in SECTION_STAGES[nm] for c in STAGES[st]['code']} | lib))
        for k in list(sec):
            v = sec[k]
            if k in DRIVER_SET: sys.exit(f'config sets {k}, which only the driver sets (r13 D-6)')
            if k not in STAGE_SWITCHES: sys.exit(f'config env key {k} is not a stage switch (typo?)')
            if k in REGION_SWITCHES and nm != 'env': sys.exit(f'{nm}.{k}: region switches belong in the top-level env (every stage must see the same region; G45.9)')
            if k not in reads: sys.exit(f'{nm}.{k}: none of the scripts of the stages this section is passed to ({", ".join(SECTION_STAGES[nm])}) reads {k} (G45.9)')
            if v is None or v == '': sys.exit(f'{nm}.{k} is empty: give a value or remove the key (G45.2)')
            if isinstance(v, (list, dict)): sys.exit(f'{nm}.{k} = {v!r}: an env switch must be a scalar')
            if isinstance(v, float) and not reads[k]: sys.exit(f'{nm}.{k} = {v!r}: a float, but no script parses {k} as a float (an integer / flag switch; G45.2)')
            sec[k] = envstr(v)
    return cfg

class _Stop(Exception):
    pass
def _on_signal(signum, frame):                                   # SIGTERM / SIGINT: stop the running stage's process group, then exit (G45.4)
    signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN); raise _Stop(signum)
def _kill_group(p):
    """terminate a stage's process group (SIGTERM, then SIGKILL after 10 s) and reap it"""
    try: os.killpg(p.pid, signal.SIGTERM)
    except OSError: pass
    try: p.wait(timeout=10)
    except subprocess.TimeoutExpired: pass
    try: os.killpg(p.pid, signal.SIGKILL)                        # (members that outlived the leader)
    except OSError: pass
    p.wait()
class _Tee:
    """--log: the driver's stdout / stderr also written to a file (G45.11)"""
    def __init__(self, a, b): self.a, self.b = a, b
    def write(self, x): self.a.write(x); self.b.write(x); self.b.flush(); return len(x)
    def flush(self): self.a.flush(); self.b.flush()
    def __getattr__(self, k): return getattr(self.a, k)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=os.path.join(ROOT, 'config', 'genveg_v3b.yml')); ap.add_argument('--set', action='append', default=[], dest='sets')
    ap.add_argument('--force', default=''); ap.add_argument('--adopt', default=''); ap.add_argument('--log', default=None); ap.add_argument('stages', nargs='*')
    a = ap.parse_args(); force = set(filter(None, a.force.split(','))); adopt = set(filter(None, a.adopt.split(',')))
    OWN = set()                                                                # files the driver itself writes into the data dir, by (device, inode)
    if a.log:
        os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True); logf = open(a.log, 'a'); s_ = os.fstat(logf.fileno()); OWN.add((s_.st_dev, s_.st_ino))
        logf.write(f'\n# {time.strftime("%F %T")} genveg_pipeline.py {" ".join(sys.argv[1:])}\n'); sys.stdout = _Tee(sys.stdout, logf); sys.stderr = _Tee(sys.stderr, logf)
    # ---- refusals, before anything is read or written
    run = a.stages or ORDER
    bad = [s for s in run if s not in STAGES]
    if bad: sys.exit(f'unknown stage(s) {bad}')
    for flag, names in (('--force', force), ('--adopt', adopt)):                 # (r13 D-2)
        if names - set(STAGES): sys.exit(f'{flag}: unknown stage(s) {sorted(names - set(STAGES))}')
        if names - set(run): sys.exit(f'{flag}: stage(s) {sorted(names - set(run))} are not among the stages being run {run}')
    if force & adopt: sys.exit(f'stage(s) {sorted(force & adopt)} both forced and adopted')
    if adopt & set(CHECKS): sys.exit(f'--adopt: {sorted(adopt & set(CHECKS))} are check stages — their result is their exit status; run them (G45.1)')
    run = [s for s in ORDER if s in run]                                           # always in pipeline order
    need, todo = set(), list(run)                                                  # the stages run or checked: run stages + their ups and gates (G45.10)
    while todo:
        s = todo.pop()
        if s not in need: need.add(s); todo += STAGES[s]['up'] + STAGES[s].get('gate', [])
    from check_comments import swallowed                                         # refuse to run code that a comment swallowed (G12; r13 D-10)
    hits = [(f, ln, fr) for f in scanned_scripts() for ln, fr in swallowed(os.path.join(SCR, f))]
    if hits: sys.exit('code inside comments:\n' + '\n'.join(f'  {f}:{ln}: {fr}' for f, ln, fr in hits))
    unk = set(env_reads_of(hashed_scripts())) - STAGE_SWITCHES
    if unk: sys.exit(f'stage scripts read environment switch(es) {sorted(unk)} not in STAGE_SWITCHES (they would be inherited from the shell): add them')
    raw = yaml.safe_load(open(a.config))
    if raw.get('frozen'): sys.exit(f'config {a.config} is frozen: {raw["frozen"]}')        # on the RAW config: --set frozen=… cannot bypass it (r13 D-1)
    try: cfg = apply_overrides(raw, a.sets)
    except (KeyError, ValueError) as e: sys.exit(f'--set: {e.args[0] if e.args else e}')
    if cfg.get('frozen'): sys.exit(f'config {a.config} is frozen (--set): {cfg["frozen"]}')
    check_config(cfg)
    D = os.path.join(ROOT, 'data', 'proto', cfg['data']); LOGD = os.path.join(D, 'pipeline_logs'); os.makedirs(LOGD, exist_ok=True)
    SR = os.path.join(ROOT, 'data', 'derived', 'runs', cfg['stands_run'])
    if not os.path.exists(f'{SR}/summary.json'): sys.exit(f'stand run {SR} missing: run scripts/stands_pipeline.py --name {cfg["stands_run"]} first')
    import fcntl                                                               # one driver per data dir (r13 D-9); its stages inherit the lock (G45.4)
    lockf = open(os.path.join(D, '.pipeline.lock'), 'a')
    try: fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: sys.exit(f'another genveg driver (or a stage it started) is running on {D}')
    stf = f'{D}/pipeline_state.json'; state = json.load(open(stf)) if os.path.exists(stf) else {}
    def save():                                                                # atomic (r13 D-9)
        tmp = stf + '.tmp'
        with open(tmp, 'w') as f: json.dump(state, f, indent=1); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, stf)
    r4 = cfg.get('env', {}).get('AEF90', '0') == '1'
    _srh = {}
    def srh(k):                                                                # CONTENT of the stand run's outputs genveg reads (lazy)
        if k not in _srh:
            f = os.path.join(SR, STAND_FILES[k])
            if not os.path.exists(f): sys.exit(f'stand run raster missing: {f}')
            _srh[k] = pixel_hash(f)
        return _srh[k]
    inh, dbs = inputs_fingerprint(cfg, r4, SR, D, need)
    hcfg = copy.deepcopy(cfg)                                                  # the hashed view: performance knobs dropped (r13 S-4)
    for k in PERF_TABLE: hcfg.get('table', {}).pop(k, None)
    for _, sec in env_sections(hcfg):
        for k in PERF_ENV: sec.pop(k, None)
    def cget(k):                                                             # a config section or a dotted key (verify.n_upstream)
        v = hcfg
        for part in k.split('.'): v = v.get(part) if isinstance(v, dict) else None
        return v
    def stands_of(st): return STAGES[st]['stands'] + ([] if r4 else STAGES[st].get('stands_gafl', []))
    prod_b = cfg['map']['use_b'][-1]; keep = (f"dynamics_check_{cfg['name']}_history_s{cfg['model_seed']}.json",)   # (TAG as cmds_of sets it)
    py = [sys.executable, '-I']
    tenv = dict(cfg['train']['env'])
    base_env = dict({k: v for k, v in os.environ.items() if k not in STAGE_SWITCHES}, GV_CURATED_DB=dbs['curated_db'], GV_FIADB=dbs['fiadb'], STANDS_RUN=SR,
                    DATA=cfg['data'], WORKSPACE=WS, FE_GEOJSON=os.path.join(ROOT, cfg['fe']['geojson']), **cfg.get('env', {}))
    model = f"{cfg['train']['model_prefix']}_s{cfg['model_seed']}.pt"
    def cmds_of(st):
        S = lambda f, *x: py + [os.path.join(SCR, f), *map(str, x)]
        if st == 'table': return [(S('g02b_build_gafl.py'), dict(cfg['table']))]
        if st == 'localisation': return [(S('g02d_localisation.py'), {})]
        if st == 'records': return [(S('g02e_fia_records.py'), {})]
        if st == 'history': return [(S('g02g_history_consistency.py'), {})]
        if st == 'stands_fp': return [(S('g02j_stand_footprints.py'), {})]
        if st == 'vup': return [(S('gv_verify_upstream.py', cfg['verify']['n_upstream']), dict(GV_SKIP_U8='1'))]   # U8 (stand mapping) runs in verify
        if st == 'train': return [(S('g03b_train_gafl.py'), dict(tenv, SEED=str(s), MODEL=f"{cfg['train']['model_prefix']}_s{s}.pt")) for s in cfg['train']['seeds']]
        if st == 'dynamics':
            return [(S('g06_dynamics_check.py'), dict(tenv, **cfg['dynamics']['env'], MODEL=f"{cfg['train']['model_prefix']}_s{s}.pt",
                                                       EVAL_LOC=L, TAG=f"{cfg['name']}_{L}_s{s}", G06_ENFORCE='1' if s == cfg['model_seed'] else '0'))
                    for s in cfg['train']['seeds'] for L in cfg['dynamics']['eval_loc']]   # (only the production seed's acceptance gates mapping: r6 I2)
        if st == 'downscale': return [(S('g04_bayes_downscale.py'), dict(tenv, **cfg['downscale']['env'], MODEL=model))]
        if st == 'map':
            out = [(S('g05_map_stands.py'), dict(tenv, **cfg['map']['env'], MODEL=model, USE_B=str(b))) for b in cfg['map']['use_b']]
            b = cfg['map']['use_b'][-1]                                       # the production map: written under its own name only after every run (r6 M2)
            out.append((['cp', f'{D}/stand_veg_v3_B{b}.parquet', f'{D}/stand_veg_v3.parquet'], {}))
            out.append((['cp', f'{D}/stand_cohorts_v3_B{b}.parquet', f'{D}/stand_cohorts_v3.parquet'], {}))
            return out
        if st == 'fe':
            gj = os.path.join(ROOT, cfg['fe']['geojson'])
            return [(S('24_export_fe.py', os.path.join(ROOT, cfg['fe']['area']), gj, f'{SR}/finalize/stands.gpkg'), {}), (S('g07_fe_attach.py', gj, cfg['fe']['tag']), {})]
        tab_ = {k: str(cfg['table'][k]) for k in ('TOPK', 'NCAND', 'RADIAL_P') if k in cfg.get('table', {})}   # (gv_verify T1q checks meta against them)
        if st == 'vtab': return [(S('gv_verify.py', cfg['verify']['n']), dict(tenv, **tab_, GV_REQUIRE_MAP='0'))]
        if st == 'verify': return [(S('gv_verify.py', cfg['verify']['n']), dict(tenv, **tab_, MODEL=model, GV_REQUIRE_MAP='1', GV_DYN_JSON=f'{D}/{keep[0]}')),
                                   (S('gv_verify_upstream.py', cfg['verify']['n_upstream']), dict(GV_REQUIRE_MAP='1'))]
    # ---- the check stages' inputs = their verifiers' VERIFY_INPUTS / VERIFY_CODE under the stage's own env (G45.7), re-evaluated after files changed
    def verifier_lists(st):
        ins, code = set(), set()
        for args, env in cmds_of(st):
            f = args[len(py)] if len(args) > len(py) else ''
            if os.path.basename(f) not in VERIFIERS: continue
            r = subprocess.run(py + ['-c', _VI, f], env=dict(base_env, **env), cwd=ROOT, capture_output=True, text=True)
            o = r.stdout.rsplit(_VI_MARK, 1)
            if r.returncode != 0 or len(o) != 2:
                sys.exit(f'{st}: cannot evaluate VERIFY_INPUTS of {os.path.basename(f)} under its env (exit {r.returncode}):\n{(r.stderr or r.stdout).strip()[-1500:]}')
            d_ = json.loads(o[1]); ins |= set(d_['inputs']); code |= set(d_['code'])
        return sorted(ins), sorted(code)
    VL = {st: verifier_lists(st) for st in ORDER if st in need and any(os.path.basename(x[len(py)]) in VERIFIERS for x, _ in cmds_of(st) if len(x) > len(py))}
    gen, vl_at = [0], {st: 0 for st in VL}
    def vfp(st):
        ins, code = VL[st]
        return dict(verify_inputs=hashlib.sha256('|'.join(path_fp(p, D, SR, srh) for p in ins).encode()).hexdigest()[:16],
                    verify_code=hashlib.sha256('|'.join(f'{os.path.relpath(p, ROOT)}:' + (hashlib.sha256(open(p, 'rb').read()).hexdigest() if os.path.exists(p) else 'missing')
                                                        for p in code).encode()).hexdigest()[:16])
    # stage hash = its config sections, code, region switches and inputs, and per upstream stage its hash AND the fingerprint of the outputs that run wrote
    # (state[u]['out']): a re-run or --force of a stage (e.g. a non-deterministic retrain) makes every downstream stage stale (audit 2026-10-26 F11)
    H = {}
    def rehash():
        for st in ORDER:
            if st not in need: continue
            d = STAGES[st]
            H[st] = hashlib.sha256(json.dumps(dict(stage=st, cfg={k: cget(k) for k in d['cfg']}, data=cfg['data'], code=code_hash(d['code']),
                                                   up={u: (H[u], state.get(u, {}).get('out')) for u in d['up']}, stands={s: srh(s) for s in stands_of(st)},
                                                   **({'env': hcfg['env']} if hcfg.get('env') else {}), **({'inputs': inh[st]} if st in inh else {}),
                                                   **(vfp(st) if st in VL else {})),
                                              sort_keys=True, default=str).encode()).hexdigest()[:12]
    rehash()
    def deps(st):                                                              # the stages whose currency decides st's: ups (recursively) + its gates
        out, todo = set(STAGES[st].get('gate', [])), [st]
        while todo:
            for u in STAGES[todo.pop()]['up']:
                if u not in out: out.add(u); todo.append(u)
        return out | {st}
    def closure(st):                                                           # st and every stage it depends on (ups and gates, recursively)
        out, todo = set(), [st]
        while todo:
            s = todo.pop()
            if s not in out: out.add(s); todo += STAGES[s]['up'] + STAGES[s].get('gate', [])
        return out
    def ensure(sts):
        """verifier lists of these stages up to date with the data dir; refused when one lists a data-dir file a LATER stage writes (it could never be current)"""
        todo = [s for s in sts if s in VL and vl_at[s] != gen[0]]
        for s in todo: VL[s] = verifier_lists(s); vl_at[s] = gen[0]
        if todo: rehash()
        for s in sts:
            if s not in VL: continue
            cl = closure(s); rel = [os.path.relpath(os.path.abspath(p), D) for p in VL[s][0] if _under(os.path.abspath(p), D)]
            late = sorted({f'{k} ({W[k][0]})' for k in W for r in rel if (k == r or k.startswith(r + os.sep)) and W[k][0] in STAGES and W[k][0] not in cl})
            if late: sys.exit(f'!! {s}: its verifier lists data-dir file(s) written by stages that run after it: {late[:8]} — it could never be current; '
                              f'the verifier must not list them under this stage (G45.7)')
    # the driver's own stdout / stderr when redirected into the data dir (r9 I-2), by (device, inode): portable (r13 I-4); --log's file (G45.11)
    for fd in (1, 2):
        try: s_ = os.fstat(fd); OWN.add((s_.st_dev, s_.st_ino))
        except OSError: pass
    SKIP = {'pipeline_state.json', 'pipeline_state.json.tmp', '.pipeline.lock'}
    def snap():
        """(size, mtime) of every file in the data dir — logs, the pipeline state / lock, the driver's own output and top-level *.out / *.log
        excluded. No other process may write into the data dir during a run: its files would be attributed to the running stage."""
        out = {}
        for root, _, fs in os.walk(D):
            if _under(os.path.abspath(root), LOGD): continue
            for f in fs:
                if root == D and (f in SKIP or f.endswith('.out') or f.endswith('.log')): continue
                fp = os.path.join(root, f); rel = os.path.relpath(fp, D)
                if os.path.islink(fp) and not os.path.exists(fp): print(f'   (skipping dangling symlink {rel})'); continue   # (r13 D-8)
                try: st_ = os.stat(fp)
                except OSError as e: sys.exit(f'cannot stat {fp}: {e}')
                if (st_.st_dev, st_.st_ino) in OWN: continue
                out[rel] = [st_.st_size, st_.st_mtime_ns]
        return out
    def out_fp(files):                                                         # the output fingerprint: the non-report-only files
        return hashlib.sha256(json.dumps(sorted((k, v) for k, v in files.items() if not report_only(k, prod_b, keep))).encode()).hexdigest()[:12]
    def outputs_of(before):
        """the files a stage run created or changed (snapshot diff: no time slack, nothing of the previous stage attributed to it): fingerprint of the
        non-report-only ones + the full list"""
        files = {k: v for k, v in snap().items() if before.get(k) != v}
        return out_fp(files), files
    W = state.setdefault('_writers', {})                                     # file -> [the stage that last wrote it, size, mtime]
    def stat_(fp): return [os.stat(fp).st_size, os.stat(fp).st_mtime_ns] if os.path.exists(fp) else None
    def later(w, st): return bool(w) and w[0] != st and w[0] in STAGES and ORDER.index(w[0]) > ORDER.index(st)
    def files_ok(st):
        """the files a stage's run (or adoption) recorded are unchanged — or were rewritten by a LATER stage of the pipeline (g02e rewrites visits.parquet,
        g02g localisation.parquet: intended) and still are as that stage left them (audit r6 M3 / r7 B1); a rewrite by an earlier stage (localisation
        re-run after history) makes it stale. Report-only files do not count; fe's layer (r13 M-7) does."""
        e = state.get(st, {}); fl = e.get('files')
        if fl is None: return False                                            # (no file list: a failed run, or an adoption before G45.1)
        for rel, rec in fl.items():
            if report_only(rel, prod_b, keep): continue
            cur = stat_(os.path.join(D, rel)); w = W.get(rel)
            if later(w, st):
                if cur != w[1:]: return False
            elif cur != list(rec): return False
        return all(stat_(os.path.join(ROOT, p)) == rec for p, rec in (e.get('ext') or {}).items())
    def current(st):
        """hash matches, the files its run wrote are intact, and every up stage is (recursively) current (r13 D-4); gates are not part of currency"""
        return state.get(st, {}).get('hash') == H[st] and files_ok(st) and all(current(u) for u in STAGES[st]['up'])
    def failed_gates(st):                                                      # gates (of st or its ups) whose last run FAILED (G21 / G45.12)
        return sorted({g for g in STAGES[st].get('gate', []) if state.get(g, {}).get('how') == 'FAILED'} | {g for u in STAGES[st]['up'] for g in failed_gates(u)})
    child = [None]
    def sh(stage, cmds):
        with open(f'{LOGD}/{stage}.log', 'w') as lf:
            for args, env in cmds:
                lf.write(f'$ {" ".join(args)}  [{", ".join(f"{k}={v}" for k, v in env.items())}]\n'); lf.flush()
                # own process group, holding the data-dir lock: a killed driver never frees the dir while a stage of it still writes (G45.4)
                p = child[0] = subprocess.Popen(args, env=dict(base_env, **env), stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True, pass_fds=(lockf.fileno(),))
                try: rc = p.wait()
                except BaseException: _kill_group(p); raise
                child[0] = None
                if rc != 0: return rc
        return 0
    def fe_ext():                                                              # fe's layer (+ g07's cohorts sidecar), checkout-relative (G45.10): size and mtime
        gj = cfg['fe']['geojson']; cj = gj[:-len('.geojson')] + '_cohorts.json' if gj.endswith('.geojson') else gj + '_cohorts.json'
        return {p: stat_(os.path.join(ROOT, p)) for p in (gj, cj) if os.path.exists(os.path.join(ROOT, p))}
    def adopt_stage(st):
        """G45.1: the stage's last recorded files as they are now (a file a LATER stage rewrote as intended keeps this stage's record), refused if any is missing"""
        e = state.get(st, {}); fl = e.get('files')
        if fl is None: sys.exit(f'!! {st}: cannot adopt — no recorded run of it (file list) in {stf}: run it')
        new, miss, own = {}, [], []
        for rel, rec in fl.items():
            cur = stat_(os.path.join(D, rel))
            if cur is None:
                if not report_only(rel, prod_b, keep): miss.append(rel)
                continue
            if later(W.get(rel), st) and cur == W[rel][1:]: new[rel] = list(rec)
            else: new[rel] = cur; own.append(rel)
        ext = {p: stat_(os.path.join(ROOT, p)) for p in (e.get('ext') or {})}
        miss += [p for p, v in ext.items() if v is None]
        if miss: sys.exit(f'!! {st}: cannot adopt — recorded output(s) missing: {miss[:10]}')
        state[st] = dict(hash=H[st], at=time.strftime('%F %T'), how='adopted', out=out_fp(new), files=new, **({'ext': ext} if 'ext' in e else {}))
        for rel in own: W[rel] = [st, *new[rel]]
    signal.signal(signal.SIGTERM, _on_signal); signal.signal(signal.SIGINT, _on_signal)
    try:
        for st in run:
            d = STAGES[st]; ensure(deps(st))
            if st in adopt:                                                    # only on current up stages and gates (r13 D-3)
                nc = [u for u in d['up'] + d.get('gate', []) if not current(u)]
                if nc: sys.exit(f'!! {st}: cannot adopt — upstream stage(s) / gate(s) {nc} not current')
                adopt_stage(st); save(); rehash()
                print(f'== {st}: ADOPTED its recorded outputs as current ({H[st]})'); continue
            if st not in force and current(st):
                gf = failed_gates(st)
                print(f'== {st}: current{" (gate FAILED: " + ", ".join(gf) + ")" if gf else ""} ({state[st]["how"]} {state[st]["at"]})'); continue
            stale = [u for u in d['up'] + d.get('gate', []) if not current(u)]
            if stale: sys.exit(f'!! {st}: upstream stage(s) {stale} not current — run them first')
            if st in ('downscale', 'map', 'verify'): shutil.copy(f'{D}/{model}', f'{D}/model.pt')      # tools that read the default model see the chosen one
            before = snap()                                                    # (every refusal happens before the state is touched: r13 D-8)
            state.pop(st, None); save()                                        # running = not current: a failed (forced) re-run leaves no stale 'current' mark (audit r3 I4)
            t0 = time.time(); print(f'== {st}: running (log {os.path.relpath(LOGD, ROOT)}/{st}.log)', flush=True)
            rc = sh(st, cmds_of(st))
            tail = open(f'{LOGD}/{st}.log').read().strip().splitlines()[-1:] or ['']
            if rc != 0:
                state[st] = dict(how='FAILED', at=time.strftime('%F %T'), rc=rc); save()          # not current (no hash / files); marks gated stages (G45.12)
                print(f'!! {st} FAILED (exit {rc}): {tail[0][:300]}'); sys.exit(1)
            oh, ofl = outputs_of(before); state[st] = dict(hash=H[st], at=time.strftime('%F %T'), how=f'ran {time.time()-t0:.0f}s', out=oh, files=ofl)
            if st == 'fe': state[st]['ext'] = fe_ext()
            for rel, rec in ofl.items(): W[rel] = [st, *rec]
            if ofl: gen[0] += 1
            save()
            rehash()                                                                        # downstream hashes now see this run's outputs
            print(f'   {tail[0][:300]}', flush=True)
    except _Stop as e:
        save(); print(f'!! interrupted (signal {e.args[0]}): the running stage and its processes were stopped; it is not current', file=sys.stderr); sys.exit(128 + e.args[0])
    save()

if __name__ == '__main__':
    main()
