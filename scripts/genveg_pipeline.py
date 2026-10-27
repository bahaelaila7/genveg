"""Generative-vegetation pipeline driver — every stage in order, one configuration (default config/genveg_v3b.yml; docs/GENVEG_PIPELINE.md).
  .venv/bin/python -I scripts/genveg_pipeline.py [--config config/genveg_v3b.yml] [--set key=value ...] [--force stage,...] [--adopt stage,...] [stage ...]
Stages (default all, in order):
  table         g02b_build_gafl.py       visits, cohort tokens, footprint candidates + location prior (forest mask = the stand run's land_clean; AEF90: data/r4/land.tif)
  localisation  g02d_localisation.py     per-plot localisation numbers keyed by FIA plot identity
  records       g02e_fia_records.py      typed FIA events / negative-evidence periods (fia_events, fia_negs, visits.fia_blocked)
  history       g02g_history_consistency.py  candidates consistent with each plot's whole FIA history (cand_hist; overwrites localisation)
  stands_fp     g02j_stand_footprints.py stand footprints of the mapping set from the stand run (stands.parquet, stand_feat.npy)
  vup           gv_verify_upstream.py    upstream logic checks (grid, forest-land rules, stands, FE display; without U8) [check: exit 0 required]
  vtab          gv_verify.py             the table-level data checks before any training (T0-T10; the map checks are skipped) [check: exit 0 required]
  train         g03b_train_gafl.py       temporal JEPA model, one per seed (<model_prefix>_s<seed>.pt)
  dynamics      g06_dynamics_check.py    held-out disturbance-response test of every seed and eval_loc
  downscale     g04_bayes_downscale.py   landscape prior (MDN) + likelihood B + mapping-path calibration (model = model_seed)
  map           g05_map_stands.py        stand vegetation per use_b (stand_veg_v3_B<b>.parquet; the last = stand_veg_v3.parquet) + stand cohorts (stand_cohorts_v3[_B<b>].parquet)
  fe            24_export_fe.py + g07_fe_attach.py   Forest Explorer stands layer of the stand run + GEN_* attributes (the host serves it)
  verify        gv_verify.py + gv_verify_upstream.py   independent recomputation of every stage (gv_verify with the map checks + gv_verify_upstream with U8) [check: exit 0 required]
Outputs are the scripts' own files under data/proto/<data>/. The driver keeps data/proto/<data>/pipeline_state.json: per stage the hash of its config
section (performance knobs excluded), its code (scripts + the genveg shared modules + the definitions it imports from the stand pipeline's modules),
its upstream stages (hash + output fingerprint), the inputs it reads (STAGE_INPUTS) and pixel-content fingerprints of the stand-run rasters it reads.
A stage is current if its hash matches, the files its run wrote are unchanged and its upstream stages are current; a current stage is skipped;
--force re-runs it; --adopt records existing outputs as current WITHOUT running (only for artifacts known to come from this code + config).
Logs: data/proto/<data>/pipeline_logs/<stage>.log. One driver per data dir (a lock); no other process may write into the data dir during a run."""
import os, sys, json, time, argparse, hashlib, subprocess, shutil, glob, ast, re, copy, fnmatch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stands_stages import apply_overrides
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); SCR = os.path.join(ROOT, 'scripts'); WS = os.environ.get('WORKSPACE', os.path.dirname(ROOT))
# shared modules: the genveg ones are hashed whole; features.py / common.py are STAND-pipeline modules, so only the top-level definitions a stage's
# scripts import from them (and what those reference) enter its hash (r13 I-1: an edit to the stand featurizer must not retrain genveg)
LIB_WHOLE = ['stand_runs.py', 'region.py']
LIB_DEFS = ['features.py', 'common.py']
SCAN_ONLY = ['stands_cfg.py', 'r4_grid.py', 'r4_aef90.py', 'r4_rasters.py', 'genveg_pipeline.py', 'check_comments.py']   # scanned by check_comments, not hashed
# stage: scripts (code hash), config sections, up = hash dependencies (their hash + output fingerprint enter this stage's hash), gate = stages that must
# be current (so they ran and passed) before this one runs WITHOUT entering its hash (G21; r13 S-1 / S-2), stands = stand-run rasters it reads
# (stands_gafl: read only outside AEF90, where data/r4/land.tif replaces the stand run's land_clean; r13 I-2)
STAGES = {
    'table':        dict(code=['g02b_build_gafl.py'], cfg=['table', 'data'], up=[], stands=[], stands_gafl=['clean']),
    'localisation': dict(code=['g02d_localisation.py'], cfg=[], up=['table'], stands=[]),
    'records':      dict(code=['g02e_fia_records.py', 'g02b_build_gafl.py'], cfg=[], up=['table'], stands=[]),
    # (localisation: g02g OVERWRITES localisation.parquet, it does not read it — an ordering gate, r13 S-1)
    'history':      dict(code=['g02g_history_consistency.py', 'g02b_build_gafl.py'], cfg=[], up=['records'], gate=['localisation'], stands=[]),
    'stands_fp':    dict(code=['g02j_stand_footprints.py', 'g02b_build_gafl.py'], cfg=[], up=['table'], stands=['finalize'], stands_gafl=['clean']),   # (clean: g02b.feats' forest filter, r13 A-11)
    'vup':          dict(code=['gv_verify_upstream.py'], cfg=['verify.n_upstream'], up=['stands_fp'], stands=['land', 'clean', 'finalize']),
    # vtab: gv_verify BEFORE training — the table-level data checks (T0-T10, incl. the pre-registered T1k / T1l) must pass before any seed is trained (r11 M-3)
    'vtab':         dict(code=['gv_verify.py', 'g03b_train_gafl.py', 'g02b_build_gafl.py', 'g04_bayes_downscale.py', 'check_comments.py'], cfg=['verify.n', 'train', 'table.TOPK', 'table.NCAND', 'table.RADIAL_P'], up=['history', 'stands_fp'], stands=['finalize'], stands_gafl=['clean']),
    # train: up = history only (g03b reads the table, the FIA records and the history-consistent candidates; its stand map is the opt-in STAND_DIAG
    # diagnostic, so no stands_fp: G27); gated on the upstream checks (vup, r10 M-1) and the table checks (vtab, r11 M-3)
    'train':        dict(code=['g03b_train_gafl.py'], cfg=['train'], up=['history'], gate=['vup', 'vtab'], stands=[]),
    'dynamics':     dict(code=['g06_dynamics_check.py', 'g03b_train_gafl.py', 'g04_bayes_downscale.py'], cfg=['dynamics', 'train', 'model_seed'], up=['train'], stands=[]),   # (model_seed: whose acceptance is enforced; r8 I1)
    # (dynamics: its acceptance gates mapping, audit r5 I5; g04 reads none of its files — a gate, r13 S-2)
    'downscale':    dict(code=['g04_bayes_downscale.py', 'g03b_train_gafl.py'], cfg=['downscale', 'train', 'model_seed'], up=['train'], gate=['dynamics'], stands=[]),
    'map':          dict(code=['g05_map_stands.py', 'g04_bayes_downscale.py', 'g03b_train_gafl.py'], cfg=['map', 'train', 'model_seed'], up=['downscale', 'stands_fp'], stands=['finalize'], stands_gafl=['clean']),
    'fe':           dict(code=['24_export_fe.py', 'g07_fe_attach.py'], cfg=['fe'], up=['map'], stands=[]),
    # (dynamics: T15 re-derives the production seed's acceptance from its dynamics_check file)
    'verify':       dict(code=['gv_verify.py', 'gv_verify_upstream.py', 'g03b_train_gafl.py', 'g02b_build_gafl.py', 'check_comments.py'], cfg=['verify', 'train', 'table.TOPK', 'table.NCAND', 'table.RADIAL_P'], up=['fe', 'history', 'dynamics'], stands=['finalize', 'land', 'clean']),
}
ORDER = list(STAGES)
def scanned_scripts():
    """every script check_comments scans (check_comments.pipeline_scripts; gv_verify T0): stage code, shared modules, stands_cfg, the R4 builders, driver"""
    return sorted({c for d in STAGES.values() for c in d['code']} | set(LIB_WHOLE) | set(LIB_DEFS) | set(SCAN_ONLY))

STAND_FILES = dict(clean='clean/land_clean.tif', land='land/land.tif', finalize='finalize/stands.tif')
# every switch a stage script reads from its environment: never inherited from the shell (an exported AEF90 / REGION / ARCH … would otherwise build or
# train another region / architecture under this config's hash; audit 2026-10-26 P1) — only the config (top-level env, stage env) sets them. The driver
# refuses to run a stage script that reads a switch missing here.
STAGE_SWITCHES = {'STAND_DIAG', 'GV_REQUIRE_MAP', 'GV_CURATED_DB', 'GV_FIADB', 'VAL_SPLIT', 'G06_ENFORCE', 'AEF90', 'ARCH', 'RADIAL_P', 'T_PAIR', 'SITE_ATTR', 'PAIR_ALL', 'PAIR_HMAX', 'MAP_Y0', 'AGE_CH', 'W_AGE', 'SMOKE_BATCHES', 'BATCH', 'BUF_KM', 'CF16', 'CURRIC', 'DATA', 'DZ', 'EM_ITERS', 'EP8', 'EPB', 'EPOCHS', 'EVAL_LOC', 'FE_GEOJSON', 'G06_SET',
                  'GV_SKIP_U8', 'KC', 'LOAD', 'LOC', 'LR', 'MDN_EP', 'MIN_FSHARE', 'MODEL', 'NCAND', 'NPROC', 'NT', 'PAIR_ANCHOR', 'QD_WIN', 'RAD_M', 'REGION',
                  'SEED', 'SEVERITY_ANCHOR', 'STANDS_RUN', 'TAG', 'TOPK', 'USE_B', 'WRITE_LOC', 'W_COMP', 'W_SUM',
                  'STANDS_STAGE_JSON',                       # stands_cfg (imported by features): a stand-pipeline stage context must not leak in (r13 I-1)
                  'WORKSPACE', 'SIGREG_MEAN', 'JUMP_RESET', 'STATE_ATTR',   # (G29-G31)
                  'EVK_REGION_OK',                           # (G34: g02b refuses a non-GA+FL table until the region's evidence table is re-derived)
                  'GV_DYN_JSON'}                             # (gv_verify T15: an explicit dynamics acceptance file)
# switches only the driver sets (per stage / from other config keys): a config env section setting one is refused (r13 D-6)
DRIVER_SET = {'GV_SKIP_U8', 'LOAD', 'SMOKE_BATCHES', 'STAND_DIAG', 'G06_ENFORCE', 'GV_REQUIRE_MAP', 'MODEL', 'SEED', 'USE_B', 'TAG', 'EVAL_LOC', 'DATA',
              'STANDS_RUN', 'WORKSPACE', 'FE_GEOJSON', 'GV_CURATED_DB', 'GV_FIADB', 'STANDS_STAGE_JSON'}
# performance knobs: passed to the stages but kept out of every hash (r13 S-4). Thread counts can change the order of float reductions, so a re-run
# with another NT / NPROC need not be bit-identical — it is not a different model.
PERF_TABLE = {'NPROC'}; PERF_ENV = {'NT'}
def env_sections(cfg):
    """the config sections passed to stage scripts as environment variables"""
    return [s for s in (cfg.get('env'), cfg.get('table'), cfg['train'].get('env'), cfg['dynamics'].get('env'), cfg['downscale'].get('env'), cfg['map'].get('env')) if s is not None]
def envstr(v):                                               # YAML booleans as the scripts test them ('1' / '0'; r13 D-7)
    return '1' if v is True else '0' if v is False else str(v)

# ---------------------------------------------------------------- inputs each stage READS (r12 I-1, G25), fingerprinted into that stage's hash only
# rasters / tables: path + size + mtime (the mosaics are GBs); vector files: their CONTENT; small text files (the stand run's resolved config): content
DIST_CODES = 'data/landfire/dist/dist_codes.csv'
GAFL_RASTERS = ['data/aef/aef_2022.tif', 'data/aef/aef_2025.tif', 'data/landfire/dist/dist_*.tif', 'data/hansen/lossyear.tif', 'data/soils/*.tif']
# Region 4 (AEF90) rasters the table / footprint / check stages read instead (audit P2); the code that built them is scanned, not hashed (r13 S-3)
R4_INPUTS = ['data/r4/land.tif', 'data/r4/hansen_lossyear.tif', 'data/r4/dist/dist_*.tif', 'data/r4/soils90/*.tif', 'data/r4/soils90/fill.json',
             'data/aef90/aef90_2022.tif', 'data/aef90/aef90_2025.tif', 'data/grid/grid_r4.json', 'data/grid/grid_r4_90.json', 'data/grid/aoi_r4_mask90.tif']
R4_VERIFY_EXTRA = ['data/aef/aef_2022.tif', 'data/aef/aef_2025.tif']          # gv_verify T3c2 compares the 90 m means with the 30 m mosaics
R4_TIGER = [f'TIGER/tl_2024_us_county/tl_2024_us_county.{x}' for x in ('shp', 'shx', 'dbf', 'prj')]   # under WORKSPACE (gv_verify T1 state polygons; r13 M-9)
# gv_verify_upstream without U8 (r11 I-2, r13 M-3 / M-4): U1 grids, U6 forest-land rules, U7 stand polygons, U9 the FE stands.js; U1r / U6r under AEF90
SOILS6 = ('clay030', 'sand030', 'soc030', 'awc030', 'ph030', 'restrdepth')
VUP_FILES = ['data/grid/grid.json', 'data/aef/aef_2022.tif', 'data/aef/aef_2025.tif', 'data/hansen/lossyear.tif', 'data/hansen/treecover2000.tif',
             'data/hansen/datamask.tif', 'data/landfire/evt2022.tif', 'data/landfire/evt_classes.csv', 'data/treemap/tm_id.tif', 'data/grid/aoi_mask.tif',
             *[f'data/landfire/dist/dist_{y}.tif' for y in (2000, 2012, 2024, 2025)], *[f'data/soils/{k}.tif' for k in SOILS6]]
VUP_R4 = ['data/grid/grid_r4.json', 'data/grid/grid_r4_90.json', 'data/r4/land.tif', 'data/r4/hansen_lossyear.tif', 'data/r4/hansen_treecover2000.tif',
          'data/r4/hansen_datamask.tif', *[f'data/r4/dist/dist_{y}.tif' for y in (2000, 2012, 2024, 2025)], 'data/grid/aoi_r4_mask90.tif',
          'data/aef90/aef90_2022.tif', 'data/aef90/aef90_2025.tif', *[f'data/r4/soils90/{k}.tif' for k in SOILS6],
          'data/landfire/raw/LF2022_EVT_CONUS/Tif/LF2022_EVT_CONUS.tif']
VUP_WS = ['fe_pan/apps/forest-explorer/public/stands.js']; VUP_R4_WS = ['FVSclaude/treemap/Data/TreeMap2022_CONUS.tif']
# files in the data dir that no stage writes but gv_verify reads when they exist (g02c / g02f analyses; r13 M-5)
D_EXTRA = ['visit_aef/*.npy', 'cand_prior.parquet', 'location_em.json', 'cand_loc_aef.parquet']
def stage_inputs(st, cfg, r4):
    """{kind: [paths]}: db (config keys), req (ROOT globs, must exist), opt (ROOT globs, 'missing' recorded), ws (WORKSPACE globs, optional),
    data (data-dir globs, optional), content (content-hashed: ROOT path or SR:<path in the stand run>), vec (vector files: STANDS_GPKG, cfg:<key> or a ROOT path)"""
    I = dict(db=(), req=[], opt=[], ws=[], data=[], content=[], vec=[])
    feat = R4_INPUTS if r4 else GAFL_RASTERS
    def vup_part():
        return dict(opt=VUP_FILES + (VUP_R4 if r4 else []), ws=VUP_WS + (VUP_R4_WS if r4 else []), content=['SR:config_resolved.yml'] + (['config/pilot.yml'] if r4 else []), vec=['STANDS_GPKG'])
    def vtab_part():
        return dict(db=('curated_db', 'fiadb'), req=[DIST_CODES, *feat, *(R4_VERIFY_EXTRA if r4 else [])], ws=R4_TIGER if r4 else [], data=D_EXTRA)
    if st == 'table': I.update(db=('curated_db', 'fiadb'), req=[DIST_CODES, *feat])
    elif st == 'records': I.update(db=('fiadb',))
    elif st == 'stands_fp': I.update(req=[DIST_CODES, *feat], vec=['STANDS_GPKG', 'data/derived/baker_county.gpkg'])
    elif st == 'vup': I.update(vup_part())
    elif st == 'vtab': I.update(vtab_part())
    elif st == 'train':
        loc = str(cfg['train']['env'].get('LOC', 'history'))         # g03b's optional location posteriors (LOC=aef / plot)
        I.update(req=['data/proto/baker/plots.parquet'], data={'aef': ['cand_loc_aef.parquet'], 'plot': ['cand_prior.parquet']}.get(loc, []))
    elif st == 'map': I.update(vec=['STANDS_GPKG'])
    elif st == 'fe': I.update(vec=['STANDS_GPKG', 'cfg:fe.area'])                    # (24_export_fe's area polygon = fe.area; r13 M-1)
    elif st == 'verify':                                                               # gv_verify (vtab's inputs) + gv_verify_upstream (vup's; r13 M-2)
        a, b = vtab_part(), vup_part(); I.update(a); I.update(opt=b['opt'], ws=a['ws'] + b['ws'], content=b['content'], vec=b['vec'])
    return I
def vector_hash(path):
    """content fingerprint of a vector file (per layer: CRS, column names + dtypes, attributes, geometries as WKB; row order kept), cached next to it by (size, mtime)"""
    import geopandas as gpd
    real = os.path.realpath(path); st = os.stat(real); cf = real + '.vechash.json'
    if os.path.exists(cf):
        c = json.load(open(cf))
        if c.get('v') == 2 and c.get('size') == st.st_size and c.get('mtime') == st.st_mtime_ns: return c['hash']
    h = hashlib.sha256()
    for layer in sorted(__import__('pyogrio').list_layers(real)[:, 0]):
        g = gpd.read_file(real, layer=layer); h.update(layer.encode()); h.update(str(g.crs).encode())
        h.update(json.dumps([(str(c), str(t)) for c, t in g.dtypes.items()]).encode())          # (column names + dtypes: r13 M-8)
        h.update(pd_bytes(g.drop(columns='geometry'))); h.update(b''.join(g.geometry.to_wkb()))
    out = h.hexdigest()[:16]; json.dump(dict(v=2, size=st.st_size, mtime=st.st_mtime_ns, hash=out), open(cf, 'w')); return out
def pd_bytes(df):
    import pandas as pd
    return pd.util.hash_pandas_object(df, index=True).values.tobytes()
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
def inputs_fingerprint(cfg, r4, sr, D):
    """per-stage input fingerprints {stage: hash} and the configured database paths"""
    inp = cfg.get('inputs', {})
    dbs = {'curated_db': inp.get('curated_db', 'landis_region4/data_r4/slim_r4.duckdb'), 'fiadb': inp.get('fiadb', 'FIA/FIASQLITE2PGSQL/FIADB.duckdb')}
    dbfp = {}
    for k, rel in dbs.items():                       # the CONFIGURED path (relative to WORKSPACE) + size / mtime: a moved checkout keeps its hashes (r13 I-3)
        f = os.path.join(WS, rel)
        if not os.path.exists(f): sys.exit(f'input database missing: {k} = {f}')
        st = os.stat(os.path.realpath(f)); dbfp[k] = f'{k}:{rel}:{st.st_size}:{st.st_mtime_ns}'
    def stamp(base, pats, need):
        out = []
        for pat in pats:
            fs = sorted(glob.glob(os.path.join(base, pat)))
            if not fs:
                if need: sys.exit(f'input missing: {pat}')
                out.append(f'{pat}:missing'); continue
            out += [f'{os.path.relpath(f, base)}:{os.stat(f).st_size}:{os.stat(f).st_mtime_ns}' for f in fs]
        return out
    def content(p):
        f = os.path.join(sr, p[3:]) if p.startswith('SR:') else os.path.join(ROOT, p)
        return f'{p}:' + (hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16] if os.path.exists(f) else 'missing')   # (r13 S-5: the stand run rewrites it every run)
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
        I = stage_inputs(st_, cfg, r4)
        parts = [dbfp[k] for k in I['db']] + stamp(ROOT, I['req'], True) + stamp(ROOT, I['opt'], False) + [f'ws:{x}' for x in stamp(WS, I['ws'], False)]
        parts += [f'D:{x}' for x in stamp(D, I['data'], False)] + [content(p) for p in I['content']] + [vec(v) for v in I['vec']]
        if parts: out[st_] = hashlib.sha256('|'.join(parts).encode()).hexdigest()[:16]
    return out, dbs

# ---------------------------------------------------------------- code
def _bound(s):
    """the module-level names a top-level statement binds"""
    if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): return {s.name}
    if isinstance(s, (ast.Import, ast.ImportFrom)): return {(a.asname or a.name).split('.')[0] for a in s.names}
    return {n.id for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
def defs_hash(mod, names):
    """hash of the top-level definitions of `names` in module `mod` and, transitively, of the module-level names they reference (ast dump: comments
    and formatting do not count); '*' = the whole file"""
    src = open(os.path.join(SCR, mod)).read()
    if '*' in names: return hashlib.sha256(src.encode()).hexdigest()
    body = ast.parse(src).body; binds = {}
    for i, s in enumerate(body):
        for nm in _bound(s): binds.setdefault(nm, []).append(i)
    keep, todo, seen = set(), list(names), set()
    while todo:
        nm = todo.pop()
        if nm in seen: continue
        seen.add(nm)
        if nm not in binds:
            if nm in names: sys.exit(f'{mod}: {nm} (imported by a genveg stage) is not defined at module level')
            continue
        for i in binds[nm]:
            if i not in keep: keep.add(i); todo += [n.id for n in ast.walk(body[i]) if isinstance(n, ast.Name)]
    return hashlib.sha256('\n'.join(f'{mod}:{ast.dump(body[i])}' for i in sorted(keep)).encode()).hexdigest()
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
ENV_READ = re.compile(r"""environ(?:\.get)?\s*[\(\[]\s*['"]([A-Z][A-Z0-9_]*)['"]|getenv\(\s*['"]([A-Z][A-Z0-9_]*)['"]|['"]([A-Z][A-Z0-9_]*)['"]\s+(?:not\s+)?in\s+os\.environ""")
def env_reads(files):
    return {x for f in files for m in ENV_READ.finditer(open(os.path.join(SCR, f)).read()) for x in m.groups() if x}

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

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=os.path.join(ROOT, 'config', 'genveg_v3b.yml')); ap.add_argument('--set', action='append', default=[], dest='sets')
    ap.add_argument('--force', default=''); ap.add_argument('--adopt', default=''); ap.add_argument('stages', nargs='*')
    a = ap.parse_args(); force = set(filter(None, a.force.split(','))); adopt = set(filter(None, a.adopt.split(',')))
    # ---- refusals, before anything is read or written
    run = a.stages or ORDER
    bad = [s for s in run if s not in STAGES]
    if bad: sys.exit(f'unknown stage(s) {bad}')
    for flag, names in (('--force', force), ('--adopt', adopt)):                 # (r13 D-2)
        if names - set(STAGES): sys.exit(f'{flag}: unknown stage(s) {sorted(names - set(STAGES))}')
        if names - set(run): sys.exit(f'{flag}: stage(s) {sorted(names - set(run))} are not among the stages being run {run}')
    if force & adopt: sys.exit(f'stage(s) {sorted(force & adopt)} both forced and adopted')
    run = [s for s in ORDER if s in run]                                           # always in pipeline order
    from check_comments import swallowed                                         # refuse to run code that a comment swallowed (G12; r13 D-10)
    hits = [(f, ln, fr) for f in scanned_scripts() for ln, fr in swallowed(os.path.join(SCR, f))]
    if hits: sys.exit('code inside comments:\n' + '\n'.join(f'  {f}:{ln}: {fr}' for f, ln, fr in hits))
    hashed = sorted({c for d in STAGES.values() for c in d['code']} | set(LIB_WHOLE) | set(LIB_DEFS) | {'stands_cfg.py'})
    unk = env_reads(hashed) - STAGE_SWITCHES
    if unk: sys.exit(f'stage scripts read environment switch(es) {sorted(unk)} not in STAGE_SWITCHES (they would be inherited from the shell): add them')
    raw = yaml.safe_load(open(a.config))
    if raw.get('frozen'): sys.exit(f'config {a.config} is frozen: {raw["frozen"]}')        # on the RAW config: --set frozen=… cannot bypass it (r13 D-1)
    cfg = apply_overrides(raw, a.sets)
    if cfg.get('frozen'): sys.exit(f'config {a.config} is frozen (--set): {cfg["frozen"]}')
    for sec in env_sections(cfg):
        for k in list(sec):
            if k in DRIVER_SET: sys.exit(f'config sets {k}, which only the driver sets (r13 D-6)')
            if k not in STAGE_SWITCHES: sys.exit(f'config env key {k} is not a stage switch (typo?)')
            sec[k] = envstr(sec[k])
    if cfg['model_seed'] not in cfg['train']['seeds']:                     # the production seed must be one this config trains (r11 I-1)
        sys.exit(f"model_seed {cfg['model_seed']} is not in train.seeds {cfg['train']['seeds']}")
    if 'history' not in cfg['dynamics']['eval_loc']:                          # the production location model's acceptance must be tested (r13 D-5)
        sys.exit(f"dynamics.eval_loc {cfg['dynamics']['eval_loc']} lacks 'history'")
    D = os.path.join(ROOT, 'data', 'proto', cfg['data']); LOGD = os.path.join(D, 'pipeline_logs'); os.makedirs(LOGD, exist_ok=True)
    SR = os.path.join(ROOT, 'data', 'derived', 'runs', cfg['stands_run'])
    if not os.path.exists(f'{SR}/summary.json'): sys.exit(f'stand run {SR} missing: run scripts/stands_pipeline.py --name {cfg["stands_run"]} first')
    import fcntl                                                               # one driver per data dir (r13 D-9)
    lockf = open(os.path.join(D, '.pipeline.lock'), 'a')
    try: fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: sys.exit(f'another genveg driver is running on {D}')
    stf = f'{D}/pipeline_state.json'; state = json.load(open(stf)) if os.path.exists(stf) else {}
    def save():                                                                # atomic (r13 D-9)
        tmp = stf + '.tmp'
        with open(tmp, 'w') as f: json.dump(state, f, indent=1); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, stf)
    r4 = cfg.get('env', {}).get('AEF90', '0') == '1'
    srh = {k: pixel_hash(os.path.join(SR, f)) for k, f in STAND_FILES.items()}          # CONTENT of the stand run's outputs genveg reads
    inh, dbs = inputs_fingerprint(cfg, r4, SR, D)
    hcfg = copy.deepcopy(cfg)                                                  # the hashed view: performance knobs dropped (r13 S-4)
    for k in PERF_TABLE: hcfg['table'].pop(k, None)
    for sec in env_sections(hcfg):
        for k in PERF_ENV: sec.pop(k, None)
    def cget(k):                                                             # a config section or a dotted key (verify.n_upstream)
        v = hcfg
        for part in k.split('.'): v = v.get(part) if isinstance(v, dict) else None
        return v
    def stands_of(st): return STAGES[st]['stands'] + ([] if r4 else STAGES[st].get('stands_gafl', []))
    # stage hash = its config sections, code, region switches and inputs, and per upstream stage its hash AND the fingerprint of the outputs that run wrote
    # (state[u]['out']): a re-run or --force of a stage (e.g. a non-deterministic retrain) makes every downstream stage stale (audit 2026-10-26 F11)
    H = {}
    def rehash():
        for st in ORDER:
            d = STAGES[st]
            H[st] = hashlib.sha256(json.dumps(dict(stage=st, cfg={k: cget(k) for k in d['cfg']}, data=cfg['data'], code=code_hash(d['code']),
                                                   up={u: (H[u], state.get(u, {}).get('out')) for u in d['up']}, stands={s: srh[s] for s in stands_of(st)},
                                                   **({'env': hcfg['env']} if 'env' in hcfg else {}), **({'inputs': inh[st]} if st in inh else {})),
                                              sort_keys=True, default=str).encode()).hexdigest()[:12]
    rehash()
    prod_b = int(cfg['map']['use_b'][-1]); keep = (f"dynamics_check_{cfg['name']}_history_s{cfg['model_seed']}.json",)   # (TAG as cmds_of sets it)
    # the driver's own stdout / stderr when redirected into the data dir (r9 I-2), by (device, inode): portable (r13 I-4)
    OWN = set()
    for fd in (1, 2):
        try: s_ = os.fstat(fd); OWN.add((s_.st_dev, s_.st_ino))
        except OSError: pass
    SKIP = {'pipeline_state.json', 'pipeline_state.json.tmp', '.pipeline.lock'}
    def snap():
        """(size, mtime) of every file in the data dir — logs, the pipeline state / lock, the driver's own output and top-level *.out / *.log
        excluded. No other process may write into the data dir during a run: its files would be attributed to the running stage."""
        out = {}
        for root, _, fs in os.walk(D):
            if os.path.abspath(root).startswith(os.path.abspath(LOGD)): continue
            for f in fs:
                if root == D and (f in SKIP or f.endswith('.out') or f.endswith('.log')): continue
                fp = os.path.join(root, f); rel = os.path.relpath(fp, D)
                if os.path.islink(fp) and not os.path.exists(fp): print(f'   (skipping dangling symlink {rel})'); continue   # (r13 D-8)
                try: st_ = os.stat(fp)
                except OSError as e: sys.exit(f'cannot stat {fp}: {e}')
                if (st_.st_dev, st_.st_ino) in OWN: continue
                out[rel] = [st_.st_size, st_.st_mtime_ns]
        return out
    def outputs_of(before):
        """the files a stage run created or changed (snapshot diff: no time slack, nothing of the previous stage attributed to it): fingerprint of the
        non-report-only ones + the full list"""
        files = {k: v for k, v in snap().items() if before.get(k) != v}
        return hashlib.sha256(json.dumps(sorted((k, v) for k, v in files.items() if not report_only(k, prod_b, keep))).encode()).hexdigest()[:12], files
    W = state.setdefault('_writers', {})                                     # file -> [the stage that last wrote it, size, mtime]
    def stat_(fp): return [os.stat(fp).st_size, os.stat(fp).st_mtime_ns] if os.path.exists(fp) else None
    def files_ok(st):
        """the files a stage's run wrote are unchanged — or were rewritten by a LATER stage of the pipeline (g02e rewrites visits.parquet, g02g
        localisation.parquet: intended) and still are as that stage left them (audit r6 M3 / r7 B1); a rewrite by an earlier stage (localisation re-run
        after history) makes it stale. Report-only files do not count; fe's layer outside the data dir does (r13 M-7). Adopted stages carry no file list."""
        e = state.get(st, {})
        if e.get('how') == 'adopted': return True
        fl = e.get('files')
        if fl is None: return False
        for rel, rec in fl.items():
            if report_only(rel, prod_b, keep): continue
            cur = stat_(os.path.join(D, rel)); w = W.get(rel)
            if w and w[0] != st and w[0] in STAGES and ORDER.index(w[0]) > ORDER.index(st):
                if cur != w[1:]: return False
            elif cur != list(rec): return False
        return all(stat_(os.path.join(WS, p)) == rec for p, rec in (e.get('ext') or {}).items())
    def current(st):
        """hash matches, the files its run wrote are intact, and every up stage is (recursively) current (r13 D-4); gates are not part of currency"""
        return state.get(st, {}).get('hash') == H[st] and files_ok(st) and all(current(u) for u in STAGES[st]['up'])
    py = [sys.executable, '-I']
    tenv = dict(cfg['train']['env'])
    base_env = dict({k: v for k, v in os.environ.items() if k not in STAGE_SWITCHES}, GV_CURATED_DB=dbs['curated_db'], GV_FIADB=dbs['fiadb'], STANDS_RUN=SR,
                    DATA=cfg['data'], WORKSPACE=WS, FE_GEOJSON=os.path.join(WS, cfg['fe']['geojson']), **cfg.get('env', {}))
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
            gj = os.path.join(WS, cfg['fe']['geojson'])
            return [(S('24_export_fe.py', os.path.join(ROOT, cfg['fe']['area']), gj, f'{SR}/finalize/stands.gpkg'), {}), (S('g07_fe_attach.py', gj, cfg['fe']['tag']), {})]
        tab_ = {k: str(cfg['table'][k]) for k in ('TOPK', 'NCAND', 'RADIAL_P') if k in cfg.get('table', {})}   # (gv_verify T1q checks meta against them)
        if st == 'vtab': return [(S('gv_verify.py', cfg['verify']['n']), dict(tenv, **tab_, GV_REQUIRE_MAP='0'))]
        if st == 'verify': return [(S('gv_verify.py', cfg['verify']['n']), dict(tenv, **tab_, MODEL=model, GV_REQUIRE_MAP='1')), (S('gv_verify_upstream.py', cfg['verify']['n_upstream']), dict(GV_REQUIRE_MAP='1'))]
    def fe_ext():                                                              # fe's layer (+ g07's cohorts sidecar), WORKSPACE-relative: [size, mtime]
        gj = cfg['fe']['geojson']; cj = gj[:-len('.geojson')] + '_cohorts.json' if gj.endswith('.geojson') else gj + '_cohorts.json'
        return {p: stat_(os.path.join(WS, p)) for p in (gj, cj) if os.path.exists(os.path.join(WS, p))}
    for st in run:
        d = STAGES[st]
        if st in adopt:                                                        # only on current up stages and gates (r13 D-3)
            nc = [u for u in d['up'] + d.get('gate', []) if not current(u)]
            if nc: sys.exit(f'!! {st}: cannot adopt — upstream stage(s) / gate(s) {nc} not current')
            state[st] = dict(hash=H[st], at=time.strftime('%F %T'), how='adopted', out=state.get(st, {}).get('out')); save(); rehash()
            print(f'== {st}: ADOPTED existing outputs as current ({H[st]})'); continue
        if st not in force and current(st):
            print(f'== {st}: current ({state[st]["how"]} {state[st]["at"]})'); continue
        stale = [u for u in d['up'] + d.get('gate', []) if not current(u)]
        if stale: sys.exit(f'!! {st}: upstream stage(s) {stale} not current — run them first')
        if st in ('downscale', 'map', 'verify'): shutil.copy(f'{D}/{model}', f'{D}/model.pt')      # tools that read the default model see the chosen one
        before = snap()                                                        # (every refusal happens before the state is touched: r13 D-8)
        state.pop(st, None); save()                                            # running = not current: a failed (forced) re-run leaves no stale 'current' mark (audit r3 I4)
        t0 = time.time(); print(f'== {st}: running (log {os.path.relpath(LOGD, ROOT)}/{st}.log)', flush=True)
        rc = sh(st, cmds_of(st))
        tail = open(f'{LOGD}/{st}.log').read().strip().splitlines()[-1:] or ['']
        if rc != 0:
            print(f'!! {st} FAILED (exit {rc}): {tail[0][:300]}'); save(); sys.exit(1)
        oh, ofl = outputs_of(before); state[st] = dict(hash=H[st], at=time.strftime('%F %T'), how=f'ran {time.time()-t0:.0f}s', out=oh, files=ofl)
        if st == 'fe': state[st]['ext'] = fe_ext()
        for rel, rec in ofl.items(): W[rel] = [st, *rec]
        save()
        rehash()                                                                            # downstream hashes now see this run's outputs
        print(f'   {tail[0][:300]}', flush=True)
    save()

if __name__ == '__main__':
    main()
