"""Stand-delineation pipeline: configuration, stage graph and stage directories (docs/STANDS_PIPELINE.md). Shared by the driver
(scripts/stands_pipeline.py) and every stage script.

A stage script reads its resolved configuration and its input / output directories from the env STANDS_STAGE_JSON written by the driver:
  CFG          the resolved configuration (config file + --set overrides)
  OUT          this stage's output directory (a content-hashed cache dir: data/derived/cache/<stage>-<hash>)
  IN(stage)    the output directory of an upstream stage
  WINDOW       (r0, c0, r1, c1) processing window in pixels, or None = full extent
Run without the driver (no STANDS_STAGE_JSON), a stage falls back to config/pilot.yml and data/derived for every directory (the legacy layout).

The hash of a stage = its configuration keys (STAGES[...]['keys']) + the window (if the stage is windowed) + the code of the stage script and the
shared modules + the hashes of its upstream stages, so a tuning variant re-runs exactly the stages its changes affect and reuses the rest."""
import os, sys, json, hashlib, copy, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, 'scripts')
DEFAULT_CONFIG = os.path.join(ROOT, 'config', 'pilot.yml')
# stage -> script + the local modules it imports (code hash), the configuration keys it depends on, its raw input files (fingerprint: size + mtime;
# '{barriers}' = the configured barrier rasters), upstream stages, whether the window enters its hash
TM = ['treemap/tm_id.tif', 'treemap/vat.parquet', 'treemap/trees.parquet']; SOIL = [f'soils/{k}.tif' for k in ('clay030', 'sand030', 'soc030', 'awc030', 'ph030', 'restrdepth')]
AEF = ['aef/aef_2025.tif', 'aef/aef_2022.tif']
STAGES = {
    'land':     dict(script='10_forest_land.py', mods=['common.py', 'stands_cfg.py'], keys=['forest_land'], up=[], windowed=False,
                     inputs=['grid/grid.json', 'grid/aoi_mask.tif', 'treemap/tm_id.tif', 'landfire/evt2022.tif', 'landfire/evt_classes.csv', 'hansen/lossyear.tif',
                             'hansen/treecover2000.tif', 'hansen/datamask.tif']),
    'clean':    dict(script='11_land_clean.py', mods=['common.py', 'stands_cfg.py'], keys=['segmentation.mmu_ha', 'clean'], up=['land'], windowed=False, inputs=[]),
    'fit':      dict(script='20_fit_features.py', mods=['common.py', 'features.py', 'stands_cfg.py'], keys=['segmentation.aef_pcs', 'fit'], up=['clean'], windowed=False,
                     inputs=AEF + SOIL + TM),
    'sp':       dict(script='21_superpixels.py', mods=['common.py', 'features.py', 'stands_cfg.py'], keys=['segmentation.weights', 'segmentation.superpixel_ha',
                     'segmentation.slic_compactness', 'segmentation.barriers', 'segmentation.aef_pcs', 'segmentation.sp_tile'], up=['clean', 'fit'], windowed=True,
                     inputs=AEF + SOIL + TM + ['{barriers}']),
    'merge':    dict(script='22_merge.py', mods=['features.py', 'rag_merge.py', 'stands_cfg.py'], keys=['segmentation.merge_threshold', 'segmentation.target_median_ha',
                     'segmentation.max_ha', 'segmentation.mmu_ha', 'segmentation.shape_weight', 'segmentation.status_is_hard', 'merge'], up=['sp'], windowed=False, inputs=[]),
    'finalize': dict(script='23_finalize.py', mods=['common.py', 'features.py', 'stands_cfg.py'], keys=['segmentation.mmu_ha', 'segmentation.status_is_hard',
                     'segmentation.barriers', 'finalize'], up=['merge', 'sp', 'clean', 'land'], windowed=True, inputs=TM + SOIL + ['{barriers}']),
}
PXHA = 0.09                                            # ha per 30 m pixel
def mmu_px(cfg):
    """The MMU in pixels — the one threshold every stage, check and metric uses."""
    return int(round(cfg['segmentation']['mmu_ha'] / PXHA))
def barrier_files(cfg):
    fs = [os.path.join(ROOT, 'data', 'barriers', f'{b}.tif') for b in cfg['segmentation']['barriers']]
    miss = [f for f in fs if not os.path.exists(f)]
    if miss: raise FileNotFoundError(f'configured barrier raster(s) missing: {miss}')
    return fs
def grid_shape():
    g = json.load(open(os.path.join(ROOT, 'data', 'grid', 'grid.json'))); return g['height'], g['width']
def sp_extent(cfg, window):
    """The pixel extent the superpixels cover: every sp tile overlapping the window (tile-aligned, outward). finalize / verify / metrics work on it,
    so stands are never cut by the window; stands touching this extent's border are 'edge stands' (cut by the data extent)."""
    H, W = grid_shape()
    if not window: return (0, 0, H, W)
    T = int(cfg['segmentation']['sp_tile']); r0, c0, r1, c1 = window
    return ((r0 // T) * T, (c0 // T) * T, min(-(-r1 // T) * T, H), min(-(-c1 // T) * T, W))
ORDER = list(STAGES)

def parse_value(s):
    return yaml.safe_load(s)

def apply_overrides(cfg, sets):
    """sets: ['segmentation.weights.aef=0.8', ...] (values parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for s in sets or []:
        k, v = s.split('=', 1); path = k.strip().split('.'); d = cfg
        for p in path[:-1]:
            if p not in d or not isinstance(d[p], dict): raise KeyError(f'--set {k}: no section {p}')
            d = d[p]
        if path[-1] not in d: raise KeyError(f'--set {k}: unknown key (typo?) — add it to the config file first')
        d[path[-1]] = parse_value(v)
    return cfg

def get_key(cfg, key):
    d = cfg
    for p in key.split('.'):
        if not isinstance(d, dict) or p not in d: return None
        d = d[p]
    return d

def code_hash(stage):
    h = hashlib.sha256()
    for f in [STAGES[stage]['script']] + STAGES[stage]['mods']:
        h.update(f.encode()); h.update(open(os.path.join(SCRIPTS, f), 'rb').read())
    return h.hexdigest()

def input_fingerprint(stage, cfg):
    out = []
    for f in STAGES[stage]['inputs']:
        for p in (barrier_files(cfg) if f == '{barriers}' else [os.path.join(ROOT, 'data', f)]):
            st = os.stat(p); out.append((os.path.relpath(p, ROOT), st.st_size, st.st_mtime_ns))
    return out

def stage_hashes(cfg, window):
    """stage -> 12-hex hash, in topological order."""
    out = {}
    for st in ORDER:
        d = STAGES[st]
        payload = dict(stage=st, keys={k: get_key(cfg, k) for k in d['keys']}, grid=cfg.get('grid'), code=code_hash(st), inputs=input_fingerprint(st, cfg),
                       up={u: out[u] for u in d['up']}, window=list(window) if (d['windowed'] and window) else None)
        out[st] = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return out

def stage_dir(stage, h):
    return os.path.join(ROOT, 'data', 'derived', 'cache', f'{stage}-{h}')

# ---------------------------------------------------------------- the view of a running stage
_ctx = json.loads(os.environ['STANDS_STAGE_JSON']) if 'STANDS_STAGE_JSON' in os.environ else None
if _ctx is not None:
    CFG = yaml.safe_load(open(_ctx['cfg']))
    OUT = _ctx['out']
    _IN = _ctx['in']
    WINDOW = tuple(_ctx['window']) if _ctx.get('window') else None
else:
    CFG = yaml.safe_load(open(DEFAULT_CONFIG))
    OUT = os.path.join(ROOT, 'data', 'derived')
    _IN = {}
    WINDOW = None
def IN(stage):
    return _IN.get(stage, os.path.join(ROOT, 'data', 'derived'))
def seg(key, default=None):
    return CFG['segmentation'].get(key, default)
