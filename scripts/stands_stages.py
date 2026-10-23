"""Stand-delineation stage graph + cache hashing (driver side; docs/STANDS_PIPELINE.md). Not part of any stage's code hash: a stage's own entry (its
configuration keys and their VALUES, its inputs, its module list) enters its hash directly, so adding a tunable key to one stage does not invalidate the
others. The hash of a stage = its configuration keys and values + the window (if windowed) + the code of its script and the modules it imports + a
fingerprint of its raw inputs + the hashes of its upstream stages."""
import os, json, hashlib, copy, yaml
from stands_cfg import ROOT, SCRIPTS, DEFAULT_CONFIG, barrier_files
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
    'embed':    dict(script='20b_embed.py', mods=['common.py', 'features.py', 'stands_cfg.py', 'embed_iwae.py'], keys=['embed'], up=['clean', 'fit'], windowed=False,
                     inputs=AEF + SOIL + TM),
    'sp':       dict(script='21_superpixels.py', mods=['common.py', 'features.py', 'stands_cfg.py', 'embed_iwae.py'], keys=['segmentation.weights', 'segmentation.superpixel_ha',
                     'segmentation.slic_compactness', 'segmentation.barriers', 'segmentation.aef_pcs', 'segmentation.sp_tile', 'segmentation.slic_scale', 'segmentation.slic_range',
                     'segmentation.slic_groups', 'segmentation.treemap_smooth_px', 'segmentation.merge_groups', 'segmentation.sp_seam_halo', 'segmentation.sp_seam_mode'], up=['clean', 'fit', 'embed'], windowed=True,
                     inputs=AEF + SOIL + TM + ['{barriers}']),
    'merge':    dict(script='22_merge.py', mods=['features.py', 'rag_merge.py', 'graph_slic.py', 'stands_cfg.py'], keys=['segmentation.merge_method', 'graph_slic', 'segmentation.merge_threshold', 'segmentation.target_median_ha',
                     'segmentation.max_ha', 'segmentation.mmu_ha', 'segmentation.shape_weight', 'segmentation.status_is_hard', 'merge'], up=['sp'], windowed=False, inputs=[]),
    'levels':   dict(script='25_levels.py', mods=['common.py', 'rag_merge.py', 'stands_cfg.py', 'heldout.py'], keys=['levels', 'segmentation.shape_weight', 'segmentation.barriers', 'segmentation.mmu_ha',
                     'metrics'], up=['finalize', 'sp', 'clean'], windowed=True, inputs=['{barriers}']),
    'finalize': dict(script='23_finalize.py', mods=['common.py', 'features.py', 'stands_cfg.py'], keys=['segmentation.mmu_ha', 'segmentation.status_is_hard',
                     'segmentation.barriers', 'finalize'], up=['merge', 'sp', 'clean', 'land'], windowed=True, inputs=TM + SOIL + ['{barriers}']),
}
ORDER = ['land', 'clean', 'fit', 'embed', 'sp', 'merge', 'finalize', 'levels']
assert set(ORDER) == set(STAGES)

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

