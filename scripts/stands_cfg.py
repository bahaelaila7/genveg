"""Stand-delineation pipeline: configuration, stage graph and stage directories (docs/STANDS_PIPELINE.md). Shared by the driver
(scripts/stands_pipeline.py) and every stage script.

A stage script reads its resolved configuration and its input / output directories from the env STANDS_STAGE_JSON written by the driver:
  CFG          the resolved configuration (config file + --set overrides)
  OUT          this stage's output directory (a content-hashed cache dir: data/derived/cache/<stage>-<hash>)
  IN(stage)    the output directory of an upstream stage
  WINDOW       (r0, c0, r1, c1) processing window in pixels, or None = full extent
Run without the driver (no STANDS_STAGE_JSON), a stage falls back to config/pilot.yml and data/derived for every directory (the legacy layout).

The stage graph, configuration keys per stage and the cache hashing live in scripts/stands_stages.py (driver side).
"""
import os, sys, json, hashlib, copy, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, 'scripts')
DEFAULT_CONFIG = os.path.join(ROOT, 'config', 'pilot.yml')
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
