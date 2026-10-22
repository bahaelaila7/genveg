import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def stands_run():
    """The stand-delineation run (scripts/stands_pipeline.py) whose forest mask and stands the generative-vegetation stages use: env STANDS_RUN
    (a data/derived/runs/<name> dir; config/genveg.yml stands_run via scripts/genveg_pipeline.py), default the base run gafl_pilot. No silent fallback."""
    d = os.environ.get('STANDS_RUN', f'{ROOT}/data/derived/runs/gafl_pilot')
    f = dict(dir=d, tif=f'{d}/finalize/stands.tif', gpkg=f'{d}/finalize/stands.gpkg', land_clean=f'{d}/clean/land_clean.tif', land=f'{d}/land/land.tif')
    for k in ('tif', 'gpkg', 'land_clean', 'land'):
        if not os.path.exists(f[k]): raise FileNotFoundError(f'stand run {d}: {f[k]} missing (run scripts/stands_pipeline.py)')
    return f
