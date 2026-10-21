"""Attach the generative-vegetation stand map (g05: data/proto/gafl/stand_veg_v3.parquet) to a Forest Explorer stands GeoJSON (replaces the ad-hoc
attachment of 2026-10-20). Properties: GEN_BA (ft²/ac), GEN_TPA (trees/ac), GEN_QMD (in), GEN_BA_SD (sample spread, ft²/ac), GEN_DOM (dominant species
key by BA), GEN_SRC (version tag). Units: m²/ha × 4.356 → ft²/ac; /ha ÷ 2.471 → /ac; cm ÷ 2.54 → in. The input file is backed up next to itself once.
Usage: python scripts/g07_fe_attach.py <stands.geojson> [tag]"""
import os, sys, json, shutil, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import ROOT
fn = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'v3'
D = f'{ROOT}/data/proto/gafl'; S = pd.read_parquet(f'{D}/stand_veg_v3.parquet').set_index('stand_id')
if not os.path.exists(fn + '.bak_v2'): shutil.copy2(fn, fn + '.bak_v2')
g = json.load(open(fn)); n = 0
for f in g['features']:
    p = f['properties']; sid = int(p['stand_id'])
    for k in ('GEN_BA', 'GEN_TPA', 'GEN_QMD', 'GEN_BA_SD', 'GEN_DOM', 'GEN_SRC'): p.pop(k, None)
    if sid in S.index:
        r = S.loc[sid]; n += 1
        p.update(GEN_BA=round(float(r.ba) * 4.356, 1), GEN_TPA=round(float(r.tph) / 2.471, 1), GEN_QMD=round(float(r.qmd) / 2.54, 2),
                 GEN_BA_SD=round(float(r.ba_sd) * 4.356, 1), GEN_DOM=str(r.dom_key), GEN_SRC=tag)
json.dump(g, open(fn + '.tmp', 'w'), separators=(',', ':')); os.replace(fn + '.tmp', fn)
print(f'{fn}: {n} of {len(g["features"])} stands carry generative attributes ({tag})')
