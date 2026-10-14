# Status / resume checkpoint (2026-10-14 13:35 UTC, before container restart)

## Data (all on the TreeMap 30 m EPSG:5070 grid, data/grid/grid.json; validated by scripts/99_validate.py -> data/validate1.txt)
DONE: grid + AOI mask, TreeMap (tm_id.tif byte-identical to source crop; vat.parquet, trees.parquet — PLT_CN stored as
'1234.0' strings: strip the '.0'), Hansen GFC-2024 (lossyear, treecover2000, datamask), LANDFIRE EVT 2022 (+ evt_classes.csv),
soils (awc030, clay030, sand030, soc030, ph030, restrdepth, lithdepth), TIGER roads/areawater/linearwater -> barriers/{roads,water}.tif,
parcels (FL 5 counties) -> barriers/parcels.tif.
AlphaEarth: tiles are resumable (per-tile files, atomic .part->rename; done tiles skipped; local sparse COG cache data/aef/cache 39 GB).
- 2022: all 1026 tiles done; the mosaic data/aef/aef_2022.tif was being assembled when stopped -> deleted; RE-RUN
  `.venv/bin/python scripts/02b_aef_assemble.py 2022`.
- 2025: 824/1026 tile files done -> RE-RUN `.venv/bin/python scripts/02_aef.py 2025` (skips done tiles), then
  `scripts/02b_aef_assemble.py 2025`, then `scripts/02c_aef_sanity.py`.

## Pipeline (written, not yet run on real data)
10_forest_land.py -> 11_land_clean.py -> 20_fit_features.py -> 21_superpixels.py [r0 c0 r1 c1 pilot window] -> 22_merge.py
(rag_merge.py unit-tested: scripts/test_rag_merge.py). Config: config/pilot.yml.
Still to write: 23_finalize (stands.tif via sp2stand, barrier-pixel reassignment, polygonize -> stands_gafl.gpkg),
30_stand_veg (TreeMap plot shares per stand -> FVS_StandInit/TreeInit composite in stand_inputs.sqlite + Pan
stand_cohorts.parquet; donors by kNN in AEF-2025 space for harvested-after-2022 / no-TreeMap stands), FE integration, Tessera.
Next: run 10 + 11 now (inputs ready) and check land_summary.json; finish AEF; pilot-window segmentation and tuning.

## Other processes stopped
Forest Explorer server (fe_pan, port 8088) and Pan worker — stateless; restart with apps/forest-explorer/serve_pan.sh.
PnET relaxed-alternatives watcher (landis_scen runs_local/pnet_alts3.log) — informational only.
