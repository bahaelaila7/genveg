# Stand delineation pilot — Georgia + Florida (2026-10-14)

Goal: stand polygons for GA+FL from TreeMap 2022, soils (SOLUS), satellite embeddings (AlphaEarth 2022 + 2025; Tessera
later), cut by roads, streams/water and parcels (where available), each carrying enough vegetation data to run FVSjl and
Pan directly (stand = simulation unit, instead of TreeMap pixel → FIA plot). Owner choices: MMU adjustable (default
2 ha, typical stands 5–50 ha); AlphaEarth 2022 + 2025 (Tessera next); hard boundaries roads + streams/water + parcels where
available; deliverable = GeoPackage + Forest Explorer stands layer with per-stand FVS + Pan inputs.

## Phases
1. **Data** (GA+FL, all on the TreeMap 30 m grid, EPSG:5070):
   TreeMap 2022 tm_id (+ VAT, tree table); AlphaEarth 2022 & 2025 (64-band int8, read from the 20 m COG overview on
   source.coop through the provided VRTs, resampled to 30 m); Hansen GFC v1.12 lossyear / treecover2000 / datamask;
   LANDFIRE 2022 EVT (developed / agriculture / water classes); SOLUS (clay, sand, AWC, depth to restriction, SOC; 100 m →
   30 m); TIGER 2024 roads + area/linear water per county; FL 5-county parcels.
2. **Forest land incl. clear-cuts**: forest = TreeMap forest pixel, OR (Hansen loss 2001–2024 AND treecover2000 ≥ 25 % AND
   not developed/agriculture/water in EVT 2022), OR (AlphaEarth 2017/2022 forest-like → 2025 change, as a check). Each
   pixel gets a status: `forest`, `harvested` (+ loss year), `regenerating` (loss before 2016 and TreeMap forest now).
3. **Segmentation** (tiled, stitched): features = AlphaEarth 2022 (+2025) PCA, TreeMap stand attributes (forest-type group,
   height, BA, QMD), soils; hard barriers = roads / water / parcel lines / status boundaries; superpixels then
   region-merging to the MMU and a max size; adjustable parameters.
4. **Stand vegetation**: per stand, the TreeMap plots of its pixels with pixel-share weights →
   FVS (FVS_StandInit + FVS_TreeInit: composite tree list, TPA × share; site attributes from the dominant plot) and
   Pan (cohorts: curated FIA cohorts of those plots × share; eco cell, L4, lat). Harvested stands get donors by k-nearest
   neighbours in AlphaEarth 2025 space among young / sapling–seedling TreeMap stands of the same ecoregion.
5. **Outputs**: `stands_gafl.gpkg` (polygons + attributes + donor table), `stand_inputs.sqlite` (FVS tables) and
   `stand_cohorts.parquet` (Pan); Forest Explorer fork: stands layer + simulate-by-stand for FVSjl and Pan.
6. **Tessera**: same pipeline with Tessera embeddings once access is sorted (host unreachable on 2026-10-14).
