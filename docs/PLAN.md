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

## Tessera access (checked 2026-10-14)
Mirrored on Source Cooperative: `https://data.source.coop/tessera/tessera/npy/v1/` (also `zarr/v1.1`): per (year, 0.1° tile)
int8 [H, W, 128] + float32 per-pixel scale, ~10 m, EPSG:4326, years 2017–2025, range-readable (GeoParquet index
`index/tessera_<year>.parquet`). No overviews ⇒ GA+FL ≈ 2,700 tiles ≈ 390 GB per year to stream; plan: stream tile by tile,
dequantize, aggregate 3×3 → 30 m, PCA-reduce (fit on a sample) to 32 dims int8 ≈ 12 GB per year on disk; ~5–6 h per year
at the measured ~20 MB/s. Phase 6, after the AlphaEarth pipeline works end to end.

## Segmentation design (phase 3)
- Hard barriers (roads, water, parcel lines) are rasterized as 1-pixel separators; connected components of
  (forest land ∧ ¬barrier) are **independent blocks** — each is segmented on its own (parallel, no tile seams); barrier
  pixels on forest land are re-assigned afterwards to an adjacent stand (so the area is not lost, and no stand crosses a
  barrier). Very large blocks (> ~20 M px) are split into overlapping tiles with seam merging.
- Per block: features = PCA of AlphaEarth 2022 (and 2025 change), TreeMap stand attributes (forest-type group, STANDHT,
  BALIVE, QMD), soil PCs, and land status (forest / harvested) with a large weight; Felzenszwalb graph segmentation with
  scale tied to the target stand size, then hierarchical merging of segments < MMU into their most similar neighbour, and
  splitting of segments > max size. All thresholds in a YAML config (MMU default 2 ha, target 5–50 ha, max 200 ha).
