# Stand-delineation pipeline — one place to run, tune and verify

Driver: `scripts/stands_pipeline.py`. Configuration: `config/pilot.yml`. It holds every threshold and weight; `--set key=value` overrides any key
for one run, and unknown keys are refused. Shared code: `scripts/stands_cfg.py` (configuration, stage graph, cache), `features.py`, `rag_merge.py`.

```bash
.venv/bin/python -I scripts/stands_pipeline.py                                    # all stages, config window (Baker County FL pilot)
.venv/bin/python -I scripts/stands_pipeline.py --set segmentation.target_median_ha=25 --name big25
.venv/bin/python -I scripts/stands_pipeline.py --window 12288 14336 16384 18432 sp merge finalize verify metrics
.venv/bin/python -I scripts/stands_tune.py config/tune_baker.yml                   # tuning sweep
.venv/bin/python -I scripts/24_export_fe.py data/derived/baker_county.gpkg <out.geojson> data/derived/runs/<name>/finalize/stands.gpkg
```

## Stages

| stage | script | extent | depends on (config) | outputs (stage dir) |
|---|---|---|---|---|
| `land` | `10_forest_land.py` | full | `forest_land.*` | `land.tif` status classes 0/1 forest/2 harvested/3 regenerating/4 forest_unmapped/9 converted, `loss_year.tif`, `land_summary.json` |
| `clean` | `11_land_clean.py` | full | `segmentation.mmu_ha`, `clean.*` | `land_clean.tif`: status patches < MMU take the class of the nearest larger patch |
| `fit` | `20_fit_features.py` | full | `segmentation.aef_pcs`, `fit.*` | `feature_fit.npz`: AEF-2025 PCA basis, scales, TreeMap / soil standardisation |
| `sp` | `21_superpixels.py` | window tiles | weights, `superpixel_ha`, `slic_compactness`, `barriers`, `sp_tile` | `sp.tif`, `sp_stats.npz`: SLIC per tile × status, never across barriers / statuses / tiles; ids in tile order (deterministic) |
| `merge` | `22_merge.py` | window | `merge_threshold` (auto: calibrated to `target_median_ha`), `max_ha`, `mmu_ha`, `shape_weight`, `status_is_hard` | `sp2stand.npy`, `stand_feat.npz` |
| `finalize` | `23_finalize.py` | window | `mmu_ha`, `status_is_hard`, `finalize.*` | `stands.tif`, `stands.gpkg` (polygons + attributes), `generalisation.json` |
| `verify` | `stands_verify.py` | window | `verify.*` | run dir `verify.txt` (PASS/FAIL; non-zero exit on FAIL) |
| `metrics` | `stands_metrics.py` | window | `metrics.heldout` | run dir `metrics.json` |

**Cache.**
- Every stage writes to `data/derived/cache/<stage>-<hash>`. The hash covers:
  - the stage's config keys;
  - the window (for windowed stages);
  - the code of the stage script and of the local modules **it** imports (`STAGES[...]['mods']` in `stands_cfg.py`);
  - a fingerprint (size + mtime) of its raw input files, including the configured barrier rasters;
  - the upstream hashes.
- Changing a merge parameter therefore re-runs only `merge` and `finalize`; editing `features.py` re-runs from `fit`. Library versions are not hashed.
- A configured barrier raster that does not exist is an error, not a silent skip.
- A stage writes to `<dir>.partial` and is renamed on success, with a `DONE` marker.
- `data/derived/runs/<name>/` links a run's stage dirs and holds `config_resolved.yml`, `summary.json`, `verify.txt` and `metrics.json`.
- The default name is the config name only for the configuration as written, including its window. With `--set` or another `--window` it is
  `<name>_<finalize hash>`.
- Old verify / metrics results are deleted before each run, and a failed metrics stage is an error.
- **Extent.** `sp` covers every tile overlapping the window. finalize, verify and metrics work on that tile-aligned **superpixel extent**, so stands
  are never cut by the window. Stands touching the extent's border are cut by the data extent; they are excluded from stand-level checks and metrics.

## Method (as implemented)

1. **Forest land and status.**
   - TreeMap 2022 forest = forest.
   - Recent Hansen loss (≥ `recent_loss_from`) on former forest (treecover2000 ≥ min) = harvested.
   - Outside TreeMap: LANDFIRE recently disturbed / plantation / ruderal types, older loss, or shrub/herb with high 2000 canopy = regenerating.
   - LANDFIRE tree life form without TreeMap = forest_unmapped.
   - Loss into developed / agriculture / water = converted, which is not forest land. The precedence is documented in the script header.
2. **Speckle removal.** 4-connected status patches below the MMU take the status of the nearest larger patch within `clean.max_dist_px`.
3. **Features.** Each group is scaled to a total variance of weight²:
   - AEF-2025 (16 PCs);
   - AEF-2022 on the same basis (8 PCs, `aef_old`);
   - 1 − cos(AEF-2022, AEF-2025) (`aef_change`);
   - TreeMap 2022 attributes BALIVE, STANDHT, QMD, CANOPYPCT, TPA, softwood share (`treemap`; only on unharvested TreeMap forest);
   - soil PCs (`soil`).
4. **Superpixels.** SLIC of about `superpixel_ha` per tile and status, inside forest land minus barrier pixels (roads, water, parcel lines). They are
   split into 4-connected parts.
5. **Merging.** Region-adjacency graph over superpixels; barriers disconnect it.
   - Each round, mutual-best pairs merge when their Ward cost n_a n_b/(n_a+n_b)·|μ_a−μ_b|² × (1 + shape_weight·(1 − shared / min perimeter)) is below the
     threshold. Pairs must have the same status, and the merged size must stay ≤ `max_ha`.
   - Then regions below the MMU are absorbed into their cheapest neighbour: same status only when `status_is_hard`.
   - Auto threshold: bisection so that the area-weighted median stand of 300 sampled large graph components is `target_median_ha`.
6. **Finalise.**
   - Barrier and sliver pixels are grown into 4-adjacent stands ring by ring: same status first (up to `hole_dist_px` rings), then any status. Every assigned
     pixel therefore touches its stand. Pixels that no stand can reach become stands of their own.
   - MMU generalisation: a fragment below the MMU joins the neighbour with the longest shared boundary. With `status_is_hard` it must have the same status;
     with `mmu_across_barriers` it may sit across a barrier line. It must also be larger, so merges are acyclic; passes repeat until nothing merges.
   - Polygons and attributes are pixel-exact: status = majority, TreeMap means, main TM_ID and its share, soil means, loss year (harvested).

## Verification (`verify`, on the window)

| check | invariant |
|---|---|
| V1 | every forest-land pixel is in a stand; no stand pixel outside forest land |
| V2 | stands are single 4-connected pieces (< 0.5 % of the area in disconnected stands) |
| V3 | status purity when `status_is_hard` (< 0.5 % of pixels outside their stand's majority status, i.e. reassigned barrier pixels) |
| V4 | no superpixel split between stands |
| V5 | after generalisation no interior stand below the MMU has an admissible neighbour: same status if hard; barrier pixels excluded when crossing is off. Fails if generalisation was skipped (extent > 120 Mpx) |
| V6 | stands > `max_ha` (+5 % for barrier pixels) ≤ 0.1 % |
| V7 | gpkg ↔ raster: same ids; area_ha = pixels × 0.09; polygon area = area_ha; status = majority |
| V8 | TreeMap stand means recomputed from pixels for a sample of stands |
| V9 | the auto threshold reaches the target median (± 35 %, after generalisation) |
| V10 | `test_rag_merge.py` unit tests |

## Tuning (`stands_tune.py`, `metrics`)

The objective uses only layers that the segmentation does **not** see (`metrics.heldout`):
- Hansen treecover2000;
- LANDFIRE terrain: elevation, slope, TPI-300;
- LANDFIRE EVC tree cover.

For each layer, R² = 1 − within-stand SS / total SS over the forest-land pixels of interior stands. It is compared with an **equal-count square-block
partition** (blocks × status). The block side is searched until the number of block × status units equals the number of stands. The gain is how much
more of a layer's spatial structure the stands capture than arbitrary units of the same number. The MMU is one pixel count everywhere:
round(mmu_ha / 0.09) = 22 px.

**Independence caveats.** No layer is fully independent:
- TreeMap's imputation used LANDFIRE EVC / EVT and topography as predictors. EVC and terrain gains are therefore inflated by the TreeMap feature group,
  and tuning `weights.treemap` against them would be circular; the shipped spec does not tune it.
- Hansen treecover2000 decides part of the status classes (former forest ≥ 25 %, shrub/herb ≥ 50 %). The block baseline is split by status too, so
  only the within-status variation of treecover2000 is informative.

**Selection.** A sweep ranks variants on the tuning window, so the winner's gain there is optimistic. With `report_window`, the base and the top
`report_top` variants are re-run on a second window (Waycross GA). Those numbers, not the ranking ones, are the estimate to report.

A tuning spec (e.g. `config/tune_baker.yml`) is a grid, random combinations or explicit variants of `--set` overrides, plus size constraints: accepted
area-weighted median and maximum area share below the MMU. Results are ranked by mean gain in `data/derived/tune/<spec>/results.{csv,md}`, with
verify FAIL counts. Variant 0 is the base configuration.

## Fixes made while consolidating (2026-10-21)

- MMU generalisation could cycle: two equal-size fragments remapped onto each other, so the same 88 fragments "merged" every pass and 5 % of stands
  stayed below 0.2 ha. Merges now go only toward a larger neighbour, and chains resolve to their end.
- `status_is_hard` was not read anywhere. The force phase of the merge and the generalisation crossed statuses; both now honour it.
- Superpixel ids followed worker completion order (`imap_unordered`). Ids decide merge tie-breaking, so runs were not reproducible; ids now follow tile order.
- The fragment status was the status of its first pixel; it is now the majority.
- Barrier and sliver pixels were assigned to the NEAREST stand pixel (EDT). That made 1,110 stands disconnected, with 3,919 px in detached pieces on the Baker
  window; region growing from 4-adjacent stands gives 0 (V2).
- Second review (2026-10-22), all confirmed and fixed:
  - Stands cut by the window were generalised and measured as complete; finalize / verify / metrics now use the superpixel extent and interior stands.
  - The "equal-count" blocks were 1.53× more numerous than the stands; the block side is now searched.
  - Three MMU variants were in use (2.0 ha, 22 px, 22.2 px); there is now one pixel count.
  - The feature fit sampled converted (class 9) pixels and could crash on small edge tiles.
  - `--window` without `--name` overwrote the base run; a failed metrics stage left stale results.
  - V5 was vacuous when barrier crossing was off.
  - Shared-module hashing over-invalidated the land / clean stages, and raw inputs were not fingerprinted.
  - Missing barrier rasters were skipped silently.
  - The tuner skipped land / clean / fit for later variants.
  - Forest-free tiles dropped class 9 in land_clean.
  - Calibration edge cases now raise errors. Generalisation skipped on large extents is recorded and fails V5.
  - Ties are stable; loss_year is set only on harvested stands.
- **Status of the Baker pilot** (base configuration, superpixel extent rows 15872–17920 × cols 16384–18432; verify all PASS):
  - 39,639 stands, of which 748 are edge stands; area-weighted median 13.6 ha.
  - 7,631 interior stands < MMU (0.65 % of the area) are isolated, with no same-status neighbour.
  - Held-out R² gain over equal-count blocks: Hansen 2000 canopy +0.110, TPI-300 +0.123, EVC tree cover +0.055, slope +0.030, elevation ≈ 0.
    Mind the independence caveats above.
