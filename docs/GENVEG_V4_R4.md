# Generative vegetation v4: deeper networks (GA+FL, 30 m) and Region 4 (90 m)

User request 2026-10-25: the v2/v3 networks are too shallow (encoder, disturbance jump, decoder) and the satellite heads should not be linear;
apply that to the current GA+FL data (30 m), and expand the model to EPA Region 4 (AL FL GA KY MS NC SC TN) at 90 m, because AlphaEarth at
30 m for all of Region 4 does not fit on disk. Branch `genveg-v4` (from `stands-pipeline` a49e73a). Runs in parallel with the nested-CV campaign.

## A. Architecture v4 (`ARCH=v4` in `g03b_train_gafl.py`; the downstream stages load the model through g03b)
| part | v2/v3 | v4 |
|---|---|---|
| encoder E_v | DeepSets: token MLP 17→64→64, mean+max pool, MLP 129→64→16 | Set Transformer: token embedding → 4 pre-LN self-attention blocks (d 128, 4 heads, FF 256) → attention pooling (PMA, 1 seed) + year → MLP → z |
| latent z | 16 | 32 (`DZ`) |
| flow (LTC cell) | gate MLP (z, soils)→64→z | unchanged form; gate MLP 2 hidden layers (128) |
| disturbance jump J | MLP (z, class one-hot, severity, Hansen)→64→z | class embedding + 4 residual blocks (d 128, pre-LN) → z, bounded (tanh) |
| decoder | MLP z→128→slots×5 | 4 residual blocks (d 256, pre-LN) → slots×5 |
| satellite heads | linear z→64 | linear skip + 2-layer MLP (128) |
| inverse encoder Q | MLP 256/128 | unchanged |
Rationale for keeping a linear skip in the heads: the MIL localisation compares predicted with candidate AlphaEarth embeddings; the linear path keeps
the latent's geometry tied to satellite space, the MLP removes the 16/32→64 linear bottleneck.
Comparison: v3 vs v4 on the same table and held-out 10-km blocks (g03b direct predictor; g04 mapping path; null and kNN baselines), 2 seeds each.
GA+FL run: data dir `gafl_v4` (the `gafl_pipe` table, stand footprints and stage outputs adopted, model retrained); FE outputs stay in data/proto
until the comparison is reviewed (the live Forest Explorer layer is not replaced).

## B. Region 4 at 90 m
* Grids: `data/grid/grid_r4.json` (30 m, TreeMap-aligned, Region 4 AOI = the 8 states' counties dissolved, buffered 2 km) and `grid_r4_90.json`
  (90 m, origin on the 30 m lattice; a 90 m pixel = 3×3 30 m pixels).
* AlphaEarth 2022 / 2025 at 90 m (`r4_aef90.py`): per 256×256 output window (90 m), the 20 m overviews of the source COGs that cover it, copied locally
  as sparse spans (`/vsicurl` windowed reads were ~200 s per window; local copies are fetched ahead within `AEF90_CACHE_GB`, default 25, and deleted when
  no pending window needs them), reprojected per band to 30 m (average) in the de-quantised domain, renormalised and int8-quantised exactly as
  `02_aef.py`, de-quantised again and averaged over each 3×3 block (nodata excluded) = the footprint mean of `g02b.feats`, then re-quantised with the same
  scheme for storage (sign·(v/127.5)²). Resumable per window (`data/aef90/tiles_<y>/`), then `assemble` → `data/aef90/aef90_<y>.tif`.
  Equivalence vs the GA+FL 30 m mosaic (the 30 m stage): 94 % of int8 values identical, mean |Δ| 0.0011 (≈1 % of magnitude), max 0.011 (GDAL warper
  approximation; scattered, no spatial pattern). The 90 m re-quantisation adds ≤ half a quantisation step (verified per run: gv_verify T3c2).
* 30 m rasters on `grid_r4` (`r4_rasters.py`): forest land (TreeMap 2022 CONUS + EVT 2022 CONUS + Hansen, the rule of `10_forest_land.py` with
  `config/pilot.yml`; NOT speckle-cleaned — the table reads only footprint centres), LANDFIRE annual disturbance (CONUS archive, the codes of
  `dist_codes.csv`), Hansen lossyear / treecover2000 / datamask (+ granules 30N_100W, 40N_100W). Soils (SOLUS 100 m, bilinear) on `grid_r4_90`;
  `soilfill` writes the Region 4 forest-land medians (`data/r4/soils90/fill.json`) that replace `SOIL_FILL` (GA+FL medians) for SOLUS nodata.
* Footprints: plot windows start on a multiple of 3 in the 30 m grid, so footprint centres are 90 m block centres (row, col ≡ 1 mod 3); GA+FL uses a
  per-plot-window stride-3 lattice (statistically equivalent, not bit-identical). Distances are from the plot pixel to the block centre (≤ 1600 m).
* Switches (env; `config/genveg_r4.yml` top-level `env:` is passed to every stage and enters every stage hash only when present, so existing configs
  keep their hashes): `REGION` (FIA STATECDs; default 12,13), `AEF90=1` (the Region 4 rasters above), `CF16=1` (g03b keeps the candidate features
  float16 in memory and casts to float32 on indexing — ~4 M candidates would need ~12 GB as float32). g02b (6 SQL queries + rasters + aligned windows),
  g02j / g05 (stand pixels shifted by `g02b.stand_offset`, a multiple of 3 px from the GA+FL stand raster to `grid_r4`; windows on 90 m blocks),
  g02e / g02g (via g02b's FIADB queries), gv_verify (REGION / AEF90: state-polygon test T2c, 90 m reads in T3c/T3d/T3g/T5/T6b, T3c0 lattice, T3c2 90 m
  route vs the GA+FL 30 m mosaic, T13 skipped without g02c windows), gv_verify_upstream (U1r grids, U6r forest-land rules re-derived with CONUS
  TreeMap / EVT read by coordinates, U8 with 90 m footprints). GA+FL defaults: byte-identical (old vs new g02b `plot_job` on 40 plots + FIADB queries).
* Table: `NCAND` 128 (GA+FL 256; ~3.5× the plots: ~30k plots, ~115k visits in curated_trees since 2000), `TOPK` 24 (GA+FL 16; Region 4 is more diverse).
* Train v4 on Region 4; evaluate on held-out 10-km blocks across Region 4 (per state too); downscale; map the Baker stands (the GA+FL stand run
  `gafl_pilot`, features from the Region 4 rasters), outputs in `data/proto/r4_v4` (not the live FE layer).

## C. Run (resumable; every step skips finished work)
```
.venv/bin/python -I scripts/r4_grid.py
for y in 2022 2025; do .venv/bin/python -I scripts/r4_aef90.py $y 4 && .venv/bin/python -I scripts/r4_aef90.py $y assemble; done   # resumable per window
for s in hansen_dl hansen dist land soils90 soilfill; do .venv/bin/python -I scripts/r4_rasters.py $s 2; done
.venv/bin/python -I scripts/genveg_pipeline.py --config config/genveg_r4.yml          # table … verify (stage hashes; reruns only stale stages)
.venv/bin/python -I scripts/genveg_pipeline.py --config config/genveg_v4.yml          # GA+FL v4 (inputs adopted from gafl_pipe)
```
Code changes to g02b / g02j / g05 / gv_verify change the stage code hashes of the GA+FL pipelines too (by design); their outputs are byte-identical
for the defaults, so an existing GA+FL data dir can re-adopt its stages (`--adopt`) instead of rebuilding.

## D. Audit round 1 (2026-10-26, independent Opus review; findings reproduced before fixing)
| id | finding | fix |
|---|---|---|
| M1 | g05 stand cohort tables (the TreeMap-independent initial state) were the MEAN of 16 samples × CAL, the stand summaries the calibrated MEDIAN: cohort sums 1.41 × TPH, 1.14 × BA (median over 16,214 Baker stands; reproduced) | per stand one density and one diameter factor so the cohorts sum to the calibrated TPH / BA (composition and ages kept); gv_verify T11b |
| M2 | Q-decode loss trained the DECODER against visit-time cohorts for Q's z(2025) (visits on average ~4 yr before 2025) | decoder frozen in that term (only Q learns), visits within QD_WIN = 1.5 yr of 2025 only |
| M3 | bare visits (no live tree) had a QMD target of 0 | QMD term masked for bare visits (as composition) |
| M4 | gv_verify T4e–T4j built the v3 Flow whatever ARCH; forced NT 4 | the configured flow class; NT from the environment |
| P1 | the shell environment leaked unhashed into every stage (an exported AEF90 / REGION / ARCH … would change a run under the same hash) | stage switches stripped from the inherited environment; meta.json records region / aef90; g02e / g02g / g03b / gv_verify assert them (`scripts/region.py`) |
| P2 | the AEF90 rasters were not part of any stage hash | size + mtime fingerprint of the Region 4 inputs and the r4_*.py code in the table / stands_fp hashes |
| R1 | r4_v4 stand footprints from superseded code (1,827 stands without a footprint) | stands_fp re-run with c8e6a9f: 16,214 / 16,214 stands |
| ops | gv_verify_upstream U6r took 4.8 s per boundless 1-px read of the strip TreeMap CONUS | plain reads in row order (minutes) |
Verified correct by the audit (independent recomputation): the LTC flow (float64 re-implementation, ≤ 8e-16), sparse vs dense jump, EncV4 masks /
permutation invariance / bare visits, held-out blocks (3.2 km buffer), the Region 4 rasters vs GA+FL on the overlap, same-plot same-footprint
features, a direct recomputation of 40 plots outside GA+FL, GA+FL default byte-identity. Not changed (documented): an event's jump applies at the end
of its year (the event year's AlphaEarth state is post-event); 6 Kentucky pairs after 2025 are cut at the end of 2025.
The models trained before these fixes (gafl_pipe v3, the first v4 runs) were trained with M2 / M3; v4 GA+FL and Region 4 are retrained.

## E. Audit round 2 (2026-10-26, fresh independent Opus review of 79c73b7; findings reproduced before fixing)
| id | finding | fix |
|---|---|---|
| F1 | Kentucky: 1,199 plot keys reuse the number of a 1986-88 periodic plot km away; g02b averaged LAT/LON over all PLOT rows (547 R4 plots > 1.6 km off) | coordinates from the MEASYEAR ≥ 2000 rows; gv_verify T2a0 (a key's rows agree) |
| F8 | densities divided by the forested share of the SAMPLED area; TPA_UNADJ expands to the whole plot → low by (1 − nonsampled share) (0.5 % / 0.9 % of visits) | per forested area = Σ TPA / f1s (the sampled share stays the visit filter); gv_verify recomputation likewise |
| F7 | uniform location prior over the 1.6 km disk (75 % of the mass beyond 0.8 km); FIA: perturbed within 1 mile, only a small share beyond 0.5 mile | radial prior RADIAL_P = 0.9 within 0.5 mile (uniform per zone), recorded in meta.json; T3f re-implements it |
| F6 | remeasurement-pair loss averaged over K candidate histories (22 % of pairs mix stand-replacing and other histories) | mixture (soft-min, T_PAIR = 0.1) as the MIL; flowed-state decoding weighted by the detached posterior |
| F5 | every seed wrote eval.json / stand_veg.parquet; the direct predictor's geometric-mean estimator was biased low | per-model files (eval_<model>.json, stand_veg_q_<model>.parquet); prior-weighted median |
| F2/F3 | stand cohort tables = ensemble mean of 16 draws (median 48 cohorts, 13 keys / stand vs FIA 8 / 4); mean-share argmax made 94.5 % of stands PIEL-dominant | one plausible stand: the medoid draw (closest to the calibrated median summary) with its present slots (p ≥ 0.5), scaled to the calibrated TPH / BA; dominant key from that table |
| F4 | the production map was USE_B = 1 (stand likelihood), unvalidated (no localisation skill; worse TPH) | use_b [1, 0]: B0 (validated prior path) is the production map; B1 kept as an alternative output |
| F10 | GEN_QMD = median of the samples' QMDs, inconsistent with the calibrated TPH / BA | QMD of the calibrated TPH / BA |
| F11 | stage hashes ignored outputs: a forced / re-run stage left downstream stages current | each run records an output fingerprint; downstream hashes include it |
| F14 | g05 docstring ("same uniform subsample as the plots") | corrected |
Documented, not changed: F9 — the mapping path is calibrated at the visits' dates (mean 2023.9) while g05 maps the end-2025 state; F12 — held-out candidate priors
use the test plot's own FIA records (stands have none), so localisation-dependent held-out scores are optimistic for stand mapping; F13 — g02g never returns an
empty consistent set (the "empty set keeps its prior" branch of g03b is unused). FIA also SWAPS up to 20 % of private plots' coordinates with a similar plot of the
same county: for those plots no footprint within 1.6 km is the plot's — not modelled. The satellite likelihood B shows no localisation skill (audit): the mapped
stands' composition comes from the landscape prior; whether v4's non-linear heads change that is part of the v3 vs v4 comparison.
