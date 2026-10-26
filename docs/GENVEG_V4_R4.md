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
Comparison: v3 vs v4 on the same table and held-out 10-km blocks (g03b direct predictor for every seed; g04 mapping path for model_seed only; null and kNN baselines).
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
* Train v4 on Region 4; evaluate on held-out 10-km blocks across Region 4 (per-state breakdown NOT implemented yet: a TODO, audit r6 M8); downscale; map the Baker stands (the GA+FL stand run
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
| F2/F3 | stand cohort tables = ensemble mean of 16 draws (median 48 cohorts, 13 keys / stand vs FIA 8 / 4); mean-share argmax made 94.5 % of stands PIEL-dominant | one plausible stand: the medoid draw (closest to the calibrated median summary) with its present slots (sampled per slot with the calibrated presence p, seeded by the stand id), scaled to the calibrated TPH / BA; dominant key from that table |
| F4 | the production map was USE_B = 1 (stand likelihood), unvalidated (no localisation skill; worse TPH) | use_b [1, 0]: B0 (validated prior path) is the production map; B1 kept as an alternative output |
| F10 | GEN_QMD = median of the samples' QMDs, inconsistent with the calibrated TPH / BA | QMD of the calibrated TPH / BA |
| F11 | stage hashes ignored outputs: a forced / re-run stage left downstream stages current | each run records an output fingerprint; downstream hashes include it |
| F14 | g05 docstring ("same uniform subsample as the plots") | corrected |
Documented, not changed: F9 — the mapping path is calibrated at the visits' dates (mean 2023.9) while g05 maps the end-2025 state; F12 — held-out candidate priors
use the test plot's own FIA records (stands have none), so localisation-dependent held-out scores are optimistic for stand mapping; F13 — g02g never returns an
empty consistent set (the "empty set keeps its prior" branch of g03b is unused). FIA also SWAPS up to 20 % of private plots' coordinates with a similar plot of the
same county: for those plots no footprint within 1.6 km is the plot's — not modelled. The satellite likelihood B shows no localisation skill (audit): the mapped
stands' composition comes from the landscape prior; whether v4's non-linear heads change that is part of the v3 vs v4 comparison.

## F. Audit round 3 (2026-10-26, fresh independent Opus review of b60cd87) and the user's decisions
Mechanical fixes (7fd09c9): gv_verify T3h on the prior's evidence part (B3); the live v3 config's production map is B0 (B4); a failed / forced run is not left
current (I4); the Q-decode window centred on z(2025) = the end-2025 state; per-stand seeding of g05's draws; g04 in the dynamics stage hash; dead code / docstrings.
User decisions (2026-10-26): (1) the curated tree table misses about half the live saplings in AL / GA / NC / SC / TN (a RECONCILECD-based tree exclusion) —
**fixed upstream** in FIA_DATA_PREP (branch curation-fix), every model to re-derive its inputs; (2) **fix the dynamics, then map from 2000**; (3) retrain v3
on the rebuilt table too (config/genveg_v3b.yml, data gafl_v3b); (4) runs halted until the pipelines are vetted.
Dynamics / mapping changes (switches; the new configs enable them, defaults keep the older models loadable):
* `SITE_ATTR` — the flow's attractor depends on the site: A + MLP(soil), zero-initialised (one global attractor made every undisturbed stand revert to it).
* `PAIR_ALL` (`PAIR_HMAX` 25 yr) — every ordered visit pair of a plot trains the flow (multi-horizon; a training visit samples one of its pairs); an
  "undisturbed" span requires no hiding FIA code at ANY visit in between.
* `PAIR_SYN` (REMOVED in round 4, I1) — an FIA-recorded event cluster that no candidate's LANDFIRE / Hansen evidence matches is injected into the pair's histories at its FIA year
  (its classes, severity 2): recorded but undetected disturbances no longer feed the no-event drift.
* `MAP_Y0` 2000 — the mapping path (g04 calibration and g05) draws z from the landscape prior at 2000 and flows through the footprint's whole 2000–2025 history
  (from 2021, a stand cut in 2016–20 was mapped as mature).
* g06 reports undisturbed held-out pairs by horizon (< 8, 8–15, > 15 yr): predicted vs observed BA log change per year.
* Fixed while smoke-testing: the F6 pair-loss tensor was named `ep`, shadowing the epoch counter. The `for` loop rebinds `ep` every epoch, so the curriculum
  switch was NOT affected; only the epoch log printed a tensor (round 4 corrected this note).
* `SMOKE_BATCHES` — a code-path check of the training step (no checkpoint, no evaluation).

## G. Audit round 4 (2026-10-26, fresh independent Opus review of 1ff0178); findings reproduced before fixing
* **B1** `MAP_Y0` 2000 made the held-out mapping path WORSE for v3 (BA error 0.449 vs 0.405 at 2021; a shuffled-landscape map correlated 0.87 with the real
  one: the 25-yr flow forgets the landscape prior). → g04 now **gates** the requested start year with a pre-registered rule: it is used only if its held-out
  TPH and BA errors are ≤ those of the 2021 path and ≤ the null's, and the shuffled-landscape BA correlation is < 0.5; else 2021. `bayes_eval.json` records
  `map_y0` + `map_y0_gate`, and g05 maps with `map_y0` (not the env). gv_verify T11c.
* **I1** `PAIR_SYN` injected impossible histories (clear-cuts of severity 2 on stands that kept their BA) → removed; a recorded-undetected event now only
  excludes the pair from the no-event rules (see I3).
* **I2** the severity of a multi-horizon pair used only the end-points' BA ratio → the MINIMUM BA ratio over the span's consecutive visits.
* **I3** undisturbed long pairs could be credited to spurious event histories → with no hiding FIA code in the span and no detectable record, the candidate
  set is narrowed to event-free histories (where any exist), as for the one-step pairs. gv_verify T8 / **T8c** (multi-horizon) recompute this independently.
* **I4** the contracting flow cannot age a stand (decoded mean age +0.1 yr / yr on undisturbed pairs vs +0.87 observed) → `AGE_CH`: latent dimension 0 is
  an **age clock** (BA-weighted mean cohort age, (A − 40) / 40), supervised at every visit (`W_AGE`), advanced by exactly dt / 40 per year by the flow
  (jumps may reset it), excluded from SIGReg. g06 adds the decoded-age progression and a **pre-registered acceptance**: undisturbed BA change per year
  within ±0.01 of observed in every horizon bin, decoded / observed age gain 0.7–1.3. gv_verify T4k (site attractor + clock), T4l (age target).
* **I5** `SMOKE_BATCHES` leaked into evaluation runs → honoured only when EPOCHS > 0 and LOAD = 0; added to the stage switches.
* **I6** gv_verify did not cover the new logic → T4k, T4l, T8c, T11c. Running it also exposed a round-2 regression in gv_verify itself (the live-tree merge
  dropped the `fsw` / `fmw` shares): the verifier had not run since 63e8837.
* Minor: g06 blocked-visit fallback = ones; `_n_consec` counters; Am(soil) computed once per run (soil is constant); stale gafl_v4 outputs of the smoke
  run (m_v4_s0.pt, eval.json, stand_veg.parquet) deleted so the pipeline cannot adopt them.

## H. Audit round 5 (2026-10-26, fresh independent Opus review of 60d03fd); findings reproduced before fixing
* **B1** g06's long-horizon section crashed (NameError): the `vs_of = …` assignment sat after a `#` on the previous line. Fixed; g06 ran end to end on an
  untrained scratch checkpoint (deleted afterwards).
* **I1 / M3** the age clock could not be reset by a disturbance (a bounded jump moves it < 1 = 40 yr; 22.6 % of stand-replacing pairs need more) and its
  rate was fixed at 1 yr / yr (observed BA-weighted mean age +0.87–0.91 yr / yr). → `Flow.jump`: with `AGE_CH` the jump network has one more output, a
  reset fraction r: z0 ← z0 + r (−AGE_C / AGE_S − z0) + 0.25 tanh(·) (r starts at sigmoid(−2)); the clock advances by dt · exp(lrho) / AGE_S (learned
  rate). gv_verify T4k checks the rate, the reset algebra and a full reset of a 100-yr stand.
* **I2** the r4 "event-free" rule removed FIA-recorded ground fires (and insects): their likelihood ratio is 2 (< 4, not a cluster constraint) and their codes
  do not block. → the event-free test ignores the evidence of the period's recorded weak events inside their windows (g03b.pair_set; gv_verify T8 spec).
* **I3** FIA records were assigned to pairs by a mid-year rule on the event year, not by the visit that recorded them (3.3 % of records made at v2 fell out
  of (v1, v2]; 218 next pairs re-applied a cut after v2). → g02e writes the RECORDING visit (`fia_events.rmd`); `g03b.recs_in` gives a pair the records
  made at visits in (v1, v2] (re-reports of an event recorded at a visit ≤ v1 dropped); the flow takes `rs` / `re` (the start / end visit recorded an
  event of its own calendar year: that year's evidence lies before the visit) and every narrowing window is clipped to the years the flow actually applies
  (`act_span`, which also fixes **M2**, boundary-year jumps leaking into "event-free" histories). Used by g03b (MIL, pairs, Q decode mask, evaluation),
  g04 (mapping path end visit), g06. gv_verify `pair_spec` re-derives all of it from FIADB COND (T8, T8b, T8c, T8d, new T8e) and T4m tests the flow rule.
* **I4** the stage hashes did not cover the FIA databases or the GA+FL rasters: a rebuilt curated table would have passed as current. → config `inputs`
  (curated_db, fiadb; relative to WORKSPACE) passed to every stage as GV_CURATED_DB / GV_FIADB (`common.curated_db()` / `fiadb()`), and fingerprinted
  (path, size, mtime) with the GA+FL inputs into table / records / history / stands_fp / verify. The v4 / r4 / v3b configs point at
  `landis_region4/data_r4/slim_r4_v2.duckdb` (the curation-fix rebuild): the driver refuses to run until it exists.
* **I5** the g06 acceptance was not enforced → g06 exits 3 when it fails (history location), and downscale (hence map / fe) depends on the dynamics stage.
* **M1** the MAP_Y0 gate selected on the held-out plots it reports → decided on the calibration plots, 2-fold by plot (factors from one half, errors on
  the other); held-out errors of both paths are reported only.
* **M4** g02b docstring: the stand age at the first visit is not used as negative evidence (corrected). **M6** the epoch log divides by the visits seen.
* M5 (verifier scope) addressed by the independent `pair_spec`; T8 still runs the period rule (T8b covers PAIR=both + severity).

## I. Audit round 6 (2026-10-26, fresh independent Opus review of e5992ef); findings reproduced before fixing
* **I1** FIA DSTRBCD 11 / 12 and 21 / 22 were swapped in the "cannot hide a stand-replacing event" set (g02b.NONBLOCK, gv_verify HARMLESS): per the
  FIADB library, 11 / 21 = insect / disease damage to UNDERSTORY vegetation, 12 / 22 = damage to TREES (e.g. southern pine beetle). → {11, 21, 31, 40–46,
  60}. gafl_v4: visits with a hiding code 9,521 → 10,636; negative-evidence periods 25,009 → 23,891 (the location prior / history sets change at the next
  table rebuild).
* **I2** the g06 acceptance was decided on the held-out TEST plots (and every seed had to pass) → `VAL_SPLIT` (train env; the v4 / r4 / v3b configs set
  it): 10 % of the non-test blocks (minus the test buffer) are a validation set with its own 3.2 km buffer, excluded from training; g06 decides the
  acceptance on the validation plots (test reported as `test_undisturbed_*`) and enforces it only for `model_seed` (pipeline sets G06_ENFORCE); without a
  validation set the acceptance cannot pass (exit 4). gv_verify T4d checks the three sets are disjoint and ≥ 3.2 km apart.
* **M1** a pair spanning an FIA measurement missing from the table (non-forest / non-sampled visit) was read as undisturbed → g02e writes
  `visits.fia_gap` (FIA's PREV_PLT_CN measurement is not the previous table visit; 125 visits), g03b / g06 treat it as blocking; the negative-evidence
  period of a visit starts at the later of PREV_PLT_CN's measurement and the previous forested visit (g02b.fia_records). gv_verify re-derives both.
* **M2** the map stage wrote the unvalidated B1 map under the production name first → g05 writes `_B<b>` files only; the pipeline promotes the last
  `use_b` (B0) to stand_veg_v3.parquet / stand_cohorts_v3.parquet after all runs.
* **M3** stage currency by hash only → the state records each run's output files (size, mtime) and a stage is current only if they are unchanged.
* **M4** inputs no stage hashed → train depends on stands_fp; baker plots, dist_codes.csv (both regions), the Baker county and stand-run polygons are
  fingerprinted.
* **M5** verifier: T8f compares g03b's own act_span / recs_in / pair_clusters with the spec; T4m also runs with the production switches; T4l on 300 visits;
  T4d tests the buffers.
* **M6** g04's B flow ignored the visit's recorded own-year event (rs) → passed. **M7** MDN-fit plots within 3.2 km of a calibration plot are not fitted;
  the gate's null baseline excludes the plots it judges. **M8** comments / doc corrected (per-state R4 evaluation is a TODO; the mapping path runs
  model_seed only).
* Verification of r6 on gafl_v4 (table built by the PRE-r6 g02b / g02g): gv_verify 50 / 52 — the two failures are the expected staleness of the table:
  T3f (location prior: max ratio spread exactly 4 = the ×0.25 negative-evidence factor of the corrected DSTRBCD set / periods) and T10 (history sets,
  2 of 60 plots). Both are rebuilt from the fixed curation (slim_r4_v2) before any training. Smoke: training step, untrained-checkpoint g06 (validation
  acceptance enforced: exit 3), g04 and g05 end to end (scratch outputs deleted). While smoke-testing, an r6 edit had put `n_ = …` inside a comment in
  g04.fwd_dist (the B1 bug class again) — fixed; a scan of every stage for code after `;` in comments finds none.

## J. Audit round 7 (2026-10-26, fresh independent Opus review of 64db7c8); findings reproduced before fixing
* **B1** the r6 file-level currency could never finish (stages that intentionally rewrite an upstream file — g02e visits.parquet, g02g localisation.parquet
  — invalidated it; a 1-s time slack attributed the previous stage's outputs to the next; the driver's model.pt copy was attributed to stages). → each run's
  outputs are a SNAPSHOT DIFF (before / after); a file rewritten by a later stage is checked against that stage's version (`_writers`). Tested on the
  auditor's stub root: a full run, then a second call all current; an edited fia_events.parquet re-runs records and its dependants only, then all current.
* **B2** g06 used floor(visit year) as a flow stop without clamping: a 2026 visit (6 KY plots in R4) raised KeyError → clamped to YEARS[-1] (as g03b);
  g04's mapping path too.
* **I1** LANDFIRE severity labels of the 2013-14 / 2024-25 products ('Moderate' & co., 'No Severity', 'Unburned/Low') fell back to Low (36-39 % of the
  2024-25 disturbed pixels) → mapped (Moderate = Medium 2; mixed = 1.5 / 2.5; no rating = neutral 2; NaN = no rating); water / no-data pixels inside fire
  perimeters are no disturbance; a classed code with an unknown label now raises. gv_verify T3e checks the severity channel (an independent label table);
  the new code matches it on 400 freshly computed footprints (14 / 400 differ from the stale table → rebuilt with the table).
* **I2** a new record made at v2 dated in v1's calendar year (the event is after v1) was never applied for a second-half v1 / a v1 that recorded another
  own-year event → Flow `ra` (and `act_span(..., ra)`, `pair_ra`, `VRA` for MIL / Q / B flows from a visit); gv_verify pair_spec and T4m cover it.
* **M1** forest -> non-forest / non-sampled conversion (forested share drop >= 0.25 since the previous forested visit) carries no FIA code → blocks like a
  hiding code (g02b.fia_records; gv_verify spec CONV). gafl_v4: blocked visits 10,636 → 11,041.
* **M2** bare visits were never checked against FIADB (22 % of gafl_v4's bare visits have live FIADB trees, from the old curation) → gv_verify T1k (FAILS on
  the stale table by design; must pass after the slim_r4_v2 rebuild).
* **M3** vacuous checks: T6c removed (T11 tests the conversions on the written layer); T8d requires > 0 regrown spans; T11c reads the start year g05
  recorded in the production map (`map_y0` column) instead of g05's source text.
* Verification of r7 on gafl_v4 (table predates r6/r7 g02b): gv_verify 48 / 52; the 4 failures are the stale table only — T1k (old curation: 305 / 1,363
  bare visits with FIADB live trees), T3e (stored severity of the old label table: max Δ 1/3 = one severity step), T3f / T10 (old NONBLOCK / periods).
  New code vs the independent severity spec on 400 freshly computed footprints: 0 mismatches. Smoke: train step, g06 (val acceptance enforced, exit 3),
  g04, g05 (stand map records map_y0 = bayes_eval map_y0) end to end; scratch outputs deleted. Required after the slim_r4_v2 rebuild: 52 / 52.
