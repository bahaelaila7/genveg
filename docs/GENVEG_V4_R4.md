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
Comparison: v3 vs v4 on the same table and held-out 10-km blocks (g03b direct predictor for every seed; g04 mapping path for model_seed only; null and kNN baselines) — DECIDED on the validation metrics (g03b `validation`, g04 `mapping_path_validation_crossfit`, g06 acceptance); the test block is reported only (r9).
GA+FL run: data dir `gafl_v4` (the table and stand footprints were RUN by the driver — 1,687 s — not adopted; the model is trained there); FE outputs stay in data/proto
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
.venv/bin/python -I scripts/genveg_pipeline.py --config config/genveg_v4.yml          # GA+FL v4 (table + footprints built by the driver)
```
Code changes to g02b / g02j / g05 / gv_verify change the stage code hashes of the GA+FL pipelines too (by design), so those stages re-run there as well.
Do not `--adopt` them across a code change: `--adopt` is only for outputs known to come from the current code and configuration, and it requires the
stage's up stages and gates to be current (r13 D-3). Let the driver rebuild; it re-runs only the stale stages.

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

## K. Audit round 8 (2026-10-26, fresh independent Opus review of fda6ccd); findings reproduced before fixing
* **B1** g02e crashed for Region 4: KY's 1986-88 periodic previous measurements have MEASDAY 0 (`make_date` out of range) → every FIADB date is built only
  from a valid month / day (unknown date → no match → a measurement-chain gap); also in g02b's queries. r4_v4 g02e now runs (398 gaps).
* **I1** the dynamics stage hash ignored `model_seed`, so switching the production seed skipped its enforced acceptance → `model_seed` in the dynamics cfg.
* **I2** g04's start-year gate and mapping-path calibration used g03b TRAINING plots (held out of the MDN only) → with VAL_SPLIT the calibration plots are
  the VALIDATION plots (never trained on by g03b or the MDN); older models keep the 15 % training-block rule.
* **M1** with VAL_SPLIT=0 an enforced acceptance could pass on the test plots → exit 4 whenever enforced without a validation set.
* **M2** a RISE of the forested share by >= 0.25 is a re-delineation that changes per-forested-area densities too → blocks like a drop (|Δ| >= 0.25).
* **M3** under SEVERITY_ANCHOR=1 a measured collapse (min BA ratio < 0.3) with no record was still forced onto event-free histories → in training, a
  collapse blocks the undisturbed rules (evaluation unchanged: outcomes are never used there).
* **M4** (visits of 2023 compared with AlphaEarth 2022 un-flowed) — NOT a defect: `ok = cst <= yr + 1` admits only visits up to the END of 2022, which
  flow forward to the end-2022 state; visits in calendar 2023 (decimal year > 2023) are excluded. Recorded here, no change.
* **M5** with both rs and ra the flow re-applied the start year's (pre-visit) cut → rs wins (the later record's own-year evidence is lost: 0.2 % of pairs).
* **M6** g04 validated median-sample QMD × its factor while g05 maps QMD from the calibrated TPH / BA → g04 validates the derived QMD.
* **M7** T4i was vacuous (a single-candidate plot) → a plot with >= 3 candidates and max prior < 0.9.
* **M8** g06's synthetic section used all 2019-20 visits → held-out (test ∪ validation) visits only. Selection between v3 and v4 is to be made on the
  validation metrics (g04 gate errors on the validation plots, g06 acceptance); g03b's held-out block stays the test report.
* While smoke-testing r8 two more of my edits had put code inside an inline comment (g06 synthetic set, gv_verify T4i) — the 4th / 5th instance of that
  bug class → `scripts/check_comments.py` (a tokenizer + ast scan; since r9 also plain-call assignments of names the code reads but never binds — it now catches all five earlier instances, none left) runs in the driver before any stage
  (refuses to start) and as gv_verify T0. Verification of r8 on gafl_v4: gv_verify 49 / 53 (T1k / T3e / T3f / T10 = the stale table, by design);
  smoke: train step, g06 (val acceptance enforced: exit 3; not enforced: 0), g04 (calibration on the 676 validation plots), g05; scratch deleted.

## L. Audit round 9 (2026-10-26, fresh independent Opus review of ca57292); no blocker, findings reproduced before fixing
* **I-1** no check compared a STOCKED visit's tokens with FIADB (T1 / T4 recompute from curated_trees; T1k covers bare visits only); on the stale table
  36 % of gafl_v4 visits (GA 50 %) and 42 % of r4_v4 visits have TPH < 0.9 × FIADB → gv_verify T1l (token TPH / BA vs FIADB live trees per forested
  area, shares from FIADB COND; median within 1 ± 0.02, < 2 % of visits below 0.9). Fails on the stale table by design; must pass after the rebuild.
* **I-2** the driver's own stdout redirected into the data dir (the documented `> data/proto/<data>/pipeline.out` practice) was attributed to the
  running stage and broke its currency → the driver excludes its own stdout / stderr targets from the snapshots.
* **M-1** check_comments missed plain-call assignments (the first instance, `trp_ = set(...)`) → also flags an assignment of a name the file's code
  reads but never binds; catches all five historical instances; g05's one prose comment that looked like code reworded.
* **M-2** data/proto/gafl_pipe's production stand map was the unvalidated B1 output of an interrupted pre-r6 run → restored to the B0 files of
  2026-10-23 (byte-equal to the _B1 copies before; the live FE layer was built from B0).
* **M-3** config/genveg.yml (no validation split) can no longer pass the enforced acceptance → frozen (the driver refuses it); default config v3b.
* Process: validation metrics are now reported by g03b (`validation`) and g04 (`mapping_path_validation_crossfit`), so v3 vs v4 is chosen on them.
* Verification of r9 on gafl_v4: gv_verify 49 / 54 — T1k / T1l / T3e / T3f / T10 = the stale table (T1l: TPH ratio q10 0.289, 35.7 % of stocked visits
  below 0.9, as the auditor measured); smoke train / g06 / g04 / g05 OK; g03b `validation` (633 plots) and g04 `mapping_path_validation_crossfit` reported;
  the driver with stdout redirected into the data dir: full run, then all current (stub root). Required after the slim_r4_v2 rebuild: 54 / 54.

## M. Audit round 10 (2026-10-26, fresh independent Opus review of d065407); no blocker, findings reproduced before fixing
* **I-1** g03b's kNN / null pool held the validation plots themselves (100 % self-matches; validation kNN BA error 0.233 vs 0.425 without them) → the
  pool is the training plots only (test and validation).
* **M-1** the upstream checks (vup) gated nothing → train depends on vup (G18).
* **M-2** LANDFIRE pixels coded by both a calendar-year and the next fiscal-year product counted 2/9 → once per class and event year (G20); T3e too.
* Suspected gate issue: the absolute shuffled-landscape criterion may be unpassable → relative to the 2021 path (G19).
* Verification of r10: gv_verify 49 / 54 (the same 5 stale-table checks); new g02b fractions = the once-per-pixel rule on 300 fresh footprints (0
  mismatches; 69 / 300 differ from the stale table); smoke train / g06 / g04 / g05 OK; validation kNN bias no longer exactly 0 (self-matches gone); the
  gate records both shuffled-landscape correlations.

## N. Audit round 11 (2026-10-27, fresh independent Opus review of 1b470bc); no blocker, findings reproduced before fixing
* **I-1** a model_seed outside train.seeds bypassed the enforced acceptance and mapped a stale checkpoint → the driver refuses it (G21).
* **I-2** gating train on vup by hash retrained everything on a verifier setting / new check → `gate` list (current and passed, not hashed); vup hashes
  only verify.n_upstream + fingerprints of its own inputs (TreeMap, EVT, Hansen, AOI mask, stand-run config, FE stands.js; CONUS TreeMap / EVT for R4).
* **M-1** g05's median of 16 vs g04's of 32 (torch.median = lower middle) biased the map 2-5 % low → shared `lmed` (G22).
* **M-2** localisation / synthetic subsets were the first N in key order (one state) → random samples.
* **M-3** the table checks ran only after mapping → stage vtab (gv_verify before training) gates train.
* **M-4** T1b ranked species over all conditions → forested conditions only, as g02b.
* **M-5** VRA had no check → T8g (independent from FIADB COND); g05's applied calibration factors recorded and compared (T11c); map checks required in
  the verify stage (T11z, U8).
* Not changed (negligible, recorded): the visit-year rule is class-agnostic (moves all of a visit year's evidence; 0.02 % of visits carry unrelated
  stand-replacing evidence in that year).
* Verification of r11: gv_verify 50 / 55 (the same 5 stale-table checks; T1b forested-condition spec and new T8g VRA pass); smoke OK; driver gating tested on the stub root (rerun all current; verify.n change re-runs vtab + verify only; a model_seed outside train.seeds is refused).

## O. Audit round 12 (2026-10-27, fresh independent Opus review of 6059b6f); no blocker, findings reproduced before fixing
* **I-1** one shared input fingerprint (and check_comments.py in every code hash) invalidated stages by files they never read: a re-written stand gpkg
  rebuilt the table and retrained every seed → per-stage inputs, vector files content-hashed (G25).
* **M-1** vtab ran T11 / T11b / T11c on stale map files → skipped entirely in vtab.
* **M-2** end-side visit-year asymmetry (0.28 % of pairs) — documented (G26).
* **M-3** T8b / T8d / T8e sampled the first items in key order (one state) → random order.
* **M-4** docs: vtab row, the State section (gate, inputs, file currency, refusals), G21 wording, §A "adopted" (the gafl_v4 table was RUN by the driver,
  1,687 s).
* **M-5** T11c trusted g04's `passed` → T11d re-derives the gate decision; R4 vtab / verify hash TIGER and the 30 m AlphaEarth mosaics they read.
* **S-1** the landscape criterion mixed Monte-Carlo noise → dependence above a same-landscape redraw (G24). **S-2** documented (G26).
* Verification of r12: gv_verify 50 / 55 (the 5 stale-table checks); smoke OK; the gate records the redraw / shuffled correlations and dependences; the stub root: touched gpkgs invalidate nothing, a check_comments edit re-runs vtab + verify only, a changed stand gpkg re-runs stands_fp onward (not table / train).

## P. Audit round 13 — the first MULTI-REVIEWER round (2026-10-27; 6 parallel Opus reviewers by slice + a fix-diff review of r4–r12; HEAD 5a367c3)
No blocker. Important findings, all reproduced, by slice:
* **B (model):** B-1 latent collapse onto the encoder bound (G29); B-2 ±1 jumps cannot represent stand replacement (G30); B-3 LTC equilibria confined to a box
  (G31). Fixed with switches SIGREG_MEAN / JUMP_RESET / STATE_ATTR, set in the v4 / r4 / v3b configs. Also ltau clamped and a dead counter removed.
* **C (evaluation):** C-1 the mapped (medoid) composition / dominant key was never validated → a shared `medoid_draw`, used by g05 and validated in g04
  (G32). C-2 the g06 acceptance used the lower median → `lmed` / true median. C-3 the gate is vacuous when dep0 ≤ 0 (G33). C-7 the kNN / null
  fallbacks are counted (`baseline_fallbacks`). C-8 the cross-fit halves are a seeded random split. C-5 the g06 synthetic set is fs_ok-filtered with
  max-prior soils.
* **A (data):** A-1 no stand-replacing evidence after 2024 (G36). A-2 DSTRBCD 41 / 60 now block (G37). A-3 end-to-end forest-share change (G38).
  A-4 / A-5 lookback yr0 − 4, and a non-sampled predecessor ⇒ lookback (G39). A-6 / A-7 bare test and all-bare plots (G40). A-9 converted plots
  are out of MIL (G41). A-12 deferred (G42). The R4 evidence table must be re-derived before an R4 table build (G34; now g02b.EVK_TABLES, G46.4).
* **D (driver):** see G43 / commit 1c518b9 (stand-pipeline modules hashed by used definitions, region-aware deps, relocatable fingerprints, portable
  own-output exclusion, complete per-stage inputs, refusals, lock, atomic state, a redesigned swallow detector).
* **E / F (verifier, fix diffs):** the verifier batch is a separate commit (production-config T8h, g04 set hygiene T12, acceptance re-derivation T15,
  vacuous / key-order / circular checks, coverage gaps). F#5 → G35 (span ratios over fs_ok visits).
While implementing, the redesigned guard caught two new comment-swallows in my own g06 edits (fixed before commit).

## Q. Audit round 14 (2026-10-27; multi-reviewer, HEAD e132568); decisions G45–G47
### Q.1 Data fixes (g02b / g02e / g02g / g02j; G46.1–6)
* **Incomplete curated visits (A14-1, G46.1).** g02b vets every curated visit against FIADB: n_cur = curated live tallied trees on the forested
  conditions, n_fia = FIADB's (`FV.n`). n_fia ≥ 3 and n_cur < n_fia / 2 ⇒ the visit leaves the table (no tokens, no pair endpoint, no bare visit);
  FIADB still supplies its records and negative evidence. Old curated DB (slim_r4): GA+FL 1,112 of 34,045 curated visits (FL 280, GA 832; 261
  with no curated live tree); Region 4 4,759 of 113,443 (1,037 with no curated live tree). meta.json: `incomplete_visits`, `incomplete_visits_by_state`, `incomplete_curated_visits`, `vetted_curated_visits`,
  `incomplete_no_live`. Re-check curated_v2 the same way.
* **Share change vs FIA's previous measurement (A14-2, G46.2).** The |Δ forested share| ≥ 0.25 block also compares each visit with its
  PREV_PLT_CN measurement (pre-2000 predecessors included; PLOT_STATUS 2 ⇒ share 0; non-sampled or without condition shares: no comparison).
  Additional blocked visits (negative periods removed): GA+FL 313 (627 from the consecutive rule), Region 4 1,171 (2,374). meta.json `share_blocks`.
* **G41 refined (F3 / A14-3, G46.3).** `plot_converted` ⇔ the last sampled FIADB visit dated ≤ 2022 has no forested condition and ≥ 1
  non-forest land / water one (COND_STATUS 2–4), after an earlier forested visit. Table plots: GA+FL 385 (old rule 402: 31 forested again by
  2022 leave, 14 enter), Region 4 1,034 (old 1,108: 109 leave, 35 enter).
* **Evidence table per region (A14-5, G46.4).** `g02b.EVK_TABLES` keyed by the sorted state tuple (GA+FL = (12, 13)); a region without an entry
  is refused, and so is AEF90 with GA+FL; EVK_REGION_OK is gone. meta.json `evk` (kind → [class indices, [dlo, dhi], lr]) and `evk_hash`
  (sha256 of `json.dumps(evk, sort_keys=True, separators=(',', ':'))`); g02e / g02g (and g03b) read it through `g02b.evk_from_meta` and pass it
  to `clusters(evs, evk)`.
* **G36 as a clip (A14-6, G46.5), data side.** fia_negs periods are (y_lo, min(visit year − 1, evid_end)); a period wholly after evid_end is empty
  and not listed. meta.json `evid_end` (2024) is the single source; g02g clips again with it. The location prior uses the same periods.
* **Provenance (G46.6).** `code_hash(path)` = sha256 of `ast.dump` of the module with every docstring removed. g02b: meta.json
  `code_hash: {"g02b": …}`. g02e / g02g / g02j: the parquet schema metadata key `genveg_provenance` of each of their parquet outputs
  (`{"code_hash": {"<script>": …, "g02b": …}, "evk_hash": …}`; g02j without evk_hash). They do not touch meta.json, so stands_fp stays
  independent of the table stages.
### Q.2 Model and evaluation fixes (g03b / g04 / g05 / g06; G46.7–21)
* **Converted plots out of every AlphaEarth term (G46.7).** MIL, Q and Q-decode are weighted by `~VCONV`; g04's B fine-tune (`trB`) leaves them
  out. The pair code has no FIA-only path (every pair flows along candidate histories), so `pair_set` returns None for a converted plot and counts
  it (`converted_dropped`; the current gafl_v4, with the r13 flag: 369 consecutive / 531 multi-horizon pairs). g03b refuses a table without
  `plot_converted`.
* **Evidence table from the table (G46.4).** g03b reads `EVK = g02b.evk_from_meta(meta)` and uses `clusters(evs) = g02b.clusters(evs, EVK)`;
  g06 uses the same `clusters`. `EVID_END` is the table's `evid_end`.
* **G36 as a clip (G46.5).** `pair_set` no longer blocks a span that passes EVID_END: the negative and event-free constraints act on the
  span's years up to EVID_END, and the years after it are unconstrained. The outcome-only windows were clipped already.
* **Latent acceptance (G46.8).** `latent_stats` is ok iff, over the non-clock dims, max |mean| ≤ 2, min per-dim sd ≥ 0.3 and the share of
  |z| > 2.9 is ≤ 1 %. It writes `max_abs_mean`, `min_sd`, `share_abs_gt_2p9` and `ok`, with `effective_rank` reported only. Checked against the
  existing evidence:
  - all 22 v3 checkpoints of gafl / gafl_pipe FAIL on the test encodings: max |mean| 2.87–2.99, share > 2.9 0.47–0.76, min sd 0.02–0.53;
  - the r14 B harness OFF run FAILS at step 1200 (2.89 / 0.21 / 28.7 %);
  - the ON run PASSES at step 1200 (0.57 / 0.77 / 0.03 %; it fails only at step 300, min sd 0.24).
* **SIGREG_FLOW / AGE_FLOW (G46.9).** Two training-only switches, default 0:
  - SIGREG_FLOW: SIGReg (a weighted empirical characteristic function) on the flowed states at the pair end, weighted by the pair posterior;
  - AGE_FLOW: W_AGE · the squared error, over AGE_S, of the decoded BA-weighted mean age (`dec_age`, the age g06 measures) of the flowed state
    against v2's observed age in the decoder's form (`OBS_AGED`, class midpoints), stocked v2 only.

  The A/B against the r14 B ON harness is under "SIGREG_FLOW / AGE_FLOW evidence" below.
* **Bounded resets and cls init (G46.10).** JUMP_RESET: z ← z + r ⊙ (zt − z) with zt = clamp(z_post + sc·tanh(shift), −3, 3), and the clock
  target ≥ age 0. EncV4.cls ~ N(0, 0.02): the step-1 pre-clip gradient norm on the harness is 41 (it was 1422).
  - With the switches off, outputs are bit-identical to r13 HEAD and to pre-r13 (B's bitid harness: v3 gafl_pipe/model.pt, and v4 at random
    init with SITE_ATTR + AGE_CH; max |Δ| = 0 for z, flow states 2010 / 2025, decoder, head and sigreg).
* **Evaluation (G46.11–19).**
  - One estimator (G46.13). The shared region of g04 (exec'd by g05 / g06) holds `MS = 32`, `tmed` (the true median),
    `calibrated_median(s, cf) = e^cf · lmed(s)` and `medoid_draw`. g05 draws MS iid prior samples (USE_B = 1: 64 reweighted and resampled to MS)
    and takes its medoid on the uncalibrated draws, as g04 does.
  - Replicates (G46.11). The start-year gate statistics and CALF are means of R_REP = 8 replicates, with per-replicate values and the sd recorded.
    Each replicate uses a fresh derangement of the landscapes.
  - NaN (G46.18). A NaN gate statistic fails the gate with `fail_reasons`. The fallback counters are kept per query set.
  - Map acceptance (G46.12). The validation cross-fit (the R_REP replicates × 2 halves) is recorded with the null and kNN on the same visits.
    `map_accept` requires the TPH error, the BA error and the medoid Bray each to be below the null's; g05 refuses otherwise.
  - Report only:
    - the visit-date vs end-2025 shift (G46.14);
    - the medoid age-class Wasserstein-1 (by TPH and by BA) and the bare-visit predicted BA (G46.19).
  - True medians (G46.15) in CF / err / bias / predA / CAL / AxB / the null / g03b metrics / g06 reconstruction.
  - g06 (G46.16 / G46.17):
    - the undisturbed set is `UNDIST` from `pair_set`: no hiding code (incl. G38), no detectable record, a constrained span part (G36 clip),
      no measured collapse;
    - the pre-registered horizon bins are (0, 8), (8, 15), (15, 99) yr, each requiring ≥ 20 validation pairs. They are fixed from the table
      counts, not from model results. Validation counts: gafl_v4 1192 / 636 / 252; r4_v4 4202 / 2362 / 902 (r4_v4 is stale and has no
      `plot_converted`, so its counts are approximate);
    - g06 writes `dynamics_pairs_<TAG>.parquet` (one row per evaluated pair: set, pid, v1, v2, horizon, predicted / observed BA log change and age
      per yr) and records `pair_dump`, `bins_required`, `n_min` and `fail_reasons`.
* **Documented residuals (G46.20).** The X_L standardisation, k-means and soil statistics use all plots (no labels). USE_B = 1 is mismatched
  and not validated (g05 docstring). The 2026-visit clamp is negligible.
* **Interfaces for the verifier (G48.5).**
  - Checkpoints carry `switches`: every stage switch g03b reads → its string value (None = unset). Older checkpoints without the key still load.
  - g04 writes `g04_plots.parquet` (`bayes_eval.json` `plot_dump`), one row per (kind, y0, rep, half, visit):
    - kinds: `A_cal` (the CAL fit), `cf_fit` (a factor fit: half A / B = a cross-fit half, `all` = a CALF replicate), `score` (a judged half, or
      `test`), `baseline` (null / kNN on the validation halves and on test);
    - columns: set, y0, rep, half, pid, vid, stocked, obs_* / obs_log_* (log(obs + 1e-3)), pred_* / pred_log_* (the uncalibrated true median of the
      draws), cf_* (the factors applied), bray_medoid, dom_hit_medoid, null_*, knn_*, bray_null, bray_knn, dom_hit_knn (* = TPH, BA, QMD);
    - CAL, CALF, the validation errors and Bray, the null and map_accept are recomputed from it exactly (smoke subset).
* **SIGREG_FLOW / AGE_FLOW evidence (G46.9; B's 1200-step harness on its 1600-plot subset, NT 2).** Three runs:
  - B ON: B's r13-ON checkpoint, re-probed with the current code;
  - ctl: the current code with the r13 switches on and the two flow switches off;
  - flow: the same with SIGREG_FLOW = AGE_FLOW = 1.

  g06 is the current g06 throughout (validation bins (0, 8) / (8, 15) / (15, 99), n 221 / 115 / 54).

  | (val unless noted) | B ON | ctl | flow |
  |---|---|---|---|
  | latent max \|mean\| / min sd / share \|z\| > 2.9 / eff. rank | 0.57 / 0.77 / 0.03 % / 5.86 | 0.34 / 0.63 / 0.1 % / 6.27 | 0.43 / 0.78 / 0.05 % / 6.82 |
  | stand-replacing: closest one-event-year RMS / its BA ratio | 0.35 / 0.003 | 0.44 / 0.002 | 0.39 / 0.004 |
  | stand-replacing: trained jump BA ratio / posterior-weighted flow BA ratio | 0.083 / 0.073 | 0.034 / 0.051 | 0.014 / 0.047 |
  | g06 synthetic clear-cut / fire (ratio to none) | 0.004 / 0.92 | 0.005 / 0.92 | 0.008 / 0.82 |
  | undisturbed-like pairs: NN ratio flowed (z1 encoded) | 1.25 (1.32) | 1.36 (1.25) | 1.50 (1.47) |
  | 25-yr no-event flow: NN ratio (encoded z1) | 1.15 (1.32) | 1.41 (1.25) | 1.49 (1.47) |
  | 25-yr no-event flow: latent variance retained / eff. rank | 12 % / 2.18 | 10 % / 2.50 | 32 % / 3.21 |
  | 25-yr no-event flow: BA log change / yr | +0.0014 | −0.0010 | +0.0059 |
  | g06 BA log change / yr by bin (obs +0.022 / +0.014 / +0.013) | +0.001 / +0.000 / +0.003 | +0.000 / −0.002 / −0.002 | +0.009 / +0.007 / +0.007 |
  | g06 decoded age per yr ÷ observed, by bin | 0.43 / 0.52 / 0.61 | 0.39 / 0.49 / 0.66 | 0.39 / 0.45 / 0.59 |
  | decoded age bias at v1 / at the flowed v2 (yr) | +4.3 / +1.7 | +5.1 / +2.3 | +2.9 / +0.2 |
  | composition, undisturbed pairs ≥ 10 yr: Bray / dominant kept (obs 0.094 / 0.82) | 0.137 / 0.76 | 0.147 / 0.82 | 0.139 / 0.79 |
  | composition, 25-yr no-event flow: Bray / dominant kept | 0.335 / 0.59 | 0.375 / 0.56 | 0.281 / 0.66 |
  | g06 acceptance | fail (BA ×3, age ×3) | fail (BA ×3, age ×3) | fail (BA 0–8, age ×3) |

  **The pre-registered acceptance is not met, so the switches are not forced.**
  - Improved: latent-variance retention (×2.7), BA growth (two of three bins now within ±0.01), the decoded age level, and the long-flow
    composition.
  - Not improved:
    - the decoded age RATE (0.4–0.6×; the flowed age level is unbiased now, but the decoded ages remain compressed, so the age change lags);
    - the pair-level composition Bray;
    - the NN ratio, which rose with the retained spread to the encoded baseline (B ON's low value reflects contraction toward dense regions).
  - Small regressions: the synthetic fire ratio 0.92 → 0.82 and clear-cut 0.004 → 0.008.

  A change-based age loss (decoded Δage vs observed Δage) would target the rate; that is not in G46.9. Decision for the coordinator.
