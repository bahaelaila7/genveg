# Generative-vegetation pipeline — one place to run, document and check the model

Driver: `scripts/genveg_pipeline.py`. Configuration: `config/genveg_v3b.yml` (default; `config/genveg.yml` is the frozen pre-audit v3 config, see docs/DECISIONS.md G13), which holds every switch of every stage; `--set key=value` overrides one,
and unknown keys are refused. Method and audit history: `docs/GENERATIVE_VEG_PLAN.md`. The stands it maps come from a run of the stand-delineation
pipeline (`docs/STANDS_PIPELINE.md`), named by `stands_run`.

```bash
.venv/bin/python -I scripts/stands_pipeline.py                         # the stand run (gafl_pilot) the genveg config names
.venv/bin/python -I scripts/genveg_pipeline.py                         # every stage, in order
.venv/bin/python -I scripts/genveg_pipeline.py --set data=gafl_try --set fe.geojson=standmap/data/proto/gafl_try/baker_fl.geojson   # a separate run (new data dir AND its own FE layer)
.venv/bin/python -I scripts/genveg_pipeline.py map fe verify --force map  # re-run some stages
```
A separate run needs a NEW data dir name and its own `fe.geojson`: `--set data=<existing dir>` continues that dir's state (its stages are current or
stale against this config), and a shared `fe.geojson` would overwrite the other run's layer.

## Stages

| stage | script | output (data/proto/`<data>`/) |
|---|---|---|
| `table` | `g02b_build_gafl.py` (env: the `table` section) | `visits`, `tokens`, `pairs`, `plots`, `cands` .parquet, `cand_feat.npy`, `meta.json` — FIA visits, cohort tokens, footprint candidates in the 1.6 km fuzz disk (forest mask = the stand run's `land_clean`; AEF90: `data/r4/land.tif`), location prior |
| `localisation` | `g02d_localisation.py` | `localisation.parquet` (keyed by FIA plot identity) |
| `records` | `g02e_fia_records.py` | `fia_events`, `fia_negs` .parquet, `visits.fia_blocked` |
| `history` | `g02g_history_consistency.py` | `cand_hist.parquet`; overwrites `localisation.parquet` with history-consistent sets |
| `stands_fp` | `g02j_stand_footprints.py` | `stands.parquet`, `stand_feat.npy` — footprints of the stand run's Baker County stands |
| `vup` | `gv_verify_upstream.py` (`GV_SKIP_U8=1`: without U8) | check: grid (U1), forest-land rules from the stand run's resolved config (U6; U1r / U6r under AEF90), stand polygons vs raster (U7), FE units (U9) |
| `vtab` | `gv_verify.py` (`GV_REQUIRE_MAP=0`: map checks skipped) | check before any training: the table-level data checks (T0–T10, incl. the pre-registered T1k / T1l) |
| `train` | `g03b_train_gafl.py` per seed | `<model_prefix>_s<seed>.pt`; report-only `eval_<model>.json` |
| `dynamics` | `g06_dynamics_check.py` per seed × `eval_loc` | `dynamics_check_<name>_<eval_loc>_s<seed>.json` (report-only except the production seed's `history` record); the exit status of `model_seed` (G06_ENFORCE) is the acceptance |
| `downscale` | `g04_bayes_downscale.py` (model `model_seed`) | `bayes_models.pt`, `bayes_kmeans.pkl`, `bayes_eval.json` |
| `map` | `g05_map_stands.py` per `use_b` | `stand_veg_v3_B<b>.parquet` + `stand_cohorts_v3_B<b>.parquet`; the last `use_b` is promoted to `stand_veg_v3.parquet` / `stand_cohorts_v3.parquet`, plus `stand_nodes.npz` |
| `fe` | `24_export_fe.py` (area polygon `fe.area`) + `g07_fe_attach.py` | FE stands GeoJSON (`fe.geojson`, relative to WORKSPACE) with GEN_* attributes + its `_cohorts.json`. The host serves it; restart / reload Forest Explorer on the host |
| `verify` | `gv_verify.py` (`GV_REQUIRE_MAP=1`) + `gv_verify_upstream.py` (with U8, the stand-mapping internals) | check: independent recomputation of every stage |

**Environment.**
- The training environment (`train.env`) is passed to the stages that load the model through g03b or check it: train, dynamics, downscale, map,
  vtab and verify (gv_verify). The table gets its own section, and each of dynamics / downscale / map adds its `env` section. The top-level `env`
  (region switches) goes to every stage.
- YAML booleans are passed as `1` / `0` (as the scripts test them), and hashed in that form.
- The chosen model (`<model_prefix>_s<model_seed>.pt`) is also copied to `model.pt` before `downscale`, `map` and `verify`, for tools that read the default.
- Every switch a stage script reads (`STAGE_SWITCHES`) is stripped from the inherited shell environment; only the config sets it. The driver refuses
  to run if a stage script reads a switch missing from that list, and refuses a config env key that is not one of them (a typo).
- The driver sets `STANDS_RUN`, `DATA`, `WORKSPACE`, `FE_GEOJSON`, `GV_CURATED_DB` / `GV_FIADB` (from `inputs`) and, per stage, `MODEL`, `SEED`,
  `USE_B`, `TAG`, `EVAL_LOC`, `G06_ENFORCE`, `GV_SKIP_U8`, `GV_REQUIRE_MAP`. A config that sets any of these, or the debug switches `LOAD`,
  `SMOKE_BATCHES`, `STAND_DIAG`, is refused.

**State.** `data/proto/<data>/pipeline_state.json` (written atomically) stores, per stage, a hash of:
- its config sections (or dotted keys, e.g. vup hashes only `verify.n_upstream`), without the performance knobs `table.NPROC` and `NT` of every env
  section: a thread-count change does not invalidate anything. (It can change the order of float reductions, so a re-run with another thread count
  need not be bit-identical; it is not a different model.)
- its code: the stage's scripts, the genveg modules `stand_runs.py` / `region.py` whole, and from the stand-pipeline modules `features.py` /
  `common.py` only the top-level definitions its scripts import (and what those reference; comments do not count). `stands_cfg.py` and the
  Region 4 raster builders (`r4_grid.py`, `r4_aef90.py`, `r4_rasters.py`) are scanned by check_comments but not hashed: their outputs are fingerprinted.
- its `up` stages: their hashes and the fingerprints of the files their runs wrote;
- the stage's OWN inputs (`stage_inputs`):
  - the FIA databases it reads: the configured WORKSPACE-relative path, size and mtime (a moved checkout keeps its hashes);
  - its rasters / tables: size and mtime (GA+FL rasters, or under AEF90 the Region 4 rasters; TIGER `.shp/.shx/.dbf/.prj` for the R4 checks);
  - the CONTENT of its vector files (stand polygons, county outline, `fe.area`; geometries, attributes, column names and dtypes) and of the stand
    run's `config_resolved.yml` (the stand pipeline rewrites it on every run);
  - vup / verify: every file gv_verify_upstream reads (grid.json, AlphaEarth, Hansen, LANDFIRE EVT / dist, TreeMap, soils, AOI mask, the FE
    `stands.js` it parses; under AEF90 the Region 4 grids / rasters, `config/pilot.yml` and the CONUS TreeMap / EVT). A change to any of them
    re-runs vup and verify — and train, which is gated on vup, only after vup has passed again;
  - vtab / verify: the data-dir files no stage writes but gv_verify reads when present (`visit_aef/*.npy`, `cand_prior.parquet`, `location_em.json`,
    `cand_loc_aef.parquet`); train: `cand_loc_aef.parquet` / `cand_prior.parquet` when `LOC` is `aef` / `plot`;
- pixel-content fingerprints of the stand-run rasters it reads: `land_clean` (table, stand footprints, map, vtab; not under AEF90, which reads
  `data/r4/land.tif`), `stands.tif` (stand footprints, map, vtab), and `land` / `land_clean` / `stands.tif` (vup, verify). A stand-pipeline re-run
  with identical outputs does not invalidate genveg.

**up vs gate.** `up` stages are hash dependencies: the stage reads their outputs, so a re-run of an up stage re-runs it. A `gate` stage must be
current (so it ran and passed) before the stage runs, but does not enter its hash: train is gated on `vup` and `vtab` (a verifier setting or a new
check does not retrain), downscale on `dynamics` (its acceptance gates mapping; g04 reads none of its files), history on `localisation` (g02g
overwrites localisation.parquet; it does not read it). verify has dynamics as an `up` stage (T15 reads its acceptance record).

**Currency.** A stage is current only if (1) its hash matches, (2) the files its run wrote are unchanged on disk, or were rewritten by a LATER stage
of the pipeline and are still as that stage left them (a rewrite by an earlier stage, e.g. localisation re-run after history, makes it stale), and
(3) every `up` stage is current, recursively — so running a named stage never reports "current" over changed upstream outputs. fe also tracks its
layer outside the data dir (`fe.geojson` + `_cohorts.json`, size and mtime).
- Report-only outputs do not count for currency: an output counts only if a later stage or the product reads it. Report-only are `eval_*.json`,
  `dynamics_check_*.json` except the production acceptance record `dynamics_check_<name>_history_s<model_seed>.json` (verify T15 re-derives the
  acceptance from it), `stand_veg_q_*.parquet`, the non-production `stand_veg_v3_B<b>` / `stand_cohorts_v3_B<b>` maps (every `use_b` but the
  last) and logs. Deleting one does not re-run anything; deleting the production acceptance record re-runs dynamics and verify (not training).
- Files are attributed to a stage by a snapshot diff of the data dir. **No other process may write into the data dir during a run**: its files
  would be attributed to the running stage. Excluded: `pipeline_logs/`, the state and lock files, top-level `*.out` / `*.log` (so
  `driver | tee data/proto/<data>/run.out` is safe), the driver's own redirected stdout / stderr (matched by device + inode) and dangling symlinks.
- One driver per data dir: a second driver on the same dir is refused (a lock on `<data>/.pipeline.lock`).

**Refusals** (before any stage runs or the state is touched):
- a frozen config — `frozen:` is checked on the raw file, before `--set`, so `--set frozen=` cannot bypass it;
- a `model_seed` not in `train.seeds`; `dynamics.eval_loc` without `history`;
- a config env key the driver owns, or that is not a stage switch;
- `--force` / `--adopt` naming an unknown stage or one not among the stages being run;
- code swallowed by a comment (`check_comments.py` over `check_comments.pipeline_scripts()`, the same list gv_verify T0 scans).

A current stage is skipped; a stage whose `up` stages or gates are not current is refused. `--adopt` records existing outputs as current without
running them; it requires the stage's `up` stages and gates to be current. Use it only for artifacts known to come from the same code and
configuration — never across a code change. The `gafl` table, for example, is NOT reproducible by `table`: it is g02b under the old matching rules
re-weighted by `g02i_reprior.py`, which is not a pipeline stage.

**check_comments.py** flags a comment when the rest of it, from its start, a `;`, a `)` / `]` or a word, parses fully as Python and contains code
(an assignment, del, a block, a call; after prose words only code-dense fragments). Write formulas in comments with `·`, `←`, `≔` or in backticks.

## One-off / analysis scripts (not stages)

- `g02h_fix_visit_dates.py`: a migration, already applied.
- `g02i_reprior.py`: re-weights an existing table under new matching rules without a rebuild. It is not used by the pipeline, which rebuilds.
- `g02f_location_em.py`, `g08_sameyear_aef_test.py`, `g09_fia_landfire_association.py`, `g10_fast_viability.py`, `g11_raster_information.py`:
  analyses and viability tests.
- `g02c_fetch_visit_aef.py`: download of same-year AlphaEarth windows.

## Verification

- `vup`, `vtab` and `verify` are stages; their exit status must be 0.
- The stand run itself is verified by `scripts/stands_verify.py` (stage `verify` of the stand pipeline).

## First end-to-end run (2026-10-22, `--set data=gafl_pipe`)

- Stages ran from the table on, with the strict_unknown rules and the fixed stand map. table, localisation, records, history, train and dynamics were
  adopted after the shared-library hashing fix; their scripts and config were unchanged.
- **Bugs the run exposed (fixed):**
  - g04: the `trp_` assignment was swallowed by an inline comment.
  - gv_verify: the T8 anchor text had changed, and T4 had DATA hard-coded to gafl.
  - g02j: 2,103 sliver stands got no footprint; they now get one at their central pixel.
- **verify: 45 / 45 PASS.**
- **Mapping path, held-out visits 2022–2025, median |log error|:**

  | | trees/ha | BA | QMD |
  |---|---|---|---|
  | rebuild | 0.611 | 0.412 | 0.308 |
  | previous model | 0.607 | 0.424 | 0.299 |
  | kNN | 0.664 | 0.414 | 0.359 |
  | null | 0.609 | 0.453 | 0.327 |

  The rules change is neutral. The model is only modestly informative: better than kNN on QMD, better than null on BA, no better than null on trees/ha.
- **The direct predictor's BA level depends on the training seed.** g03b held-out BA error is 0.52 / 0.65 (old seeds) and 0.71 / 0.54 (new seeds), from a log bias
  of −0.27 to −0.62. g04's calibration absorbs this on the mapping path, but the training instability should be fixed at the source.
