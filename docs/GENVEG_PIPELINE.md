# Generative-vegetation pipeline — one place to run, document and check the model

Driver: `scripts/genveg_pipeline.py`. Configuration: `config/genveg_v3b.yml` (default; `config/genveg.yml` is the frozen pre-audit v3 config, see docs/DECISIONS.md G13), which holds every switch of every stage; `--set key=value` overrides one,
and unknown keys are refused. Method and audit history: `docs/GENERATIVE_VEG_PLAN.md`. The stands it maps come from a run of the stand-delineation
pipeline (`docs/STANDS_PIPELINE.md`), named by `stands_run`.

```bash
.venv/bin/python -I scripts/stands_pipeline.py                         # the stand run (gafl_pilot) the genveg config names
.venv/bin/python -I scripts/genveg_pipeline.py                         # every stage, in order
.venv/bin/python -I scripts/genveg_pipeline.py --set data=gafl_try --set fe.geojson=data/proto/gafl_try/baker_fl.geojson   # a separate run (new data dir AND its own FE layer)
.venv/bin/python -I scripts/genveg_pipeline.py map fe verify --force map  # re-run some stages
.venv/bin/python -I scripts/genveg_pipeline.py --log data/proto/gafl_v3b/run.txt   # keep the driver's output (NOT `| tee` into the data dir)
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
| `dynamics` | `g06_dynamics_check.py` per seed × `eval_loc` | check: `dynamics_check_<name>_<eval_loc>_s<seed>.json` (report-only except the production seed's `history` record); the exit status of `model_seed` (G06_ENFORCE) is the acceptance |
| `downscale` | `g04_bayes_downscale.py` (model `model_seed`) | `bayes_models.pt`, `bayes_kmeans.pkl`, `bayes_eval.json` |
| `map` | `g05_map_stands.py` per `use_b` | `stand_veg_v3_B<b>.parquet` + `stand_cohorts_v3_B<b>.parquet`; the last `use_b` is promoted to `stand_veg_v3.parquet` / `stand_cohorts_v3.parquet`, plus `stand_nodes.npz` |
| `fe` | `24_export_fe.py` (area polygon `fe.area`) + `g07_fe_attach.py` | FE stands GeoJSON (`fe.geojson`, relative to the checkout) with GEN_* attributes + its `_cohorts.json`. The host serves it; restart / reload Forest Explorer on the host |
| `verify` | `gv_verify.py` (`GV_REQUIRE_MAP=1`) + `gv_verify_upstream.py` (with U8, the stand-mapping internals) | check: independent recomputation of every stage |

**Environment.**
- The training environment (`train.env`) is passed to the stages that load the model through g03b or check it: train, dynamics, downscale, map,
  vtab and verify (gv_verify). The table gets its own section, and each of dynamics / downscale / map adds its `env` section. The top-level `env`
  (region switches) goes to every stage.
- Region switches (`REGION`, `AEF90`) are accepted only in the top-level `env`; any switch in a section whose stages' scripts never read it is
  refused (e.g. `AEF90` under `table`, `G06_SET` under `downscale`). `CF16` is a train switch (`train.env`).
- Config values are checked (G45.2): `dynamics.eval_loc` must be a non-empty list of names containing `history`, `map.use_b` a non-empty list of
  booleans (`true` / `false` or `1` / `0`, hashed as `1` / `0`), `train.seeds` a list of integers and `model_seed` an integer among them. A null
  `env:` section means `{}`; an empty value (`MAP_Y0:`) or a list / mapping value is refused; a float is refused for a switch no script parses with
  `float()` (`EPOCHS: 14.0`, `SITE_ATTR: 1.0`).
- YAML booleans are passed as `1` / `0` (as the scripts test them), and hashed in that form.
- The chosen model (`<model_prefix>_s<model_seed>.pt`) is also copied to `model.pt` before `downscale`, `map` and `verify`, for tools that read the default.
- Every switch a stage script reads (`STAGE_SWITCHES`) is stripped from the inherited shell environment; only the config sets it. The driver refuses
  to run if a stage script reads a switch missing from that list, and refuses a config env key that is not one of them (a typo). The reads are
  found in the scripts' syntax tree (`os.environ.get / setdefault / pop / [...]`, `in os.environ`, `os.getenv`, aliases of them); a read whose name
  is not a constant (or a loop over a literal) is refused.
- The driver sets `STANDS_RUN`, `DATA`, `WORKSPACE`, `FE_GEOJSON`, `GV_CURATED_DB` / `GV_FIADB` (from `inputs`) and, per stage, `MODEL`, `SEED`,
  `USE_B`, `TAG`, `EVAL_LOC`, `G06_ENFORCE`, `GV_SKIP_U8`, `GV_REQUIRE_MAP` and (verify) `GV_DYN_JSON` = the production acceptance record
  `dynamics_check_<name>_history_s<model_seed>.json`. A config that sets any of these, or the debug switches `LOAD`, `SMOKE_BATCHES`, `STAND_DIAG`,
  is refused.
- Library environment (`PROJ_*`, `GDAL_*`, OMP / MKL thread counts, `CUDA_VISIBLE_DEVICES`) is inherited and not hashed, like `NT`: performance /
  environment knobs, not model switches.

**State.** `data/proto/<data>/pipeline_state.json` (written atomically) stores, per stage, a hash of:
- its config sections (or dotted keys, e.g. vup hashes only `verify.n_upstream`), without the performance knobs `table.NPROC` and `NT` of every env
  section: a thread-count change does not invalidate anything. (It can change the order of float reductions, so a re-run with another thread count
  need not be bit-identical; it is not a different model.)
- its code: the stage's scripts, the genveg modules `stand_runs.py` / `region.py` whole, and from the stand-pipeline modules `features.py` /
  `common.py` every module-level statement but def / class (importing runs them: `SOILS.append(...)`, `os.environ[...] = ...`) plus the top-level
  definitions its scripts import (and what those reference); comments and docstrings do not count. `stands_cfg.py` (imported by features.py; it
  loads config/pilot.yml at import, which the definitions genveg imports do not use) and `stands_stages.py` (the driver's `--set` parser) are
  scanned by check_comments but not hashed; the Region 4 raster builders (`r4_grid.py`, `r4_aef90.py`, `r4_rasters.py`) are scanned, not hashed:
  their outputs are fingerprinted.
- its `up` stages: their hashes and the fingerprints of the files their runs wrote;
- the stage's OWN inputs (`stage_inputs`):
  - the FIA databases it reads: the configured WORKSPACE-relative path, size and mtime (a moved checkout keeps its hashes when mtimes are kept:
    `mv`, `cp -a`; a `cp -r` copy re-runs everything);
  - its rasters / tables: size and mtime (GA+FL rasters, or under AEF90 the Region 4 rasters; TIGER `.shp/.shx/.dbf/.prj` for the R4 checks);
  - the CONTENT of its vector files (stand polygons, county outline, `fe.area`; geometries, attributes, column names and dtypes) and of the stand
    run's `config_resolved.yml` (the stand pipeline rewrites it on every run);
  - the check stages vup / vtab / verify: exactly the verifiers' own lists — `VERIFY_INPUTS` (and `VERIFY_CODE`, by content) of
    gv_verify.py / gv_verify_upstream.py, evaluated in a subprocess under the stage's own env (the lists depend on the region and the table), and
    re-evaluated after a stage changed files. Stand-run rasters by pixel content, its gpkg / config by content, a shapefile with its sidecars,
    everything else by size + mtime; a missing path is recorded. A verifier that fails to import refuses the run; a list naming a data-dir file a
    LATER stage writes refuses too (that check could never be current). Remote AlphaEarth COGs (T3c2z) cannot be fingerprinted. A change to a
    vup input re-runs train, which is gated on vup, only after vup has passed again;
  - train: `cand_loc_aef.parquet` / `cand_prior.parquet` when `LOC` is `aef` / `plot`;
- pixel-content fingerprints (every band, CRS, nodata) of the stand-run rasters it reads: `land_clean` (table, stand footprints, map; not under
  AEF90, which reads `data/r4/land.tif`), `stands.tif` (stand footprints, map); the check stages as their verifiers list. A stand-pipeline re-run
  with identical outputs does not invalidate genveg. The pixel / vector hash caches (`*.pixhash.json`, `*.vechash.json` next to the file) are
  written atomically; an unreadable one is recomputed.
- `name` enters the hash of dynamics and verify (it names the acceptance records).
- r14 (G45) changed the hash definitions (module-level statements, docstrings, the check stages' verifier lists, `name`): the first run after it
  re-runs every stage of an existing data dir, and a stage adopted before r14 (no file list) is stale until it is run.

**up vs gate.** `up` stages are hash dependencies: the stage reads their outputs, so a re-run of an up stage re-runs it. A `gate` stage must be
current (so it ran and passed) before the stage runs, but does not enter its hash: train is gated on `vup` and `vtab` (a verifier setting or a new
check does not retrain), downscale on `dynamics` (its acceptance gates mapping; g04 reads none of its files), history on `localisation` (g02g
overwrites localisation.parquet; it does not read it). verify has dynamics as an `up` stage (T15 reads its acceptance record).

**Currency.** A stage is current only if (1) its hash matches, (2) the files its run wrote are unchanged on disk, or were rewritten by a LATER stage
of the pipeline and are still as that stage left them (a rewrite by an earlier stage, e.g. localisation re-run after history, makes it stale), and
(3) every `up` stage is current, recursively — so running a named stage never reports "current" over changed upstream outputs. fe also tracks its
layer (`fe.geojson` + `_cohorts.json`, size and mtime), wherever it is. A failing gate does not make the stages gated on it stale (G21): their
status prints `current (gate FAILED: <gate>)`.
- Report-only outputs do not count for currency: an output counts only if a later stage or the product reads it. Report-only are `eval_*.json`,
  `dynamics_check_*.json` except the production acceptance record `dynamics_check_<name>_history_s<model_seed>.json` (verify T15 re-derives the
  acceptance from it), `stand_veg_q_*.parquet`, the non-production `stand_veg_v3_B<b>` / `stand_cohorts_v3_B<b>` maps (every `use_b` but the
  last) and logs. Deleting one does not re-run anything; deleting the production acceptance record re-runs dynamics and verify (not training).
- Files are attributed to a stage by a snapshot diff of the data dir. **No other process may write into the data dir during a run**: its files
  would be attributed to the running stage. Excluded: `pipeline_logs/`, the state and lock files, top-level `*.out` / `*.log`, the driver's own
  redirected stdout / stderr and its `--log` file (matched by device + inode) and dangling symlinks. Use `--log FILE` to keep the output: a shell
  `| tee` into the data dir under any other name is a file that grows during every stage, so the next run re-runs everything.
- Only the inputs of the stages being run or checked (and their ups / gates) are resolved: a missing input of another stage does not refuse.
- One driver per data dir: a second driver on the same dir is refused (a lock on `<data>/.pipeline.lock`). Each stage runs in its own process group
  and inherits the lock, so the dir stays locked while a stage of a killed driver still runs; on SIGTERM / SIGINT the driver stops the stage's
  process group (SIGTERM, SIGKILL after 10 s), waits, then exits (status 128 + signal); the stopped stage is not current.

**Refusals** (before any stage runs or the state is touched):
- a frozen config — `frozen:` is checked on the raw file, before `--set`, so `--set frozen=` cannot bypass it;
- a config value of the wrong type or empty (see Environment), a `model_seed` not in `train.seeds`, `dynamics.eval_loc` without `history`;
- a config env key the driver owns, that is not a stage switch, a region switch outside the top-level `env`, or a switch no script of the
  section's stages reads;
- `--force` / `--adopt` naming an unknown stage or one not among the stages being run; `--adopt` of a check stage;
- code swallowed by a comment (`check_comments.py` over `check_comments.pipeline_scripts()`, the same list gv_verify T0 scans).

A current stage is skipped; a stage whose `up` stages or gates are not current is refused. `--adopt` records the stage's files from its last
recorded run (the file list in the state), as they are now, as its current outputs without running it: the output fingerprint is recomputed, so
an externally rebuilt output re-runs everything downstream, and later changes make it stale as for a run. It is refused when any recorded output
is missing, when the stage has no recorded run (also an adoption recorded before r14), for the check stages vup / vtab / dynamics / verify (their
result is their exit status), and unless its `up` stages and gates are current. Use it only for artifacts known to come from the same code and
configuration — never across a code change. The `gafl` table, for example, is NOT reproducible by `table`: it is g02b under the old matching rules
re-weighted by `g02i_reprior.py`, which is not a pipeline stage.

**check_comments.py** splits a comment at its start, after `;` / `:` / `)` / `]` (strong) and after every space or punctuation mark (weak), and
flags it when a remainder parses fully as Python and contains code: at a strong point any statement but a bare expression (assignments incl. alias
and walrus, raise / assert / continue / break / pass / return / global, blocks, any written call); after prose words only code-dense fragments
(dense assignments, keyword statements, method calls, calls with non-literal arguments). It also flags swallowed elements of a multi-line literal,
inline tails starting with an operator / subscript / attribute that still parse appended to the line's code, and `, k=v)` closing a bracket the
comment did not open. Its regression suite is `scripts/tests/test_check_comments.py` (0 misses required). Write formulas in comments with `·`,
`←`, `≔`, `÷`, words, or in backticks; write array shapes as `shape (n, m)`, not `[n, m]`.

## One-off / analysis scripts (not stages)

- `g02h_fix_visit_dates.py`: a migration, already applied.
- `g02i_reprior.py`: re-weights an existing table under new matching rules without a rebuild. It is not used by the pipeline, which rebuilds.
- `g02f_location_em.py`, `g08_sameyear_aef_test.py`, `g09_fia_landfire_association.py`, `g10_fast_viability.py`, `g11_raster_information.py`:
  analyses and viability tests.
- `g02c_fetch_visit_aef.py`: download of same-year AlphaEarth windows.

## Verification

- `vup`, `vtab` and `verify` are stages; their exit status must be 0 (so is the production seed's `dynamics` acceptance).
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
