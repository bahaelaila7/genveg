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
| `dynamics` | `g06_dynamics_check.py` per seed × `eval_loc` | check: `dynamics_check_<name>_<eval_loc>_s<seed>.json` + its per-pair dump `dynamics_pairs_<name>_<eval_loc>_s<seed>.parquet` (both report-only except the production seed's `history` pair, which verify T15 reads); the exit status of `model_seed` (G06_ENFORCE) is the acceptance |
| `downscale` | `g04_bayes_downscale.py` (model `model_seed`) | `bayes_models.pt`, `bayes_kmeans.pkl`, `bayes_eval.json`, the per-plot dump `g04_plots.parquet` (verify T11i) |
| `map` | `g05_map_stands.py` per `use_b` | `stand_veg_v3_B<b>.parquet` + `stand_cohorts_v3_B<b>.parquet`; the last `use_b` is promoted to `stand_veg_v3.parquet` / `stand_cohorts_v3.parquet`, plus `stand_nodes.npz` |
| `fe` | `24_export_fe.py` (area polygon `fe.area`) + `g07_fe_attach.py` | FE stands GeoJSON (`fe.geojson`, relative to the checkout) with GEN_* attributes + its `_cohorts.json`. The host serves it; restart / reload Forest Explorer on the host |
| `verify` | `gv_verify.py` (`GV_REQUIRE_MAP=1`) + `gv_verify_upstream.py` (with U8, the stand-mapping internals) | check: independent recomputation of every stage |

**Environment.**
- The training environment (`train.env`) is passed to the stages that load the model through g03b or check it: train, dynamics, downscale, map,
  vtab and verify (gv_verify). The table gets its own section, and each of dynamics / downscale / map adds its `env` section. The top-level `env`
  (region switches) goes to every stage.
- Region switches (`REGION`, `AEF90`) are accepted only in the top-level `env`; any switch in a section whose stages' scripts never read it is
  refused (e.g. `AEF90` under `table`, `G06_SET` under `downscale`). `CF16` is a train switch (`train.env`).
- Training switches (gv_verify's `T4Y_COMPARED`: the switches T4y compares with the checkpoint's `switches` record) are accepted only in
  `train.env` (G53.3): a `VAL_SPLIT` or `DZ` under `downscale.env`, `map.env`, `dynamics.env`, `table` or the top-level `env` is refused, since
  the stages that load the model must see the switches it was trained with. `NT` is a performance knob and stays allowed everywhere.
- Acceptance is file-granular (G53.6): a switch is accepted in a section when some script of its stages reads it, even if that script's reading
  lines do not run for that stage (e.g. `MAP_Y0` in `dynamics.env`: g06 execs only part of g04). Such a switch is still hashed, so it re-runs the
  stage; it cannot change a result.
- Config values are checked (G45.2, G53.5): the sections `name`, `data`, `stands_run`, `model_seed`, `table`, `train`, `dynamics`, `downscale`,
  `map`, `fe` and `verify` must be present (an empty config is refused); `name` and `train.model_prefix` are names of letters, digits, `_ . -`
  (they name files); `data` is a directory under `data/proto` (no `..`, not absolute); `verify.n` / `verify.n_upstream` are positive integers;
  `fe.area` / `fe.geojson` / `fe.tag` are non-empty strings; `dynamics.eval_loc` must be a non-empty list of names containing `history`,
  `map.use_b` a non-empty list of booleans (`true` / `false` or `1` / `0`, hashed as `1` / `0`), `train.seeds` a list of distinct integers and
  `model_seed` an integer among them. A null `env:` section means `{}`; an empty value (`MAP_Y0:`) or a list / mapping value is refused; a float
  is refused for a switch no script parses with `float()` (`EPOCHS: 14.0`, `SITE_ATTR: 1.0`), a boolean for a switch a script parses with
  `int()` as a count (`EPOCHS: true`; flags such as `VAL_SPLIT: true` stay allowed).
- YAML booleans are passed as `1` / `0` (as the scripts test them), and hashed in that form.
- The chosen model (`<model_prefix>_s<model_seed>.pt`) is also copied to `model.pt` before `downscale`, `map` and `verify`, for tools that read the default.
- Every switch a stage script reads (`STAGE_SWITCHES`) is stripped from the inherited shell environment; only the config sets it. The driver refuses
  to run if a stage script reads a switch missing from that list, and refuses a config env key that is not one of them (a typo). The reads are
  found in the scripts' syntax tree (`os.environ.get / setdefault / pop / [...]`, `in os.environ`, `os.getenv(name)` or `os.getenv(key=name)`,
  aliases of os.environ assigned to a name); writes (an `os.environ` entry assigned or deleted, `os.environ.update(NAME=...)`) are allowed. A read
  whose name is not a constant (or a loop over a literal) is refused, and so is every other reference to the environment (G53.5, fail closed):
  os.environ passed on, copied, iterated or matched, a bound method, `getattr(os, 'environ')`, `sys.modules['os']`, `__import__('os')`,
  `environb`, `expandvars`, a string exec'd that reads it. A stage starting a subprocess passes it the stage's environment, which the driver set.
- At start the driver also checks that the keys of g03b's checkpoint record `SWITCHES` equal the switches g03b reads (a new g03b switch must be
  recorded, so T4y can compare it; G53.5). `WORKSPACE` is made absolute (a relative one is taken from the driver's working directory), and the
  driver imports the stand pipeline's modules without any `STANDS_STAGE_JSON` of the shell.
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
- its code: the stage's scripts and the genveg modules `stand_runs.py` / `region.py`, whole and by bytes (a comment reword in g03b re-runs
  training: G53.6), and from the stand-pipeline modules `features.py` / `common.py` every module-level statement but def / class (importing runs
  them: `SOILS.append(...)`, `os.environ[...] = ...`) plus the top-level definitions its scripts import (and what those reference); for these two
  modules only, comments and docstrings do not count. This over-invalidates on purpose (G53.6): editing a module-level name genveg never uses
  (features' `TM_LOG`, an import there) still changes every stage's hash, a full rebuild; a featurizer method edit changes nothing.
  `stands_cfg.py` (imported by features.py; it loads config/pilot.yml at import — `24_export_fe.py` imports `CFG` from features, but uses it only
  when no stand gpkg is given, and the driver always passes one), `stands_stages.py` (the driver's `--set` parser), the driver itself and the
  Region 4 raster builders (`r4_grid.py`, `r4_aef90.py`, `r4_rasters.py`, whose outputs are fingerprinted) are in no stage's code hash, but vtab
  and verify byte-hash every scanned pipeline script through gv_verify's `VERIFY_CODE` (T0's verdict depends on all of them): an edit to any of
  them re-runs vtab and verify.
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
    LATER stage writes refuses too (that check could never be current); this is re-checked at the end of every run, on the files the run wrote,
    and for the gates of a stage's ups too (verify run alone sees vtab's list; D15-7). The list entries are made absolute in the stage's working
    directory (the checkout) and the evaluation has a timeout. Remote AlphaEarth COGs (T3c2z) cannot be fingerprinted. A change to a vup input
    makes vup stale and so blocks train (gated on vup) until vup has passed again; it does not re-run train, which does not hash its gates;
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
layer (`fe.geojson` + `_cohorts.json`, size and mtime), wherever it is. A failing gate does not make the stages that are already current stale
(G21): their status prints `current (gate FAILED: <gate>)`. But nothing new runs past a failed gate (G53.4, amending G21 / G45.12): a stage that
is not current is refused, and so is `--adopt`, when a gate of it or of any stage it depends on (transitively) FAILED its last run — `--force`
does not override this. Fix the cause and re-run the gate.
- Report-only outputs do not count for currency: an output counts only if a later stage or the product reads it. Report-only are `eval_*.json`,
  `dynamics_check_*.json` and `dynamics_pairs_*.parquet` except the production acceptance record `dynamics_check_<name>_history_s<model_seed>.json`
  and its dump `dynamics_pairs_<name>_history_s<model_seed>.parquet` (verify T15 re-derives the acceptance from them), `stand_veg_q_*.parquet`,
  the non-production `stand_veg_v3_B<b>` / `stand_cohorts_v3_B<b>` maps (every `use_b` but the last) and logs. Deleting one does not re-run
  anything; deleting the production acceptance record or dump re-runs dynamics and verify (not training).
- Files are attributed to a stage by a snapshot diff of the data dir. **No other process may write into the data dir during a run**: its files
  would be attributed to the running stage. Excluded: `pipeline_logs/`, the state and lock files, top-level `*.out` / `*.log`, the driver's own
  redirected stdout / stderr and its `--log` file (matched by device + inode) and dangling symlinks. Use `--log FILE` to keep the output: a shell
  `| tee` into the data dir under any other name is a file that grows during every stage, so the next run re-runs everything. The `--log` file is
  opened only once the data-dir lock is held (D15-5): a second driver refused by the lock does not write into the running driver's data dir. A
  refusal before the lock is written to the log only where no running driver can attribute it (outside `data/proto`, or a data dir whose lock is
  free); otherwise it is on the terminal only. A `--log` inside another data dir than the run's is refused.
- Only the inputs of the stages being run or checked (and their ups / gates) are resolved: a missing input of another stage does not refuse.
- One driver per data dir: a second driver on the same dir is refused (a lock on `<data>/.pipeline.lock`). Each stage runs in its own process group
  and inherits the lock, so the dir stays locked while a stage of a killed driver still runs; on SIGTERM / SIGINT the driver stops the stage's
  process group (SIGTERM, SIGKILL after 10 s), waits, then exits (status 128 + signal); the stopped stage is not current.
  Accepted (G53.6): a grandchild process (a subprocess a stage starts) does not inherit the lock fd, so after a SIGKILL of the driver the lock is
  free while it runs — no stage starts subprocesses (fork workers do inherit the fd); and a signal arriving in the microseconds between a stage's
  start and the driver's wait leaves that stage's process group running.

**Refusals** (before any stage runs or the state is touched):
- a frozen config — `frozen:` is checked on the raw file, before `--set`, so `--set frozen=` cannot bypass it;
- a config value of the wrong type or empty (see Environment), a `model_seed` not in `train.seeds`, `dynamics.eval_loc` without `history`;
- a config env key the driver owns, that is not a stage switch, a region switch outside the top-level `env`, or a switch no script of the
  section's stages reads;
- `--force` / `--adopt` naming an unknown stage or one not among the stages being run; `--adopt` of a check stage;
- a failing swallow-detector regression suite (`scripts/tests/test_check_comments.py`, run at start; a pass is remembered under
  `$XDG_CACHE_HOME/genveg_detector_suite/` by the content hash of the suite, the detector, every scanned script and the interpreter, so an
  unchanged tree is not re-tested; G53.2);
- code swallowed by a comment (`check_comments.py` over `check_comments.pipeline_scripts()`, the same list gv_verify T0 scans);
- a stage script reading a switch missing from `STAGE_SWITCHES`, or referring to the environment any other way; g03b's `SWITCHES` record
  differing from its reads; a training switch outside `train.env`.

A current stage is skipped; a stage whose `up` stages or gates are not current is refused. `--adopt` records the stage's files from its last
recorded run (the file list in the state), as they are now, as its current outputs without running it: the output fingerprint is recomputed, so
an externally rebuilt output re-runs everything downstream, and later changes make it stale as for a run. It is refused when any recorded output
is missing, when the stage has no recorded run (also an adoption recorded before r14), for the check stages vup / vtab / dynamics / verify (their
result is their exit status), unless its `up` stages and gates are current, past a failed gate (G53.4), and when a recorded file was last written
by another stage that is not a later one (G53.1: e.g. localisation re-run after history — that rewrite makes history stale, G43.5, and adopting
history would take over localisation's raw file; run history instead). Use it only for artifacts known to come from the same code and
configuration — never across a code change. The `gafl` table, for example, is NOT reproducible by `table`: it is g02b under the old matching rules
re-weighted by `g02i_reprior.py`, which is not a pipeline stage.

**check_comments.py** splits a comment at its start, after `;` / `:` / `)` / `]` (strong) and after every space or punctuation mark (weak), and
flags it when a remainder parses fully as Python and contains code: at a strong point any statement but a bare expression (assignments incl. alias
and walrus, raise / assert / continue / break / pass / return / global, blocks, any written call, a block header such as `if i % 2:`); after prose
words only code-dense fragments (dense assignments, keyword statements, method calls, calls with non-literal arguments). It also flags swallowed
elements of a multi-line literal, `, k=v)` closing a bracket the comment did not open, and (the broad tail scan, G53.2) every tail that, put back
in place of the comment, still parses and changes the statement: operators (`-1`, `**2` also without a space), commas, `.attr`, `[...]`, the
keywords and / or / not / in / is / if-else / for, and at the comment's start also `=`, `as` and a call `(...)` — at any bracket depth and across
the continued lines of a bracketed statement. Its regression suite is `scripts/tests/test_check_comments.py` (0 misses required; the driver runs it).
Write formulas in comments with `·`, `←`, `≔`, `÷`, words, or in backticks; write array shapes as `shape (n, m)`, not `[n, m]` or a bare
`(n, m)` after code; avoid ending a prose comment with "in X", "or X", "is X" right after code-like words (`# ... used in training` reads as
`x in training`: write "during training").

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
