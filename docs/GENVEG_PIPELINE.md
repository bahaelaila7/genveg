# Generative-vegetation pipeline — one place to run, document and check the model

Driver: `scripts/genveg_pipeline.py`. Configuration: `config/genveg.yml`, which holds every switch of every stage; `--set key=value` overrides one,
and unknown keys are refused. Method and audit history: `docs/GENERATIVE_VEG_PLAN.md`. The stands it maps come from a run of the stand-delineation
pipeline (`docs/STANDS_PIPELINE.md`), named by `stands_run`.

```bash
.venv/bin/python -I scripts/stands_pipeline.py                         # the stand run (gafl_pilot) the genveg config names
.venv/bin/python -I scripts/genveg_pipeline.py                         # every stage, in order
.venv/bin/python -I scripts/genveg_pipeline.py --set data=gafl_pipe --set fe.geojson=standmap/data/proto/gafl_pipe/baker_fl.geojson   # a separate run
.venv/bin/python -I scripts/genveg_pipeline.py map fe verify --force map  # re-run some stages
```

## Stages

| stage | script | output (data/proto/`<data>`/) |
|---|---|---|
| `table` | `g02b_build_gafl.py` (env `NCAND`, `TOPK`, `NPROC`) | `visits`, `tokens`, `pairs`, `plots`, `cands` .parquet, `cand_feat.npy`, `meta.json` — FIA visits, cohort tokens, footprint candidates in the 1.6 km fuzz disk (forest mask = the stand run's `land_clean`), location prior |
| `localisation` | `g02d_localisation.py` | `localisation.parquet` (keyed by FIA plot identity) |
| `records` | `g02e_fia_records.py` | `fia_events`, `fia_negs` .parquet, `visits.fia_blocked` |
| `history` | `g02g_history_consistency.py` | `cand_hist.parquet`; overwrites `localisation.parquet` with history-consistent sets |
| `stands_fp` | `g02j_stand_footprints.py` | `stands.parquet`, `stand_feat.npy` — footprints of the stand run's Baker County stands |
| `vup` | `gv_verify_upstream.py` | check: grid, forest-land rules (from the stand run's resolved config), stands, g05 nodes, FE units |
| `train` | `g03b_train_gafl.py` per seed (`train.env`) | `<model_prefix>_s<seed>.pt` |
| `dynamics` | `g06_dynamics_check.py` per seed × `eval_loc` | `dynamics_check.json` (tagged) |
| `downscale` | `g04_bayes_downscale.py` (model `model_seed`) | `bayes_models.pt`, `bayes_kmeans.pkl`, `bayes_eval.json` |
| `map` | `g05_map_stands.py` per `use_b` | `stand_veg_v3_B<b>.parquet`; the last one is `stand_veg_v3.parquet`, plus `stand_nodes.npz` |
| `fe` | `24_export_fe.py` + `g07_fe_attach.py` | FE stands GeoJSON (`fe.geojson`, relative to WORKSPACE) with GEN_* attributes. The host serves it; restart / reload Forest Explorer on the host |
| `verify` | `gv_verify.py` | check: independent recomputation of every stage |

**Environment.**
- The training environment (`train.env`) is passed to every stage after the table, because g04 / g05 / g06 / gv_verify load the model through g03b.
- The chosen model (`<model_prefix>_s<model_seed>.pt`) is also copied to `model.pt` before `downscale`, `map` and `verify`, for tools that read the default.
- `STANDS_RUN`, `DATA`, `WORKSPACE` and `FE_GEOJSON` are set by the driver.

**State.** `data/proto/<data>/pipeline_state.json` stores, per stage, a hash of:
- its config sections;
- the code of its scripts and the shared modules (`common`, `features`, `stands_cfg`, `stand_runs`);
- the upstream stage hashes;
- the stand run's stage hashes: `clean` for the table, `finalize` for stand footprints, map, FE and verify.

A current stage is skipped; a stage whose upstream is not current is refused. `--adopt` records existing outputs as current without running them. Use it
only for artifacts known to come from the same code and configuration. The `gafl` table, for example, is NOT reproducible by `table`: it is g02b under
the old matching rules re-weighted by `g02i_reprior.py`, which is not a pipeline stage.

## One-off / analysis scripts (not stages)

- `g02h_fix_visit_dates.py`: a migration, already applied.
- `g02i_reprior.py`: re-weights an existing table under new matching rules without a rebuild. It is not used by the pipeline, which rebuilds.
- `g02f_location_em.py`, `g08_sameyear_aef_test.py`, `g09_fia_landfire_association.py`, `g10_fast_viability.py`, `g11_raster_information.py`:
  analyses and viability tests.
- `g02c_fetch_visit_aef.py`: download of same-year AlphaEarth windows.

## Verification

- `vup` and `verify` are stages; their exit status must be 0.
- The stand run itself is verified by `scripts/stands_verify.py` (stage `verify` of the stand pipeline).
