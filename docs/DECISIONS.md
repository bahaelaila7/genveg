# Genveg decision log (v4 / Region 4 / v3b)

Every decision point, with the options considered. Since 2026-10-26 the agent takes the recommended option without asking (user: "go with your
recommendations, but document the decision point and the different options"). The audit history is in GENVEG_V4_R4.md §D–K.

## G1 — Curation sapling gap (2026-10-26, user)
- **Options:** fix the curation upstream; patch downstream; ignore.
- **Chosen:** **upstream fix** (FIA_DATA_PREP branch curation-fix). The v4 / r4 / v3b configs read the rebuilt `slim_r4_v2.duckdb`.

## G2 — Disturbed stands mapped as mature (2026-10-26, user)
- **Options:**
  - (a) **fix the dynamics, then map from 2000**;
  - (b) map from 2021 and accept the gap;
  - (c) patch the map with LANDFIRE.
- **Chosen:** (a). "Map from 2000" became a gated request (G5).

## G3 — v3 on the rebuilt table (2026-10-26, user)
- **Chosen:** **retrain v3 too** (config/genveg_v3b.yml).

## G4 — Runs during vetting (2026-10-26, user)
- **Chosen:** **halt every production run** until the audits of the component and everything upstream come back clean.

## G5 — Mapping start year (r4 B1, agent)
- **Evidence:** at MAP_Y0 = 2000, v3's held-out BA error got worse (0.449 vs 0.405), and a shuffled-landscape map correlated 0.87 with the real one.
- **Options:**
  - always 2000;
  - always 2021;
  - **a pre-registered gate**: 2000 only if its TPH and BA errors are ≤ those of the 2021 path and of the null, and the shuffled-landscape BA correlation is < 0.5.
- **Chosen:** the gate. Since r8 it is decided on the validation plots.

## G6 — Age clock (r4 I4, r5 I1 / M3, agent)
- **Evidence:** the LTC flow could not age stands (+0.1 yr/yr against +0.87 observed).
- **Options:**
  - (a) **a supervised clock dimension with a learned rate and a learned reset fraction**;
  - (b) a separate age head;
  - (c) feeding age as a covariate only.
- **Chosen:** (a).

## G7 — Dynamics acceptance (r4, r6 I2, agent)
- **Thresholds (pre-registered):** undisturbed BA log change within ±0.01 per yr of observed in every horizon bin, and a decoded/observed age-gain ratio of 0.7–1.3.
- **Set:**
  - Options: test plots; **a 10 % validation-block split with a 3.2 km buffer**; cross-validation.
  - Chosen: the validation split, enforced only for `model_seed`; it is never decided on the test plots.

## G8 — LANDFIRE severity labels (r7 I1, agent)
- **Options for 'No Severity':** Low (1, the old fallback); **neutral Medium (2)**; a separate "unknown" channel.
- **Chosen:** 2. Moderate equals Medium; the mixed labels map to 1.5 / 2.5; any unmapped label raises an error.

## G9 — Visit-year rule conflicts (r5 I3, r7 I2, r8 M5, agent)
- **Context:** when the start visit recorded an event of its own year (rs) AND a later visit recorded a new event dated in that year (ra).
- **Options:**
  - (a) ra wins and the year's evidence is applied, which re-applies the pre-visit cut;
  - (b) **rs wins**, which loses the later event's own-year evidence;
  - (c) per-class masking.
- **Chosen:** (b). It affects 0.2 % of pairs, and a double clear-cut is the worse error.

## G10 — Forest-share changes (r7 M1, r8 M2, agent)
- **Options:** drops only; **any |Δ| ≥ 0.25 blocks the undisturbed rules**; use a smaller threshold.
- **Chosen:** |Δ| ≥ 0.25. A rise is a re-delineation that also changes the per-forested-area densities.

## G11 — Audit finding rejected (r8 M4, agent)
- **Claim:** visits of 2023 were compared with AlphaEarth 2022 without being flowed.
- **Why rejected:** `ok = cst ≤ yr + 1` admits only visits up to the END of 2022, which flow forward. No change was made.

## G12 — Code swallowed by comments (r5–r8, agent)
- **Context:** five instances.
- **Options:** careful editing; **a static guard** (scripts/check_comments.py, in the driver before any stage and as gv_verify T0).
- **Chosen:** the guard.

## G13 — Legacy v3 config (r9 M-3, agent)
- **Context:** config/genveg.yml has no validation split, so the enforced acceptance (G7) cannot be decided for it.
- **Options:**
  - (a) add VAL_SPLIT, which changes the live v3 product's split and forces a retrain;
  - (b) **freeze it**: the driver refuses it, its outputs are kept, and v3b is the default config;
  - (c) exempt legacy configs from enforcement.
- **Chosen:** (b). (c) would reopen a test-set decision path.
- **To reverse:** remove `frozen:` from the config.

## G14 — Stocked-visit density check thresholds (r9 I-1, agent)
- **Options:** report only; **median TPH and BA ratio to FIADB within 1 ± 0.02, and < 2 % of visits below 0.9**; a per-state test.
- **Chosen:** the middle option, pre-registered before the rebuild. Expansion factors are identical by construction, so the residual should be about 0.

## G15 — Files written into the data dir during a stage (r9 I-2, agent)
- **Options:**
  - (a) declare each stage's outputs explicitly;
  - (b) **exclude the driver's own stdout / stderr targets**, and document that no other process may write there during a stage;
  - (c) exclude all `*.out` files.
- **Chosen:** (b). It is the smallest rule that is not ad hoc.

## G16 — gafl_pipe production map (r9 M-2, agent)
- **Options:** leave the B1 copy; **restore the 2026-10-23 B0 files**, which the live FE layer was built from; re-run map.
- **Chosen:** restore, with no re-run (runs are halted). It is reversible, because the _B1 files are kept.

## G17 — Where model choices are made (r9, agent)
- **Options:** the test block, as the old §A said; **validation**: g03b `validation`, g04 `mapping_path_validation_crossfit`, g06 acceptance.
- **Chosen:** validation. Test is reported only.

## G18 — The upstream checks gate the modelling chain (r10 M-1, agent)
- **Options:**
  - (a) leave vup ungated;
  - (b) **make train depend on vup** (so dynamics, downscale, map and fe follow);
  - (c) move fe after verify via a staging and publish step.
- **Chosen:** (b). fe stays before verify: the v4 / r4 / v3b configs write their FE layer to the data dir, not to the live layer, until verified (fe.geojson).
- **To reverse:** remove 'vup' from STAGES['train']['up'].

## G19 — Landscape-dependence part of the start-year gate (r10, agent; amends G5)
- **Evidence:** only the landscape descriptor is shuffled; the plot's own histories and soils are not. On a small model the 2021 path itself had cs 0.85, so the absolute cs < 0.5 can be unpassable for both paths.
- **Options:**
  - (a) keep cs < 0.5;
  - (b) **relative**: (1 − cs_requested) ≥ 0.5 × (1 − cs_2021);
  - (c) drop the criterion.
- **Chosen:** (b), pre-registered before any trained model. Both cs values are recorded in bayes_eval.json.

## G20 — LANDFIRE fiscal / calendar overlap (r10 M-2, agent)
- **Options:**
  - (a) sum per file (a pixel coded in two products counts 2/9);
  - (b) **count a pixel once per class and event year**;
  - (c) drop the fiscal products' previous-calendar-year codes.
- **Chosen:** (b). It keeps every pixel's evidence and counts it once. gv_verify T3e states the same rule.

## G21 — Gates vs hash dependencies (r11 I-1 / I-2 / M-3, agent; amends G18)
- **Options:**
  - (a) the check stages as hash dependencies of train (G18): a verifier setting or a new check retrains every seed;
  - (b) **a separate `gate` list**: the stages must be current (so they ran and passed) but do not enter train's hash;
  - (c) no gating.
- **Chosen:** (b).
  - train is gated on vup and on the new **vtab**: gv_verify before training, so the table-level checks incl. the pre-registered T1k / T1l fail before any seed is trained.
  - Of the verify section, vup hashes only `verify.n_upstream`; it also hashes its own code, its upstream (stands_fp), the stand-run pixels and its inputs' fingerprints.
  - The driver refuses a `model_seed` that is not in `train.seeds` (otherwise no acceptance is enforced and a stale checkpoint is mapped).

## G22 — Median of the mapping samples (r11 M-1, agent)
- **Options:**
  - (a) torch.median (the lower middle value), as before: g05's median of 16 was 2–5 % below g04's median of 32;
  - (b) the same sample count in both;
  - (c) **a shared true (interpolated) median in log space, `lmed`, in g04's mapping path and g05**.
- **Chosen:** (c). The same statistic on both paths; the sample counts may differ.

## G23 — Map checks must run after the map stage (r11 M-5, agent)
- **Chosen:** GV_REQUIRE_MAP=1 in the verify stage: missing map / FE / bayes outputs FAIL (T11z, U8) instead of being skipped. vtab runs with it off.
- **Also:** g05 records the calibration factors it applied (`map_cal_tph`, `map_cal_ba`), and T11c compares them with bayes_eval.json.

## G24 — Monte-Carlo noise in the gate's landscape criterion (r12 S-1, agent; amends G19)
- **Evidence:** the real and shuffled predictions use independent random draws. On an untrained model, a same-landscape redraw correlates only 0.974 / 0.963 (2021 / 2000 path), comparable to the criterion's scale.
- **Options:** (a) keep (1 − cs) vs (1 − cs_2021); (b) **dependence = corr(redraw) − corr(shuffled), with the requested path keeping ≥ 0.5 × the 2021 path's**; (c) common random numbers.
- **Chosen:** (b). It removes the noise floor without changing the samplers. gv_verify T11d re-derives the decision from the recorded numbers.

## G25 — Per-stage input fingerprints (r12 I-1, agent)
- **Options:** (a) one shared fingerprint, as before; (b) **each stage hashes only what it reads, and vector files by content**; (c) declared outputs only.
- **Chosen:** (b). A re-written but identical stand polygon file or county outline no longer rebuilds the table or retrains. check_comments.py only enters the check stages' code hashes.

## G26 — Known residuals, documented rather than changed (r12 M-2 / S-2, agent)
- **End-side visit-year rule:** an event recorded at a later visit, dated in a second-half v2's year, is applied before v2. 0.28 % of pairs; 0.05 % keep that evidence.
- **Validation cross-fit:** mapping_path_validation_crossfit reuses the halves that chose map_y0. Two options only, so the selection bias is small; v3 vs v4 should compare both paths' cross-fit errors.

## G27 — train does not depend on the stand footprints (r12 follow-up, agent)
- **Context:** g03b read stands.parquet only for an end-of-training diagnostic stand map (stand_veg_q, the direct Q predictor). Nothing downstream reads it.
- **Options:** (a) keep stands_fp upstream of train, so a stand-map change retrains every seed; (b) **make the diagnostic opt-in (STAND_DIAG=1) and drop the dependency**; (c) move it to its own stage.
- **Chosen:** (b).

## G28 — Generalisation beyond Region 4 (2026-10-27, agent; recorded for the expansion phase)
- **Generic:** the FIA record semantics, LANDFIRE / Hansen handling, the visit-year rule, splits / gating / driver / verifier, and the AEF90 path.
- **Must be re-derived per region:**
  - the EVK / SPEC evidence classes, windows and likelihood ratios: derived from GA+FL by g09; fire and harvest regimes differ;
  - the dynamics acceptance thresholds;
  - forested shares per size class: PNW / CA need the macroplot tier (16.6 % of live trees at TPA 0.999).

## G29 — Latent collapse onto the encoder bound (r13 B-1, agent)
- **Evidence:** SIGReg centres z, so the mean is unpenalised. On 4 trained v3 checkpoints, 11–12 of 16 dimensions sit at ±2.99 with sd 0.03–0.3, where tanh's gradient is about 0. The effective rank is 5. Undisturbed flows drift those dimensions off-manifold, and the decoded BA ratio falls to 0.60 at 25 yr. This plausibly explains r3 I1 and r4 B1.
- **Options:**
  - (a) **uncentred SIGReg** (test the projections against N(0, 1) including location);
  - (b) a softer bound;
  - (c) a VICReg-style variance term;
  - (d) leave it.
- **Chosen:** (a), plus pre-registered latent diagnostics reported by g03b and enforced in the g06 acceptance on validation encodings:
  - no non-clock dimension with |mean| > 2.0;
  - effective rank (participation ratio) ≥ D_Z / 4.
- **Switch:** SIGREG_MEAN=1. The new configs set it; older checkpoints stay loadable.

## G30 — Stand-replacing jumps (r13 B-2, agent)
- **Evidence:** jumps are tanh-bounded at ±1 per dimension per event-year, but stand-replacing pairs need a median max |Δz| of 4.48.
- **Options:**
  - (a) **reset-type jumps for every dimension**: z ← z + r ⊙ (z_post − z) + 0.5·tanh(shift), with r = sigmoid(·) per dimension and z_post = 3·tanh(W[ev, soil]), a learned post-disturbance state. This generalises the age-clock reset;
  - (b) an unbounded jump scale;
  - (c) leave it.
- **Chosen:** (a). It is a convex combination, so it stays bounded and can reach any post-disturbance state from anywhere. The r logits are initialised at −2.
- **Switch:** JUMP_RESET=1.

## G31 — LTC equilibria confined to a box (r13 B-3, agent)
- **Evidence:** zs = g·a / (1/τ + g) lies between 0 and a(soil). No stationary state is possible outside that box, so every state outside it drifts.
- **Options:** (a) **a state-dependent attractor**, Aa = A + Am([soil, z]) (bounded by 3·tanh); (b) leave the LTC form.
- **Chosen:** (a). The equilibria can then sit anywhere, and the exact per-year exponential step still holds with g and A frozen within the year.
- **Switch:** STATE_ATTR=1.
- **Also:** ltau is clamped to [−5, 10] (B-5).

## G32 — Mapped composition must be validated (r13 C-1, agent)
- **Context:** g04 validated the mean-share composition, while g05 maps one medoid draw with Bernoulli presence.
- **Options:** (a) **move the medoid selection into a shared function and validate it in g04** (Bray-Curtis and dominant-key accuracy, in the cross-fit and the test report); (b) map the mean composition instead.
- **Chosen:** (a). The cohort tables need one plausible stand.

## G33 — Start-year gate when the 2021 path has no landscape dependence (r13 C-3 / F-7, agent)
- **Options:** (a) **require dep0 > 0 as well; otherwise the requested year cannot demonstrate dependence and the gate falls back to 2021**; (b) an absolute floor.
- **Chosen:** (a).

## G34 — LANDFIRE "unknown" as cut evidence in Region 4 (r13 E-F11, agent)
- **Evidence:** the EVK / SPEC tables (where 'unknown' is accepted for T10–T40) were derived from GA+FL by g09. In KY / TN the 2023 'unknown' class is mostly Low severity and barely moves AlphaEarth (cos 0.965 vs 0.69–0.80 elsewhere).
- **Options:**
  - (a) **re-derive the evidence table for Region 4 (g09 per region) before the R4 table build; until it exists, g02b refuses AEF90 / R4 runs that use the GA+FL table**;
  - (b) drop 'unknown' everywhere;
  - (c) keep the GA+FL table.
- **Chosen:** (a). Recorded in the restart run order.

## G35 — Severity / collapse ratios over a span use only visits passing MIN_FSHARE (r13 F#5, agent)
- **Chosen:** an intermediate visit that fails the forest-share vetting does not enter the span's minimum BA ratio, because its per-forested-area BA is inflated or noisy.
- **Option rejected:** keep every visit.

## G36 — Evidence-coverage end (r13 A-1, agent)
- **Evidence:** Hansen GFC-2024 has lossyear 2001–2024. FY2025 LANDFIRE misses Oct–Dec 2025. The LANDFIRE harvest class is near-empty, so stand-replacing evidence is in practice Hansen. 2025 therefore carries no stand-replacing evidence; 2000 carries Hansen none either.
- **Options:**
  - (a) **treat 2025 as outside the evidence window**: negative-evidence periods and the neg / event-free / replacing rules are clipped to ≤ 2024; a span whose acting years reach 2025 cannot be "undisturbed"; documented for the map's end-2025 state;
  - (b) map the end-2024 state;
  - (c) acquire GFC v1.13 (lossyear 2025) when available.
- **Chosen:** (a) now, plus (c) as a data-update TODO after the restart: check availability and re-run the rasters. 2000 needs nothing, because the mapping path starts AT 2000.

## G37 — NONBLOCK DSTRBCD 41 and 60 (r13 A-2, agent)
- **Evidence:** the share of v2 visits with BA ratio < 0.3 is 3.6 % for code 41 (beaver, incl. flooding) and 2.8 % for 60 (vegetation). That is close to the blocking codes 52 (3.6 %) and 53 (2.5 %), and far from "no code" (0.6 %).
- **Chosen:** **both block**. NONBLOCK = {11, 21, 31, 40, 42, 43, 44, 45, 46}.
- **Option rejected:** keep them as NONBLOCK.

## G38 — Multi-horizon forest-share change (r13 A-3, agent)
- **Chosen:** for multi-horizon pairs, an end-to-end |Δ forested share| ≥ 0.25 also blocks the undisturbed rules. Before, only consecutive steps were compared.

## G39 — FIA lookback and initial installations (r13 A-4 / A-5, agent)
- **Lookback:** the first visit's lookback period is (yr0 − 4, yr0 − 1), conservative. It used to be yr0 − 5, which included months before "visit − 5 yr".
- **After a non-sampled visit:** a visit whose PREV_PLT_CN measurement was non-sampled (PLOT_STATUS_CD 3) uses the lookback rule, not the period since that unvisited measurement.

## G40 — Bare visits and all-bare plots (r13 A-6 / A-7, agent)
- **Bare test:** it counts live trees with TPA_UNADJ > 0 on forested conditions only.
- **All-bare plots:** FIADB plots that are bare at every visit are included, when their L3 ecoregion is in the curated table's scope. Recently cleared stands were under-represented by about 30–37 %.
- **Option rejected:** keep excluding them.

## G41 — Plots converted to non-forest before the 2022 mask (r13 A-9, agent)
- **Context:** a plot whose later FIADB visit is entirely non-forest likely has its true footprint outside the 2022 forest mask, so it has no candidate.
- **Chosen:** these plots are excluded from the MIL (AlphaEarth) loss. Their visits still train the decoder and pair losses.
- **Options rejected:** train them as now; drop them entirely.

## G42 — T10 as a hard constraint (r13 A-12, agent; deferred)
- **Evidence:** cuttings are mostly partial (median BA ratio 0.756), and the 'unknown' / Hansen base rate is about 15 % per 3-yr window.
- **Chosen:** keep T10 as a hard constraint for now; the g09 lift evidence supports it. Revisit in the per-region evidence re-derivation (G34).

## G43 — Round-13 driver decisions (implementer, 1c518b9)
1. **Swallowed-code detector — prose-shaped simple assignments** (`then w = 3`):
   - Options: (a) flag them, which also flags prose such as "invert year = year + …"; (b) **accept the miss**, since real swallows are code-dense.
   - Chosen: (b). All 5 historical instances are caught.
2. **Formulas in comments:** bare `f(x)` and expression-only fragments after prose are not flagged. Use `·`, `←` or backticks in formulas, so they no longer parse as code.
3. **Which outputs count for currency:** only those read by a later stage or the product.
   - The production acceptance record `dynamics_check_<name>_history_s<model_seed>.json` is kept as a real output; verify has dynamics as an up stage, because T15 reads it.
   - Non-production `_B<b>` maps and cohort tables are report-only.
   - Option rejected: a blanket report-only rule.
4. **Dangling symlinks** in D are skipped with a note. Any other stat error refuses, before `state.pop`.
5. **A rewrite by an earlier stage makes the later writer stale** (e.g. a localisation re-run re-runs history and everything after it). This is inherent to mtime-based output fingerprints.
6. **Imported names of features.py / common.py** are derived by parsing the scripts; a missing name refuses. Option rejected: hard-coding them.
7. **Missing required inputs** (table rasters, R4 rasters, DBs, vectors) refuse. The vup, TIGER and data-dir extras record `missing`, and the stage itself fails.
8. **fe** depends on the stand gpkg only (the stands.tif dependency was dropped). It tracks its geojson and `_cohorts.json` (WORKSPACE-relative).
- **Beyond the brief:**
  - the driver refuses any stage script that reads an env switch not in STAGE_SWITCHES;
  - config env keys that are not switches are refused;
  - train fingerprints cand_loc_aef / cand_prior under LOC aef / plot;
  - stages always run in pipeline order.

## G16 correction (2026-10-27, from the r13 verifier work)
- The live Forest Explorer layer (fe_pan/.../baker_fl.geojson, tag "v3 (pipeline)") matches NO stand_veg file in gafl / gafl_pipe (median |ΔBA| 14.6 ft²/ac against gafl_pipe B0). G16's "the live layer was built from B0" does not hold for the current files.
- The restored B0 files also predate the cohort scaling and the QMD fix.
- The live layer's provenance is unknown. It will be replaced by the rebuilt pipeline's fe output after the restart; the user restarts FE on the host.

## G44 — Verifier-side decisions (r13, implementer d8294ea)
- Check names: the old T12 became T4r, fs_ok is T4v, and T12 is g04 set hygiene.
- E-F7: independent visits carry the table's visit ids, matched by plot + date, so one T1a mismatch does not cascade.
- T5s: a per-dimension rule plus an aggregate χ² rule (the per-dimension rule alone had no power).
- T3i is per state (cells n ≥ 30).
- T3c2z covers UTM zones 15–18 through the remote AlphaEarth COGs. It needs the network; a fetch failure FAILs, with no bypass.
- T15 re-derives the G7 + G29 acceptance and requires it to pass.
- **Real data defects found:** T1m — 337 / 9,034 (3.7 %) FIADB live-tree plots in the curated L3 scope are absent from the curated product. This goes to the curation loop.
- **Stale artifacts found:** T6c — stand_feat.npy predates the G8 severity labels, so stands_fp must re-run.

## G45 — Round-14 driver decisions (r14 D / F slices, coordinator)
Options weighed per item; the chosen one is first.
1. **`--adopt` (D-1, D-2).** Adopt snapshots the stage's own files (its last recorded file list; refuse if any is missing), recomputes `out` from them and stores `files`, so later changes to them invalidate as for a run. Check stages (vup, vtab, verify, dynamics) cannot be adopted: their result is their exit status, and adopting would bypass G21. Rejected: keeping adopt as a pure state copy (it made adopted stages current forever).
2. **List-typed config values (D-3, D-14).** `dynamics.eval_loc` and `use_b` must be non-empty YAML lists; eval_loc members are strings and must contain 'history'; use_b members are booleans. A null `env:` means {}; an empty value (`MAP_Y0:`) is refused, not passed as 'None'; integer switches given as floats are refused; `model_seed` gets a clear message. Rejected: coercing a scalar into a one-element list (it hides the slip).
3. **Module-level statements of imported libraries (D-4).** `defs_hash` hashes every top-level statement of features.py / common.py that is not a def/class (assignments, calls, augmented assignments, subscript stores, `os.environ[...] = ...`), not only Name stores. Docstrings are stripped before hashing (H1), consistent with "comments do not count". Rejected: hashing the whole module (re-runs everything on unrelated edits).
4. **Lock vs children (D-5).** Stage children run in their own process group and inherit the lock fd (`pass_fds`), so the data-dir lock is held while any child lives, even after SIGKILL of the driver; on SIGTERM / SIGINT the driver kills the group, waits, then releases. Rejected: a pid file (stale-pid races).
5. **Swallow detector (D-6, F-1).** Close every reported miss class, keeping the heuristic design: strong statements (raise, assert, continue, break, global, nonlocal, return, pass-free bare calls, alias / walrus assignments); bare calls with non-literal arguments at weak points; splits after `—`, `→`, `--`, `%`, quotes, `-`, `?`; swallowed elements of multi-line literals (a comment inside a bracketed literal whose suffix parses as one or more elements); and statement tails (a suffix starting with a binary operator / subscript / attribute that, appended to the code on the line, still parses). Flagged prose is reworded (the existing policy). The detector's own regression suite (all r13 / r14 crafted cases + historical swallows) must pass with 0 misses and the tree with 0 unreworded hits. Rejected: banning inline comments in pipeline scripts (an all-file reformat; own-line comments can still swallow), and an AST-of-comments comparison (comments are not in the AST).
6. **Hash caches (D-7, D-18).** pixhash / vechash caches are written tmp + fsync + replace; an unreadable cache is recomputed, never fatal. pixel_hash covers all bands, CRS and nodata.
7. **Verifier inputs single source (D-9, F-7, E14-3).** The driver evaluates `gv_verify._verify_inputs()` / `gv_verify_upstream` VERIFY_INPUTS in a subprocess under each check stage's env and fingerprints exactly that list (plus its own code inputs); its parallel hand list for vtab / verify / vup is removed. The verifier owns the list's contents (aef_index.parquet added, over-listed files removed: verifier implementer). Remote AlphaEarth COGs cannot be fingerprinted; T3c2z needs the network (G44) — accepted.
8. **Dynamics acceptance record (D-10, F-8, E14-4).** The driver passes the exact `dynamics_check_<name>_history_s<seed>.json` to verify as GV_DYN_JSON; GV_DYN_JSON becomes driver-owned (DRIVER_SET), not config-settable. `name` enters the hash of every stage whose outputs or TAG it sets.
9. **Region switches (D-13).** REGION / AEF90 / CF16-type region switches are accepted only in the top-level env; any switch placed in a section whose stage scripts never read it is refused. CF16 moves to `train.env` in genveg_r4.yml (D-8).
10. **Small items.** stands_stages.py and stands_cfg.py are scanned (D-15); ENV_READ becomes AST-based (setdefault / pop / subscript / get with constant names; a non-constant name read is refused) and the five unread switches are removed from STAGE_SWITCHES (D-16); inputs are resolved only for the stages being run or checked and their ups (D-17); `snap()` prefix check uses a path-component compare (D-19); fe.geojson paths in configs are relative to the checkout, not WORKSPACE (D-12); docs precision items (D-20).
11. **Tee (D-11).** A new `--log FILE` option tees the driver's output to a file excluded by inode; GENVEG_PIPELINE.md tells users to use it instead of a shell tee into the data dir. Rejected: detecting a file that changes in every stage window (heuristic, misattributes real outputs).
12. **Library env (D-16).** PROJ_* / GDAL_* / thread counts / CUDA_VISIBLE_DEVICES stay unhashed performance-or-environment knobs (like NT); documented. A failing gate leaving downstream "current" (D-21) stays by design (G21); the status print marks such stages "current (gate FAILED)".

## G46 — Round-14 model / data / evaluation decisions (r14 A / B / C / F slices, coordinator)
Chosen option first; rejected options after "Rejected".
**Data (g02b / g02e / g02g)**
1. **Incomplete curated visits (A14-1, IMPORTANT).** g02b vets every curated visit against FIADB: n_cur = curated live tallied trees, n_fia = FIADB live tallied trees on forested conditions (`FV.n`). A visit with n_fia ≥ 3 and n_cur < 0.5·n_fia is *incomplete*: it is dropped from the table (no tokens, not a pair endpoint, not a bare visit); it stays a FIADB visit for records and negative evidence. Counts go to meta.json. The defect itself goes to the curation loop (the new curated_v2 must be checked the same way). Rejected: flag-only (the decoder would still train on false-bare targets); filling the visit from FIADB (bypasses curation).
2. **Share change against older / non-forest predecessors (A14-2).** The |Δ forest share| ≥ 0.25 block also compares a first post-2000 visit with its pre-2000 FIADB predecessor, and treats an entirely non-forest predecessor (PLOT_STATUS 2) as share 0.
3. **G41 converted plots, refined (F3 / A14-3).** A plot is converted iff its last FIADB visit dated ≤ 2022 (the mask year) is non-forest — non-forest land or water — and an earlier visit was forested. Plots forested again by 2022 are not converted; non-forest only after 2022 is not converted. g03b refuses a table without `plot_converted` (F9; no VCONV = 0 fallback).
4. **Evidence table per region (A14-5, E14-2).** EVK becomes `EVK_TABLES` keyed by the sorted region set (GA+FL = today's EVK). g02b refuses a region with no entry; this replaces the EVK_REGION_OK switch, which is removed. g02b writes the resolved EVK and its hash to meta.json, and g02e / g02g / g03b read EVK from meta.json, not from g02b's module. REGION is compared as a sorted set; AEF90 with GA+FL is refused.
5. **G36 becomes a clip (A14-6).** A pair whose acting span passes EVID_END keeps its negative / event-free constraints over the part ≤ EVID_END, and the part after it is unconstrained, matching the location prior and g02g. This replaces the block, and `cand_hist` follows the same rule.
6. **Provenance (E14 triage).** g02b, g02e, g02g and g02j record a code hash (the script's AST hash) in meta.json, so the verifier can label a failure stale vs divergent.

**Model (g03b)**
7. **Converted plots out of every AlphaEarth term (F2 / B14-2 / B14-3 / A14-4 / B14-8).** They are excluded from MIL, Q, Q-decode, g04's B likelihood (trB), and from the candidate-history-derived pair constraints. If the pair code has no FIA-only path, their pairs are dropped and the count is reported. Rejected: MIL-only (G41's rationale, mislocated footprints, applies to every term reading AlphaEarth candidates).
8. **Latent acceptance replaced (B14-1, IMPORTANT; supersedes G29's rank part).** latent_ok requires, on validation over the non-clock dims, all three of: max |mean| ≤ 2; min per-dim sd ≥ 0.3; share of values with |z| > 2.9 ≤ 1 %. Effective rank stays as a report only. The implementer checks this against the existing evidence: all 22 collapsed v3 checkpoints and B's OFF run must fail, and B's ON run must pass. If any does not, report rather than retune. Rejected: keeping the rank ≥ dims/4 rule (it is non-discriminative: the collapsed OFF run scores 6.10, the healthy ON run 5.86).
9. **Flow contraction and age lag (B14-4, IMPORTANT).** Two new switches, on in the production configs:
   - SIGREG_FLOW: the SIGReg term is also applied to the flowed states at the pair end, so the flowed population stays on the encoder's distribution.
   - AGE_FLOW: the age loss is also applied to the decoded age of the flowed state at v2.
   Acceptance evidence: B's 1200-step ON harness (audit_r14/B_model) re-run with the switches. NN ratio, latent-variance retention over 25 yr, decoded age rate and composition Bray must improve, and nothing in B's table may regress. Otherwise report. Rejected: waiting for full training (g06 would fail late, after hours of compute).
10. **Bounded resets (F10 / B14-6).** JUMP_RESET target zt = clamp(zp + 0.5·tanh(shift), −3, 3); the clock component of zt is ≥ the age-0 value. The jump is then a strict convex combination and is bounded. EncV4.cls is initialised N(0, 0.02) (B14-5). Old checkpoints load unchanged.

**Evaluation (g04 / g05 / g06)**
11. **Gate and calibration replicated (C1, IMPORTANT).** R = 8 replicate draws. The gate statistics (dep0, dep, errors) and CALF are the replicate means; per-replicate values and sd are recorded. The gate shuffle is a derangement, fresh per replicate (C13).
12. **Mapped-estimator acceptance (C2, IMPORTANT).** Null and kNN baselines are computed on validation too. The map is accepted only if, on validation, the chosen path's calibrated median |log err| for BA and TPH and its medoid composition Bray are each below the null's. kNN is reported, not gated. g04 writes `map_accept`; g05 refuses to map unless it is true. Pre-registered here, before any production run.
13. **One estimator (C3 / C8 / F5).** g04 and g05 share the draw count (MS = 32 iid) and one `calibrated` helper (e^CF · lmed(s), the validated form).
14. **Visit-date vs end-2025 (C4).** Accepted as a documented residual. g04 reports the median shift between the visit-date and end-2025 flows on validation.
15. **True medians everywhere (C7)**: CF, err / bias, predA / CAL, AxB, the null, g03b metrics, g06 reconstruction.
16. **Undisturbed set (C5 / F4).** g06 uses g03b's pair_set "undisturbed" rule (G36 clip, G38, the collapse block) through the shared exec'd code.
17. **Required horizon bins (C13 / E14-5c).** g06 acceptance fails if any pre-registered horizon bin has n < 20 on validation. The bins are fixed from table counts (not model results) before training. g06 writes a per-pair dump (pair ids, horizon, predicted / observed values) so the verifier can recompute the bins.
18. **Counters (C10 / F11)** are kept per query set; a NaN gate statistic is an explicit FAIL with its reason.
19. **Report-only diagnostics (C12):** a bare-visit metric for the mapping path, and an age-class Wasserstein of the mapped cohorts vs observed, on validation.
20. **Accepted, documented:**
    - transductive unsupervised statistics (C9: no labels);
    - the USE_B = 1 calibration mismatch (C11: non-production, documented in g05);
    - the 2026-visit clamp (A14-8: negligible; a verifier test is added).
    - G37's region dependence (A14-7) is noted: the decision holds in both regions.
21. **Stale text (F12 / C15 / F6 / B14-7):** fixed with the code; G29's rank text is superseded by item 8.

## G47 — Round-14 verifier decisions (r14 E / F slices, coordinator)
1. **pair_spec records (E14-1, IMPORTANT).** The spec takes every FIA record with y1 < ry ≤ y2 from the verifier's own FIADB visits, including non-table visits. T8, T8f, T8c and T8h print the mismatching pairs.
2. **Per-region spec (E14-2, IMPORTANT).** The verifier keeps its own independently written EVK per region set and FAILs (not SKIPs) for a region it has none for. T3i re-derives the evidence for every row and every class of that table, per state. It also checks that the verifier's spec equals the table's meta.json EVK (any divergence FAILs).
3. **Inputs list (E14-3):** VERIFY_INPUTS gets aef_index.parquet and drops what is not read. The driver consumes it (G45.7). VERIFY_CODE is removed or used.
4. **T15 (E14-4 / E14-5):**
   - reads only GV_DYN_JSON (FAIL if unset or missing);
   - requires the `latent` key;
   - applies the G46.8 criteria;
   - recomputes the bins and medians from g06's per-pair dump (G46.17) and requires every pre-registered bin;
   - checks `set == 'val'` against the split.
5. **T12 (E14-6):** every anchored name must have exactly one assignment in g04 (else FAIL). The regex parts are replaced by behavioural tests. The halves test rejects sorted and first-half splits too.
6. **Configured switches (E14-7):** T8h / T8b read PAIR_ANCHOR / SEVERITY_ANCHOR and FAIL on an unsupported value. Their tautological size conditions are replaced by real minima.
7. **T5s power (E14-8):** it subsamples only plots with non-constant incl_w. The per-dim rule stays report-only; the aggregate χ² decides.
8. **Coverage (E14-9 / E14-10 and the coverage gaps):**
   - T3c2 covers AEF90 2022.
   - gv_verify_upstream gets a pre-registration table, and U8 compares every XL block (AEF, disturbance, Hansen, soil).
   - T0 scans the driver's `pipeline_scripts()` list (single source).
   - T13 / T14 print N/A (with reason) in R4, not SKIP.
   - New checks:
     - g05 calls the same estimator with the same MS as g04;
     - g04's null baselines and CAL are recomputed from per-plot dumps;
     - the checkpoint's training switches equal the config's;
     - the G41 exclusions (all terms, G46.7) are unit-tested;
     - the R4 derived rasters (dist, hansen_*, soils90) are compared with an independent reprojection on sampled windows;
     - the 2026 clamp;
     - the A14-1 vetting is re-derived independently (no table visit with n_cur < 0.5·n_fia);
     - G46.2, G46.3 and G46.5 are mirrored in the spec.
9. **Stale vs divergent:** every failing check prints whether the table's meta.json code hash equals the current script's hash (G46.6).

## G48 — Round-14 verifier follow-ups (verifier implementer ec428f8..dc1e11f, coordinator)
1. **T3i judging rule (set after seeing data — recorded as such).** Per state, each detectable EVK row is judged on the POOLED cells of all its compatible classes (the row's claim is "this kind is detectable", not "each compatible class is"); every class is also reported on its own. Options: (a) pooled row + per-class report — chosen; (b) the implementer's interim rule, judging only the classes every detectable row accepts (harvest, unknown, Hansen in GA+FL) — rejected, it silently drops a row whose evidence is only in its own classes; (c) judge every class — rejected, a compatible-but-weak class (GA+FL prescribed 'fire' never lowers AlphaEarth) is not a claim of the table. **Follow-up for round 15 (data slice):** check whether LANDFIRE fire classes in a detectable row's evidence dilute that row's likelihood ratio in GA+FL (prescribed burns), i.e. whether the row's lr 4 still holds with them in.
2. **G46.3 wording.** "Last FIADB visit dated ≤ 2022" means the last *sampled* visit (any COND_STATUS 1–4), as g02e implements: a non-sampled visit carries no forest status. The verifier mirrors "sampled".
3. **U10r tolerance:** treecover2000 within 1 vs the GA+FL product (GDAL `average` rounding, 5.5 % of pixels at ±1) — accepted, documented in the pre-registration table.
4. **Stale/divergent label limit:** a table with no recorded code hash labels every failure "stale". Accepted: every production table is rebuilt with provenance before its verify run, so the limit only affects pre-r14 tables.
5. **Missing interfaces assigned to the model side:** g03b saves the training switches into the checkpoint (`switches` dict, every STAGE_SWITCHES name g03b reads) for T4y; g04 writes per-plot dumps sufficient to recompute the null baselines and CAL (T11i); the verifier then lists them in G04_DUMPS.

## G49 — SIGREG_FLOW / AGE_FLOW outcome and the age-rate lag (r14 model implementer A/B, c048e50; coordinator)
The pre-registered G46.9 acceptance was only partly met on B's 1200-step subset harness (docs/GENVEG_V4_R4.md §Q.2):
- **Improved:** 25-yr variance retained 10 → 32 %; g06 BA growth +0.000 / −0.002 / −0.002 → +0.009 / +0.007 / +0.007 per yr (observed +0.022 / +0.014 / +0.013); flowed-v2 age bias +2.3 → +0.2 yr; 25-yr composition Bray 0.375 → 0.281.
- **Not improved:** the decoded age *rate* (0.39–0.66 → 0.39–0.59 of observed); pair-level composition Bray is flat.
- **Small regressions:** synthetic fire ratio 0.92 → 0.82; undisturbed-like NN ratio 1.36 → 1.50, which now matches the encoded baseline 1.47.

Options:
- (a) keep both switches on and add a change-based age loss for the rate;
- (b) turn both off, since the criterion was not met;
- (c) keep SIGREG_FLOW only.

**Chosen: (a).** Contraction and the BA-growth deficit are the defects g06 judges, and both moved strongly in the right direction. The regressions are small, and one of them is a move to the encoded baseline. The real gate remains g06's pre-registered validation acceptance on the production run, which fails or passes independently of this choice. The partial miss is recorded here rather than re-defined as a pass.

**New switch AGE_DELTA.** W_AGE · ((dec_age(flowed z2) − dec_age(z1)) − (obs_age(v2) − obs_age(v1)))² / AGE_S², posterior-weighted, on pairs with both visits stocked and undisturbed (UNDIST). It targets the rate directly; AGE_FLOW targets the level.
- Acceptance (pre-registered): on the same harness, flow + AGE_DELTA vs flow:
  - the decoded age ratio rises in every bin by ≥ 0.1 or reaches ≥ 0.8;
  - no other row of the §Q.2 table regresses by more than the flow-vs-ctl spread.
- If this fails, AGE_DELTA stays off and the age-rate lag is logged as a known residual for g06 to judge.

## G50 — AGE_DELTA stays off (G49 pre-registered criterion failed; model implementer 421d91e, coordinator)
- **What passed:** the age-rate part passed in every bin. The decoded age ratio went from 0.39 / 0.45 / 0.59 to 0.69 / 0.69 / 0.84.
- **What regressed** beyond the flow-vs-ctl spread:
  - latent min sd 0.78 → 0.47;
  - closest reachable RMS 0.39 → 0.47;
  - 25-yr effective rank 3.21 → 2.34;
  - flowed-v2 age bias +0.2 → +3.1 yr (it trades AGE_FLOW's level fit for the rate).
- **Decision:** under G49 as written, this fails, so AGE_DELTA stays 0 in every config.
- **Rejected:** re-judging with a second seed after seeing the result. The spreads are noisy single-run-pair estimates, but widening them post hoc would be moving the goalposts.
- **Known residual:** decoded age advances at about 0.4–0.6× the observed rate. g06's pre-registered age_ok on the production run decides whether this blocks the map.
- **Switch handling:** g03b reads AGE_DELTA. So it is listed in STAGE_SWITCHES (unset = off), recorded in the checkpoint `switches`, and compared by T4y. A config that turns it on is then hashed and visible.

## G51 — Round-15 model / data / evaluation decisions (r15 A / B / C / F slices, coordinator)
Reports: scratchpad/audit_r15/rep_{A,B,C,F}.md. The chosen option comes first; rejected options follow "Rejected".

**Data**
1. **TPA-weighted vetting (A15-1, IMPORTANT).**
   - A visit is incomplete if n_fia ≥ 3 and n_cur < 0.5·n_fia (G46.1), OR live TPA_cur < 0.5·TPA_fia, OR n_cur = 0 while n_fia ≥ 1 (A15-3: a curated visit is bare only when FIADB is bare too, the same definition as G40).
   - meta.json records the counts by rule, plus the curated-DB and FIADB paths and their file fingerprints (A15-6c).
   - The curation round-7 audit measured the new curated product at TPA share ≥ 0.993 in every state, so this mostly confirms the fix. It still guards against a sapling regression.
   - Rejected: count-only (blind to saplings).
2. **Raw keys (curation r7 A7-9 / D-F2, IMPORTANT).**
   - g02b joins curated trees to FIADB on the curated canonical key (via `unitcd_raw` / `countycd_raw` / `plot_raw` per record).
   - Coordinates follow the curated D28 location (latest visit).
   - FIA events and negatives cover every chain segment of a canonical plot.
3. **Float ties (A15-2).** Every |Δshare| ≥ 0.25 comparison (g02b, g03b G38, verifier) uses ≥ 0.25 − 1e-9.
4. **Single source of evid_end (A15-4).** g02e takes evid_end from meta.json and passes it into fia_records.
5. **Provenance completed (A15-6, E15-7).**
   - code_hash covers each script plus the local modules it imports (region, common, features, stand_runs).
   - It is computed at the START of the run.
   - It is version-stable: a token-based serialisation with comments and docstrings dropped, giving identical results under Python 3.12 and 3.13.
   - g02b writes a `build_id` (uuid) into meta.json. Every downstream output records the build_id it read: g02e / g02g / g02j parquet provenance, the checkpoint, bayes_eval.json and the g06 record.
   - The parquet evk_hash is checked.
   - Rejected: code-only hashes (cannot tell a stale table instance from a divergence).
6. **EVK consumers outside the contract (A15-7).**
   - g02f uses the meta EVK and region_check.
   - g02i_reprior (legacy) refuses to run.
   - **g09 becomes the per-region derivation tool.** It uses common.fiadb(), handles any region, and reproduces EVK_GAFL on GA+FL (including 'unknown') as its self-test.
   - **Bootstrapping a new region.** g02b accepts `EVK_PROVISIONAL=1` (table env, driver STAGE_SWITCHES), which builds with the GA+FL table and marks meta `evk_provisional: true`. g03b and every training/eval stage refuse a provisional table. g09 then writes a proposed EVK_TABLES entry for the coordinator to commit.
7. **Separate evidence weight from detectability (A15 G48.1 follow-up).**
   - Measured inner-zone lifts of the detectable rows are 1.7–2.6, against the coded lr 4.
   - Each EVK row gets two fields:
     - `lr`, the evidence weight, measured by g09 on the region (GA+FL: re-derived now, with the method recorded in meta);
     - `detectable`, the hard-constraint flag, unchanged: T10–T40 and Dcrown.
   - Cluster and hard-constraint semantics are unchanged; only the prior / likelihood weights become calibrated.
   - Rejected: keeping lr 4 (overconfident location prior); and deriving detectability from the measured lr (it would change the constraint set right before the rebuild).
8. **UNDIST on converted plots (A15-8).** UNDIST is set after the VCONV return, so converted plots are never UNDIST.

**Model (g03b)**
9. **Decoder age compression (B15-1, IMPORTANT).**
   - New switch DEC_AGE: a direct decoded-age loss on ENCODINGS (dec_age(enc(v)) vs the observed age, posterior-free, stocked visits). The decoder learns to read the clock.
   - Pre-registered acceptance on the B harness (1200 steps), flow + fixes + DEC_AGE vs flow + fixes, reading:
     - the oracle decoded-age rate (decoded enc(v2) − enc(v1)) ≥ 0.8 × observed in every bin;
     - the decoded-vs-observed age slope on validation encodings ≥ 0.95;
     - no other §Q.2 row regressing beyond the flow-vs-ctl spread.
   - If it fails, DEC_AGE stays off and the residual stays as recorded.
10. **Clock target (B15-2).**
    - The clock component of the reset target becomes zt0 = z_age0 + σ(shift)·(z0 + 0.25 − z_age0), a convex blend between age 0 and the current clock + 0.25. A reset can rejuvenate, keep, or slightly raise the age.
    - The gradient is non-saturating.
    - G46.10's text is amended to the actual scales (0.25 on the clock).
    - It is structural, so it is not A/B-gated, but its effect is reported in the same A/B. Partial cuts must show a clock rate ≥ 0.8 yr/yr.
11. **Latent acceptance, extended (B15-3).** Pre-registered now, before any production run. latent_ok additionally requires:
    - effective rank ≥ 3, a floor that catches rank-1 / rank-2 collapse; the healthy runs give 5.9–6.8;
    - share of |z| > 2.5 ≤ 5 % (bimodal saturation);
    - the same G46.8 criteria on the 25-yr no-event flow of the validation encodings (flowed states must stay on the distribution).

    The implementer measures every criterion on the existing checkpoints (B ON, ctl, flow, delta, the 20 distinct v3 models) and REPORTS the numbers. If a criterion would fail on every candidate, it is reported, not retuned.
12. **SIGREG_FLOW routing (B15-4).** z1 is detached in SIGREG_FLOW, so it regularises the flow only, as G46.9 intended. AGE_FLOW is unchanged. The effect is reported in the A/B.
13. **Switch consistency at load (B15-5 / C15-7).** g04, g05 and g06 compare their env's T4Y_COMPARED switches with the checkpoint `switches` and refuse a mismatch. bayes_eval.json records MODEL and the checkpoint's sha256, and g05 checks both.
14. **g06 (B15-6 / B15-7 / C15-9).**
    - The synthetic section uses validation plots only.
    - The BA tolerance (G7) is unchanged, since it is pre-registered. g06 adds report-only plot-cluster bootstrap CIs and per-bin plot counts.

**Evaluation (g04 / g05)**
15. **map_accept superiority (C15-1, IMPORTANT).** Pre-registered now, before any production map.
    - Accept only if, for each of BA, TPH and medoid Bray, the one-sided 90 % upper bound of (model − null) is < 0.
    - The bound comes from a plot-cluster bootstrap within the judged halves, B = 1000, seeded, with the factors held fixed.
    - A model that cannot be shown better than the null is not mapped.
    - Rejected: the point comparison (it flipped by seed: accepted in 5 of 12 runs on the subset).
16. **Medoid draws replicated (C15-2).** The presence uniforms are seeded by (visit, replicate, path, half). g05 uses its own generator for presence (C15-12).
17. **CAL on stocked visits (C15-3)**, as fit_cf does.
18. **Gate dump (C15-4).** g04_plots.parquet also carries the shuffled and redraw predictions, so dep / dep0 can be recomputed.
19. **Selection vs acceptance (C15-5).** map_y0 is selected on replicate set 1 (R = 8); map_accept is judged on an independent replicate set 2 (R = 8, fresh seeds). The plots are shared; this is documented.
20. **Robustness (C15-6 / C15-8 / F10):**
    - g05 checks that map_accept was judged on 'validation plots';
    - dump set labels are conditional;
    - an empty or all-bare judged half is an explicit FAIL;
    - map_accept no longer requires test visits (test is report-only);
    - the k-means sample uses min(n, 200 k).
21. **Converted plots in the MDN and the eval sets (C15-11).** Excluded from trA (the landscape descriptor reads the mislocated footprint) and from the validation and test scoring sets, consistent with G46.7.
22. **Split invariance (C15-13).** Blocks are sorted before the seeded choice.
23. **Small items:**
    - the JSON records sd of the mean alongside the per-replicate sd;
    - stale prose fixed (C15-14);
    - doc fixes (B15-8, F3, F4: G46.2's text widened to "every visit vs its PREV_PLT_CN measurement");
    - the r14 probe's 24.5-yr divisor noted in §Q.2 (B15-9).
24. **Upstream product (E15-4):** r4_aef90 takes each 90 m cell from the tile whose UTM zone contains the cell centre (zone from longitude), deterministically. The R4 product must be rebuilt (after the restart).

## G52 — Round-15 verifier decisions (r15 E / F slices, coordinator)
1. **Self-calibration and leakage (E15-1).**
   - T12 adds an AST scan: any assignment to an anchored name or to HALVES anywhere in g04 outside its anchored line FAILs. HALVES must be the literal ((hA, hB), (hB, hA)) form.
   - T11i adds set relations:
     - each cf_fit half's visits equal the score visits of the OTHER half (disjoint from the half they are applied to);
     - the score / baseline visits equal T12's independent gateE set;
     - the replicate count equals R for both replicate sets (G51.19).
2. **Trusted dumps (E15-2).**
   - T15 recomputes the observed side (obs BA, ages, horizons) from the table, and pair membership from the verifier's own undisturbed spec.
   - T15 recomputes the latent statistics from MODEL on validation encodings, applying G51.11.
   - T11i recomputes the null composition (Bray) independently, and the G51.15 bootstrap bound from the dump.
3. **T1j / T1k / T1l (E15-3 / F-F1 / F2).**
   - T1j excludes the G46.1 / G51.1 incomplete visits, which is a spec mirror of decisions made before the rebuild.
   - T1k and T1l keep their pre-registered thresholds. Under G51.1 and the new curated product they are expected to pass; if they fail on the rebuild, that is a finding and is not loosened.
4. **T3c2z (E15-4)** uses the G51.24 zone rule independently: tile by the zone of the longitude.
5. **Planted gaps (E15-5 / E15-6 / E15-8 / E15-9 / E15-11 / E15-12):**
   - T4n samples clocks near age 0;
   - T4q tests clock exclusion;
   - U8 tests each XL block separately;
   - T5 regains full coverage (the G47.7 restriction is for T5s only);
   - U10r samples randomly;
   - T4y compares the table build_id and FAILs without MODEL;
   - T3j FAILs when its input is missing.
6. **Labels (E15-7 / F-F9).** The G51.5 hash is used. A third label, "outcome", applies to checks whose failure is a model or data result (T15, T1k, T1l, T3i, T10). Labels compare table build_ids, not only code.
7. **Pre-registration table (E15-10).** Every threshold is listed with the date it was set, including T12 runs |z| < 4 and n ≥ 20, T4x ≥ 8, and U10r 0.995. Where the history cannot show that a threshold was set before its first run, that is stated.
8. **The rest of E15 INFO:**
   - PRODUCERS lists every check that reads stage outputs;
   - T8h counts the < 0.3 branch separately (≥ 20);
   - R4 soils90 fill.json is checked.
   - Mirrors of G51.1–3 / 7 / 8 / 10 / 11 / 15 / 21 / 22 in the spec.

## G53 — Round-15 driver decisions (r15 D / F slices, coordinator)
1. **Adopt (D15-1).** Refuse to adopt a file whose last writer is another stage that is not a later stage.
2. **Detector (D15-2 / F-F5 / F-F6).**
   - Adopt the D reviewer's broad tail scan: operator, keyword (and / or / not / in / is / if-else), comma, unary-without-space and call-paren tails, at any bracket depth and across continued lines. Add every new D / F case to the suite.
   - Reword the 6 prose hits.
   - The driver runs scripts/tests/test_check_comments.py at start and refuses if it fails.
   - Rejected: accepting the misses (real lines in g04 / g02b / gv_verify would swallow silently).
3. **Training switches outside train.env (D15-3 / B15-5).** Refused (T4Y_COMPARED names; NT is a perf knob and stays allowed).
4. **Failed gates (D15-4).** A stage whose gate, transitively, FAILED is refused, not only marked; `--force` does not override this. This amends G21 / G45.12: downstream stages that are already current stay current, but nothing new runs past a failed gate.
5. **Small items:**
   - `--log` is opened after the lock (D15-5);
   - non-production `dynamics_pairs_*` are REPORT_ONLY (D15-6 / C15-9 / F-F8);
   - the late-file refusal is re-evaluated at the end of every run (D15-7);
   - the AST env guard fails closed on any unrecognised os.environ / getenv / environ reference (D15-8);
   - the typed-config gaps are refused (D15-10): null / invalid name and model_prefix, verify.n, a data path outside data/proto, a bool where an int is expected, duplicate seeds, missing sections, an empty config;
   - WORKSPACE is made absolute (D15-11);
   - VERIFY_INPUTS entries are made absolute, and the evaluation subprocess gets a timeout;
   - STANDS_STAGE_JSON is stripped for the driver's own import;
   - at start, the driver checks that g03b's SWITCHES keys equal g03b's env reads.
6. **Accepted, documented:**
   - over-invalidation (D15-9: hashing by bytes and by module statements is the safe direction);
   - grandchild lock (D15-12: no stage spawns subprocesses);
   - file-granular switch acceptance (D15-13);
   - the microsecond Popen window.
7. **Docs (D15-14, F-F11).**
