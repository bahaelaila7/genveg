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
