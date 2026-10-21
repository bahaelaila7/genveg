# Generative FIA vegetation model + predictive mapping (proposal, 2026-10-19)

User request: "build a vegetation data generative model using FIA data. Then predictive mapping from SOLUS (and derived), AlphaEarth, …
to the FIA vegetation … so I don't have to take TreeMap/LANDFIRE at face value … but can get the vegetation out of the generative model."
The user has their own ideas; this is the independent proposal for discussion.

Environment facts: no GPU, 8 cores, 30 GB RAM; FIADB national (1.99 M plot records, 58 states, INVYR 1968–2025) with PUBLIC (fuzzed)
LAT/LON only.

## Target
- **What is generated:** one representation, a plot-scale tree list. Each tree has species, DBH, height, crown ratio, age and a TPA
  expansion; the plot also gets stand age, origin (planted / natural) and years since disturbance.
- **Translators** give every simulator a consistent forest:
  - FVS: TreeInit directly;
  - H3P / HBAD: cohorts (N, QMD, H, A per species × age class);
  - BiomassSuccession / PnET: cohort biomass via FIA allometries.
- **Ages:** from the curated cohort ages, not from FIA site-tree ages, whose sparse, size-selected sample would be a missing-not-at-random
  (MNAR) problem.

## Model
**v1 (hierarchical, CPU-friendly, interpretable):**
1. Stand structure (BA, TPA, QMD, dominant H, stand age, origin): a conditional flow or mixture density network, constrained to the
   Reineke SDI bounds.
2. Composition: Dirichlet-multinomial over species keys.
3. Diameter distribution per species: a Weibull mixture.
4. Height and age given D: the H3P species H–D curve plus residual, and an age(D, site, origin) model.

**v2 (only if v1 misses joint structure):** a conditional autoregressive transformer over tree tokens (species, DBH bin, H residual,
age bin, stop), with cross-attention to the covariate encoder.

## Covariates per FIA plot visit
- AlphaEarth of the measurement year (2017–2025 → training on visits from 2017 on); Tessera later.
- SOLUS soils, and the AWC bucket water balance (AET/PET).
- PRISM normals and the period's anomalies.
- 3DEP terrain: elevation, slope, aspect, TPI.
- Hansen loss year (time since stand-replacing disturbance) and treecover2000.
- TreeMap / LANDFIRE: never inputs, only baselines.

## Fuzzed coordinates (critical)
- **(a) True coordinates:** an FIA Spatial Data Services agreement, as TreeMap used.
- **(b) Latent location:** each plot gets a bag of candidate pixels within the fuzz radius, filtered by consistency with the plot's
  forest / harvest status. Training uses attention or EM over the bag (multiple-instance learning); coarse covariates are robust to the
  fuzz.

## Validation
- Spatially blocked held-out plots.
- Metrics:
  - CRPS / coverage / energy score for the structure variables;
  - Bray–Curtis for composition;
  - Wasserstein distance on DBH distributions.
- Baselines: kNN / random forest on the same covariates (TreeMap method without LANDFIRE), and TreeMap 2022 at held-out plots.
- Simulation fidelity: H3P / FVS projections from generated vs observed tree lists.
- Design-based: map totals vs FIA county / EVALIDator estimates.

## Mapping
- Per stand, K plot-scale samples conditioned on the stand's pixels → composite tree list + ensemble.
- Harvested / young stands are conditioned on time since loss; this replaces kNN donors.

## Phases
| phase | content |
|---|---|
| P0 | decisions + covariate extraction |
| P1 | training table |
| P2 | baselines |
| P3 | v1 + validation |
| P4 | v2 if needed |
| P5 | stand mapping + simulator inputs |
| P6 | simulation / design-based validation |

## Open decisions
1. True FIA coordinates available?
2. Training region: R4 or the eastern US?
3. Hierarchical v1 first?
4. How the user's own ideas fit in: generative tree list, MNAR, the predictive-coding age model (TreeAgeTransformer).

## Decisions (user, 2026-10-19/20)
1. **Location: multiple-instance learning (MIL).** The prior comes from matching disturbance records (FIA TRTCD / TRTYR / DSTRB* ↔ Hansen loss year and
   LANDFIRE annual disturbance 1999–2025). The viability test is `scripts/g01_disturbance_match.py`:
   - 88 % of dated FIA harvests have a Hansen patch in their year ± 1 within 800 m (controls: 42 %);
   - candidate area shrinks 5–8× for those plots.
2. **Region:** GA+FL first.
3. **Targets:** the simulation-engine inputs (cohort tables per subplot: species × age class → N, QMD, H, crown ratio, biomass); no tree lists, no
   FVSjl. Crown ratio was added because HBAD3P showed it is a major growth driver.
4. **Ages:** MNAR / age completion deferred.
5. **Resolution:** model support = FIA footprint (~90 m); MIL candidates = delineated stands / footprints in the fuzz disk; output = per stand
   (K samples).

## Temporal JEPA design (user, 2026-10-20: "instead of reaching for AlphaEarth of older years … a temporal model that takes disturbances into account …
LeJEPA can help") — replaces fetching AlphaEarth for every FIA year
1. **Vegetation encoder** E_v: a set encoder over cohort tokens → z_v, regularised with SIGReg (LeJEPA's isotropic Gaussian), giving a generative
   latent and a well-behaved distance.
2. **Forward predictor** P(z_v(t_FIA), site covariates, Δt, disturbance sequence in (t_FIA, t_AEF]) → AlphaEarth embedding at 2022 and 2025. The
   AlphaEarth side is fixed, so it is plain regression there; SIGReg matters on the vegetation side.
3. **MIL loss:** a soft-min over the fuzz-disk candidates, weighted by the disturbance-match prior, plus an entropy term against the "easiest
   candidate" shortcut.
4. **Remeasurement consistency:** z_v(t1) + Δ + disturbances → z_v(t2) on FIA remeasured pairs (later: HBAD3P simulations as extra pairs).
5. **Decoder:** z_v → cohort tables (the generative model).
6. **Mapping:** an inverse encoder (AlphaEarth 2022 + 2025, covariates, disturbance history of a stand) → z_v(target year) → decoded ensemble.

Risks:
- shortcut via site covariates (countered by reconstruction terms);
- no true plot locations for validation (use dated-disturbance plots + held-out vegetation skill vs kNN / RF);
- long undisturbed gaps (rely on the dynamics step).

**First viability test:**
- steps 1–3 on GA+FL visits with dated disturbances;
- (a) does the predicted embedding localise the true harvest patch better than the disturbance prior alone;
- (b) does AlphaEarth → vegetation beat kNN on held-out plots.

## Liquid flow + LeJEPA jumps (user, 2026-10-20: "Liquid NN for the temporal model with LeJEPA making the translation in the jumps of disturbance")
- **Continuous part:** a liquid time-constant / closed-form continuous-time (CfC) network evolves z_v over irregular gaps (FIA visits, AlphaEarth
  years, disturbance years on one timeline). Annual climate inputs (AET/PET, temperature anomaly) modulate the input-dependent time constants:
  the climate hook. Library: `ncps` (PyTorch, CPU).
- **Discrete part:** at each disturbance year, a jump map z⁺ = J(z⁻, LANDFIRE type / severity, Hansen loss) is trained JEPA-style, with SIGReg on all
  latent states. Signals:
  - FIA remeasurement pairs straddling recorded events;
  - AlphaEarth 2022 → 2025 across 2023–24 events;
  - undisturbed pairs constrain the flow alone.
- **Loop:**
  1. encoder(FIA vegetation) → z(t0);
  2. flow + jumps → z(2022), z(2025);
  3. head → AlphaEarth embedding;
  4. MIL soft-min over candidates (disturbance prior);
  5. decoder → cohort tables.
- **Pretraining on HBAD3P free-run trajectories** (harvests, climate scenarios): physically consistent synthetic sequences, then fine-tuning on FIA /
  AlphaEarth.
- **First test (GA+FL):** CfC flow vs Δt-MLP vs GRU-D under the same jump / LeJEPA setup. Metrics:
  1. localisation of dated harvest patches;
  2. later-visit vegetation of held-out remeasurement pairs;
  3. AlphaEarth → vegetation skill vs kNN.

## Status 2026-10-20 (prototype → v3)
**v1 (Baker, 596 plots) → v2 (all GA+FL, 8.7k plots):**
- Temporal JEPA (SIGReg set encoder, liquid flow, disturbance jumps, MIL AlphaEarth heads, decoder, inverse encoder).
- At kNN parity, with no localisation signal: the fuzzed location is the binding constraint.

**v3 = options 2 + 3 (user choice; true coordinates unavailable):**
- A: landscape prior p(z | 1.6 km landscape, year), a mixture-density network on the frozen v2 latent.
- B: forward likelihood, fine-tuned only on well-localised plots.
- Mapping: prior × likelihood per stand.
- Calibrated A (training-plot log factors) beats kNN on TPH / BA / QMD / dominant species (0.66 / 0.48 / 0.31 / 39 % vs 0.70 / 0.49 / 0.34 / 36 %).
- B was weak before the time-matched AlphaEarth fetch (`scripts/g02c_fetch_visit_aef.py`: 1,366 same-year windows of localised plots).

**Audit (user request "double check the correctness of the pipeline"), all fixed in g02b / g04, rerun via `data/proto/run_v3.sh`:**
1. Visits with no live tree (fresh clear-cuts, 1,191 = 3.5 %) were dropped. They are now empty cohort tables.
2. SOLUS −9999 nodata leaked into the soil features. It is now replaced by the regional median.
3. The Hansen loss flag fired for 52 % of footprints in year 2000 (lossyear 0 = no loss).
4. The LANDFIRE "Unknown" class (image-based change detection; 71–78 % of Hansen-loss pixels carry a LANDFIRE code, mostly this class) was
   discarded. It now has its own class and counts in the prior.
5. Planted flag: FIA STDORGCD 1 = artificial (was tested as 2; not yet used by the model).
6. Plot landscape descriptors came from the prior-biased 256-candidate subsample. Inverse-inclusion weights now make them unbiased whole-disk
   statistics, consistent with the uniform stand landscapes.
7. Stand landscapes in g04 came from 4 footprints per stand. g05 now uses all forest footprints within 1.6 km on a 1 km grid, exactly as for
   the plots.
8. The A×B evaluation compared 2025-state predictions with visit-year observations. It now works at the visit year (prior at the visit,
   likelihood flowed forward from it).

**Verification suite + semantic audit (user, 2026-10-20: "triple check again. I don't trust it yet"; "check again the semantics for the entire pipeline").**
`scripts/gv_verify.py` recomputes each stage from the raw sources by a separate code path and prints PASS/FAIL:
- T1: tokens from SQL; forested shares from FIADB.
- T2: coordinates.
- T3: candidates checked against direct raster reads at the stored row/col (stored since this audit). This covers the LANDFIRE code tables
  (decoded independently), the prior (re-implemented from its specification on FIADB COND), and the AlphaEarth-change vs event-year alignment.
- T4: model tensors and flow unit tests.
- T5: the landscape estimator vs the full disk.
- T6: stand footprints.
- T7: downstream keys.

The first run (table of the 1st audit) passed 31/34. Its three failures, plus the semantic review, gave these fixes:

9. **Typed event matching.** The location prior matched every FIA event to harvest / thinning / unknown / Hansen evidence, including ground fire
   (DSTRBCD 31, the most frequent FIA disturbance in GA+FL: about 3.7k condition records, prescribed burning) and weather / insects. Events are
   now matched by type (`EVK` in g02b), using FIADB COND of all forested conditions and codes 1–3.
10. **Negative evidence.** A remeasurement with no FIA event code, or the stand age at the first visit, makes candidate stand-replacing evidence
    (harvest class or Hansen loss) inside that period ×0.25. The old aged-history rule used the stand age at the last visit.
11. **LANDFIRE calendar year.** The 2024 / 2025 LANDFIRE files are fiscal-year products: each code carries CALENDAR_YR (Oct–Dec in the
    previous year). Events are now placed at the code's calendar year. The "Unknown" codes described as burned (BAECV ≥ 95 %, burn probability
    ≥ 95 %, "most likely fire related") are now fire, and "Mechanical Unknown" (2024+) is now thinning.
12. **Densities per forested area.** Densities were per plot area: 25 % of forested visits are < 95 % forest, 10 % ≤ 50 %. TPA_UNADJ is now
    divided by the forested share of the subplot area (trees ≥ 5 in) or the microplot area (saplings), from COND SUBPPROP / MICRPROP_UNADJ.
13. **Candidate subsample.** When > 256 footprints had an elevated prior, the top ones were kept and the rest of the disk vanished: −72 % of the
    footprint count in 7 of 60 sampled plots, so T3g failed. Sampling is now stratified and uniform within each stratum. Kept candidates carry
    prior × incl_w: the fill candidates had been under-weighted by the factor they stand for. Localisation numbers (n_hi, mass_hi) are computed
    on the full disk, and the `hi` flag is stored per candidate.
14. **AlphaEarth 3×3 mean.** It was zeroed when any pixel was nodata; it now averages the valid pixels. Soils are the centre pixel (documented;
    the test had assumed a 3×3 mean).
15. **Flow timing.** The first year integrated a full year after a mid-year visit; it now takes an exact step over the part of the year after
    the visit. An event in the visit year counts only for a first-half visit.
16. **Q decoding target.** Q (2025 state) was decoded against visit-year cohorts even when the candidate had an event after the visit. Those
    candidates are now masked. POSW now comes from training visits only.
17. **Evaluation timing.** The g03b evaluation compared 2025-state predictions with last visits of any year; it now uses visits ≥ 2019. In g04,
    kNN uses training plots within ±3 years of the query visit (A is conditioned on the year).
18. **Footprint mixture.** The g04 A×B likelihood mixed footprints after a per-footprint minimum shift, which silently re-weights them; it now
    uses log Σ_c p_c exp(−d_c/T). In g05, all footprints of a stand observe the same vegetation, so the evidence is now the mean distance (was
    a sum of per-footprint-normalised terms). The final flow spreads samples over the stand's footprints (was footprint 0 only).
19. **Plot ids.** pid is a row number of the table and shifted by 61 plots when bare visits were added. localisation.parquet (now
    `scripts/g02d_localisation.py`, reproducible) and the visit_aef files (renamed `<st>_<u>_<c>_<p>_<year>.npy`) are keyed by the FIA plot
    identity.
20. **Tree status.** STATUSCD now comes from the main query rather than a second order-aligned query. The key was unique, so there was no
    output change.
21. **Candidate sampling.** g03b drew the K = 48 MIL candidates without replacement with p ∝ prior. Both the MIL estimator
    (log-mean-exp − log K) and the posterior softmax assume iid draws from the prior, so a concentrated prior was flattened (each high-prior
    footprint at most once). Draws are now iid with replacement.

Checked and fine:
- FIA public coordinates are identical across all inventories of a plot (span 0 m).
- Live trees missing age or height: 18 of 852k.
- Harvested trees are STATUSCD 2 (tree_fate harvest_*), so they are not live.
- Hansen lossyear is 1–24.
- The AlphaEarth de-quantisation is sign·(q/127.5)².
- The calibrated-logit correction lp − log POSW is the exact inverse of pos_weight BCE.

Known limitations (not fixed):
- The candidate forest-land mask uses TreeMap 2022 forest pixels (classes 1–4). This is a TreeMap dependency of the mask only, not of the
  vegetation attributes.
- Plots converted to non-forest after their visits (class 9) have no true candidate.
- Swapped FIA plots are not modelled.
- The pair loss compares the state at the end of v2's year with z(v2) (≤ 1 yr offset).

**Third pass (user, 2026-10-20: "once more, do verify and be thorough").** Two independent read-only reviews (data/prior builder; model / evaluation / mapping),
each claim re-checked before acting. The first verifier run on the 2nd-audit table passed 37 / 39: T3i and T5 failed because their test designs were
inadequate (too few harvest-only samples; a max-over-900-values tolerance), so both were redesigned. Fixed:

22. **LANDFIRE 2008 code years.** The 2008 code table labels 135 of 215 codes DIST_YEAR = 2009. Their pixels carry Hansen loss in 2008 (25.5 %),
    not 2009 (1.8 %), so the label is a table error. The event year is now CALENDAR_YR (fiscal-year products) or else the file year; DIST_YEAR is
    ignored.
23. **'Increased Green' rows.** 75 Fire / Wildfire rows with severity 'Increased Green' (post-fire greening) were events; they are now none.
    LANDFIRE 'Biological' is a treatment (biocontrol), now grouped with chemical treatments, not insects / disease.
24. **One physical event, one factor.** FIA records of the same event (cut + site preparation + planting, re-reports a year apart) multiplied
    (×8³ and up; 39 % of event plots had a cut and a regeneration record within 2 yr). Records whose years chain within 2 yr now form one cluster
    (union of evidence and windows, max likelihood ratio, ×2 for a cut matched by a harvest class). Likelihood ratios now follow detectability
    (assumption): ×8 cut / regeneration / site preparation / human / crown fire, ×4 insects / weather, ×3 ground fire / other treatments /
    unknown.
25. **Baseline-relative strata.** The high-prior set and the sampling stratum were median-relative: when most candidates were penalised,
    unpenalised ones became "high-prior". Both are now relative to the no-evidence baseline weight 1 (hi = weight ≥ 4, up = weight > 1.5).
26. **Negative evidence per FIA protocol.** A no-code visit's period is since the previous measurement (PREV_PLT_CN: periodic inventories) or,
    for a new plot, the 5-yr lookback. The stand-age rule is removed (STDAGE is not the origin, and partial cuts do not reset it). Codes that
    cannot hide stand-replacing evidence (understory insects / disease, ground fire, animal, vegetation) no longer block. Stand-replacing evidence
    must cover ≥ half the footprint.
27. **Footprint fractions.** Disturbance and Hansen evidence were read at the centre 30 m pixel while AlphaEarth is a 90 m (3×3) mean. They are
    now 3×3 fractions per class and year, with severity = max.
28. **True bare visits.** Forested FIADB visits of the table's plots with no live tree and no curated record (447) are added as empty cohort
    tables. Visits that have live trees in FIADB but were removed by the curation remain absent (ages unavailable).
29. **Forested share of the sampled area.** Non-sampled area (status 5) is excluded from the denominator, as in FIA estimation. Visits whose
    forested share is < 0.1 (192) are dropped instead of clipped.
30. **Separate dynamics and likelihood flows.** g05 mapped through a path no evaluation covered: z(2021) → flow → decode with the visit-year
    calibration, using the B-fine-tuned flow even with USE_B = 0. B now fine-tunes a COPY of the flow + heads (likelihood only), while the v2
    flow stays the decoder-consistent dynamics.
31. **The mapping path is validated.** g04 calibrates the exact g05 path (z ~ MDN(X_L, 2021) → v2 flow through a prior-drawn candidate's
    history to the visit date → decode) on calibration plots and reports its held-out error vs kNN at visits 2022–2025. g05 uses those factors.
32. **Calibration out of sample.** Calibration plots = 15 % of the training blocks, held out of the MDN fit.
33. **Spatial buffer.** Training plots within 3.2 km of a test plot (disks could overlap) are excluded from training and from the kNN pools.
34. **Pair-loss time.** The pair loss read the flow at the end of v2's year. Flow now takes an end time: the state is AT v2's date, and an event
    of year y applies iff mid-y lies in [start, end].
35. **kNN baselines.** g03b's kNN divided by per-dimension sd, so near-constant disturbance dims dominated; feature groups are now scaled to
    unit total variance. kNN predictions are now medians of the neighbours (like the model's log-space point estimate).
36. **Metrics on stocked visits.** Bare visits had dominant key 0 and log errors ≈ 9. Errors and composition are now on stocked visits, with
    bare visits reported separately (predicted BA). The g03b localisation check used the sum over heads (training uses the mean); now the mean.
37. **MDN samples and ESS.** MDN samples are clipped to the encoder range (±3). The likelihood ESS is reported in g04 (A×B) and g05.

Checked correct by the reviews:
- Subplot / microplot TPA_UNADJ split at 5 in (no macroplot).
- SUBPPROP sums to 1.
- The TRT / DSTRB families.
- Raster grids are identical.
- Fiscal-year 2024 codes with CALENDAR_YR 2023 match Hansen 2023 (39.5 % vs 7 %).
- Horvitz–Thompson weights.
- The MIL / posterior estimators.
- The mixture likelihoods.
- The exec-slice markers.
- Unit conversions.

**Results on the verified table (2026-10-20, run_v3_train.sh; median |log error| on held-out stocked visits, spatial blocks + 3.2 km buffer):**

| | TPH | BA | QMD | composition (Bray) |
|---|---|---|---|---|
| null (year-matched training median) | 0.638 | 0.464 | 0.341 | 0.743 |
| kNN on the landscape (±3 yr) | 0.703 | 0.454 | 0.357 | **0.579** |
| A: landscape prior (calibrated) | **0.636** | **0.438** | **0.327** | 0.611 |
| mapping path (z(2021) → flow → decode), visits 2022-25 | 0.588 (null 0.608) | 0.405 (null 0.454, kNN 0.421) | 0.317 (null 0.327) | 0.604 (kNN 0.574) |

- **A vs the null baseline:** A beats null only slightly (BA −6 %, QMD −4 %, TPH ±0). kNN is worse than null on TPH, and its composition is the
  best.
- **Localisation:** none. B puts 0.127 of the posterior mass on the pinned footprints, against their 0.152 share, i.e. no better than chance;
  g03b gives 0.335 vs 0.341. A×B on the 15 localised held-out plots shows no consistent gain.
- **Inverse encoder Q:** biased low (BA −0.76 log); it is not used by the maps.
- **Stand maps (Baker):**
  - The harvested status shows almost the same generative BA as forest: 54 vs 57 ft²/ac (B0), 61 vs 66 (B1). The TreeMap value is stale for
    them.
  - Correlation with TreeMap BA: 0.15 (B0) / 0.34 (B1).
  - Likelihood ESS is about 31 / 64.
- **Diagnosis (direct test):** the flow barely learned the disturbance jumps. A 2023 clear-cut (harvest + Hansen + high severity) lowers the
  decoded 2025 BA by only 23 % (12.4 → 9.6 m²/ha before calibration); a fire lowers it by 10 %.
- **Cause:** every jump signal (MIL AlphaEarth match, remeasurement pairs) is weighted by the candidate posterior. With no localisation skill,
  the candidate histories are mostly not the plot's, so event-conditioned dynamics are averaged out.

**Proposed next step (not done):** train the jumps from FIA's own dated records at the TRUE location. The remeasurement pairs (25.6k, incl.
the bare post-harvest visits) would flow through a history built from the plot's TRT / DSTRB records mapped to the same class channels,
instead of a candidate's LANDFIRE / Hansen history. The flow keeps its interface, so mapping still uses the stand's LANDFIRE / Hansen history.
Until then the Forest Explorer Baker layer stays at v2.

**FIA-anchored dynamics (user, 2026-10-20: "that's what I meant by using LANDFIRE disturbance to anchor FIA data … COND.DSTRB* and COND.TRT*").**
The FIA records had been used only to LOCATE plots (prior). Now they also anchor the DYNAMICS:
- `scripts/g02e_fia_records.py` writes the plot's typed events, its negative-evidence periods and a per-visit "hiding code" flag. It uses the
  same functions that built the prior (`g02b.fiadb_visits / fiadb_events / fia_records`, refactored and shared), and its records reproduce the
  stored prior exactly (400 plots, ratio spread 1.0000001).
- In g03b, each remeasurement pair (v1 → v2) flows only through candidate histories consistent with FIA's record of that period:
  - every detectable event cluster (likelihood ratio ≥ 4) must be matched by compatible LANDFIRE / Hansen evidence in its window;
  - if v2 has no code that could hide a stand-replacing event, the history must contain none (harvest class or Hansen loss on ≥ half the
    footprint);
  - candidates are drawn from the prior restricted to that set and weighted uniformly (the AlphaEarth posterior has no skill).
- Counts: 25,579 pairs. 6,484 have a detectable FIA event (97.5 % matched by some candidate), 18,527 are constrained by negative evidence, and
  161 fall back to the prior.
- The flow keeps its LANDFIRE-channel inputs, so mapping with a stand's LANDFIRE / Hansen history is consistent with training.
- `scripts/g06_dynamics_check.py` is a held-out test of the dynamics: synthetic events, and test-plot pairs flowed through FIA-consistent
  histories. Its change metric = predicted log BA change (v2 vs decoded v1) vs the observed change, against zero change.
- gv_verify T8 recomputes the pair histories independently.

Before anchoring (the v2 model of the verified table):

| period (FIA record) | n pairs | pred / obs median log BA change | change error model / zero | corr |
|---|---|---|---|---|
| detectable cut | 1006 | −0.10 / −0.43 | 0.572 / 0.586 | 0.30 |
| other detectable event | 268 | −0.09 / +0.05 | 0.285 / 0.197 | 0.20 |
| no event | 3792 | +0.01 / +0.14 | 0.191 / 0.172 | −0.01 |

The synthetic 2023 clear-cut gives BA × 0.76 by 2025 and a fire × 0.90, so the dynamics had essentially no skill (neither disturbance nor
growth).

After per-PERIOD anchoring (14 epochs, same table), held-out pairs:
- cut: change error 0.537 vs zero-change 0.586 (corr 0.43, was 0.30);
- undisturbed: 0.168 vs 0.172 (corr 0.19, was −0.01; the first time below zero-change);
- other events: no gain;
- synthetic clear-cut BA ×0.67 (was ×0.76);
- the stand map still does not separate harvested stands (69.9 vs 73.3 ft²/ac).

**Plot-level location (user, 2026-10-21: a location found from coinciding FIA / LANDFIRE records holds for ALL of the plot's visits).**
Problems with the previous handling:
- The g02b prior was plot-level but mis-specified: it rewarded matches (×8) and never penalised misses, and it ignored how common the evidence
  is in the landscape.
- The pair anchoring was per period.
- The AlphaEarth likelihood was per visit.

`scripts/g02f_location_em.py` builds one posterior per plot from its whole FIA history:
- **Base prior:** the fuzz-distance mixture (share π within 800 m).
- **Likelihood ratios:**
  - per event cluster: q_g / b for a matching footprint, (1 − q_g) / (1 − b) for a miss (b = background share of that disk);
  - per FIA-undisturbed period: f / b₀ and (1 − f) / (1 − b₀);
  - an 'outside the disk' component s (swapped / converted plots).
- **Fit:** q_g, f, π and s by EM over all plots, with a year-shifted placebo fit as control.
- **Use:** every visit uses this posterior (MIL draws, pairs, Q target). B's localised plots are those whose 80 % credible set has ≤ 50 footprints.
- **Exploratory EM on 3,000 plots (assumed-shape version):**
  - strong FIA events are detected at the true footprint q ≈ 0.57, vs b ≈ 0.23 at random footprints;
  - but the total information is only ≈ 0.03 nats / plot (placebo +5 yr: about the same), so most plots stay weakly localised.
- **Verifier:** gv_verify T9 recomputes the posterior from the spec and the fitted parameters; T8 keeps testing the per-period rule
  (PAIR_ANCHOR=period).

**Plot-level location — results (2026-10-21), and why the EM posterior is not used:**

*In-sample fit, disk-wide background (400 iterations).*
- Real records: +296 nats over no-information, strong-event detection q = 0.88, f → 0, inner share π = 0.70, outside share s = 0.52.
- The fit drifts to a degenerate solution: half the plots 'outside', the rest with hard constraints.
- The control was the trouble:
  - A year-shifted placebo scored higher (π → 0). It is confounded: the shifted records contradict the true footprint's real events and push
    mass away from it.
  - A permutation placebo (another plot's records, same state) also scored higher after a refit (+371).

*Held-out test (fit on half A, score half B).*
- With the disk-wide background, permuted 194 > real 121. Cause: disturbance is more common in the ring than near the public point, so the
  inner / ring split fits ANY records. The background is now per distance class (`g02f.bg`).
- With the per-class background, real 45.5 vs permuted 31.8 nats over 4,355 plots, i.e. ≈ 0.003 nats / plot.

*Conclusion.*
- Under the public coordinates, FIA ↔ LANDFIRE coincidences carry almost no FOOTPRINT-level location information beyond regional timing.
- The disturbances are visible in the disk (g01: ≈ 24 % of the 800 m disk shows loss in the window of an FIA-recorded cut), but harvest units
  are large and clustered.
- π also goes to the boundary in both fits, so it is not trusted.
- g02f is kept as an analysis tool; its outputs are in data/proto/gafl_g02f_unused/.

*What the records DO determine: which disturbance histories the plot can have, for all of its visits.*
- `scripts/g02g_history_consistency.py`: candidates consistent with the plot's WHOLE FIA history (every detectable cluster matched, no
  stand-replacing evidence in any undisturbed period).
- Consistent footprints, median: 189 of 753 for all plots; 43 for plots with ≥ 1 detectable cluster; 8 for ≥ 2.
- The set is empty for 8 % of plots (13 % / 28 % of event plots): FIA / LANDFIRE disagreement or swapped plots. Those plots keep the prior.
- g03b LOC=history (default): every visit of a plot (MIL, pairs, Q) draws from the prior restricted to this set.
- gv_verify T10 recomputes the set independently.

**Anchoring comparison (2026-10-21).** Two seeds each, all scored by g06 under the same evaluation conditioning (records only, no outcome):

| model | cut pairs: change err (zero 0.586) / corr | undisturbed: err (zero 0.172) / corr | synthetic clear-cut |
|---|---|---|---|
| unanchored | 0.575 / 0.30 | 0.193 / −0.02 | ×0.76 |
| per-period s0 / s1 | 0.546 / 0.44 · 0.590 / 0.29 | 0.190 / 0.22 · 0.161 / 0.08 | ×0.73 / ×0.78 |
| hybrid (history ∩ period) s0 / s1 | 0.599 / 0.31 · 0.565 / 0.30 | 0.170 / 0.06 · 0.180 / 0.02 | ×0.65 / ×0.62 |

The seed-to-seed spread is as large as the differences between configurations, so none of them learns the disturbance response: on cut pairs
the predicted median log change is −0.04 to −0.15, against −0.43 observed.

Diagnosis:
- **Not the jump cap and not the relaxation.**
  - The learned clear-cut jump is tiny (median 0.05 per latent dim) against the median 0.41 needed to reach the bare state. Its direction is
    right (cosine 0.61).
  - The flow relaxes slowly (λ ≈ 0.02 / yr, half-life ≈ 40 yr); the jump persists from 2023 to 2025.
- **The model learned the AVERAGE outcome of an FIA cut** (×0.65 observed: mostly partial cuts or regrowth before v2), applied to evidence at
  candidates that are mostly not the plot's.
  - Nothing tied strong stand-replacing evidence to complete removal.
  - In real candidate histories (weak evidence fractions) the response shrinks further.

**Severity anchoring (training only).** FIA's own outcome now says what the true footprint must show:
- a period with a detectable FIA event and BA(v2) < 0.3 BA(v1) → candidates with stand-replacing evidence (harvest class or Hansen loss on ≥
  half the footprint) in the window;
- BA(v2) > 0.6 BA(v1) → candidates without it;
- the evaluation stays records-only (g06 runs with LOAD=1, which switches it off);
- gv_verify T8b checks it.

**Severity anchoring results.**
- The synthetic clear-cut response strengthened consistently: ×0.51 / ×0.49, against ×0.62–0.78 before.
- Held-out cut pairs were still no better than zero change, even the best-anchored ones (≤ 10 consistent footprints: predicted −0.00 to −0.11
  vs observed −0.37).
- **Training pairs too:** the model predicted −0.16 vs −0.44 observed on cut pairs, +0.04 vs +0.15 on undisturbed ones. So the problem is
  underfitting, not location or generalisation.

**ROOT CAUSE (2026-10-21): the autoencoder did not reconstruct plot summaries.**
- decode(encode(visit)) vs the visit itself: log slope 0.22 / 0.24 / 0.07 (TPH / BA / QMD), corr 0.50 / 0.48 / 0.24, the same on train and
  test.
- Every downstream quantity passes through that decoder, so all changes were compressed about 4×, and the prior / maps were barely better
  than null.
- The decoder had been trained slot-wise only (presence BCE per species × age-class slot + masked value MSE); the expected plot totals
  regressed to the mean.

**Fix (g03b):**
- a summary-reconstruction loss (squared log errors of the decoded TPH, BA, QMD) on all three decoder uses: encoder states, flowed v1 → v2
  states, and Q;
- absent-slot values pulled toward the slot's training mean (they entered the expected totals untrained);
- a physical QMD cap of 150 cm, also in g04.summaries.

After 2 epochs, held-out reconstruction:
- TPH slope 0.95 / corr 0.96;
- BA 0.76 / 0.92;
- QMD 0.79 / 0.92.

g06 now reports the reconstruction fidelity. Full 14-epoch runs (2 seeds) → g04 → g05 → gv_verify: data/proto/run_sum.sh.

**Results with the reconstruction fix (14 epochs, seeds 0 / 1; gv_verify 42 / 42 PASS):**
- **Reconstruction** (held-out): slope 0.91–0.97, corr 0.97–0.98 for TPH / BA / QMD (was 0.07–0.24 / 0.24–0.50).
- **Dynamics** (held-out pairs, records-only conditioning):
  - undisturbed growth: corr 0.70 / 0.61, change error 0.143 / 0.141 vs zero-change 0.172;
  - cut pairs: corr 0.56 / 0.52, median predicted change −0.29 / −0.34 vs −0.43 observed, but the absolute error is 0.618 / 0.704 vs zero
    0.586 (candidate histories are still not the plot's own);
  - the best-anchored cut pairs (≤ 10 consistent footprints) beat zero change for seed 0 (0.494 vs 0.561);
  - other events: predicted declines that did not happen (error 0.32 / 0.44 vs 0.197);
  - synthetic 2023 clear-cut: BA ×0.10 / ×0.03 by 2025 (was ×0.5–0.8).
- **Mapping path, held-out visits 2022–25:** BA 0.395 vs null 0.454 and kNN 0.421; TPH 0.574 vs 0.608; QMD 0.307 vs 0.327.
- **Landscape prior A:** BA 0.432 vs null 0.464.
- **Composition got worse** (prior Bray 0.706 vs 0.611 before; kNN 0.579). A composition term is now in the summary loss (W_COMP), rerun via
  data/proto/run_comp.sh.
- **Stand map (Baker):** harvested stands are now clearly lower — B1 37 vs forest 56 ft²/ac, B0 44 vs 61 (were ≈ equal).
- **B localisation:** still at chance.
- **Forest Explorer:** the Baker layer now carries these v3 values via `scripts/g07_fe_attach.py`, a reproducible exporter that replaces the
  ad-hoc attach. GEN_SRC is the version tag; the backup is baker_fl.geojson.bak_v2. The server was restarted.

**+ Composition term (W_COMP = 1; seeds 0 / 1; gv_verify 42 / 42 PASS) — current model (m_comp_s0 = model.pt):**
- **Reconstruction:** slope 0.90–0.97, corr 0.96–0.98.
- **Dynamics:** growth corr 0.68 / 0.65, change error 0.147 / 0.157 vs zero-change 0.172. Cut pairs corr 0.55 / 0.50, median predicted −0.21 /
  −0.36 vs −0.43 observed, error 0.651 / 0.706 vs zero 0.586. Synthetic clear-cut ×0.08.
- **Composition:** prior Bray 0.609 (was 0.706 without the term; kNN 0.579, null 0.743); dominant-species accuracy 0.414 vs kNN 0.366. Mapping
  path Bray 0.599 vs kNN 0.574.
- **Totals:** prior BA 0.465 ≈ null 0.464 (sum-only run 0.432). Mapping path BA 0.414 vs null 0.454 / kNN 0.421 (sum-only 0.395). These are
  single g04 runs, so part of the gap may be run-to-run variance.
- **Baker map, B1:** harvested 30.0 vs forest 64.2 ft²/ac; corr with TreeMap BA 0.37 (the highest so far); likelihood ESS 53 / 64. The
  Forest Explorer layer carries this version (GEN_SRC tag).

**Open problems:**
1. Cut pairs: the change magnitude is right, but per-plot accuracy is no better than zero change (candidate histories are not the plot's own).
2. Minor events (fire, insects, weather) produce spurious declines.
3. The AlphaEarth likelihood B is at chance.
4. The landscape prior's totals are ≈ null.
5. There is no TreeMap-independent stand-level validation yet.

**Outcome anchoring for every pair (SEVERITY_ANCHOR = 2, 2026-10-22): not adopted.** In training, the measured BA outcome constrains every pair:
- kept above 0.6 → no stand-replacing evidence between the visits (17.5k pairs);
- lost below 0.3 → stand-replacing evidence required (193 pairs).

Held-out vs the composition model:

| pairs | outcome-anchored | composition model |
|---|---|---|
| minor events (error) | 0.347 / 0.349 | 0.332 / 0.366 |
| undisturbed (error) | 0.141 / 0.142 | 0.147 / 0.157 |

Cut pairs are about the same and the synthetic clear-cut is ×0.02. Everything is within seed noise; the current model stays m_comp_s0.

The remaining errors on disturbed pairs are LOCATION-limited. The records-only evaluation draws candidate histories, and for weather / insect
events the compatible evidence includes Hansen loss, so the model (correctly) predicts a decline for those candidates. Stands mapped through
their own co-located history do not have this problem.

**Next:** same-year AlphaEarth at the candidates (the g02c windows, fetched for the g02g localised plots) as a direct visit ↔ footprint match
without long flows — the last unused source of location information.

**Same-year AlphaEarth carries location information (g08, 2026-10-22; early run on the windows available).**
Setup:
- Visits 2017–2024 of the g02g-localised plots; the visit-year AlphaEarth at each history-consistent candidate (2022 mosaic, other years from the
  g02c windows).
- A head maps the frozen encoder z to that embedding, trained by MIL on training plots (597 visits).

Window georeferencing check: median cosine to the 2022 mosaic 0.977 at the same candidates vs 0.885 at shuffled ones.

Held-out results (137 visits):
- log-likelihood gain of the real visit latent over a permutation placebo (another held-out visit of the same year): +2.1 to +2.5 nats / visit
  (95 % CI 1.7–3.3 in the first run), positive for 69 %; the FIA records gave ≈ 0.003 / plot;
- FOOTPRINT level: two visits of the same held-out plot put their posteriors on the same footprints more than chance — overlap 0.094 vs 0.065
  permuted, real > permuted in 82 % of 28 pairs.

Next:
- complete the windows (2,478 for the current localised set) and rerun;
- build a JOINT per-plot location posterior from all visits' same-year matches × the history-consistent set (training plots only — localising
  a test plot with its own vegetation would leak);
- use it for every visit of those plots in training.

**CORRECTION (2026-10-22, after an independent review; user: "double check the correctness of everything").**
The g08 claim above ("same-year AlphaEarth carries location information … footprint level") is RETRACTED.
- The 'gain' compared the real visit latent with another plot's over ALL of the plot's candidates. That mixture is invariant to which candidate is
  true, so it measured only how well the plot's vegetation fits its landscape's AlphaEarth. A simulation with zero location signal gives +19.6
  nats.
- The 'overlap vs permuted' baseline destroyed the shared prior, the 'generic' candidates and the year-to-year stability, all of which inflate
  the real overlap without any location information (simulation: real > permuted 99.9 %).
- Corrected tests (g08 now; bootstrap by plot; footprint statistic = cosine similarity of the two visits' posteriors, insensitive to sharpness;
  placebo = ANOTHER plot's two latents on the SAME candidates / AlphaEarth / years / prior), 292 held-out visits:
  - plot ↔ landscape fit +1.85 nats / visit (95 % CI 1.45–2.26): landscape information;
  - FOOTPRINT selection +0.014 (95 % CI −0.042 to +0.073): no evidence of footprint-level location.
- LOC=aef is dropped (code kept as a tool: cross-fitted, common candidates, tempered by the number of visits). The window georeferencing itself
  is fine: gv_verify T13 shows the agreement with the 2022 mosaic peaks at offset (0, 0) among ±2 px shifts.

Other review findings, fixed:
- **Visit year:** year = y + doy/366 put 23 leap-year Dec-31 visits in the next calendar year (every floor(year) consumer was off). g02b now
  keeps the visit date (md) and uses decimal_year = y + (doy − 0.5) / days-in-year. `scripts/g02h_fix_visit_dates.py` migrated the current
  table (exact inverse); g02e reran.
- **Pair construction (g03b):**
  - A constraint that would empty the candidate set is now skipped (`_narrow`). The earlier ones are kept; before, a later empty result
    discarded everything.
  - A measured major loss with no recorded code no longer also gets the 'undisturbed period' constraint (they conflicted).
  - Pairs weight candidates by the same location prior as MIL; the g02b prior is used only outside it.
- **Stale files:** cand_hist / cand_loc_aef carry pid and are joined on (cid, pid) with full coverage required. LOC=history without
  cand_hist.parquet fails loudly. g02f writes localisation_em.parquet (no longer overwriting g02g's localisation.parquet).
- **gv_verify:**
  - T8 follows the sequential-with-skip rule.
  - T8b now checks: severity anchoring is OFF under LOAD=1 (evaluation sets = records-only sets); it is ON in training on the default
    PAIR=both path; it is non-vacuous (only pairs whose event-consistent set holds strong-evidence candidates).
  - New checks: T11 (Forest Explorer export = stand map × units), T12 (summary-loss targets = obs_summ, zero loss at a perfect prediction),
    T13 (pixel-exact windows), T14 (the AlphaEarth posterior only for training plots).

Interpretation note (review): held-out skill is 'given the plot's FIA records'. Candidate sets of held-out plots are restricted by their own
TRT / DSTRB records, and kNN sees the same sets, so comparisons are fair, but the absolute numbers are not satellite-only.

Remaining design concern: pair losses average over K sampled histories, which smears the jumps. A MIL soft-min over histories by fit to v2
(training only) is a possible next step.

**Final model after the correctness review (m_final_s0 = model.pt, 2026-10-22). gv_verify 45 / 45, gv_verify_upstream 5 / 5 PASS.**
- **Reconstruction** (held-out): slope 0.91–0.98, corr 0.96–0.98.
- **Dynamics**, held-out pairs (s0 / s1):
  - undisturbed: err 0.148 / 0.155 vs zero 0.172 (corr 0.68 / 0.65);
  - cut: err 0.639 / 0.703 vs zero 0.586 (corr 0.55 / 0.50; predicted −0.25 / −0.35 vs observed −0.43);
  - other events: 0.345 / 0.365 vs 0.197 (location-limited);
  - synthetic clear-cut ×0.07, fire ×0.85.
- **Landscape prior:** TPH 0.645 / BA 0.447 / QMD 0.326 vs null 0.638 / 0.464 / 0.341 and kNN 0.703 / 0.454 / 0.357. Bray 0.582 (kNN 0.579,
  null 0.743); dominant species 0.412 (kNN 0.367).
- **Mapping path, visits 2022–25** (n = 1,073): TPH 0.607 / BA 0.424 / QMD 0.299 vs null 0.608 / 0.454 / 0.327 and kNN 0.674 / 0.421 / 0.354;
  Bray 0.575 (kNN 0.574).
- **B localisation:** chance (0.046 vs 0.043).
- **Baker stand map** (Forest Explorer layer, GEN_SRC 'v3 2026-10-22'), harvested vs forest stands:
  - B1: 39.8 vs 66.0 ft²/ac; corr with TreeMap BA 0.31;
  - B0: 42.4 vs 67.7.
- The numbers are within seed noise of the composition model (m_comp). The review fixes changed correctness, not skill.

**Location ablation (g06 EVAL_LOC, 2026-10-22).** Same model and same held-out pairs; only WHERE the plot's disturbance history is taken from
varies:
- public = the footprint at the public coordinate;
- uniform = all disk footprints;
- history = FIA-consistent footprints (the default).

Change error (zero-change baseline in parentheses), seeds 0 / 1:

| | cut (0.586) | other events (0.197) | undisturbed (0.172) |
|---|---|---|---|
| history | 0.639 / 0.703 | 0.345 / 0.365 | 0.148 / 0.155 |
| public | 0.714 / 0.681 | 0.219 / 0.234 | 0.162 / 0.170 |
| uniform | **0.597 / 0.599** | **0.201 / 0.208** | **0.147 / 0.154** |

FIA-record-guided placement does NOT reduce held-out error. For event pairs it is WORSE than not using the records at all (uniform), and no
placement beats zero change on event pairs.

Likely cause: the matching is permissive. Any pixel of the 90 m footprint with any compatible class, incl. 'unknown' and Hansen loss, in the
±1-yr window counts as a match. It therefore selects footprints with severe evidence (e.g. Hansen loss for FIA weather / insect records, which
are usually mild), and the model then predicts large declines.
