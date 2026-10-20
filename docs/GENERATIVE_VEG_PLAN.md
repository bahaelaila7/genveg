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
