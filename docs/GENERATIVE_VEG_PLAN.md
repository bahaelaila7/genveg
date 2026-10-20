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
