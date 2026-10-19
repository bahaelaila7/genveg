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
