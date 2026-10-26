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
