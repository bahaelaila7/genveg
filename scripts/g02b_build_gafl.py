"""Generative-vegetation prototype v2 (all GA+FL plots), training table. Same content as g02_build_proto.py but:
  visits   = every (plot, MEASDATE) of curated_trees (common.curated_db(), read-only) for GA+FL since 2000 with a forested condition (tree records on FIADB COND
             COND_STATUS_CD 1 only: non-forest conditions of mixed plots and wholly non-forest visits are dropped), single-visit plots included; live trees with
             estimated_age, DIA, HT, CR, TPA_UNADJ (the ages / filters of the Pan interval product); species key = top-K species, else _S/_H.
             A curated visit is INCOMPLETE and left out (G46.1 / G51.1) when it keeps fewer than half of FIADB's live tallied trees (FIADB >= 3), less
             than half of their live TPA (saplings expand to 75 TPA, subplot trees to 6: A15-1), or none of them (FIADB >= 1: bare only when FIADB is).
             Curated trees carry the CANONICAL plot key of the curation chain (D29); they are joined to FIADB on their raw key (unitcd_raw /
             countycd_raw / plot_raw), every FIADB visit is read under the canonical key of its PLOT CN, and coordinates are the curated PLOT's
             latest visit (the D28 location) (G51.2).
             Densities are per FORESTED area: TPA_UNADJ / (forest share of the subplot (DIA >= 5 in) or microplot (saplings) area, FIADB COND
             SUBPPROP_UNADJ / MICRPROP_UNADJ of COND_STATUS_CD 1), so a half-forested plot is not read as a half-stocked stand
  pairs    = consecutive visits of a plot
  cands    = up to NCAND footprints per plot (stratified: high-prior stratum + random fill), features as in g02 (AlphaEarth 2022/25 3x3 means,
             disturbance history 2000-2025 by class + severity + Hansen at the EVENT's calendar year, centre-pixel soils)
  prior    = FIA-dated events (FIADB COND of the forested conditions, codes 1-3) matched BY TYPE to candidate evidence in the event year (±1; planting /
             regeneration -2..+1) → × the kind's likelihood ratio (EVK lr, measured per region by g09; cutting matched by a LANDFIRE harvest class ×2 more); negative evidence: a remeasurement whose visit carries no FIA
             code that could hide a stand-replacing event (fia_records) makes candidate stand-replacing evidence (harvest class or Hansen loss) inside
             that period ×0.25 (the stand age at the first visit is NOT used: audit r5 M4). Kept candidates carry prior × inverse inclusion weight (each stands for incl_w footprints); hi = footprint in the full disk's
             high-prior set (evidence weight >= that of one detectable match, hi_weight; G51.7); plots carry n_hi / mass_hi on the full disk (localisation)
  conc     = per plot, prior mass on its top 10 % of footprints (localisation confidence; curriculum)
Mapping set: Baker stands as in g02. Output data/proto/gafl/…
Region switches (docs/GENVEG_V4_R4.md §B): REGION = FIA STATECDs (default 12,13 = GA+FL); AEF90=1 reads the Region 4 rasters of r4_grid.py /
r4_aef90.py / r4_rasters.py (30 m land / disturbance / Hansen on grid_r4, AlphaEarth footprint means and soils at 90 m on grid_r4_90): plot windows
start on a multiple of 3 so footprint centres are the 90 m block centres, and the AlphaEarth / soil features are the 90 m pixel of the block.
The evidence table is per region (EVK_TABLES, G46.4): a region without an entry is refused unless EVK_PROVISIONAL=1 (G51.6: builds with the GA+FL
table, marked `evk_provisional`, for g09 to derive the region's table; training / evaluation stages refuse it). meta.json records the table used
(evk, evk_hash, evk_lr_method), the evidence end (evid_end), a build id (build_id, new per run), the code hash taken at the start (code_hash, G51.5),
the input databases' fingerprints (inputs) and the vetting counts (G51.1)."""
import sys, os, json, uuid, hashlib, numpy as np, pandas as pd, duckdb, rasterio, geopandas as gpd, pyarrow as pa, pyarrow.parquet as pq
from rasterio.windows import Window
from pyproj import Transformer
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from features import ROOT, WS, SOILS, dequant
from common import visit_key_remap, curated_db, fiadb, fiadb_con, code_hash, code_files, file_fingerprint
from stand_runs import stands_run
# location prior: FIA public coordinates are perturbed within 1 mile (1.6 km) of the plot, and "the proportion perturbed more than 0.5 mi (0.8 km) is small"
# (Coulston et al., FIA perturbation studies; audit F7: a uniform disk put 75 % of the mass beyond 0.8 km): RADIAL_P of the prior mass inside 0.5 mile,
# uniform over the area within each zone
RADIAL_P = float(os.environ.get('RADIAL_P', 0.9)); R_IN = 804.672
def radial_weight(d):
    """Per-footprint location weight at distance d [m] (relative; footprints tile the disk uniformly): RADIAL_P / area within R_IN, (1 − RADIAL_P) / annulus area."""
    a_in = np.pi * R_IN ** 2; a_out = np.pi * (R ** 2 - R_IN ** 2)
    return np.where(d <= R_IN, RADIAL_P / a_in, (1 - RADIAL_P) / a_out) * a_in / RADIAL_P
R = 1600.0; NCAND = int(os.environ.get('NCAND', 256)); OUT = f"{ROOT}/data/proto/{os.environ.get('DATA', 'gafl')}"; os.makedirs(OUT, exist_ok=True)
YEARS = list(range(2000, 2026))
REGION = tuple(sorted({int(x) for x in os.environ.get('REGION', '12,13').split(',')})); ST_SQL = f"({', '.join(map(str, REGION))})"
AEF90 = os.environ.get('AEF90', '0') == '1'
CLS = {'harvest': ['Clearcut', 'Harvest'], 'thin': ['Thinning', 'Mastication', 'Other Mechanical', 'Mechanical Remove', 'Mechanical Add', 'Mechanical Unknown'],
       'fire': ['Wildfire', 'Fire', 'Prescribed Fire', 'Wildland Fire Use', 'Wildland Fire', 'Unknown/Possibly Fire'],
       'insdis': ['Insects/Disease', 'Insects', 'Disease'], 'weather': ['Weather'], 'chem': ['Herbicide', 'Chemical', 'Insecticide', 'Biological'], 'dev': ['Development'],
       'unknown': ['Unknown', 'Unknown/Fire Doubtful']}   # image-based change without an assigned cause (MIICA): most harvests in GA+FL; 'Unknown' rows whose
# description says burned (BAECV >= 95 %, burn probability >= 95 %) or 'most likely fire related' are fire (see init)
NC = len(CLS); NF = 64 + 64 + len(YEARS) * (NC + 2) + len(SOILS)
SOIL_FILL = {'clay030': 7.58, 'sand030': 83.2, 'soc030': 0.964, 'awc030': 0.052, 'ph030': 5.18, 'restrdepth': 200.0}   # GA+FL forest-land medians
if AEF90: SOIL_FILL = json.load(open(f'{ROOT}/data/r4/soils90/fill.json'))      # Region 4 forest-land medians (r4_rasters.py soilfill)
def land_path(): return f'{ROOT}/data/r4/land.tif' if AEF90 else stands_run()['land_clean']
_S = {}
def init():
    codes = pd.read_csv(f'{ROOT}/data/landfire/dist/dist_codes.csv')
    cid = {t_: i + 1 for i, (c_, ts) in enumerate(CLS.items()) for t_ in ts}
    # severity scale 1-3; the 2013-14 / 2024-25 products say 'Moderate' for Medium (audit r7 I1: 36-39 % of their disturbed pixels fell back to Low);
    # 'No Severity' (no rating assigned) = the neutral Medium; a classed code with any other label is an error, not a silent Low
    SEV = {'Low': 1, 'Unburned/Low': 1, 'Low-Medium': 1.5, 'Medium-Low': 1.5, 'Low-Moderate': 1.5, 'Moderate-Low': 1.5, 'Medium': 2, 'Moderate': 2,
           'No Severity': 2, 'Medium-High': 2.5, 'High-Medium': 2.5, 'Moderate-High': 2.5, 'High-Moderate': 2.5, 'High': 3}
    lut = {}
    for r in codes.itertuples():
        typ = str(r.DIST_TYPE).strip(); c_ = cid.get(typ, 0); de = ' '.join(str(getattr(r, 'DESCRIPTION', '')) .split() + str(getattr(r, 'DESCRIPTIO', '')).split())
        if typ == 'Unknown' and ('indicates burned' in de or 'burn probability is 95% or greater' in de or 'most likely fire' in de): c_ = cid['Wildfire']
        if 'Green' in str(r.SEVERITY): c_ = 0                    # 'Increased Green' = post-fire greening, not a disturbance
        if str(r.SEVERITY).strip() in ('Water', 'Nodata/Non-processing mask'): c_ = 0   # water / unmapped pixels inside a fire perimeter: no disturbance
        # event calendar year: the 2024 / 2025 products are fiscal-year files whose codes carry CALENDAR_YR (Oct-Dec belong to the previous year); otherwise the
        # file year — NOT DIST_YEAR: the 2008 table labels 135 of 215 codes 2009, but their pixels carry Hansen loss in 2008 (25.5 %), not 2009 (1.8 %)
        ey = int(r.CALENDAR_YR) if pd.notna(getattr(r, 'CALENDAR_YR', np.nan)) else int(r.year)
        sv_ = 'No Severity' if pd.isna(r.SEVERITY) else str(r.SEVERITY).strip()   # (the 2008 codes 910-1080 carry no severity: neutral, as 'No Severity')
        if c_ > 0 and sv_ not in SEV: raise ValueError(f'LANDFIRE {r.year} code {r.VALUE} ({typ}): unmapped severity {sv_!r}')
        lut[(int(r.year), int(r.VALUE))] = (c_, SEV.get(sv_, 0.0), ey)
    if AEF90:
        _S.update(lut=lut, land=rasterio.open(land_path()), d={y: rasterio.open(f'{ROOT}/data/r4/dist/dist_{y}.tif') for y in YEARS},
                  h=rasterio.open(f'{ROOT}/data/r4/hansen_lossyear.tif'), a22=rasterio.open(f'{ROOT}/data/aef90/aef90_2022.tif'),
                  a25=rasterio.open(f'{ROOT}/data/aef90/aef90_2025.tif'), soil={k: rasterio.open(f'{ROOT}/data/r4/soils90/{k}.tif') for k in SOILS})
        return
    _S.update(lut=lut, land=rasterio.open(stands_run()['land_clean']), d={y: rasterio.open(f'{ROOT}/data/landfire/dist/dist_{y}.tif') for y in YEARS},
              h=rasterio.open(f'{ROOT}/data/hansen/lossyear.tif'), a22=rasterio.open(f'{ROOT}/data/aef/aef_2022.tif'), a25=rasterio.open(f'{ROOT}/data/aef/aef_2025.tif'),
              soil={k: rasterio.open(f'{ROOT}/data/soils/{k}.tif') for k in SOILS})
def stand_offset(sr):
    """(dR, dC) to add to a pixel of the stand raster sr (the mapped stands' grid) for the feature grid of _S['land']: 0, 0 unless AEF90 (grid_r4, a multiple of 3 px off)"""
    if not AEF90: return 0, 0
    lt, stt = _S['land'].transform, sr.transform; fr, fc = (lt.f - stt.f) / 30, (stt.c - lt.c) / 30; dR, dC = round(fr), round(fc)
    assert abs(fr - dR) < 1e-6 and abs(fc - dC) < 1e-6 and dR % 3 == 0 and dC % 3 == 0, 'stand raster not on the 90 m lattice of the feature grid'
    return dR, dC
def feats(r0, c0, h, w, keep_all=False):
    """Footprint features of the stride-3 centres of a window (forest-land centres only; keep_all: every centre — g02j's 90 m sliver blocks)."""
    s = _S; win = Window(c0, r0, w, h)
    L = s['land'].read(1, window=win, boundless=True, fill_value=0)
    rr, cc = np.mgrid[1:h - 1:3, 1:w - 1:3]; rr, cc = rr.ravel(), cc.ravel(); ok = (L[rr, cc] >= 1) & (L[rr, cc] <= 4) | keep_all; rr, cc = rr[ok], cc[ok]
    if len(rr) == 0: return rr, cc, np.zeros((0, NF), np.float32)
    HL = s['h'].read(1, window=win, boundless=True, fill_value=0); DS = np.stack([s['d'][y].read(1, window=win, boundless=True, fill_value=0) for y in YEARS])
    if AEF90:                          # 90 m rasters: footprint (3x3 block) j of the window = 90 m pixel (r0 // 3 + j); r0, c0 multiples of 3, centres = 1 mod 3
        assert r0 % 3 == 0 and c0 % 3 == 0
        w9 = Window(c0 // 3, r0 // 3, (w + 2) // 3, (h + 2) // 3); br, bc = (rr - 1) // 3, (cc - 1) // 3
        SO = np.stack([s['soil'][k].read(1, window=w9, boundless=True, fill_value=-9999).astype(np.float32) for k in SOILS])
        for i_, k in enumerate(SOILS): SO[i_][SO[i_] <= -9000] = SOIL_FILL[k]
        F = [np.nan_to_num(dequant(s[a].read(window=w9, boundless=True, fill_value=-128))[:, br, bc]).T for a in ('a22', 'a25')]; n = len(rr)
    else:
        A2 = dequant(s['a22'].read(window=win, boundless=True, fill_value=-128)); A5 = dequant(s['a25'].read(window=win, boundless=True, fill_value=-128))
        SO = np.stack([s['soil'][k].read(1, window=win, boundless=True, fill_value=-9999).astype(np.float32) for k in SOILS])
        for i_, k in enumerate(SOILS): SO[i_][SO[i_] <= -9000] = SOIL_FILL[k]      # SOLUS nodata (-9999) -> regional median
        def m3n(X):                        # 3x3 mean over the valid AlphaEarth pixels (nodata = NaN); all-nodata -> 0
            S_ = sum(np.nan_to_num(X[:, rr + dr, cc + dc]) for dr in (-1, 0, 1) for dc in (-1, 0, 1)); N_ = sum((~np.isnan(X[:1, rr + dr, cc + dc])).astype(np.float32) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
            return (S_ / np.maximum(N_, 1)).T
        F = [m3n(A2), m3n(A5)]; n = len(rr); br, bc = rr, cc
    OH = np.zeros((n, len(YEARS), NC), np.float32); SV = np.zeros((n, len(YEARS)), np.float32)
    # disturbance evidence = FRACTION of the footprint's 3x3 pixels (90 m, as for AlphaEarth) per class and event calendar year; severity: the max
    # a pixel counts ONCE per class and event year: the fiscal-year products (2024 / 2025) repeat Oct-Dec of the previous calendar year that the
    # previous file also codes (r10 M-2: 8-10 % of those pixels counted 2/9)
    PX = np.zeros((n, len(YEARS), NC, 9), bool)
    for j, y in enumerate(YEARS):                                    # file years; events land at their calendar year
        for q_, (dr, dc) in enumerate((a_, b_) for a_ in (-1, 0, 1) for b_ in (-1, 0, 1)):
            code = DS[j, rr + dr, cc + dc]
            for k in np.flatnonzero(code > 0):
                cl, sv, ey = s['lut'].get((y, int(code[k])), (0, 0, y))
                if cl > 0 and YEARS[0] <= ey <= YEARS[-1]: PX[k, ey - YEARS[0], cl - 1, q_] = True; SV[k, ey - YEARS[0]] = max(SV[k, ey - YEARS[0]], sv)
    OH = PX.sum(-1).astype(np.float32) / 9
    HF = np.stack([sum(((HL[rr + dr, cc + dc] == y - 2000) & (HL[rr + dr, cc + dc] > 0)).astype(np.float32) for dr in (-1, 0, 1) for dc in (-1, 0, 1)) / 9
                   for y in YEARS], 1).astype(np.float32)            # lossyear 0 = no loss
    for j in range(len(YEARS)): F += [OH[:, j], SV[:, j:j + 1], HF[:, j:j + 1]]
    F.append(np.nan_to_num(SO[:, br, bc].T)); return rr, cc, np.concatenate(F, 1).astype(np.float32)
# FIA event kinds -> compatible candidate evidence (class indices of CLS; 'H' = Hansen loss), year window (relative to the FIA year), evidence weight lr
# (the likelihood ratio of the location prior) and detectability (G51.7: two fields). DETECTABLE rows (cuts / site preparation / regeneration / crown
# fire, well visible from space) are the hard constraints of the history-consistent sets and pairs (g02g / g03b: a cluster with a detectable record);
# the others (ground fire, other treatments, insects, weather, human) are weak and never a hard constraint. lr is MEASURED per region by g09 (the
# inner-zone lift of the row's classes in its window vs the plot's quiet years, EVK_LR_METHOD); detectability is the fixed design choice.
H_ = NC + 1; IX = {k: i for i, k in enumerate(CLS)}
# STRICT, data-derived matching (2026-10-22, g09_fia_landfire_association.py; replaces the permissive table where any compatible class incl. the
# generic 'unknown' change class and Hansen loss matched every record type). Evidence classes per record type are those with a clear association in
# the data (lift at the FIA year vs the plot's quiet years: cutting -> harvest ×4.4-4.8, Hansen ×2.5-2.8; site preparation / planting -> chemical
# ×7-8; natural regeneration -> fire ×5.6; fire -> fire ×4.9, Hansen ×9 (crown); wind -> weather ×15; human -> harvest ×5.6). Stand-replacing evidence
# (Hansen loss) is compatible only with records whose FIA-measured outcome is stand-replacing (site preparation / regeneration median `BA(v2)/BA(v1)`
# 0.17-0.30, cutting 0.57 with 34 % < 0.3, crown fire). Mild records (wind, insects, human, other treatments: median 0.84-1.13) never match Hansen
# or 'unknown'. The coded lr 4 (detectable) / 2 (weak) of r6-r14 overstated the evidence: the measured inner-zone lifts of the detectable rows are
# 1.7-2.6 (A15 G48.1 follow-up); the GA+FL lr below are g09's measurement on gafl_v4 (G51.7).
# T10-T40 also accept LANDFIRE 'unknown' (2026-10-22, data: 61 % of Hansen-loss footprint-years carry 'unknown' within ±1 yr vs 0.5 % 'harvest'; the
# MIICA change-detection class is how most harvests appear; g10: strict_unknown gives the best stand-replacing discrimination)
EVK_GAFL = {'T10': ([IX['harvest'], IX['thin'], IX['unknown'], H_], (-1, 1), 2.456, True),                             # cutting
       'T20': ([IX['harvest'], IX['chem'], IX['fire'], IX['unknown'], H_], (-1, 1), 2.526, True),                  # site preparation
       'T30': ([IX['harvest'], IX['chem'], IX['unknown'], H_], (-2, 1), 2.281, True),                              # artificial regeneration (after a harvest)
       'T40': ([IX['harvest'], IX['fire'], IX['unknown'], H_], (-2, 1), 2.291, True),                              # natural regeneration
       'T50': ([IX['thin'], IX['chem']], (-1, 1), 1.625, False),                                                   # other silvicultural treatment
       'Dfire': ([IX['fire']], (-1, 1), 2.082, False), 'Dcrown': ([IX['fire'], H_], (-1, 1), 2.407, True),        # ground (or unspecified) / crown fire
       # ground fire is weak: prescribed burns recur every 2-3 yr, the fire class passes ~17 % of footprints -> never a hard constraint
       'Dins': ([IX['insdis']], (-1, 1), 2.183, False), 'Dwx': ([IX['weather']], (-1, 1), 2.445, False),
       'Dhum': ([IX['harvest'], IX['dev']], (-1, 1), 1.19, False)}                                                # 'unknown' (70): not matched
# how each region's lr were obtained (meta.json `evk_lr_method`; g09 writes the measurement into its proposal). GA+FL: g09 on gafl_v4 (2026-10-27,
# n records 5415 / 1641 / 1742 / 1281 / 507 / 3091 / 13 / 420 / 1142 / 398 in the order above; Dcrown has n < 30: the pooled detectable lift)
EVK_LR_METHOD = {(12, 13): ('g09 (G51.7) on gafl_v4: inner-zone lift = (mean f + PSEUDO) / (mean base + PSEUDO) over the row\'s FIA records, f = incl_w '
                            'share of the candidate footprints within R_IN = 804.672 m showing any of the row\'s classes (> 0 of 3x3) in [year + dlo, '
                            'year + dhi], base = mean of that share over the plot\'s quiet windows (>= 4 yr from any FIA code year, >= 3 windows), '
                            'PSEUDO = 0.002; rows with < 30 records take the pooled lift of their detectability; floored at 1; rounded to 3 decimals')}
# the evidence table per region (G34 / G46.4), keyed by the SORTED tuple of the region's FIA state codes: the table is derived per region (g09); a
# region without an entry is refused by g02b (main) unless EVK_PROVISIONAL=1 (G51.6: the GA+FL table, marked provisional in meta.json, only to build
# the table g09 derives the region's entry from), and later stages read the table's EVK from its meta.json (evk_from_meta), not from here
EVK_TABLES = {(12, 13): EVK_GAFL}
EVK_PROVISIONAL = os.environ.get('EVK_PROVISIONAL', '0') == '1'
EVK = EVK_TABLES.get(REGION, EVK_GAFL if EVK_PROVISIONAL else None)      # (None outside the listed regions: g02b refuses, clusters() raises)
def hi_weight(evk):
    """the evidence weight of ONE detectable match (the smallest lr of the detectable rows): a footprint whose weight reaches it is in the plot's
    high-prior set `hi` (lr 4 under the coded r6-r14 table, as before; G51.7)"""
    return min([float(lr) for _, _, lr, det in evk.values() if det] or [np.inf])
def evk_json(evk):
    """the JSON form of an evidence table: kind -> [class indices, [dlo, dhi], lr, detectable] (meta.json `evk`; G51.7 added `detectable`)"""
    return {k: [[int(c) for c in cols], [int(w[0]), int(w[1])], float(lr), bool(det)] for k, (cols, w, lr, det) in evk.items()}
def evk_hash(evk_j):
    """sha256 of the canonical JSON (sorted keys, no spaces) of an evidence table in its JSON form (meta.json `evk_hash`)"""
    return hashlib.sha256(json.dumps(evk_j, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def evk_from_meta(meta):
    """the evidence table a training table was built with (meta.json `evk`, checked against `evk_hash`): kind -> (class indices, (dlo, dhi), lr,
    detectable). Refuses a table without one (G46.4) or whose rows lack `detectable` (built before G51.7)"""
    if 'evk' not in meta or 'evk_hash' not in meta:
        sys.exit('meta.json has no evidence table (evk / evk_hash): the table predates G46.4; rebuild it with scripts/g02b_build_gafl.py')
    if evk_hash(meta['evk']) != meta['evk_hash']: sys.exit('meta.json: evk does not match evk_hash (edited by hand?); rebuild the table')
    if any(len(r) != 4 or not isinstance(r[3], bool) for r in meta['evk'].values()):
        sys.exit('meta.json evk rows lack the detectable flag (table built before G51.7): rebuild it with scripts/g02b_build_gafl.py')
    return {k: (list(cols), (int(w[0]), int(w[1])), float(lr), bool(det)) for k, (cols, w, lr, det) in meta['evk'].items()}
PROV_KEY = b'genveg_provenance'
def to_parquet_prov(df, path, prov):
    """write df as parquet with the provenance record `prov` (JSON) in the schema metadata under PROV_KEY (G46.6: g02e / g02g / g02j outputs)"""
    tb = pa.Table.from_pandas(df); md = dict(tb.schema.metadata or {}); md[PROV_KEY] = json.dumps(prov, sort_keys=True).encode()
    pq.write_table(tb.replace_schema_metadata(md), path)
def to_parquet_prov_all(items, prov):
    """write several (df, path) with the same provenance record: every file to a temporary name first, then all renamed (A15-6e: a crash between
    writes no longer leaves one script's outputs with mixed provenance)"""
    tmp = [f'{path}.tmp{os.getpid()}' for _, path in items]
    for (df, _), t_ in zip(items, tmp): to_parquet_prov(df, t_, prov)
    for (_, path), t_ in zip(items, tmp): os.replace(t_, path)
def read_prov(path):
    """the provenance record of a parquet output (PROV_KEY in its schema metadata), None without one"""
    md = pq.read_schema(path).metadata or {}
    return json.loads(md[PROV_KEY]) if PROV_KEY in md else None
def meta_build_id(meta):
    """the build id of the table (meta.json `build_id`, G51.5); refuses a table built before G51.5"""
    if not meta.get('build_id'): sys.exit('meta.json has no build_id (table built before G51.5): rebuild it with scripts/g02b_build_gafl.py')
    return meta['build_id']
def check_prov(path, meta):
    """an input parquet written by a later data stage must belong to THIS table: its provenance build_id and evk_hash equal meta.json's (G51.5;
    A15-6d: the evk_hash was written but never read)"""
    pr = read_prov(path)
    if pr is None: sys.exit(f'{path}: no provenance record ({PROV_KEY.decode()}); rerun the stage that writes it')
    if pr.get('build_id') != meta_build_id(meta): sys.exit(f'{path}: written for table build {pr.get("build_id")}, not this table ({meta["build_id"]}); rerun its stage')
    if pr.get('evk_hash') != meta.get('evk_hash'): sys.exit(f'{path}: evk_hash {pr.get("evk_hash")} != meta.json evk_hash {meta.get("evk_hash")}; rerun its stage')
    return pr
def write_json_atomic(obj, path):
    """write JSON via a temporary file + rename (a crash never leaves a half-written meta.json)"""
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w') as f: json.dump(obj, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
NONBLOCK = {11, 21, 31, 40, 42, 43, 44, 45, 46}   # (41 beaver incl. flooding and 60 vegetation BLOCK: their stand-loss rates match the blocking codes, r13 A-2 / G37)
# |Δ forested share| >= 0.25 blocks (r7 M1 / G46.2 here, G38 in g03b, the verifier): shares are sums of quarter-subplot proportions, so a change of
# exactly 0.25 can evaluate to 0.24999999999999994; every such comparison is >= SHARE_DELTA - SHARE_TOL (A15-2 / G51.3)
SHARE_DELTA, SHARE_TOL = 0.25, 1e-9
EVID_END = 2024   # last calendar year with stand-replacing evidence: Hansen GFC-2024 lossyear ends 2024, FY2025 LANDFIRE misses Oct-Dec 2025 (r13 A-1 / G36)
# NONBLOCK = the DSTRBCD that cannot hide stand-replacing evidence: insect / disease damage to UNDERSTORY vegetation (11 / 21; 12 / 22 are damage to
# TREES and can — FIADB v9.4 / EVALIDATOR library, audit r6 I1), ground fire (31), animal (40, 42-46; 41 beaver blocks)
def ev_kind(trt, code):
    c = int(code)
    if trt: return {10: 'T10', 20: 'T20', 30: 'T30', 40: 'T40', 50: 'T50'}.get(c)
    if c in (30, 31): return 'Dfire'
    if c == 32: return 'Dcrown'
    if 10 <= c <= 22: return 'Dins'
    if 50 <= c <= 54: return 'Dwx'
    if c == 80: return 'Dhum'
    return None                                     # animal (40s), vegetation (60), geologic (90): rarely visible from space
def clusters(evs, evk=None):
    """FIA records of ONE physical event (a cut + site preparation + planting, or the same event re-reported a year apart) must not multiply:
    events whose years chain within 2 years form one cluster -> (evidence columns = union, window = union, likelihood ratio = max, has cutting,
    det = has a DETECTABLE record: the hard constraint of g02g / g03b / g06, G51.7 — no longer lr >= 4).
    evk: the evidence table (stages after g02b pass the table's own, evk_from_meta); None = this region's EVK_TABLES entry."""
    evk = EVK if evk is None else evk
    if evk is None: raise SystemExit(f'no evidence table (EVK_TABLES) for region {REGION}: pass the table\'s own (evk_from_meta) or add the entry (G34 / G46.4)')
    out = []
    for kind, yv in sorted(evs, key=lambda e: e[1]):
        cols, (dlo, dhi), lr, det = evk[kind]
        if out and yv - out[-1]['ylast'] <= 2:
            c = out[-1]; c['cols'] |= set(cols); c['lo'] = min(c['lo'], yv + dlo); c['hi'] = max(c['hi'], yv + dhi); c['lr'] = max(c['lr'], lr); c['cut'] |= kind == 'T10'; c['ylast'] = yv
            c['det'] |= bool(det)
        else: out.append(dict(cols=set(cols), lo=yv + dlo, hi=yv + dhi, lr=lr, cut=kind == 'T10', ylast=yv, det=bool(det)))
    return out
def prior_weights(per, evs, negs):
    """per [n, years, NC+2] candidate evidence (footprint fractions); evs [(kind, year)]; negs [(y_lo, y_hi)] periods without stand-replacing
    disturbance. Returns the UNNORMALISED weight (1 = no evidence either way)."""
    w = np.ones(len(per)); Y0 = YEARS[0]
    for c in clusters(evs):
        lo, hi = max(c['lo'] - Y0, 0), min(c['hi'] - Y0, len(YEARS) - 1) + 1
        if hi <= lo: continue
        w *= np.where(per[:, lo:hi][:, :, sorted(c['cols'])].sum(-1).any(1), c['lr'], 1.0)
        if c['cut']: w *= np.where(per[:, lo:hi, IX['harvest']].any(1), 2.0, 1.0)
    sr = np.maximum(per[:, :, IX['harvest']], per[:, :, H_]) >= 0.5                 # stand-replacing evidence on most of the footprint
    for ylo, yhi in negs:
        lo, hi = max(ylo - Y0, 0), min(yhi - Y0, len(YEARS) - 1) + 1
        if hi > lo: w *= np.where(sr[:, lo:hi].any(1), 0.25, 1.0)
    return w
HI_W = hi_weight(EVK) if EVK is not None else np.inf
def plot_job(a):
    """candidates of one plot. Elevated-prior stratum = net positive evidence (weight > 1; under the coded lr 2 / 4 table the same set as the
    r6-r14 weight > 1.5, whose weights were powers of 2: G51.7 keeps the sampling design independent of the measured lr)"""
    pid, x, y, evs, negs = a; s = _S; row, col = s['land'].index(x, y); RP = int(R // 30) + 2
    if AEF90: r0, c0 = 3 * ((row - RP) // 3), 3 * ((col - RP) // 3); hw = 3 * ((2 * RP + 1 + 2) // 3) + 3     # 90 m-aligned window covering the disk
    else: r0, c0, hw = row - RP, col - RP, 2 * RP + 1
    rr, cc, F = feats(r0, c0, hw, hw)
    d = np.hypot(rr + r0 - row, cc + c0 - col) * 30; m = d <= R; F, d = F[m], d[m]
    ar, ac = (rr[m] + r0).astype(np.int32), (cc[m] + c0).astype(np.int32)        # absolute grid row / col of each footprint centre (verifiability)
    if len(d) == 0: return pid, None
    per = F[:, 128:128 + len(YEARS) * (NC + 2)].reshape(len(d), len(YEARS), NC + 2)
    wr = prior_weights(per, evs, negs); w = wr * radial_weight(d); w = w / w.sum()      # evidence x radial location prior
    conc = float(np.sort(w)[::-1][:max(1, len(w) // 10)].sum())
    hi = wr >= HI_W * (1 - 1e-9); n_hi, mass_hi = int(hi.sum()), float(w[hi].sum())   # localisation on the full disk: weight >= one detectable match (hi_weight)
    iw = np.ones(len(d)); keep = np.arange(len(d))
    if len(d) > NCAND:     # stratified sample: elevated-prior stratum (all, or a uniform subset of 3/4 NCAND) + uniform fill of the rest
        rg = np.random.default_rng(pid); up = np.flatnonzero(wr > 1 + 1e-9); rest = np.setdiff1d(np.arange(len(d)), up)   # net positive evidence vs the rest
        k_up = min(len(up), (3 * NCAND) // 4) if len(rest) > 0 else min(len(up), NCAND); ku = rg.choice(up, k_up, replace=False) if k_up < len(up) else up
        nfill = min(NCAND - len(ku), len(rest)); kf = rg.choice(rest, nfill, replace=False) if nfill > 0 else np.zeros(0, int)
        iw[ku] = len(up) / max(len(ku), 1); iw[kf] = len(rest) / max(nfill, 1); keep = np.r_[ku, kf]
    pk = w[keep] * iw[keep]; pk = pk / pk.sum()          # each kept candidate stands for incl_w footprints of the disk
    return pid, dict(F=F[keep].astype(np.float16), d=d[keep].astype(np.float32), w=pk.astype(np.float32), iw=iw[keep].astype(np.float32), hi=hi[keep],
                     ar=ar[keep], ac=ac[keep], conc=conc, n_hi=n_hi, mass_hi=mass_hi, n=len(d))
def decimal_year(md):
    """visit date -> decimal year at mid-day: year + (day of year - 0.5) / days in year (never reaches the next integer; the old year + doy / 366
    put leap-year Dec-31 visits at the next year)."""
    md = pd.to_datetime(md); return md.dt.year + (md.dt.dayofyear - 0.5) / np.where(md.dt.is_leap_year, 366.0, 365.0)
def key_remap(cur, fc):
    """the FIADB visits (PLOT CN) whose curated plot key differs from FIADB's (G51.2): the curation chain (D29) files the later segments of a plot
    under its canonical key, and the curated PLOT table keeps every CN with that key. -> DataFrame cn, st, u, c, p (canonical), ru, rc, rp (FIADB's
    raw key). Empty for a curated DB without remapped plots (the old slim_r4)."""
    return visit_key_remap(cur, ST_SQL)
def canon(df, remap):
    """FIADB rows (with their PLOT CN `cn`) under the curated canonical plot key: the remapped CNs take the canonical key, every other row keeps
    FIADB's (G51.2: events, negative evidence, visits of every chain segment belong to the canonical plot)"""
    if remap is None or not len(remap): return df
    m = dict(zip(remap.cn, zip(remap.u, remap.c, remap.p))); hit = df.cn.astype(str).map(m)
    if hit.notna().any():
        df = df.copy(); ix = hit.notna().values
        for j, k in enumerate(('u', 'c', 'p')): df.loc[ix, k] = [t[j] for t in hit[ix]]
    return df
def fiadb_visits(fc, remap=None):
    """FIADB visits (read-only) with a forested condition: forested shares of the WHOLE plot (fsub / fmic: TPA_UNADJ expands a tree to the whole plot area,
    so trees per forested area = Σ TPA / f1s — audit 2026-10-26 F8; the earlier f1s / (1 − n5s) made densities low by (1 − n5s)) and of the SAMPLED area
    (fsub_s, non-sampled COND_STATUS_CD 5 excluded: the visit filter), stand origin, previous measurement year (PREV_PLT_CN), live-tree count;
    prev_f1s = the forested share of the whole plot at that previous measurement (NaN when none of its conditions carries SUBPPROP_UNADJ); live
    TPA (Σ TPA_UNADJ of the same live tallied trees: the G51.1 vetting). Keys are the canonical ones (canon)."""
    FV = fc.execute(f"""select p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, (case when p.MEASMON between 1 and 12 and p.MEASDAY between 1 and 31 then make_date(p.MEASYEAR::int, p.MEASMON::int, p.MEASDAY::int) end) md, p.MEASYEAR my, p.CN cn,
                               any_value(q.MEASYEAR) prev_my, any_value(q.PLOT_STATUS_CD) prev_ps, any_value(qc.pf1s) prev_f1s,
                               sum(case when c.COND_STATUS_CD = 1 then c.SUBPPROP_UNADJ else 0 end) f1s, sum(case when c.COND_STATUS_CD = 5 then c.SUBPPROP_UNADJ else 0 end) n5s,
                               sum(case when c.COND_STATUS_CD = 1 then c.MICRPROP_UNADJ else 0 end) f1m, sum(case when c.COND_STATUS_CD = 5 then c.MICRPROP_UNADJ else 0 end) n5m,
                               max(case when c.COND_STATUS_CD = 1 then c.STDORGCD end) org
                        from PLOT p join COND c on c.PLT_CN = p.CN left join PLOT q on q.CN = p.PREV_PLT_CN
                        left join (select PLT_CN, sum(case when COND_STATUS_CD = 1 then SUBPPROP_UNADJ else 0 end) pf1s from COND
                                   group by 1 having count(SUBPPROP_UNADJ) > 0) qc on qc.PLT_CN = q.CN
                        where p.STATECD in {ST_SQL} and p.MEASYEAR >= 2000 group by all having f1s > 0""").df()
    # live TALLIED trees on forested conditions (the bare-visit test; r13 A-6 / G40: every live row of the plot was counted, any condition / TPA)
    nlive = fc.execute(f"""select t.PLT_CN cn, count(*) n, sum(t.TPA_UNADJ) tpa from TREE t join COND c on c.PLT_CN = t.PLT_CN and c.CONDID = t.CONDID
                            where t.STATECD in {ST_SQL} and t.STATUSCD = 1 and t.TPA_UNADJ > 0 and t.SUBP <= 4 and c.COND_STATUS_CD = 1 group by 1""").df()
    FV = FV.merge(nlive, on='cn', how='left'); FV['n'] = FV.n.fillna(0); FV['tpa'] = FV.tpa.fillna(0.0); FV['md'] = pd.to_datetime(FV.md)
    FV = canon(FV, remap); dup = FV.duplicated(['st', 'u', 'c', 'p', 'md'], keep=False)
    if dup.any(): sys.exit(f'two FIADB visits of one canonical plot on the same date (key remap): {FV[dup][["cn", "st", "u", "c", "p", "md"]].head(6).to_dict("records")}')
    FV['fsub_s'] = (FV.f1s / (1 - FV.n5s).clip(lower=1e-6)).clip(upper=1.0)
    FV['fsub'] = FV.f1s.clip(upper=1.0); FV['fmic'] = np.where(FV.f1m > 0, FV.f1m.clip(upper=1.0), FV.fsub)
    return FV
def fiadb_events(fc, remap=None):
    """TRT / DSTRB codes of the forested conditions of every FIADB visit since 2000 (read-only), under the canonical keys (canon)."""
    FE = fc.execute(f"""select p.CN cn, p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, (case when p.MEASMON between 1 and 12 and p.MEASDAY between 1 and 31 then make_date(p.MEASYEAR::int, p.MEASMON::int, p.MEASDAY::int) end) md,
                               c.TRTCD1, c.TRTYR1, c.TRTCD2, c.TRTYR2, c.TRTCD3, c.TRTYR3, c.DSTRBCD1, c.DSTRBYR1, c.DSTRBCD2, c.DSTRBYR2, c.DSTRBCD3, c.DSTRBYR3
                        from PLOT p join COND c on c.PLT_CN = p.CN where p.STATECD in {ST_SQL} and p.MEASYEAR >= 2000 and c.COND_STATUS_CD = 1""").df()
    FE['md'] = pd.to_datetime(FE.md); return canon(FE, remap)
def fia_records(FE, FV, P, key, stats=None, evid_end=EVID_END):
    """FIA events per plot {pid: {(kind, year)}} from every forested condition; negative evidence {pid: [(y_lo, y_hi)]} = measurement periods whose
    visit carries no code that could hide stand-replacing evidence: (previous measurement + 1 .. visit - 1); for the first visit since 2000 the
    previous measurement (PREV_PLT_CN, periodic inventories) or, for a new plot, FIA's 5-year lookback; blocked = {(plot key, visit date)} with such a code.
    A period is CLIPPED to the evidence window (y_hi <= evid_end, G36 / G46.5; later stages pass the table's meta.json evid_end, G51.4); one lying
    wholly after it is empty and not listed.
    Shared by g02b (location prior) and g03b (FIA-anchored remeasurement pairs). stats (a dict, optional): filled with counts of the blocking rules."""
    st_ = {'share_prev_forested': 0, 'share_prev_measurement': 0} if stats is None else stats
    st_.setdefault('share_prev_forested', 0); st_.setdefault('share_prev_measurement', 0)
    pidk = dict(zip(map(tuple, P[key].values), P.pid)); evs, negs = {}, {}
    cols = [('TRTCD1', 'TRTYR1', 1), ('TRTCD2', 'TRTYR2', 1), ('TRTCD3', 'TRTYR3', 1), ('DSTRBCD1', 'DSTRBYR1', 0), ('DSTRBCD2', 'DSTRBYR2', 0), ('DSTRBCD3', 'DSTRBYR3', 0)]
    blocked = set()
    for r in FE.itertuples():
        k_ = (r.st, r.u, r.c, r.p)
        for cc_, yy_, trt in cols:
            code, yr = getattr(r, cc_), getattr(r, yy_)
            if pd.notna(code) and code > 0:
                if trt or int(code) not in NONBLOCK: blocked.add((k_, r.md))
                kind = ev_kind(trt, code)
                if kind and pd.notna(yr) and 2000 <= yr <= 2025 and k_ in pidk: evs.setdefault(pidk[k_], set()).add((kind, int(yr)))
    for k_, g_ in FV.sort_values('md').groupby(key):
        if k_ not in pidk: continue
        yrs = g_.my.values.astype(int); mds = g_.md.values; pm = g_.prev_my.values; pps = g_.prev_ps.values; nl = []; fs_ = g_.f1s.values
        pfs = g_.prev_f1s.values
        for j in range(len(yrs)):
            vk_ = (k_, pd.Timestamp(mds[j])); was = vk_ in blocked
            # a forest -> non-forest (or non-sampled) conversion of >= 1/4 of the plot since the previous forested visit carries no TRT / DSTRB code
            # (codes live on forested conditions only): it can hide a clearing, so it blocks like a hiding code (audit r7 M1)
            if j > 0 and abs(fs_[j - 1] - fs_[j]) >= SHARE_DELTA - SHARE_TOL:   # (either direction: a re-delineation changes the per-forested-area densities too —
                blocked.add(vk_); st_['share_prev_forested'] += not was; was = True   # median BA log change −0.16 vs +0.14 for unchanged shares; r8 M2)
            # the same against FIA's PREVIOUS measurement (PREV_PLT_CN; r14 A14-2 / G46.2): a pre-2000 predecessor of the first visit since 2000, and
            # an entirely non-forest one (PLOT_STATUS_CD 2: forested share 0); a non-sampled one (3) or one without condition shares says nothing
            q_ = 0.0 if pd.notna(pps[j]) and int(pps[j]) == 2 else (float(pfs[j]) if pd.notna(pps[j]) and int(pps[j]) == 1 and pd.notna(pfs[j]) else None)
            if q_ is not None and abs(q_ - fs_[j]) >= SHARE_DELTA - SHARE_TOL: blocked.add(vk_); st_['share_prev_measurement'] += not was
            if vk_ in blocked: continue
            # the visit's codes cover the period since FIA's PREVIOUS measurement (PREV_PLT_CN, any status): a non-forest / non-sampled visit in between
            # shortens it (audit r6 M1); the later of that and the previous forested visit is used
            # a NON-SAMPLED previous measurement (PLOT_STATUS_CD 3) vouches for nothing: the visit's own lookback applies (r13 A-5 / G39); the FIA lookback
            # is 5 yr before the visit date, conservatively the calendar years yr-4 .. yr-1 (A-4)
            if pd.notna(pm[j]) and not (pd.notna(pps[j]) and int(pps[j]) == 3): a_ = max(int(pm[j]), yrs[j - 1] if j > 0 else -9999) + 1
            elif j > 0: a_ = max(yrs[j] - 4, yrs[j - 1] + 1) if pd.notna(pps[j]) and int(pps[j]) == 3 else yrs[j - 1] + 1
            else: a_ = yrs[0] - 4
            b_ = min(yrs[j] - 1, evid_end)                                     # no negative evidence beyond the evidence window (G36)
            if b_ >= a_: nl.append((a_, b_))
        negs[pidk[k_]] = nl
    return evs, negs, blocked
if __name__ == '__main__':
    # provenance taken at the START (G51.5 / A15-6f): code hash of this script + its local imports, a new build id, the input databases' fingerprints
    CODE_HASH = {'g02b': code_hash(os.path.abspath(__file__))}; CODE_FILES = {'g02b': [os.path.basename(f) for f in code_files(os.path.abspath(__file__))]}
    BUILD_ID = uuid.uuid4().hex
    if EVK_PROVISIONAL and REGION in EVK_TABLES:
        sys.exit(f'EVK_PROVISIONAL=1 with region {REGION}, which has its own EVK_TABLES entry: the provisional GA+FL table is only for a region without one (G51.6)')
    if EVK is None:
        sys.exit(f'no FIA-record -> evidence table (EVK_TABLES: classes / windows / likelihood ratios) for region {REGION}: the GA+FL table was derived '
                 'by g09 on GA+FL; re-derive it for the region and add its EVK_TABLES entry first (docs/DECISIONS.md G34 / G46.4: in KY / TN the 2023 '
                 '"unknown" class barely moves AlphaEarth), or bootstrap with EVK_PROVISIONAL=1 (G51.6: a provisional table for g09 only)')
    if AEF90 and REGION == (12, 13):
        sys.exit('AEF90=1 with REGION 12,13: the GA+FL evidence table and table builds are 30 m; the 90 m Region 4 rasters need a Region 4 table (G46.4)')
    INPUTS = {'curated_db': file_fingerprint(curated_db()), 'fiadb': file_fingerprint(fiadb())}
    print(f'build {BUILD_ID}; code hash {CODE_HASH["g02b"][:16]} over {CODE_FILES["g02b"]}; evidence table '
          f'{"PROVISIONAL (GA+FL rows, G51.6)" if EVK_PROVISIONAL else str(REGION)}', flush=True)
    con = duckdb.connect(curated_db(), read_only=True)
    # curated trees: canonical plot key (st, u, c, p) + the raw FIADB key of the record (ru, rc, rp; = the canonical one in a curated DB without them)
    tcols = {r[0] for r in con.execute('describe curated_trees').fetchall()}; RAWK = {'unitcd_raw', 'countycd_raw', 'plot_raw'} <= tcols
    rawsel = 't.unitcd_raw ru, t.countycd_raw rc, t.plot_raw rp' if RAWK else 't.UNITCD ru, t.COUNTYCD rc, t.PLOT rp'
    t = con.execute(f"""select t.STATECD st, t.UNITCD u, t.COUNTYCD c, t.PLOT p, {rawsel}, t.SUBP sb, t.MEASDATE md, t.DIA, t.HT, t.CR, t.TPA_UNADJ tpa, t.estimated_age age, r.SPECIES_SYMBOL sym, r.SFTWD_HRDWD grp,
                              t.STATUSCD status, t.CONDID cond, t.cond_stdage stdage, t.TRTCD1, t.TRTYR1, t.TRTCD2, t.TRTYR2, t.DSTRBCD1, t.DSTRBYR1, t.DSTRBCD2, t.DSTRBYR2, t.cond_stdorgcd org
                       from curated_trees t join REF_SPECIES r on r.SPCD = t.SPCD
                       where t.STATECD in {ST_SQL} and t.MEASDATE >= '2000-01-01' order by t.STATECD, t.UNITCD, t.COUNTYCD, t.PLOT, t.MEASDATE, t.SUBP, t.TREE""").df()
    t['live'] = t.status == 1          # every visit is kept (bare / just-harvested visits = no live tree = empty cohort table); cohorts from live trees only
    t['md'] = pd.to_datetime(t.md); key = ['st', 'u', 'c', 'p']
    fc = fiadb_con()
    REMAP = key_remap(con, fc); remapped_ = (t.u != t.ru) | (t.c != t.rc) | (t.p != t.rp)
    KEYS_INFO = dict(curated_raw_keys=bool(RAWK), remapped_cns=int(len(REMAP)), remapped_plots=int(len(REMAP[key].drop_duplicates())),
                     remapped_records=int(remapped_.sum()), remapped_live_records=int((remapped_ & t.live).sum()),
                     remapped_visits=int(len(t[remapped_][key + ['md']].drop_duplicates())))
    print(f"curated keys: raw-key columns {'present' if RAWK else 'absent (keys = FIADB keys)'}; remapped FIADB visits (CN) {KEYS_INFO['remapped_cns']} of "
          f"{KEYS_INFO['remapped_plots']} plots; curated records on remapped visits {KEYS_INFO['remapped_records']} (live {KEYS_INFO['remapped_live_records']}, "
          f"visits {KEYS_INFO['remapped_visits']}), joined to FIADB on their raw key (G51.2)", flush=True)
    CS = fc.execute(f"""select p.STATECD st, p.UNITCD ru, p.COUNTYCD rc, p.PLOT rp, (case when p.MEASMON between 1 and 12 and p.MEASDAY between 1 and 31 then make_date(p.MEASYEAR::int, p.MEASMON::int, p.MEASDAY::int) end) md, c.CONDID cond, c.COND_STATUS_CD cs
                        from PLOT p join COND c on c.PLT_CN = p.CN where p.STATECD in {ST_SQL} and p.MEASYEAR >= 2000""").df(); CS['md'] = pd.to_datetime(CS.md)
    t = t.merge(CS, on=['st', 'ru', 'rc', 'rp', 'md', 'cond'], how='left'); n0_ = len(t)
    print(f"tree records on non-forest / unmatched conditions dropped: {int((t.cs != 1).sum())} of {n0_} (live {int(((t.cs != 1) & (t.status == 1)).sum())}; "
          f"on remapped visits {int(((t.cs != 1) & ((t.u != t.ru) | (t.c != t.rc) | (t.p != t.rp))).sum())})", flush=True)
    t = t[t.cs == 1].reset_index(drop=True)            # the cohort table describes the FORESTED conditions (visits with only non-forest conditions drop out)
    FV = fiadb_visits(fc, REMAP); FE = fiadb_events(fc, REMAP)
    kp_ = t[key].drop_duplicates()             # plots of the curated table (the scope of bare visits / all-bare plots: before the vetting below)
    # INCOMPLETE curated visits (r14 A14-1 / G46.1, G51.1): n_cur / tpa_cur = curated live tallied trees on the forested conditions and their live TPA,
    # n_fia / tpa_fia = FIADB's (FV.n / FV.tpa: the same tree population). A visit leaves the table (no tokens, no pair endpoint, no bare visit) by
    #   count:         n_fia >= 3 and n_cur < n_fia / 2 (curated kept only dead records or a small subset; G46.1)
    #   tpa:           tpa_cur < tpa_fia / 2 (A15-1: a microplot sapling expands to 74.965 TPA, a subplot tree to 6.018, so a curated visit missing
    #                  the saplings passes the count rule)
    #   bare_mismatch: n_cur = 0 and n_fia >= 1 (A15-3: a curated visit is bare only when FIADB is bare too, as the FIADB-added bare visits of G40)
    # It stays a FIADB visit for the FIA records and negative evidence (fia_records reads FIADB, not the table)
    lv_ = t[t.live & (t.tpa > 0) & (t.sb <= 4)]
    nc_ = lv_.groupby(key + ['md']).agg(n_cur=('tpa', 'size'), tpa_cur=('tpa', 'sum')).reset_index()
    vv_ = t[key + ['md']].drop_duplicates().merge(nc_, on=key + ['md'], how='left').merge(FV[key + ['md', 'n', 'tpa', 'fsub', 'fsub_s']], on=key + ['md'], how='inner')
    vv_['n_cur'] = vv_.n_cur.fillna(0); vv_['tpa_cur'] = vv_.tpa_cur.fillna(0.0)
    RULES_ = {'count': (vv_.n >= 3) & (vv_.n_cur < 0.5 * vv_.n),
              'tpa': vv_.tpa_cur < 0.5 * vv_.tpa * (1 - 1e-9),          # (relative tolerance: an exact half of float TPA sums is not incomplete)
              'bare_mismatch': (vv_.n_cur == 0) & (vv_.n >= 1)}
    vv_['inc'] = RULES_['count'] | RULES_['tpa'] | RULES_['bare_mismatch']; inc_ = vv_[vv_.inc]
    tabm_ = vv_.fsub_s >= 0.1; inc_tab = inc_[inc_.fsub_s >= 0.1]   # the ones that would otherwise be table visits (forested share filter below)
    kept_ = vv_[tabm_ & ~vv_.inc]
    VET = dict(incomplete_visits=int(len(inc_tab)), incomplete_visits_by_state={str(int(k)): int(v) for k, v in inc_tab.st.value_counts().sort_index().items()},
               incomplete_curated_visits=int(len(inc_)), vetted_curated_visits=int(len(vv_)), incomplete_no_live=int((inc_tab.n_cur == 0).sum()),
               incomplete_by_rule={k: int((m_ & tabm_).sum()) for k, m_ in RULES_.items()},
               incomplete_by_rule_only={k: int((m_ & tabm_ & ~np.logical_or.reduce([m2 for k2, m2 in RULES_.items() if k2 != k])).sum()) for k, m_ in RULES_.items()},
               tpa_share_kept=float(kept_.tpa_cur.sum() / max(kept_.tpa.sum(), 1e-9)), n_share_kept=float(kept_.n_cur.sum() / max(kept_.n.sum(), 1e-9)))
    print(f"incomplete curated visits dropped (G51.1): {VET['incomplete_visits']} table visits (no curated live tree {VET['incomplete_no_live']}; all curated "
          f"visits with a forested FIADB condition {len(inc_)} of {len(vv_)}); by rule (overlapping) {VET['incomplete_by_rule']}, by that rule alone "
          f"{VET['incomplete_by_rule_only']}; by state {VET['incomplete_visits_by_state']}; kept visits: curated / FIADB live TPA {VET['tpa_share_kept']:.4f}, "
          f"trees {VET['n_share_kept']:.4f}", flush=True)
    t = t.merge(inc_[key + ['md']].assign(_inc=1), on=key + ['md'], how='left'); t = t[t._inc.isna()].drop(columns='_inc').reset_index(drop=True)
    # top species among LIVE trees (of the vetted visits)
    cnt = t[t.live].sym.value_counts(); top = list(cnt.index[:int(os.environ.get('TOPK', 16))]); KEYS = sorted(top) + ['_H', '_S']; kix = {k: i for i, k in enumerate(KEYS)}
    t['key'] = [kix[s] if s in top else kix['_' + (g if g in ('S', 'H') else 'H')] for s, g in zip(t.sym, t.grp)]
    # visits = forested visits with curated tree records + truly bare forested visits (no live tree in FIADB, absent from the curated table) of the same plots
    vt = t[key + ['md']].drop_duplicates()
    # + FIADB plots that are bare at EVERY visit, inside the curated table's L3 ecoregion scope (r13 A-7 / G40: recently cleared stands were missing);
    # data_plot_eco is per FIADB key: a remapped segment's row also counts for its canonical plot
    eco_ = fc.execute(f"select statecd st, unitcd u, countycd c, plot p, epa_l3 l3 from data_plot_eco where statecd in {ST_SQL}").df()
    if len(REMAP):
        rk_ = REMAP[['st', 'ru', 'rc', 'rp', 'u', 'c', 'p']].drop_duplicates()
        eco_ = pd.concat([eco_, eco_.rename(columns={'u': 'ru', 'c': 'rc', 'p': 'rp'}).merge(rk_, on=['st', 'ru', 'rc', 'rp'])[key + ['l3']]]).drop_duplicates()
    l3s_ = set(kp_.merge(eco_, on=key).l3.dropna()); ab_ = FV.groupby(key).n.max(); ab_ = ab_[ab_ == 0].reset_index()[key].merge(eco_, on=key)
    ab_ = ab_[ab_.l3.isin(l3s_)][key].drop_duplicates(); print(f"all-bare FIADB plots added (L3 scope {len(l3s_)}): {len(ab_)}", flush=True)
    bare = FV[FV.n == 0].merge(pd.concat([kp_, ab_]).drop_duplicates(), on=key).merge(vt.assign(_in=1), on=key + ['md'], how='left'); bare = bare[bare._in.isna()][key + ['md']]
    vis = pd.concat([vt, bare]).merge(FV[key + ['md', 'fsub', 'fsub_s', 'fmic', 'org']], on=key + ['md'], how='left')
    print(f"visits: with curated trees {len(vt)}, bare FIADB visits added {len(bare)}, without a forested FIADB condition {int(vis.fsub.isna().sum())}", flush=True)
    lowf = vis.fsub_s < 0.1; print(f"visits with forested share < 0.1 of the sampled area dropped: {int(lowf.sum())}", flush=True)
    vis = vis[vis.fsub.notna() & ~lowf].sort_values(key + ['md']).reset_index(drop=True); vis['vid'] = np.arange(len(vis))
    vis['year'] = decimal_year(vis.md); t = t.merge(vis[key + ['md', 'vid']], on=key + ['md'])
    t = t.merge(vis[['vid', 'fsub', 'fmic']], on='vid')
    t['D'] = t.DIA * 2.54; t['H'] = t.HT * 0.3048; t['CRf'] = t.CR / 100; t['A'] = t.age.astype(float)
    t['W'] = t.tpa * 2.47105 / np.where(t.DIA < 5.0, t.fmic, t.fsub)              # trees / forested ha (microplot saplings: microplot forest share)
    tl = t[t.live & (t.DIA > 0) & (t.tpa > 0) & (t.A > 0) & (t.H > 0)].copy()                   # cohorts from live trees
    tl['ac'] = np.clip((tl.A // 10).astype(int), 0, 15); tl['WD2'] = tl.W * tl.D ** 2; tl['WH'] = tl.W * tl.H; tl['WA'] = tl.W * tl.A
    tl['crok'] = tl.CRf.notna() & (tl.CRf > 0); tl['WCR'] = np.where(tl.crok, tl.W * tl.CRf, 0); tl['WCRW'] = np.where(tl.crok, tl.W, 0)
    g = tl.groupby(['vid', 'key', 'ac'])[['W', 'WD2', 'WH', 'WA', 'WCR', 'WCRW']].sum().reset_index()
    TK = pd.DataFrame({'vid': g.vid, 'key': g.key, 'ac': g.ac, 'N': g.W, 'QMD': np.sqrt(g.WD2 / g.W), 'H': g.WH / g.W, 'A': g.WA / g.W,
                       'CR': np.where(g.WCRW > 0, g.WCR / g.WCRW.replace(0, np.nan), np.nan)})
    pv = vis.groupby(key).ngroup(); vis['pid'] = pv
    vis['planted'] = vis.org.fillna(0).values == 1; vis = vis.drop(columns=['org'])          # FIA COND STDORGCD (forested conditions): 0 natural, 1 artificial
    pairs = vis.sort_values(['pid', 'year']).groupby('pid').apply(lambda x: pd.DataFrame({'v1': x.vid.values[:-1], 'v2': x.vid.values[1:]}), include_groups=False).reset_index(level=0)
    rl = 'GA+FL' if REGION == (12, 13) else f'states {",".join(map(str, REGION))}'
    print(f'{rl}: plots {vis.pid.nunique()} visits {len(vis)} tokens {len(TK)} pairs {len(pairs)} keys {KEYS}', flush=True)
    # plot coordinates (G51.2): the curated PLOT row of the plot's LATEST visit since 2000 (the D28 location: the curation writes it on every row of a
    # remapped plot), FIADB's latest row (canonical key) for a plot without curated PLOT rows. Annual-inventory rows only (MEASYEAR >= 2000): in Kentucky
    # 1,199 plot keys reuse the number of a 1986-88 periodic plot several km away (audit 2026-10-26 F1); the spread of a key's rows is reported
    def latest_xy(df):
        df = df.sort_values(key + ['my', 'mm', 'dd', 'iy', 'cn']); g_ = df.groupby(key)
        out = g_.tail(1)[key + ['lat', 'lon']].merge(g_.agg(la0=('lat', 'min'), la1=('lat', 'max'), lo0=('lon', 'min'), lo1=('lon', 'max'), lam=('lat', 'mean'), lom=('lon', 'mean')).reset_index(), on=key)
        out['spread_km'] = (out.la1 - out.la0) * 111.0 + (out.lo1 - out.lo0) * 111.0 * np.cos(np.radians(out.lam)); return out
    q_ = f"""select CN::varchar cn, STATECD st, UNITCD u, COUNTYCD c, PLOT p, MEASYEAR my, coalesce(MEASMON, 0) mm, coalesce(MEASDAY, 0) dd, INVYR iy, LAT lat, LON lon
             from PLOT where STATECD in {ST_SQL} and MEASYEAR >= 2000 and LAT is not null and LON is not null"""
    plc = latest_xy(con.execute(q_).df()); plf = latest_xy(canon(fc.execute(q_).df(), REMAP))
    pl = pd.concat([plc.assign(src='curated'), plf[~plf.set_index(key).index.isin(plc.set_index(key).index)].assign(src='fiadb')])
    mv_ = plf.merge(plc, on=key, suffixes=('_f', ''))
    mv_ = np.hypot((mv_.lat - mv_.lam_f) * 111.0, (mv_.lon - mv_.lom_f) * 111.0 * np.cos(np.radians(mv_.lat))) > 0.1
    print(f"plot coordinates (rows since 2000, latest visit): keys {len(pl)} (curated PLOT {int((pl.src == 'curated').sum())}, FIADB {int((pl.src == 'fiadb').sum())}); "
          f"spread of a key's rows > 0.1 km: {int((pl.spread_km > 0.1).sum())}; curated location > 0.1 km from FIADB's mean of the key's rows: {int(mv_.sum())}", flush=True)
    P = vis.drop_duplicates('pid')[['pid', *key]].merge(pl[key + ['lat', 'lon', 'spread_km']], on=key, how='left'); land = rasterio.open(land_path())
    COORDS = dict(source='curated PLOT latest visit since 2000 (D28), else FIADB latest', table_plots_curated=int(P.merge(plc[key], on=key).shape[0]),
                  table_plots_fiadb=int(P.merge(pl[pl.src == 'fiadb'][key], on=key).shape[0]), moved_vs_fiadb_mean_gt_0_1km=int(mv_.sum()))
    tf = Transformer.from_crs(4326, land.crs, always_xy=True); P['x'], P['y'] = tf.transform(P.lon.values, P.lat.values); P = P[P.lat.notna()]
    RST = {}; evs, negs, blocked = fia_records(FE, FV, P, key, RST, EVID_END)
    print(f"plots with typed FIA events {len(evs)} (events {sum(map(len, evs.values()))}); negative-evidence periods {sum(map(len, negs.values()))}; visits "
          f"blocked by a forest-share change >= 0.25 (no hiding code): vs the previous forested visit {RST['share_prev_forested']}, only vs FIA's previous "
          f"measurement (A14-2) {RST['share_prev_measurement']}", flush=True)
    jobs = [(int(r.pid), r.x, r.y, sorted(evs.get(r.pid, set())), negs.get(r.pid, [])) for r in P.itertuples()]
    rows = []; CF = []; n0 = 0; concs = {}
    with Pool(int(os.environ.get('NPROC', 6)), initializer=init) as pool:
        for i, (pid, o) in enumerate(pool.imap(plot_job, jobs, chunksize=8)):
            if o is None: continue
            nd_ = len(o['d']); rows.append(pd.DataFrame({'pid': pid, 'cid': np.arange(n0, n0 + nd_), 'dist_m': o['d'], 'prior': o['w'], 'incl_w': o['iw'], 'hi': o['hi'],
                                                        'row': o['ar'], 'col': o['ac']})); CF.append(o['F']); n0 += nd_
            concs[pid] = (o['conc'], o['n_hi'], o['mass_hi'], o['n'])
            if i % 500 == 0: print(f'plots {i}/{len(jobs)} candidates {n0}', flush=True)
    C = pd.concat(rows, ignore_index=True)
    for j_, nm in enumerate(['conc', 'n_hi', 'mass_hi', 'n_fp']): P[nm] = P.pid.map({k: v[j_] for k, v in concs.items()})
    vis.to_parquet(f'{OUT}/visits.parquet'); TK.to_parquet(f'{OUT}/tokens.parquet'); pairs.to_parquet(f'{OUT}/pairs.parquet'); P.to_parquet(f'{OUT}/plots.parquet')
    C.to_parquet(f'{OUT}/cands.parquet'); np.save(f'{OUT}/cand_feat.npy', np.concatenate(CF))
    REMAP.to_parquet(f'{OUT}/key_remap.parquet')     # (the CN -> canonical key map g02e / g09 read FIADB with: the same map as this build)
    # stand footprints of the mapping set: scripts/g02j_stand_footprints.py (pipeline stage stands_fp; depends on the stand run, not on this table)
    # meta.json: the table's switches, its evidence table (G46.4: later stages read EVK from here), the evidence end (G36 / G46.5), the build id, the
    # code hash taken at the start and the inputs (G51.5; g02e / g02g / g02j record theirs in their own parquet outputs, PROV_KEY), the vetting counts
    # (A14-1 / G51.1), the key remap and coordinate source (G51.2)
    ej_ = evk_json(EVK)
    write_json_atomic(dict(keys=KEYS, years=YEARS, classes=list(CLS), soils=SOILS, nf=NF, radius_m=R, ncand=NCAND, region=list(REGION), aef90=AEF90, radial_p=RADIAL_P,
                           radial_in_m=R_IN, evk=ej_, evk_hash=evk_hash(ej_), evk_provisional=bool(EVK_PROVISIONAL),
                           evk_lr_method=EVK_LR_METHOD[(12, 13)] if EVK_PROVISIONAL else EVK_LR_METHOD[REGION], hi_weight=HI_W, evid_end=EVID_END,
                           build_id=BUILD_ID, code_hash=CODE_HASH, code_files=CODE_FILES, inputs=INPUTS, **VET, share_blocks=RST, keys_canonical=KEYS_INFO,
                           coords=COORDS), f'{OUT}/meta.json')
    print(f'done: build {BUILD_ID}; candidates {len(C)} plots {C.pid.nunique()}; localisation concentration quantiles {np.quantile([v[0] for v in concs.values()], [.1, .5, .9]).round(3)}')
