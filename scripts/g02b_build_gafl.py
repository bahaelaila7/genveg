"""Generative-vegetation prototype v2 (all GA+FL plots), training table. Same content as g02_build_proto.py but:
  visits   = every (plot, MEASDATE) of curated_trees (common.curated_db(), read-only) for GA+FL since 2000 with a forested condition (tree records on FIADB COND
             COND_STATUS_CD 1 only: non-forest conditions of mixed plots and wholly non-forest visits are dropped), single-visit plots included; live trees with
             estimated_age, DIA, HT, CR, TPA_UNADJ (the ages / filters of the Pan interval product); species key = top-K species, else _S/_H.
             Densities are per FORESTED area: TPA_UNADJ / (forest share of the subplot (DIA >= 5 in) or microplot (saplings) area, FIADB COND
             SUBPPROP_UNADJ / MICRPROP_UNADJ of COND_STATUS_CD 1), so a half-forested plot is not read as a half-stocked stand
  pairs    = consecutive visits of a plot
  cands    = up to NCAND footprints per plot (stratified: high-prior stratum + random fill), features as in g02 (AlphaEarth 2022/25 3x3 means,
             disturbance history 2000-2025 by class + severity + Hansen at the EVENT's calendar year, centre-pixel soils)
  prior    = FIA-dated events (FIADB COND of the forested conditions, codes 1-3) matched BY TYPE to candidate evidence in the event year (±1; planting /
             regeneration -2..+1) → ×8 (cutting matched by a LANDFIRE harvest class ×2 more); negative evidence: a remeasurement whose visit carries no FIA
             code that could hide a stand-replacing event (fia_records) makes candidate stand-replacing evidence (harvest class or Hansen loss) inside
             that period ×0.25 (the stand age at the first visit is NOT used: audit r5 M4). Kept candidates carry prior × inverse inclusion weight (each stands for incl_w footprints); hi = footprint in the full disk's
             high-prior set (prior >= 4 x median); plots carry n_hi / mass_hi on the full disk (localisation)
  conc     = per plot, prior mass on its top 10 % of footprints (localisation confidence; curriculum)
Mapping set: Baker stands as in g02. Output data/proto/gafl/…
Region switches (docs/GENVEG_V4_R4.md §B): REGION = FIA STATECDs (default 12,13 = GA+FL); AEF90=1 reads the Region 4 rasters of r4_grid.py /
r4_aef90.py / r4_rasters.py (30 m land / disturbance / Hansen on grid_r4, AlphaEarth footprint means and soils at 90 m on grid_r4_90): plot windows
start on a multiple of 3 so footprint centres are the 90 m block centres, and the AlphaEarth / soil features are the 90 m pixel of the block."""
import sys, os, json, numpy as np, pandas as pd, duckdb, rasterio, geopandas as gpd
from rasterio.windows import Window
from pyproj import Transformer
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from features import ROOT, WS, SOILS, dequant
from common import curated_db, fiadb
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
REGION = tuple(int(x) for x in os.environ.get('REGION', '12,13').split(',')); ST_SQL = f"({', '.join(map(str, REGION))})"
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
    # disturbance evidence = FRACTION of the footprint's 3x3 pixels (90 m, as for AlphaEarth) per class and event calendar year; severity = max
    for j, y in enumerate(YEARS):                                    # file years; events land at their calendar year
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                code = DS[j, rr + dr, cc + dc]
                for k in np.flatnonzero(code > 0):
                    cl, sv, ey = s['lut'].get((y, int(code[k])), (0, 0, y))
                    if cl > 0 and YEARS[0] <= ey <= YEARS[-1]: OH[k, ey - YEARS[0], cl - 1] += 1 / 9; SV[k, ey - YEARS[0]] = max(SV[k, ey - YEARS[0]], sv)
    OH = np.minimum(OH, 1.0)                                         # (fiscal-year duplicates of the same pixel)
    HF = np.stack([sum(((HL[rr + dr, cc + dc] == y - 2000) & (HL[rr + dr, cc + dc] > 0)).astype(np.float32) for dr in (-1, 0, 1) for dc in (-1, 0, 1)) / 9
                   for y in YEARS], 1).astype(np.float32)            # lossyear 0 = no loss
    for j in range(len(YEARS)): F += [OH[:, j], SV[:, j:j + 1], HF[:, j:j + 1]]
    F.append(np.nan_to_num(SO[:, br, bc].T)); return rr, cc, np.concatenate(F, 1).astype(np.float32)
# FIA event kinds -> compatible candidate evidence (class indices of CLS; 'H' = Hansen loss), year window (relative to the FIA year), likelihood ratio
# (assumed detectability: cuts / regeneration / site preparation / human / crown fire are well visible from space, ground fire / other treatments weakly)
H_ = NC + 1; IX = {k: i for i, k in enumerate(CLS)}
# STRICT, data-derived matching (2026-10-22, g09_fia_landfire_association.py; replaces the permissive table where any compatible class incl. the
# generic 'unknown' change class and Hansen loss matched every record type). Evidence classes per record type are those with a clear association in
# the data (lift at the FIA year vs the plot's quiet years: cutting -> harvest ×4.4-4.8, Hansen ×2.5-2.8; site preparation / planting -> chemical
# ×7-8; natural regeneration -> fire ×5.6; fire -> fire ×4.9, Hansen ×9 (crown); wind -> weather ×15; human -> harvest ×5.6). Stand-replacing evidence
# (Hansen loss) is compatible only with records whose FIA-measured outcome is stand-replacing (site preparation / regeneration median BA(v2)/BA(v1)
# 0.17-0.30, cutting 0.57 with 34 % < 0.3, crown fire). Mild records (wind, insects, human, other treatments: median 0.84-1.13) never match Hansen
# or 'unknown'. Likelihood ratio 4 = a detectable record (a hard constraint for the history-consistent sets / pairs, threshold lr >= 4; the old ×8
# was far stronger than the observed lifts ≈ 2); 2 = weak (never a hard constraint).
# T10-T40 also accept LANDFIRE 'unknown' (2026-10-22, data: 61 % of Hansen-loss footprint-years carry 'unknown' within ±1 yr vs 0.5 % 'harvest'; the
# MIICA change-detection class is how most harvests appear; g10: strict_unknown gives the best stand-replacing discrimination)
EVK = {'T10': ([IX['harvest'], IX['thin'], IX['unknown'], H_], (-1, 1), 4.0),                                     # cutting
       'T20': ([IX['harvest'], IX['chem'], IX['fire'], IX['unknown'], H_], (-1, 1), 4.0),                          # site preparation
       'T30': ([IX['harvest'], IX['chem'], IX['unknown'], H_], (-2, 1), 4.0),                                      # artificial regeneration (after a harvest)
       'T40': ([IX['harvest'], IX['fire'], IX['unknown'], H_], (-2, 1), 4.0),                                      # natural regeneration
       'T50': ([IX['thin'], IX['chem']], (-1, 1), 2.0),                                                           # other silvicultural treatment
       'Dfire': ([IX['fire']], (-1, 1), 2.0), 'Dcrown': ([IX['fire'], H_], (-1, 1), 4.0),                         # ground (or unspecified) / crown fire
       # ground fire is weak: prescribed burns recur every 2-3 yr, the fire class passes ~17 % of footprints, lift 1.85 -> never a hard constraint
       'Dins': ([IX['insdis']], (-1, 1), 2.0), 'Dwx': ([IX['weather']], (-1, 1), 2.0),
       'Dhum': ([IX['harvest'], IX['dev']], (-1, 1), 2.0)}                                                       # 'unknown' (70): not matched
NONBLOCK = {11, 21, 31, 40, 41, 42, 43, 44, 45, 46, 60}   # DSTRBCD that cannot hide stand-replacing evidence: insect / disease damage to UNDERSTORY vegetation (11 / 21;
# 12 / 22 are damage to TREES and can — FIADB v9.4 / EVALIDATOR library, audit r6 I1), ground fire, animal, vegetation
def ev_kind(trt, code):
    c = int(code)
    if trt: return {10: 'T10', 20: 'T20', 30: 'T30', 40: 'T40', 50: 'T50'}.get(c)
    if c in (30, 31): return 'Dfire'
    if c == 32: return 'Dcrown'
    if 10 <= c <= 22: return 'Dins'
    if 50 <= c <= 54: return 'Dwx'
    if c == 80: return 'Dhum'
    return None                                     # animal (40s), vegetation (60), geologic (90): rarely visible from space
def clusters(evs):
    """FIA records of ONE physical event (a cut + site preparation + planting, or the same event re-reported a year apart) must not multiply:
    events whose years chain within 2 years form one cluster -> (evidence columns = union, window = union, likelihood ratio = max, has cutting)."""
    out = []
    for kind, yv in sorted(evs, key=lambda e: e[1]):
        cols, (dlo, dhi), lr = EVK[kind]
        if out and yv - out[-1]['ylast'] <= 2:
            c = out[-1]; c['cols'] |= set(cols); c['lo'] = min(c['lo'], yv + dlo); c['hi'] = max(c['hi'], yv + dhi); c['lr'] = max(c['lr'], lr); c['cut'] |= kind == 'T10'; c['ylast'] = yv
        else: out.append(dict(cols=set(cols), lo=yv + dlo, hi=yv + dhi, lr=lr, cut=kind == 'T10', ylast=yv))
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
def plot_job(a):
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
    hi = wr >= 4.0; n_hi, mass_hi = int(hi.sum()), float(w[hi].sum())       # localisation on the full disk: footprints POSITIVELY matched (baseline weight 1)
    iw = np.ones(len(d)); keep = np.arange(len(d))
    if len(d) > NCAND:     # stratified sample: elevated-prior stratum (all, or a uniform subset of 3/4 NCAND) + uniform fill of the rest
        rg = np.random.default_rng(pid); up = np.flatnonzero(wr > 1.5); rest = np.setdiff1d(np.arange(len(d)), up)      # positive evidence vs the rest
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
def fiadb_visits(fc):
    """FIADB visits (read-only) with a forested condition: forested shares of the WHOLE plot (fsub / fmic: TPA_UNADJ expands a tree to the whole plot area,
    so trees per forested area = Σ TPA / f1s — audit 2026-10-26 F8; the earlier f1s / (1 − n5s) made densities low by (1 − n5s)) and of the SAMPLED area
    (fsub_s, non-sampled COND_STATUS_CD 5 excluded: the visit filter), stand origin, previous measurement year (PREV_PLT_CN), live-tree count."""
    FV = fc.execute(f"""select p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, (case when p.MEASMON between 1 and 12 and p.MEASDAY between 1 and 31 then make_date(p.MEASYEAR::int, p.MEASMON::int, p.MEASDAY::int) end) md, p.MEASYEAR my, p.CN cn,
                               any_value(q.MEASYEAR) prev_my,
                               sum(case when c.COND_STATUS_CD = 1 then c.SUBPPROP_UNADJ else 0 end) f1s, sum(case when c.COND_STATUS_CD = 5 then c.SUBPPROP_UNADJ else 0 end) n5s,
                               sum(case when c.COND_STATUS_CD = 1 then c.MICRPROP_UNADJ else 0 end) f1m, sum(case when c.COND_STATUS_CD = 5 then c.MICRPROP_UNADJ else 0 end) n5m,
                               max(case when c.COND_STATUS_CD = 1 then c.STDORGCD end) org
                        from PLOT p join COND c on c.PLT_CN = p.CN left join PLOT q on q.CN = p.PREV_PLT_CN
                        where p.STATECD in {ST_SQL} and p.MEASYEAR >= 2000 group by all having f1s > 0""").df()
    nlive = fc.execute(f"select PLT_CN cn, count(*) n from TREE where STATECD in {ST_SQL} and STATUSCD = 1 group by 1").df()
    FV = FV.merge(nlive, on='cn', how='left'); FV['n'] = FV.n.fillna(0); FV['md'] = pd.to_datetime(FV.md)
    FV['fsub_s'] = (FV.f1s / (1 - FV.n5s).clip(lower=1e-6)).clip(upper=1.0)
    FV['fsub'] = FV.f1s.clip(upper=1.0); FV['fmic'] = np.where(FV.f1m > 0, FV.f1m.clip(upper=1.0), FV.fsub)
    return FV
def fiadb_events(fc):
    """TRT / DSTRB codes of the forested conditions of every FIADB visit since 2000 (read-only)."""
    FE = fc.execute(f"""select p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, (case when p.MEASMON between 1 and 12 and p.MEASDAY between 1 and 31 then make_date(p.MEASYEAR::int, p.MEASMON::int, p.MEASDAY::int) end) md,
                               c.TRTCD1, c.TRTYR1, c.TRTCD2, c.TRTYR2, c.TRTCD3, c.TRTYR3, c.DSTRBCD1, c.DSTRBYR1, c.DSTRBCD2, c.DSTRBYR2, c.DSTRBCD3, c.DSTRBYR3
                        from PLOT p join COND c on c.PLT_CN = p.CN where p.STATECD in {ST_SQL} and p.MEASYEAR >= 2000 and c.COND_STATUS_CD = 1""").df()
    FE['md'] = pd.to_datetime(FE.md); return FE
def fia_records(FE, FV, P, key):
    """FIA events per plot {pid: {(kind, year)}} from every forested condition; negative evidence {pid: [(y_lo, y_hi)]} = measurement periods whose
    visit carries no code that could hide stand-replacing evidence: (previous measurement + 1 .. visit - 1); for the first visit since 2000 the
    previous measurement (PREV_PLT_CN, periodic inventories) or, for a new plot, FIA's 5-year lookback; blocked = {(plot key, visit date)} with such a code.
    Shared by g02b (location prior) and g03b (FIA-anchored remeasurement pairs)."""
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
        yrs = g_.my.values.astype(int); mds = g_.md.values; pm = g_.prev_my.values; nl = []; fs_ = g_.f1s.values
        for j in range(len(yrs)):
            # a forest -> non-forest (or non-sampled) conversion of >= 1/4 of the plot since the previous forested visit carries no TRT / DSTRB code
            # (codes live on forested conditions only): it can hide a clearing, so it blocks like a hiding code (audit r7 M1)
            if j > 0 and abs(fs_[j - 1] - fs_[j]) >= 0.25: blocked.add((k_, pd.Timestamp(mds[j])))   # (either direction: a re-delineation changes the
            # per-forested-area densities too — median BA log change −0.16 vs +0.14 for unchanged shares; audit r8 M2)
            if (k_, pd.Timestamp(mds[j])) in blocked: continue
            # the visit's codes cover the period since FIA's PREVIOUS measurement (PREV_PLT_CN, any status): a non-forest / non-sampled visit in between
            # shortens it (audit r6 M1); the later of that and the previous forested visit is used
            if pd.notna(pm[j]): nl.append((max(int(pm[j]), yrs[j - 1] if j > 0 else -9999) + 1, yrs[j] - 1))
            elif j > 0: nl.append((yrs[j - 1] + 1, yrs[j] - 1))
            else: nl.append((yrs[0] - 5, yrs[0] - 1))
        negs[pidk[k_]] = nl
    return evs, negs, blocked
if __name__ == '__main__':
    con = duckdb.connect(curated_db(), read_only=True)
    t = con.execute(f"""select t.STATECD st, t.UNITCD u, t.COUNTYCD c, t.PLOT p, t.SUBP sb, t.MEASDATE md, t.DIA, t.HT, t.CR, t.TPA_UNADJ tpa, t.estimated_age age, r.SPECIES_SYMBOL sym, r.SFTWD_HRDWD grp,
                              t.STATUSCD status, t.CONDID cond, t.cond_stdage stdage, t.TRTCD1, t.TRTYR1, t.TRTCD2, t.TRTYR2, t.DSTRBCD1, t.DSTRBYR1, t.DSTRBCD2, t.DSTRBYR2, t.cond_stdorgcd org
                       from curated_trees t join REF_SPECIES r on r.SPCD = t.SPCD
                       where t.STATECD in {ST_SQL} and t.MEASDATE >= '2000-01-01' order by t.STATECD, t.UNITCD, t.COUNTYCD, t.PLOT, t.MEASDATE, t.SUBP, t.TREE""").df()
    t['live'] = t.status == 1          # every visit is kept (bare / just-harvested visits = no live tree = empty cohort table); cohorts from live trees only
    t['md'] = pd.to_datetime(t.md); key = ['st', 'u', 'c', 'p']
    fc = duckdb.connect(fiadb(), read_only=True)
    CS = fc.execute(f"""select p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, (case when p.MEASMON between 1 and 12 and p.MEASDAY between 1 and 31 then make_date(p.MEASYEAR::int, p.MEASMON::int, p.MEASDAY::int) end) md, c.CONDID cond, c.COND_STATUS_CD cs
                        from PLOT p join COND c on c.PLT_CN = p.CN where p.STATECD in {ST_SQL} and p.MEASYEAR >= 2000""").df(); CS['md'] = pd.to_datetime(CS.md)
    t = t.merge(CS, on=key + ['md', 'cond'], how='left'); n0_ = len(t)
    print(f"tree records on non-forest / unmatched conditions dropped: {int((t.cs != 1).sum())} of {n0_} (live {int(((t.cs != 1) & (t.status == 1)).sum())})", flush=True)
    t = t[t.cs == 1].reset_index(drop=True)            # the cohort table describes the FORESTED conditions (visits with only non-forest conditions drop out)
    # top species among LIVE trees
    cnt = t[t.live].sym.value_counts(); top = list(cnt.index[:int(os.environ.get('TOPK', 16))]); KEYS = sorted(top) + ['_H', '_S']; kix = {k: i for i, k in enumerate(KEYS)}
    t['key'] = [kix[s] if s in top else kix['_' + (g if g in ('S', 'H') else 'H')] for s, g in zip(t.sym, t.grp)]
    FV = fiadb_visits(fc); FE = fiadb_events(fc)
    # visits = forested visits with curated tree records + truly bare forested visits (no live tree in FIADB, absent from the curated table) of the same plots
    vt = t[key + ['md']].drop_duplicates(); kp_ = vt[key].drop_duplicates()
    bare = FV[FV.n == 0].merge(kp_, on=key).merge(vt.assign(_in=1), on=key + ['md'], how='left'); bare = bare[bare._in.isna()][key + ['md']]
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
    # plot coordinates from the annual-inventory rows (MEASYEAR >= 2000, the table's visits) only: in Kentucky 1,199 plot keys reuse the number of a
    # 1986-88 periodic plot several km away (audit 2026-10-26 F1); the spread of a key's rows is reported
    pl = fc.execute(f"""select STATECD st, UNITCD u, COUNTYCD c, PLOT p, avg(LAT) lat, avg(LON) lon, (max(LAT) - min(LAT)) * 111.0 + (max(LON) - min(LON)) * 111.0 * cos(radians(avg(LAT))) spread_km
                         from PLOT where STATECD in {ST_SQL} and MEASYEAR >= 2000 group by all""").df()
    print(f"plot coordinates (rows since 2000): keys {len(pl)}, spread of a key's rows > 0.1 km: {int((pl.spread_km > 0.1).sum())}", flush=True)
    P = vis.drop_duplicates('pid')[['pid', *key]].merge(pl, on=key, how='left'); land = rasterio.open(land_path())
    tf = Transformer.from_crs(4326, land.crs, always_xy=True); P['x'], P['y'] = tf.transform(P.lon.values, P.lat.values); P = P[P.lat.notna()]
    evs, negs, blocked = fia_records(FE, FV, P, key)
    print(f"plots with typed FIA events {len(evs)} (events {sum(map(len, evs.values()))}); negative-evidence periods {sum(map(len, negs.values()))}", flush=True)
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
    # stand footprints of the mapping set: scripts/g02j_stand_footprints.py (pipeline stage stands_fp; depends on the stand run, not on this table)
    json.dump(dict(keys=KEYS, years=YEARS, classes=list(CLS), soils=SOILS, nf=NF, radius_m=R, ncand=NCAND, region=list(REGION), aef90=AEF90, radial_p=RADIAL_P, radial_in_m=R_IN), open(f'{OUT}/meta.json', 'w'))
    print(f'done: candidates {len(C)} plots {C.pid.nunique()}; localisation concentration quantiles {np.quantile([v[0] for v in concs.values()], [.1, .5, .9]).round(3)}')
