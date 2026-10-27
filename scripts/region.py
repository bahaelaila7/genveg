"""Region switches of the generative-vegetation stages (docs/GENVEG_V4_R4.md §B): REGION (FIA STATECDs, default 12,13) and AEF90 (Region 4 rasters at
90 m). A table records them in meta.json (`region`, `aef90`; absent = the GA+FL 30 m default); every later stage checks them (audit 2026-10-26 P1).
REGION is a SET of state codes: both sides are compared as the sorted tuple of unique codes, as g02b keys EVK_TABLES (G46.4)."""
import os
def region_set(codes):
    """the canonical form of a region: the sorted tuple of its unique FIA state codes (from an iterable or a comma-separated string)"""
    if isinstance(codes, str): codes = codes.split(',')
    return tuple(sorted({int(x) for x in codes}))
def region_check(meta):
    """The table's region switches must equal this process's REGION / AEF90 (else a stage would mix a table with another region's rasters / settings)."""
    want = (region_set(os.environ.get('REGION', '12,13')), os.environ.get('AEF90', '0') == '1')
    have = (region_set(meta.get('region', (12, 13))), bool(meta.get('aef90', False)))
    if want != have: raise SystemExit(f'region switches {want} (env REGION / AEF90) != the table\'s {have} (meta.json)')
