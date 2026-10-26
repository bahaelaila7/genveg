"""Region switches of the generative-vegetation stages (docs/GENVEG_V4_R4.md §B): REGION (FIA STATECDs, default 12,13) and AEF90 (Region 4 rasters at
90 m). A table records them in meta.json (`region`, `aef90`; absent = the GA+FL 30 m default); every later stage checks them (audit 2026-10-26 P1)."""
import os
def region_check(meta):
    """The table's region switches must equal this process's REGION / AEF90 (else a stage would mix a table with another region's rasters / settings)."""
    want = (tuple(int(x) for x in os.environ.get('REGION', '12,13').split(',')), os.environ.get('AEF90', '0') == '1')
    have = (tuple(meta.get('region', (12, 13))), bool(meta.get('aef90', False)))
    if want != have: raise SystemExit(f'region switches {want} (env REGION / AEF90) != the table\'s {have} (meta.json)')
