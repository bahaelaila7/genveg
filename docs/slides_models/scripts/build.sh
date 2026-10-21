#!/usr/bin/env bash
# Build the Beamer PDF with Tectonic. TECTONIC = path to the binary (default: the one installed for the PERSEUS deck, a sibling folder).
set -euo pipefail
cd "$(dirname "$0")/.."
T="${TECTONIC:-$(cd ../../.. && pwd)/PERSEUS_PRESENTATION/.tools/env/bin/tectonic}"
$T -X compile --keep-logs --outdir build main.tex > build/main.stdout 2>&1 || { tail -40 build/main.stdout; exit 1; }
cp build/main.pdf Stand_Delineation_and_HBAD_Models.pdf
grep -h "Overfull" build/main.log | sort | uniq -c | sort -rn | head -10 || true
