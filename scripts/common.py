import json, numpy as np, rasterio
from rasterio.transform import Affine
import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # the standmap checkout (relocatable)
WS=os.environ.get('WORKSPACE', os.path.dirname(ROOT))                      # sibling projects (FIA databases)
# the FIA databases genveg reads (read-only), relative to WORKSPACE unless absolute; the pipeline sets them from the config `inputs` and fingerprints
# them into the stage hashes (audit r5 I4: a rebuilt curated table must not pass as current)
def curated_db(): return os.path.join(WS, os.environ.get('GV_CURATED_DB', 'FIA/truth/fia_truth_r4.duckdb'))
def fiadb(): return os.path.join(WS, os.environ.get('GV_FIADB', 'FIA/truth/fia_truth_r4.duckdb'))
def fiadb_con():
    """The FIA record, read-only: schema raw of the truth DB (FIA curation D59): every visit, as-recorded keys, PLT_CN / CN = the natural visit id."""
    import duckdb
    c = duckdb.connect(fiadb(), read_only=True); c.execute("SET schema = 'raw'"); return c
def visit_key_remap(cur, st_sql):
    """FIA visits filed under another (canonical) plot key: the truth DB's plot_key_remap (evidence-checked, D59). -> DataFrame cn = the natural visit id
    of the as-recorded key (raw schema's CN), st, u, c, p (canonical), ru, rc, rp (recorded)."""
    m = cur.execute(f"""select STATECD st, UNITCD u, COUNTYCD c, PLOT p, UNITCD_RAW ru, COUNTYCD_RAW rc, PLOT_RAW rp, INVYR y,
        CAST(CAST(STATECD AS BIGINT) * 100000000000000 + CAST(UNITCD_RAW AS BIGINT) * 1000000000000 + CAST(COUNTYCD_RAW AS BIGINT) * 1000000000
             + CAST(PLOT_RAW AS BIGINT) * 10000 + CAST(INVYR AS BIGINT) AS VARCHAR) cn
        from plot_key_remap where STATECD in {st_sql} order by cn""").df()
    return m[['cn', 'st', 'u', 'c', 'p', 'ru', 'rc', 'rp']].astype({k: 'int64' for k in ('st', 'u', 'c', 'p', 'ru', 'rc', 'rp')}).reset_index(drop=True)
def grid():
    g=json.load(open(f'{ROOT}/data/grid/grid.json'))
    return Affine(*g['transform']), g['width'], g['height'], g['crs']
PROFILE=dict(driver='GTiff',tiled=True,blockxsize=512,blockysize=512,compress='ZSTD',zstd_level=9,BIGTIFF='YES')
def aoi_mask():
    return f'{ROOT}/data/grid/aoi_mask.tif'
# provenance helpers (G51.5): the code hash of a script and its local imports, the fingerprint of an input file
import ast as _ast, hashlib as _hashlib, io as _io, tokenize as _tokenize
def code_files(path):
    """the script plus every LOCAL module it imports, transitively: an `import m` / `from m import ...` (any depth of the module, level-0 imports)
    whose m.py lies in the script's directory. Sorted absolute paths."""
    d = os.path.dirname(os.path.abspath(path)); seen = set(); todo = [os.path.abspath(path)]
    while todo:
        p = todo.pop()
        if p in seen: continue
        seen.add(p)
        for n in _ast.walk(_ast.parse(open(p, encoding='utf-8').read())):
            names = [a.name for a in n.names] if isinstance(n, _ast.Import) else ([n.module] if isinstance(n, _ast.ImportFrom) and n.level == 0 and n.module else [])
            for m in names:
                f = os.path.join(d, m.split('.')[0] + '.py')
                if os.path.exists(f) and f not in seen: todo.append(f)
    return sorted(seen)
def _code_tokens(src):
    """the token stream of a module that the code hash reads: (token type name, exact source text) of every NAME / NUMBER / OP / STRING token,
    NEWLINE / INDENT / DEDENT by type only (the indentation width does not count), an f-string as ONE STRING token of its source text (its inner
    FSTRING_* tokens are not read); comments, NL (blank / continued lines), the encoding and end markers and every docstring (the first statement of
    the module or of a def / class when it is a string constant, with the NEWLINE ending it) are dropped"""
    lines = src.splitlines(keepends=True)
    def chars(ln, bcol): return len(lines[ln - 1].encode('utf-8')[:bcol].decode('utf-8'))       # ast columns are UTF-8 byte offsets
    spans = []
    for n in _ast.walk(_ast.parse(src)):
        if isinstance(n, (_ast.Module, _ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)) and n.body and isinstance(n.body[0], _ast.Expr) \
                and isinstance(n.body[0].value, _ast.Constant) and isinstance(n.body[0].value.value, str):
            e = n.body[0]; spans.append(((e.lineno, chars(e.lineno, e.col_offset)), (e.end_lineno, chars(e.end_lineno, e.end_col_offset))))
    def in_doc(pos): return any(a <= pos < b for a, b in spans)
    def text(a, b):
        if a[0] == b[0]: return lines[a[0] - 1][a[1]:b[1]]
        return lines[a[0] - 1][a[1]:] + ''.join(lines[a[0]:b[0] - 1]) + lines[b[0] - 1][:b[1]]
    out = []; fdepth = 0; fstart = None; after_doc = False
    for tk in _tokenize.generate_tokens(_io.StringIO(src).readline):
        nm = _tokenize.tok_name[tk.type]
        if nm == 'FSTRING_START':
            if fdepth == 0: fstart = tk.start
            fdepth += 1; continue
        if nm == 'FSTRING_END':
            fdepth -= 1
            if fdepth == 0: out.append(('STRING', text(fstart, tk.end)))
            continue
        if fdepth or nm in ('COMMENT', 'NL', 'ENCODING', 'ENDMARKER'): continue
        if in_doc(tk.start): after_doc = True; continue
        if after_doc and nm == 'NEWLINE': after_doc = False; continue
        after_doc = False
        out.append((nm, '' if nm in ('NEWLINE', 'INDENT', 'DEDENT') else tk.string))
    return out
def code_hash(path):
    """provenance hash of a script (G51.5, replaces the G46.6 ast.dump hash): sha256 over code_files(path) in sorted order, each as its file name and
    the token stream _code_tokens of its source. Comments, docstrings, blank lines, line continuations, spacing and indentation width do not count; any
    other edit of the script or of a local module it imports does. Version independence by construction: the serialisation reads only token
    types every 3.12+ tokenizer emits and the exact SOURCE TEXT of names, numbers, operators and string literals (the lexical grammar is the same in
    3.12 and 3.13); an f-string counts as its whole source text, so the FSTRING_MIDDLE text and positions (whose brace unescaping and columns have
    changed between releases) are never read; docstrings are located by their ast start / end positions. Nothing reads ast.dump, whose output
    changed in 3.13 (E15-7). Compute it at the START of a run (A15-6f)."""
    h = _hashlib.sha256()
    for f in code_files(path):
        h.update(os.path.basename(f).encode() + b'\0')
        for nm, tx in _code_tokens(open(f, encoding='utf-8').read()): h.update(nm.encode() + b'\x1f' + tx.encode('utf-8') + b'\x1e')
        h.update(b'\x1d')
    return h.hexdigest()
FP_CHUNK = 64 * 2 ** 20
def file_fingerprint(path):
    """identity of an input file (A15-6c): path, size, mtime_ns and sha256 — of the whole file up to 2 x 64 MiB, else of its first and last 64 MiB
    (`sha256_scope` says which); the size and mtime catch most other edits of a large database"""
    st = os.stat(path); h = _hashlib.sha256()
    with open(path, 'rb') as f:
        if st.st_size <= 2 * FP_CHUNK: h.update(f.read()); scope = 'full'
        else: h.update(f.read(FP_CHUNK)); f.seek(st.st_size - FP_CHUNK); h.update(f.read(FP_CHUNK)); scope = 'first+last 64 MiB'
    return dict(path=os.path.abspath(path), size=int(st.st_size), mtime_ns=int(st.st_mtime_ns), sha256=h.hexdigest(), sha256_scope=scope)
