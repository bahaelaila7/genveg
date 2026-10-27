"""Regression suite of the swallowed-code detector (scripts/check_comments.py; docs/DECISIONS.md G12 / G43 / G45.5 / G53.2): every crafted case of the
r9-r15 audits and the historical swallows. A CODE case must be flagged (0 misses); ACCEPTED cases are the prose-shaped simple assignments / no-argument
calls after a prose word that G43.1 accepts; PROSE cases are reported (a flagged prose comment is reworded). Also, over the pipeline code (the
exposure checks, 0 misses):
  - every statement that follows another one on its line, swallowed after a `;` (any kind but a bare expression) or after a prose word (calls,
    keyword statements, dense assignments);
  - every one-line statement cut before a depth-0 operator / keyword / comma / attribute / subscript token whose prefix still parses, the rest
    moved into a comment (directly and after a prose word);
  - every multi-line bracketed statement cut on a non-last line before an operator / keyword / comma / attribute / subscript token, the rest of
    that line moved into a comment, when the statement still parses (r15 D15-2 harnesses).
The genveg driver runs this suite at start and refuses when it fails (G53.2).
  .venv/bin/python -I scripts/tests/test_check_comments.py [-v]   -> exit 1 on any miss (also runs under pytest)"""
import ast, io, os, sys, tempfile, textwrap, tokenize
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(HERE))
from check_comments import swallowed, swallowed_src, pipeline_scripts
# one-line cases: the comment text of `x = 1  #<text>`
CODE = [  # r14 D-6 (crafted, strong points)
 ' note; y = f()', ' note; main()', ' note; G2.init()', ' note; raise SystemExit(f"bad {x}")', ' note; assert n > 0, n', ' note; continue', ' note; break',
 ' note; global NT', ' note; nonlocal NT', ' note; return out', ' note; pass', ' note; def g(v): return v', ' note; n += 1', ' note; y = x', ' note; V = V0',
 ' note; (n := len(a))', ' note; print(x)', ' note; f(x)', ' note; lf.flush()', ' note; x, y = y, x', ' note; y = x if a else z', ' note; y = -x',
 ' note; y = [v for v in a]', ' note; y = x * 0.25', ' r13; if bad: sys.exit(1)', ' r13: if bad: sys.exit(1)', ' note; y = f(x)  # orig', ' note;y=f(x)',
 ' note; y = f(x).', ' note; np.random.seed(0)', ' note; del a[0]', ' note; ok = ok and n > 20', " note; D = f'{ROOT}/x'", ' note; y: int = 3',
 ' see G12) y = 0', ' [r13] y = 0',
 # r14 D-6 (crafted, weak points: dense statements, calls with non-literal arguments, keyword statements)
 ' then y = f(x)', ' then f(x)', ' then print(x)', ' then raise ValueError(x)', ' then assert ok', ' then x, y = 1, 2', ' then n -= 1',
 ' note — y = f(x)', ' note -> y = f(x)', ' note (y = f(x))', ' note\ty = f(x)', ' 3 y = f(x)', ' note np.random.seed(0)', ' note torch.manual_seed(SEED)',
 ' note y = a[0]', ' note y = a.b', ' note: y = f(x, k=1)', ' note del a', ' note import pandas as pd', ' (r13) region_check(M)',
 " the T4b test check('T4b x 2.471', ok, d)", ' then (n := len(a))',
 # r14 D-6 (c): statement tails and call arguments
 ' scale * 2', ' keep live .reset_index(drop=True)', ' forest & (b < 3)', ' + w_age * L_age', ' clip .clip(0, None)', ' note: [1:]', ' note, axis=1)',
 # r14 F-1 (sw_cases / sw_cases2)
 ' note; y = f(x)', ' (r13 C-5) vv_ = vv_[fs_ok[vv_]]', ' keep the order: out.append(r)', ' the filter trp_ = set(vpid[tr_v])', ' note: torch.manual_seed(0)',
 ' (G36) be_ = min(b, G2.EVID_END - YEARS[0])', ' see above; return out', ' note) if ok: z = 2', " (r13) FALLBACK['n'] += 1",
 ' see G33 ok = all(e1[k] <= e0[k] for k in K) and dep0 > 0', ' (C-8) hA = [v for v in cal_m if vpid[v] in _hp]', ' (G35) span = [u for u in span if fs_ok[u]] or [v2]',
 ' note: blk = True', ' (G38) if abs(a - b) >= 0.25: blk = True', " (g06) out['latent'] = latent_stats(np.asarray(va_v))", ' note: i_ = int(dist_.argmin())',
 " the medoid draw — md_ = medoid_draw(lp_, val_, s_, int(sid)); i_, pres = md_['i'], md_['pres']", ' note, z = enc(vt, vyear[vt])', ' →  z = enc(vt, vyear[vt])',
 ' (fs_ok: r13)   vv_ = np.random.default_rng(33).choice(vv_, 800)', ' -- res[kind] = float(torch.quantile(s[1], 0.5))', ' ... a, b = b, a', ' note: global FOO',
 " note: assert ok, 'bad'", ' note: raise SystemExit(3)', ' note: sys.exit(4)', " note: print('x')", ' 2.5 % y = f(x, z)', ' the stand id — md_ = medoid_draw(lp_, val_, s_, int(sid))',
 " 'unknown' y = f(x, z)", ' see "G36" y = f(x, z)', ' r13-C5 - y = f(x, z)', ' (ok)? y = f(x, z)', ' note:y = f(x, z)', ' note:  y = f(x, z)  # trailing',
 # r13 case files (D s1-s5, E cc_cases, F cc_test) and the historical swallows (r5-r12; g04 trp_)
 ' pad; n = len(x)', ' k += 1', ' b = f(a)', ' note; y = np.sum(x)', ' note; y = int(x * 2)', ' scale; w = w * 0.25', ' count; n += 1', ' log; out.append(c)',
 ' note: w = f(d)', ' note; print(e)', ' tmp; del out', ' x = 1', ' w[0] = 9', ' w, n = f(1), f(2)', ' if n: w = 0', ' (fix) k2 = f(3)', ' note; b = np.zeros(3)',
 ' note; opt.step()', ' note; out.append(d)', " (no column) vs_of = V.groupby('pid').vid.apply(list)", ' see below; n_ = len(a)', ' x; y2 = a + 1',
 ' count; n8 = n8 + 1', ' last visit passing the forest-share filter; trp_ = set(vpid[tr_v])', ' last visit passing the forest-share filter trp_ = set(vpid[tr_v])',
 ' the visit year (clamped); yy = min(int(math.floor(float(yv))), YEARS[-1])', ' note: n_ = len(cid)', " (r5 B1) vs_of = V.assign(vid=np.arange(nv)).sort_values('year')",
 ' a, b = act_span(y1, y2, rs, re)',
 # r15 F-F5 (crafted inline tails; G53.2) and D15-2 one-line tails
 ' and not blocked[v]', ' or fs_ok[v]', ' , b', ' **2', ' -b[0]', ' *math.pi', ' -1', ' if VAL_SPLIT else 0.5', ' (G46) if VAL_SPLIT else 0.5',
 ' or 4', ' and m > 20', ' note and n > 20', ' note if a else b',
]
ACCEPTED = [  # G43.1: prose-shaped simple assignments / calls without non-literal arguments after a prose word
 ' then y = f()', ' then y = f(1)', ' then n = n + 1', ' then w = w * 0.25', ' then y = x', ' then n = 0', ' then y = -x', ' then y = x if a else z',
 ' then y = [v for v in a]', ' then y = {k: 1 for k in a}', ' then main()', ' then continue', ' note ok = ok and n > 20', " note D = f'{ROOT}/x'", ' note y: int = 3',
 ' note — blk = True', ' 2. mil = mil * w', ' then w = 3', ' reset q = 0 before loop',
]
PROSE = [' severity = max', ' at x = 45', ' invert year = year + day-of-year / 366', ' passes N(0, I)', ' A + Am(z)', ' BA(v2)/BA(v1)', ' g02b.EVK',
 ' (r13 S-1) see g02b.feats()', ' default n = 300', ' (default) n = 300', ' e.g. NT=2', ' i.e. y = f(x)', ' the share p = n_hi / n', ' rule: keep if tph > 0',
 ' x = 1 is the default', ' TPA × 2.47105', ' log(y) - log(x)', ' (see G7) y = f(x) is the definition', ' [r13] the rule p_hi = mass_hi / mass',
 ' (G31) attractor = W @ z', ' r13: mask = a & b', ' cap (r13): n = min(n, 20)', ' i.e. sd = std(x)', ' (sigma = exp(ltau))', ' ratio r = a[0] / a[1]',
 ' fallback: D.get(k)', ' see self.fit()', ' (cf. np.median(x))', ' stored as float16; x.astype(np.float16) later', ' the checks pass', ' the loop may break',
 ' (r13 S-1, see G7)', ' [r13]', ' the mosaic — 30 m', ' 0.5 mile (FIA)']
# multi-line cases (r14 D-6 (b): swallowed elements of multi-line literals; the driver's own tables are such literals)
LIT = {
 'list element after prose': "L = ['a.tif',  # note 'data/treemap/tm_id.tif', 'data/grid/aoi_mask.tif',\n     'z.tif']\n",
 'list element after ;': "L = ['a.tif',  # note; 'data/treemap/tm_id.tif', 'data/grid/aoi_mask.tif',\n     'z.tif']\n",
 'set element (STAGE_SWITCHES style)': "S = {'A', 'B',  # (G29) 'SIGREG_MEAN', 'JUMP_RESET',\n     'C'}\n",
 'dict entry': "D = dict(a=1,  # note b=2,\n         c=3)\n",
 'dict entry call': "D = dict(a=1,  # note b=f(x),\n         c=3)\n",
 'kwarg in call': "r = f(x,  # note; weights_only=False,\n      y=1)\n",
 'STAGES entry': "ST = {\n  'a': dict(code=['x.py']),  # note 'b': dict(code=['y.py'], cfg=[]),\n  'c': 1}\n",
 'own-line comment element': "L = ['a.tif',\n     # see 'b.tif', 'c.tif',\n     'z.tif']\n",
 'tail inside parentheses': "loss = (l1  # + w_age * L_age\n        )\n",
 'tail after a closer': "x = f(a,\n      b)  # .clip(0, None)\n",
 'keyword argument closing a call': "y = f(x,\n      a  # note, axis=1)\n      )\n",
}
# r15 D15-2 (ml_cases.py), F-F5 and the real r14 lines that would have swallowed silently: whole sources, each must be flagged. F's probes with
# the comment tails tilde-b, `lambda: 0` and `:-1]` are not swallows: put back after the code before them, they do not parse.
SRC = {
 'paren formula tail + more lines': "loss = (lmil + W_COMP * lcomp  # + W_AGE * lage\n        + W_SUM * lsum)\n",
 'paren method chain tail + more': "df = (pd.read_parquet(p)  # .query('x > 0')\n      .reset_index())\n",
 'paren bool tail + more': "ok = (a > 0  # and b > 0\n      and c > 0)\n",
 'paren subscript tail + more': "v = (arr  # [mask]\n     .sum())\n",
 'ternary tail inline': "w = 1.0  # if VAL_SPLIT else 0.5\n",
 'ternary tail after prose': "w = 1.0  # (G46) if VAL_SPLIT else 0.5\n",
 'or-default tail': "n = int(os.environ.get('NT', 0))  # or 4\n",
 'and tail': "ok = n > 20  # and m > 20\n",
 'not-in tail': "keep = k  # not in DROP\n",
 'in tail': "hit = k  # in SEEN\n",
 'tuple tail': "lo, hi = 0  # , 1\n",
 'return tuple tail': "def f():\n    return a  # , b\n",
 'import tail': "import os  # , sys\n",
 'chained assignment tail': "a = b  # = c\n",
 'call-args swallowed after closer (method)': "x = f(a)  # (b)\n",
 'kwarg in dict literal own-line': "D = dict(\n    a=1,\n    # b=f(x),\n    c=3)\n",
 'decorator arg': "@lru_cache  # (maxsize=None)\ndef f(): pass\n",
 'with-item tail': "with open(a) as fa:  # , open(b) as fb:\n    pass\n",
 'for-target tail': "for i in range(n):  # if i % 2:\n    pass\n",
 'lambda tail': "f = lambda x: x  # + 1\n",
 'slice tail': "y = x[:5]  # [::-1]\n",
 'negative tail no space': "y = x  # -1\n",
 'power tail no space': "y = x  # **2\n",
 'starred element': "L = [a,  # note *rest,\n     b]\n",
 'dict-spread element': "D = {'a': 1,  # note **extra,\n     'b': 2}\n",
 'keyword arg mid-call (no closer next)': "r = f(x,  # note weights_only=False,\n      y=1)\n",
 'swallowed elif cond': "if a:  # or b:\n    pass\n",
 'condition tail in while': "while n > 0:  # and not done:\n    n -= 1\n",
 'assert message tail': "assert ok  # , 'bad table'\n",
 'f-string arg': "print(x)  # ; print(f'{y}')\n",
 'semicolon no space': "x = 1  #;y = f(x)\n",
 'unicode lead': "x = 1  # ← y = f(x)\n",
 'tab before code': "x = 1  #\ty = f(x)\n",
 'F5 word operator and-not': "ok = a > 0  # and not blocked[v]\n",
 'F5 word operator or': "ok = a > 0  # or fs_ok[v]\n",
 'F5 not in': "ok = k  # not in excl\n",
 'F5 in': "ok = k  # in allowed\n",
 'F5 is None': "ok = k  # is None\n",
 'F5 depth-0 comma': "def f():\n    return a  # , b\n",
 'F5 power no space': "d2 = d  # **2\n",
 'F5 unary minus subscript': "y = a  # -b[0]\n",
 'F5 times attribute': "y = a  # *math.pi\n",
 'F5 tuple element subscript': "t = x  # , y[0]\n",
 'F5 ternary call': "y = f(x)  # if AGE_CH else g(x)\n",
 'F5 comprehension': "y = sum(w)  # for w in ws\n",
 'F5 call not in': "y = g(x)  # not in excl\n",
 'F5 call is None': "y = g(x)  # is None\n",
 'F5 statement after ;': "x = [1, 2]  # is the list; x.sort()\n",
 'F5 call': "r = f  # (x, y)\n",
 'r14 g04 map year gate': "y0u = MAP_Y0  # if gate.get('passed') else 2021.0\n",
 'r14 g02b evidence table': "evk = EVK  # if evk is None else evk\n",
 'r14 gv_verify T4x': "ok4x = len(b4) >= 8  # and all(res4['flagged'][k] == 0.0 for k in ('mil', 'q', 'qdec')) and okB4\n",
 'r14 gv_verify T4m multi-line': "check('T4m',\n      prod_ok and (s_rsra - s_0).abs().max() < 1e-7  # and (s_late - s_0b).abs().max() < 1e-7 and\n      (e_no - e_0).abs().max() < 1e-7)\n",
}
LIT_PROSE = {
 'prose in a list': "L = ['a.tif',  # the mosaic\n     'z.tif']\n",
 'prose after the last element': "L = ['a.tif',  # see b\n     ]\n",
 'aside closing a call': "x = f(a,  # see, b)\n      c)\n",
 'aside in a dict': "ST = {\n  'a': dict(code=['x.py']),   # (clean: g02b.feats' forest filter, r13 A-11)\n  'c': 1}\n",
}
SRC_PROSE = {  # r15 D15-2 fresh prose comments (ml_cases.py): reported (a flagged one would be reworded)
 'p1': "x = 1  # or None, if unset\n", 'p2': "n = 3  # and the rest of the plots\n", 'p3': "y = f(x)  # if any\n", 'p4': "t = 0  # in seconds\n",
 'p5': "z = 2  # (see G12)\n", 'p6': "a = b  # = the default\n", 'p7': "k = 5  # , i.e. five\n", 'p8': "v = w  # not in the table\n",
 'p9': "q = r  # is the default\n", 'p10': "s = 0.5  # or 0.25 for R4\n", 'p11': "m = 32  # or 16 at most\n", 'p12': "c = d  # if c else the fallback\n",
 'p13': "e = 1  # in km\n", 'p14': "h = g  # or g2\n", 'p15': "p = 0  # and q = 1 elsewhere\n", 'p16': "x = 1  # use the global settings\n",
 'p17': "y = 2  # it may return early\n", 'p18': "z = 3  # raise the threshold later\n", 'p19': "w = 4  # assert-free\n", 'p20': "u = 5  # import time only\n",
 'p21': "r = 6  # del the old cache first\n", 'p22': "t = 7  # pass-through value\n", 'p23': "n = 8  # if NT else 1\n", 'p24': "n = 9  # or NT\n",
 'p25': "x = 1  # see f(x) above\n", 'p26': "y = 2  # ratio a / b\n", 'p27': "k = 3  # the [0, 1] range\n", 'p28': "m = 4  # .5 m resolution\n",
 'p29': "x = 1  # return value of main\n", 'p30': "x = 1  # global NT is set by main\n", 'p31': "x = 1  # nonlocal state\n", 'p32': "x = 1  # yield per hectare\n",
 'p33': "x = 1  # lambda functions are avoided\n", 'p34': "x = 1  # print the table\n", 'p35': "x = 1  # x is the year, not the index\n",
 'p36': "x = 1  # try harder\n", 'p37': "x = 1  # while waiting\n", 'p38': "x = 1  # for each plot\n", 'p39': "x = 1  # with replacement\n",
 'p40': "x = 1  # class 1-4 only\n", 'p41': "x = 1  # def. 2.5 cm\n", 'p42': "x = 1  # await the lock\n", 'p43': "x = 1  # from FIADB\n",
 'p44': "x = 1  # then exit(1)\n", 'p45': "x = 1  # call main() twice\n", 'p46': "x = 1  # i.e. len(a)\n", 'p47': "x = 1  # raise\n",
 'p48': "x = 1  # see main\n", 'p49': "x = 1  # global\n", 'p50': "x = 1  # return\n",
}
_CUT = {'+', '-', '*', '/', '//', '%', '**', '@', '&', '|', '^', '<<', '>>', '<', '>', '<=', '>=', '==', '!=', 'and', 'or', 'in', 'not', 'is', 'if', 'else',
        ',', '.', '[', '='}
def _ctx_tree(s):
    """the source parsed as it stands, as a block header, an else / except clause or a function body line; None if it does not parse"""
    for w in ('{}', '{}\n pass', 'if 1: pass\n{}\n pass', 'try: pass\n{}\n pass', 'def _f():\n {}\n'):
        try: return ast.parse(w.format(s))
        except SyntaxError: pass
    return None
def _ctx_ok(s): return _ctx_tree(s) is not None
def _statements(src):
    """[(first row, last row, significant tokens)] of the logical lines without a comment or a multi-line string; f-string internals are marked"""
    out, cur = [], []
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        cur.append(t)
        if t.type != tokenize.NEWLINE: continue
        sig, fs = [], 0
        for x in cur:
            if x.type in (tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.NEWLINE): continue
            if x.type == getattr(tokenize, 'FSTRING_START', -1): fs += 1
            sig.append((x, fs > 0))
            if x.type == getattr(tokenize, 'FSTRING_END', -1): fs -= 1
        cur = []
        if sig and not any(x.type == tokenize.COMMENT or (x.type == tokenize.STRING and x.start[0] != x.end[0]) for x, _ in sig):
            out.append((sig[0][0].start[0], sig[-1][0].end[0], sig))
    return out
def _tail_exposure():
    """[(file:line, lead, cut)] one-line statements cut before a depth-0 tail token (the prefix parses on its own) that are not flagged"""
    miss = []
    for f in pipeline_scripts(ROOT):
        src = open(f).read(); lines = src.splitlines()
        for r0, r1, sig in _statements(src):
            if r0 != r1: continue
            line = lines[r0 - 1]; col0 = sig[0][0].start[1]
            if not _ctx_ok(line.strip()): continue
            depth = 0
            for x, in_fs in sig:
                if x.type == tokenize.OP and x.string in '([{': depth += 1
                elif x.type == tokenize.OP and x.string in ')]}': depth -= 1
                if in_fs or x.string not in _CUT or depth != (1 if x.string == '[' else 0): continue
                pre, rest = line[col0:x.start[1]].rstrip(), line[x.start[1]:].rstrip()
                if not pre or not _ctx_ok(pre): continue
                for lead in ('', ' note '):
                    if not swallowed_src(f'{pre}  #{lead}{rest}\n'): miss.append((f'{os.path.basename(f)}:{r0}', lead, f'{pre[-30:]}  #{lead}{rest[:60]}'))
    return miss
def _multiline_exposure():
    """[(file:line, cut)] multi-line bracketed statements cut on a non-last line before a tail token (the statement still parses) that are not flagged"""
    miss = []
    for f in pipeline_scripts(ROOT):
        src = open(f).read(); lines = src.splitlines(keepends=True)
        for r0, r1, sig in _statements(src):
            if r0 == r1: continue
            t0 = _ctx_tree(textwrap.dedent(''.join(lines[r0 - 1:r1])))
            if t0 is None: continue
            base, depth = ast.dump(t0), 0
            for x, in_fs in sig:
                if x.type == tokenize.OP and x.string in '([{': depth += 1
                elif x.type == tokenize.OP and x.string in ')]}': depth -= 1
                if depth == 0 or in_fs or x.end[0] == r1 or x.string not in _CUT or x.type == tokenize.STRING: continue
                row, col = x.start; L = lines[row - 1].rstrip('\n'); rest = L[col:].rstrip()
                if not rest or not L[:col].strip(): continue
                new = lines[:]; new[row - 1] = L[:col].rstrip() + '  # note ' + rest + '\n'; snip = textwrap.dedent(''.join(new[r0 - 1:r1]))
                t = _ctx_tree(snip)
                if t is not None and ast.dump(t) != base and not swallowed_src(snip): miss.append((f'{os.path.basename(f)}:{row}', f'{L[max(0, col - 30):col]} # note {rest[:60]}'))
    return miss
def _hit(src):
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f: f.write(src); p = f.name
    try: return bool(swallowed(p))
    finally: os.unlink(p)
def _exposure():
    """[(file:line, mode, fragment)] statements of the pipeline code that follow another one on their line and are not flagged when swallowed"""
    miss = []
    for f in pipeline_scripts(ROOT):
        src = open(f).read(); lines = src.splitlines()
        for node in ast.walk(ast.parse(src)):
            for fld in ('body', 'orelse', 'finalbody'):
                body = getattr(node, fld, None)
                if not isinstance(body, list): continue
                for a, b in zip(body, body[1:]):
                    if not isinstance(b, ast.stmt) or b.lineno != a.end_lineno or b.end_lineno != b.lineno: continue
                    rest = lines[b.lineno - 1][b.col_offset:]
                    if isinstance(b, ast.Expr) and not isinstance(b.value, (ast.Call, ast.NamedExpr)): continue    # a bare expression is not code
                    modes = ['strong']
                    if isinstance(b, (ast.Expr, ast.AugAssign, ast.Raise, ast.Assert, ast.Return, ast.Delete, ast.Global, ast.Nonlocal, ast.If, ast.For, ast.With)): modes.append('weak')
                    elif isinstance(b, ast.Assign) and (len(b.targets) > 1 or not isinstance(b.targets[0], ast.Name) or any(isinstance(n, (ast.Subscript, ast.Attribute)) or (isinstance(n, ast.Call) and n.args and not all(isinstance(x_, ast.Constant) for x_ in n.args)) for n in ast.walk(b.value))): modes.append('weak')
                    for mode in modes:
                        if mode == 'weak' and isinstance(b, ast.Expr) and isinstance(b.value, ast.Call) and not isinstance(b.value.func, ast.Attribute) and not b.value.keywords and all(isinstance(x_, ast.Constant) for x_ in b.value.args): continue
                        if not _hit(f"x = 1  # note{';' if mode == 'strong' else ''} {rest}\n"): miss.append((f'{os.path.basename(f)}:{b.lineno}', mode, rest[:100]))
    return miss
def run(verbose=False):
    res = {}
    res['code'] = [c for c in CODE if not _hit(f'x = 1  #{c}\n')]
    res['lit'] = [k for k, s in LIT.items() if not _hit(s)]
    res['accepted_flagged'] = [c for c in ACCEPTED if _hit(f'x = 1  #{c}\n')]
    res['prose_flagged'] = [c for c in PROSE if _hit(f'x = 1  #{c}\n')] + [k for k, s in LIT_PROSE.items() if _hit(s)]
    res['src'] = [k for k, s in SRC.items() if not swallowed_src(s)]
    res['src_prose_flagged'] = [k for k, s in SRC_PROSE.items() if swallowed_src(s)]
    res['exposure'] = _exposure() + _tail_exposure() + _multiline_exposure()
    n = len(CODE) + len(LIT) + len(SRC)
    print(f'CODE cases {n}: misses {len(res["code"]) + len(res["lit"]) + len(res["src"])} {res["code"] + res["lit"] + res["src"]}')
    print(f'ACCEPTED (G43.1) {len(ACCEPTED)}: flagged {len(res["accepted_flagged"])}; PROSE {len(PROSE) + len(LIT_PROSE)}: flagged {len(res["prose_flagged"])}'
          + (f' {res["prose_flagged"]}' if verbose else '') + f'; r15 fresh PROSE {len(SRC_PROSE)}: flagged {len(res["src_prose_flagged"])}'
          + (f' {[SRC_PROSE[k].split("#", 1)[1].strip() for k in res["src_prose_flagged"]]}' if verbose else ''))
    print(f'exposure (pipeline statements swallowed after `; ` / a prose word; one-line and multi-line tail cuts): misses {len(res["exposure"])}'
          + ''.join(f'\n  {m}' for m in res['exposure'][:40]))
    return res
def _failed(r): return bool(r['code'] or r['lit'] or r['src'] or r['exposure'])
def test_check_comments():
    assert not _failed(run())
if __name__ == '__main__':
    sys.exit(1 if _failed(run('-v' in sys.argv)) else 0)
