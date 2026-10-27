"""Regression suite of the swallowed-code detector (scripts/check_comments.py; docs/DECISIONS.md G12 / G43 / G45.5): every crafted case of the r9-r14
audits and the historical swallows. A CODE case must be flagged (0 misses); ACCEPTED cases are the prose-shaped simple assignments / no-argument
calls after a prose word that G43.1 accepts; PROSE cases are reported (a flagged prose comment is reworded). Also: every statement of the pipeline
code that follows another one on its line, swallowed after a `;` (any kind but a bare expression) or after a prose word (calls, keyword statements,
dense assignments), must be flagged.
  .venv/bin/python -I scripts/tests/test_check_comments.py [-v]   -> exit 1 on any miss (also runs under pytest)"""
import ast, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(HERE))
from check_comments import swallowed, pipeline_scripts
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
LIT_PROSE = {
 'prose in a list': "L = ['a.tif',  # the mosaic\n     'z.tif']\n",
 'prose after the last element': "L = ['a.tif',  # see b\n     ]\n",
 'aside closing a call': "x = f(a,  # see, b)\n      c)\n",
 'aside in a dict': "ST = {\n  'a': dict(code=['x.py']),   # (clean: g02b.feats' forest filter, r13 A-11)\n  'c': 1}\n",
}
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
    res['exposure'] = _exposure()
    n = len(CODE) + len(LIT)
    print(f'CODE cases {n}: misses {len(res["code"]) + len(res["lit"])} {res["code"] + res["lit"]}')
    print(f'ACCEPTED (G43.1) {len(ACCEPTED)}: flagged {len(res["accepted_flagged"])}; PROSE {len(PROSE) + len(LIT_PROSE)}: flagged {len(res["prose_flagged"])}'
          + (f' {res["prose_flagged"]}' if verbose else ''))
    print(f'exposure (pipeline statements swallowed after `; ` / a prose word): misses {len(res["exposure"])}' + ''.join(f'\n  {m}' for m in res['exposure'][:40]))
    return res
def test_check_comments():
    r = run(); assert not r['code'] and not r['lit'] and not r['exposure']
if __name__ == '__main__':
    r = run('-v' in sys.argv); sys.exit(1 if r['code'] or r['lit'] or r['exposure'] else 0)
