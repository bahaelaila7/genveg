"""Code swallowed by comments: an inline comment inserted mid-line turns the rest of the line into a comment (audit rounds 5-8: five instances; r13
D-10: the r8-r12 heuristic missed bare calls, re-bound names, augmented / subscript / tuple assignments, del, if; r14 D-6 / F-1, G45.5: keyword
statements, splits after any punctuation, swallowed literal elements and statement tails).

Rule. A comment is split at its STRONG points (its start, after each ';', ':', ')' / ']') and at its WEAK points (every other start of a word or
symbol after a space or a punctuation mark: `—`, `→`, `--`, `%`, quotes, `-`, `?` ...). The remainder from a split point (trailing `)` / `]` / `.`
dropped when that is what keeps it from parsing) is code when it parses fully as Python and contains code:
  - at a strong point: any statement but a bare expression — an assignment (alias and walrus included), del, a block or def, return, import, raise,
    assert, continue, break, pass, global / nonlocal, or a call written `f(` (a bare `name()` included);
  - at a weak point (the remainder follows a prose word, as in `... the forest-share filter trp_ = set(vpid[tr_v])`): only code-DENSE statements —
    an assignment whose value holds a call with a non-literal argument, a subscript or an attribute (or with a subscript / attribute / tuple target),
    an augmented assignment, a walrus, del, a block, return, import, raise, assert, global / nonlocal, a method call or a call with a non-literal
    argument or a keyword. Prose such as `at x = 45`, `invert year = year + day-of-year / 366`, `then w = 3` or `see main()` stays allowed (G43.1);
  - in context: a comment inside a bracketed literal / call whose remainder, put back in place of the comment, still parses and adds elements that
    are not bare names (`'a.tif',  # note 'b.tif', 'c.tif',`); an inline comment whose remainder starts with a binary operator, a subscript or an
    attribute and, appended to the code before the comment, still parses (`y = f(x)  # + w_age * L_age`); a remainder `, name=value)` closing a
    bracket the comment did not open.
Prose rarely parses fully; swallowed code always does (it was a working statement). A prose comment the rule flags is reworded. The regression
suite is scripts/tests/test_check_comments.py.
  python scripts/check_comments.py [FILE ...]   (no files: every pipeline script, pipeline_scripts()) -> exit 1 and file:line for each hit."""
import ast, io, os, re, sys, tokenize
_BLOCK = (ast.Delete, ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Return, ast.Import, ast.ImportFrom, ast.Try,
          ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Raise, ast.Assert, ast.Global, ast.Nonlocal)
_STRONG = (ast.Continue, ast.Break, ast.Pass)                       # flagged only at strong points (prose: "the checks pass", "may break")
def _call_written(c, txt):
    """a call written `f(` (no space before the parenthesis: prose writes `word (aside)`)"""
    f = c.func; line = txt.splitlines()[f.end_lineno - 1]
    return line[f.end_col_offset:f.end_col_offset + 1] == '('
def _alias(s):
    return isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) and isinstance(s.value, ast.Name)
def _dense(node, txt):
    """code-dense: a written call with a non-literal argument / keywords or a method call, a subscript or an attribute"""
    for n in ast.walk(node):
        if isinstance(n, (ast.Subscript, ast.Attribute)): return True
        if isinstance(n, ast.Call) and _call_written(n, txt) and (n.keywords or any(not isinstance(a, ast.Constant) for a in n.args)): return True
    return False
def _stmt_code(s, txt, dense):
    if isinstance(s, _BLOCK) or isinstance(s, ast.AugAssign): return True
    if isinstance(s, _STRONG): return not dense
    if isinstance(s, ast.Expr) and isinstance(s.value, ast.NamedExpr): return True
    if _alias(s): return not dense
    if isinstance(s, ast.AnnAssign): return s.value is not None and (not dense or _dense(s, txt))
    if isinstance(s, ast.Assign): return not dense or len(s.targets) > 1 or not isinstance(s.targets[0], ast.Name) or _dense(s.value, txt)
    if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and _call_written(s.value, txt):   # a statement-level call
        c = s.value
        return not dense or isinstance(c.func, ast.Attribute) or bool(c.keywords) or any(not isinstance(a, ast.Constant) for a in c.args)
    return False                                                     # a bare expression (prose: `g02b.EVK`, `BA(v2)/BA(v1)`, `0.5 % 'harvest'`)
def _parse(s):
    try: return ast.parse(s)
    except Exception: return None
def _stmt_tree(rem):
    """the remainder parsed, else with trailing closers / periods dropped (`y = f(x))`, `y = f(x).`), else None"""
    while rem:
        t = _parse(rem)
        if t is not None: return t, rem
        if rem[-1] not in ')].,': return None, rem
        rem = rem[:-1].rstrip()
    return None, rem
def _splits(txt):
    """[(position, weak)]: strong = start, after ';' / ':' / ')' / ']'; weak = every other non-space after a space or a non-word character"""
    strong = {0} | {m.end() for m in re.finditer(r'[;:)\]]', txt)}
    weak = {i for i in range(1, len(txt)) if not txt[i].isspace() and (txt[i - 1].isspace() or not (txt[i - 1].isalnum() or txt[i - 1] == '_'))} - strong
    return sorted([(p, False) for p in strong] + [(p, True) for p in weak])
_OPS = re.compile(r'(?:\*\*|//|<<|>>|<=|>=|==|!=|[-+*/%@&|^<>])(?=\s)|\.(?=[A-Za-z_])|\[')
def _ctx_parse(src):
    """the logical line parses (a block header gets a body; else / elif / except / finally a leading block)"""
    for w in ('{}', '{}\n pass', 'if 1: pass\n{}\n pass', 'try: pass\n{}\n pass'):
        t = _parse(w.format(src))
        if t is not None: return t
    return None
def _element_like(suffix):
    """the remainder as elements of a list / set / dict / call: one that is not a bare name (a word of prose)"""
    for w in ('[{}\n]', '{{{}\n}}', '_f({}\n)'):
        t = _parse('_ = ' + w.format(suffix))
        if t is None: continue
        v = t.body[0].value; els = v.elts if isinstance(v, (ast.List, ast.Set)) else v.keys + v.values if isinstance(v, ast.Dict) else v.args + [k.value for k in v.keywords]
        if isinstance(v, ast.Call) and v.keywords: return True
        if any(e is not None and not isinstance(e, ast.Name) for e in els): return True
    return False
def _logical_lines(toks):
    """per comment token index: (index of the first token of its logical line, index of its NEWLINE / ENDMARKER, bracket depth at the comment)"""
    out, start, depth = {}, None, 0; pend = []
    for i, t in enumerate(toks):
        if t.type == tokenize.OP and t.string in '([{': depth += 1
        elif t.type == tokenize.OP and t.string in ')]}': depth = max(depth - 1, 0)
        if t.type == tokenize.COMMENT: pend.append((i, depth)); continue
        if t.type in (tokenize.NEWLINE, tokenize.ENDMARKER) or (t.type == tokenize.NL and depth == 0):
            for j, dp in pend: out[j] = (start, i, dp)
            pend, start = [], None; continue
        if start is None and t.type not in (tokenize.NL, tokenize.ENCODING, tokenize.INDENT, tokenize.DEDENT): start = i
    return out
def swallowed(path):
    """[(line, fragment)] of the comments in `path` that contain code"""
    out = []; src = open(path).read(); lines = src.splitlines(keepends=True)
    try: toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, SyntaxError) as e: return [((e.args[1] or (0,))[0] if len(e.args) > 1 else 0, f'(the file does not tokenize: {e.args[0]})')]
    ll = _logical_lines(toks)
    def seg(a, b):                                                   # source between two (row, col) positions
        if a[0] == b[0]: return lines[a[0] - 1][a[1]:b[1]] if a[0] <= len(lines) else ''
        return lines[a[0] - 1][a[1]:] + ''.join(lines[a[0]:b[0] - 1]) + (lines[b[0] - 1][:b[1]] if b[0] <= len(lines) else '')
    for i, tok in enumerate(toks):
        if tok.type != tokenize.COMMENT: continue
        txt = tok.string[1:]; hit = None
        for p, dense in _splits(txt):                                # the remainder as statements
            t, rem = _stmt_tree(txt[p:].strip())
            if t is not None and any(_stmt_code(s, rem, dense) for s in t.body): hit = rem; break
        st, nl, depth = ll.get(i, (None, None, 0))
        if hit is None and st is not None:                           # in context: literal elements, statement tails, call-argument tails
            before = seg(toks[st].start, tok.start); after = seg(tok.end, toks[nl].start)
            base = _ctx_parse(before + after)
            unmatched = sum(txt.count(c) for c in ')]}') > sum(txt.count(c) for c in '([{')
            for p, _ in ([] if base is None else _splits(txt)):
                if p >= len(txt) or txt[p].isspace(): continue
                suf = txt[p:].rstrip()
                if depth > 0: ok = (suf.endswith(',') or after.lstrip()[:1] in (')', ']', '}')) and _element_like(suf.rstrip(','))   # (at an element boundary)
                else: ok = bool(_OPS.match(suf)) and (p == 0 or txt[p - 1].isspace())     # (an inline comment: code before it on its line)
                if ok:
                    t = _ctx_parse(before + ' ' + suf + after)
                    if t is not None and ast.dump(t) != ast.dump(base): hit = suf.strip(); break
            for m in (re.finditer(',', txt) if hit is None and unmatched else ()):
                suf = txt[m.start():].rstrip(); c_ = f'_f(_x{suf}\n'; t = _parse(c_)
                if t is not None and (t.body[0].value.keywords or any(_dense(a, c_) for a in t.body[0].value.args[1:])): hit = suf; break
        if hit is not None: out.append((tok.start[0], hit[:120]))
    return out
def pipeline_scripts(root):
    """The canonical list (absolute paths) of the scripts the genveg driver hashes and scans: every stage script, the shared modules, stands_cfg.py
    (imported by features.py), stands_stages.py (imported by the driver), the Region 4 raster builders, the driver and this file. gv_verify T0 scans
    exactly this list (r13 M-5)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('_genveg_driver', os.path.join(root, 'scripts', 'genveg_pipeline.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return [os.path.join(root, 'scripts', f) for f in m.scanned_scripts()]
if __name__ == '__main__':
  fs = sys.argv[1:] or pipeline_scripts(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  bad = 0
  for f in fs:
    for ln, fr in swallowed(f): print(f'{f}:{ln}: {fr}'); bad += 1
  sys.exit(1 if bad else 0)
