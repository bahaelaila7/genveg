"""Code swallowed by comments: an inline comment inserted mid-line turns the rest of the line into a comment (audit rounds 5-8: five instances; r13
D-10: the r8-r12 heuristic missed bare calls, re-bound names, augmented / subscript / tuple assignments, del, if).

Rule. A comment is split at its STRONG points (its start, after each ';', after each ')' / ']') and at its WEAK points (every other word start). The
remainder of the comment from a split point is code when it parses FULLY as Python (ast.parse of the whole remainder) and contains code:
  - at a strong point: an assignment (=, +=, annotated with a value), del, if / for / while / with / return / import, or a call written `f(` that has
    arguments or is a method call (a bare `name()` alone is a prose reference to a function);
  - at a weak point (the remainder follows a prose word, as in `... the forest-share filter trp_ = set(vpid[tr_v])`; not after an operator, so
    `A + Am(z)` is a formula): only code-DENSE statements — an assignment whose value holds a call with a non-literal argument, a subscript or an
    attribute (or with a subscript / attribute / tuple target), an augmented assignment, del, a block, or a method call. Prose such as `at x = 45`,
    `invert year = year + day-of-year / 366` or `passes N(0, I)` stays allowed;
  - never: a lone alias `name = other_name` (prose such as `severity = max`).
Prose rarely parses fully; swallowed code always does (it was a working statement). A prose comment the rule flags is reworded.
  python scripts/check_comments.py [FILE ...]   (no files: every pipeline script, pipeline_scripts()) -> exit 1 and file:line for each hit."""
import ast, io, os, re, sys, tokenize
_BLOCK = (ast.Delete, ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Return, ast.Import, ast.ImportFrom, ast.Try)
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
    if _alias(s): return False
    if isinstance(s, _BLOCK) or isinstance(s, ast.AugAssign): return True
    if isinstance(s, ast.AnnAssign): return s.value is not None and (not dense or _dense(s, txt))
    if isinstance(s, ast.Assign): return not dense or len(s.targets) > 1 or not isinstance(s.targets[0], ast.Name) or _dense(s.value, txt)
    if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and _call_written(s.value, txt):   # a statement-level call
        c = s.value
        return isinstance(c.func, ast.Attribute) or bool(c.keywords) or (not dense and bool(c.args))   # (after prose words a bare `f(x)` is notation: `N(0, I)`)
    return False                                                     # a bare expression (prose: `g02b.EVK`, `BA(v2)/BA(v1)`, `0.5 % 'harvest'`)
def _code(tree, txt, dense):
    return any(_stmt_code(s, txt, dense) for s in tree.body)
def _splits(txt):
    strong = {0} | {m.end() for m in re.finditer(r'[;)\]]', txt)}
    weak = {m.end() for m in re.finditer(r'[\w.,:]\s+(?=\S)', txt)} - strong      # a word start after a word (not after an operator: `A + Am(z)` is a formula)
    return sorted([(p, False) for p in strong] + [(p, True) for p in weak])
def swallowed(path):
    """[(line, fragment)] of the comments in `path` that contain code"""
    out = []; src = open(path).read()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type != tokenize.COMMENT: continue
        txt = tok.string.lstrip('#')
        for p, dense in _splits(txt):
            rem = txt[p:].strip()
            if not rem: continue
            try: tree = ast.parse(rem)
            except (SyntaxError, ValueError): continue
            if _code(tree, rem, dense): out.append((tok.start[0], rem[:120])); break
    return out
def pipeline_scripts(root):
    """The canonical list (absolute paths) of the scripts the genveg driver hashes and scans: every stage script, the shared modules, stands_cfg.py
    (imported by features.py), the Region 4 raster builders, the driver and this file. gv_verify T0 scans exactly this list (r13 M-5)."""
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
