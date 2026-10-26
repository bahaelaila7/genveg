"""Code swallowed by comments (audit rounds 5-8 found it 4x: an inline comment inserted mid-line turned the rest of the line into a comment): flags a
comment whose text parses as Python and contains several statements, an assignment with a method / keyword call, or an assignment of a
name the file's code reads but never binds.
  python scripts/check_comments.py FILE ... -> exit 1 and file:line for each hit."""
import ast, io, re, sys, tokenize, glob
def _code_names(src):
    """names the file's code reads but never binds (a swallowed assignment leaves exactly such a name: the NameError signature)"""
    try: tree = ast.parse(src)
    except SyntaxError: return set()
    load, bind = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name): (load if isinstance(n.ctx, ast.Load) else bind).add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): bind.add(n.name)
        elif isinstance(n, ast.arg): bind.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)): bind.update((a.asname or a.name).split('.')[0] for a in n.names)
    return load - bind
def swallowed(path):
    out = []; src = open(path).read(); unbound = _code_names(src)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type != tokenize.COMMENT: continue
        txt = tok.string.lstrip('#')
        for m in re.finditer(r'(?:^|[\s;)\]])([A-Za-z_][\w\.]*(?:\[[^\]]*\])?(?:\s*,\s*[A-Za-z_][\w\.]*)*\s*(?:=|\+=|-=|\*=)\s*[^=])', txt):
            frag = txt[m.start(1):]
            for cut in range(len(frag), 0, -1):          # the longest prefix that parses as statements containing an assignment
                try: tree = ast.parse(frag[:cut])
                except SyntaxError: continue
                nodes = list(ast.walk(tree)); codey = len(tree.body) >= 2 or any(isinstance(n, ast.Call) and (isinstance(n.func, ast.Attribute) or n.keywords) for n in nodes)
                tgt = {t.id for n in nodes if isinstance(n, (ast.Assign, ast.AugAssign)) for t in ast.walk(n) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
                codey = codey or bool(tgt & unbound)          # a plain-call assignment of a name the code reads but never binds (r9 M-1)
                if any(isinstance(n, (ast.Assign, ast.AugAssign)) for n in nodes) and codey:
                    out.append((tok.start[0], frag[:cut][:120]))
                break
    return out
if __name__ == '__main__':
  bad = 0
  for f in sys.argv[1:]:
    for ln, fr in swallowed(f): print(f'{f}:{ln}: {fr}'); bad += 1
  sys.exit(1 if bad else 0)
