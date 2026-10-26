"""Code swallowed by comments (audit rounds 5-8 found it 4x: an inline comment inserted mid-line turned the rest of the line into a comment): flags a
comment whose text contains an assignment with a method / keyword call, or several statements, that parses as Python.
  python scripts/check_comments.py FILE ... -> exit 1 and file:line for each hit."""
import ast, io, re, sys, tokenize, glob
def swallowed(path):
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(open(path).read()).readline):
        if tok.type != tokenize.COMMENT: continue
        txt = tok.string.lstrip('#')
        for m in re.finditer(r'(?:^|[\s;)\]])([A-Za-z_][\w\.]*(?:\[[^\]]*\])?(?:\s*,\s*[A-Za-z_][\w\.]*)*\s*(?:=|\+=|-=|\*=)\s*[^=])', txt):
            frag = txt[m.start(1):]
            for cut in range(len(frag), 0, -1):          # the longest prefix that parses as statements containing an assignment
                try: tree = ast.parse(frag[:cut])
                except SyntaxError: continue
                nodes = list(ast.walk(tree)); codey = len(tree.body) >= 2 or any(isinstance(n, ast.Call) and (isinstance(n.func, ast.Attribute) or n.keywords) for n in nodes)
                if any(isinstance(n, (ast.Assign, ast.AugAssign)) for n in nodes) and codey:
                    out.append((tok.start[0], frag[:cut][:120]))
                break
    return out
if __name__ == '__main__':
  bad = 0
  for f in sys.argv[1:]:
    for ln, fr in swallowed(f): print(f'{f}:{ln}: {fr}'); bad += 1
  sys.exit(1 if bad else 0)
