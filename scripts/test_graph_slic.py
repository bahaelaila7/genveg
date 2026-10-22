import sys, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); from graph_slic import graph_slic, aggregate
from rag_merge import merge
# synthetic 60 x 60 grid of unit "superpixels": two feature blocks (left / right), a status-2 square, a barrier column (no edges) at x = 45
H = W = 60; N = H * W; yy, xx = np.divmod(np.arange(N), W)
f = np.where(xx < 30, 0.0, 5.0)[:, None] + 0.05 * np.random.default_rng(0).standard_normal((N, 1)); st = np.ones(N, int); st[(yy >= 10) & (yy < 20) & (xx >= 5) & (xx < 15)] = 2
n = np.ones(N); S = f * n[:, None]
a, b = [], []
for i in range(N):
    y, x = divmod(i, W)
    if x + 1 < W and x != 44: a.append(i); b.append(i + 1)
    if y + 1 < H: a.append(i); b.append(i + W)
a, b = np.array(a), np.array(b)
lab = graph_slic(n, S, st, yy.astype(float), xx.astype(float), a, b, G=10.0, m=1.0)
ok_status = all(len(set(st[lab == k])) == 1 for k in np.unique(lab))
ok_feat = all(len(set((xx[lab == k] < 30).tolist())) == 1 for k in np.unique(lab))       # no cluster across the feature edge
ok_barrier = all(len(set((xx[lab == k] <= 44).tolist())) == 1 for k in np.unique(lab))   # nor across the cut column
sizes = np.bincount(lab); print('clusters', lab.max() + 1, 'size q10/50/90', np.quantile(sizes, [.1, .5, .9]))
# connectivity of each cluster
from scipy.sparse import coo_matrix; from scipy.sparse.csgraph import connected_components
conn = True
for k in np.unique(lab):
    m_ = lab == k; ia = m_[a] & m_[b]; g = coo_matrix((np.ones(ia.sum()), (a[ia], b[ia])), shape=(N, N)); nc, c = connected_components(g, directed=False)
    conn &= len(np.unique(c[m_])) == 1
rn, rS, rst, ra, rb, rln, rper = aggregate(lab, n, S, st, a, b, np.ones(a.size), 4 * n)
inner = np.bincount(lab[a][lab[a] == lab[b]], minlength=lab.max() + 1); ok_agg = np.isclose(rn.sum(), N) and np.allclose(rper, 4 * np.bincount(lab) - 2 * inner) and not (ra == rb).any()
lab2 = merge(rn, rS, rst, ra, rb, rln, rper, 0.0, 1e12, min_n=30, status_hard=True)
print('status pure', ok_status, '| no cluster across the feature edge', ok_feat, '| none across the barrier', ok_barrier, '| connected', conn, '| aggregate ok', ok_agg,
      '| after MMU absorption: regions', lab2.max() + 1, 'min size', np.bincount(lab2, rn).min())
assert ok_status and ok_feat and ok_barrier and conn and ok_agg; print('OK')
