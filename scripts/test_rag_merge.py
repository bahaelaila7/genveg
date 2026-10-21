import os as _os; _SM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _WS = _os.environ.get('WORKSPACE', _os.path.dirname(_SM))   # relocatable paths
import sys, numpy as np; sys.path.insert(0, _SM + '/scripts')
from rag_merge import merge
# 1-D chain of 12 regions: two homogeneous groups (features 0 and 5) separated, + status split in group 1
f = np.array([0,0.1,0,0.05,0, 5,5.1,5,4.9, 5,5,5.05], float)[:, None]
n = np.ones(12); st = np.array([1]*9 + [2]*3)
a = np.arange(11); b = a + 1; ln = np.ones(11); perim = np.full(12, 4.0)
lab = merge(n, f * n[:, None], st, a, b, ln, perim, threshold=1.0, max_n=100)
print(lab); assert len(set(lab[:5])) == 1 and len(set(lab[5:9])) == 1 and len(set(lab[9:])) == 1 and lab[0] != lab[5] != lab[9]
lab = merge(n, f * n[:, None], st, a, b, ln, perim, threshold=1.0, max_n=3)
print(lab); assert np.bincount(lab).max() <= 3
# min_n absorption: region 5 alone (size 1) with status 1 neighbours
n2 = np.array([10, 1, 10.]); lab = merge(n2, np.array([[0], [100], [0.]]) * n2[:, None], np.array([1, 1, 1]), np.array([0, 1]), np.array([1, 2]), np.ones(2), np.full(3, 4.), threshold=1, max_n=100, min_n=5)
print(lab); assert lab[1] in (lab[0], lab[2])
print('OK')
