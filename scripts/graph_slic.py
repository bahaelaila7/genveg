"""Hierarchical SLIC, level 2 (user, 2026-10-22): SLIC on the superpixel graph instead of the Ward merge (segmentation.merge_method: graph_slic).
Superpixels are points with a mean feature vector (the merge feature space) and a centroid; stands are clusters grown SLIC-style:
  seeds      one per grid cell of side G = sqrt(graph_slic_ha / 0.09) px, per land status (snapped to the nearest superpixel of that status)
  assign     each superpixel joins the centre (same status) within 2 G minimising |mu_i - c_mu|^2 / s2 + m^2 |xy_i - c_xy|^2 / G^2
             (s2 = the area-weighted total feature variance, so m = graph_slic_compactness has the same meaning for any feature space)
  update     centres = area-weighted means of features and centroids; graph_slic_iters iterations
  connect    a cluster that is not connected in the superpixel graph (barriers cut the graph) is split into its connected pieces
Then the existing MMU absorption (rag_merge.merge force phase, status-hard) removes pieces below the MMU. Size is controlled by G, not by a threshold."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def graph_slic(n, S, status, cy, cx, a, b, G, m, iters=10):
    """n (N,) px, S (N, d) feature sums, status (N,), cy / cx (N,) centroid row / col, edges a / b (superpixel indices). Returns labels (N,) 0..K-1,
    every label a connected set of superpixels of one status."""
    N = n.size; mu = S / n[:, None]; xy = np.stack([cy, cx], 1).astype(np.float64)
    w = n / n.sum(); s2 = float((w[:, None] * (mu - (w[:, None] * mu).sum(0)) ** 2).sum()) or 1.0
    lab = np.full(N, -1, np.int64); nxt = 0
    for st in np.unique(status):
        idx = np.flatnonzero(status == st)
        if idx.size == 0: continue
        P = xy[idx]; lo, hi = P.min(0), P.max(0)
        gy, gx = np.meshgrid(np.arange(lo[0] + G / 2, hi[0] + G / 2, G), np.arange(lo[1] + G / 2, hi[1] + G / 2, G), indexing='ij')
        t = cKDTree(P); d, j = t.query(np.stack([gy.ravel(), gx.ravel()], 1), distance_upper_bound=G / 2 * 1.42)
        seeds = np.unique(j[np.isfinite(d)])
        if seeds.size == 0: seeds = np.array([0])
        cmu, cxy = mu[idx[seeds]].copy(), P[seeds].copy()
        for _ in range(iters):
            ct = cKDTree(cxy); cand = ct.query_ball_point(P, 2 * G)
            a_ = np.full(idx.size, -1, np.int64)
            for k, cs in enumerate(cand):
                if not cs: continue
                cs = np.asarray(cs); dd = ((mu[idx[k]] - cmu[cs]) ** 2).sum(1) / s2 + m * m * ((P[k] - cxy[cs]) ** 2).sum(1) / (G * G)
                a_[k] = cs[np.argmin(dd)]
            orphan = a_ < 0                                                         # no centre within 2G: become centres themselves
            if orphan.any():
                newc = np.arange(cmu.shape[0], cmu.shape[0] + orphan.sum()); a_[orphan] = newc
                cmu = np.vstack([cmu, mu[idx[orphan]]]); cxy = np.vstack([cxy, P[orphan]])
            wn = n[idx]; K = cmu.shape[0]; cnt = np.bincount(a_, wn, K); keep = cnt > 0
            cmu = np.stack([np.bincount(a_, wn * mu[idx][:, j_], K) for j_ in range(mu.shape[1])], 1)[keep] / cnt[keep][:, None]
            cxy = np.stack([np.bincount(a_, wn * P[:, j_], K) for j_ in range(2)], 1)[keep] / cnt[keep][:, None]
            remap = -np.ones(K, np.int64); remap[keep] = np.arange(keep.sum()); a_ = remap[a_]
        lab[idx] = a_ + nxt; nxt += cmu.shape[0]
    # connectivity: split every cluster into its connected pieces in the superpixel graph
    same = lab[a] == lab[b]
    g = coo_matrix((np.ones(same.sum()), (a[same], b[same])), shape=(N, N))
    _, comp = connected_components(g, directed=False)
    _, out = np.unique(comp, return_inverse=True)                                  # comp already separates clusters (edges only within a cluster)
    return out

def aggregate(lab, n, S, status, a, b, ln, perim):
    """Region-level arrays of a labelling (for the MMU absorption by rag_merge.merge): n, S, status (one per region), region edges + shared lengths,
    perimeters (sum of member perimeters minus twice the internal shared boundary)."""
    R = lab.max() + 1; rn = np.bincount(lab, n, R); rS = np.zeros((R, S.shape[1])); np.add.at(rS, lab, S)
    rst = np.zeros(R, status.dtype); rst[lab] = status
    la, lb = lab[a], lab[b]; internal = la == lb
    rper = np.bincount(lab, perim, R) - 2 * np.bincount(la[internal], ln[internal], R)
    return rn, rS, rst, la[~internal], lb[~internal], ln[~internal], rper
