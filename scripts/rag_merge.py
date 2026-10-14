"""Region-adjacency-graph merging (vectorized mutual-best-neighbour rounds, Ward criterion).

Regions carry size n, feature sum S (n x d), a status code and (optionally) perimeter. Edges (a, b, shared boundary
length). A merge's cost is the Ward increase of within-region squared error
    w = n_a n_b / (n_a + n_b) * |mu_a - mu_b|^2
times a shape factor 1 + shape_w * (1 - shared / min(perim_a, perim_b)) that prefers merging along long shared
boundaries (compact stands). Each round, every region finds its cheapest admissible neighbour; mutual pairs below
`threshold` merge. Admissible: same status, n_a + n_b <= max_n.
"""
import numpy as np

def _unique_edges(a, b, ln):
    lo = np.minimum(a, b); hi = np.maximum(a, b); keep = lo != hi
    lo, hi, ln = lo[keep], hi[keep], ln[keep]
    key = lo.astype(np.int64) * (np.int64(1) << 32) + hi
    u, inv = np.unique(key, return_inverse=True)
    L = np.zeros(u.size, ln.dtype); np.add.at(L, inv, ln)
    return (u >> 32).astype(np.int64), (u & 0xFFFFFFFF).astype(np.int64), L

def _best_per_node(n_nodes, a, b, cost):
    """index of the cheapest edge per node (−1 if none)."""
    order = np.argsort(cost, kind='stable')
    best = np.full(n_nodes, -1, np.int64)
    # walk in increasing cost; first edge seen for a node is its best
    ends = np.concatenate([a[order], b[order]]); eidx = np.concatenate([order, order])
    pos = np.concatenate([np.arange(order.size), np.arange(order.size)])
    srt = np.lexsort((pos, ends))
    ends, eidx = ends[srt], eidx[srt]
    first = np.ones(ends.size, bool); first[1:] = ends[1:] != ends[:-1]
    best[ends[first]] = eidx[first]
    return best

def merge(n, S, status, a, b, ln, perim, threshold, max_n, shape_w=0.5, min_n=0, max_rounds=10_000, verbose=False):
    """Returns label array (len = #regions) mapping each input region to its merged region id (0..k-1)."""
    n = n.astype(np.float64).copy(); S = S.astype(np.float64).copy(); perim = perim.astype(np.float64).copy()
    status = status.copy(); N = n.size
    parent = np.arange(N)
    a, b, ln = _unique_edges(a.astype(np.int64), b.astype(np.int64), ln.astype(np.float64))
    phase_force = False
    for rnd in range(max_rounds):
        if a.size == 0: break
        mu_a = S[a] / n[a, None]; mu_b = S[b] / n[b, None]
        ward = n[a] * n[b] / (n[a] + n[b]) * ((mu_a - mu_b) ** 2).sum(1)
        shape = 1 + shape_w * (1 - np.clip(ln / np.minimum(perim[a], perim[b]), 0, 1))
        cost = ward * shape
        if not phase_force:
            ok = (status[a] == status[b]) & (n[a] + n[b] <= max_n)
            cost = np.where(ok, cost, np.inf)
        else:   # absorb regions smaller than min_n: only edges touching a small region; prefer same status
            small_a = n[a] < min_n; small_b = n[b] < min_n
            ok = small_a | small_b
            cost = np.where(ok, cost * np.where(status[a] == status[b], 1.0, 1e6), np.inf)
        best = _best_per_node(N, a, b, cost)
        e = np.arange(a.size)
        mutual = (best[a] == e) & (best[b] == e) & np.isfinite(cost)
        if not phase_force: mutual &= cost < threshold
        else:
            # a small region may merge into its best neighbour even if not mutual (neighbour picks only one partner)
            sm = np.where(n[a] < min_n, a, -1); sm2 = np.where(n[b] < min_n, b, -1)
            cand = ((best[a] == e) & (n[a] < min_n)) | ((best[b] == e) & (n[b] < min_n))
            cand &= np.isfinite(cost)
            # avoid chains: one merge per node per round
            idx = np.flatnonzero(cand); used = np.zeros(N, bool); sel = []
            for i in idx[np.argsort(cost[idx])]:
                if not used[a[i]] and not used[b[i]]:
                    used[a[i]] = used[b[i]] = True; sel.append(i)
            mutual = np.zeros(a.size, bool); mutual[np.array(sel, np.int64)] = True
        m = np.flatnonzero(mutual)
        if m.size == 0:
            if not phase_force and min_n > 0:
                phase_force = True; continue
            break
        keep, drop = a[m], b[m]
        # absorbed status: larger region's status wins
        status[keep] = np.where(n[drop] > n[keep], status[drop], status[keep])
        n[keep] += n[drop]; S[keep] += S[drop]
        perim[keep] = perim[keep] + perim[drop] - 2 * ln[m]
        parent[drop] = keep
        remap = np.arange(N); remap[drop] = keep
        a, b, ln = _unique_edges(remap[a], remap[b], ln)
        if verbose and rnd % 10 == 0:
            print(f'round {rnd}{" force" if phase_force else ""}: merged {m.size}, edges {a.size}', flush=True)
    # resolve parents
    for _ in range(64):
        p2 = parent[parent]
        if np.array_equal(p2, parent): break
        parent = p2
    _, lab = np.unique(parent, return_inverse=True)
    return lab
