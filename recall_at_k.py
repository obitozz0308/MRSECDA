"""Shared recall@k (top-k disease recall) computation.

For each test circRNA, rank all diseases by predicted score and check whether
the true (test) associated diseases fall in the top-k. Micro-averaged:
    recall@k = (# test-positive pairs whose disease is in top-k of its circRNA)
             / (total # test-positive pairs).
"""
from __future__ import annotations

import numpy as np

KS = (10, 20, 30, 40, 50)


def compute_recall_at_k(S, test_pos_circ, test_pos_dis, ks=KS):
    """Compute recall@k from a full score matrix.

    Args:
        S: (n_circ, n_disease) predicted score matrix (higher = more associated).
        test_pos_circ: (n_pos,) int array of circRNA node indices.
        test_pos_dis: (n_pos,) int array of disease node indices.
        ks: iterable of top-k cutoffs.

    Returns:
        dict {k: recall} (floats, one per fold).
    """
    S = np.asarray(S, dtype=float)
    n_disease = S.shape[1]
    # rank of each disease for each circRNA (0 = highest score)
    order = np.argsort(-S, axis=1)   # descending indices
    rank = np.argsort(order, axis=1)  # rank[d] = position (0-based)

    n_pos = len(test_pos_circ)
    out = {}
    if n_pos == 0:
        return {int(k): 0.0 for k in ks}

    for k in ks:
        k_eff = min(int(k), n_disease)
        hits = 0
        for c, d in zip(test_pos_circ, test_pos_dis):
            if rank[int(c), int(d)] < k_eff:
                hits += 1
        out[int(k)] = hits / n_pos
    return out
