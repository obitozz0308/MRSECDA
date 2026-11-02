"""
Pairwise Structural Features for circRNA-Disease Link Prediction.

Computes 4 scalar topological features for each candidate (circRNA, disease) pair
on the heterogeneous graph, inspired by SEAL/DE-GNN but without subgraph extraction.

Features:
  1. Jaccard coefficient: |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
  2. Adamic-Adar index: Σ_{k∈N(i)∩N(j)} 1/log(deg(k))
  3. Resource Allocation index: Σ_{k∈N(i)∩N(j)} 1/deg(k)
  4. Shortest path distance: d(i, j) on merged adjacency (capped at max_dist)
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from typing import Dict, Tuple, Optional, Union


class StructuralFeatureComputer:
    """
    Precomputes and caches structural features for circRNA-disease pairs.

    Designed to work with the heterogeneous graph built by build_hetero_graph().
    """

    # Canonical feature order (matches compute_features output):
    #   [0] jaccard  [1] adamic_adar  [2] resource_alloc  [3] inv_distance
    FEATURE_NAMES = ("jaccard", "adamic_adar", "resource_alloc", "inv_distance")

    def __init__(
        self,
        edge_index: np.ndarray,          # (2, E) — all edges in merged graph
        N_total: int,                     # total nodes (circ + disease + miRNA + lncRNA)
        N_circ: int,
        N_disease: int,
        node_counts: Dict[str, int],
        max_dist: int = 5,               # cap shortest path at this distance
        drop_feature: Optional[Union[str, list]] = None,  # per-feature ablation: drop one or more FEATURE_NAMES
    ):
        self.N_total = N_total
        self.N_circ = N_circ
        self.N_disease = N_disease
        self.max_dist = max_dist

        # Per-feature ablation mask (True = keep, False = drop)
        self.feature_mask = [True, True, True, True]
        if drop_feature is not None:
            if isinstance(drop_feature, str):
                drop_feature = [drop_feature]
            for f in drop_feature:
                if f not in self.FEATURE_NAMES:
                    raise ValueError(
                        f"drop_feature must be a subset of {self.FEATURE_NAMES}, "
                        f"got {drop_feature!r}")
                self.feature_mask[self.FEATURE_NAMES.index(f)] = False

        # ── Build sparse adjacency (symmetric, unweighted) ──
        src = edge_index[0]
        dst = edge_index[1]
        n_edges = src.shape[0] if hasattr(src, 'shape') else len(src)

        data = np.ones(n_edges, dtype=np.float32)
        self.adj = sp.coo_matrix(
            (data, (src, dst)), shape=(N_total, N_total)
        ).tocsr()
        # Symmetrize
        self.adj = self.adj.maximum(self.adj.T).tocsr()
        # Binarize
        self.adj.data = np.ones_like(self.adj.data)

        # ── Precompute degrees ──
        self.degrees = np.array(self.adj.sum(axis=1)).flatten()
        # Avoid log(0) / div-by-0
        self.degrees_safe = np.maximum(self.degrees, 1)

        # ── Precompute only the features the mask keeps (dropped features are
        # masked out in compute_features, so their precompute is pure waste). ──
        keep = self.feature_mask  # [jaccard, aa, ra, inv_dist]
        circ_offset = 0
        dis_offset = N_circ

        # Shortest-path distances (inv_dist). Kept arrays stay at their init
        # values (max_dist / 0) so compute_features indexing never fails.
        self._dist = np.full((N_disease, N_circ), max_dist, dtype=np.int32)
        if keep[3]:
            self._precompute_distances()

        # Common-neighbor count + union (jaccard)
        self._cn = np.zeros((N_disease, N_circ), dtype=np.int32)
        self._union = np.zeros((N_disease, N_circ), dtype=np.float32)
        if keep[0]:
            self._precompute_common_neighbors(circ_offset, dis_offset)
            self._precompute_union(circ_offset, dis_offset)

        # Adamic-Adar / Resource-Allocation
        self._aa = np.zeros((N_disease, N_circ), dtype=np.float32)
        self._ra = np.zeros((N_disease, N_circ), dtype=np.float32)
        if keep[1] or keep[2]:
            self._precompute_aa_ra(circ_offset, dis_offset)

    def _precompute_distances(self):
        """BFS from each disease node to all circRNA nodes.

        Uses scipy's C-level unweighted shortest-path (BFS) instead of a Python
        `deque` loop with per-node CSR row slicing — the original did ~62K scipy
        `__getitem__` row slices, which dominated setup time. Identical output:
        distances are capped at `max_dist` and unreachable nodes remain `max_dist`.
        """
        from scipy.sparse.csgraph import shortest_path
        N_d, N_c = self.N_disease, self.N_circ
        dis_offset = self.N_circ  # disease nodes start after circRNA nodes
        sources = np.arange(dis_offset, dis_offset + N_d, dtype=np.int32)

        # (N_d, N_total) unweighted BFS distances in C; inf = unreachable
        dist = shortest_path(self.adj, method='D', directed=False, unweighted=True,
                             indices=sources)
        dist_circ = dist[:, :N_c]                       # (N_d, N_c)
        dist_circ = np.minimum(dist_circ, self.max_dist)  # cap; inf → max_dist
        dist_circ = np.nan_to_num(dist_circ, posinf=self.max_dist)
        self._dist = dist_circ.astype(np.int32)

    def _precompute_common_neighbors(self, circ_offset: int, dis_offset: int):
        """Compute |N(dis) ∩ N(circ)| for all pairs via sparse dot product."""
        N_d, N_c = self.N_disease, self.N_circ
        # adj is N_total × N_total, binary
        # Extract rows for circ and disease nodes
        adj_circ = self.adj[circ_offset:circ_offset + N_c, :]    # N_c × N_total
        adj_dis = self.adj[dis_offset:dis_offset + N_d, :]        # N_d × N_total
        # CN = adj_dis @ adj_circ.T  →  (N_d × N_c)
        cn_mat = adj_dis @ adj_circ.T
        self._cn = np.array(cn_mat.toarray() if sp.issparse(cn_mat) else cn_mat, dtype=np.int32)

    def _precompute_aa_ra(self, circ_offset: int, dis_offset: int):
        """Compute AA and RA indices.

        AA = Σ_{k∈N(i)∩N(j)} 1/log(deg(k))
        RA = Σ_{k∈N(i)∩N(j)} 1/deg(k)

        We approximate via matrix operations:
        For each common neighbor k, contribution weight = 1/log(deg(k)) for AA.
        """
        # Use the adj rows directly
        adj_circ = self.adj[circ_offset:circ_offset + self.N_circ, :]  # N_c × N_total
        adj_dis = self.adj[dis_offset:dis_offset + self.N_disease, :]   # N_d × N_total

        keep = self.feature_mask  # [jaccard, aa, ra, inv_dist]

        # AA: adj_dis @ D_aa @ adj_circ.T = Σ_k adj_dis[i,k] * w_k * adj_circ[j,k]
        if keep[1]:
            log_deg = np.log(self.degrees_safe)
            log_deg = np.maximum(log_deg, 1e-8)  # avoid div by zero
            D_aa = sp.diags(1.0 / log_deg)
            aa_mat = adj_dis @ D_aa @ adj_circ.T
            self._aa = np.array(aa_mat.toarray() if sp.issparse(aa_mat) else aa_mat, dtype=np.float32)

        # RA similarly
        if keep[2]:
            D_ra = sp.diags(1.0 / self.degrees_safe)
            ra_mat = adj_dis @ D_ra @ adj_circ.T
            self._ra = np.array(ra_mat.toarray() if sp.issparse(ra_mat) else ra_mat, dtype=np.float32)

    def _precompute_union(self, circ_offset: int, dis_offset: int):
        """|N(i) ∪ N(j)| = deg(i) + deg(j) - |N(i) ∩ N(j)|"""
        deg_circ = self.degrees[circ_offset:circ_offset + self.N_circ]   # (N_c,)
        deg_dis = self.degrees[dis_offset:dis_offset + self.N_disease]    # (N_d,)
        # Broadcast: (N_d, N_c) = (N_d, 1) + (1, N_c) - (N_d, N_c)
        self._union = (deg_dis[:, np.newaxis] + deg_circ[np.newaxis, :] - self._cn).astype(np.float32)
        self._union = np.maximum(self._union, 1.0)  # avoid div by zero

    # ── Public API ──

    def compute_features(
        self,
        circ_indices: np.ndarray,   # (B,) or scalar — circRNA indices [0, N_circ)
        dis_indices: np.ndarray,    # (B,) or scalar — disease indices [0, N_disease)
    ) -> np.ndarray:                # (B, 4) — [Jaccard, AA, RA, 1/dist]
        """
        Compute the 4 structural features for given pairs.

        Features (in order):
          0: Jaccard coefficient  [0, 1]
          1: Adamic-Adar index    [0, ∞), normalized below
          2: Resource Allocation  [0, ∞), normalized below
          3: Inverse distance     1/d(i,j) ∈ (0, 1], 0 if unreachable
        """
        circ_idx = np.asarray(circ_indices, dtype=np.int64)
        dis_idx = np.asarray(dis_indices, dtype=np.int64)
        scalar_input = circ_idx.ndim == 0
        if scalar_input:
            circ_idx = np.array([circ_idx])
            dis_idx = np.array([dis_idx])

        n_pairs = len(circ_idx)

        # Jaccard = CN / union
        cn = self._cn[dis_idx, circ_idx]
        union = self._union[dis_idx, circ_idx]
        jaccard = cn / union

        # AA (log-scale: add 1 to avoid 0 → -inf, then normalize by typical range)
        aa = self._aa[dis_idx, circ_idx]
        # Apply log1p normalization to compress range
        aa_norm = np.log1p(aa) / np.log1p(aa.max()) if aa.max() > 0 else aa

        # RA similarly
        ra = self._ra[dis_idx, circ_idx]
        ra_norm = np.log1p(ra) / np.log1p(ra.max()) if ra.max() > 0 else ra

        # Inverse distance
        dist = self._dist[dis_idx, circ_idx].astype(np.float32)
        inv_dist = np.where(dist > 0, 1.0 / dist, 0.0)

        feats = np.stack([jaccard, aa_norm, ra_norm, inv_dist], axis=-1)
        feats = feats[..., self.feature_mask]

        if scalar_input:
            return feats[0]
        return feats

    @property
    def feat_names(self) -> list:
        return [n for n, keep in zip(self.FEATURE_NAMES, self.feature_mask) if keep]

    @property
    def feat_dim(self) -> int:
        return sum(self.feature_mask)


# =============================================================================
# Multi-View Structural Feature Computer (Phase 2a)
# =============================================================================

class MultiViewStructuralFeatureComputer:
    """
    Computes pairwise structural features on MULTIPLE subgraphs of the
    heterogeneous graph, then concatenates them.

    Motivation (DeepSeek recommendation):
      "在单一融合图上计算会丢失信息" — different edge types contribute to
      different topological patterns. Computing features on separate subgraphs
      preserves view-specific structural signals.

    Three views:
      View 0 (Sim+Assoc):  Type 0 (within-type similarity) + Type 1 (circ-dis assoc)
                           → "guilt by similarity + association" topology
      View 1 (MetaPath):   Type 1 (circ-dis assoc) + Type 2 (miRNA/lncRNA mediated)
                           → "miRNA-mediated indirect path" topology
      View 2 (Full):       Type 0 + Type 1 + Type 2 (all edges)
                           → complete heterogeneous topology (Phase 1 baseline)

    Total: 3 views × 4 features = 12-dim structural features.
    """

    FEAT_DIM_PER_VIEW = 4
    VIEW_NAMES = ["sim_assoc", "metapath", "full"]

    def __init__(
        self,
        edge_index_by_type: dict,       # Dict[int, Tensor] — PyG edge_index per type
        N_total: int,
        N_circ: int,
        N_disease: int,
        node_counts: Dict[str, int],
        max_dist: int = 5,
        drop_feature: Optional[Union[str, list]] = None,  # per-feature ablation (forwarded to each view)
        drop_view: Optional[Union[int, list]] = None,     # per-view ablation: drop subgraph view(s) 0/1/2
    ):
        self.N_circ = N_circ
        self.N_disease = N_disease

        import torch
        # View definitions: (name, edge types).  0=Sim+Assoc, 1=MetaPath, 2=Full
        view_defs = [
            ("sim_assoc", [0, 1]),
            ("metapath", [1, 2]),
            ("full", [0, 1, 2]),
        ]

        # Parse drop_view into a set of indices
        if drop_view is None:
            drop_idx = set()
        elif isinstance(drop_view, int):
            drop_idx = {drop_view}
        else:
            drop_idx = set(drop_view)

        self.active_views = [(name, types) for i, (name, types) in enumerate(view_defs)
                             if i not in drop_idx]
        self.num_views = len(self.active_views)

        self.computers = [
            StructuralFeatureComputer(
                self._merge_edges(edge_index_by_type, types),
                N_total, N_circ, N_disease, node_counts, max_dist, drop_feature,
            )
            for (name, types) in self.active_views
        ]

    @staticmethod
    def _merge_edges(edge_index_by_type: dict, types: list) -> 'np.ndarray':
        """Concatenate edge_index tensors for specified types into numpy array."""
        import torch
        parts = []
        for t in types:
            if t in edge_index_by_type and edge_index_by_type[t].shape[1] > 0:
                parts.append(edge_index_by_type[t])
        if parts:
            return torch.cat(parts, dim=1).numpy()
        else:
            import numpy as np
            return np.zeros((2, 0), dtype=np.int64)

    def compute_features(
        self,
        circ_indices: 'np.ndarray',
        dis_indices: 'np.ndarray',
    ) -> 'np.ndarray':                # (B, 12) — 3 views × 4 features
        """Compute multi-view structural features for given pairs."""
        import numpy as np
        feats = []
        for comp in self.computers:
            f = comp.compute_features(circ_indices, dis_indices)
            feats.append(f)
        return np.concatenate(feats, axis=-1)

    @property
    def feat_dim(self) -> int:
        return self.num_views * self.computers[0].feat_dim  # 3 × (4 or 3)

    @property
    def feat_names(self) -> list:
        base = self.computers[0].feat_names
        return [f"{v}_{f}" for (v, _types) in self.active_views for f in base]
