"""
Meta-Path Neighbor Encoding (MPNE) — Multi-hop topology via heterogeneous meta-paths.

Replaces MSNE (random-walk type distributions — failed: 3 scales near-identical on
small graphs). Instead of counting "what types are reachable at step k" (which
converges instantly), MPNE computes per-node features by aggregating actual node
embeddings along specific meta-paths.

Key difference from MSNE:
  - MSNE:  type distribution histogram (converges fast → scales identical)
  - MPNE:  aggregate node embeddings along meta-path instances (genuinely multi-scale)

Meta-paths (3 paths, each captures a distinct biological relationship):
  MP1 (CMC): circRNA → miRNA → circRNA
      Captures: circRNA functional similarity via shared miRNA targets.
      For circRNA c: "what do other circRNAs that share my miRNAs look like?"

  MP2 (DMD): disease → miRNA → disease
      Captures: disease similarity via shared miRNA involvement.
      For disease d: "what do other diseases that share my miRNAs look like?"

  MP3 (CMD): circRNA → miRNA → disease
      Captures: the molecular sponge mechanism (circRNA regulates disease via miRNA).
      For circRNA c: "what diseases does my miRNA neighborhood connect to?"
      For disease d: "what circRNAs target the miRNAs linked to this disease?"

Each meta-path produces a (N_total, hidden_dim) feature matrix via 2-step mean-pooling
aggregation. These can be used in two modes:
  - CONCAT: concatenated to input features → TypeSpecificProjection handles dim
  - CRT:    each meta-path as an independent CRT token alongside HGT token

Zero learned parameters in the aggregation itself. Optional learnable projection
after aggregation (Linear(hidden_dim → hidden_dim)) per meta-path.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict


class MetaPathAggregator:
    """Non-learnable meta-path neighbor aggregation.

    Computes per-node embeddings by mean-pooling features along meta-path instances.
    All aggregation is done in the provided feature space (typically hidden_dim=96).

    Args:
        edge_index_by_type: Dict[int, LongTensor] — per-type edges from hetero graph
        N_circ, N_disease, N_mirna, N_lncrna: node counts
    """

    def __init__(
        self,
        edge_index_by_type: Dict[int, torch.Tensor],
        N_circ: int,
        N_disease: int,
        N_mirna: int,
        N_lncrna: int,
    ):
        self.N_c = N_circ
        self.N_d = N_disease
        self.N_m = N_mirna
        self.N_l = N_lncrna
        self.N_total = N_circ + N_disease + N_mirna + N_lncrna

        self.off_d = N_circ
        self.off_m = N_circ + N_disease
        self.off_l = N_circ + N_disease + N_mirna

        # ── Extract adjacency sub-blocks from edge_index_by_type ──
        # Type 2 contains: circRNA↔miRNA, miRNA↔disease, lncRNA↔disease
        ei2 = edge_index_by_type.get(2, torch.zeros((2, 0), dtype=torch.long))
        ei2_np = ei2.numpy()

        # Build sparse sub-blocks (directed, then symmetrize for undirected graph)
        self._build_adj_blocks(ei2_np)

    def _build_adj_blocks(self, ei2: np.ndarray):
        """Extract C↔M, M↔D, C↔D sub-blocks from edge_index_by_type[2]."""
        N_c, N_d, N_m = self.N_c, self.N_d, self.N_m
        off_d, off_m = self.off_d, self.off_m
        N_total = self.N_total

        if ei2.shape[1] == 0:
            # No type-2 edges — all adj blocks are zero
            self.A_cm = sp.csr_matrix((N_c, N_m), dtype=np.float32)
            self.A_md = sp.csr_matrix((N_m, N_d), dtype=np.float32)
            self.A_cd_direct = sp.csr_matrix((N_c, N_d), dtype=np.float32)
            return

        src = ei2[0]
        tgt = ei2[1]

        # ── circRNA ↔ miRNA ──
        mask_cm = (src < N_c) & (tgt >= off_m) & (tgt < off_m + N_m)
        mask_mc = (src >= off_m) & (src < off_m + N_m) & (tgt < N_c)
        cm_src = np.concatenate([src[mask_cm], tgt[mask_mc]])
        cm_tgt = np.concatenate([tgt[mask_cm] - off_m, src[mask_mc] - off_m])
        cm_data = np.ones(len(cm_src), dtype=np.float32)
        A_cm_full = sp.coo_matrix(
            (cm_data, (cm_src, cm_tgt)), shape=(N_c, N_m)
        ).tocsr()
        A_cm_full.data = np.ones_like(A_cm_full.data)  # binarize
        self.A_cm = A_cm_full

        # ── miRNA ↔ disease ──
        mask_md = (src >= off_m) & (src < off_m + N_m) & (tgt >= off_d) & (tgt < off_d + N_d)
        mask_dm = (src >= off_d) & (src < off_d + N_d) & (tgt >= off_m) & (tgt < off_m + N_m)
        md_src = np.concatenate([src[mask_md] - off_m, tgt[mask_dm] - off_m])
        md_tgt = np.concatenate([tgt[mask_md] - off_d, src[mask_dm] - off_d])
        md_data = np.ones(len(md_src), dtype=np.float32)
        A_md_full = sp.coo_matrix(
            (md_data, (md_src, md_tgt)), shape=(N_m, N_d)
        ).tocsr()
        A_md_full.data = np.ones_like(A_md_full.data)
        self.A_md = A_md_full

        # ── circRNA ↔ disease (direct, via type 1 edges) — for CDC meta-path ──
        # Type 1 contains circRNA-disease training associations
        # We'll extract this separately, but it's also useful for normalization
        self.A_cd_direct = sp.csr_matrix((N_c, N_d), dtype=np.float32)

    def set_cd_edges(self, edge_index_by_type: Dict[int, torch.Tensor]):
        """Set circRNA-disease direct edges from type 1 (for CDC meta-path)."""
        N_c, N_d = self.N_c, self.N_d
        off_d = self.off_d

        ei1 = edge_index_by_type.get(1, torch.zeros((2, 0), dtype=torch.long))
        ei1_np = ei1.numpy()
        if ei1_np.shape[1] == 0:
            return

        src = ei1_np[0]
        tgt = ei1_np[1]

        # circ → disease (direct)
        mask_cd = (src < N_c) & (tgt >= off_d) & (tgt < off_d + N_d)
        mask_dc = (src >= off_d) & (src < off_d + N_d) & (tgt < N_c)

        cd_src = np.concatenate([src[mask_cd], tgt[mask_dc]])
        cd_tgt = np.concatenate([tgt[mask_cd] - off_d, src[mask_dc] - off_d])
        cd_data = np.ones(len(cd_src), dtype=np.float32)
        A_cd = sp.coo_matrix(
            (cd_data, (cd_src, cd_tgt)), shape=(N_c, N_d)
        ).tocsr()
        A_cd.data = np.ones_like(A_cd.data)
        self.A_cd_direct = A_cd

    # ── Meta-Path Aggregation ──

    @staticmethod
    def _row_norm(A: sp.spmatrix) -> sp.spmatrix:
        """Row-normalize sparse matrix: D^{-1} A."""
        deg = np.array(A.sum(axis=1)).flatten()
        deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
        D_inv = sp.diags(deg_inv)
        return D_inv @ A

    @staticmethod
    def _col_norm(A: sp.spmatrix) -> sp.spmatrix:
        """Column-normalize sparse matrix: A @ D_col^{-1}.

        Each column sums to 1 (or 0 if column has no entries).
        """
        col_sum = np.array(A.sum(axis=0)).flatten()
        col_inv = np.where(col_sum > 0, 1.0 / col_sum, 0.0)
        D_inv = sp.diags(col_inv)
        return A @ D_inv

    def aggregate_cmc(self, h: np.ndarray) -> np.ndarray:
        """CMC meta-path: circRNA → miRNA → circRNA.

        Two-step mean-pooling:
          Step 1 (C→M): each miRNA ← mean of neighbor circRNA embeddings
          Step 2 (M→C): each circRNA ← mean of neighbor miRNA embeddings

        Args:
            h: (N_total, D) numpy array — node embeddings

        Returns:
            h_cmc: (N_total, D) — non-zero for circRNA and miRNA nodes.
        """
        N_c, N_m = self.N_c, self.N_m
        N_total, D = h.shape
        off_m = self.off_m

        if self.A_cm.nnz == 0:
            return np.zeros((N_total, D), dtype=np.float32)

        h_c = h[:N_c]  # (N_c, D)

        # Step 1: circRNA → miRNA (mean pool)
        # row_norm(A_cm^T): (N_m, N_c) — each miRNA row sums to 1
        # h_m = row_norm(A_cm^T) @ h_c  →  (N_m, D)
        A_mc_norm = self._row_norm(self.A_cm.T)  # (N_m, N_c)
        h_m_from_c = np.asarray(A_mc_norm @ h_c, dtype=np.float32)

        # Step 2: miRNA → circRNA (mean pool)
        # row_norm(A_cm): (N_c, N_m) — each circRNA row sums to 1
        # h_c_cmc = row_norm(A_cm) @ h_m_from_c  →  (N_c, D)
        A_cm_norm = self._row_norm(self.A_cm)  # (N_c, N_m)
        h_c_cmc = np.asarray(A_cm_norm @ h_m_from_c, dtype=np.float32)

        # miRNA embedding = midpoint (aggregated from circRNA neighbors)
        h_m_cmc = h_m_from_c

        result = np.zeros((N_total, D), dtype=np.float32)
        result[:N_c] = h_c_cmc
        result[off_m:off_m + N_m] = h_m_cmc
        return result

    def aggregate_dmd(self, h: np.ndarray) -> np.ndarray:
        """DMD meta-path: disease → miRNA → disease.

        Two-step mean-pooling:
          Step 1 (D→M): each miRNA ← mean of neighbor disease embeddings
          Step 2 (M→D): each disease ← mean of neighbor miRNA embeddings

        A_md is (N_m, N_d). A_dm = A_md^T is (N_d, N_m).

        Args:
            h: (N_total, D) numpy array

        Returns:
            h_dmd: (N_total, D) — non-zero for disease and miRNA nodes.
        """
        N_disease = self.N_d
        N_m = self.N_m
        N_total, D = h.shape
        off_d, off_m = self.off_d, self.off_m

        if self.A_md.nnz == 0:
            return np.zeros((N_total, D), dtype=np.float32)

        h_d = h[off_d:off_d + N_disease]  # (N_d, D)

        # Step 1: disease → miRNA (mean pool)
        # A_dm = A_md^T is (N_d, N_m). row_norm(A_dm^T) → no.
        # A_md^T is (N_d, N_m). Row-norm: (N_d, N_m) each disease row sums to 1...
        # No! We want: each miRNA gets mean of its disease neighbors.
        # miRNA is the TARGET, so we normalize by miRNA in-degree from diseases.
        # A_dm = A_md^T is (N_d, N_m): disease→miRNA edges.
        # row_norm(A_dm) normalizes by disease out-degree — wrong.
        # col_norm(A_dm) normalizes by miRNA in-degree — correct for mean-pooling.
        # Wait, col_norm(A_dm) = A_dm @ D_mi^{-1} where D_mi = diag(col_sum(A_dm)).
        # h_m = col_norm(A_dm)^T @ h_d = D_mi^{-1} @ A_dm^T @ h_d.
        #
        # Alternatively: A_md is (N_m, N_d): miRNA→disease edges.
        # row_norm(A_md) is (N_m, N_d): each miRNA row sums to 1.
        # This is miRNA→disease direction, not disease→miRNA.
        #
        # Simplest correct approach:
        #   Build A_md as (N_m, N_d) and use col_norm for D→M step
        #   h_m = col_norm(A_md)^T @ h_d = (A_md @ D_di^{-1})^T @ h_d
        #       = D_di^{-1} @ A_md^T @ h_d
        #   where D_di = diag(col_sum(A_md)) — disease in-degree from miRNAs.
        #   Actually this would normalize by disease degree, not miRNA degree!
        #
        # Let me just be explicit. The correct mean-pool for D→M:
        #   A_dm is (N_d, N_m): disease→miRNA.
        #   For miRNA m: h_m[m] = mean({h_d[d] : A_dm[d,m] = 1})
        #   = sum_d A_dm[d,m] * h_d[d] / sum_d A_dm[d,m]
        #   = (col_norm(A_dm))^T @ h_d
        #   col_norm(A_dm) = A_dm @ D_mi^{-1} where D_mi = diag(col_sum(A_dm))
        #   → h_m = D_mi^{-1} @ A_dm^T @ h_d
        #
        # Since A_dm = A_md^T, col_sum(A_dm) = row_sum(A_md).
        # D_mi = diag(row_sum(A_md)) = miRNA degree (how many diseases per miRNA).
        #
        # h_m = D_mi^{-1} @ A_md @ h_d  (since A_dm^T = A_md)
        #     = row_norm(A_md) @ h_d
        #
        # Beautiful! row_norm(A_md) directly gives us the D→M mean-pool.

        # Step 1: disease → miRNA (mean pool via row_norm of A_md)
        # row_norm(A_md): (N_m, N_d) — each miRNA row sums to 1
        A_md_norm = self._row_norm(self.A_md)  # (N_m, N_d)
        # h_m = A_md_norm @ h_d  →  (N_m, D)
        h_m_from_d = np.asarray(A_md_norm @ h_d, dtype=np.float32)

        # Step 2: miRNA → disease (mean pool)
        # row_norm(A_md^T): (N_d, N_m) — each disease row sums to 1
        A_dm_norm = self._row_norm(self.A_md.T)  # (N_d, N_m)
        # h_d_dmd = A_dm_norm @ h_m_from_d  →  (N_d, D)
        h_d_dmd = np.asarray(A_dm_norm @ h_m_from_d, dtype=np.float32)

        h_m_dmd = h_m_from_d

        result = np.zeros((N_total, D), dtype=np.float32)
        result[off_d:off_d + N_disease] = h_d_dmd
        result[off_m:off_m + N_m] = h_m_dmd
        return result

    def aggregate_cmd(self, h: np.ndarray) -> np.ndarray:
        """CMD meta-path: circRNA → miRNA → disease.

        Captures the molecular sponge mechanism.
          Forward:  circRNA → miRNA → disease (C→M→D)
          Backward: disease → miRNA → circRNA (D→M→C)

        Both directions are computed independently, then the results are
        placed in the respective node type positions.

        Args:
            h: (N_total, D) numpy array

        Returns:
            h_cmd: (N_total, D) — non-zero for circRNA, disease, and miRNA nodes.
        """
        N_c = self.N_c
        N_disease = self.N_d
        N_m = self.N_m
        N_total, D = h.shape
        off_d, off_m = self.off_d, self.off_m

        if self.A_cm.nnz == 0 or self.A_md.nnz == 0:
            return np.zeros((N_total, D), dtype=np.float32)

        h_c = h[:N_c]  # (N_c, D)
        h_d = h[off_d:off_d + N_disease]  # (N_d, D)

        # ── Forward: circRNA → miRNA → disease ──
        # Step 1: C→M. row_norm(A_cm^T): (N_m, N_c) — each miRNA row sums to 1
        A_mc_norm = self._row_norm(self.A_cm.T)  # (N_m, N_c)
        h_m_from_c = np.asarray(A_mc_norm @ h_c, dtype=np.float32)  # (N_m, D)

        # Step 2: M→D. row_norm(A_md^T): (N_d, N_m) — each disease row sums to 1
        # Wait: A_md is (N_m, N_d). We want M→D which is exactly A_md^T row-normed.
        # row_norm(A_md^T): (N_d, N_m) — each disease row sums to 1
        A_dm_norm = self._row_norm(self.A_md.T)  # (N_d, N_m)
        h_d_cmd = np.asarray(A_dm_norm @ h_m_from_c, dtype=np.float32)  # (N_d, D)

        # ── Backward: disease → miRNA → circRNA ──
        # Step 1': D→M. row_norm(A_md): (N_m, N_d) — each miRNA row sums to 1
        A_md_norm = self._row_norm(self.A_md)  # (N_m, N_d)
        h_m_from_d = np.asarray(A_md_norm @ h_d, dtype=np.float32)  # (N_m, D)

        # Step 2': M→C. row_norm(A_cm): (N_c, N_m) — each circRNA row sums to 1
        A_cm_norm = self._row_norm(self.A_cm)  # (N_c, N_m)
        h_c_cmd = np.asarray(A_cm_norm @ h_m_from_d, dtype=np.float32)  # (N_c, D)

        result = np.zeros((N_total, D), dtype=np.float32)
        result[:N_c] = h_c_cmd
        result[off_d:off_d + N_disease] = h_d_cmd
        # miRNA: combine both midpoints
        h_m_both = (h_m_from_c + h_m_from_d) * 0.5
        result[off_m:off_m + N_m] = np.asarray(h_m_both, dtype=np.float32)
        return result

    def compute_all(self, h: np.ndarray,
                    include_cmd: bool = True) -> Dict[str, np.ndarray]:
        """Compute all meta-path aggregations.

        Args:
            h: (N_total, D) numpy array — input node embeddings
            include_cmd: include CMD meta-path (may leak prediction target info
                         if used in encoder rather than predictor)

        Returns:
            Dict with keys 'cmc', 'dmd', 'cmd' (if include_cmd=True)
            → each value is (N_total, D) float32
        """
        result = {
            'cmc': self.aggregate_cmc(h),
            'dmd': self.aggregate_dmd(h),
        }
        if include_cmd:
            result['cmd'] = self.aggregate_cmd(h)
        return result


def compute_metapath_degrees(
    edge_index_by_type: Dict[int, torch.Tensor],
    N_circ: int,
    N_disease: int,
    N_mirna: int,
    N_lncrna: int,
) -> np.ndarray:
    """Compute per-node meta-path degree features.

    For each meta-path, computes the number of path instances starting/ending
    at each node. These are simple structural statistics — zero learnable params,
    concatenated to raw input features for TypeSpecificProjection.

    Meta-paths:
      CMC (circRNA→miRNA→circRNA): 2-hop circRNA connectivity via shared miRNAs
      DMD (disease→miRNA→disease): 2-hop disease connectivity via shared miRNAs
      CMD (circRNA→miRNA→disease): cross-type reachability via miRNA sponge

    Features per node (3-dim):
      [0]: CMC-degree — # of meta-path instances through this node
      [1]: DMD-degree
      [2]: CMD-degree

    All values are log1p-normalized.

    Returns:
        (N_total, 3) float32 array
    """
    N_total = N_circ + N_disease + N_mirna + N_lncrna
    off_d = N_circ
    off_m = N_circ + N_disease

    result = np.zeros((N_total, 3), dtype=np.float32)

    # ── Extract adjacency sub-blocks from type-2 edges ──
    ei2 = edge_index_by_type.get(2, torch.zeros((2, 0), dtype=torch.long))
    ei2_np = ei2.numpy()
    if ei2_np.shape[1] == 0:
        return result

    src = ei2_np[0]
    tgt = ei2_np[1]

    # circRNA ↔ miRNA
    mask_cm = (src < N_circ) & (tgt >= off_m) & (tgt < off_m + N_mirna)
    mask_mc = (src >= off_m) & (src < off_m + N_mirna) & (tgt < N_circ)
    cm_src = np.concatenate([src[mask_cm], tgt[mask_mc]])
    cm_tgt = np.concatenate([tgt[mask_cm] - off_m, src[mask_mc] - off_m])
    cm_data = np.ones(len(cm_src), dtype=np.float32)
    A_cm = sp.coo_matrix((cm_data, (cm_src, cm_tgt)),
                         shape=(N_circ, N_mirna)).tocsr()
    A_cm.data = np.ones_like(A_cm.data)

    # miRNA ↔ disease
    mask_md = (src >= off_m) & (src < off_m + N_mirna) & (tgt >= off_d) & (tgt < off_d + N_disease)
    mask_dm = (src >= off_d) & (src < off_d + N_disease) & (tgt >= off_m) & (tgt < off_m + N_mirna)
    md_src = np.concatenate([src[mask_md] - off_m, tgt[mask_dm] - off_m])
    md_tgt = np.concatenate([tgt[mask_md] - off_d, src[mask_dm] - off_d])
    md_data = np.ones(len(md_src), dtype=np.float32)
    A_md = sp.coo_matrix((md_data, (md_src, md_tgt)),
                         shape=(N_mirna, N_disease)).tocsr()
    A_md.data = np.ones_like(A_md.data)

    # ── CMC degree (circRNA → miRNA → circRNA) ──
    if A_cm.nnz > 0:
        cmc_mat = A_cm @ A_cm.T  # (N_c, N_c) — symmetric, counts shared miRNAs
        cmc_circ = np.array(cmc_mat.sum(axis=1)).flatten()  # (N_c,)
        result[:N_circ, 0] = cmc_circ

        # miRNA CMC-degree = deg_c * (deg_c - 1) / 2
        mirna_deg_c = np.array(A_cm.sum(axis=0)).flatten()  # (N_m,)
        cmc_mirna = mirna_deg_c * (mirna_deg_c - 1) / 2.0
        result[off_m:off_m + N_mirna, 0] = cmc_mirna

    # ── DMD degree (disease → miRNA → disease) ──
    if A_md.nnz > 0:
        dmd_mat = A_md.T @ A_md  # (N_d, N_d) — shared miRNAs between diseases
        dmd_dis = np.array(dmd_mat.sum(axis=1)).flatten()  # (N_d,)
        result[off_d:off_d + N_disease, 1] = dmd_dis

        # miRNA DMD-degree = deg_d * (deg_d - 1) / 2
        mirna_deg_d = np.array(A_md.sum(axis=1)).flatten()  # (N_m,)
        dmd_mirna = mirna_deg_d * (mirna_deg_d - 1) / 2.0
        result[off_m:off_m + N_mirna, 1] += dmd_mirna

    # ── CMD degree (circRNA → miRNA → disease) ──
    if A_cm.nnz > 0 and A_md.nnz > 0:
        cmd_cd = A_cm @ A_md  # (N_c, N_d)
        cmd_circ = np.array(cmd_cd.sum(axis=1)).flatten()  # (N_c,)
        result[:N_circ, 2] = cmd_circ

        cmd_dis = np.array(cmd_cd.sum(axis=0)).flatten()  # (N_d,)
        result[off_d:off_d + N_disease, 2] = cmd_dis

        # miRNA CMD-degree = deg_c * deg_d
        cmd_mirna = mirna_deg_c * mirna_deg_d
        result[off_m:off_m + N_mirna, 2] = cmd_mirna

    # ── Log1p normalize ──
    result = np.log1p(result)

    return result.astype(np.float32)


class MetaPathProjector(nn.Module):
    """Learnable projection for meta-path aggregated features.

    Projects each meta-path's (N, D) features through a small MLP, optionally
    combining multiple meta-paths into a single tensor.
    """

    def __init__(self, hidden_dim: int, num_metapaths: int = 3,
                 dropout: float = 0.3, share_proj: bool = True):
        super().__init__()
        self.num_metapaths = num_metapaths
        self.share_proj = share_proj

        if share_proj:
            # Shared projection: all meta-paths use the same weights
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            # Per-meta-path independent projections
            self.proj = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ) for _ in range(num_metapaths)
            ])

    def forward(self, mp_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Project and combine meta-path features.

        Args:
            mp_features: {'cmc': (N,D), 'dmd': (N,D), 'cmd': (N,D)} tensors

        Returns:
            (N, D) combined meta-path features (mean of all paths)
        """
        paths = ['cmc', 'dmd', 'cmd']
        projected = []
        for i, key in enumerate(paths):
            if key not in mp_features:
                continue
            feat = mp_features[key]
            if self.share_proj:
                projected.append(self.proj(feat))
            else:
                projected.append(self.proj[i](feat))

        if not projected:
            return None
        return torch.stack(projected, dim=0).mean(dim=0)  # (N, D)


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Test with random data
    N_c, N_d, N_m, N_l = 100, 50, 80, 30
    N_total = N_c + N_d + N_m + N_l

    # Create fake type-2 edges: circ↔mir, mir↔dis
    rng = np.random.RandomState(42)
    cm_edges = rng.randint(0, N_c, 200), rng.randint(0, N_m, 200)
    md_edges = rng.randint(0, N_m, 200), rng.randint(0, N_d, 200)

    off_d, off_m, off_l = N_c, N_c + N_d, N_c + N_d + N_m

    ei2_src = np.concatenate([
        cm_edges[0],                          # circ → mir
        cm_edges[1] + off_m,                  # mir → circ (reverse)
        md_edges[0] + off_m,                  # mir → dis
        md_edges[1] + off_d,                  # dis → mir (reverse)
    ])
    ei2_tgt = np.concatenate([
        cm_edges[1] + off_m,
        cm_edges[0],
        md_edges[1] + off_d,
        md_edges[0] + off_m,
    ])

    edge_index_by_type = {
        2: torch.tensor(np.stack([ei2_src, ei2_tgt]), dtype=torch.long)
    }

    agg = MetaPathAggregator(edge_index_by_type, N_c, N_d, N_m, N_l)
    print(f"A_cm nnz: {agg.A_cm.nnz}, A_md nnz: {agg.A_md.nnz}")

    # Test with random embeddings
    h = rng.randn(N_total, 96).astype(np.float32)
    mp = agg.compute_all(h)

    for key, val in mp.items():
        nz = (np.abs(val).sum(axis=1) > 1e-6).sum()
        print(f"  {key}: shape={val.shape}, non-zero rows={nz}/{N_total}, "
              f"mean={val.mean():.6f}, std={val.std():.6f}")

    print("MetaPathAggregator test passed!")
