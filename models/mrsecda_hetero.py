"""
MRSECDA v3.3: Heterogeneous Multi-Scale CDA Predictor.

Naming note: the pairwise structural feature encoder is "PSE"
(Pairwise Structural Encoder), formerly "SEGCA" (renamed 2026-08-30 to avoid
collision with the medical acronym "Subependymal Giant Cell Astrocytoma").

Extension of MRSECDA to truly heterogeneous graphs with 4+ node types:
  circRNA | disease | miRNA | lncRNA

Architecture (v3.3):
  Stage 0 — Meta-Path HGT Edge Types: CMC (circRNA↔circRNA via shared miRNAs)
            and DMD (disease↔disease via shared miRNAs) as independent HGT edge
            types (3→5), each with dedicated QKV attention projections.
  Stage 1 — Type-Specific Projection (one per node type) + LapPE/RWSE
  Stage 2 — CRF Multi-Resolution:
            Token₀: Self (projected features)
            Token₁: 1-hop HGT (5 edge types) + DropEdge(0.05)
            Token₃: Cross-Type Context Pool (global cross-type attention)
  Stage 3 — CRF Adaptive Fusion (learned per-node per-token softmax weights)
  Predictor — Dual-Pathway + Multi-View PSE structural features

Key differences from v3.2:
  - ADDED: CMC/DMD meta-path edges as HGT edge types 3 and 4 (+2 QKV projections)
    → 2-hop topology captured via dedicated attention, not as scalar features
  - CHANGED: num_edge_types 3→5 (within-type / assoc / aux / CMC / DMD)
  - PRESERVED: CRF multi-resolution (Token₀ + Token₁ + Token₃) from v3.2
  - DISABLED: MPNE concat (meta-path degree features → input) — replaced by edges
  - PSE multi-view structural features (12-dim) — unchanged
  - Focal Loss (γ=2.0) — unchanged

Results (seed=42, 5-fold CV):
  circR2Cancer: 0.9933 ± 0.0018 (+0.0033 vs v3.2)
  circR2Disease: 0.9855 ± 0.0118 (matching v3.2)
  First encoder-level innovation to surpass v3.2 after 10+ failed attempts.

Usage:
  python scripts/run_mrsecda.py --datasets circR2Cancer circR2Disease --use_metapath_edges
"""



from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Tuple, Optional, Dict


# =============================================================================
# Utility: Scatter Softmax for HGT per-target-node attention
# =============================================================================

def _scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int,
             reduce: str = 'sum') -> torch.Tensor:
    """Scatter-reduce using PyTorch native scatter_reduce_.

    Args:
        src: (E, ...) values to scatter
        index: (E,) target indices in [0, dim_size)
        dim_size: size of output dimension 0
        reduce: 'sum' or 'max'
    Returns:
        (dim_size, ...) reduced tensor
    """
    reduce_op = 'amax' if reduce == 'max' else 'sum'
    # Expand index to match src shape for scatter_reduce_
    idx_expanded = index
    for _ in range(src.dim() - 1):
        idx_expanded = idx_expanded.unsqueeze(-1)
    idx_expanded = idx_expanded.expand_as(src)

    out = src.new_zeros(dim_size, *src.shape[1:])
    out.scatter_reduce_(0, idx_expanded, src, reduce=reduce_op, include_self=False)
    return out


def scatter_softmax(scores: torch.Tensor, index: torch.Tensor,
                    num_nodes: int) -> torch.Tensor:
    """Per-target-node softmax normalization.

    For each target node j, softmax over all edges where tgt == j.

    Args:
        scores: (E, H) raw attention scores per edge
        index: (E,) target node index for each edge
        num_nodes: N — total number of target nodes
    Returns:
        (E, H) normalized scores (sum to 1 per target node)
    """
    scores_max = _scatter(scores, index, dim_size=num_nodes, reduce='max')
    scores = scores - scores_max[index]
    scores_exp = torch.exp(scores)
    scores_sum = _scatter(scores_exp, index, dim_size=num_nodes, reduce='sum')
    return scores_exp / (scores_sum[index] + 1e-8)


# =============================================================================
# Stage 1: Type-Specific Tokenization
# =============================================================================

class TypeSpecificProjection(nn.Module):
    """Two-stage MLP for projecting heterogeneous input features to unified space."""

    def __init__(self, input_dim: int, hidden_dim: int, pe_dim: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        total_dim = input_dim + pe_dim
        mid_dim = hidden_dim * 2

        self.reduce = nn.Sequential(
            nn.Linear(total_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.refine = nn.Sequential(
            nn.Linear(mid_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, pe], dim=-1)
        h = self.reduce(h)
        h = self.refine(h)
        return h


# =============================================================================
# Stage 2: GCN-based Multi-Resolution Token Construction
# =============================================================================

class GCNEncoder(nn.Module):
    """Edge-type-specific GCN layers for heterogeneous message passing.

    Uses separate GCNConv per edge type, aggregates with learned attention weights.
    """

    def __init__(self, hidden_dim: int, num_layers: int = 2, num_edge_types: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.num_edge_types = num_edge_types
        # Per-edge-type GCNConv per layer
        self.convs = nn.ModuleList([
            nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_edge_types)])
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        # Learnable edge-type attention weights per layer (softmaxed)
        self.edge_type_weights = nn.ParameterList([
            nn.Parameter(torch.ones(num_edge_types) / num_edge_types)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor,
                edge_index_by_type: dict = None,
                edge_index: torch.Tensor = None,
                edge_weight: Optional[torch.Tensor] = None,
                num_layers: int = None) -> torch.Tensor:
        n_layers = num_layers if num_layers is not None else len(self.convs)
        use_typed = edge_index_by_type is not None

        for li in range(n_layers):
            if use_typed:
                alpha = F.softmax(self.edge_type_weights[li], dim=0)
                h_agg = torch.zeros_like(h)
                n_active = 0
                for t in range(self.num_edge_types):
                    if t in edge_index_by_type and edge_index_by_type[t].shape[1] > 0:
                        h_t = self.convs[li][t](h, edge_index_by_type[t])
                        h_agg = h_agg + alpha[t] * h_t
                        n_active += 1
                h = h_agg if n_active > 0 else h
            else:
                # Uniform fallback: use first conv on combined edge_index
                h = self.convs[li][0](h, edge_index, edge_weight)

            h = self.norms[li](F.relu(h))
            h = self.dropout(h)
        return h


# =============================================================================
# Stage 2b: HGT — Heterogeneous Graph Transformer (replaces GCNEncoder)
# =============================================================================

class HGTLayer(nn.Module):
    """Single HGT (Heterogeneous Graph Transformer) layer.

    Replaces per-edge-type GCNConv with per-edge-type multi-head attention.
    Key differences from GCNEncoder:
      - GCN: h_i = Σ α_t * (Σ_{j∈N_t(i)} W_t · h_j / √(d_i·d_j))  [mean pooling]
      - HGT: h_i = Σ α_t * (Σ_{j∈N_t(i)} attn_t(i,j) · V_t(h_j))   [learned attention]

    The attention mechanism allows the model to selectively focus on
    important neighbors within each edge type.

    v3.0: DropEdge regularization — randomly drops edges during training
          to prevent overfitting on small biological graphs.
    v3.1: Learnable per-edge-type per-head attention temperature —
          allows different edge types to have different attention sharpness.
    """

    def __init__(self, hidden_dim: int, num_edge_types: int = 3,
                 num_heads: int = 4, dropout: float = 0.3,
                 edge_dropout: float = 0.0, use_attn_temp: bool = False,
                 use_sigmoid_gate: bool = False):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.edge_dropout_rate = edge_dropout
        self.use_attn_temp = use_attn_temp
        self.use_sigmoid_gate = use_sigmoid_gate

        # Per-edge-type per-head learnable attention temperature (v3.1)
        # temp > 1 → softer attention, temp < 1 → sharper attention
        if use_attn_temp:
            self.attn_temp = nn.Parameter(torch.ones(num_edge_types, num_heads))
        else:
            self.attn_temp = None

        # Per-edge-type Q, K, V projections
        self.W_q = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_edge_types)
        ])
        self.W_k = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_edge_types)
        ])
        self.W_v = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_edge_types)
        ])
        # Per-edge-type output projection
        self.W_o = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_edge_types)
        ])

        # Edge-type fusion weights (softmaxed — same pattern as GCNEncoder)
        self.edge_type_weights = nn.Parameter(
            torch.ones(num_edge_types) / num_edge_types)

        # FFN block (transformer-style, 2x expansion — lighter than standard 4x
        # to avoid overfitting on small biological datasets)
        ffn_hidden = hidden_dim * 2
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
        )

        # Pre-norm layers (modern transformer convention)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def set_edge_dropout(self, rate: float):
        """Update edge dropout rate (for annealing schedules)."""
        self.edge_dropout_rate = rate

    def forward(self, h: torch.Tensor,
                edge_index_by_type: dict) -> torch.Tensor:
        """
        Args:
            h: (N, D) node embeddings
            edge_index_by_type: Dict[int, LongTensor(2, E_t)] per-type edges
        Returns:
            h_out: (N, D)
        """
        N, D = h.shape[0], self.hidden_dim
        H, d = self.num_heads, self.head_dim
        # v3.2: sigmoid gating allows independent contribution per edge type
        # (softmax forces competition — complementary types shouldn't compete)
        if self.use_sigmoid_gate:
            alpha = torch.sigmoid(self.edge_type_weights)  # per-type ∈ (0,1)
        else:
            alpha = F.softmax(self.edge_type_weights, dim=0)

        h_agg = torch.zeros(N, D, device=h.device, dtype=h.dtype)
        deg_sum = torch.zeros(N, device=h.device, dtype=h.dtype)
        n_active = 0

        for t in range(self.num_edge_types):
            if t not in edge_index_by_type or edge_index_by_type[t].shape[1] == 0:
                continue

            ei = edge_index_by_type[t].to(h.device)      # (2, E_t)

            # DropEdge: randomly drop edges during training (v3.0)
            if self.training and self.edge_dropout_rate > 0:
                n_edges = ei.shape[1]
                keep_mask = torch.rand(n_edges, device=ei.device) > self.edge_dropout_rate
                ei = ei[:, keep_mask]
                if ei.shape[1] == 0:
                    continue  # all edges dropped for this type
            src_idx = ei[0]                  # (E_t,)
            tgt_idx = ei[1]                  # (E_t,)

            # Compute in-degree per target node for GCN-style normalization
            deg_t = torch.zeros(N, device=h.device, dtype=h.dtype)
            deg_t.scatter_add_(0, tgt_idx, torch.ones(src_idx.shape[0], device=h.device))
            deg_sum = deg_sum + deg_t

            # Multi-head projections: (N, D) → (N, H, d)
            q = self.W_q[t](h).view(N, H, d)   # (N, H, d)
            k = self.W_k[t](h).view(N, H, d)
            v = self.W_v[t](h).view(N, H, d)

            # Gather edge endpoints
            q_tgt = q[tgt_idx]                  # (E_t, H, d)
            k_src = k[src_idx]
            v_src = v[src_idx]

            # Dot-product attention per head
            attn_raw = (q_tgt * k_src).sum(dim=-1)       # (E_t, H)
            # v3.1: per-edge-type per-head learnable temperature
            temp = self.attn_temp[t:t+1, :] if self.use_attn_temp else 1.0
            attn_raw = attn_raw / (self.scale * temp)

            # Per-target-node softmax (key HGT innovation over GCN)
            attn = scatter_softmax(attn_raw, tgt_idx, N)          # (E_t, H)
            attn = self.dropout(attn)

            # Weighted message: (E_t, H, d)
            msg = attn.unsqueeze(-1) * v_src

            # Scatter-aggregate to target nodes: (E_t, H*d) → (N, H*d)
            msg_flat = msg.reshape(-1, H * d)
            agg_flat = _scatter(msg_flat, tgt_idx, dim_size=N, reduce='sum')

            # Output projection
            h_t = self.W_o[t](agg_flat)       # (N, D)
            h_agg = h_agg + alpha[t] * h_t
            n_active += 1

        # GCN-style degree normalization (prevents high-degree nodes dominating)
        if n_active > 0:
            # v3.2: sigmoid gate normalization (softmax has Σα=1 built in)
            if self.use_sigmoid_gate:
                alpha_active = torch.stack([alpha[t] for t in range(self.num_edge_types)
                                           if t in edge_index_by_type
                                           and edge_index_by_type[t].shape[1] > 0])
                alpha_sum = alpha_active.sum()
                if alpha_sum > 0:
                    h_agg = h_agg / alpha_sum
            deg_safe = deg_sum.clamp(min=1).unsqueeze(-1)
            h_agg = h_agg / deg_safe.sqrt()

        # Residual + pre-norm + ReLU (matching GCNEncoder activation pattern)
        if n_active > 0:
            h = h + self.dropout(F.relu(h_agg))
        h = self.norm_attn(h)

        # Light FFN with pre-norm residual
        h = h + self.dropout(self.ffn(h))
        h = self.norm_ffn(h)

        return h


class HGTEncoder(nn.Module):
    """Stacked HGT layers for heterogeneous message passing.

    Drop-in replacement for GCNEncoder with identical interface.
    Uses edge-type-specific multi-head attention instead of mean-pooling GCNConv.
    """

    def __init__(self, hidden_dim: int, num_layers: int = 2,
                 num_edge_types: int = 3, num_heads: int = 4,
                 dropout: float = 0.3, edge_dropout: float = 0.0,
                 use_attn_temp: bool = False, use_sigmoid_gate: bool = False):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.layers = nn.ModuleList([
            HGTLayer(hidden_dim, num_edge_types, num_heads, dropout,
                     edge_dropout=edge_dropout, use_attn_temp=use_attn_temp,
                     use_sigmoid_gate=use_sigmoid_gate)
            for _ in range(num_layers)
        ])

    def set_edge_dropout(self, rate: float):
        """Update edge dropout rate across all layers (v3.2: annealing schedule)."""
        for layer in self.layers:
            layer.set_edge_dropout(rate)

    def forward(self, h: torch.Tensor,
                edge_index_by_type: dict = None,
                edge_index: torch.Tensor = None,
                edge_weight: Optional[torch.Tensor] = None,
                num_layers: int = None) -> torch.Tensor:
        """
        Args:
            h: (N, D)
            edge_index_by_type: Dict[int, LongTensor] per-type edges
            edge_index: ignored (backward compat — HGT requires typed edges)
            edge_weight: ignored
            num_layers: override number of layers (1 for Token1, 2 for Token2)
        Returns:
            h_out: (N, D)
        """
        n_layers = num_layers if num_layers is not None else len(self.layers)

        for li in range(n_layers):
            h = self.layers[li](h, edge_index_by_type)

        return h


# =============================================================================
# Stage 2c: Cross-Type Context Pool
# =============================================================================

class CrossTypeContextPool(nn.Module):
    """
    Token₃: Global cross-type context for heterogeneous graph.

    Each node type attends to ALL other node types, enabling
    circRNA↔miRNA↔lncRNA↔disease cross-modal information flow.
    """

    def __init__(self, hidden_dim: int, num_node_types: int = 4, dropout: float = 0.3):
        super().__init__()
        self.num_types = num_node_types
        self.hidden_dim = hidden_dim

        # One QKV set per source→target type pair
        self.W_q = nn.ModuleList([
            nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False)
                           for _ in range(num_node_types)])
            for _ in range(num_node_types)
        ])
        self.W_k = nn.ModuleList([
            nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False)
                           for _ in range(num_node_types)])
            for _ in range(num_node_types)
        ])
        self.W_v = nn.ModuleList([
            nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False)
                           for _ in range(num_node_types)])
            for _ in range(num_node_types)
        ])

        self.W_o = nn.Linear(hidden_dim * num_node_types, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_all: torch.Tensor,
                type_boundaries: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        Args:
            h_all: (N_total, D)
            type_boundaries: (N_circ, N_disease, N_mirna, N_lncrna)
        Returns:
            h_ctx: (N_total, D) cross-type enriched representations
        """
        N_c, N_d, N_m, N_l = type_boundaries
        D = self.hidden_dim
        n_types = self.num_types

        # Split by type
        h_types = [
            h_all[:N_c],
            h_all[N_c:N_c + N_d],
            h_all[N_c + N_d:N_c + N_d + N_m],
            h_all[N_c + N_d + N_m:],
        ]

        # For each source type, compute cross-attention to all target types
        ctx_parts = [[] for _ in range(n_types)]

        for src_t in range(n_types):
            h_src = h_types[src_t]   # (N_src, D)
            ctx_sources = []
            for tgt_t in range(n_types):
                h_tgt = h_types[tgt_t]  # (N_tgt, D)
                q = self.W_q[src_t][tgt_t](h_src)      # (N_src, D)
                k = self.W_k[src_t][tgt_t](h_tgt)      # (N_tgt, D)
                v = self.W_v[src_t][tgt_t](h_tgt)      # (N_tgt, D)
                attn = F.softmax(q @ k.T / math.sqrt(D), dim=-1)
                ctx_src_tgt = attn @ v                  # (N_src, D)
                ctx_sources.append(ctx_src_tgt)
            # Concatenate all target-type contexts
            ctx_parts[src_t] = torch.cat(ctx_sources, dim=-1)  # (N_src, D * n_types)

        # Project and residual
        ctx_all = torch.cat(ctx_parts, dim=0)           # (N_total, D * n_types)
        ctx_all = self.norm(h_all + self.dropout(self.W_o(ctx_all)))
        return ctx_all


# =============================================================================
# Stage 2d: Hypergraph Convolution (v2.4) — Meta-Path HGNN + Hyperedge Gating
# =============================================================================

class HypergraphConv(nn.Module):
    """Hypergraph Convolution with learned per-hyperedge gating.

    Constructs a meta-path-based hypergraph from edge_index_by_type,
    then performs two-stage message passing:
      1. Node → Hyperedge: mean_pool node features within each hyperedge
      2. Hyperedge → Node: scatter hyperedge features back to constituent nodes,
         with learned scalar gate per hyperedge.

    Architecture:
      E_j     = mean_pool({h_i : i in hyperedge_members[j]})     # (K, D)
      E'_j    = sigmoid(gate_j) * E_j                             # (K, D)
      h_hyper_i = mean_pool({E'_j : j in node_hyperedges[i]})    # (N, D)
      h_out   = h + Dropout(LayerNorm(Linear(h_hyper) + h_hyper))# (N, D)

    v2.5: Outputs a standalone Token_Hyper (meso-scale).
    CRF handles fusion with Token₀, Token₁, Token₃ adaptively.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # Output projection after hypergraph convolution
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # Lazy-initialized hyperedge gating (created on first forward)
        self.hyperedge_gate: Optional[nn.Embedding] = None
        self._hypergraph_cached: Optional[Tuple[List[List[int]],
                                                 List[List[int]]]] = None

    # -----------------------------------------------------------------
    # Hypergraph Builder
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_edges(ei: torch.Tensor, src_lo: int, src_hi: int,
                       tgt_lo: int, tgt_hi: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter edge_index for edges from [src_lo, src_hi) to [tgt_lo, tgt_hi).

        Returns (src_local, tgt_local) — local indices within each range.
        """
        if ei.shape[1] == 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        src = ei[0]
        tgt = ei[1]
        mask = (src >= src_lo) & (src < src_hi) & (tgt >= tgt_lo) & (tgt < tgt_hi)
        return src[mask] - src_lo, tgt[mask] - tgt_lo

    def _build_hypergraph(self, edge_index_by_type: dict,
                          node_counts: dict) -> Tuple[List[List[int]], List[List[int]]]:
        """Build meta-path hypergraph incidence structure.

        Three hyperedge types:
          - Disease-centered:   {d} ∪ {c : c-d edge in type 1}
          - miRNA-centered:     {m} ∪ {c : c-m edge} ∪ {d : m-d edge}
          - circRNA-centered:   {c} ∪ {d : c-d edge} ∪ {m : c-m edge}

        Returns:
            hyperedge_members:  list of lists, hyperedge_members[j] = [node_idx, ...]
            node_hyperedges:    list of lists, node_hyperedges[i] = [hyperedge_idx, ...]
        """
        N_c = node_counts['N_circ']
        N_d = node_counts['N_disease']
        N_m = node_counts['N_mirna']
        N_total = N_c + N_d + N_m + node_counts.get('N_lncrna', 0)
        off_d = N_c
        off_m = N_c + N_d

        # ── Extract adjacency from edge_index_by_type ──
        ei1 = edge_index_by_type.get(1, torch.zeros((2, 0), dtype=torch.long))
        ei2 = edge_index_by_type.get(2, torch.zeros((2, 0), dtype=torch.long))

        # circRNA-disease (type 1, undirected — take circ→dis direction)
        cd_src, cd_tgt = self._extract_edges(ei1, 0, N_c, off_d, off_d + N_d)

        # miRNA-disease (type 2)
        md_src, md_tgt = self._extract_edges(ei2, off_m, off_m + N_m, off_d, off_d + N_d)

        # circRNA-miRNA (type 2)
        cm_src, cm_tgt = self._extract_edges(ei2, 0, N_c, off_m, off_m + N_m)

        # Build adjacency dicts: disease→set(circRNA), miRNA→set(circRNA), miRNA→set(disease)
        disease_circs = {d: [] for d in range(N_d)}
        for i in range(cd_src.shape[0]):
            disease_circs[int(cd_tgt[i])].append(int(cd_src[i]))

        mirna_circs = {m: [] for m in range(N_m)}
        for i in range(cm_src.shape[0]):
            mirna_circs[int(cm_tgt[i])].append(int(cm_src[i]))

        mirna_diseases = {m: [] for m in range(N_m)}
        for i in range(md_src.shape[0]):
            mirna_diseases[int(md_src[i])].append(int(md_tgt[i]))

        hyperedge_members = []

        # ── Type 1: Disease-centered hyperedges ──
        for d in range(N_d):
            circs = disease_circs[d]
            if circs:
                hyperedge_members.append([off_d + d] + circs)

        # ── Type 2: miRNA-centered (circRNA + disease) hyperedges ──
        for m in range(N_m):
            circs = mirna_circs.get(m, [])
            diseases = mirna_diseases.get(m, [])
            members = ([off_m + m] + circs +
                       [off_d + d_idx for d_idx in diseases])
            if len(members) >= 2:  # skip trivial singletons
                hyperedge_members.append(members)

        # ── Type 3: circRNA-centered (disease + miRNA) hyperedges ──
        # Build reverse index: circ→diseases
        circ_diseases = {c: [] for c in range(N_c)}
        for i in range(cd_src.shape[0]):
            circ_diseases[int(cd_src[i])].append(int(cd_tgt[i]))
        # Build reverse index: circ→miRNAs
        circ_mirnas = {c: [] for c in range(N_c)}
        for i in range(cm_src.shape[0]):
            circ_mirnas[int(cm_src[i])].append(int(cm_tgt[i]))

        for c in range(N_c):
            diseases = circ_diseases.get(c, [])
            mirnas = circ_mirnas.get(c, [])
            members = ([c] +
                       [off_d + d_idx for d_idx in diseases] +
                       [off_m + m_idx for m_idx in mirnas])
            if len(members) >= 2:
                hyperedge_members.append(members)

        # ── Build reverse index: node → hyperedges ──
        node_hyperedges = [[] for _ in range(N_total)]
        for j, members in enumerate(hyperedge_members):
            for node_idx in set(members):
                if node_idx < N_total:
                    node_hyperedges[node_idx].append(j)

        return hyperedge_members, node_hyperedges

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(self, h: torch.Tensor, edge_index_by_type: dict,
                node_counts: dict) -> torch.Tensor:
        """Apply hypergraph convolution with per-hyperedge gating.

        Args:
            h: (N_total, D) node embeddings (Token1 output)
            edge_index_by_type: Dict[int, LongTensor] per-type edges
            node_counts: dict with N_circ, N_disease, N_mirna, N_lncrna
        Returns:
            h_hyper: (N_total, D) hypergraph-convolved embeddings
        """
        N = h.shape[0]
        D = self.hidden_dim
        device = h.device

        # ── Build/load hypergraph structure (once, cached) ──
        if self._hypergraph_cached is None:
            hyperedge_members, node_hyperedges = self._build_hypergraph(
                edge_index_by_type, node_counts)
            self._hypergraph_cached = (hyperedge_members, node_hyperedges)
        else:
            hyperedge_members, node_hyperedges = self._hypergraph_cached

        K = len(hyperedge_members)
        if K == 0:
            return h  # No hyperedges → pass-through

        # ── Lazy init: per-hyperedge gating weights ──
        if self.hyperedge_gate is None or self.hyperedge_gate.weight.shape[0] != K:
            self.hyperedge_gate = nn.Embedding(K, 1).to(device)
            nn.init.zeros_(self.hyperedge_gate.weight)  # init to sigmoid(0) = 0.5
        gate_vals = torch.sigmoid(self.hyperedge_gate.weight).squeeze(-1)  # (K,)

        # ── Step 1: Node → Hyperedge (mean pool) ──
        edge_ids = []   # hyperedge index for each node occurrence
        node_ids = []   # node index
        for j, members in enumerate(hyperedge_members):
            for ni in members:
                if ni < N:
                    edge_ids.append(j)
                    node_ids.append(ni)

        edge_ids_t = torch.tensor(edge_ids, dtype=torch.long, device=device)
        node_ids_t = torch.tensor(node_ids, dtype=torch.long, device=device)

        h_nodes_for_edges = h[node_ids_t]                        # (total_incidence, D)
        h_hyperedge = torch.zeros(K, D, device=device, dtype=h.dtype)
        h_hyperedge.scatter_reduce_(0,
            edge_ids_t.unsqueeze(-1).expand(-1, D),
            h_nodes_for_edges, reduce='mean', include_self=False)

        # ── Step 2: Per-hyperedge gating ──
        h_hyperedge = h_hyperedge * gate_vals.unsqueeze(-1)      # (K, D)

        # ── Step 3: Hyperedge → Node (mean pool) ──
        node_edge_ids = []
        he_ids_for_node = []
        for i, edges in enumerate(node_hyperedges):
            for ej in edges:
                node_edge_ids.append(i)
                he_ids_for_node.append(ej)

        node_edge_ids_t = torch.tensor(node_edge_ids, dtype=torch.long, device=device)
        he_ids_for_node_t = torch.tensor(he_ids_for_node, dtype=torch.long, device=device)

        h_from_he = h_hyperedge[he_ids_for_node_t]                # (total_incidence, D)
        h_hyper = torch.zeros(N, D, device=device, dtype=h.dtype)
        h_hyper.scatter_reduce_(0,
            node_edge_ids_t.unsqueeze(-1).expand(-1, D),
            h_from_he, reduce='mean', include_self=False)

        # ── Step 4: Skip connection + projection ──
        h_hyper = h + self.dropout(h_hyper)
        h_hyper = h_hyper + self.dropout(self.proj(F.relu(h_hyper)))
        h_hyper = self.norm(h_hyper)

        return h_hyper


# =============================================================================
# Graph Contrastive Learning Loss (v3.1)
# =============================================================================

def multi_view_contrastive_loss(token0: torch.Tensor, token1: torch.Tensor,
                                 temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE loss between two views (resolutions) of the same nodes.

    Maximizes agreement between Token₀ (Self) and Token₁ (1-hop HGT)
    for the same node while minimizing agreement with other nodes.
    Zero extra parameters — pure self-supervised regularization.

    Args:
        token0: (N, D) Self-view (projected features)
        token1: (N, D) 1-hop HGT view (graph-convolved features)
        temperature: softmax temperature (lower = sharper, default 0.07)

    Returns:
        scalar contrastive loss
    """
    N = token0.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=token0.device)

    # L2-normalize embeddings
    z0 = F.normalize(token0, dim=-1)
    z1 = F.normalize(token1, dim=-1)

    # Cosine similarity matrix: (N, N)
    sim = (z0 @ z1.T) / temperature

    # Positive pairs on diagonal — each node matches itself across views
    labels = torch.arange(N, device=token0.device)

    # Symmetric InfoNCE
    loss_01 = F.cross_entropy(sim, labels)
    loss_10 = F.cross_entropy(sim.T, labels)

    return (loss_01 + loss_10) / 2.0


# =============================================================================
# Stage 3: Cross-Resolution Fusion (CRF) — ★ Core Innovation ★
# =============================================================================

class CrossResolutionFusion(nn.Module):
    """Resolution PE + adaptive weighted fusion (v2.2: attention layers removed)."""

    def __init__(self, hidden_dim: int, num_resolutions: int = 4,
                 num_groups: int = 8, drop_token_prob: float = 0.0):
        super().__init__()
        self.res_pe = nn.Parameter(torch.randn(1, num_resolutions, hidden_dim) * 0.02)
        self.W_fuse = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: list) -> torch.Tensor:
        S = torch.stack(tokens, dim=1)  # (N, B, D)
        S = S + self.res_pe
        alpha = self.W_fuse(S).squeeze(-1)      # (N, B)
        alpha = F.softmax(alpha, dim=-1)
        h_fused = (alpha.unsqueeze(-1) * S).sum(dim=1)
        self.last_alpha = alpha
        return h_fused


# =============================================================================
# Dual-Pathway Predictor (unchanged from v1)
# =============================================================================

class DualPathwayPredictor(nn.Module):
    """
    Path A: Direct bilinear interaction
    Path B: Cross-attention disease context
    Adaptive gate: σ(W·[h_c||h_d]) · s_A + (1−σ) · s_B

    Supports optional pairwise structural features (Jaccard, AA, RA, inv_dist)
    injected via concat into both pathways.
    """

    def __init__(self, hidden_dim: int, decoder_hidden: Tuple[int, ...] = (96, 48),
                 dropout: float = 0.3, struct_feat_dim: int = 0,
                 mean_gate: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.struct_feat_dim = struct_feat_dim
        self.mean_gate = mean_gate

        # Small projection for structural features: raw_dim → 16D
        if struct_feat_dim > 0:
            self.struct_proj = nn.Sequential(
                nn.Linear(struct_feat_dim, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.struct_proj = None
        struct_emb_dim = 16 if struct_feat_dim > 0 else 0

        # Path A: [h_c || h_d || h_c⊙h_d || |h_c−h_d|] + optional struct_emb
        in_dim = hidden_dim * 4 + struct_emb_dim
        path_a = []
        prev = in_dim
        for h in decoder_hidden:
            path_a.extend([nn.Linear(prev, h), nn.BatchNorm1d(h),
                           nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        path_a.append(nn.Linear(prev, 1))
        self.mlp_a = nn.Sequential(*path_a)

        # Path B: Cross-attention over all diseases
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        in_dim_b = hidden_dim * 4 + struct_emb_dim
        path_b = []
        prev = in_dim_b
        for h in decoder_hidden:
            path_b.extend([nn.Linear(prev, h), nn.BatchNorm1d(h),
                           nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        path_b.append(nn.Linear(prev, 1))
        self.mlp_b = nn.Sequential(*path_b)

        # Adaptive gate
        self.W_gate = nn.Linear(hidden_dim * 2, 1)

    def forward(self, h_circ: torch.Tensor, h_disease: torch.Tensor,
                circ_indices: torch.Tensor, disease_indices: torch.Tensor,
                return_attention: bool = False,
                struct_feat: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        D = self.hidden_dim
        hc = h_circ[circ_indices]
        hd = h_disease[disease_indices]

        # Structural feature embedding
        if self.struct_proj is not None and struct_feat is not None:
            s_emb = self.struct_proj(struct_feat)
        else:
            s_emb = None

        # Path A: concat structural embedding if available
        parts_a = [hc, hd, hc * hd, (hc - hd).abs()]
        if s_emb is not None:
            parts_a.append(s_emb)
        f_a = torch.cat(parts_a, dim=-1)
        s_a = self.mlp_a(f_a).squeeze(-1)

        # Path B
        q = self.W_q(hc)
        k = self.W_k(h_disease)
        v = self.W_v(h_disease)
        attn_weights = F.softmax(q @ k.T / math.sqrt(D), dim=-1)
        context = attn_weights @ v
        parts_b = [hc, hd, context, hc * context]
        if s_emb is not None:
            parts_b.append(s_emb)
        f_b = torch.cat(parts_b, dim=-1)
        s_b = self.mlp_b(f_b).squeeze(-1)

        # Adaptive gate (or simple mean for ablation)
        if self.mean_gate:
            logits = 0.5 * s_a + 0.5 * s_b
        else:
            gate = torch.sigmoid(self.W_gate(torch.cat([hc, hd], dim=-1)))
            logits = gate.squeeze(-1) * s_a + (1 - gate.squeeze(-1)) * s_b

        if return_attention:
            return logits, attn_weights
        return logits, None


# =============================================================================
# Full MRSECDA v2: Heterogeneous Multi-Type Model
# =============================================================================

class MRSECDAHetero(nn.Module):
    """
    MRSECDA v3.0 for heterogeneous graphs (circRNA, disease, miRNA, lncRNA).

    ┌──────────────────────────────────────────────────────┐
    │ Stage 1: Type-Specific Projection (4 independent)    │
    │   circRNA: MLP(input_dim_circ → hidden)              │
    │   disease: MLP(input_dim_dis  → hidden)              │
    │   miRNA:   MLP(input_dim_mir  → hidden)              │
    │   lncRNA:  MLP(input_dim_lnc  → hidden)              │
    │                                                      │
    │ Stage 2: Multi-Resolution Token Construction         │
    │   Token₀:    Self (projected features)               │
    │   Token₁:    1-hop HGT + DropEdge(0.05)              │
    │   Token_Hyper: HGNN on meta-path hypergraph (Meso)   │ ← optional
    │   Token₃:    Cross-Type Context Pool (Macro: global) │
    │                                                      │
    │ Stage 3: CRF Adaptive Fusion                         │
    │   → Learned per-node scalar weights across tokens    │
    │                                                      │
    │ Predictor: Dual-Pathway + PSE Structural Features  │
    │   → Bilinear (Path A) + Cross-Attention (Path B)     │
    │   → Multi-view struct features (12-dim) injected     │
    └──────────────────────────────────────────────────────┘

    Experimental flags (off by default):
      use_attn_temp, token_dropout, consistency_weight,
      contrastive_weight, aux_task_weight, use_hypergraph, use_llm
    """

    def __init__(self,
                 input_dim_circrna: int,
                 input_dim_disease: int,
                 input_dim_mirna: int,
                 input_dim_lncrna: int,
                 hidden_dim: int = 96,
                 lap_pe_dim: int = 16,
                 rwse_dim: int = 16,
                 gcn_layers: int = 1,
                 gcn_dropout: float = 0.3,
                 decoder_hidden: Tuple[int, ...] = (96, 48),
                 decoder_dropout: float = 0.3,
                 num_resolutions: int = 4,
                 num_node_types: int = 4,
                 struct_feat_dim: int = 0,
                 use_hgt: bool = True,
                 hgt_heads: int = 2,
                 use_hypergraph: bool = False,
                 hgcn_dropout: float = 0.3,
                 use_llm: bool = False,
                 llm_proj_dim: int = 64,
                 llm_proj_dim_circ: int = 0,
                 llm_proj_dim_disease: int = 0,
                 llm_proj_dim_mirna: int = 0,
                 edge_dropout: float = 0.0,
                 token_dropout: float = 0.0,
                 use_attn_temp: bool = False,
                 use_sigmoid_gate: bool = False,
                 use_multiscale_topo: bool = True,
                 msne_num_scales: int = 3,
                 use_metapath: bool = True,
                 metapath_mode: str = "concat",
                 metapath_num_paths: int = 3,
                 metapath_residual_weight: float = 0.1,
                 num_edge_types: int = 3,
                 use_metapath_edges: bool = False,
                 crt_single_token: bool = False,
                 mean_gate: bool = False,
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_node_types = num_node_types
        self.struct_feat_dim = struct_feat_dim
        self.use_hgt = use_hgt
        self.use_hypergraph = use_hypergraph
        self.use_llm = use_llm
        self.llm_proj_dim = llm_proj_dim
        self.token_dropout = token_dropout  # v3.1: TokenDrop rate
        self.use_metapath = use_metapath
        self.metapath_mode = metapath_mode
        self.metapath_num_paths = metapath_num_paths
        self.metapath_residual_weight = metapath_residual_weight
        self.num_edge_types = num_edge_types
        self.use_metapath_edges = use_metapath_edges
        self.crt_single_token = crt_single_token
        self.mean_gate = mean_gate
        self.pe_dim = lap_pe_dim + rwse_dim
        pe_dim = self.pe_dim
        # Per-type LLM projection dims (0 = no LLM for that type)
        # If all zero and use_llm=True, fall back to llm_proj_dim for all (backward compat)
        _pc = llm_proj_dim_circ if (llm_proj_dim_circ > 0 or llm_proj_dim_disease > 0 or llm_proj_dim_mirna > 0) else llm_proj_dim
        _pd = llm_proj_dim_disease if (llm_proj_dim_circ > 0 or llm_proj_dim_disease > 0 or llm_proj_dim_mirna > 0) else llm_proj_dim
        _pm = llm_proj_dim_mirna if (llm_proj_dim_circ > 0 or llm_proj_dim_disease > 0 or llm_proj_dim_mirna > 0) else llm_proj_dim
        self._llm_pdims = (_pc, _pd, _pm)  # for encode() to check which types have LLM

        # ─── LLM Feature Projection (v2.7): trainable Linear(768→proj_dim) ───
        # Projects frozen LLM embeddings to a compact space before concatenation.
        # Only created for types with non-zero projection dim.
        if use_llm:
            if _pc > 0:
                self.llm_proj_circ = nn.Linear(768, _pc)
            if _pd > 0:
                self.llm_proj_disease = nn.Linear(768, _pd)
            if _pm > 0:
                self.llm_proj_mirna = nn.Linear(768, _pm)
            # lncRNA: no LLM features (no sequence data available)

        # ─── Stage 1: Type-Specific Projections ───
        # When use_llm=True, input_dim includes the projected LLM features
        _extra_c = _pc if use_llm else 0
        _extra_d = _pd if use_llm else 0
        _extra_m = _pm if use_llm else 0
        _extra_l = 0  # lncRNA never has LLM features
        self.proj_circ = TypeSpecificProjection(
            input_dim_circrna + _extra_c, hidden_dim, pe_dim, gcn_dropout)
        self.proj_disease = TypeSpecificProjection(
            input_dim_disease + _extra_d, hidden_dim, pe_dim, gcn_dropout)
        self.proj_mirna = TypeSpecificProjection(
            input_dim_mirna + _extra_m, hidden_dim, pe_dim, gcn_dropout)
        self.proj_lncrna = TypeSpecificProjection(
            input_dim_lncrna + _extra_l, hidden_dim, pe_dim, gcn_dropout)

        # ─── Stage 2: Message Passing (GCN or HGT) + Cross-Type Pool ───
        if use_hgt:
            self.gcn = HGTEncoder(hidden_dim, gcn_layers, num_edge_types=num_edge_types,
                                  num_heads=hgt_heads, dropout=gcn_dropout,
                                  edge_dropout=edge_dropout,
                                  use_attn_temp=use_attn_temp,
                                  use_sigmoid_gate=use_sigmoid_gate)
        else:
            self.gcn = GCNEncoder(hidden_dim, gcn_layers, num_edge_types=num_edge_types,
                                  dropout=gcn_dropout)
        self.cross_type_pool = CrossTypeContextPool(
            hidden_dim, num_node_types, gcn_dropout)

        # ─── Hypergraph Convolution (v2.5): independent meso-scale token ───
        if use_hypergraph:
            self.hgcn = HypergraphConv(hidden_dim, dropout=hgcn_dropout)

        self.use_multiscale_topo = use_multiscale_topo
        self.msne_num_scales = msne_num_scales

        # ─── v3.3: Meta-Path Neighbor Encoding (MPNE) ───
        # Concat mode: meta-path features are computed externally and added as
        #   residual to projected embeddings before HGT (zero extra params in model).
        # CRF mode: each meta-path (CMC/DMD/CMD) becomes a CRF token alongside HGT.
        #   Requires small projection layers (Linear(hidden→hidden)) per meta-path.
        if use_metapath and metapath_mode == "crt":
            self.mp_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(gcn_dropout),
                ) for _ in range(metapath_num_paths)
            ])
        else:
            self.mp_proj = None

        # ─── Stage 3: CRF (Resolution PE + adaptive fusion, v2.2: no attention) ───
        # v3.3: CRF only used in metapath CRF mode (1 HGT + N meta-path tokens).
        # In concat mode: single HGT token (num_resolutions=1, CRF is identity pass-through).
        # v3.3: use_metapath_edges → CRF multi-resolution (Token₀ + Token₁ + Token₃).
        if crt_single_token:
            crt_num_res = 1  # Ablation: only Token₁ (HGT), drop Token₀+Token₃
        elif use_metapath_edges:
            crt_num_res = 3  # Self + HGT + CrossType (v3.2-style)
        elif use_metapath and metapath_mode == "crt":
            crt_num_res = metapath_num_paths + 1  # meta-paths + HGT
        else:
            crt_num_res = 1  # single HGT token, no multi-resolution
        if use_hypergraph:
            crt_num_res += 1
        self.crt = CrossResolutionFusion(hidden_dim, crt_num_res)

        # ─── Predictor ───
        self.predictor = DualPathwayPredictor(
            hidden_dim, decoder_hidden, decoder_dropout,
            struct_feat_dim=struct_feat_dim,
            mean_gate=mean_gate)

        # ─── Learnable type embeddings (for node-type awareness) ───
        self.type_embed = nn.Parameter(
            torch.randn(num_node_types, hidden_dim) * 0.02)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self,
               x_circ: torch.Tensor,
               x_disease: torch.Tensor,
               x_mirna: torch.Tensor,
               x_lncrna: torch.Tensor,
               edge_index: torch.Tensor,
               node_counts: Dict[str, int],
               edge_weight: Optional[torch.Tensor] = None,
               edge_index_by_type: dict = None,
               llm_circ: Optional[torch.Tensor] = None,
               llm_disease: Optional[torch.Tensor] = None,
               llm_mirna: Optional[torch.Tensor] = None,
               msne_feat: Optional[torch.Tensor] = None,
               mp_feat: Optional[torch.Tensor] = None,
               mp_feat_dict: Optional[Dict[str, torch.Tensor]] = None,
               return_tokens: bool = False,
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Heterogeneous encoder — 3-stage hierarchical multi-scale encoding.

        v2.7 Pipeline:
          Stage 0: (optional) Project frozen LLM embeddings via trainable
                   Linear(768→64) and concatenate to raw features
          Stage 1: Type-specific projection → Token₀ (Self)
          Stage 2: Token₁ (Micro: 1-hop HGT) → Token_Hyper (Meso: HGNN)
                   → Token₃ (Macro: CrossType Context)
          Stage 3: CRF adaptive fusion across 4 resolutions

        Args:
            x_circ:    (N_c, F_c) circRNA features
            x_disease: (N_d, F_d) disease features
            x_mirna:   (N_m, F_m) miRNA features
            x_lncrna:  (N_l, F_l) lncRNA features
            edge_index: (2, E) combined edges (fallback uniform mode)
            node_counts: dict with N_circ, N_disease, N_mirna, N_lncrna
            edge_weight: optional edge weights
            edge_index_by_type: Dict[int, Tensor] — per-type edges (hetero mode)
            llm_circ:    (N_c, 768) optional frozen DNABERT circRNA embeddings
            llm_disease: (N_d, 768) optional frozen PubMedBERT disease embeddings
            llm_mirna:   (N_m, 768) optional frozen DNABERT miRNA embeddings

        Returns:
            h_circ:   (N_circ, D) fused circRNA embeddings
            h_disease: (N_disease, D) fused disease embeddings
            (if return_tokens: also token0, token1, token3, h_mirna_out)
        """
        N_c = node_counts['N_circ']
        N_d = node_counts['N_disease']
        N_m = node_counts['N_mirna']
        N_l = node_counts['N_lncrna']
        D = self.hidden_dim

        # ─── Stage 0: LLM Feature Projection (v2.7) ───
        # Project frozen LLM embeddings through trainable Linear(768→proj_dim),
        # then concatenate with existing features. Only applies to node types
        # with non-zero projection dim (controlled by --llm_types).
        if self.use_llm:
            _pc, _pd, _pm = self._llm_pdims
            if llm_circ is not None and _pc > 0 and hasattr(self, 'llm_proj_circ'):
                llm_c_proj = self.llm_proj_circ(llm_circ.to(device=x_circ.device))
                x_circ = torch.cat([x_circ, llm_c_proj], dim=-1)
            if llm_disease is not None and _pd > 0 and hasattr(self, 'llm_proj_disease'):
                llm_d_proj = self.llm_proj_disease(llm_disease.to(device=x_disease.device))
                x_disease = torch.cat([x_disease, llm_d_proj], dim=-1)
            if llm_mirna is not None and _pm > 0 and hasattr(self, 'llm_proj_mirna'):
                llm_m_proj = self.llm_proj_mirna(llm_mirna.to(device=x_mirna.device))
                x_mirna = torch.cat([x_mirna, llm_m_proj], dim=-1)
            # lncRNA: no LLM features (no sequence data)

        # ─── Stage 1: Type-Specific Tokenization ───
        # Use LapPE+RWSE from node_counts if available; else fall back to zeros
        pe_lap = node_counts.get('pe_lap', None)
        pe_rw = node_counts.get('pe_rw', None)
        _pe_dim = getattr(self, 'pe_dim', 32)
        if pe_lap is not None and pe_rw is not None and _pe_dim > 0:
            pe_lap = pe_lap.to(device=x_circ.device)
            pe_rw = pe_rw.to(device=x_circ.device)
            pe_all = torch.cat([pe_lap, pe_rw], dim=-1)[:, :_pe_dim]  # truncate to configured dims
            pe_c = pe_all[:N_c]
            pe_d = pe_all[N_c:N_c + N_d]
            pe_m = pe_all[N_c + N_d:N_c + N_d + N_m]
            pe_l = pe_all[N_c + N_d + N_m:]
        else:
            pe_c = torch.zeros(N_c, _pe_dim, device=x_circ.device)
            pe_d = torch.zeros(N_d, _pe_dim, device=x_disease.device)
            pe_m = torch.zeros(N_m, _pe_dim, device=x_mirna.device)
            pe_l = torch.zeros(N_l, _pe_dim, device=x_lncrna.device)

        h0_c = self.proj_circ(x_circ, pe_c)
        h0_d = self.proj_disease(x_disease, pe_d)
        h0_m = self.proj_mirna(x_mirna, pe_m)
        h0_l = self.proj_lncrna(x_lncrna, pe_l)

        # Add type embeddings for node-type awareness
        h0_c = h0_c + self.type_embed[0]
        h0_d = h0_d + self.type_embed[1]
        h0_m = h0_m + self.type_embed[2]
        h0_l = h0_l + self.type_embed[3]

        h_all = torch.cat([h0_c, h0_d, h0_m, h0_l], dim=0)

        # ─── Stage 2: Meta-Path Neighbor Residual + HGT ───
        # v3.3 MPNE (concat mode): meta-path degree features added as residual BEFORE HGT.
        # v3.3 Metapath Edges: CMC/DMD as separate HGT edge types (no feature injection).
        #   When use_metapath_edges=True: skip MPNE concat, use CRF multi-resolution.

        if self.use_metapath_edges:
            # v3.3: Meta-path edges mode — use CMC/DMD as HGT edge types.
            # Skip MPNE features; meta-path info flows through dedicated HGT projections.
            h_for_hgt = h_all
        elif self.use_metapath and self.metapath_mode == "concat" and mp_feat is not None:
            # v3.3: MPNE concat — inject meta-path degree features as input residual
            mp_feat_d = mp_feat.to(device=h_all.device, dtype=h_all.dtype)
            h_for_hgt = h_all + self.metapath_residual_weight * mp_feat_d
        else:
            h_for_hgt = h_all

        # 1-hop HGT (primary token)
        n_layers_available = len(self.gcn.layers) if hasattr(self.gcn, 'layers') else len(self.gcn.convs)
        if n_layers_available >= 1:
            token_hgt = self.gcn(h_for_hgt, edge_index_by_type=edge_index_by_type,
                                 edge_index=edge_index, edge_weight=edge_weight,
                                 num_layers=None)
        else:
            token_hgt = h_for_hgt

        # Build type embedding indices (used for CRF mode)
        t_idx = torch.zeros(N_c, dtype=torch.long, device=h_all.device)
        t_idx = torch.cat([t_idx, torch.full((N_d,), 1, dtype=torch.long, device=h_all.device)])
        t_idx = torch.cat([t_idx, torch.full((N_m,), 2, dtype=torch.long, device=h_all.device)])
        t_idx = torch.cat([t_idx, torch.full((N_l,), 3, dtype=torch.long, device=h_all.device)])
        t_emb_all = self.type_embed[t_idx]  # (N, D)

        if self.use_metapath_edges:
            if self.crt_single_token:
                # Ablation: only Token₁ (1-hop HGT with 5 edge types), no CRF fusion
                crt_tokens = [token_hgt]
            else:
                # v3.3: CRF multi-resolution — Token₀ (Self) + Token₁ (1-hop HGT) + Token₃ (CrossType)
                token0 = h_all
                token1 = token_hgt
                token3 = self.cross_type_pool(h_all, (N_c, N_d, N_m, N_l))
                crt_tokens = [token0, token1, token3]

                if self.use_hypergraph and edge_index_by_type is not None:
                    token_hyper = self.hgcn(token1, edge_index_by_type, node_counts)
                    crt_tokens.append(token_hyper)
        elif self.use_metapath and self.metapath_mode == "crt" and mp_feat_dict is not None:
            # ── CRF mode: each meta-path as a separate token ──
            crt_tokens = []
            path_keys = ['cmc', 'dmd', 'cmd']
            for i, key in enumerate(path_keys):
                if key in mp_feat_dict and i < len(self.mp_proj):
                    mp_tok = mp_feat_dict[key].to(device=h_all.device, dtype=h_all.dtype)
                    mp_tok = self.mp_proj[i](mp_tok) + t_emb_all
                    crt_tokens.append(mp_tok)

            token_hgt = token_hgt + t_emb_all
            crt_tokens.append(token_hgt)

            if self.use_hypergraph and edge_index_by_type is not None:
                token_hyper = self.hgcn(token_hgt, edge_index_by_type, node_counts)
                token_hyper = token_hyper + t_emb_all
                crt_tokens.append(token_hyper)
        else:
            # Single-token (no CRF multi-resolution)
            crt_tokens = [token_hgt]

        # ─── Stage 3: CRF Adaptive Fusion ───

        # ── v3.1 TokenDrop: per-node stochastic token masking during training ──
        # Randomly drops resolution tokens per node, forcing CRF to be robust
        # to missing resolutions. Like Dropout but for multi-scale tokens.
        # At inference (eval mode), all tokens are used unchanged.
        if self.training and self.token_dropout > 0:
            crt_tokens = list(crt_tokens)  # ensure mutable
            for i in range(len(crt_tokens)):
                # Per-node binary mask: (N, 1)
                mask = torch.bernoulli(
                    torch.full((crt_tokens[i].shape[0], 1),
                               1.0 - self.token_dropout,
                               device=crt_tokens[i].device))
                # No rescaling needed — CRF softmax naturally adapts
                crt_tokens[i] = crt_tokens[i] * mask
        h_fused = self.crt(crt_tokens)

        # ─── Split back: only circRNA and disease for prediction ───
        h_circ_out = h_fused[:N_c]
        h_dis_out = h_fused[N_c:N_c + N_d]

        if return_tokens:
            h_mirna_out = h_fused[N_c + N_d:N_c + N_d + N_m]
            return h_circ_out, h_dis_out, h_mirna_out
        return h_circ_out, h_dis_out

    def predict(self,
                h_circ: torch.Tensor,
                h_disease: torch.Tensor,
                circ_indices: torch.Tensor,
                disease_indices: torch.Tensor,
                struct_feat: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        logits, _ = self.predictor(h_circ, h_disease, circ_indices, disease_indices,
                                   struct_feat=struct_feat)
        return torch.sigmoid(logits)

    def forward(self,
                x_circ: torch.Tensor,
                x_disease: torch.Tensor,
                x_mirna: torch.Tensor,
                x_lncrna: torch.Tensor,
                edge_index: torch.Tensor,
                node_counts: Dict[str, int],
                circ_indices: torch.Tensor,
                disease_indices: torch.Tensor,
                edge_index_by_type: dict = None,
                return_attention: bool = False,
                struct_feat: Optional[torch.Tensor] = None,
                llm_circ: Optional[torch.Tensor] = None,
                llm_disease: Optional[torch.Tensor] = None,
                llm_mirna: Optional[torch.Tensor] = None,
                msne_feat: Optional[torch.Tensor] = None,
                mp_feat: Optional[torch.Tensor] = None,
                mp_feat_dict: Optional[Dict[str, torch.Tensor]] = None,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h_circ, h_disease = self.encode(
            x_circ, x_disease, x_mirna, x_lncrna,
            edge_index, node_counts, edge_index_by_type=edge_index_by_type,
            llm_circ=llm_circ, llm_disease=llm_disease, llm_mirna=llm_mirna,
            msne_feat=msne_feat, mp_feat=mp_feat, mp_feat_dict=mp_feat_dict)

        logits, attn = self.predictor(
            h_circ, h_disease, circ_indices, disease_indices,
            return_attention=return_attention,
            struct_feat=struct_feat)
        return logits, attn

    def get_attention_summary(self) -> dict:
        info = {}
        if hasattr(self.crt, 'last_alpha'):
            info['resolution_alpha'] = self.crt.last_alpha
        return info
