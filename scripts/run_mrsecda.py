"""
MRSECDA v2: Heterogeneous Graph Training Script

Usage:
  python scripts/run_mrsecda.py --dataset circR2Cancer               # Single dataset
  python scripts/run_mrsecda.py --dataset circR2Cancer --fold 0      # Single fold
  python scripts/run_mrsecda.py --datasets circR2Cancer circR2Disease # Multiple
  python scripts/run_mrsecda.py --ablation no_mirna                  # Ablation
"""
from __future__ import annotations
import os, sys, time, json, argparse, warnings, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Tuple, Dict, List, Optional

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config
from models.mrsecda_hetero import MRSECDAHetero, multi_view_contrastive_loss
from data.dataset import CDADataset, kfold_split, train_val_split
from data.hetero_data import build_hetero_graph
from data.structural_features import StructuralFeatureComputer, MultiViewStructuralFeatureComputer
from trainer import sample_negative_pairs, prepare_batches
from recall_at_k import compute_recall_at_k

RESULT_DIR = os.path.join(ROOT, "results", "mrsecda")
os.makedirs(RESULT_DIR, exist_ok=True)

ALL_DATASETS = ["circR2Disease", "circR2Cancer", "circRNADisease", "circ2Traits", "circad"]


# =============================================================================
# LLM Embedding Helpers
# =============================================================================

def _align_embeddings(
    embeddings: "torch.Tensor",
    target_names: list,
    expected_n: int,
    tag: str = "",
) -> "torch.Tensor":
    """Align precomputed embedding tensor with target name list.

    If the embedding row count matches expected_n, returns as-is.
    Otherwise zero-pads or truncates with a warning.

    Args:
        embeddings: (M, D) tensor from precomputed .pt file
        target_names: ordered list of node names (length N)
        expected_n: expected number of rows (should equal len(target_names))
        tag: label for log messages

    Returns:
        (N, D) float32 tensor aligned to target order
    """
    M, D = embeddings.shape
    N = expected_n
    if M == N:
        return embeddings.to(torch.float32)

    # Mismatch: pad or warn
    print(f"  [LLM] {tag} embedding count mismatch: got {M}, expected {N}. "
          f"{'Zero-padding' if M < N else 'Truncating'}.")
    result = torch.zeros(N, D, dtype=torch.float32)
    n_copy = min(M, N)
    result[:n_copy] = embeddings[:n_copy].to(torch.float32)
    return result


def _align_embeddings_from_dict(
    emb_dict: dict,
    target_names: list,
    expected_n: int,
    tag: str = "",
) -> "torch.Tensor":
    """Align {name: tensor} embedding dict with target name list.

    Looks up each target name in the dict. Missing names get zero vectors.

    Args:
        emb_dict: {name: (D,) or (1, D) tensor} mapping
        target_names: ordered list of names
        expected_n: expected number of rows
        tag: label for log messages

    Returns:
        (N, D) float32 tensor.
    """
    N = expected_n
    if not emb_dict:
        print(f"  [LLM] {tag}: empty embedding dict — all zeros")
        return torch.zeros(N, 768, dtype=torch.float32)  # fallback to 768

    # Get D from first embedding
    first_emb = next(iter(emb_dict.values()))
    if first_emb.dim() == 1:
        D = first_emb.shape[0]
    elif first_emb.dim() == 2:
        D = first_emb.shape[1]
    else:
        D = 768  # fallback

    # Build case-insensitive lookup for fuzzy matching
    # (HMDD uses "mir-", miRBase uses "miR-" — same miRNA, different case)
    emb_dict_lower = {k.lower(): (k, v) for k, v in emb_dict.items()}

    result = torch.zeros(N, D, dtype=torch.float32)
    n_found = 0
    for i, name in enumerate(target_names):
        emb = emb_dict.get(name)
        if emb is None:
            # Fuzzy match: case-insensitive
            lower_name = name.lower()
            if lower_name in emb_dict_lower:
                _, emb = emb_dict_lower[lower_name]
        if emb is not None:
            if emb.dim() == 1:
                result[i] = emb.to(torch.float32)
            else:
                result[i] = emb.squeeze(0).to(torch.float32)
            n_found += 1
        # else: stay zero

    # Mean-fill for missing entries (instead of zero-fill)
    if n_found > 0 and n_found < N:
        found_mask = (result.abs().sum(dim=1) > 1e-8)
        mean_emb = result[found_mask].mean(dim=0)
        result[~found_mask] = mean_emb
        print(f"  [LLM] {tag}: {n_found}/{N} matched from dict "
              f"({n_found/N*100:.1f}%), {N - n_found} mean-filled")
    elif n_found == 0:
        print(f"  [LLM] {tag}: 0/{N} matched — all zeros (no mean available)")
    return result


# =============================================================================
# Metrics
# =============================================================================

def find_best_threshold(
    scores: np.ndarray, labels: np.ndarray,
    low: float = 0.05, high: float = 0.95, step: float = 0.01,
) -> float:
    """Search for threshold that maximizes F1 on validation set."""
    from sklearn.metrics import f1_score as _f1
    best_thr = 0.5
    best_f1 = 0.0
    for thr in np.arange(low, high + step, step):
        preds = (scores > thr).astype(int)
        f1 = _f1(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr)


def compute_metrics(scores: np.ndarray, labels: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, accuracy_score,
        precision_score, recall_score, f1_score,
    )
    auc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    preds = (scores > threshold).astype(int)
    return {
        "auc": auc, "aupr": aupr,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_model_hetero(
    model: MRSECDAHetero,
    x_circ: torch.Tensor, x_disease: torch.Tensor,
    x_mirna: torch.Tensor, x_lncrna: torch.Tensor,
    edge_index: torch.Tensor, node_counts: Dict[str, int],
    circ_idx: np.ndarray, dis_idx: np.ndarray, labels: np.ndarray,
    batch_size: int = 8192, device: str = "cpu",
    edge_index_by_type: dict = None,
    struct_feat_computer: StructuralFeatureComputer = None,
    llm_circ: torch.Tensor = None,
    llm_disease: torch.Tensor = None,
    llm_mirna: torch.Tensor = None,
    msne_feat_all: torch.Tensor = None,
    threshold: float = 0.5,
    return_scores: bool = False,
) -> Dict[str, float]:
    model.eval()
    h_circ, h_disease = model.encode(
        x_circ.to(device), x_disease.to(device),
        x_mirna.to(device), x_lncrna.to(device),
        edge_index.to(device), node_counts,
        edge_index_by_type=edge_index_by_type,
        llm_circ=llm_circ, llm_disease=llm_disease, llm_mirna=llm_mirna,
        msne_feat=None.to(device) if msne_feat_all is not None else None)

    all_scores = []
    n = len(labels)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ci = torch.tensor(circ_idx[start:end], dtype=torch.long, device=device)
        di = torch.tensor(dis_idx[start:end], dtype=torch.long, device=device)
        sf = None
        if struct_feat_computer is not None:
            sf_np = struct_feat_computer.compute_features(
                circ_idx[start:end], dis_idx[start:end])
            sf = torch.tensor(sf_np, dtype=torch.float32, device=device)
        scores = model.predict(h_circ, h_disease,
                               circ_indices=ci, disease_indices=di, struct_feat=sf)
        all_scores.append(scores.cpu())

    scores_np = torch.cat(all_scores).numpy()
    metrics = compute_metrics(scores_np, labels, threshold=threshold)
    if return_scores:
        metrics["_scores"] = scores_np
    return metrics


def score_full_matrix(
    model: MRSECDAHetero,
    x_circ: torch.Tensor, x_disease: torch.Tensor,
    x_mirna: torch.Tensor, x_lncrna: torch.Tensor,
    edge_index: torch.Tensor, node_counts: Dict[str, int],
    edge_index_by_type: dict = None,
    struct_feat_computer: StructuralFeatureComputer = None,
    llm_circ: torch.Tensor = None,
    llm_disease: torch.Tensor = None,
    llm_mirna: torch.Tensor = None,
    batch_size: int = 8192, device: str = "cpu",
) -> np.ndarray:
    """Encode the full graph and score every (circRNA, disease) pair.

    Returns S (n_circ, n_disease) probability matrix (higher = more associated).
    """
    model.eval()
    with torch.no_grad():
        h_circ, h_disease = model.encode(
            x_circ.to(device), x_disease.to(device),
            x_mirna.to(device), x_lncrna.to(device),
            edge_index.to(device), node_counts,
            edge_index_by_type=edge_index_by_type,
            llm_circ=llm_circ, llm_disease=llm_disease, llm_mirna=llm_mirna,
            msne_feat=None)

        n_circ = h_circ.shape[0]
        n_dis = h_disease.shape[0]
        circ_grid = np.repeat(np.arange(n_circ), n_dis)
        dis_grid = np.tile(np.arange(n_dis), n_circ)

        all_scores = []
        total = len(circ_grid)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            ci = torch.tensor(circ_grid[start:end], dtype=torch.long, device=device)
            di = torch.tensor(dis_grid[start:end], dtype=torch.long, device=device)
            sf = None
            if struct_feat_computer is not None:
                sf_np = struct_feat_computer.compute_features(
                    circ_grid[start:end], dis_grid[start:end])
                sf = torch.tensor(sf_np, dtype=torch.float32, device=device)
            scores = model.predict(h_circ, h_disease,
                                   circ_indices=ci, disease_indices=di, struct_feat=sf)
            all_scores.append(scores.cpu())

        S = torch.cat(all_scores).detach().cpu().numpy().reshape(n_circ, n_dis)
    return S


# =============================================================================
# Training (one fold)
# =============================================================================

def _smooth_labels(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Apply label smoothing: hard {0,1} → soft {ε/2, 1−ε/2}."""
    if smoothing > 0:
        return labels * (1.0 - smoothing) + smoothing / 2.0
    return labels


def focal_bce_loss(logits: torch.Tensor, targets: torch.Tensor,
                   gamma: float = 1.0, alpha: float = 0.25,
                   pos_weight: torch.Tensor = None) -> torch.Tensor:
    """Focal BCE Loss — down-weights easy samples, focuses on hard cases.

    FL(p_t) = -α_t * (1 − p_t)^γ * log(p_t)

    For small datasets (647 associations), easy negatives dominate the loss.
    Focal loss forces the encoder to learn from the most informative pairs.

    Args:
        logits: (B,) raw logits
        targets: (B,) 0/1 labels
        gamma: focusing strength (0=standard BCE, 2=strong focusing)
        alpha: class weight for positives
        pos_weight: optionally scale positive class (applied after focal scaling)
    Returns:
        scalar loss
    """
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)  # p if y=1, 1-p if y=0
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    loss = (focal_weight * bce)
    if pos_weight is not None:
        pw_factor = pos_weight * targets + (1 - targets)
        loss = loss * pw_factor.squeeze(-1) if pw_factor.dim() > 1 else loss * pw_factor
    return loss.mean()


def ranking_aware_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                              gamma: float = 2.0, alpha: float = 0.25,
                              pos_weight: torch.Tensor = None,
                              bpr_weight: float = 0.1, bpr_k: int = 5) -> torch.Tensor:
    """RA-Focal: Focal Loss + Pairwise BPR Ranking Loss (v3.3).

    Focal Loss focuses on hard samples per-point.
    BPR (Bayesian Personalized Ranking) loss optimizes pairwise ordering —
    for each positive, its score should exceed all negative scores.
    This is directly aligned with AUC (area under ROC = ranking quality).

    Zero extra model parameters. Only changes training loss.

    Args:
        logits: (B,) raw logits
        targets: (B,) 0/1 labels
        gamma: focal focusing strength
        alpha: focal class balance
        pos_weight: optional positive class scaling
        bpr_weight: BPR loss weight (0.05-0.2 range)
        bpr_k: negatives to compare each positive against (subsampled)

    Returns:
        scalar combined loss
    """
    # ── 1. Focal Loss (pointwise hard-sample focusing) ──
    focal = focal_bce_loss(logits, targets, gamma=gamma, alpha=alpha,
                           pos_weight=pos_weight)

    # ── 2. BPR Ranking Loss (pairwise ordering) ──
    pos_mask = targets > 0.5
    neg_mask = targets < 0.5

    if pos_mask.any() and neg_mask.any():
        pos_idx = torch.where(pos_mask)[0]
        neg_idx = torch.where(neg_mask)[0]

        # Subsample K negatives per positive for efficiency
        k = min(bpr_k, len(neg_idx))
        rand_idx = torch.randint(0, len(neg_idx), (len(pos_idx), k),
                                 device=logits.device)
        chosen_neg = neg_idx[rand_idx]                       # (P, K)

        pos_scores = logits[pos_idx].unsqueeze(1)            # (P, 1)
        neg_scores = logits[chosen_neg]                      # (P, K)
        diff = pos_scores - neg_scores                       # (P, K)
        bpr = -F.logsigmoid(diff).mean()
    else:
        bpr = torch.tensor(0.0, device=logits.device)

    return focal + bpr_weight * bpr


# =============================================================================
# v3.3: Meta-Path Structural Features (miRNA-mediated view)
# =============================================================================

def compute_metapath_features(circ_to_mirnas: list, dis_to_mirnas: list,
                              mirna_degrees: dict,
                              circ_indices: np.ndarray,
                              dis_indices: np.ndarray) -> np.ndarray:
    """Compute 4 structural features based on circRNA→miRNA→disease meta-path.

    Captures the molecular sponge mechanism: circRNA regulates disease by
    competitively binding miRNAs. Unlike existing PSE views (graph adjacency),
    this view measures miRNA-mediated connectivity directly.

    Features: miRNA-Jaccard, miRNA-AA, miRNA-RA, meta-inv_dist (always 2-hop).
    Zero extra model parameters. Complements PSE's 12-dim → 16-dim total.
    """
    import math as _math
    n_pairs = len(circ_indices)
    jaccard = np.zeros(n_pairs, dtype=np.float32)
    aa = np.zeros(n_pairs, dtype=np.float32)
    ra = np.zeros(n_pairs, dtype=np.float32)
    inv_dist = np.full(n_pairs, 1.0 / 3.0, dtype=np.float32)

    for i, (c, d) in enumerate(zip(circ_indices, dis_indices)):
        m_c = circ_to_mirnas[c] if c < len(circ_to_mirnas) else set()
        m_d = dis_to_mirnas[d] if d < len(dis_to_mirnas) else set()
        if not m_c or not m_d:
            continue
        common = m_c & m_d
        if common:
            inv_dist[i] = 0.5
            union = len(m_c | m_d)
            jaccard[i] = len(common) / max(union, 1)
            aa_val = sum(1.0 / max(_math.log(max(mirna_degrees.get(m, 2), 2)), 1e-8)
                        for m in common)
            aa[i] = aa_val
            ra_val = sum(1.0 / max(mirna_degrees.get(m, 1), 1) for m in common)
            ra[i] = ra_val

    if aa.max() > 0:
        aa = np.log1p(aa) / np.log1p(aa.max())
    if ra.max() > 0:
        ra = np.log1p(ra) / np.log1p(ra.max())
    return np.stack([jaccard, aa, ra, inv_dist], axis=-1)


class CombinedStructFeatComputer:
    """PSE multi-view (12-dim) + meta-path (4-dim) = 16-dim features."""

    def __init__(self, base_computer, circ_to_mirnas, dis_to_mirnas, mirna_degrees):
        self.base = base_computer
        self.circ_to_mirnas = circ_to_mirnas
        self.dis_to_mirnas = dis_to_mirnas
        self.mirna_degrees = mirna_degrees

    @property
    def feat_dim(self) -> int:
        return self.base.feat_dim + 4

    def compute_features(self, circ_indices, dis_indices):
        base_feat = self.base.compute_features(circ_indices, dis_indices)
        meta_feat = compute_metapath_features(
            self.circ_to_mirnas, self.dis_to_mirnas, self.mirna_degrees,
            circ_indices, dis_indices)
        return np.concatenate([base_feat, meta_feat], axis=-1)


# =============================================================================
# v3.3: Topology-Guided Negative Sampling (TGNS)
# =============================================================================

def _build_mirna_adjacency(edge_index_by_type: dict,
                           N_circ: int, N_disease: int, N_mirna: int):
    """Extract circRNA↔miRNA and disease↔miRNA adjacency from typed edges.

    Edge type 2 contains both circRNA-miRNA and miRNA-disease edges.
    Node index layout:
      circRNA: [0, N_circ)
      disease: [N_circ, N_circ+N_disease)
      miRNA:   [N_circ+N_disease, N_circ+N_disease+N_mirna)

    Returns:
        circ_to_mirnas: list of sets, circ_to_mirnas[c] = {m_local_idx, ...}
        mirna_to_diseases: list of sets, mirna_to_diseases[m] = {d_local_idx, ...}
    """
    off_d = N_circ
    off_m = N_circ + N_disease

    circ_to_mirnas = [set() for _ in range(N_circ)]
    mirna_to_diseases = [set() for _ in range(N_mirna)]

    ei2 = edge_index_by_type.get(2, None)
    if ei2 is None or ei2.shape[1] == 0:
        return circ_to_mirnas, mirna_to_diseases

    src = ei2[0].cpu().numpy()
    tgt = ei2[1].cpu().numpy()

    for s, t in zip(src, tgt):
        # circRNA → miRNA
        if s < N_circ and off_m <= t < off_m + N_mirna:
            circ_to_mirnas[int(s)].add(int(t - off_m))
        # miRNA → circRNA (reverse edge)
        elif off_m <= s < off_m + N_mirna and t < N_circ:
            circ_to_mirnas[int(t)].add(int(s - off_m))
        # miRNA → disease
        elif off_m <= s < off_m + N_mirna and off_d <= t < off_d + N_disease:
            mirna_to_diseases[int(s - off_m)].add(int(t - off_d))
        # disease → miRNA (reverse edge)
        elif off_d <= s < off_d + N_disease and off_m <= t < off_m + N_mirna:
            mirna_to_diseases[int(t - off_m)].add(int(s - off_d))

    return circ_to_mirnas, mirna_to_diseases


# =============================================================================
# v3.4: Meta-Path HGT Edge Types (CMC, DMD)
# =============================================================================

def build_metapath_hgt_edges(edge_index_by_type: dict,
                              N_circ: int, N_disease: int, N_mirna: int,
                              min_shared: int = 1, topk: int = 15):
    """Build CMC and DMD meta-path edges as new HGT edge types.

    CMC (CircRNA-MiRNA-CircRNA, type 3):
        Undirected edges between circRNAs that share >= min_shared miRNAs.
        Captures circRNA-circRNA functional similarity via the sponge mechanism.

    DMD (Disease-MiRNA-Disease, type 4):
        Undirected edges between diseases that share >= min_shared miRNAs.
        Captures disease-disease similarity via shared miRNA regulation.

    Both use Jaccard similarity + top-K sparsification to keep the graph
    from becoming too dense. Returns both the new edge dict and the extracted
    adjacency lists (for potential downstream use in struct features).
    """
    off_d = N_circ
    off_m = N_circ + N_disease

    # ── Extract miRNA adjacency from edge type 2 ──
    circ_to_mirnas = [set() for _ in range(N_circ)]
    dis_to_mirnas = [set() for _ in range(N_disease)]

    ei2 = edge_index_by_type.get(2, None)
    if ei2 is None or ei2.shape[1] == 0:
        return {}, circ_to_mirnas, dis_to_mirnas

    src = ei2[0].cpu().numpy()
    tgt = ei2[1].cpu().numpy()

    for s, t in zip(src, tgt):
        if s < N_circ and off_m <= t < off_m + N_mirna:
            circ_to_mirnas[int(s)].add(int(t - off_m))
        elif off_m <= s < off_m + N_mirna and t < N_circ:
            circ_to_mirnas[int(t)].add(int(s - off_m))
        elif off_d <= s < off_d + N_disease and off_m <= t < off_m + N_mirna:
            dis_to_mirnas[int(s - off_d)].add(int(t - off_m))
        elif off_m <= s < off_m + N_mirna and off_d <= t < off_d + N_disease:
            dis_to_mirnas[int(t - off_d)].add(int(s - off_m))

    # ── CMC edges: circRNA↔circRNA via shared miRNAs (Jaccard + topK) ──
    cmc_src, cmc_tgt = [], []
    for c1 in range(N_circ):
        m1 = circ_to_mirnas[c1]
        if not m1:
            continue
        scores = []
        for c2 in range(N_circ):
            if c1 == c2:
                continue
            m2 = circ_to_mirnas[c2]
            if not m2:
                continue
            shared = len(m1 & m2)
            if shared < min_shared:
                continue
            jaccard = shared / max(len(m1 | m2), 1)
            scores.append((c2, jaccard))
        scores.sort(key=lambda x: x[1], reverse=True)
        for c2, _ in scores[:topk]:
            cmc_src.append(c1)
            cmc_tgt.append(c2)

    # ── DMD edges: disease↔disease via shared miRNAs (Jaccard + topK) ──
    dmd_src, dmd_tgt = [], []
    for d1 in range(N_disease):
        m1 = dis_to_mirnas[d1]
        if not m1:
            continue
        scores = []
        for d2 in range(N_disease):
            if d1 == d2:
                continue
            m2 = dis_to_mirnas[d2]
            if not m2:
                continue
            shared = len(m1 & m2)
            if shared < min_shared:
                continue
            jaccard = shared / max(len(m1 | m2), 1)
            scores.append((d2, jaccard))
        scores.sort(key=lambda x: x[1], reverse=True)
        for d2, _ in scores[:topk]:
            dmd_src.append(d1 + off_d)
            dmd_tgt.append(d2 + off_d)

    cmc_ei = torch.tensor([cmc_src, cmc_tgt], dtype=torch.long) if cmc_src else \
        torch.zeros((2, 0), dtype=torch.long)
    dmd_ei = torch.tensor([dmd_src, dmd_tgt], dtype=torch.long) if dmd_src else \
        torch.zeros((2, 0), dtype=torch.long)

    new_edges = {3: cmc_ei, 4: dmd_ei}
    return new_edges, circ_to_mirnas, dis_to_mirnas


# =============================================================================
# Ablation: Remove Node Type from Heterogeneous Graph
# =============================================================================

def _filter_edge_index(ei: torch.Tensor, remove_start: int, remove_end: int,
                       N_remove: int) -> torch.Tensor:
    """Filter edges involving removed nodes and remap remaining indices.

    Args:
        ei: (2, E) edge_index tensor
        remove_start: first index of removed node block
        remove_end: past-the-end index of removed node block
        N_remove: number of removed nodes (remove_end - remove_start)

    Returns:
        (2, E') filtered and remapped edge_index
    """
    if ei is None or ei.shape[1] == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    src = ei[0]
    tgt = ei[1]
    src_bad = (src >= remove_start) & (src < remove_end)
    tgt_bad = (tgt >= remove_start) & (tgt < remove_end)
    keep = ~(src_bad | tgt_bad)

    if not keep.any():
        return torch.zeros((2, 0), dtype=torch.long)

    src_f = src[keep].clone()
    tgt_f = tgt[keep].clone()
    src_f[src_f >= remove_end] -= N_remove
    tgt_f[tgt_f >= remove_end] -= N_remove
    return torch.stack([src_f, tgt_f], dim=0)


def filter_node_type_from_graph(
    x_all: torch.Tensor,
    edge_index: torch.Tensor,
    edge_index_by_type: dict,
    node_counts: dict,
    remove_type: str,
) -> tuple:
    """Remove a node type from the heterogeneous graph and remap all indices.

    Node layout: [0, N_circ) circ, [N_circ, N_c+N_d) disease,
                 [N_c+N_d, N_c+N_d+N_m) miRNA, rest lncRNA.

    Also filters positional encoding tensors (pe_lap, pe_rw) if present
    in node_counts, so the model receives properly aligned PE.

    Args:
        x_all: (N_total, F) concatenated features
        edge_index: (2, E) combined edges
        edge_index_by_type: {type_id: (2, E_type)} or None
        node_counts: dict with N_circ, N_disease, N_mirna, N_lncrna
                     (and optionally pe_lap, pe_rw tensors)
        remove_type: 'mirna' or 'lncrna'

    Returns:
        (x_all_new, edge_index_new, edge_index_by_type_new, node_counts_new)
    """
    N_c = node_counts['N_circ']
    N_d = node_counts['N_disease']
    N_m = node_counts['N_mirna']
    N_l = node_counts['N_lncrna']

    if remove_type == 'mirna':
        remove_start = N_c + N_d
        remove_end = N_c + N_d + N_m
        node_counts_new = {**node_counts, 'N_mirna': 0}
    elif remove_type == 'lncrna':
        remove_start = N_c + N_d + N_m
        remove_end = N_c + N_d + N_m + N_l
        node_counts_new = {**node_counts, 'N_lncrna': 0}
    else:
        raise ValueError(f"Unknown remove_type: {remove_type}")

    N_remove = remove_end - remove_start
    if N_remove == 0:
        return x_all, edge_index, edge_index_by_type, node_counts

    # Filter x_all
    N_total = x_all.shape[0]
    keep_mask = torch.ones(N_total, dtype=torch.bool)
    keep_mask[remove_start:remove_end] = False
    x_all_new = x_all[keep_mask]

    # Filter positional encodings in node_counts (if present)
    for pe_key in ('pe_lap', 'pe_rw'):
        pe_tensor = node_counts.get(pe_key)
        if pe_tensor is not None and isinstance(pe_tensor, torch.Tensor):
            if pe_tensor.shape[0] == N_total:
                node_counts_new[pe_key] = pe_tensor[keep_mask]

    # Filter combined edge_index
    edge_index_new = _filter_edge_index(edge_index, remove_start, remove_end, N_remove)

    # Filter per-type edges
    edge_index_by_type_new = None
    if edge_index_by_type is not None:
        edge_index_by_type_new = {}
        for etype, ei in edge_index_by_type.items():
            # For no_mirna: CMC (type 3) and DMD (type 4) depend on miRNAs → drop entirely
            if remove_type == 'mirna' and etype in (3, 4):
                continue
            ei_f = _filter_edge_index(ei, remove_start, remove_end, N_remove)
            edge_index_by_type_new[etype] = ei_f

    return x_all_new, edge_index_new, edge_index_by_type_new, node_counts_new


def _build_hard_negative_pool(circ_to_mirnas: list,
                              mirna_to_diseases: list,
                              N_circ: int,
                              exclude_set: set) -> np.ndarray:
    """Build pool of topology-guided hard negative pairs.

    A hard negative = (circRNA, disease) that shares >=1 miRNA neighbor
    (2-hop path via miRNA) but is NOT a known positive association.

    Args:
        circ_to_mirnas: per-circRNA set of local miRNA indices
        mirna_to_diseases: per-miRNA set of local disease indices
        N_circ: number of circRNA nodes
        exclude_set: set of (c, d) tuples to exclude (positives + val + test)

    Returns:
        hard_pool: (N_hard, 2) int64 array of [circ_idx, dis_idx]
    """
    hard_pairs = []
    for c in range(N_circ):
        c_mirnas = circ_to_mirnas[c]
        if not c_mirnas:
            continue
        # All diseases reachable within 2 hops via shared miRNAs
        candidate_dis = set()
        for m in c_mirnas:
            candidate_dis.update(mirna_to_diseases[m])
        for d in candidate_dis:
            if (c, d) not in exclude_set:
                hard_pairs.append([c, d])

    if not hard_pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(hard_pairs, dtype=np.int64)


def sample_topo_negatives(hard_pool: np.ndarray,
                          n_hard: int,
                          n_random: int,
                          train_mask: np.ndarray,
                          exclude_set: set,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Sample negatives mixing topology-guided (hard) and random.

    Hard negatives are sampled from the precomputed pool (pairs sharing
    miRNA neighbours). Random negatives fill the remainder to maintain
    coverage of the full negative distribution.

    Args:
        hard_pool: (N_hard, 2) precomputed hard negative pairs
        n_hard: number of hard negatives to sample
        n_random: number of random negatives to sample
        train_mask, exclude_set: standard sampling params

    Returns:
        neg_circ: (n_total,) circRNA indices
        neg_dis: (n_total,) disease indices
    """
    neg_pairs = []

    # ── Hard negatives: topology-guided ──
    if n_hard > 0 and hard_pool.shape[0] > 0:
        if hard_pool.shape[0] <= n_hard:
            idx = np.arange(hard_pool.shape[0])
        else:
            idx = np.random.choice(hard_pool.shape[0], n_hard, replace=False)
        neg_pairs.append(hard_pool[idx])

    # ── Random negatives: exploration ──
    if n_random > 0:
        rand_circ, rand_dis = sample_negative_pairs(
            train_mask, n_samples=n_random, exclude_pairs=exclude_set)
        neg_pairs.append(np.stack([rand_circ, rand_dis], axis=1))

    if not neg_pairs:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    all_pairs = np.concatenate(neg_pairs, axis=0)
    # Shuffle to interleave hard + random within minibatches
    perm = np.random.permutation(all_pairs.shape[0])
    all_pairs = all_pairs[perm]
    return all_pairs[:, 0].astype(np.int64), all_pairs[:, 1].astype(np.int64)


def train_one_fold_hetero(
    model: MRSECDAHetero,
    x_circ: torch.Tensor, x_disease: torch.Tensor,
    x_mirna: torch.Tensor, x_lncrna: torch.Tensor,
    edge_index: torch.Tensor, node_counts: Dict[str, int],
    fold_data: dict, train_mask: np.ndarray,
    config: Config, device: str = "cpu", verbose: bool = True,
    edge_index_by_type: dict = None,
    struct_feat_computer: StructuralFeatureComputer = None,
    llm_circ: torch.Tensor = None,
    llm_disease: torch.Tensor = None,
    llm_mirna: torch.Tensor = None,
    msne_feat_all: torch.Tensor = None,
    fixed_threshold: float = None,
    save_score_path: str = None,
) -> Tuple[MRSECDAHetero, Dict[str, float], Dict[str, float]]:
    model = model.to(device)
    model.train()

    train_circ = fold_data["train_circ"]
    train_dis = fold_data["train_dis"]
    val_circ = fold_data["val_circ"]
    val_dis = fold_data["val_dis"]
    val_neg_circ = fold_data["val_neg_circ"]
    val_neg_dis = fold_data["val_neg_dis"]
    test_pos_circ = fold_data["test_pos_circ"]
    test_pos_dis = fold_data["test_pos_dis"]
    test_neg_circ = fold_data["test_neg_circ"]
    test_neg_dis = fold_data["test_neg_dis"]
    n_train = len(train_circ)

    # Build exclude set
    exclude_set = set()
    for i, j in zip(train_circ, train_dis):
        exclude_set.add((i, j))
    for i, j in zip(val_circ, val_dis):
        exclude_set.add((i, j))
    for i, j in zip(test_pos_circ, test_pos_dis):
        exclude_set.add((i, j))

    val_all_c = np.concatenate([val_circ, val_neg_circ])
    val_all_d = np.concatenate([val_dis, val_neg_dis])
    val_all_l = np.concatenate([np.ones(len(val_circ)), np.zeros(len(val_neg_circ))])

    test_all_c = np.concatenate([test_pos_circ, test_neg_circ])
    test_all_d = np.concatenate([test_pos_dis, test_neg_dis])
    test_all_l = np.concatenate([np.ones(len(test_pos_circ)), np.zeros(len(test_neg_circ))])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.finetune_lr,
        weight_decay=config.train.finetune_weight_decay,
    )
    # Support configurable pos_weight from extra_args
    _pw = getattr(config.train, 'pos_weight', None)
    if _pw is not None and _pw > 0:
        pw_val = _pw
    else:
        pw_val = max(1.0, config.data.neg_ratio)
    pos_weight = torch.tensor([pw_val]).to(device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # LR scheduler
    _use_scheduler = getattr(config.train, 'use_lr_scheduler', False)
    if _use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.train.finetune_epochs, eta_min=1e-6)
    else:
        scheduler = None

    # LR warmup: linear ramp from lr/10 to target_lr over warmup_epochs
    _warmup_epochs = getattr(config.train, 'lr_warmup_epochs', 0)
    _base_lr = config.train.finetune_lr

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    x_circ_d = x_circ.to(device)
    x_disease_d = x_disease.to(device)
    x_mirna_d = x_mirna.to(device)
    x_lncrna_d = x_lncrna.to(device)
    edge_index_d = edge_index.to(device)

    # ── v3.2: DropEdge Annealing ──
    _do_anneal = getattr(config.model, 'edge_dropout_anneal', False)
    _init_edge_drop = getattr(config.model, 'edge_dropout', 0.0)

    # ── v3.2: Label Smoothing & Focal Loss ──
    _label_smooth = getattr(config.model, 'label_smoothing', 0.0)
    _focal_gamma = getattr(config.model, 'focal_gamma', 0.0)
    # ── v3.3: Ranking-Aware Focal Loss ──
    _bpr_weight = getattr(config.model, 'bpr_weight', 0.0)
    _bpr_k = getattr(config.model, 'bpr_k', 5)

    # ── v3.3: Topology-Guided Negative Sampling (TGNS) ──
    # Build miRNA adjacency + hard negative pool once per fold.
    # Hard negatives = (circRNA, disease) sharing >=1 miRNA neighbour
    # but not a known positive. These are biologically informative negatives
    # because circRNA→miRNA→disease is the molecular sponge mechanism.
    _topo_neg = getattr(config.model, 'topo_neg_ratio', 0.0)
    if _topo_neg > 0 and edge_index_by_type is not None:
        N_circ_topo = node_counts['N_circ']
        N_d_topo = node_counts['N_disease']
        N_m_topo = node_counts.get('N_mirna', 0)
        if N_m_topo > 0:
            circ_to_mir, mirna_to_dis = _build_mirna_adjacency(
                edge_index_by_type, N_circ_topo, N_d_topo, N_m_topo)
            if any(len(s) > 0 for s in circ_to_mir):
                hard_neg_pool = _build_hard_negative_pool(
                    circ_to_mir, mirna_to_dis, N_circ_topo, exclude_set)
                if verbose:
                    print(f"  [TGNS] Hard negative pool: {hard_neg_pool.shape[0]} pairs "
                          f"(miRNA-sharing, unassociated)")
            else:
                hard_neg_pool = None
                if verbose:
                    print(f"  [TGNS] No miRNA adj found — falling back to random negatives")
        else:
            hard_neg_pool = None
    else:
        hard_neg_pool = None

    for epoch in range(config.train.finetune_epochs):
        model.train()

        # DropEdge annealing: decay from initial rate → 0 over training
        # Curriculum regularization — strong early (learn robust features),
        # weak late (fine-tune with full graph structure)
        if _do_anneal and _init_edge_drop > 0:
            progress = epoch / max(1, config.train.finetune_epochs - 1)
            current_drop = _init_edge_drop * (1.0 - progress)
            # Apply to HGT layers (hasattr check for backward compat)
            if hasattr(model, 'gcn') and hasattr(model.gcn, 'set_edge_dropout'):
                model.gcn.set_edge_dropout(current_drop)

        n_neg = n_train * config.data.neg_ratio
        # ── v3.3: Topology-Guided Negative Sampling ──
        # Mix hard negatives (miRNA-sharing unassociated pairs) with random
        # negatives. Hard negatives force the encoder to learn fine-grained
        # distinctions; random negatives maintain distribution coverage.
        if _topo_neg > 0 and hard_neg_pool is not None and hard_neg_pool.shape[0] > 0:
            n_hard = int(n_neg * _topo_neg)
            n_random = n_neg - n_hard
            neg_circ, neg_dis = sample_topo_negatives(
                hard_neg_pool, n_hard, n_random, train_mask, exclude_set)
        else:
            neg_circ, neg_dis = sample_negative_pairs(
                train_mask, n_samples=n_neg, exclude_pairs=exclude_set)

        batches = prepare_batches(
            train_circ, train_dis, neg_circ, neg_dis,
            batch_size=config.train.batch_size, shuffle=True)

        # Standard single-view encoding (v2.3 / v2.7 with optional LLM)
        # v3.1/v3.2: optionally request tokens + miRNA embeddings for auxiliary tasks
        cont_weight = getattr(config.model, 'contrastive_weight', 0.0)
        aux_weight = getattr(config.model, 'aux_task_weight', 0.0)
        deep_sup_weight = getattr(config.model, 'deep_sup_weight', 0.0)
        need_tokens = (cont_weight > 0) or (aux_weight > 0) or (deep_sup_weight > 0)

        if need_tokens:
            # NOTE: Deprecated experimental path (v3.1/v3.2 aux tasks, all off by default).
            # v3.3 MSNE architecture doesn't expose per-token decompositions.
            h_circ, h_disease, h_mirna_all = model.encode(
                x_circ_d, x_disease_d, x_mirna_d, x_lncrna_d,
                edge_index_d, node_counts, edge_index_by_type=edge_index_by_type,
                llm_circ=llm_circ, llm_disease=llm_disease, llm_mirna=llm_mirna,
                msne_feat=None, return_tokens=True)

            # ── Token-wise Deep Supervision (v3.2) ──
            # Extract per-token circ/disease embeddings once per epoch.
            # Each token is forced to be independently predictive via the shared
            # predictor during minibatch processing — zero extra parameters.
            if deep_sup_weight > 0:
                N_c_ds = node_counts['N_circ']
                N_d_ds = node_counts['N_disease']
                # Token₀: Self (projected features, no graph)
                h_circ_t0 = token0[:N_c_ds]
                h_dis_t0 = token0[N_c_ds:N_c_ds + N_d_ds]
                # Token₁: 1-hop HGT (local graph structure)
                h_circ_t1 = token1[:N_c_ds]
                h_dis_t1 = token1[N_c_ds:N_c_ds + N_d_ds]
                # Token₃: Cross-Type Context (global semantics)
                h_circ_t3 = token3[:N_c_ds]
                h_dis_t3 = token3[N_c_ds:N_c_ds + N_d_ds]
            else:
                h_circ_t0 = h_dis_t0 = h_circ_t1 = h_dis_t1 = None
                h_circ_t3 = h_dis_t3 = None

            # ── Contrastive loss (v3.1) ──
            if cont_weight > 0:
                N_cd = node_counts['N_circ'] + node_counts['N_disease']
                cont_loss = multi_view_contrastive_loss(
                    token0[:N_cd], token1[:N_cd], temperature=0.07)
            else:
                cont_loss = None

            # ── Auxiliary multi-task loss (v3.1) ──
            # Reconstruct miRNA-disease edges using dot-product similarity
            # Provides additional training signal for the shared encoder
            if aux_weight > 0 and 'mirna_dis_adj' in node_counts:
                md_adj = node_counts['mirna_dis_adj']  # (N_m, N_d) torch tensor
                md_adj = md_adj.to(device=device)
                # Predict all miRNA-disease scores
                md_pred = h_mirna_all @ h_disease.T  # (N_m, N_d)
                # BCE loss: push known edges to 1, unknown to 0
                aux_md_loss = F.binary_cross_entropy_with_logits(
                    md_pred, md_adj, reduction='mean')
            else:
                aux_md_loss = None
        else:
            h_circ, h_disease = model.encode(
                x_circ_d, x_disease_d, x_mirna_d, x_lncrna_d,
                edge_index_d, node_counts, edge_index_by_type=edge_index_by_type,
                llm_circ=llm_circ, llm_disease=llm_disease, llm_mirna=llm_mirna,
                msne_feat=None)
            cont_loss = None
            aux_md_loss = None
            h_circ_t0 = h_dis_t0 = h_circ_t1 = h_dis_t1 = None
            h_circ_t3 = h_dis_t3 = None
            h_mirna_all = None

        # ── v3.1 Consistency Regularization: second encoding pass ──
        # Different DropEdge mask produces different node representations.
        # MSE between predictions enforces DropEdge-invariant predictions.
        cons_weight = getattr(config.model, 'consistency_weight', 0.0)
        if cons_weight > 0:
            h_circ2, h_disease2 = model.encode(
                x_circ_d, x_disease_d, x_mirna_d, x_lncrna_d,
                edge_index_d, node_counts, edge_index_by_type=edge_index_by_type,
                llm_circ=llm_circ, llm_disease=llm_disease, llm_mirna=llm_mirna,
                msne_feat=None)
        else:
            h_circ2, h_disease2 = None, None

        n_batches = len(batches)
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (circ_idx, dis_idx, labels) in enumerate(batches):
            circ_idx = circ_idx.to(device)
            dis_idx = dis_idx.to(device)
            labels = labels.to(device)

            sf = None
            if struct_feat_computer is not None:
                sf_np = struct_feat_computer.compute_features(
                    circ_idx.cpu().numpy(), dis_idx.cpu().numpy())
                sf = torch.tensor(sf_np, dtype=torch.float32, device=device)

            logits, _ = model.predictor(h_circ, h_disease, circ_idx, dis_idx,
                                        struct_feat=sf)
            smooth_lbls = _smooth_labels(labels, _label_smooth)
            # ── v3.3: Ranking-Aware Focal Loss ──
            # RA-Focal adds a pairwise BPR ranking term to Focal Loss.
            # BPR directly optimizes AUC (ranking) by ensuring each positive
            # scores higher than negatives. Zero extra model parameters.
            if _bpr_weight > 0:
                _gamma_use = _focal_gamma if _focal_gamma > 0 else 0.0
                loss = ranking_aware_focal_loss(
                    logits.squeeze(-1), smooth_lbls,
                    gamma=_gamma_use, pos_weight=pos_weight,
                    bpr_weight=_bpr_weight, bpr_k=_bpr_k)
            elif _focal_gamma > 0:
                loss = focal_bce_loss(logits.squeeze(-1), smooth_lbls,
                                     gamma=_focal_gamma, pos_weight=pos_weight)
            else:
                loss = bce_loss(logits.squeeze(-1), smooth_lbls)
            is_last = (batch_idx == n_batches - 1)

            # ── v3.1 Loss composition: BCE + Aux MTL + Contrastive + Consistency ──
            batch_loss = loss
            # ── v3.2: Token-wise Deep Supervision ──
            # Each resolution token independently predicts the same pairs
            # using the shared predictor. Forces all scales to be informative.
            if deep_sup_weight > 0:
                logits_t0, _ = model.predictor(
                    h_circ_t0, h_dis_t0, circ_idx, dis_idx, struct_feat=sf)
                logits_t1, _ = model.predictor(
                    h_circ_t1, h_dis_t1, circ_idx, dis_idx, struct_feat=sf)
                logits_t3, _ = model.predictor(
                    h_circ_t3, h_dis_t3, circ_idx, dis_idx, struct_feat=sf)
                smooth_lbls = _smooth_labels(labels, _label_smooth)
                ds_loss = (bce_loss(logits_t0.squeeze(-1), smooth_lbls) +
                           bce_loss(logits_t1.squeeze(-1), smooth_lbls) +
                           bce_loss(logits_t3.squeeze(-1), smooth_lbls)) / 3.0
                batch_loss = batch_loss + deep_sup_weight * ds_loss
            if aux_md_loss is not None:
                batch_loss = batch_loss + aux_weight * aux_md_loss
            if cont_loss is not None:
                batch_loss = batch_loss + cont_weight * cont_loss
            if cons_weight > 0 and h_circ2 is not None:
                logits2, _ = model.predictor(h_circ2, h_disease2, circ_idx, dis_idx,
                                            struct_feat=sf)
                cons_loss_val = F.mse_loss(
                    torch.sigmoid(logits.squeeze(-1)),
                    torch.sigmoid(logits2.squeeze(-1)))
                batch_loss = batch_loss + cons_weight * cons_loss_val
            batch_loss = batch_loss / n_batches

            batch_loss.backward(retain_graph=not is_last)
            epoch_loss += loss.item()

        if config.train.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        # LR warmup: linear ramp from base_lr/10 to base_lr over warmup_epochs
        if _warmup_epochs > 0 and epoch < _warmup_epochs:
            warmup_factor = (epoch + 1) / _warmup_epochs  # 0→1
            warmup_lr = _base_lr * (0.1 + 0.9 * warmup_factor)  # lr/10 → lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        epoch_loss /= n_batches

        if epoch % config.train.log_interval == 0 or epoch == config.train.finetune_epochs - 1:
            val_metrics = evaluate_model_hetero(
                model, x_circ, x_disease, x_mirna, x_lncrna,
                edge_index, node_counts,
                val_all_c, val_all_d, val_all_l,
                batch_size=config.train.batch_size, device=device,
                edge_index_by_type=edge_index_by_type,
                struct_feat_computer=struct_feat_computer,
                llm_circ=llm_circ, llm_disease=llm_disease, llm_mirna=llm_mirna,
                msne_feat_all=None)

            val_auc = val_metrics["auc"]
            best_str = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                best_str = " [BEST]"
            else:
                patience_counter += 1

            if verbose:
                print(f"  Epoch {epoch:4d} | Loss: {epoch_loss:.6f} | "
                      f"Val AUC: {val_auc:.4f} | Val AUPR: {val_metrics['aupr']:.4f}{best_str}")

        if patience_counter >= config.train.finetune_patience:
            if verbose:
                print(f"  Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Per-fold optimal threshold: find on val, apply to test ──
    if fixed_threshold is not None:
        best_threshold = fixed_threshold
    else:
        val_eval = evaluate_model_hetero(
            model, x_circ, x_disease, x_mirna, x_lncrna,
            edge_index, node_counts,
            val_all_c, val_all_d, val_all_l,
            batch_size=config.train.batch_size, device=device,
            edge_index_by_type=edge_index_by_type,
            struct_feat_computer=struct_feat_computer,
            llm_circ=llm_circ, llm_disease=llm_disease, llm_mirna=llm_mirna,
            msne_feat_all=None,
            threshold=0.5, return_scores=True)
        val_scores = val_eval.pop("_scores")
        best_threshold = find_best_threshold(val_scores, val_all_l)

    test_metrics = evaluate_model_hetero(
        model, x_circ, x_disease, x_mirna, x_lncrna,
        edge_index, node_counts,
        test_all_c, test_all_d, test_all_l,
        batch_size=config.train.batch_size * 4, device=device,
        edge_index_by_type=edge_index_by_type,
        struct_feat_computer=struct_feat_computer,
        llm_circ=llm_circ, llm_disease=llm_disease, llm_mirna=llm_mirna,
        msne_feat_all=None,
        threshold=best_threshold)

    if verbose:
        print(f"  Test AUC: {test_metrics['auc']:.4f} | Test AUPR: {test_metrics['aupr']:.4f} "
              f"| F1: {test_metrics['f1']:.4f} (thr={best_threshold:.2f})")

    # ── recall@k: full-matrix scoring against this fold's test positives ──
    try:
        _t_inf = time.time()
        S = score_full_matrix(
            model, x_circ, x_disease, x_mirna, x_lncrna,
            edge_index, node_counts,
            edge_index_by_type=edge_index_by_type,
            struct_feat_computer=struct_feat_computer,
            llm_circ=llm_circ, llm_disease=llm_disease, llm_mirna=llm_mirna,
            device=device)
        test_metrics["infer_time_s"] = float(time.time() - _t_inf)
        rak = compute_recall_at_k(S, test_pos_circ, test_pos_dis)
        test_metrics["recall_at_k"] = {str(k): float(v) for k, v in rak.items()}
        if save_score_path:
            os.makedirs(os.path.dirname(save_score_path), exist_ok=True)
            np.save(save_score_path, S)
        if verbose:
            print(f"  [INFER_TIME] full-matrix scoring: {test_metrics['infer_time_s']:.4f}s")
            print(f"  recall@10={rak[10]:.4f} recall@20={rak[20]:.4f} recall@30={rak[30]:.4f} "
                  f"recall@40={rak[40]:.4f} recall@50={rak[50]:.4f}")
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"  [recall@k] skipped: {e}")

    val_metrics = {"auc": best_val_auc, "best_threshold": best_threshold}
    return model, val_metrics, test_metrics


# =============================================================================
# Run one dataset (5-fold CV)
# =============================================================================

def run_dataset_hetero(
    dataset_name: str, config: Config, device: str = "cpu",
    ablation: str = None, seed: int = 42,
    folds_to_run: list = None, verbose: bool = True,
    fixed_threshold: float = None,
    drop_struct_feat: list = None,
    drop_struct_view: list = None,
    save_score_dir: str = None,
) -> Dict[str, list]:
    if verbose:
        print(f"\n{'='*60}")
        print(f"  MRSECDA v2 (Hetero) on {dataset_name}")
        if ablation:
            print(f"  Ablation: {ablation}")
        print(f"{'='*60}")

    dataset = CDADataset(
        name=dataset_name,
        data_dir=os.path.join(ROOT, "data", "CompareExperiment", "Data"),
        seed=seed, use_external_knowledge=True, use_kmer_features=True,
    )
    adj = dataset.adj
    folds = kfold_split(adj, n_folds=5, seed=seed)

    fold_results = []
    for fold_idx, (train_mask, test_mask) in enumerate(folds):
        if folds_to_run is not None and fold_idx not in folds_to_run:
            continue

        if verbose:
            print(f"\n  --- Fold {fold_idx + 1}/5 ---")

        np.random.seed(seed + fold_idx)
        torch.manual_seed(seed + fold_idx * 100)

        train_inner, val_mask = train_val_split(train_mask, val_ratio=0.2)
        train_mask_full = np.clip(train_inner + val_mask, 0, 1)

        # ── Build HETEROGENEOUS graph (★ key difference from v1) ──
        circ_mirna_h5_dir = os.path.join(ROOT, "data", "CompareExperiment",
                                         "Data", "circRNA_miRNA")
        graph_data, x_all, node_counts = \
            build_hetero_graph(
                dataset, train_mask_full,
                khgat_data_dir=os.path.join(ROOT, "data", "CompareExperiment",
                                            "AE-RF main", "KHGAT", "data"),
                circ_mirna_h5_dir=circ_mirna_h5_dir,
            )

        N_c = node_counts['N_circ']
        N_d = node_counts['N_disease']
        N_m = node_counts['N_mirna']
        N_l = node_counts['N_lncrna']
        D_feat = node_counts['feat_dim']

        # Extract per-type features from x_all
        x_circ = x_all[:N_c]
        x_disease = x_all[N_c:N_c + N_d]
        x_mirna = x_all[N_c + N_d:N_c + N_d + N_m]
        x_lncrna = x_all[N_c + N_d + N_m:]

        edge_index = graph_data.edge_index
        edge_index_by_type = getattr(graph_data, 'edge_index_by_type', None)

        # Inject positional encodings into node_counts EARLY (before ablation filter)
        # so they get properly filtered if node types are removed.
        pe_lap = getattr(graph_data, 'lap_pe', None)
        pe_rw = getattr(graph_data, 'rwse', None)
        if pe_lap is not None and pe_rw is not None:
            node_counts['pe_lap'] = pe_lap
            node_counts['pe_rw'] = pe_rw

        # ── Node type removal (ablation or MRSECDA default) ──
        _use_lncrna = getattr(config.model, 'use_lncrna', True)
        _rm_type = None
        if ablation == "no_mirna":
            _rm_type = "mirna"
        elif ablation == "no_lncrna" or not _use_lncrna:
            _rm_type = "lncrna"
        if _rm_type is not None:
            _n_removed = N_m if _rm_type == "mirna" else N_l
            x_all, edge_index, edge_index_by_type, node_counts = \
                filter_node_type_from_graph(
                    x_all, edge_index, edge_index_by_type, node_counts, _rm_type)
            N_c = node_counts['N_circ']
            N_d = node_counts['N_disease']
            N_m = node_counts['N_mirna']
            N_l = node_counts['N_lncrna']
            # Re-extract per-type features after filtering
            x_circ = x_all[:N_c]
            x_disease = x_all[N_c:N_c + N_d]
            x_mirna = x_all[N_c + N_d:N_c + N_d + N_m]
            x_lncrna = x_all[N_c + N_d + N_m:]
            if _rm_type == "mirna":
                # Keep use_metapath_edges=True so CRT multi-resolution works.
                # CMC/DMD edges will naturally be empty (no miRNAs to form paths).
                if verbose:
                    print(f"  Removed {_n_removed} miRNA nodes, "
                          f"{N_c}+{N_d} nodes remain")
            elif verbose:
                print(f"  Removed {_n_removed} lncRNA nodes, "
                      f"{N_c}+{N_d}+{N_m} nodes remain")

        # ── v3.4: Meta-Path HGT Edge Types ──
        # Build CMC (circRNA↔circRNA) and DMD (disease↔disease) edges via shared
        # miRNAs. These become new HGT edge types (3, 4) with independent QKV
        # projections, allowing the model to learn how to weight 2-hop topology.
        _use_metapath_edges = getattr(config.model, 'use_metapath_edges', False)
        if _use_metapath_edges and edge_index_by_type is not None:
            _mp_topk = getattr(config.model, 'metapath_edge_topk', 15)
            _mp_min = getattr(config.model, 'metapath_edge_min_shared', 1)
            mp_new_edges, _, _ = build_metapath_hgt_edges(
                edge_index_by_type, N_c, N_d, N_m,
                min_shared=_mp_min, topk=_mp_topk)
            # Merge into edge_index_by_type (mutate a copy to avoid side effects)
            edge_index_by_type = dict(edge_index_by_type)
            edge_index_by_type.update(mp_new_edges)
            _num_edge_types = 5
            n_cmc = mp_new_edges.get(3, torch.zeros(2, 0)).shape[1]
            n_dmd = mp_new_edges.get(4, torch.zeros(2, 0)).shape[1]
            if verbose:
                print(f"  Metapath Edges: CMC={n_cmc}, DMD={n_dmd} (types 3,4; total edge types=5)")
        else:
            _num_edge_types = 3

        # ── MRSECDA default: 2-feature PSE (jaccard+AA) unless explicitly overridden ──
        if drop_struct_feat is None:
            _keep = getattr(config.model, 'struct_feat_names', None)
            if _keep:
                _all_feats = ["jaccard", "adamic_adar", "resource_alloc", "inv_distance"]
                drop_struct_feat = [f for f in _all_feats if f not in _keep]

        # ── Build structural feature computer (PSE Phase 2a: Multi-View) ──
        _use_struct = getattr(config.model, 'use_struct_feat', False)
        _multi_view = getattr(config.model, 'use_multi_view_struct', True)  # default multi-view
        if _use_struct and edge_index_by_type is not None and _multi_view:
            struct_feat_computer = MultiViewStructuralFeatureComputer(
                edge_index_by_type=edge_index_by_type,
                N_total=N_c + N_d + N_m + N_l,
                N_circ=N_c, N_disease=N_d,
                node_counts=node_counts,
                max_dist=5,
                drop_feature=drop_struct_feat,
                drop_view=drop_struct_view,
            )
            struct_feat_dim = struct_feat_computer.feat_dim  # 12
            if verbose:
                _nv = getattr(struct_feat_computer, 'num_views', 3)
                _nf = struct_feat_dim // _nv if _nv else 0
                print(f"  StructFeat: Multi-View ({_nv} views × {_nf} features = {struct_feat_dim}-dim)")
            # ── v3.3: Meta-Path Structural Features (miRNA-mediated view) ──
            # Adds a 4th view capturing circRNA→miRNA→disease sponge mechanism.
            # Augments PSE from 12-dim → 16-dim. Zero extra model parameters.
            _use_metapath = getattr(config.model, 'use_metapath_feat', False)
            if _use_metapath and N_m > 0:
                circ_to_mir_mp, mirna_to_dis_mp = _build_mirna_adjacency(
                    edge_index_by_type, N_c, N_d, N_m)
                # Compute miRNA degrees (total connections to circRNA + disease)
                mirna_degrees_mp = {}
                for m_idx in range(N_m):
                    deg = len(mirna_to_dis_mp[m_idx])
                    # Also count circRNA connections to this miRNA
                    for c_set in circ_to_mir_mp:
                        if m_idx in c_set:
                            deg += 1
                    mirna_degrees_mp[m_idx] = max(deg, 1)
                # Also build disease→miRNA mapping (reverse of miRNA→disease)
                dis_to_mir_mp = [set() for _ in range(N_d)]
                for m_idx, d_set in enumerate(mirna_to_dis_mp):
                    for d in d_set:
                        dis_to_mir_mp[d].add(m_idx)
                struct_feat_computer = CombinedStructFeatComputer(
                    struct_feat_computer, circ_to_mir_mp, dis_to_mir_mp,
                    mirna_degrees_mp)
                struct_feat_dim = struct_feat_computer.feat_dim  # 16
                if verbose:
                    _n_shared = sum(1 for c_s in circ_to_mir_mp if c_s)
                    print(f"  MetaPathFeat: 4-dim miRNA-mediated view enabled "
                          f"({_n_shared}/{N_c} circRNAs have miRNAs, "
                          f"total {struct_feat_dim}-dim struct features)")
        elif _use_struct:
            # Fallback: single-view on full adjacency
            struct_feat_computer = StructuralFeatureComputer(
                edge_index=edge_index.numpy(),
                N_total=N_c + N_d + N_m + N_l,
                N_circ=N_c, N_disease=N_d,
                node_counts=node_counts,
                max_dist=5,
                drop_feature=drop_struct_feat,
            )
            struct_feat_dim = struct_feat_computer.feat_dim  # 4
            if verbose:
                print(f"  StructFeat: Single-View (4-dim)")
        else:
            struct_feat_computer = None
            struct_feat_dim = 0

        # ── v3.3: Meta-Path Neighbor Encoding (MPNE) ──
        # Precomputes per-node meta-path degree features (3-dim):
        #   [0] CMC-degree: # circRNA↔miRNA↔circRNA paths through this node
        #   [1] DMD-degree: # disease↔miRNA↔disease paths through this node
        #   [2] CMD-degree: # circRNA↔miRNA↔disease paths through this node
        # Concatenated to raw input features → TypeSpecificProjection handles them.
        # Zero extra model parameters — purely structural statistics.
        _use_metapath = getattr(config.model, 'use_metapath', True)
        _metapath_mode = getattr(config.model, 'metapath_mode', 'concat')
        mp_feat_all = None  # (N_total, 3) tensor, concatenated to raw input
        if _use_metapath and edge_index_by_type is not None:
            from data.metapath_encoder import compute_metapath_degrees
            mp_deg_np = compute_metapath_degrees(
                edge_index_by_type, N_c, N_d, N_m, N_l)
            mp_feat_all = torch.from_numpy(mp_deg_np).float()
            # Concat to raw per-type features (3 extra dims per node type)
            # Need to handle both torch tensor and numpy array cases
            x_circ_np = x_circ.cpu().numpy() if isinstance(x_circ, torch.Tensor) else x_circ
            x_disease_np = x_disease.cpu().numpy() if isinstance(x_disease, torch.Tensor) else x_disease
            x_mirna_np = x_mirna.cpu().numpy() if isinstance(x_mirna, torch.Tensor) else x_mirna
            x_lncrna_np = x_lncrna.cpu().numpy() if isinstance(x_lncrna, torch.Tensor) else x_lncrna
            x_circ = torch.from_numpy(np.concatenate([x_circ_np, mp_deg_np[:N_c]], axis=1)).float()
            x_disease = torch.from_numpy(np.concatenate([x_disease_np, mp_deg_np[N_c:N_c+N_d]], axis=1)).float()
            x_mirna = torch.from_numpy(np.concatenate([x_mirna_np, mp_deg_np[N_c+N_d:N_c+N_d+N_m]], axis=1)).float()
            x_lncrna = torch.from_numpy(np.concatenate([x_lncrna_np, mp_deg_np[N_c+N_d+N_m:]], axis=1)).float()
            if verbose:
                nz_cmc = (mp_deg_np[:, 0] > 1e-6).sum()
                nz_dmd = (mp_deg_np[:, 1] > 1e-6).sum()
                nz_cmd = (mp_deg_np[:, 2] > 1e-6).sum()
                print(f"  MPNE: Meta-Path Degree features (3-dim → concat to input) "
                      f"CMC-nz={nz_cmc}, DMD-nz={nz_dmd}, CMD-nz={nz_cmd}")

        # ── LLM Feature Loading (v2.7) ──
        # Load precomputed frozen LLM embeddings. They are passed separately
        # to the model, which projects them through trainable Linear(768→64)
        # before concatenating with existing features.
        _use_llm = getattr(config.model, 'use_llm', False)
        llm_circ_tensor = None
        llm_disease_tensor = None
        llm_mirna_tensor = None
        if _use_llm:
            llm_dir = config.model.llm_dir
            if not os.path.isabs(llm_dir):
                llm_dir = os.path.join(ROOT, llm_dir)

            # Which node types get LLM features? (default: all three)
            _llm_types = getattr(config.model, 'llm_types', ["circrna", "disease", "mirna"])
            _llm_circ = "circrna" in _llm_types
            _llm_dis = "disease" in _llm_types
            _llm_mir = "mirna" in _llm_types

            # Load circRNA embeddings (DNABERT) — ordered same as dataset.circrna_names
            if _llm_circ:
                circ_llm_path = os.path.join(llm_dir, f"circ_{dataset_name}_dnabert2.pt")
                if os.path.exists(circ_llm_path):
                    llm_circ_tensor = torch.load(circ_llm_path, map_location='cpu', weights_only=True)
                    llm_circ_tensor = _align_embeddings(llm_circ_tensor, dataset.circrna_names, N_c, "circRNA")
            else:
                llm_circ_tensor = None

            # Load disease embeddings (PubMedBERT) — ordered same as dataset.disease_names
            if _llm_dis:
                dis_llm_path = os.path.join(llm_dir, f"disease_{dataset_name}_pubmedbert.pt")
                if os.path.exists(dis_llm_path):
                    llm_disease_tensor = torch.load(dis_llm_path, map_location='cpu', weights_only=True)
                    llm_disease_tensor = _align_embeddings(llm_disease_tensor, dataset.disease_names, N_d, "disease")
            else:
                llm_disease_tensor = None

            # Load miRNA embeddings (DNABERT) — name dict → aligned matrix
            if _llm_mir:
                mir_llm_path = os.path.join(llm_dir, f"mirna_{dataset_name}_dnabert2.pt")
                if os.path.exists(mir_llm_path):
                    mirna_names = node_counts.get('mirna_names', None)
                    if mirna_names is not None:
                        llm_mirna = torch.load(mir_llm_path, map_location='cpu', weights_only=True)
                        # miRNA .pt is a {name: embedding} dict
                        if isinstance(llm_mirna, dict):
                            llm_mirna_tensor = _align_embeddings_from_dict(llm_mirna, mirna_names, N_m, "miRNA")
                        else:
                            llm_mirna_tensor = _align_embeddings(llm_mirna, mirna_names, N_m, "miRNA")
                    else:
                        if verbose:
                            print("  [LLM] miRNA names not in node_counts — skipping miRNA LLM features")
            else:
                llm_mirna_tensor = None

            if verbose:
                llm_extra = getattr(config.model, 'llm_proj_dim', 128)
                _llm_types_display = getattr(config.model, 'llm_types', ["circrna", "disease", "mirna"])
                parts = []
                if llm_circ_tensor is not None: parts.append("circ")
                if llm_disease_tensor is not None: parts.append("dis")
                if llm_mirna_tensor is not None: parts.append("mir")
                print(f"  [LLM] Loaded frozen embeddings → trainable Linear(768→{llm_extra}) "
                      f"on [{', '.join(parts)}]. "
                      f"Base feature dims: c={x_circ.shape[1]}, d={x_disease.shape[1]}, "
                      f"m={x_mirna.shape[1]}, l={x_lncrna.shape[1]}")

        in_dim_c = x_circ.shape[1]
        in_dim_d = x_disease.shape[1]
        in_dim_m = x_mirna.shape[1]
        in_dim_l = x_lncrna.shape[1]

        # Prepare fold data
        train_pos = np.where(train_inner > 0)
        val_pos = np.where(val_mask > 0)
        test_pos = np.where(test_mask > 0)

        all_pos_mask = np.clip(train_mask + test_mask, 0, 1)
        exclude_set = set()
        for i, j in zip(train_pos[0], train_pos[1]):
            exclude_set.add((i, j))
        for i, j in zip(val_pos[0], val_pos[1]):
            exclude_set.add((i, j))
        for i, j in zip(test_pos[0], test_pos[1]):
            exclude_set.add((i, j))

        val_neg = sample_negative_pairs(
            all_pos_mask, n_samples=len(val_pos[0]), exclude_pairs=exclude_set)
        test_neg = sample_negative_pairs(
            all_pos_mask, n_samples=len(test_pos[0]), exclude_pairs=exclude_set)

        fold_data = {
            "train_circ": train_pos[0], "train_dis": train_pos[1],
            "val_circ": val_pos[0], "val_dis": val_pos[1],
            "val_neg_circ": val_neg[0], "val_neg_dis": val_neg[1],
            "test_pos_circ": test_pos[0], "test_pos_dis": test_pos[1],
            "test_neg_circ": test_neg[0], "test_neg_dis": test_neg[1],
        }

        # Compute LLM effective dims early (for both verbose display and model construction)
        _llm_proj_dim = getattr(config.model, 'llm_proj_dim', 128)
        _llm_pdim_c_eff = _llm_proj_dim if llm_circ_tensor is not None else 0
        _llm_pdim_d_eff = _llm_proj_dim if llm_disease_tensor is not None else 0
        _llm_pdim_m_eff = _llm_proj_dim if llm_mirna_tensor is not None else 0

        if verbose:
            print(f"  Nodes: circ={N_c}, dis={N_d}, mir={N_m}, lnc={N_l} (total={N_c+N_d+N_m+N_l})")
            print(f"  Edges: {edge_index.shape[1]}")
            eff_c = in_dim_c + _llm_pdim_c_eff
            eff_d = in_dim_d + _llm_pdim_d_eff
            eff_m = in_dim_m + _llm_pdim_m_eff
            eff_l = in_dim_l
            print(f"  Features: c={in_dim_c}D→{eff_c}D, d={in_dim_d}D→{eff_d}D, "
                  f"m={in_dim_m}D→{eff_m}D, l={eff_l}D")

        # Apply ablation
        gcn_layers = config.model.gcn_layers
        num_resolutions = config.model.num_resolutions

        if ablation == "no_global":
            num_resolutions = 2  # remove Token₃ (CrossType), keep [Self, 1-hop]
        elif ablation == "no_2hop":
            gcn_layers = 2
            num_resolutions = 4  # ADD 2-hop back (inverse of v2.1 default)
        elif ablation == "no_gcn":
            gcn_layers = 0
            num_resolutions = 1  # only Self (no graph processing)
        elif ablation == "token1_only":
            gcn_layers = 1
            num_resolutions = 1  # only Token₁ (1-hop HGT)

        # Build model (heterogeneous version, v2.2: no CRT attention layers)
        # _llm_proj_dim already computed above; reuse
        _llm_types = getattr(config.model, 'llm_types', ["circrna", "disease", "mirna"])
        _llm_pdim_c = _llm_proj_dim if "circrna" in _llm_types else 0
        _llm_pdim_d = _llm_proj_dim if "disease" in _llm_types else 0
        _llm_pdim_m = _llm_proj_dim if "mirna" in _llm_types else 0
        model = MRSECDAHetero(
            input_dim_circrna=in_dim_c,
            input_dim_disease=in_dim_d,
            input_dim_mirna=in_dim_m,
            input_dim_lncrna=in_dim_l,
            hidden_dim=config.model.hidden_dim,
            lap_pe_dim=config.model.lap_pe_dim,
            rwse_dim=config.model.rwse_dim,
            gcn_layers=gcn_layers,
            gcn_dropout=config.model.gcn_dropout,
            decoder_hidden=config.model.decoder_hidden,
            decoder_dropout=config.model.decoder_dropout,
            num_resolutions=num_resolutions,
            num_node_types=4,
            struct_feat_dim=struct_feat_dim,
            use_hgt=config.model.use_hgt,
            hgt_heads=config.model.hgt_heads,
            use_hypergraph=config.model.use_hypergraph,
            hgcn_dropout=config.model.hgcn_dropout,
            use_llm=_use_llm,
            llm_proj_dim=_llm_proj_dim,
            llm_proj_dim_circ=_llm_pdim_c,
            llm_proj_dim_disease=_llm_pdim_d,
            llm_proj_dim_mirna=_llm_pdim_m,
            edge_dropout=config.model.edge_dropout,
            token_dropout=config.model.token_dropout,
            use_attn_temp=config.model.use_attn_temp,
            use_sigmoid_gate=config.model.use_sigmoid_gate,
            use_multiscale_topo=False,
            msne_num_scales=3,
            use_metapath=_use_metapath,
            metapath_mode=_metapath_mode,
            metapath_num_paths=getattr(config.model, 'metapath_num_paths', 3),
            metapath_residual_weight=getattr(config.model, 'metapath_residual_weight', 0.1),
            num_edge_types=_num_edge_types,
            use_metapath_edges=_use_metapath_edges,
            crt_single_token=getattr(config.model, 'crt_single_token', False),
            mean_gate=getattr(config.model, 'mean_gate', False),
        )

        n_params = sum(p.numel() for p in model.parameters())
        if verbose:
            print(f"  Model params: {n_params:,}")

        _, val_metrics, test_metrics = train_one_fold_hetero(
            model, x_circ, x_disease, x_mirna, x_lncrna,
            edge_index, node_counts, fold_data, train_inner,
            config, device, verbose=verbose,
            edge_index_by_type=edge_index_by_type,
            struct_feat_computer=struct_feat_computer,
            llm_circ=llm_circ_tensor, llm_disease=llm_disease_tensor,
            llm_mirna=llm_mirna_tensor,
            msne_feat_all=None,
            fixed_threshold=fixed_threshold,
            save_score_path=(os.path.join(save_score_dir, f"{dataset_name}_fold{fold_idx}.npy")
                             if save_score_dir else None))

        fold_results.append(test_metrics)

    if not fold_results:
        return {}

    avg = {}
    for k in fold_results[0].keys():
        first = fold_results[0][k]
        if isinstance(first, dict):
            # recall_at_k: {k: value} averaged per cutoff
            avg[k] = {str(kk): float(np.mean([r[k][kk] for r in fold_results]))
                      for kk in first}
        else:
            vals = [r[k] for r in fold_results]
            avg[k] = np.mean(vals)
            avg[f"{k}_std"] = np.std(vals)

    if verbose:
        print(f"\n  --- {dataset_name} Summary ---")
        for k in ["auc", "aupr", "accuracy", "f1"]:
            if k in avg:
                print(f"  {k}: {avg[k]:.4f} ± {avg[f'{k}_std']:.4f}")
        if "recall_at_k" in avg:
            print(f"  recall@k: " + " ".join(
                f"{kk}={v:.4f}" for kk, v in sorted(avg["recall_at_k"].items(), key=lambda x: int(x[0]))))

    return {"fold_results": fold_results, "summary": avg}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MRSECDA v2 Heterogeneous Training")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["no_global", "no_2hop", "no_gcn",
                                 "no_mirna", "no_lncrna", "token1_only"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--pos_weight", type=float, default=None,
                        help="Positive class weight for BCE loss (default: neg_ratio)")
    parser.add_argument("--lr_scheduler", type=str, default=None,
                        choices=["cosine"], help="LR scheduler type")
    parser.add_argument("--lr_warmup", type=int, default=None,
                        help="LR linear warmup epochs (default: 0, recommended: 20-50 for lr>=1e-3)")
    parser.add_argument("--struct_feat", action="store_true", default=False,
                        help="Enable pairwise structural features (PSE)")
    parser.add_argument("--no_struct_feat", action="store_true", default=False,
                        help="Disable PSE structural features (ablation)")
    parser.add_argument("--single_view", action="store_true", default=False,
                        help="Use single-view struct features (default: multi-view 12-dim)")
    parser.add_argument("--drop_struct_feat", type=str, nargs="+", default=None,
                        choices=["jaccard", "adamic_adar", "resource_alloc", "inv_distance"],
                        help="Per-feature PSE ablation: drop one or more features from all 3 views")
    parser.add_argument("--drop_struct_view", type=int, nargs="+", default=None,
                        choices=[0, 1, 2],
                        help="Per-view PSE ablation: drop subgraph view(s) "
                             "(0=sim+assoc, 1=metapath, 2=full)")
    parser.add_argument("--no_hgt", action="store_true", default=False,
                        help="Disable HGT, use GCN encoder instead (HGT is default)")
    parser.add_argument("--hgt_heads", type=int, default=None,
                        help="Number of attention heads in HGT (default: 2 from config)")
    parser.add_argument("--hgt_layers", type=int, default=None,
                        help="Number of HGT layers (default: same as gcn_layers)")
    parser.add_argument("--no_hypergraph", action="store_true", default=False,
                        help="Disable hypergraph convolution module (v2.4)")
    parser.add_argument("--use_llm", action="store_true", default=False,
                        help="Enable LLM feature concatenation (DNABERT-2 + PubMedBERT)")
    parser.add_argument("--llm_dir", type=str, default=None,
                        help="Directory for precomputed LLM embeddings")
    parser.add_argument("--llm_proj_dim", type=int, default=None,
                        help="LLM projection dimension (default: 128 from config)")
    parser.add_argument("--llm_types", type=str, nargs="+",
                        default=["circrna", "disease", "mirna"],
                        choices=["circrna", "disease", "mirna"],
                        help="Node types to apply LLM features (default: all three)")
    # ── v3.1 Encoder Innovations ──
    parser.add_argument("--token_dropout", type=float, default=None,
                        help="TokenDrop rate: randomly mask resolution tokens (0=disabled, 0.1 rec.)")
    parser.add_argument("--no_attn_temp", action="store_true", default=False,
                        help="Disable per-edge-type attention temperature")
    parser.add_argument("--consistency_weight", type=float, default=None,
                        help="Consistency regularization weight (0=disabled, 0.1 rec.)")
    parser.add_argument("--contrastive_weight", type=float, default=None,
                        help="Multi-view contrastive loss weight (0=disabled, 0.05 rec.)")
    parser.add_argument("--aux_task_weight", type=float, default=None,
                        help="Auxiliary multi-task loss weight (miRNA-disease reconstruction)")
    # ── v3.2 Encoder Innovations ──
    parser.add_argument("--deep_sup_weight", type=float, default=None,
                        help="Token-wise deep supervision weight (0=disabled, 0.3 rec.)")
    parser.add_argument("--sigmoid_gate", action="store_true", default=False,
                        help="Use independent sigmoid edge-type gating (replaces softmax)")
    parser.add_argument("--edge_dropout", type=float, default=None,
                        help="HGT edge dropout rate (0=disabled, 0.05 default, 0.2 for annealing)")
    parser.add_argument("--edge_dropout_anneal", action="store_true", default=False,
                        help="Enable DropEdge annealing (decay from edge_dropout→0)")
    parser.add_argument("--label_smoothing", type=float, default=None,
                        help="Label smoothing factor (0=disabled, 0.05 rec.)")
    parser.add_argument("--focal_gamma", type=float, default=None,
                        help="Focal loss gamma (0=standard BCE, 1.0-2.0 for hard mining)")
    # ── v3.3 Meta-Path Neighbor Encoding ──
    parser.add_argument("--use_metapath", type=lambda x: x.lower() == 'true',
                        default=None, help="Enable/disable MPNE meta-path degree features")
    parser.add_argument("--metapath_mode", type=str, default=None,
                        choices=["concat", "crt"],
                        help="MPNE mode: 'concat' (input concat) or 'crt' (token fusion)")
    # ── v3.4 Meta-Path HGT Edge Types ──
    parser.add_argument("--use_metapath_edges", action="store_true", default=False,
                        help="Add CMC/DMD 2-hop edges as HGT edge types (types 3,4)")
    parser.add_argument("--metapath_edge_topk", type=int, default=None,
                        help="Top-K meta-path neighbors per node (default: 15)")
    parser.add_argument("--metapath_edge_min_shared", type=int, default=None,
                        help="Min shared miRNAs for meta-path edge (default: 1)")
    parser.add_argument("--use_lncrna", action="store_true", default=False,
                        help="MRSECDA: re-enable lncRNA nodes (default: disabled)")
    # ── v3.3 Ablation ──
    parser.add_argument("--crt_single_token", action="store_true", default=False,
                        help="Ablation: use only Token1 (HGT), drop Token0+Token3 (single-token CRT)")
    parser.add_argument("--mean_gate", action="store_true", default=False,
                        help="Ablation: replace adaptive gate with simple 0.5 mean in dual-path decoder")
    parser.add_argument("--lap_pe_dim", type=int, default=None,
                        help="LapPE dimension (0=disable, default: 16)")
    parser.add_argument("--rwse_dim", type=int, default=None,
                        help="RWSE dimension (0=disable, default: 16)")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout rate (sets both gcn_dropout and decoder_dropout)")
    parser.add_argument("--fixed_threshold", type=float, default=None,
                        help="Use fixed threshold for F1/ACC/PRE/REC instead of per-fold optimal (e.g. 0.5)")
    parser.add_argument("--save_score_dir", type=str, default=None,
                        help="Save the full per-fold score matrix S (n_circ x n_disease) as .npy under this dir")
    args = parser.parse_args()

    config = Config()
    config.train.device = args.device
    if args.epochs is not None:
        config.train.finetune_epochs = args.epochs
    if args.lr is not None:
        config.train.finetune_lr = args.lr
    if args.hidden_dim is not None:
        config.model.hidden_dim = args.hidden_dim
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.pos_weight is not None:
        config.train.pos_weight = args.pos_weight
    if args.lr_scheduler is not None:
        config.train.use_lr_scheduler = True
    if args.lr_warmup is not None:
        config.train.lr_warmup_epochs = args.lr_warmup
    if args.struct_feat:
        config.model.use_struct_feat = True
    if args.no_struct_feat:
        config.model.use_struct_feat = False
    if args.single_view:
        config.model.use_multi_view_struct = False
    if args.no_hgt:
        config.model.use_hgt = False
    if args.hgt_heads is not None:
        config.model.hgt_heads = args.hgt_heads
    if args.hgt_layers is not None:
        config.model.hgt_layers = args.hgt_layers
        config.model.gcn_layers = args.hgt_layers  # keep in sync for token construction
    if args.no_hypergraph:
        config.model.use_hypergraph = False
    if args.use_llm:
        config.model.use_llm = True
    if args.llm_dir is not None:
        config.model.llm_dir = args.llm_dir
    if args.llm_proj_dim is not None:
        config.model.llm_proj_dim = args.llm_proj_dim
    if args.llm_types is not None:
        config.model.llm_types = args.llm_types
    # ── v3.1 Encoder Innovations ──
    if args.token_dropout is not None:
        config.model.token_dropout = args.token_dropout
    if args.no_attn_temp:
        config.model.use_attn_temp = False
    if args.consistency_weight is not None:
        config.model.consistency_weight = args.consistency_weight
    if args.contrastive_weight is not None:
        config.model.contrastive_weight = args.contrastive_weight
    if args.aux_task_weight is not None:
        config.model.aux_task_weight = args.aux_task_weight
    # ── v3.2 Encoder Innovations ──
    if args.deep_sup_weight is not None:
        config.model.deep_sup_weight = args.deep_sup_weight
    if args.sigmoid_gate:
        config.model.use_sigmoid_gate = True
    if args.edge_dropout is not None:
        config.model.edge_dropout = args.edge_dropout
    if args.edge_dropout_anneal:
        config.model.edge_dropout_anneal = True
    if args.label_smoothing is not None:
        config.model.label_smoothing = args.label_smoothing
    if args.focal_gamma is not None:
        config.model.focal_gamma = args.focal_gamma
    # ── v3.3 Meta-Path Neighbor Encoding ──
    if args.use_metapath is not None:
        config.model.use_metapath = args.use_metapath
    if args.metapath_mode is not None:
        config.model.metapath_mode = args.metapath_mode
    # ── v3.4 Meta-Path HGT Edge Types ──
    if args.use_metapath_edges:
        config.model.use_metapath_edges = True
        config.model.use_metapath = False  # disable MPNE concat (no interference)
    if args.use_lncrna:
        config.model.use_lncrna = True  # MRSECDA ablation: re-add lncRNA nodes
    if args.crt_single_token:
        config.model.crt_single_token = True
    if args.mean_gate:
        config.model.mean_gate = True
    if args.metapath_edge_topk is not None:
        config.model.metapath_edge_topk = args.metapath_edge_topk
    if args.metapath_edge_min_shared is not None:
        config.model.metapath_edge_min_shared = args.metapath_edge_min_shared
    if args.lap_pe_dim is not None:
        config.model.lap_pe_dim = args.lap_pe_dim
    if args.rwse_dim is not None:
        config.model.rwse_dim = args.rwse_dim
    if args.dropout is not None:
        config.model.gcn_dropout = args.dropout
        config.model.decoder_dropout = args.dropout
        config.model.hgcn_dropout = args.dropout

    print(f"Device: {args.device}")
    print(f"Config: hidden_dim={config.model.hidden_dim}, lr={config.train.finetune_lr}, "
          f"epochs={config.train.finetune_epochs}, batch={config.train.batch_size}, "
          f"pos_weight={getattr(config.train, 'pos_weight', 'auto')}, "
          f"scheduler={'cosine' if getattr(config.train, 'use_lr_scheduler', False) else 'none'}, "
          f"gcn_layers={config.model.gcn_layers}, "
          f"mpne={getattr(config.model, 'use_metapath', True)}, "
          f"mp_mode={getattr(config.model, 'metapath_mode', 'concat')}, "
          f"mp_edges={getattr(config.model, 'use_metapath_edges', False)}")

    if args.datasets:
        datasets = args.datasets
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ALL_DATASETS  # default: all 5

    all_results = {}
    folds_to_run = [args.fold] if args.fold is not None else None
    ablation_tag = f"_{args.ablation}" if args.ablation else ""
    if args.drop_struct_feat:
        _drop_tag = {"jaccard": "no_jaccard", "adamic_adar": "no_aa",
                     "resource_alloc": "no_ra", "inv_distance": "no_inv_dist"}
        _tags = [_drop_tag[f] for f in args.drop_struct_feat]
        ablation_tag = "_" + "_and_".join(_tags)

    for ds_name in datasets:
        result = run_dataset_hetero(
            ds_name, config, args.device,
            ablation=args.ablation, seed=args.seed,
            folds_to_run=folds_to_run,
            fixed_threshold=args.fixed_threshold,
            drop_struct_feat=args.drop_struct_feat,
            drop_struct_view=args.drop_struct_view,
            save_score_dir=args.save_score_dir)
        if result:
            all_results[ds_name] = result

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(
        RESULT_DIR, f"mrsecda{ablation_tag}_{timestamp}.json")

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(result_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\nResults saved to: {result_path}")

    print(f"\n{'='*70}")
    print(f"MRSECDA v2 Hetero Summary{' (' + args.ablation + ')' if args.ablation else ''}")
    print(f"{'='*70}")
    for ds_name, result in all_results.items():
        s = result.get("summary", {})
        if s:
            print(f"  {ds_name:20s}  AUC={s.get('auc', 0):.4f}±{s.get('auc_std', 0):.4f}  "
                  f"AUPR={s.get('aupr', 0):.4f}±{s.get('aupr_std', 0):.4f}  "
                  f"F1={s.get('f1', 0):.4f}±{s.get('f1_std', 0):.4f}")

    return all_results


if __name__ == "__main__":
    main()
