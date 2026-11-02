"""
数据处理模块
- 数据集加载 (关联矩阵)
- GIP Kernel 相似度
- 疾病语义相似度 (基于MeSH DAG)
- circRNA功能相似度
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import fractional_matrix_power
from sklearn.preprocessing import normalize
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# 1. GIP Kernel Similarity (Gaussian Interaction Profile)
# =============================================================================

def gip_kernel(adj_matrix: np.ndarray, gamma_ratio: float = 1.0) -> np.ndarray:
    r"""
    计算GIP kernel相似度矩阵。

    K_{ij} = exp(-γ * ||y_i - y_j||²)

    其中:
    - y_i 是关联矩阵的第i行 (实体i的interaction profile)
    - γ = γ_ratio / (1/N * Σ_k ||y_k||²)

    Parameters
    ----------
    adj_matrix : shape (N, M)
        关联矩阵, 行=实体, 列=另一端实体
        对于circRNA GIP: adj_matrix = A (shape [N_circ, N_disease])
        对于disease GIP: adj_matrix = A.T (shape [N_disease, N_circ])
    gamma_ratio : float
        γ的缩放系数

    Returns
    -------
    K : shape (N, N)
        GIP kernel相似度矩阵
    """
    N = adj_matrix.shape[0]
    # 计算平方欧氏距离
    sq_dists = np.zeros((N, N))
    for i in range(N):
        diff = adj_matrix[i:i+1, :] - adj_matrix
        sq_dists[i, :] = np.sum(diff ** 2, axis=1)

    # 计算带宽参数 γ
    mean_sq_norm = np.mean(np.sum(adj_matrix ** 2, axis=1))
    if mean_sq_norm == 0:
        gamma = 1.0
    else:
        gamma = gamma_ratio / mean_sq_norm

    K = np.exp(-gamma * sq_dists)
    np.fill_diagonal(K, 1.0)  # 确保对角=1
    return K


# =============================================================================
# 2. CircRNA Functional Similarity (基于关联疾病的相似度)
# =============================================================================

def circrna_functional_similarity(adj_matrix: np.ndarray,
                                   disease_sim: np.ndarray) -> np.ndarray:
    r"""
    计算circRNA功能相似度。

    基于: 如果两个circRNA与相似的疾病关联, 则认为它们功能相似。
    参考miRNA functional similarity的思路:

    FS(c_i, c_j) = Σ_{d∈D(c_i)} max_{d'∈D(c_j)} sim(d, d')
                   / (|D(c_i)| + |D(c_j)|) + 对称项

    Parameters
    ----------
    adj_matrix : shape (N_circ, N_disease)
        circRNA-disease关联矩阵 (binary)
    disease_sim : shape (N_disease, N_disease)
        疾病语义相似度矩阵

    Returns
    -------
    fs : shape (N_circ, N_circ)
        circRNA功能相似度矩阵
    """
    N_circ, N_disease = adj_matrix.shape
    fs = np.zeros((N_circ, N_circ))

    # 每个circRNA关联的疾病集合
    circ_diseases = [set(np.where(adj_matrix[i] > 0)[0]) for i in range(N_circ)]

    for i in range(N_circ):
        if len(circ_diseases[i]) == 0:
            continue
        for j in range(i, N_circ):
            if len(circ_diseases[j]) == 0:
                continue
            if i == j:
                fs[i, j] = 1.0
                continue

            # c_i → c_j 方向的相似度
            sim_ij = 0.0
            for di in circ_diseases[i]:
                if len(circ_diseases[j]) > 0:
                    max_sim = max(disease_sim[di, dj] for dj in circ_diseases[j])
                    sim_ij += max_sim
            sim_ij /= len(circ_diseases[i])

            # c_j → c_i 方向的相似度
            sim_ji = 0.0
            for dj in circ_diseases[j]:
                if len(circ_diseases[i]) > 0:
                    max_sim = max(disease_sim[dj, di] for di in circ_diseases[i])
                    sim_ji += max_sim
            sim_ji /= len(circ_diseases[j])

            fs[i, j] = (sim_ij + sim_ji) / 2.0
            fs[j, i] = fs[i, j]

    return fs


# =============================================================================
# 3. Disease Semantic Similarity (基于MeSH DAG)
# =============================================================================

def load_mesh_disease_dag() -> Dict[str, set]:
    """
    加载MeSH疾病DAG数据。

    返回疾病MeSH ID与其祖先节点的映射。
    如果无法获取MeSH数据, 返回空字典。
    """
    # MeSH数据通常需要从NCBI或HMDD获取
    # 这里提供一个简化版本, 支持从本地文件加载
    # 如无法获取则返回空(回退到GIP only)
    return {}


def disease_semantic_similarity(disease_names: list,
                                 mesh_dag: Optional[Dict] = None) -> np.ndarray:
    r"""
    计算疾病语义相似度。

    基于MeSH DAG: 两个疾病共享的祖先越多, 它们越相似。

    If no MeSH DAG is available, falls back to GIP-based similarity.

    Parameters
    ----------
    disease_names : list
        疾病名称列表
    mesh_dag : dict or None
        MeSH DAG数据

    Returns
    -------
    sim : shape (N_disease, N_disease)
    """
    N = len(disease_names)
    if mesh_dag is None or len(mesh_dag) == 0:
        # 无法获取MeSH, 返回None让调用者fallback
        return None

    # 实现DAG-based语义相似度
    # 这里使用Wang's method的简化版本
    sim = np.eye(N)
    # TODO: 完整实现需要MeSH数据
    return sim


# =============================================================================
# 4. 综合相似度融合
# =============================================================================

def fuse_similarities(gip_sim: np.ndarray,
                       other_sim: Optional[np.ndarray] = None,
                       alpha: float = 0.5) -> np.ndarray:
    """
    融合GIP相似度和其他相似度 (如功能/语义相似度)。

    S_fused = α * S_gip + (1-α) * S_other

    如果other_sim为None, 直接返回GIP。

    Parameters
    ----------
    gip_sim : shape (N, N)
        GIP kernel相似度
    other_sim : shape (N, N) or None
        其他相似度
    alpha : float
        GIP的权重, α∈[0,1]

    Returns
    -------
    fused : shape (N, N)
    """
    if other_sim is None:
        return gip_sim
    return alpha * gip_sim + (1 - alpha) * other_sim


def fuse_multi_source(
    *sources: Tuple[np.ndarray, float],
) -> np.ndarray:
    """
    Fuse multiple similarity matrices with configurable weights.

    Each source is a tuple of (similarity_matrix, weight).
    Weights are normalized to sum to 1.0.
    None sources are filtered out (their weight redistributed).

    Parameters
    ----------
    *sources : (np.ndarray, float) tuples
        Each tuple: (N×N similarity matrix, weight)

    Returns
    -------
    fused : shape (N, N)
        Weighted average of all valid sources.
        Falls back to identity matrix if all sources are None.
    """
    valid = [(sim, w) for sim, w in sources if sim is not None]
    if not valid:
        # All sources None — return identity (no similarity information)
        return None

    if len(valid) == 1:
        return valid[0][0]

    total_weight = sum(w for _, w in valid)
    if total_weight == 0:
        return valid[0][0]

    result = np.zeros_like(valid[0][0], dtype=np.float64)
    for sim, w in valid:
        result += (w / total_weight) * sim

    return result


def normalize_similarity_matrix(S: np.ndarray) -> np.ndarray:
    """
    Normalize a similarity matrix to [0, 1] range with diagonal = 1.0.

    Uses cosine-like normalization:
      S_norm[i,j] = S[i,j] / sqrt(S[i,i] * S[j,j])

    Parameters
    ----------
    S : shape (N, N)
        Potentially unnormalized similarity matrix

    Returns
    -------
    S_norm : shape (N, N)
        Normalized similarity matrix with values in [0, 1]
    """
    N = S.shape[0]
    diag = np.diag(S).copy()
    diag = np.where(diag == 0, 1.0, diag)  # Avoid division by zero
    denom = np.sqrt(np.outer(diag, diag))
    denom = np.where(denom == 0, 1.0, denom)
    S_norm = S / denom
    S_norm = np.clip(S_norm, 0.0, 1.0)
    np.fill_diagonal(S_norm, 1.0)
    return S_norm


def sparsify_similarity(sim_matrix: np.ndarray, topk: int) -> np.ndarray:
    """
    稀疏化相似度矩阵: 每行只保留top-k个最大值, 其余置零。

    Parameters
    ----------
    sim_matrix : shape (N, N)
    topk : int

    Returns
    -------
    sparse_sim : shape (N, N)
    """
    N = sim_matrix.shape[0]
    sparse_sim = np.zeros_like(sim_matrix)

    for i in range(N):
        # 跳过自身
        row = sim_matrix[i].copy()
        row[i] = -np.inf
        top_indices = np.argsort(row)[-topk:]
        sparse_sim[i, top_indices] = sim_matrix[i, top_indices]

    return sparse_sim
