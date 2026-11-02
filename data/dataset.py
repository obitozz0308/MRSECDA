"""
数据集加载模块
- 加载circRNA-disease关联矩阵
- 构建异构图
- 生成拓扑编码 (Laplacian PE + Random Walk SE)
"""

import os
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected, degree

from data.similarity import (
    gip_kernel,
    circrna_functional_similarity,
    fuse_similarities,
    fuse_multi_source,
    sparsify_similarity,
)


# =============================================================================
# 1. Benchmark数据集加载
# =============================================================================

class CDADataset:
    """
    CDA benchmark数据集类。

    每个数据集包含:
    - circRNA-disease关联矩阵 A (N_circ × N_disease)
    - circRNA名称列表
    - disease名称列表
    - (可选) 预计算的相似度矩阵
    """

    def __init__(
        self,
        name: str,
        data_dir: str = "./data/raw",
        topk_circrna: int = 15,
        topk_disease: int = 15,
        gip_gamma: float = 1.0,
        seed: int = 42,
        use_external_knowledge: bool = True,
        use_kmer_features: bool = True,
        use_go_features: bool = False,
    ):
        self.name = name
        self.data_dir = data_dir
        self.topk_circrna = topk_circrna
        self.topk_disease = topk_disease
        self.gip_gamma = gip_gamma
        self.seed = seed

        # use_external_knowledge acts as a master switch for backward compat
        self.use_external_knowledge = use_external_knowledge
        # Fine-grained feature toggles (only active when use_external_knowledge=True)
        self.use_kmer_features = use_kmer_features
        self.use_go_features = use_go_features
        np.random.seed(seed)

        # 加载数据
        self._load()

        # 加载circRNA/Disease名称列表 (用于外部知识库查询)
        self._load_names()

        # 预计算相似度
        self._compute_similarities()

    def _load(self):
        """加载关联矩阵。支持多种数据格式。"""
        data_path = os.path.join(self.data_dir, self.name)

        # 数据集到h5文件路径的映射
        h5_file_map = {
            "circR2Disease": "disease-circRNA.h5",
            "circR2Cancer": "circRNA_cancer/circRNA_cancer.h5",
            "circRNADisease": "circRNA_disease_from_circRNADisease/association.h5",
            "circ2Traits": "circ2Traits/circRNA_disease.h5",
            "circad": "circad/circrna_disease.h5",
        }

        # 尝试多种格式
        loaded = False

        # 0. 尝试从h5文件加载 (benchmark标准格式)
        h5_rel_path = h5_file_map.get(self.name)
        if h5_rel_path:
            h5_path = os.path.join(self.data_dir, h5_rel_path)
            if os.path.exists(h5_path):
                import h5py
                with h5py.File(h5_path, "r") as f:
                    self.adj = f["infor"][:].astype(np.float64)
                loaded = True

        # 1. 尝试从.npz文件加载
        npz_path = data_path + ".npz"
        if not loaded and os.path.exists(npz_path):
            data = np.load(npz_path, allow_pickle=True)
            self.adj = data["adj"]
            self.circrna_names = data.get("circrna_names", None)
            self.disease_names = data.get("disease_names", None)
            loaded = True

        # 2. 尝试从.pkl文件加载
        pkl_path = data_path + ".pkl"
        if not loaded and os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            self.adj = data["adj"]
            self.circrna_names = data.get("circrna_names", None)
            self.disease_names = data.get("disease_names", None)
            loaded = True

        # 3. 尝试从CSV文件加载
        csv_path = data_path + ".csv"
        if not loaded and os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path, index_col=0)
            self.adj = df.values.astype(np.float64)
            self.circrna_names = list(df.index)
            self.disease_names = list(df.columns)
            loaded = True

        # 4. 生成合成数据用于演示
        if not loaded:
            print(f"[{self.name}] 未找到数据文件, 生成合成演示数据集...")
            self._generate_synthetic()

        self.adj = np.asarray(self.adj, dtype=np.float64)
        self.N_circ = self.adj.shape[0]
        self.N_disease = self.adj.shape[1]

        print(f"[{self.name}] circRNA: {self.N_circ}, "
              f"Disease: {self.N_disease}, "
              f"Associations: {int(np.sum(self.adj > 0))}")

    def _generate_synthetic(self):
        """生成合成数据集用于测试管线。"""
        N_circ, N_disease = 500, 80
        n_assoc = 600
        adj = np.zeros((N_circ, N_disease))
        idx = np.random.choice(N_circ * N_disease, n_assoc, replace=False)
        adj.flat[idx] = 1.0
        self.adj = adj
        self.circrna_names = [f"circ_{i}" for i in range(N_circ)]
        self.disease_names = [f"disease_{i}" for i in range(N_disease)]

    def _load_names(self):
        """Load circRNA and disease name lists for external knowledge queries."""
        # Per-dataset subdirectory mapping (same structure as _load h5_file_map)
        dataset_subdir_map = {
            "circR2Disease": "",                    # root data_dir
            "circR2Cancer": "circRNA_cancer",
            "circRNADisease": "circRNA_disease_from_circRNADisease",
            "circ2Traits": "circ2Traits",
            "circad": "circad",
        }
        subdir = dataset_subdir_map.get(self.name, "")
        ds_dir = os.path.join(self.data_dir, subdir) if subdir else self.data_dir

        # Per-dataset circRNA name file candidates
        circ_candidates = [
            (os.path.join(ds_dir, "circRNA_List.csv"), "csv"),
            (os.path.join(ds_dir, "circRNA_list.csv"), "csv"),
            (os.path.join(ds_dir, "all_circRNAs.csv"), "csv"),
            (os.path.join(ds_dir, "circrna_list.xls"), "xls"),
            (os.path.join(ds_dir, "circRNA_list.xlsx"), "xlsx"),
            (os.path.join(ds_dir, "circrna_list.xls"), "xls"),
        ]
        # Also search root for shared files
        circ_candidates += [
            (os.path.join(self.data_dir, "circRNA_List.csv"), "csv"),
        ]

        dis_candidates = [
            (os.path.join(ds_dir, "Disease_List.csv"), "csv"),
            (os.path.join(ds_dir, "disease_list.csv"), "csv"),
            (os.path.join(ds_dir, "all_diseases.csv"), "csv"),
            (os.path.join(ds_dir, "disease_list.xls"), "xls"),
            (os.path.join(ds_dir, "disease_list.xlsx"), "xlsx"),
        ]
        dis_candidates += [
            (os.path.join(self.data_dir, "Disease_List.csv"), "csv"),
        ]

        self.circrna_names = getattr(self, "circrna_names", None)
        self.disease_names = getattr(self, "disease_names", None)

        def _read_names(path, fmt):
            norm = os.path.normpath(path)
            if not os.path.exists(norm):
                return None
            if fmt == "csv":
                with open(norm, "r") as f:
                    return [l.strip() for l in f if l.strip()]
            elif fmt in ("xls", "xlsx"):
                try:
                    import pandas as pd
                    df = pd.read_excel(norm, header=None)
                    return [str(v).strip() for v in df.iloc[:, 0].tolist() if str(v).strip()]
                except Exception:
                    return None
            return None

        # Load circRNA names
        if self.circrna_names is None:
            for path, fmt in circ_candidates:
                names = _read_names(path, fmt)
                if names:
                    self.circrna_names = names
                    break

        # Load disease names
        if self.disease_names is None:
            for path, fmt in dis_candidates:
                names = _read_names(path, fmt)
                if names:
                    self.disease_names = names
                    break

        # Generate default names if loading failed
        if self.circrna_names is None:
            self.circrna_names = [f"circRNA_{i}" for i in range(self.N_circ)]
        if self.disease_names is None:
            self.disease_names = [f"disease_{i}" for i in range(self.N_disease)]

    def _compute_similarities(self):
        """计算相似度矩阵 (GIP kernel + MeSH + functional similarity)。"""
        # ── GIP Kernel ──
        self.circrna_gip = gip_kernel(self.adj, self.gip_gamma)           # (N_circ, N_circ)
        self.disease_gip = gip_kernel(self.adj.T, self.gip_gamma)         # (N_disease, N_disease)

        # ── Load MeSH disease semantic similarity ──
        self.disease_mesh = None
        try:
            from data.external_knowledge import load_mesh_similarity
            self.disease_mesh = load_mesh_similarity(
                self.disease_names, self.data_dir,
                cache_key=f"mesh_disease_sim_{self.name}",
            )
        except Exception as e:
            print(f"  [Dataset] MeSH not available: {e}")

        # ── Disease Similarity: GIP + MeSH fusion ──
        self.disease_sim = fuse_similarities(self.disease_gip, self.disease_mesh, alpha=0.5)

        # ── CircRNA Functional Similarity (uses fused disease sim) ──
        self.circrna_func = circrna_functional_similarity(self.adj, self.disease_sim)

        # ── Load external knowledge: k-mer similarity ──
        self.circrna_kmer = None
        self.circrna_go = None
        self.circrna_kmer_feat = None
        self.circrna_go_feat = None

        if self.use_external_knowledge:
            try:
                from data.external_knowledge import compute_kmer_similarity
                self.circrna_kmer = compute_kmer_similarity(
                    self.circrna_names, cache_prefix=f"{self.name}_")
            except Exception as e:
                print(f"  [Dataset] k-mer similarity not available: {e}")

            try:
                from data.external_knowledge import compute_go_similarity
                self.circrna_go = compute_go_similarity(
                    self.circrna_names, cache_prefix=f"{self.name}_")
            except Exception as e:
                print(f"  [Dataset] GO similarity not available: {e}")

            try:
                from data.external_knowledge import compute_kmer_features, compute_go_features
                if self.use_kmer_features:
                    self.circrna_kmer_feat = compute_kmer_features(
                        self.circrna_names, cache_prefix=f"{self.name}_")
                if self.use_go_features:
                    self.circrna_go_feat = compute_go_features(
                        self.circrna_names, cache_prefix=f"{self.name}_")
            except Exception as e:
                print(f"  [Dataset] External features not available: {e}")

        # ── CircRNA Similarity: GIP + functional + k-mer + GO fusion ──
        if self.use_external_knowledge:
            self.circrna_sim = fuse_multi_source(
                (self.circrna_gip, 1.0),
                (self.circrna_func, 1.0),
                (self.circrna_kmer, 0.5),
                (self.circrna_go, 0.5),
            )
            if self.circrna_sim is None:
                self.circrna_sim = fuse_similarities(self.circrna_gip, self.circrna_func, alpha=0.5)
        else:
            self.circrna_sim = fuse_similarities(self.circrna_gip, self.circrna_func, alpha=0.5)

        # ── Sparsify ──
        self.circrna_sim_sparse = sparsify_similarity(self.circrna_sim, self.topk_circrna)
        self.disease_sim_sparse = sparsify_similarity(self.disease_sim, self.topk_disease)


# =============================================================================
# 2. 异构图构建
# =============================================================================

def build_hetero_graph(
    dataset: CDADataset,
    use_edge_weights: bool = True,
) -> HeteroData:
    """
    构建PyG HeteroData异构图。

    节点类型:
    - circRNA (N_circ nodes)
    - disease (N_disease nodes)

    边类型:
    - ("circRNA", "assoc", "disease"): 关联边 (已知circRNA-disease关联)
    - ("circRNA", "similar", "circRNA"): circRNA相似边
    - ("disease", "similar", "disease"): disease相似边

    Parameters
    ----------
    dataset : CDADataset
    use_edge_weights : bool
        是否使用边权重 (None则无权)

    Returns
    -------
    data : HeteroData
    """
    data = HeteroData()

    # 节点数量
    data["circRNA"].num_nodes = dataset.N_circ
    data["disease"].num_nodes = dataset.N_disease

    # --- circRNA → disease 关联边 (双向) ---
    circ_idx, dis_idx = np.where(dataset.adj > 0)
    edge_index_cd = torch.tensor(np.stack([circ_idx, dis_idx]), dtype=torch.long)
    edge_index_dc = torch.tensor(np.stack([dis_idx, circ_idx]), dtype=torch.long)

    data["circRNA", "assoc", "disease"].edge_index = edge_index_cd
    data["disease", "assoc", "circRNA"].edge_index = edge_index_dc

    # --- circRNA → circRNA 相似边 (无向 -> 变为双向) ---
    rows_c, cols_c = np.where(dataset.circrna_sim_sparse > 0)
    # 滤除自环 (将在建模时通过skip connection处理)
    non_self = rows_c != cols_c
    rows_c, cols_c = rows_c[non_self], cols_c[non_self]
    edge_index_cc = torch.tensor(np.stack([rows_c, cols_c]), dtype=torch.long)

    if use_edge_weights:
        weights_cc = dataset.circrna_sim_sparse[rows_c, cols_c]
        data["circRNA", "similar", "circRNA"].edge_weight = torch.tensor(
            weights_cc, dtype=torch.float32
        )

    # 转换为无向 (每条边双向)
    edge_index_cc_undir = to_undirected(edge_index_cc)
    data["circRNA", "similar", "circRNA"].edge_index = edge_index_cc_undir

    # --- disease → disease 相似边 ---
    rows_d, cols_d = np.where(dataset.disease_sim_sparse > 0)
    non_self = rows_d != cols_d
    rows_d, cols_d = rows_d[non_self], cols_d[non_self]
    edge_index_dd = torch.tensor(np.stack([rows_d, cols_d]), dtype=torch.long)

    if use_edge_weights:
        weights_dd = dataset.disease_sim_sparse[rows_d, cols_d]
        data["disease", "similar", "disease"].edge_weight = torch.tensor(
            weights_dd, dtype=torch.float32
        )

    edge_index_dd_undir = to_undirected(edge_index_dd)
    data["disease", "similar", "disease"].edge_index = edge_index_dd_undir

    return data


# =============================================================================
# 3. 拓扑编码: Laplacian Positional Encoding
# =============================================================================

def compute_laplacian_pe(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int = 16,
) -> torch.Tensor:
    r"""
    计算Laplacian Positional Encoding。

    L = I - D^{-1/2} A D^{-1/2}
    L v = λ v

    取前k个最小的非零特征向量作为位置编码。

    Parameters
    ----------
    edge_index : shape (2, E)
    num_nodes : int
    k : int
        特征向量个数 (不包含λ=0对应的)

    Returns
    -------
    pe : shape (num_nodes, k)
    """
    # 构建稀疏邻接矩阵
    row, col = edge_index[0].numpy(), edge_index[1].numpy()
    A = sp.coo_matrix(
        (np.ones(len(row)), (row, col)),
        shape=(num_nodes, num_nodes)
    ).tocsr()

    # 确保对称
    A = A.maximum(A.T)

    # 归一化拉普拉斯
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    I = sp.eye(num_nodes)
    L = I - D_inv_sqrt @ A @ D_inv_sqrt

    # 计算特征向量 (最小的k+1个, 去掉λ≈0的)
    try:
        eigenvalues, eigenvectors = eigsh(
            L, k=k + 1, which="SM", tol=1e-4, maxiter=5000
        )
        # 去掉第一个 (λ≈0)
        pe = eigenvectors[:, 1:k+1]
        # 随机翻转符号 (解决符号歧义)
        sign = np.random.choice([-1, 1], size=k)
        pe = pe * sign[np.newaxis, :]
    except Exception:
        # eigsh可能不收敛, 使用随机PE作为fallback
        print("Warning: eigsh不收敛, 使用随机PE")
        pe = np.random.randn(num_nodes, k)

    return torch.tensor(pe, dtype=torch.float32)


# =============================================================================
# 4. 拓扑编码: Random Walk Structural Encoding
# =============================================================================

def compute_rwse(
    edge_index: torch.Tensor,
    num_nodes: int,
    steps: int = 16,
) -> torch.Tensor:
    r"""
    计算Random Walk Structural Encoding。

    对于每步t, 计算转移矩阵P的t次幂的对角元素:
    RWSE_i = [diag(P^1)_i, diag(P^2)_i, ..., diag(P^steps)_i]

    这编码了节点的局部拓扑结构 (每个节点出发的随机游走返回概率)。

    Parameters
    ----------
    edge_index : shape (2, E)
    num_nodes : int
    steps : int

    Returns
    -------
    rwse : shape (num_nodes, steps)
    """
    row, col = edge_index[0].numpy(), edge_index[1].numpy()
    A = sp.coo_matrix(
        (np.ones(len(row)), (row, col)),
        shape=(num_nodes, num_nodes)
    ).tocsr()

    # 对称化
    A = A.maximum(A.T)

    # 度数矩阵
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    D_inv = sp.diags(deg_inv)

    # 转移矩阵 P = D^{-1} A
    P = D_inv @ A

    # 计算各步的返回概率
    rwse_list = []
    P_k = P.copy()
    for _ in range(steps):
        # 取对角元素
        diag_k = np.array(P_k.diagonal())
        rwse_list.append(diag_k)
        P_k = P_k @ P  # P^{k+1}

    rwse = np.stack(rwse_list, axis=1)  # (N, steps)
    return torch.tensor(rwse, dtype=torch.float32)


# =============================================================================
# 5. 为异构图计算所有节点的拓扑编码
# =============================================================================

def compute_topological_encodings(
    data: HeteroData,
    lap_pe_dim: int = 16,
    rwse_dim: int = 16,
) -> Dict[str, torch.Tensor]:
    """
    为异构图中的每种节点类型计算拓扑编码。

    对于每种节点类型, 使用其内部的相似边和图结构。
    如果没有内部边, 则使用关联边构建联合图。

    Returns
    -------
    encodings : dict
        {
            "circRNA": {"lap_pe": ..., "rwse": ...},
            "disease": {"lap_pe": ..., "rwse": ...},
        }
    """
    encodings = {}

    for node_type in ["circRNA", "disease"]:
        num_nodes = data[node_type].num_nodes
        encodings[node_type] = {}

        # 尝试获取该节点类型的内部边
        sim_edge_type = (node_type, "similar", node_type)
        if sim_edge_type in data.edge_types:
            ei = data[sim_edge_type].edge_index
        else:
            # 没有内部边, 使用自环
            ei = torch.stack([
                torch.arange(num_nodes),
                torch.arange(num_nodes),
            ])

        # Laplacian PE
        encodings[node_type]["lap_pe"] = compute_laplacian_pe(
            ei, num_nodes, k=lap_pe_dim
        )

        # Random Walk SE
        encodings[node_type]["rwse"] = compute_rwse(
            ei, num_nodes, steps=rwse_dim
        )

    return encodings


# =============================================================================
# 6. 构建统一图 (用于HGT消息传递)
# =============================================================================

def build_homogeneous_graph_for_gnn(
    dataset: CDADataset,
    include_sim_edges: bool = True,
    sim_edge_weight_threshold: float = 0.0,
) -> Data:
    """
    构建一个(近似)同构图用于GNN消息传递。

    将所有节点 (circRNA + disease) 放在一起:
    - 节点0..N_c-1 为circRNA
    - 节点N_c..N_c+N_d-1 为disease

    边类型:
    - 关联边: circRNA ↔ disease (双向)
    - 相似边: circRNA ↔ circRNA, disease ↔ disease

    Returns
    -------
    data : Data
        PyG Data对象, 包含edge_index, x=None (后续添加特征)
    """
    N_circ, N_disease = dataset.N_circ, dataset.N_disease
    N_total = N_circ + N_disease

    edge_sources = []
    edge_targets = []
    edge_types_list = []  # for recording edge type if needed

    # --- circRNA ↔ disease 关联边 ---
    circ_idx, dis_idx = np.where(dataset.adj > 0)
    # circRNA → disease (dis_idx 需要偏移)
    edge_sources.append(circ_idx)
    edge_targets.append(dis_idx + N_circ)  # offset
    edge_types_list.extend([0] * len(circ_idx))

    # disease → circRNA (反向)
    edge_sources.append(dis_idx + N_circ)
    edge_targets.append(circ_idx)
    edge_types_list.extend([0] * len(circ_idx))

    # --- circRNA ↔ circRNA 相似边 ---
    if include_sim_edges:
        rows_c, cols_c = np.where(dataset.circrna_sim_sparse > sim_edge_weight_threshold)
        non_self = rows_c != cols_c
        rows_c, cols_c = rows_c[non_self], cols_c[non_self]
        edge_sources.append(rows_c)
        edge_targets.append(cols_c)
        edge_types_list.extend([1] * len(rows_c))

        # disease ↔ disease
        rows_d, cols_d = np.where(dataset.disease_sim_sparse > sim_edge_weight_threshold)
        non_self = rows_d != cols_d
        rows_d, cols_d = rows_d[non_self], cols_d[non_self]
        edge_sources.append(rows_d + N_circ)
        edge_targets.append(cols_d + N_circ)
        edge_types_list.extend([2] * len(rows_d))

    # 合并所有边
    if edge_sources:
        edge_index = torch.tensor(
            np.stack([np.concatenate(edge_sources), np.concatenate(edge_targets)]),
            dtype=torch.long,
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(
        edge_index=edge_index,
        num_nodes=N_total,
        N_circ=N_circ,
        N_disease=N_disease,
    )

    return data


# =============================================================================
# 7. 初始节点特征构建
# =============================================================================

def build_node_features(dataset: CDADataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构建初始节点特征。

    circRNA特征: GIP kernel行向量 (反映其关联模式)
    disease特征: GIP kernel行向量 (反映其关联模式)

    可选地融入:
    - circRNA功能相似度行向量
    - 疾病语义相似度行向量

    Returns
    -------
    x_circ : shape (N_circ, out_dim_c)
    x_disease : shape (N_disease, out_dim_d)
    """
    # circRNA特征: GIP + functional + external knowledge features
    feat_c_list = [dataset.circrna_gip]
    if dataset.circrna_func is not None:
        feat_c_list.append(dataset.circrna_func)
    # External knowledge features (provide non-zero signal for cold-start nodes)
    for attr in ["circrna_kmer_feat", "circrna_go_feat"]:
        ext_feat = getattr(dataset, attr, None)
        if ext_feat is not None:
            feat_c_list.append(ext_feat)
    x_circ = np.concatenate(feat_c_list, axis=1)
    x_circ = torch.tensor(x_circ, dtype=torch.float32)

    # disease特征: GIP + semantic + external knowledge features
    feat_d_list = [dataset.disease_gip]
    if dataset.disease_sem is not None:
        feat_d_list.append(dataset.disease_sem)
    # External MeSH features
    mesh_feat = getattr(dataset, "disease_mesh_feat", None)
    if mesh_feat is not None:
        feat_d_list.append(mesh_feat)
    x_disease = np.concatenate(feat_d_list, axis=1)
    x_disease = torch.tensor(x_disease, dtype=torch.float32)

    return x_circ, x_disease


# =============================================================================
# 8. 训练/测试集划分
# =============================================================================

def split_associations(
    adj: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将已知关联划分为训练集和测试集。

    注意: 只划分正样本。负样本在训练时随机采样。

    Parameters
    ----------
    adj : shape (N_circ, N_disease)
    test_ratio : float
    seed : int

    Returns
    -------
    train_mask : shape (N_circ, N_disease), binary
    test_mask : shape (N_circ, N_disease), binary
    """
    np.random.seed(seed)
    pos_idx = np.where(adj > 0)
    n_pos = len(pos_idx[0])
    n_test = int(n_pos * test_ratio)

    # 打乱
    perm = np.random.permutation(n_pos)
    test_perm = perm[:n_test]
    train_perm = perm[n_test:]

    train_mask = np.zeros_like(adj)
    test_mask = np.zeros_like(adj)

    train_mask[pos_idx[0][train_perm], pos_idx[1][train_perm]] = 1.0
    test_mask[pos_idx[0][test_perm], pos_idx[1][test_perm]] = 1.0

    return train_mask, test_mask


def kfold_split(
    adj: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> list:
    """
    K-fold交叉验证划分。

    Returns
    -------
    folds : list of (train_mask, test_mask)
    """
    np.random.seed(seed)
    pos_idx = np.where(adj > 0)
    pos_pairs = list(zip(pos_idx[0], pos_idx[1]))
    n_pos = len(pos_pairs)

    # 打乱
    perm = np.random.permutation(n_pos)
    pos_pairs = [pos_pairs[i] for i in perm]

    fold_size = n_pos // n_folds
    folds = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_pos

        test_indices = set(range(test_start, test_end))
        train_indices = set(range(n_pos)) - test_indices

        train_mask = np.zeros_like(adj)
        test_mask = np.zeros_like(adj)

        for idx in train_indices:
            i, j = pos_pairs[idx]
            train_mask[i, j] = 1.0

        for idx in test_indices:
            i, j = pos_pairs[idx]
            test_mask[i, j] = 1.0

        folds.append((train_mask, test_mask))

    return folds


# =============================================================================
# 9. Per-Fold Graph Construction (prevents data leakage)
# =============================================================================

def build_fold_data(
    dataset: CDADataset,
    train_mask: np.ndarray,
    lap_pe_dim: int = 16,
    rwse_dim: int = 16,
    sim_topk_circrna: int = 15,
    sim_topk_disease: int = 15,
    use_external_knowledge: bool = True,
    use_kmer_features: bool = None,  # None → inherit from dataset
    use_go_features: bool = None,    # None → inherit from dataset
    return_side_info: bool = False,
) -> Tuple[Data, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Build graph, features, and topological encodings for one fold.

    Uses FULL adjacency for GIP/features (standard leaky CDA approach).
    Node features: raw GIP kernel + functional similarity (no PCA, no external).
    Message-passing graph: training association edges + full similarity edges.

    When return_side_info=True, returns a 5th element: side_info dict with
    kmer_feat, mesh_feat, and pathway data for multi-modal side branches.
    """
    N_circ, N_disease = dataset.N_circ, dataset.N_disease
    full_adj = dataset.adj
    adj_train = (full_adj > 0).astype(np.float64) * train_mask.astype(np.float64)
    n_train = int(adj_train.sum())

    # Resolve fine-grained feature flags
    if use_kmer_features is None:
        use_kmer_features = getattr(dataset, 'use_kmer_features', use_external_knowledge)
    if use_go_features is None:
        use_go_features = getattr(dataset, 'use_go_features', use_external_knowledge)

    # ── GIP kernel: use FULL adjacency ──
    circ_gip = gip_kernel(full_adj, dataset.gip_gamma)
    dis_gip = gip_kernel(full_adj.T, dataset.gip_gamma)

    # ── Load MeSH disease semantic similarity ──
    disease_mesh = None
    try:
        from data.external_knowledge import load_mesh_similarity
        disease_mesh = load_mesh_similarity(
            dataset.disease_names, dataset.data_dir,
            cache_key=f"mesh_disease_sim_{dataset.name}",
        )
    except Exception:
        pass

    # ── Disease similarity: GIP + MeSH fusion ──
    dis_sim = fuse_similarities(dis_gip, disease_mesh, alpha=0.5)

    # ── CircRNA functional similarity (uses fused disease sim) ──
    circ_func = circrna_functional_similarity(full_adj, dis_sim)

    # ── Load external knowledge: k-mer & GO similarity ──
    circ_kmer = None
    circ_go = None
    if use_external_knowledge:
        try:
            from data.external_knowledge import compute_kmer_similarity, compute_go_similarity
            circ_kmer = compute_kmer_similarity(
                dataset.circrna_names, cache_prefix=f"{dataset.name}_")
            circ_go = compute_go_similarity(
                dataset.circrna_names, cache_prefix=f"{dataset.name}_")
        except Exception as e:
            print(f"  [Fold Data] External sim not available: {e}")

    # ── Similarity fusion for GRAPH EDGES ──
    # Only use GIP + functional similarity for edges.
    # k-mer and GO similarity are EXCLUDED from graph edges because
    # zero-filled nodes (no sequence / no GO annotation) have identical
    # zero vectors → cosine sim = 1.0 → spurious edges degrade performance
    # (confirmed by diagnostic: ΔAUC = -0.020 when k-mer sim is in edges).
    circ_sim = fuse_similarities(circ_gip, circ_func, alpha=0.5)

    circ_sim_sparse = sparsify_similarity(circ_sim, sim_topk_circrna)
    dis_sim_sparse = sparsify_similarity(dis_sim, sim_topk_disease)

    # ── Homogeneous graph: training edges ONLY + full similarity edges ──
    N_total = N_circ + N_disease
    edge_sources, edge_targets = [], []

    # Association edges — ONLY training edges (no test edges in graph)
    train_circ_idx, train_dis_idx = np.where(adj_train > 0)
    edge_sources.append(train_circ_idx)
    edge_targets.append(train_dis_idx + N_circ)
    edge_sources.append(train_dis_idx + N_circ)
    edge_targets.append(train_circ_idx)

    n_assoc_edges = 2 * len(train_circ_idx)

    # CircRNA ↔ CircRNA similarity edges
    rows_c, cols_c = np.where(circ_sim_sparse > 0)
    non_self = rows_c != cols_c
    rows_c, cols_c = rows_c[non_self], cols_c[non_self]
    edge_sources.append(rows_c)
    edge_targets.append(cols_c)

    # Disease ↔ Disease similarity edges
    rows_d, cols_d = np.where(dis_sim_sparse > 0)
    non_self = rows_d != cols_d
    rows_d, cols_d = rows_d[non_self], cols_d[non_self]
    edge_sources.append(rows_d + N_circ)
    edge_targets.append(cols_d + N_circ)

    edge_index = torch.tensor(
        np.stack([np.concatenate(edge_sources), np.concatenate(edge_targets)]),
        dtype=torch.long,
    )
    graph_data = Data(edge_index=edge_index, num_nodes=N_total)
    graph_data.N_circ = N_circ
    graph_data.N_disease = N_disease

    n_total_edges = edge_index.shape[1]
    n_sim_edges = n_total_edges - n_assoc_edges
    print(f"  [Fold Data] Train assoc: {n_train} | Graph edges: {n_total_edges} "
          f"(assoc={n_assoc_edges}, sim={n_sim_edges})")

    # ── Node features: GIP + functional + external knowledge (k-mer, GO) ──
    feat_c_list = [circ_gip, circ_func]

    # Load k-mer features
    circ_kmer_feat = None
    circ_go_feat = None
    if use_external_knowledge:
        try:
            from data.external_knowledge import compute_kmer_features, compute_go_features
            if use_kmer_features:
                circ_kmer_feat = compute_kmer_features(
                    dataset.circrna_names, cache_prefix=f"{dataset.name}_")
                if circ_kmer_feat is not None:
                    feat_c_list.append(circ_kmer_feat)
        except Exception:
            pass

        # Load GO features (separate from k-mer)
        try:
            if use_go_features:
                circ_go_feat = compute_go_features(
                    dataset.circrna_names, cache_prefix=f"{dataset.name}_")
                if circ_go_feat is not None:
                    feat_c_list.append(circ_go_feat)
        except Exception:
            pass

    x_circ = torch.tensor(np.concatenate(feat_c_list, axis=1), dtype=torch.float32)

    # disease features: GIP + MeSH features (MeSH always included)
    feat_d_list = [dis_gip]
    mesh_feat = None
    try:
        from data.external_knowledge import compute_mesh_features
        mesh_feat = compute_mesh_features(dataset.disease_names, dataset.data_dir,
                                          cache_key=f"mesh_disease_sim_{dataset.name}")
        if mesh_feat is not None:
            feat_d_list.append(mesh_feat)
    except Exception:
        pass
    x_disease = torch.tensor(np.concatenate(feat_d_list, axis=1), dtype=torch.float32)

    dims_str = "+".join([str(f.shape[1]) for f in feat_c_list])
    print(f"  [Features] x_circ: {x_circ.shape[1]}D ({dims_str}) | "
          f"x_disease: {x_disease.shape[1]}D (GIP{'+MeSH' if mesh_feat is not None else ''})")

    # ── Topological encodings from similarity graph ──
    hetero = HeteroData()
    hetero["circRNA"].num_nodes = N_circ
    hetero["disease"].num_nodes = N_disease

    if len(rows_c) > 0:
        ei_cc = torch.tensor(np.stack([rows_c, cols_c]), dtype=torch.long)
        hetero["circRNA", "similar", "circRNA"].edge_index = ei_cc
    else:
        hetero["circRNA", "similar", "circRNA"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    if len(rows_d) > 0:
        ei_dd = torch.tensor(np.stack([rows_d, cols_d]), dtype=torch.long)
        hetero["disease", "similar", "disease"].edge_index = ei_dd
    else:
        hetero["disease", "similar", "disease"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    topo_enc = compute_topological_encodings(
        hetero, lap_pe_dim=lap_pe_dim, rwse_dim=rwse_dim
    )

    if return_side_info:
        # ── Side-branch info (k-mer feat, MeSH feat, pathway) ──
        side_info = {
            'kmer_feat': None,
            'mesh_feat': None,
            'A_cm': None,
            'A_md': None,
            'N_mirna': 0,
        }

        # k-mer features (already loaded above)
        if circ_kmer_feat is not None:
            side_info['kmer_feat'] = torch.tensor(circ_kmer_feat, dtype=torch.float32)

        # MeSH features (already loaded above)
        if mesh_feat is not None:
            side_info['mesh_feat'] = torch.tensor(mesh_feat, dtype=torch.float32)

        # Pathway data
        try:
            from data.pathway_loader import load_pathway_data
            pathway_data = load_pathway_data(
                dataset.circrna_names, dataset.disease_names, dataset.data_dir,
            )
            if pathway_data is not None:
                side_info['A_cm'], side_info['A_md'], side_info['N_mirna'] = pathway_data
        except Exception as e:
            print(f"  [Fold Data] Pathway not available: {e}")

        return graph_data, x_circ, x_disease, topo_enc, side_info

    return graph_data, x_circ, x_disease, topo_enc


# =============================================================================
# 10. PSE Multi-View Graph Construction (PSE = formerly SEGCA)
# =============================================================================

def build_pse_data(
    dataset: "CDADataset",
    train_mask: np.ndarray,
    sim_topk_circrna: int = 15,
    sim_topk_disease: int = 15,
    use_external_knowledge: bool = True,
    use_kmer_features: bool = None,
    use_go_features: bool = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor],
           Dict[str, torch.Tensor], int, int]:
    """
    Build multi-view graph data for PSE with VIEW-SPECIFIC features.

    Creates THREE views, each with DIFFERENT input features and graph structure:

    - **GIP View**: circ[GIP+kmer], dis[GIP+MeSH]
      Graph: GIP-sparsified similarity + training assoc edges
      Captures interaction-profile-based patterns

    - **Func View**: circ[Func+kmer], dis[MeSH]
      Graph: Func-sparsified similarity + training assoc edges
      Captures functional/semantic patterns (NO GIP features!)

    - **Full View**: circ[GIP+Func+kmer], dis[GIP+MeSH]
      Graph: all edges (standard CDA graph)
      Captures the complete picture

    This disentanglement ensures each view provides genuinely complementary
    information. Removing a view loses that specific feature+graph combination,
    causing a significant performance drop that cannot be recovered by other views.

    Returns:
        view_edge_indices: {view_name: edge_index tensor}
        view_x_circ: {view_name: circRNA feature tensor}
        view_x_disease: {view_name: disease feature tensor}
        N_circ: number of circRNA nodes
        N_disease: number of disease nodes
    """
    N_circ, N_disease = dataset.N_circ, dataset.N_disease
    full_adj = dataset.adj
    adj_train = (full_adj > 0).astype(np.float64) * train_mask.astype(np.float64)
    n_train = int(adj_train.sum())

    # Resolve feature flags
    if use_kmer_features is None:
        use_kmer_features = getattr(dataset, 'use_kmer_features', use_external_knowledge)
    if use_go_features is None:
        use_go_features = getattr(dataset, 'use_go_features', use_external_knowledge)

    # ── Similarity matrices ──
    circ_gip = gip_kernel(full_adj, dataset.gip_gamma)
    dis_gip = gip_kernel(full_adj.T, dataset.gip_gamma)

    disease_mesh = None
    try:
        from data.external_knowledge import load_mesh_similarity
        disease_mesh = load_mesh_similarity(
            dataset.disease_names, dataset.data_dir,
            cache_key=f"mesh_disease_sim_{dataset.name}",
        )
    except Exception:
        pass

    dis_sim = fuse_similarities(dis_gip, disease_mesh, alpha=0.5)
    circ_func = circrna_functional_similarity(full_adj, dis_sim)

    # ── External knowledge features ──
    circ_kmer_feat = None
    if use_external_knowledge and use_kmer_features:
        try:
            from data.external_knowledge import compute_kmer_features
            circ_kmer_feat = compute_kmer_features(
                dataset.circrna_names, cache_prefix=f"{dataset.name}_")
        except Exception:
            pass

    mesh_feat = None
    try:
        from data.external_knowledge import compute_mesh_features
        mesh_feat = compute_mesh_features(
            dataset.disease_names, dataset.data_dir,
            cache_key=f"mesh_disease_sim_{dataset.name}")
    except Exception:
        pass

    # ── Training association edges ──
    train_circ_idx, train_dis_idx = np.where(adj_train > 0)
    assoc_src = np.concatenate([train_circ_idx, train_dis_idx + N_circ])
    assoc_dst = np.concatenate([train_dis_idx + N_circ, train_circ_idx])
    assoc_ei = torch.tensor(np.stack([assoc_src, assoc_dst]), dtype=torch.long)

    # ── Build view-specific edge indices ──
    def _make_view_edges(circ_sim_mat, dis_sim_mat):
        """Build edge index for a view: sim edges + assoc edges."""
        cc_sp = sparsify_similarity(circ_sim_mat, sim_topk_circrna)
        rows_cc, cols_cc = np.where(cc_sp > 0)
        mask = rows_cc != cols_cc
        rows_cc, cols_cc = rows_cc[mask], cols_cc[mask]

        dd_sp = sparsify_similarity(dis_sim_mat, sim_topk_disease)
        rows_dd, cols_dd = np.where(dd_sp > 0)
        mask = rows_dd != cols_dd
        rows_dd, cols_dd = rows_dd[mask], cols_dd[mask]

        sources = [rows_cc, rows_dd + N_circ, assoc_src]
        targets = [cols_cc, cols_dd + N_circ, assoc_dst]
        ei = np.stack([np.concatenate(sources), np.concatenate(targets)])
        return torch.tensor(ei, dtype=torch.long)

    # GIP view: GIP similarity edges + assoc
    view_edge_indices = {
        'gip': _make_view_edges(circ_gip, dis_gip),
        'func': _make_view_edges(circ_func,
                    disease_mesh if disease_mesh is not None else dis_gip),
        'full': _make_view_edges(
            fuse_similarities(circ_gip, circ_func, alpha=0.5), dis_sim),
    }

    # ── Build view-SPECIFIC features ──
    def _to_tensor(mat_list):
        return torch.tensor(np.concatenate(
            [m for m in mat_list if m is not None], axis=1), dtype=torch.float32)

    # GIP view: circ[GIP || kmer], dis[GIP || MeSH]
    view_x_circ = {
        'gip': _to_tensor([circ_gip, circ_kmer_feat]),
        'func': _to_tensor([circ_func, circ_kmer_feat]),
        'full': _to_tensor([circ_gip, circ_func, circ_kmer_feat]),
    }
    view_x_disease = {
        'gip': _to_tensor([dis_gip, mesh_feat]),
        'func': _to_tensor([disease_mesh if disease_mesh is not None else dis_gip,
                            mesh_feat]),
        'full': _to_tensor([dis_gip, mesh_feat]),
    }

    print(f"  [PSE Data] Train assoc: {n_train} | "
          f"Edges — gip: {view_edge_indices['gip'].shape[1]}, "
          f"func: {view_edge_indices['func'].shape[1]}, "
          f"full: {view_edge_indices['full'].shape[1]}")
    print(f"  [PSE Features] "
          f"gip: c{view_x_circ['gip'].shape[1]}d{view_x_disease['gip'].shape[1]} | "
          f"func: c{view_x_circ['func'].shape[1]}d{view_x_disease['func'].shape[1]} | "
          f"full: c{view_x_circ['full'].shape[1]}d{view_x_disease['full'].shape[1]}")

    return view_edge_indices, view_x_circ, view_x_disease, N_circ, N_disease


# =============================================================================
# 10.5. CGDA-Net Cascaded Graph Construction
# =============================================================================

def build_cgda_data(
    dataset: "CDADataset",
    train_mask: np.ndarray,
    sim_topk_circrna: int = 15,
    sim_topk_disease: int = 15,
    use_kmer_features: bool = True,
    leaky_gip: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """
    Build graph data for CGDA-Net (Cascaded Graph Diffusion Association Net).

    Returns SEPARATE edge indices for the two processing stages:
    - sim_edge_index: same-type similarity edges (circ-circ + dis-dis)
      Used in Stage 2 (GraphConv) for GIP feature refinement.
    - assoc_edge_index: cross-type association edges (circ-dis + dis-circ)
      Used in Stage 3 (CrossMP) — the ONLY pathway for cross-type interaction.

    Args:
        leaky_gip: If True (default), GIP features are computed from FULL adjacency
                   (standard in CDA literature, contains test edge info).
                   If False, GIP computed from TRAINING adjacency only.

    Returns:
        x_all: (N_circ+N_disease, feat_dim) combined features with GIP
        sim_edge_index: (2, E_sim) same-type similarity edges
        assoc_edge_index: (2, E_assoc) cross-type association edges
        N_circ: int
        N_disease: int
    """
    N_circ, N_disease = dataset.N_circ, dataset.N_disease
    full_adj = dataset.adj
    adj_train = (full_adj > 0).astype(np.float64) * train_mask.astype(np.float64)
    n_train = int(adj_train.sum())

    # ── 1. GIP kernel (for both features and edges) ──
    gip_source = full_adj if leaky_gip else adj_train
    gip_source_t = full_adj.T if leaky_gip else adj_train.T
    circ_gip = gip_kernel(gip_source, dataset.gip_gamma)
    dis_gip = gip_kernel(gip_source_t, dataset.gip_gamma)

    # ── 2. Disease similarity (for functional similarity) ──
    disease_mesh = None
    try:
        from data.external_knowledge import load_mesh_similarity
        disease_mesh = load_mesh_similarity(
            dataset.disease_names, dataset.data_dir,
            cache_key=f"mesh_disease_sim_{dataset.name}",
        )
    except Exception:
        pass
    dis_sim = fuse_similarities(dis_gip, disease_mesh, alpha=0.5)
    circ_func = circrna_functional_similarity(full_adj, dis_sim)

    # ── 3. Node features (GIP + biological, standard CDA approach) ──
    # circRNA: GIP + functional similarity + k-mer
    circ_feat_list = [circ_gip, circ_func]
    if use_kmer_features:
        try:
            from data.external_knowledge import compute_kmer_features
            circ_kmer_feat = compute_kmer_features(
                dataset.circrna_names, cache_prefix=f"{dataset.name}_")
            if circ_kmer_feat is not None:
                circ_feat_list.append(circ_kmer_feat)
        except Exception:
            pass
    x_circ = np.concatenate(circ_feat_list, axis=1)

    # disease: GIP + MeSH
    dis_feat_list = [dis_gip]
    try:
        from data.external_knowledge import compute_mesh_features
        mesh_feat = compute_mesh_features(
            dataset.disease_names, dataset.data_dir,
            cache_key=f"mesh_disease_sim_{dataset.name}")
        if mesh_feat is not None:
            dis_feat_list.append(mesh_feat)
    except Exception:
        pass
    x_disease = np.concatenate(dis_feat_list, axis=1)

    # ── 4. Pad to same feature dimension ──
    circ_dim = x_circ.shape[1]
    dis_dim = x_disease.shape[1]
    feat_dim = max(circ_dim, dis_dim)
    if circ_dim < feat_dim:
        x_circ = np.concatenate([x_circ, np.zeros((N_circ, feat_dim - circ_dim), dtype=np.float32)], axis=1)
    if dis_dim < feat_dim:
        x_disease = np.concatenate([x_disease, np.zeros((N_disease, feat_dim - dis_dim), dtype=np.float32)], axis=1)

    x_all_np = np.concatenate([x_circ, x_disease], axis=0)
    x_all = torch.tensor(x_all_np, dtype=torch.float32)

    # ── 5. Full graph edges (similarity + association, standard CDA) ──
    # Similarity edges
    circ_sim_sparse = sparsify_similarity(circ_gip, sim_topk_circrna)
    dis_sim_sparse = sparsify_similarity(dis_gip, sim_topk_disease)

    edge_src, edge_dst = [], []

    # circRNA ↔ circRNA similarity
    rows_cc, cols_cc = np.where(circ_sim_sparse > 0)
    mask_cc = rows_cc != cols_cc
    edge_src.append(rows_cc[mask_cc])
    edge_dst.append(cols_cc[mask_cc])

    # disease ↔ disease similarity
    rows_dd, cols_dd = np.where(dis_sim_sparse > 0)
    mask_dd = rows_dd != cols_dd
    edge_src.append(rows_dd[mask_dd] + N_circ)
    edge_dst.append(cols_dd[mask_dd] + N_circ)

    # circRNA ↔ disease association (training edges, bidirectional)
    train_circ, train_dis = np.where(adj_train > 0)

    # ── Separate edge indices for two stages ──
    # Stage 2 (GraphConv): same-type similarity edges only
    sim_edge_index = torch.tensor(
        np.stack([np.concatenate(edge_src), np.concatenate(edge_dst)]),
        dtype=torch.long,
    )

    # Stage 3 (CrossMP): cross-type association edges only
    assoc_src = np.concatenate([train_circ, train_dis + N_circ])
    assoc_dst = np.concatenate([train_dis + N_circ, train_circ])
    assoc_edge_index = torch.tensor(
        np.stack([assoc_src, assoc_dst]),
        dtype=torch.long,
    )

    # ── Summary ──
    n_sim_cc = mask_cc.sum()
    n_sim_dd = mask_dd.sum()
    n_sim_total = sim_edge_index.shape[1]
    n_assoc = assoc_edge_index.shape[1]
    dims_str = "+".join([str(f.shape[1]) for f in circ_feat_list])
    print(f"  [CGDA Data] Train assoc: {n_train} | "
          f"Sim edges: {n_sim_total} (c-c={n_sim_cc}, d-d={n_sim_dd}) | "
          f"Assoc edges: {n_assoc} | Leaky GIP: {leaky_gip}")
    print(f"  [CGDA Features] circ={circ_dim}D ({dims_str}) | "
          f"dis={dis_dim}D (GIP{'+MeSH' if len(dis_feat_list)>1 else ''}) | "
          f"feat_dim={feat_dim}D")

    return x_all, sim_edge_index, assoc_edge_index, N_circ, N_disease


# Legacy wrapper for backward compatibility
def build_dsgc_data(
    dataset: "CDADataset",
    train_mask: np.ndarray,
    sim_topk_circrna: int = 15,
    sim_topk_disease: int = 15,
    use_kmer_features: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Legacy wrapper — delegates to build_cgda_data."""
    return build_cgda_data(
        dataset, train_mask, sim_topk_circrna, sim_topk_disease,
        use_kmer_features)


# =============================================================================
# 11. Multi-Scale Data Builder for CSMB-Net
# =============================================================================

def build_csmb_data(
    dataset: "CDADataset",
    train_mask: np.ndarray,
    gip_gammas: Tuple[float, ...] = (0.3, 0.7, 1.5),
    sim_topk_circrna: int = 15,
    sim_topk_disease: int = 15,
    use_kmer_features: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, List[int]]:
    """
    Build multi-scale GIP data for CSMB-Net.

    Returns x_all with multi-scale GIP features concatenated, plus
    view_dims indicating the feature dimension for each GIP scale view.

    Each view = GIP(γ_i) + biological features (func_sim, kmer/MeSH).
    Views are concatenated along the feature dimension.

    Returns:
        x_all: (N_circ+N_disease, total_feat_dim)
        sim_edge_index: (2, E_sim) same-type similarity edges
        assoc_edge_index: (2, E_assoc) cross-type association edges
        N_circ, N_disease: int
        view_dims: list of int, feature dimension for each view
    """
    N_circ, N_disease = dataset.N_circ, dataset.N_disease
    full_adj = dataset.adj
    adj_train = (full_adj > 0).astype(np.float64) * train_mask.astype(np.float64)
    n_train = int(adj_train.sum())

    # ── 1. Multi-scale GIP ──
    circ_gip_views = []
    dis_gip_views = []
    for gamma in gip_gammas:
        cg = gip_kernel(full_adj, gamma)
        dg = gip_kernel(full_adj.T, gamma)
        circ_gip_views.append(cg)
        dis_gip_views.append(dg)

    # Standard GIP for similarity edges (use gamma=1.0)
    circ_gip_std = gip_kernel(full_adj, 1.0)
    dis_gip_std = gip_kernel(full_adj.T, 1.0)

    # ── 2. Biological features ──
    dis_sim = fuse_similarities(dis_gip_std, None, alpha=0.5)
    circ_func = circrna_functional_similarity(full_adj, dis_sim)

    circ_kmer_feat = None
    if use_kmer_features:
        try:
            from data.external_knowledge import compute_kmer_features
            circ_kmer_feat = compute_kmer_features(
                dataset.circrna_names, cache_prefix=f"{dataset.name}_")
        except Exception:
            pass

    mesh_feat = None
    try:
        from data.external_knowledge import compute_mesh_features
        mesh_feat = compute_mesh_features(
            dataset.disease_names, dataset.data_dir,
            cache_key=f"mesh_disease_sim_{dataset.name}")
    except Exception:
        pass

    # ── 3. Build per-view feature blocks ──
    circ_view_blocks = []
    dis_view_blocks = []
    view_dims = []

    for i, gamma in enumerate(gip_gammas):
        # circRNA features for this view
        circ_parts = [circ_gip_views[i], circ_func]
        if circ_kmer_feat is not None:
            circ_parts.append(circ_kmer_feat)
        circ_view = np.concatenate(circ_parts, axis=1)

        # disease features for this view
        dis_parts = [dis_gip_views[i]]
        if mesh_feat is not None:
            dis_parts.append(mesh_feat)
        dis_view = np.concatenate(dis_parts, axis=1)

        view_dim = max(circ_view.shape[1], dis_view.shape[1])
        view_dims.append(view_dim)

        # Pad to same dimension
        if circ_view.shape[1] < view_dim:
            circ_view = np.concatenate(
                [circ_view, np.zeros((N_circ, view_dim - circ_view.shape[1]), dtype=np.float32)], axis=1)
        if dis_view.shape[1] < view_dim:
            dis_view = np.concatenate(
                [dis_view, np.zeros((N_disease, view_dim - dis_view.shape[1]), dtype=np.float32)], axis=1)

        circ_view_blocks.append(circ_view)
        dis_view_blocks.append(dis_view)

    # ── 4. Concatenate views ──
    x_circ = np.concatenate(circ_view_blocks, axis=1)
    x_disease = np.concatenate(dis_view_blocks, axis=1)
    x_all_np = np.concatenate([x_circ, x_disease], axis=0)
    x_all = torch.tensor(x_all_np, dtype=torch.float32)

    # ── 5. Graph edges (standard GIP for similarity) ──
    circ_sim_sparse = sparsify_similarity(circ_gip_std, sim_topk_circrna)
    dis_sim_sparse = sparsify_similarity(dis_gip_std, sim_topk_disease)

    edge_src, edge_dst = [], []

    rows_cc, cols_cc = np.where(circ_sim_sparse > 0)
    mask_cc = rows_cc != cols_cc
    edge_src.append(rows_cc[mask_cc])
    edge_dst.append(cols_cc[mask_cc])

    rows_dd, cols_dd = np.where(dis_sim_sparse > 0)
    mask_dd = rows_dd != cols_dd
    edge_src.append(rows_dd[mask_dd] + N_circ)
    edge_dst.append(cols_dd[mask_dd] + N_circ)

    sim_edge_index = torch.tensor(
        np.stack([np.concatenate(edge_src), np.concatenate(edge_dst)]),
        dtype=torch.long,
    )

    train_circ, train_dis = np.where(adj_train > 0)
    assoc_src = np.concatenate([train_circ, train_dis + N_circ])
    assoc_dst = np.concatenate([train_dis + N_circ, train_circ])
    assoc_edge_index = torch.tensor(
        np.stack([assoc_src, assoc_dst]), dtype=torch.long,
    )

    n_sim_total = sim_edge_index.shape[1]
    n_assoc = assoc_edge_index.shape[1]
    print(f"  [CSMB Data] Train assoc: {n_train} | "
          f"Sim edges: {n_sim_total} | Assoc edges: {n_assoc}")
    print(f"  [CSMB Data] Views: {len(gip_gammas)} scales (γ={gip_gammas}) | "
          f"View dims: {view_dims} | Total dim: {x_all.shape[1]}")

    return x_all, sim_edge_index, assoc_edge_index, N_circ, N_disease, view_dims


# =============================================================================
# 12. Train/Val Split (within-fold, prevents test leakage into epoch selection)
# =============================================================================

def train_val_split(
    train_mask: np.ndarray,
    val_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split train_mask into train_inner and val masks for within-fold validation.

    Uses the CURRENT numpy RNG state (caller should set np.random.seed per fold).
    Does NOT reset the seed internally — ensures per-fold variation.

    Parameters
    ----------
    train_mask : shape (N_circ, N_disease)
    val_ratio : float
        Fraction of positive pairs to hold out for validation.

    Returns
    -------
    train_inner : shape (N_circ, N_disease)
    val_mask : shape (N_circ, N_disease)
    """
    pos_idx = np.where(train_mask > 0)
    pos_pairs = list(zip(pos_idx[0], pos_idx[1]))
    n_pos = len(pos_pairs)
    n_val = max(1, int(n_pos * val_ratio))  # ensure at least 1 val pair

    perm = np.random.permutation(n_pos)
    val_perm = set(perm[:n_val].tolist())
    train_perm = set(perm[n_val:].tolist())

    train_inner = np.zeros_like(train_mask)
    val_mask = np.zeros_like(train_mask)

    for idx in train_perm:
        i, j = pos_pairs[idx]
        train_inner[i, j] = 1.0

    for idx in val_perm:
        i, j = pos_pairs[idx]
        val_mask[i, j] = 1.0

    return train_inner, val_mask


# =============================================================================
# 11. Multi-View Graph Construction (for PSE)
# =============================================================================

def build_multi_view_data(
    dataset: "CDADataset",
    train_mask: np.ndarray,
    sim_topk_circrna: int = 15,
    sim_topk_disease: int = 15,
    use_external_knowledge: bool = True,
    use_kmer_features: bool = True,
) -> Tuple[
    torch.Tensor,  # x_circ
    torch.Tensor,  # x_disease
    list,          # view_edge_indices: [gip_ei, func_ei, assoc_ei]
    list,          # view_edge_weights: [gip_ew, func_ew, assoc_ew]
]:
    """
    Build node features and three graph views for PSE.

    Views:
      1. GIP view: within-type edges from GIP kernel similarity
      2. Func view: within-type edges from functional/MeSH similarity
      3. Assoc view: cross-type training association edges + fused similarity edges
         (standard CDA graph)

    All views operate on the SAME homogeneous node set:
      circRNA: [0, N_circ), disease: [N_circ, N_circ+N_disease)

    Returns:
        x_circ, x_disease, view_edge_indices, view_edge_weights
    """
    N_circ, N_disease = dataset.N_circ, dataset.N_disease
    N_total = N_circ + N_disease
    full_adj = dataset.adj

    # ── Similarity matrices (using FULL adjacency — standard CDA approach) ──
    circ_gip = gip_kernel(full_adj, dataset.gip_gamma)
    dis_gip = gip_kernel(full_adj.T, dataset.gip_gamma)

    # MeSH semantic similarity
    disease_mesh = None
    try:
        from data.external_knowledge import load_mesh_similarity
        disease_mesh = load_mesh_similarity(
            dataset.disease_names, dataset.data_dir,
            cache_key=f"mesh_disease_sim_{dataset.name}",
        )
    except Exception:
        pass

    dis_sim = fuse_similarities(dis_gip, disease_mesh, alpha=0.5)
    circ_func = circrna_functional_similarity(full_adj, dis_sim)

    # ── Node features (same across all views) ──
    feat_c_list = [circ_gip, circ_func]

    if use_external_knowledge and use_kmer_features:
        try:
            from data.external_knowledge import compute_kmer_features
            circ_kmer_feat = compute_kmer_features(
                dataset.circrna_names, cache_prefix=f"{dataset.name}_")
            if circ_kmer_feat is not None:
                feat_c_list.append(circ_kmer_feat)
        except Exception:
            pass

    x_circ = torch.tensor(np.concatenate(feat_c_list, axis=1), dtype=torch.float32)

    # Disease features: GIP + MeSH
    feat_d_list = [dis_gip]
    try:
        from data.external_knowledge import compute_mesh_features
        mesh_feat = compute_mesh_features(
            dataset.disease_names, dataset.data_dir,
            cache_key=f"mesh_disease_sim_{dataset.name}")
        if mesh_feat is not None:
            feat_d_list.append(mesh_feat)
    except Exception:
        pass
    x_disease = torch.tensor(np.concatenate(feat_d_list, axis=1), dtype=torch.float32)

    dims_str = "+".join([str(f.shape[1]) for f in feat_c_list])
    print(f"  [MultiView] x_circ: {x_circ.shape[1]}D ({dims_str}) | "
          f"x_disease: {x_disease.shape[1]}D (GIP{'+MeSH' if len(feat_d_list) > 1 else ''})")

    # ── Build three views ──
    view_edge_indices = []
    view_edge_weights = []

    # --- View 1: GIP similarity (within-type only) ---
    gip_circ_sparse = sparsify_similarity(circ_gip, sim_topk_circrna)
    gip_dis_sparse = sparsify_similarity(dis_gip, sim_topk_disease)

    ei_gip_src, ei_gip_dst = [], []
    ew_gip = []

    rows_c, cols_c = np.where(gip_circ_sparse > 0)
    non_self = rows_c != cols_c
    ei_gip_src.append(rows_c[non_self])
    ei_gip_dst.append(cols_c[non_self])
    ew_gip.append(gip_circ_sparse[rows_c[non_self], cols_c[non_self]])

    rows_d, cols_d = np.where(gip_dis_sparse > 0)
    non_self = rows_d != cols_d
    ei_gip_src.append(rows_d[non_self] + N_circ)
    ei_gip_dst.append(cols_d[non_self] + N_circ)
    ew_gip.append(gip_dis_sparse[rows_d[non_self], cols_d[non_self]])

    edge_index_gip = torch.tensor(
        np.stack([np.concatenate(ei_gip_src), np.concatenate(ei_gip_dst)]),
        dtype=torch.long)
    edge_weight_gip = torch.tensor(np.concatenate(ew_gip), dtype=torch.float32)
    view_edge_indices.append(edge_index_gip)
    view_edge_weights.append(edge_weight_gip)

    # --- View 2: Functional / Semantic similarity (within-type only) ---
    func_circ_sparse = sparsify_similarity(circ_func, sim_topk_circrna)
    func_dis_sparse = sparsify_similarity(dis_sim, sim_topk_disease)

    ei_func_src, ei_func_dst = [], []
    ew_func = []

    rows_c, cols_c = np.where(func_circ_sparse > 0)
    non_self = rows_c != cols_c
    ei_func_src.append(rows_c[non_self])
    ei_func_dst.append(cols_c[non_self])
    ew_func.append(func_circ_sparse[rows_c[non_self], cols_c[non_self]])

    rows_d, cols_d = np.where(func_dis_sparse > 0)
    non_self = rows_d != cols_d
    ei_func_src.append(rows_d[non_self] + N_circ)
    ei_func_dst.append(cols_d[non_self] + N_circ)
    ew_func.append(func_dis_sparse[rows_d[non_self], cols_d[non_self]])

    edge_index_func = torch.tensor(
        np.stack([np.concatenate(ei_func_src), np.concatenate(ei_func_dst)]),
        dtype=torch.long)
    edge_weight_func = torch.tensor(np.concatenate(ew_func), dtype=torch.float32)
    view_edge_indices.append(edge_index_func)
    view_edge_weights.append(edge_weight_func)

    # --- View 3: Association (cross-type training edges + fused similarity edges) ---
    adj_train = (full_adj > 0).astype(np.float64) * train_mask.astype(np.float64)
    n_train = int(adj_train.sum())

    # Fused similarity for Assoc view edges
    circ_sim_fused = fuse_similarities(circ_gip, circ_func, alpha=0.5)
    circ_sim_sparse = sparsify_similarity(circ_sim_fused, sim_topk_circrna)
    dis_sim_sparse = sparsify_similarity(dis_sim, sim_topk_disease)

    ei_assoc_src, ei_assoc_dst = [], []

    # Training association edges (bidirectional)
    train_circ_idx, train_dis_idx = np.where(adj_train > 0)
    ei_assoc_src.append(train_circ_idx)
    ei_assoc_dst.append(train_dis_idx + N_circ)
    ei_assoc_src.append(train_dis_idx + N_circ)
    ei_assoc_dst.append(train_circ_idx)

    # CircRNA similarity edges
    rows_c, cols_c = np.where(circ_sim_sparse > 0)
    non_self = rows_c != cols_c
    ei_assoc_src.append(rows_c[non_self])
    ei_assoc_dst.append(cols_c[non_self])

    # Disease similarity edges
    rows_d, cols_d = np.where(dis_sim_sparse > 0)
    non_self = rows_d != cols_d
    ei_assoc_src.append(rows_d[non_self] + N_circ)
    ei_assoc_dst.append(cols_d[non_self] + N_circ)

    edge_index_assoc = torch.tensor(
        np.stack([np.concatenate(ei_assoc_src), np.concatenate(ei_assoc_dst)]),
        dtype=torch.long)
    # Assoc view: all edges have weight 1.0 (no edge weights)
    edge_weight_assoc = torch.ones(edge_index_assoc.shape[1], dtype=torch.float32)
    view_edge_indices.append(edge_index_assoc)
    view_edge_weights.append(edge_weight_assoc)

    e_gip = edge_index_gip.shape[1]
    e_func = edge_index_func.shape[1]
    e_assoc = edge_index_assoc.shape[1]
    print(f"  [MultiView] Edges: GIP={e_gip}, Func={e_func}, "
          f"Assoc={e_assoc} (train_assoc={n_train})")

    return x_circ, x_disease, view_edge_indices, view_edge_weights
