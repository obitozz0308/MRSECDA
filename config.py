"""
MRSECDA: Multi-Resolution Structural Encoding for CircRNA-Disease Association prediction
全局配置文件 (精简版)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

@dataclass
class ExternalKnowledgeConfig:
    """外部生物知识库配置"""
    # Enable/disable sources
    use_mesh: bool = True          # MeSH disease semantic similarity (ready)
    use_kmer: bool = False         # circRNA sequence k-mer similarity (coverage too low)
    use_go: bool = False           # GO host-gene functional similarity (coverage too low)

    # k-mer parameters
    kmer_k: int = 4                # k-mer size (4^4 = 256 features)
    kmer_timeout: float = 5.0      # circBase API timeout (seconds)
    kmer_max_workers: int = 5      # concurrent circBase requests
    kmer_var_keep: float = 0.3     # variance threshold: keep top N% k-mers (0=all, 0.3=top 30%)

    # GO parameters
    go_obo_path: str = ""          # path to go-basic.obo (empty = auto-download)
    go_goa_path: str = ""          # path to goa_human.gaf (empty = auto-download)
    go_ic_percentile: float = 50.0  # IC pruning: keep terms above N%ile (0=all, 50=above median)

    # Fusion weights
    alpha_disease_gip: float = 0.5      # disease: GIP weight in GIP+MeSH fusion
    alpha_disease_mesh: float = 0.5     # disease: MeSH weight
    alpha_circ_internal: float = 0.5    # circRNA: (GIP+func)/2 weight
    alpha_circ_external: float = 0.5    # circRNA: (kmer+GO)/2 weight

    # Cache
    cache_dir: str = "./data/external"


@dataclass
class DataConfig:
    """数据处理配置"""
    dataset_names: Tuple[str, ...] = (
        "circR2Disease",
        "circR2Cancer",
        "circRNADisease",
        "circ2Traits",
        "circad",
    )

    phase1_dataset: str = "circR2Disease"

    sim_topk_circrna: int = 15
    sim_topk_disease: int = 15

    gip_gamma_ratio: float = 1.0

    neg_ratio: int = 2

    seed: int = 42

    # External knowledge
    external: ExternalKnowledgeConfig = field(default_factory=ExternalKnowledgeConfig)


@dataclass
class ModelConfig:
    """模型架构配置 (精简版)"""
    input_dim_circrna: int = -1    # 运行时设置
    input_dim_disease: int = -1    # 运行时设置

    # 嵌入维度
    hidden_dim: int = 96
    out_dim: int = 96

    # 拓扑编码
    lap_pe_dim: int = 16
    rwse_dim: int = 16
    rwse_steps: int = 16

    # ── HGT: Heterogeneous Graph Transformer (multi-resolution encoder) ──
    gcn_layers: int = 3                # HGT depth L (paper final: L=3)
    gcn_dropout: float = 0.3           # GCN/HGT dropout

    # ── HGT (Heterogeneous Graph Transformer) ──
    use_hgt: bool = True               # Use HGT by default (better weak-fold stability)
    hgt_heads: int = 8                 # Number of attention heads in HGT (paper final: H=8)

    num_resolutions: int = 4           # v3.3: 3 MSNE scales (micro/meso/macro) + 1 HGT

    # ── Hypergraph Convolution (v2.4) ──
    use_hypergraph: bool = False      # v2.5: Meta-path hypergraph conv (disabled — no gain vs v2.3)
    hgcn_dropout: float = 0.3         # Dropout in HypergraphConv

    # ── DropEdge Regularization (v3.0) ──
    edge_dropout: float = 0.05         # Edge dropout rate during training (0=disabled, 0.05 optimal)

    # ── Encoder Innovations (v3.1) ──
    token_dropout: float = 0.0          # TokenDrop rate: randomly mask resolution tokens (0=disabled)
    use_attn_temp: bool = False         # Per-edge-type per-head learnable attention temperature (neutral)
    consistency_weight: float = 0.0     # Consistency regularization weight (0=disabled, 0.1 rec.)
    contrastive_weight: float = 0.0     # Multi-view contrastive loss weight (0=disabled, 0.05 rec.)
    aux_task_weight: float = 0.0        # Auxiliary MTL weight (miRNA-disease edge reconstruction)

    # ── Encoder Innovations (v3.2) ──
    deep_sup_weight: float = 0.0        # Token-wise deep supervision weight (0=disabled, 0.3 rec.)
    use_sigmoid_gate: bool = False      # Independent sigmoid edge-type gating (replaces softmax competition)
    edge_dropout_anneal: bool = False   # DropEdge annealing: decay from edge_dropout→0 over training
    label_smoothing: float = 0.0        # Label smoothing (0=disabled, 0.05-0.1 rec. for high variance)
    focal_gamma: float = 2.0            # Focal loss γ (v3.2: 2.0 optimal, +0.0015 AUC vs BCE)

    # ── v3.3: Meta-Path Neighbor Encoding (MPNE) ──
    use_metapath: bool = True               # Meta-path neighbor aggregation (CMC/DMD/CMD)
    metapath_mode: str = "concat"           # "concat" (residual to hidden) or "crt" (CRT token fusion)
    metapath_num_paths: int = 3             # Meta-paths: CMC, DMD, CMD
    metapath_residual_weight: float = 0.1   # Residual weight for concat mode (0=off, 0.1 default)

    # ── v3.3: Meta-Path HGT Edge Types ──
    use_metapath_edges: bool = True         # Add CMC/DMD 2-hop meta-path edges as HGT edge types (paper final)
    metapath_edge_topk: int = 15            # Top-K meta-path neighbors per node (sparsity)
    metapath_edge_min_shared: int = 1       # Min shared miRNAs to create meta-path edge
    crt_single_token: bool = False          # Ablation: use only Token₁ (HGT), drop Token₀+Token₃

    # ── Structural Features (PSE, formerly SEGCA) ──
    use_struct_feat: bool = True       # Multi-view pairwise structural features (proven +0.02 AUC)
    use_multi_view_struct: bool = True # Multi-view (12-dim) vs single-view (4-dim)
    struct_feat_names: Tuple[str, ...] = ("jaccard", "adamic_adar")  # MRSECDA: keep only these 2 features (drop resource_alloc + inv_distance)

    # ── MRSECDA: node type scope ──
    use_lncrna: bool = False            # MRSECDA: exclude lncRNA nodes (3 types only: circ/disease/miRNA)

    # ── LLM Feature Enhancement (v2.7) ──
    use_llm: bool = False                  # Enable LLM embedding projection (DNABERT-2 + PubMedBERT)
    llm_dir: str = "data/external/llm"     # Precomputed .pt files directory
    llm_proj_dim: int = 64                 # Projection dim: Linear(768 → llm_proj_dim) before concat

    # 解码器
    decoder_hidden: Tuple[int, ...] = (96, 48)
    decoder_dropout: float = 0.3

    # 激活函数
    activation: str = "relu"
    use_layer_norm: bool = True


@dataclass
class TrainConfig:
    """训练配置 (精简版: 仅监督训练)"""
    # 监督训练
    finetune_epochs: int = 300
    finetune_lr: float = 5e-4
    finetune_weight_decay: float = 1e-4
    finetune_patience: int = 80

    # 优化器
    optimizer: str = "adamw"

    # 学习率调度
    lr_scheduler: str = "cosine"
    lr_step_size: int = 50
    lr_gamma: float = 0.5
    lr_min: float = 1e-6

    # Batch
    batch_size: int = 4096

    # 梯度裁剪
    grad_clip: float = 1.0

    # 日志
    log_interval: int = 50

    # 设备
    device: str = "cpu"

    # ═══ 以下为兼容保留, 精简版训练不使用 ═══
    pretrain_epochs: int = 200
    pretrain_lr: float = 5e-4
    pretrain_weight_decay: float = 1e-5
    pretrain_patience: int = 40


@dataclass
class EvalConfig:
    """评估配置"""
    n_folds: int = 5
    n_repeats: int = 1

    topk_list: Tuple[int, ...] = (10, 20, 50, 100)

    save_model: bool = True
    model_dir: str = "./checkpoints"
    result_dir: str = "./results"


@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    experiment_name: str = "MRSECDA"
    mode: str = "phase1"


default_config = Config()
