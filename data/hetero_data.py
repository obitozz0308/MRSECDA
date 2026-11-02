"""
Heterogeneous Graph Data Builder for circRNA-Disease Prediction.

Builds a truly heterogeneous graph with 4+ node types:
  - circRNA (GIP + k-mer + functional similarity features)
  - disease  (GIP + MeSH features)
  - miRNA    (k-mer sequence features + GIP)
  - lncRNA   (k-mer sequence features + GIP)

Edge types:
  - circRNA-disease:  known associations (training only)
  - miRNA-disease:    from HMDD / miRCancer databases
  - lncRNA-disease:   from LncRNADisease database
  - circRNA-miRNA:    from ENCORI / starBase interactions
  - circRNA-circRNA:  GIP similarity (within-type)
  - disease-disease:  GIP+MeSH similarity (within-type)
  - miRNA-miRNA:      GIP similarity (within-type)
  - lncRNA-lncRNA:    GIP similarity (within-type)
"""

from __future__ import annotations
import os
import re
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple

from data.similarity import gip_kernel, fuse_similarities, sparsify_similarity


# =============================================================================
# Fuzzy name matching for cross-database disease/circRNA name mapping
# =============================================================================

# Known disease name synonyms/subtypes (circR2Cancer → KHGAT / HMDD)
DISEASE_SYNONYMS = {
    # Subtype → parent mappings
    "bladder urothelial carcinoma": "bladder cancer",
    "castration-resistant prostate cancer": "prostate cancer",
    "esophageal squamous cell cancer": "esophageal cancer",
    "kidney clear cell carcinoma": "clear cell renal cell carcinoma",
    "renal carcinoma": "renal cell carcinoma",
    "triple-negative breast cancer": "breast cancer",
    "urothelial carcinoma of bladder": "bladder cancer",
}

# MeSH disease names → common cancer names (for HMDD v4.0 data)
MESH_TO_COMMON = {
    # Breast
    "breast neoplasms": "breast cancer",
    "breast carcinoma in situ": "breast cancer",
    "carcinoma, ductal, breast": "breast cancer",
    "carcinoma, lobular, breast": "breast cancer",
    "breast neoplasms, male": "breast cancer",
    "breast neoplasms, triple-negative": "breast cancer",
    "inflammatory breast neoplasms": "breast cancer",
    "canine mammary carcinoma": "breast cancer",
    # Lung
    "lung neoplasms": "lung cancer",
    "carcinoma, non-small-cell lung": "lung cancer",
    "carcinoma, small cell lung": "lung cancer",
    "adenocarcinoma of lung": "lung cancer",
    "adenocarcinoma, bronchiolo": "lung cancer",
    "carcinoma, bronchogenic": "lung cancer",
    "carcinoma, lewis lung": "lung cancer",
    "small cell lung carcinoma": "lung cancer",
    # Colorectal / Colon
    "colorectal neoplasms": "colorectal cancer",
    "colon neoplasms": "colon cancer",
    "rectal neoplasms": "rectal cancer",
    "adenocarcinoma, colon": "colon cancer",
    "colorectal neoplasms, hereditary nonpolyposis": "colorectal cancer",
    # Prostate
    "prostatic neoplasms": "prostate cancer",
    "prostatic neoplasms, castration-resistant": "prostate cancer",
    "carcinoma, prostatic": "prostate cancer",
    "adenocarcinoma, prostate": "prostate cancer",
    # Liver
    "liver neoplasms": "liver cancer",
    "carcinoma, hepatocellular": "liver cancer",
    "hepatocellular carcinoma": "liver cancer",
    "liver neoplasms, experimental": "liver cancer",
    "adenoma, liver cell": "liver cancer",
    "cholangiocarcinoma": "liver cancer",
    "bile duct neoplasms": "bile duct cancer",
    "biliary tract neoplasms": "bile duct cancer",
    # Gastric/Stomach
    "stomach neoplasms": "gastric cancer",
    "adenocarcinoma, gastric": "gastric cancer",
    "gastric cancer, hereditary diffuse": "gastric cancer",
    "adenocarcinoma, gastric-esophageal junction": "gastric cancer",
    # Esophageal
    "esophageal neoplasms": "esophageal cancer",
    "adenocarcinoma, esophageal": "esophageal cancer",
    "esophageal squamous cell carcinoma": "esophageal cancer",
    "adenocarcinoma of esophagus": "esophageal cancer",
    # Pancreatic
    "pancreatic neoplasms": "pancreatic cancer",
    "carcinoma, pancreatic ductal": "pancreatic cancer",
    "adenocarcinoma, pancreatic ductal": "pancreatic cancer",
    "carcinoma, islet cell": "pancreatic cancer",
    # Ovarian
    "ovarian neoplasms": "ovarian cancer",
    "carcinoma, ovarian epithelial": "ovarian cancer",
    # Cervical
    "uterine cervical neoplasms": "cervical cancer",
    "cervical intraepithelial neoplasia": "cervical cancer",
    "adenocarcinoma, cervical": "cervical cancer",
    # Endometrial/Uterine
    "endometrial neoplasms": "endometrial cancer",
    "uterine neoplasms": "uterine cancer",
    "adenocarcinoma, endometrial": "endometrial cancer",
    # Bladder
    "urinary bladder neoplasms": "bladder cancer",
    "carcinoma, bladder": "bladder cancer",
    "carcinoma, urothelial": "bladder cancer",
    # Kidney/Renal
    "kidney neoplasms": "kidney cancer",
    "carcinoma, renal cell": "renal cell carcinoma",
    "renal cell carcinoma": "renal cell carcinoma",
    "carcinoma, renal cell, clear cell": "clear cell renal cell carcinoma",
    "wilms tumor": "wilms tumor",
    # Thyroid
    "thyroid neoplasms": "thyroid cancer",
    "carcinoma, papillary thyroid": "thyroid cancer",
    "thyroid cancer, papillary": "thyroid cancer",
    "carcinoma, thyroid": "thyroid cancer",
    "carcinoma, anaplastic thyroid": "thyroid cancer",
    "carcinoma, follicular thyroid": "thyroid cancer",
    "carcinoma, medullary thyroid": "thyroid cancer",
    # Head and Neck
    "head and neck neoplasms": "head and neck cancer",
    "squamous cell carcinoma of head and neck": "head and neck cancer",
    "mouth neoplasms": "oral cancer",
    "tongue neoplasms": "tongue cancer",
    "laryngeal neoplasms": "laryngeal cancer",
    "nasopharyngeal neoplasms": "nasopharyngeal cancer",
    "nasopharyngeal carcinoma": "nasopharyngeal cancer",
    "pharyngeal neoplasms": "pharyngeal cancer",
    "oropharyngeal neoplasms": "oropharyngeal cancer",
    "hypopharyngeal neoplasms": "hypopharyngeal cancer",
    "lip neoplasms": "lip cancer",
    # Brain/CNS
    "brain neoplasms": "brain cancer",
    "glioma": "glioma",
    "glioblastoma": "glioblastoma",
    "glioblastoma multiforme": "glioblastoma",
    "astrocytoma": "astrocytoma",
    "medulloblastoma": "medulloblastoma",
    "neuroblastoma": "neuroblastoma",
    "retinoblastoma": "retinoblastoma",
    "meningioma": "meningioma",
    "brain stem neoplasms": "brain cancer",
    # Leukemia
    "leukemia": "leukemia",
    "leukemia, myeloid, acute": "leukemia",
    "leukemia, myeloid, chronic": "leukemia",
    "leukemia, lymphocytic, acute": "leukemia",
    "leukemia, lymphocytic, chronic, b-cell": "leukemia",
    "leukemia, t-cell": "leukemia",
    "leukemia, b-cell": "leukemia",
    "leukemia, myeloid": "leukemia",
    "leukemia, myelomonocytic, chronic": "leukemia",
    "leukemia, promyelocytic, acute": "leukemia",
    "leukemia, hairy cell": "leukemia",
    "leukemia, myelogenous, chronic, bcr-abl positive": "leukemia",
    "acute lymphoblastic leukemia, b-cell": "leukemia",
    "acute lymphoblastic leukemia, t-cell": "leukemia",
    "acute lymphocytic leukemia, b-cell": "leukemia",
    # Lymphoma
    "lymphoma": "lymphoma",
    "burkitt lymphoma": "lymphoma",
    "lymphoma, b-cell": "lymphoma",
    "lymphoma, t-cell": "lymphoma",
    "lymphoma, non-hodgkin": "lymphoma",
    "lymphoma, hodgkin": "lymphoma",
    "lymphoma, mantle-cell": "lymphoma",
    "lymphoma, large b-cell, diffuse": "lymphoma",
    "lymphoma, follicular": "lymphoma",
    # Skin
    "melanoma": "melanoma",
    "melanoma, cutaneous": "melanoma",
    "melanoma, experimental": "melanoma",
    "skin neoplasms": "skin cancer",
    "carcinoma, basal cell": "skin cancer",
    "carcinoma, squamous cell": "skin cancer",
    "carcinoma, merkel cell": "skin cancer",
    # Bone/Soft Tissue
    "bone neoplasms": "bone cancer",
    "osteosarcoma": "osteosarcoma",
    "sarcoma": "sarcoma",
    "sarcoma, ewing": "sarcoma",
    "sarcoma, kaposi": "sarcoma",
    "sarcoma, synovial": "sarcoma",
    "chondrosarcoma": "chondrosarcoma",
    "liposarcoma": "liposarcoma",
    "rhabdomyosarcoma": "rhabdomyosarcoma",
    "leiomyosarcoma": "leiomyosarcoma",
    "soft tissue neoplasms": "sarcoma",
    # Myeloma
    "multiple myeloma": "multiple myeloma",
    # Other cancers
    "mesothelioma": "mesothelioma",
    "mesothelioma, malignant": "mesothelioma",
    "thymoma": "thymoma",
    "thymus neoplasms": "thymus cancer",
    "testicular neoplasms": "testicular cancer",
    "penile neoplasms": "penile cancer",
    "vulvar neoplasms": "vulvar cancer",
    "vaginal neoplasms": "vaginal cancer",
    "adrenal cortex neoplasms": "adrenal cancer",
    "adrenocortical carcinoma": "adrenal cancer",
    "pheochromocytoma": "pheochromocytoma",
    "gallbladder neoplasms": "gallbladder cancer",
    "duodenal neoplasms": "duodenal cancer",
    "gastrointestinal neoplasms": "gastrointestinal cancer",
    "gastrointestinal stromal tumors": "gist",
    "peritoneal neoplasms": "peritoneal cancer",
    "anus neoplasms": "anal cancer",
    "salivary gland neoplasms": "salivary gland cancer",
    "parathyroid neoplasms": "parathyroid cancer",
    "pituitary neoplasms": "pituitary cancer",
    "adenocarcinoma": "adenocarcinoma",
    "carcinoma": "carcinoma",
    "carcinoma in situ": "carcinoma",
    "adenoid cystic carcinoma": "adenoid cystic carcinoma",
    # Cancer-related conditions
    "neoplasm metastasis": "cancer metastasis",
    "neoplasm recurrence, local": "cancer recurrence",
    "neoplastic cells, circulating": "circulating tumor cells",
    "neoplastic processes": "cancer",
    "precancerous conditions": "precancerous",
}

# Known circRNA name variants (KHGAT format → dataset format)
# Most differences are just case/hyphen, handled by normalization


def _normalize_name(name: str) -> str:
    """Normalize entity names for cross-database matching."""
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9]', '', name)  # remove non-alphanumeric
    return name


def _fuzzy_match_disease(name: str, ref_disease_set: set, ref_disease_list: list, is_hmdd: bool = False) -> Optional[str]:
    """Try to match a circR2Cancer disease name to reference database disease names.

    Strategy:
    1. Exact match (already done before calling this)
    2. Known synonym lookup
    3. Normalized (non-alphanumeric removed) match
    4. Sub-word overlap (e.g. "bladder urothelial carcinoma" ⊃ "bladder cancer")
    """
    name_lower = name.lower().strip()

    # Step 1: Known synonym
    if name_lower in DISEASE_SYNONYMS:
        synonym = DISEASE_SYNONYMS[name_lower]
        if synonym in ref_disease_set:
            return synonym

    # Step 2: If reference data uses MeSH terms, try reverse matching
    if is_hmdd:
        for mesh_term, common in MESH_TO_COMMON.items():
            if common == name_lower and mesh_term in ref_disease_set:
                return mesh_term

    # Step 3: Normalized match (strip all non-alphanumeric)
    name_norm = _normalize_name(name_lower)
    norm_map = {_normalize_name(d): d for d in ref_disease_list}
    if name_norm in norm_map:
        return norm_map[name_norm]

    # Step 4: Check if reference disease is a substring of our disease name (or vice versa)
    name_words = set(name_lower.split())
    for ref_d in ref_disease_list:
        khg_words = set(ref_d.split())
        # If KHGAT disease words are mostly contained in our disease name
        if len(khg_words & name_words) >= min(len(khg_words), len(name_words)) * 0.7:
            return ref_d

    return None


def _fuzzy_match_circrna(name: str, khgat_circ_set: set, our_circ_list: list) -> Optional[str]:
    """Try to match a KHGAT circRNA name to our dataset's circRNA names."""
    name_lower = name.lower().strip()

    # Normalized match
    name_norm = _normalize_name(name_lower)
    norm_map = {_normalize_name(c): c for c in our_circ_list}
    if name_norm in norm_map:
        return norm_map[name_norm]

    return None


# =============================================================================
# Helper: load KHGAT-style TSV data
# =============================================================================

def _load_khgat_tsv(path: str) -> List[Tuple[str, str]]:
    """Load miRNA/lncRNA association files (source, weight, target)."""
    pairs = []
    if not os.path.exists(path):
        return pairs
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                src, _, tgt = parts[0], parts[1], parts[2]
                pairs.append((src.lower().strip(), tgt.lower().strip()))
    return pairs


def _load_khgat_list(path: str) -> List[str]:
    """Load entity list file (one per line)."""
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]


# =============================================================================
# Main Heterogeneous Graph Builder
# =============================================================================

def _map_h5_mirna_to_hmdd(
    h5_mirna_name: str,
    hmdd_lookup: Dict[str, str],
) -> Optional[str]:
    """Map H5 mature miRNA name (miR-182-5p) → HMDD precursor name (hsa-mir-182)."""
    n = h5_mirna_name.lower().strip()
    n = re.sub(r'^hsa-', '', n)                         # strip species prefix
    n = n.replace('miR-', 'mir-')                        # normalize case

    # Try exact match first
    if n in hmdd_lookup:
        return hmdd_lookup[n]

    # Try without -5p/-3p mature suffix
    n_base = re.sub(r'-\dp$', '', n)
    if n_base in hmdd_lookup:
        return hmdd_lookup[n_base]

    # Try without paralog number (-1, -2)
    n_no_para = re.sub(r'-\d$', '', n_base)
    if n_no_para in hmdd_lookup:
        return hmdd_lookup[n_no_para]

    return None


def build_hetero_graph(
    dataset,                      # CDADataset instance
    train_mask: np.ndarray,
    khgat_data_dir: str = "data/CompareExperiment/AE-RF main/KHGAT/data",
    lap_pe_dim: int = 16,
    rwse_dim: int = 16,
    sim_topk: int = 15,
    use_hmdd_v4: bool = True,     # Use HMDD v4.0 for miRNA-disease data
    hmdd_data_dir: str = "data/external",  # Path to HMDD preprocessed files
    circ_mirna_h5_dir: str = "",  # Path to circRNA_miRNA h5 file dir (better circRNA coverage)
) -> Tuple[Data, torch.Tensor, Dict[str, int]]:
    """
    Build a truly heterogeneous graph with circRNA, disease, miRNA, lncRNA nodes.

    Args:
        dataset: CDADataset instance (already loaded)
        train_mask: (N_circ, N_disease) binary training mask
        khgat_data_dir: path to KHGAT data directory with miRNA/lncRNA files
        lap_pe_dim: Laplacian PE dimension
        rwse_dim: Random Walk SE dimension
        sim_topk: top-K similarity edges per node
        use_hmdd_v4: if True, use HMDD v4.0 miRNA-disease pairs (more comprehensive)
        hmdd_data_dir: path to HMDD v4.0 preprocessed data files

    Returns:
        data: PyG Data object with edge_index and node features
        x_all: (N_total, D) combined node features
        node_counts: dict with N_circ, N_disease, N_mirna, N_lncrna
    """
    N_circ = dataset.N_circ
    N_disease = dataset.N_disease
    full_adj = dataset.adj
    adj_train = (full_adj > 0).astype(np.float64) * train_mask.astype(np.float64)

    # ─── Load lncRNA data from KHGAT (HMDD doesn't have lncRNA data) ───
    lncrna_list = _load_khgat_list(os.path.join(khgat_data_dir, "lncRNA_list.txt"))
    lncrna_list = list(dict.fromkeys(lncrna_list))

    # ─── Load circRNA-miRNA interactions ───
    # Priority: 1) REAL CLIP/curated data (ENCORI + RNAInter, from parse_rna_databases.py)
    #           2) dataset-specific comprehensive pairs (seed-match + KHGAT + disease + GIP)
    #           3) KHGAT starBase v2.0 pairs (fallback)
    _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # data/.. → project root
    _ext_dir = os.path.join(_proj_root, "data", "external")
    _real_cm_path = os.path.join(_ext_dir, "circrna_mirna_real.txt")
    _ds_pairs_path = os.path.join(_ext_dir, f"{dataset.name}_circrna_mirna_pairs.txt")
    if os.path.exists(_real_cm_path):
        circ_mirna_pairs = _load_khgat_tsv(_real_cm_path)
        print(f"  [HeteroGraph] Using REAL circRNA-miRNA: {len(circ_mirna_pairs)} pairs "
              f"from circrna_mirna_real.txt")
    elif os.path.exists(_ds_pairs_path):
        circ_mirna_pairs = _load_khgat_tsv(_ds_pairs_path)
        print(f"  [HeteroGraph] Loaded {len(circ_mirna_pairs)} circRNA-miRNA pairs "
              f"from {dataset.name}_circrna_mirna_pairs.txt (comprehensive)")
    else:
        circ_mirna_pairs = _load_khgat_tsv(os.path.join(khgat_data_dir, "circRNA_miRNA.txt"))
        print(f"  [HeteroGraph] Using KHGAT circRNA-miRNA: {len(circ_mirna_pairs)} pairs")

    # ─── (H5 supplement loaded after miRNA-disease data, see below) ───

    # ─── Load miRNA-disease data: HMDD v4.0 (preferred) or KHGAT (fallback) ───
    if use_hmdd_v4:
        # Use HMDD v4.0 miRNA-disease pairs — prefer FULL (cancer + non-cancer), fallback cancer-only
        hmdd_full_path = os.path.join(hmdd_data_dir, "hmdd_v4_all_mirna_disease.txt")
        hmdd_cancer_path = os.path.join(hmdd_data_dir, "hmdd_v4_cancer_mirna_disease.txt")
        hmdd_path = hmdd_full_path if os.path.exists(hmdd_full_path) else hmdd_cancer_path
        hmdd_label = "FULL" if hmdd_path == hmdd_full_path else "cancer"
        if os.path.exists(hmdd_path):
            mirna_disease_pairs = _load_khgat_tsv(hmdd_path)
            # Derive miRNA list from the pairs (HMDD uses mature names matching KHGAT)
            mirna_list = list(dict.fromkeys([p[0] for p in mirna_disease_pairs]))
            print(f"  [HeteroGraph] Using HMDD v4.0 {hmdd_label} miRNA-disease: "
                  f"{len(mirna_disease_pairs)} pairs, {len(mirna_list)} miRNAs")
        else:
            print(f"  [HeteroGraph] WARNING: HMDD v4.0 file not found at {hmdd_cancer_path}")
            print(f"  [HeteroGraph] Run 'python data/preprocess_hmdd_v4.py' first.")
            print(f"  [HeteroGraph] Falling back to KHGAT miRNA data.")
            use_hmdd_v4 = False

    if not use_hmdd_v4:
        # Fallback: use original KHGAT miRNA data
        mirna_list = _load_khgat_list(os.path.join(khgat_data_dir, "miRNA_list.txt"))
        mirna_list = list(dict.fromkeys(mirna_list))
        mirna_disease_pairs = _load_khgat_tsv(os.path.join(khgat_data_dir, "miRNA_cancer.txt"))
        print(f"  [HeteroGraph] Using KHGAT miRNA-disease data: "
              f"{len(mirna_disease_pairs)} pairs, {len(mirna_list)} miRNAs")

    # Load lncRNA-disease pairs: prefer LncRNADisease v3.0 (non-cancer), fallback KHGAT
    _lncd_path = os.path.join(_ext_dir, "lncrnadisease_lncrna_disease.txt")
    if os.path.exists(_lncd_path):
        lncrna_disease_pairs = _load_khgat_tsv(_lncd_path)
        print(f"  [HeteroGraph] Using LncRNADisease v3.0 lncRNA-disease: "
              f"{len(lncrna_disease_pairs)} pairs")
    else:
        lncrna_disease_pairs = _load_khgat_tsv(os.path.join(khgat_data_dir, "lncRNA_cancer.txt"))
        print(f"  [HeteroGraph] Using KHGAT lncRNA-disease: {len(lncrna_disease_pairs)} pairs")

    # ─── Load circRNA-lncRNA ceRNA pairs (inferred via shared miRNA, parse_rna_databases.py) ───
    _clc_path = os.path.join(_ext_dir, "circrna_lncrna_ceRNA.txt")
    if os.path.exists(_clc_path):
        circ_lncrna_pairs = _load_khgat_tsv(_clc_path)   # (circRNA, lncRNA)
        print(f"  [HeteroGraph] Using ceRNA circRNA-lncRNA: {len(circ_lncrna_pairs)} pairs")
    else:
        circ_lncrna_pairs = []
        print(f"  [HeteroGraph] No ceRNA circRNA-lncRNA file (run parse_rna_databases.py first)")

    # Augment lncRNA list with lncRNAs from LncRNADisease + ceRNA pairs
    _lnc_extra = [p[0] for p in lncrna_disease_pairs] + [p[1] for p in circ_lncrna_pairs]
    lncrna_list = list(dict.fromkeys(lncrna_list + _lnc_extra))

    # ─── Supplement: Load circRNA-miRNA from H5 file (better circRNA coverage) ───
    h5_circ_mirna_extra = 0
    if circ_mirna_h5_dir:
        circ_mirna_h5_path = os.path.join(circ_mirna_h5_dir, "circRNA_miRNA.h5")
        mirna_xlsx_path = os.path.join(circ_mirna_h5_dir, "miRNA_list.xlsx")
        if os.path.exists(circ_mirna_h5_path) and os.path.exists(mirna_xlsx_path):
            try:
                import h5py
                import pandas as pd
                with h5py.File(circ_mirna_h5_path, 'r') as f:
                    h5_mat = f['infor'][:]
                mirna_df = pd.read_excel(mirna_xlsx_path, header=None)
                h5_mirna_names = [str(v).strip() for v in mirna_df.iloc[:, 0].tolist()]

                # Build HMDD miRNA name → canonical lookup for H5 mapping
                hmdd_lookup = {}
                for m_name, _ in mirna_disease_pairs:
                    c = m_name.lower().replace('hsa-', '').replace('miR-', 'mir-')
                    if c not in hmdd_lookup:
                        hmdd_lookup[c] = m_name
                    c_base = re.sub(r'-\dp$', '', c)
                    if c_base not in hmdd_lookup:
                        hmdd_lookup[c_base] = m_name
                    c_np = re.sub(r'-\d$', '', c_base)
                    if c_np not in hmdd_lookup:
                        hmdd_lookup[c_np] = m_name

                # Build local circRNA name list (dataset-aligned, same order as h5 matrix)
                if dataset.circrna_names is not None:
                    _h5_circ_names = [str(c).lower().strip() for c in dataset.circrna_names]
                else:
                    _h5_circ_names = [f"circrna_{i}" for i in range(h5_mat.shape[0])]

                # Convert H5 matrix → (miRNA_HMDD_name, circRNA_name) pairs
                existing_pairs = set(circ_mirna_pairs)
                rows, cols = np.where(h5_mat > 0)
                for r, c in zip(rows, cols):
                    if c < len(h5_mirna_names):
                        hmdd_name = _map_h5_mirna_to_hmdd(
                            h5_mirna_names[c], hmdd_lookup)
                        if hmdd_name and r < len(_h5_circ_names):
                            pair = (hmdd_name, _h5_circ_names[r])
                            if pair not in existing_pairs:
                                circ_mirna_pairs.append(pair)
                                existing_pairs.add(pair)
                                h5_circ_mirna_extra += 1

                if h5_circ_mirna_extra > 0:
                    print(f"  [HeteroGraph] H5 supplement: +{h5_circ_mirna_extra} "
                          f"circRNA-miRNA pairs from local h5")
            except Exception as e:
                print(f"  [HeteroGraph] WARNING: Failed to load H5 circRNA-miRNA: {e}")

    # ─── Filter miRNA/lncRNA: only keep those with enough disease associations ───
    MIN_ASSOC_MIRNA = 3   # miRNAs: quality > quantity. Adding more miRNAs dilutes GCN signal
    MIN_ASSOC_LNCRNA = 3  # lncRNAs: keep strict threshold (better data quality)

    # Count disease associations per miRNA
    mirna_dis_count = {}
    for mirna_name, disease_name in mirna_disease_pairs:
        mirna_dis_count[mirna_name] = mirna_dis_count.get(mirna_name, 0) + 1
    # Filter miRNA list by disease association count
    mirna_list_filtered = [m for m in mirna_list
                           if mirna_dis_count.get(m, 0) >= MIN_ASSOC_MIRNA]

    # Count disease associations per lncRNA
    lncrna_dis_count = {}
    for lncrna_name, disease_name in lncrna_disease_pairs:
        lncrna_dis_count[lncrna_name] = lncrna_dis_count.get(lncrna_name, 0) + 1
    lncrna_list_filtered = [l for l in lncrna_list
                            if lncrna_dis_count.get(l, 0) >= MIN_ASSOC_LNCRNA]

    # Track which miRNAs have circRNA interactions (for edge building, not for filtering)
    mirna_with_circ = set()
    for mirna_name, _ in circ_mirna_pairs:
        mirna_with_circ.add(mirna_name)

    n_mirna_total = len(mirna_list_filtered)
    n_mirna_with_circ = sum(1 for m in mirna_list_filtered if m in mirna_with_circ)
    n_mirna_no_circ = n_mirna_total - n_mirna_with_circ

    print(f"  [HeteroGraph] miRNA: {len(mirna_list)}→{n_mirna_total} "
          f"(min {MIN_ASSOC_MIRNA} disease assoc; {n_mirna_with_circ} with circRNA, {n_mirna_no_circ} without)")

    # IMPORTANT: Only include miRNAs that have circRNA interactions.
    # miRNAs without circRNA interactions can only reach circRNAs via 2-hop paths
    # (miRNA→disease→circRNA), which dilutes the GCN signal. Keeping them degrades AUC.
    mirna_list_filtered = [m for m in mirna_list_filtered if m in mirna_with_circ]

    n_mirna_kept = len(mirna_list_filtered)
    print(f"  [HeteroGraph] Keeping {n_mirna_kept} miRNAs with circRNA interactions "
          f"(dropped {n_mirna_no_circ} without circRNA)")

    print(f"  [HeteroGraph] lncRNA: {len(lncrna_list)}→{len(lncrna_list_filtered)} "
          f"(min {MIN_ASSOC_LNCRNA} disease assoc)")

    mirna_list = mirna_list_filtered
    lncrna_list = lncrna_list_filtered

    # Build index mappings
    mirna_to_idx = {m: i for i, m in enumerate(mirna_list)}
    lncrna_to_idx = {l: i for i, l in enumerate(lncrna_list)}

    # Disease name mapping (dataset disease names → lowercase)
    disease_names_lower = []
    if dataset.disease_names is not None:
        disease_names_lower = [str(d).lower().strip() for d in dataset.disease_names]
    else:
        disease_names_lower = [f"disease_{i}" for i in range(N_disease)]

    disease_to_idx = {d: i for i, d in enumerate(disease_names_lower)}

    # CircRNA name mapping
    circ_names_lower = []
    if dataset.circrna_names is not None:
        circ_names_lower = [str(c).lower().strip() for c in dataset.circrna_names]
    else:
        circ_names_lower = [f"circrna_{i}" for i in range(N_circ)]

    N_mirna = len(mirna_list)
    N_lncrna = len(lncrna_list)
    N_total = N_circ + N_disease + N_mirna + N_lncrna

    print(f"  [HeteroGraph] circRNA={N_circ}, disease={N_disease}, "
          f"miRNA={N_mirna}, lncRNA={N_lncrna} → total={N_total}")

    # ─── Compute GIP kernels for all node types ───
    # circRNA GIP (from circRNA-disease associations)
    circ_gip = gip_kernel(full_adj, dataset.gip_gamma)
    # disease GIP (from disease-circRNA associations)
    dis_gip = gip_kernel(full_adj.T, dataset.gip_gamma)

    # Build miRNA-disease association matrix for miRNA GIP
    # Use fuzzy disease name matching for unmatched diseases
    ref_disease_set = set()
    ref_disease_list = []
    for _, d in mirna_disease_pairs:
        if d.lower().strip() not in ref_disease_set:
            ref_disease_set.add(d.lower().strip())
            ref_disease_list.append(d.lower().strip())
    for _, d in lncrna_disease_pairs:
        if d.lower().strip() not in ref_disease_set:
            ref_disease_set.add(d.lower().strip())
            ref_disease_list.append(d.lower().strip())

    # Build fuzzy disease map (bidirectional)
    # Forward: circR2Cancer disease → reference disease (for building the map)
    # Reverse: reference disease → circR2Cancer disease (for edge construction)
    disease_fuzzy_map = {}      # circR2Cancer → reference
    reverse_disease_map = {}    # reference → circR2Cancer
    fuzzy_match_count = 0
    for dis_name in disease_names_lower:
        if dis_name in disease_to_idx:
            # Check if this disease name exists in reference data
            if dis_name in ref_disease_set:
                disease_fuzzy_map[dis_name] = dis_name
                reverse_disease_map[dis_name] = dis_name
            else:
                # Try fuzzy match
                matched = _fuzzy_match_disease(dis_name, ref_disease_set, ref_disease_list, is_hmdd=use_hmdd_v4)
                if matched:
                    disease_fuzzy_map[dis_name] = matched
                    reverse_disease_map[matched] = dis_name
                    fuzzy_match_count += 1
    if fuzzy_match_count > 0:
        print(f"  [HeteroGraph] Fuzzy-matched {fuzzy_match_count} disease names to reference DB")

    mirna_dis_adj = np.zeros((N_mirna, N_disease), dtype=np.float64)
    mirna_dis_count = 0
    for mirna_name, disease_name in mirna_disease_pairs:
        if mirna_name not in mirna_to_idx:
            continue
        # Find matching dataset disease via reverse fuzzy map (reference → circR2Cancer)
        dis_lower = disease_name.lower().strip()
        matched_dis = reverse_disease_map.get(dis_lower)
        if matched_dis is None and dis_lower in disease_to_idx:
            # Direct match (reference disease name = circR2Cancer disease name)
            matched_dis = dis_lower
        if matched_dis is not None and matched_dis in disease_to_idx:
            mi = mirna_to_idx[mirna_name]
            di = disease_to_idx[matched_dis]
            mirna_dis_adj[mi, di] = 1.0
            mirna_dis_count += 1
    print(f"  [HeteroGraph] miRNA-disease edges mapped: {mirna_dis_count}/{len(mirna_disease_pairs)}")

    # Build lncRNA-disease association matrix for lncRNA GIP
    lncrna_dis_adj = np.zeros((N_lncrna, N_disease), dtype=np.float64)
    lncrna_dis_count = 0
    for lncrna_name, disease_name in lncrna_disease_pairs:
        if lncrna_name not in lncrna_to_idx:
            continue
        # Find matching dataset disease via reverse fuzzy map (reference → circR2Cancer)
        dis_lower = disease_name.lower().strip()
        matched_dis = reverse_disease_map.get(dis_lower)
        if matched_dis is None and dis_lower in disease_to_idx:
            matched_dis = dis_lower
        if matched_dis is not None and matched_dis in disease_to_idx:
            li = lncrna_to_idx[lncrna_name]
            di = disease_to_idx[matched_dis]
            lncrna_dis_adj[li, di] = 1.0
            lncrna_dis_count += 1
    print(f"  [HeteroGraph] lncRNA-disease edges mapped: {lncrna_dis_count}/{len(lncrna_disease_pairs)}")

    # Build circRNA-miRNA interaction matrix (with fuzzy circRNA name matching)
    khgat_circ_set = set()
    for mirna_name, circ_name in circ_mirna_pairs:
        khgat_circ_set.add(circ_name.lower().strip())

    # Build exact + fuzzy circRNA name map
    circ_fuzzy_map = {}  # KHGAT circRNA name → our circRNA name
    circ_fuzzy_count = 0
    our_circ_set = set(circ_names_lower)
    for khg_circ in khgat_circ_set:
        if khg_circ in our_circ_set:
            circ_fuzzy_map[khg_circ] = khg_circ
        else:
            matched = _fuzzy_match_circrna(khg_circ, khgat_circ_set, circ_names_lower)
            if matched:
                circ_fuzzy_map[khg_circ] = matched
                circ_fuzzy_count += 1
    if circ_fuzzy_count > 0:
        print(f"  [HeteroGraph] Fuzzy-matched {circ_fuzzy_count} circRNA names to KHGAT")

    circ_mirna_adj = np.zeros((N_circ, N_mirna), dtype=np.float64)
    circ_mirna_count = 0
    for mirna_name, circ_name in circ_mirna_pairs:
        circ_name_lower = circ_name.lower().strip()
        matched_circ = circ_fuzzy_map.get(circ_name_lower)
        if matched_circ is None:
            # Fallback: direct lookup
            if circ_name_lower in our_circ_set:
                matched_circ = circ_name_lower
        if matched_circ is not None and mirna_name in mirna_to_idx:
            try:
                ci = circ_names_lower.index(matched_circ)
            except ValueError:
                continue
            mi = mirna_to_idx[mirna_name]
            circ_mirna_adj[ci, mi] = 1.0
            circ_mirna_count += 1
    print(f"  [HeteroGraph] circRNA-miRNA edges mapped: {circ_mirna_count}/{len(circ_mirna_pairs)}")

    # Build circRNA-lncRNA interaction matrix (ceRNA; fuzzy circRNA + exact lncRNA name match)
    circ_lncrna_adj = np.zeros((N_circ, N_lncrna), dtype=np.float64)
    circ_lncrna_count = 0
    for circ_name, lncrna_name in circ_lncrna_pairs:
        circ_name_lower = circ_name.lower().strip()
        matched_circ = circ_fuzzy_map.get(circ_name_lower)
        if matched_circ is None:
            if circ_name_lower in our_circ_set:
                matched_circ = circ_name_lower
        lnc_lower = lncrna_name.lower().strip()
        if matched_circ is not None and lnc_lower in lncrna_to_idx:
            try:
                ci = circ_names_lower.index(matched_circ)
            except ValueError:
                continue
            li = lncrna_to_idx[lnc_lower]
            circ_lncrna_adj[ci, li] = 1.0
            circ_lncrna_count += 1
    print(f"  [HeteroGraph] circRNA-lncRNA edges mapped: {circ_lncrna_count}/{len(circ_lncrna_pairs)}")

    # miRNA GIP from miRNA-disease association matrix
    if mirna_dis_adj.sum() > 0:
        mirna_gip = gip_kernel(mirna_dis_adj, dataset.gip_gamma)  # (N_mirna, N_mirna)
    else:
        mirna_gip = np.eye(N_mirna, dtype=np.float64)

    # lncRNA GIP from lncRNA-disease association matrix
    if lncrna_dis_adj.sum() > 0:
        lncrna_gip = gip_kernel(lncrna_dis_adj, dataset.gip_gamma)
    else:
        lncrna_gip = np.eye(N_lncrna, dtype=np.float64)

    # ─── Load external features (MeSH, k-mer, GO) ───
    # MeSH for disease
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
    circ_func = gip_kernel(full_adj, dataset.gip_gamma)  # functional via GIP for now
    # Actually use proper functional similarity
    try:
        from data.similarity import circrna_functional_similarity
        circ_func = circrna_functional_similarity(full_adj, dis_sim)
    except Exception:
        pass

    # Load k-mer features for circRNA
    circ_kmer_feat = None
    try:
        from data.external_knowledge import compute_kmer_features
        circ_kmer_feat = compute_kmer_features(
            dataset.circrna_names, cache_prefix=f"{dataset.name}_")
    except Exception:
        pass

    # Load MeSH features for disease
    mesh_feat = None
    try:
        from data.external_knowledge import compute_mesh_features
        mesh_feat = compute_mesh_features(dataset.disease_names, dataset.data_dir,
                                          cache_key=f"mesh_disease_sim_{dataset.name}")
    except Exception:
        pass

    # Load miRNA k-mer features from miRBase sequences
    mirna_kmer_feat = None
    mirna_kmer_sim = None
    try:
        from data.external_knowledge import (compute_mirna_kmer_features,
                                             compute_mirna_kmer_similarity)
        mirna_kmer_feat = compute_mirna_kmer_features(
            mirna_list, cache_key=f"mirna_kmer_features_{dataset.name}")
        mirna_kmer_sim = compute_mirna_kmer_similarity(
            mirna_list, cache_key=f"mirna_kmer_sim_{dataset.name}")
    except Exception as e:
        pass

    # ─── Node Features (with SVD reduction to avoid dimension explosion) ───
    # GIP features are N×N matrices — we reduce via SVD to a fixed dim
    SVD_DIM = 128  # Fixed latent dimension for GIP features

    def reduce_svd(matrix, target_dim=SVD_DIM):
        """Reduce a (N, D) matrix to (N, target_dim) via truncated SVD."""
        if matrix.shape[1] <= target_dim:
            return matrix
        n = min(matrix.shape[0], matrix.shape[1])
        k = min(target_dim, n - 1, matrix.shape[1] - 1)
        k = max(k, 1)
        try:
            U, S, Vt = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
            # Use top-k singular vectors scaled by singular values
            reduced = U[:, :k] * S[:k]
            if reduced.shape[1] < target_dim:
                pad = np.zeros((reduced.shape[0], target_dim - reduced.shape[1]))
                reduced = np.concatenate([reduced, pad], axis=1)
            return reduced.astype(np.float64)
        except Exception:
            # Fallback: random projection
            rng = np.random.RandomState(42)
            proj = rng.randn(matrix.shape[1], target_dim) / np.sqrt(target_dim)
            return (matrix @ proj).astype(np.float64)

    # circRNA features: GIP(SVD) + func_sim(SVD) + k-mer
    feat_c_list = [reduce_svd(circ_gip)]
    if circ_func is not None:
        feat_c_list.append(reduce_svd(circ_func))
    if circ_kmer_feat is not None:
        feat_c_list.append(circ_kmer_feat)
    x_circ = np.concatenate(feat_c_list, axis=1)

    # disease features: GIP(SVD) + MeSH
    feat_d_list = [reduce_svd(dis_gip)]
    if mesh_feat is not None:
        feat_d_list.append(mesh_feat)
    x_disease = np.concatenate(feat_d_list, axis=1)

    # miRNA features: GIP(SVD) from miRNA-disease matrix + k-mer sequence features
    mirna_has_edges = mirna_dis_adj.sum() >= 5
    feat_m_list = [reduce_svd(mirna_gip)]  # GIP 128D
    if mirna_kmer_feat is not None and mirna_kmer_feat.shape[0] == N_mirna:
        # Reduce k-mer to 128D for consistency
        kmer_reduced = reduce_svd(mirna_kmer_feat)
        feat_m_list.append(kmer_reduced)
    x_mirna = np.concatenate(feat_m_list, axis=1)
    # Add random positional signal if too few edges
    if not mirna_has_edges:
        rng = np.random.RandomState(42)
        pos_signal = rng.randn(N_mirna, 32).astype(np.float64) * 0.01
        x_mirna = np.concatenate([x_mirna, pos_signal], axis=1)
        print(f"  [HeteroGraph] miRNA: few disease edges ({int(mirna_dis_adj.sum())}), "
              f"adding pos signal")

    # lncRNA features: GIP(SVD) from lncRNA-disease matrix
    lncrna_has_edges = lncrna_dis_adj.sum() >= 5
    x_lncrna = reduce_svd(lncrna_gip)
    if not lncrna_has_edges:
        rng = np.random.RandomState(42)
        pos_signal = rng.randn(N_lncrna, 32).astype(np.float64) * 0.01
        x_lncrna = np.concatenate([x_lncrna, pos_signal], axis=1)

    # Unify dimensions: pad to max across all types
    D_max = max(x_circ.shape[1], x_disease.shape[1], x_mirna.shape[1], x_lncrna.shape[1])

    def pad_features(x, target_dim):
        if x.shape[1] < target_dim:
            pad = np.zeros((x.shape[0], target_dim - x.shape[1]), dtype=np.float64)
            return np.concatenate([x, pad], axis=1)
        return x

    x_circ = pad_features(x_circ, D_max)
    x_disease = pad_features(x_disease, D_max)
    x_mirna = pad_features(x_mirna, D_max)
    x_lncrna = pad_features(x_lncrna, D_max)

    x_all = np.concatenate([x_circ, x_disease, x_mirna, x_lncrna], axis=0)
    x_all = torch.tensor(x_all, dtype=torch.float32)

    feat_dim = D_max
    feat_parts = (x_circ.shape[1], x_disease.shape[1], x_mirna.shape[1], x_lncrna.shape[1])
    print(f"  [HeteroGraph] Feature dim: {feat_dim}D (circ={feat_parts[0]}, dis={feat_parts[1]}, "
          f"mir={feat_parts[2]}, lnc={feat_parts[3]})")

    # ─── Build Graph Edges (with per-type grouping for heterogeneous GNN) ───
    offset_disease = N_circ
    offset_mirna = N_circ + N_disease
    offset_lncrna = N_circ + N_disease + N_mirna

    # We group edges into 3 meta-types for edge-type-specific GCN:
    #   TYPE 0 — within-type similarity: circ-circ, dis-dis, mir-mir, lnc-lnc
    #   TYPE 1 — circRNA-disease training assoc (the prediction target)
    #   TYPE 2 — auxiliary cross-type: mir-dis, lnc-dis, circ-mir

    edge_sources = {0: [], 1: [], 2: []}  # per-type
    edge_targets = {0: [], 1: [], 2: []}

    # ── TYPE 1: circRNA-disease association edges (training only) ──
    train_circ, train_dis = np.where(adj_train > 0)
    edge_sources[1].append(train_circ)
    edge_targets[1].append(train_dis + offset_disease)
    edge_sources[1].append(train_dis + offset_disease)
    edge_targets[1].append(train_circ)
    n_assoc = 2 * len(train_circ)

    # ── TYPE 0: circRNA-circRNA similarity edges ──
    circ_sim_sparse = sparsify_similarity(
        fuse_similarities(circ_gip, circ_func, alpha=0.5), sim_topk)
    rows_c, cols_c = np.where(circ_sim_sparse > 0)
    non_self = rows_c != cols_c
    edge_sources[0].append(rows_c[non_self])
    edge_targets[0].append(cols_c[non_self])

    # ── TYPE 0: disease-disease similarity edges ──
    dis_sim_sparse = sparsify_similarity(dis_sim, sim_topk)
    rows_d, cols_d = np.where(dis_sim_sparse > 0)
    non_self = rows_d != cols_d
    edge_sources[0].append(rows_d[non_self] + offset_disease)
    edge_targets[0].append(cols_d[non_self] + offset_disease)

    # ── TYPE 2: miRNA-disease association edges ──
    mirna_row, dis_col = np.where(mirna_dis_adj > 0)
    if len(mirna_row) > 0:
        edge_sources[2].append(mirna_row + offset_mirna)
        edge_targets[2].append(dis_col + offset_disease)
        edge_sources[2].append(dis_col + offset_disease)
        edge_targets[2].append(mirna_row + offset_mirna)
    n_mirna_dis = 2 * len(mirna_row)

    # ── TYPE 2: lncRNA-disease association edges ──
    lncrna_row, dis_col2 = np.where(lncrna_dis_adj > 0)
    if len(lncrna_row) > 0:
        edge_sources[2].append(lncrna_row + offset_lncrna)
        edge_targets[2].append(dis_col2 + offset_disease)
        edge_sources[2].append(dis_col2 + offset_disease)
        edge_targets[2].append(lncrna_row + offset_lncrna)
    n_lncrna_dis = 2 * len(lncrna_row)

    # ── TYPE 2: circRNA-miRNA interaction edges ──
    cm_row, cm_col = np.where(circ_mirna_adj > 0)
    if len(cm_row) > 0:
        edge_sources[2].append(cm_row)                      # circRNA indices
        edge_targets[2].append(cm_col + offset_mirna)       # miRNA indices
        edge_sources[2].append(cm_col + offset_mirna)
        edge_targets[2].append(cm_row)
    n_circ_mirna = 2 * len(cm_row)

    # ── TYPE 2: circRNA-lncRNA ceRNA edges ──
    cl_row, cl_col = np.where(circ_lncrna_adj > 0)
    if len(cl_row) > 0:
        edge_sources[2].append(cl_row)                      # circRNA indices
        edge_targets[2].append(cl_col + offset_lncrna)      # lncRNA indices
        edge_sources[2].append(cl_col + offset_lncrna)
        edge_targets[2].append(cl_row)
    n_circ_lncrna = 2 * len(cl_row)

    # ── TYPE 0: miRNA-miRNA similarity edges ──
    mirna_sim = mirna_gip
    if mirna_kmer_sim is not None and mirna_kmer_sim.shape == mirna_gip.shape:
        # Fuse GIP + k-mer similarity for richer miRNA edges
        mirna_sim = fuse_similarities(mirna_gip, mirna_kmer_sim, alpha=0.5)
    mirna_sim_sparse = sparsify_similarity(mirna_sim, min(sim_topk, N_mirna - 1))
    rows_m, cols_m = np.where(mirna_sim_sparse > 0)
    non_self_m = rows_m != cols_m
    edge_sources[0].append(rows_m[non_self_m] + offset_mirna)
    edge_targets[0].append(cols_m[non_self_m] + offset_mirna)

    # ── TYPE 0: lncRNA-lncRNA similarity edges ──
    lncrna_sim_sparse = sparsify_similarity(lncrna_gip, min(sim_topk, N_lncrna - 1))
    rows_l, cols_l = np.where(lncrna_sim_sparse > 0)
    non_self_l = rows_l != cols_l
    edge_sources[0].append(rows_l[non_self_l] + offset_lncrna)
    edge_targets[0].append(cols_l[non_self_l] + offset_lncrna)

    # ─── Assemble edge_index per type ───
    edge_index_by_type = {}
    n_edge_by_type = {}
    for t in [0, 1, 2]:
        if edge_sources[t]:
            ei = torch.tensor(
                np.stack([np.concatenate(edge_sources[t]),
                          np.concatenate(edge_targets[t])]),
                dtype=torch.long,
            )
        else:
            ei = torch.zeros((2, 0), dtype=torch.long)
        edge_index_by_type[t] = ei
        n_edge_by_type[t] = ei.shape[1]

    # ─── Combined edge_index (for backward compat / Uniform mode) ───
    all_src = []
    all_tgt = []
    for t in [0, 1, 2]:
        if edge_sources[t]:
            all_src.append(np.concatenate(edge_sources[t]))
            all_tgt.append(np.concatenate(edge_targets[t]))
    edge_index = torch.tensor(
        np.stack([np.concatenate(all_src), np.concatenate(all_tgt)]),
        dtype=torch.long,
    )

    n_total_edges = edge_index.shape[1]
    print(f"  [HeteroGraph] Total edges: {n_total_edges} "
          f"(within={n_edge_by_type[0]}, assoc={n_edge_by_type[1]}, aux={n_edge_by_type[2]})")

    # ─── Build PyG Data ───
    data = Data(edge_index=edge_index, num_nodes=N_total)
    data.N_circ = N_circ
    data.N_disease = N_disease
    data.N_mirna = N_mirna
    data.N_lncrna = N_lncrna
    data.x_all = x_all
    data.feat_dim = feat_dim
    data.edge_index_by_type = edge_index_by_type  # Dict[int, Tensor] for heterogeneous GNN

    # ─── Positional Encodings: LapPE + RWSE ───
    # LapPE: top-k eigenvectors of normalized Laplacian → global structural position
    # RWSE: random walk return probabilities at steps 1..k → local neighborhood density
    from data.dataset import compute_laplacian_pe, compute_rwse
    print(f"  [HeteroGraph] Computing positional encodings "
          f"(LapPE k={lap_pe_dim}, RWSE steps={rwse_dim})...")
    data.lap_pe = compute_laplacian_pe(edge_index, N_total, k=lap_pe_dim)
    data.rwse = compute_rwse(edge_index, N_total, steps=rwse_dim)

    node_counts = {
        'N_circ': N_circ, 'N_disease': N_disease,
        'N_mirna': N_mirna, 'N_lncrna': N_lncrna,
        'N_total': N_total, 'feat_dim': feat_dim,
        'n_edges_by_type': n_edge_by_type,
        # Node name lists (for LLM feature alignment)
        'circ_names': dataset.circrna_names,
        'disease_names': dataset.disease_names,
        'mirna_names': mirna_list,
        'lncrna_names': lncrna_list,
        # Adjacency matrices for auxiliary multi-task learning (v3.1)
        'mirna_dis_adj': torch.tensor(mirna_dis_adj, dtype=torch.float32),
        'circ_mirna_adj': torch.tensor(circ_mirna_adj, dtype=torch.float32),
        'circ_lncrna_adj': torch.tensor(circ_lncrna_adj, dtype=torch.float32),
    }

    return data, x_all, node_counts


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.dataset import CDADataset

    ds = CDADataset(name='circR2Cancer', data_dir='data')
    ds.load()
    train_mask = (ds.adj > 0).astype(np.float64)  # use all for quick test

    data, x_all, counts = build_hetero_graph(ds, train_mask)
    print(f"\nDone. x_all shape: {x_all.shape}")
    print(f"Node counts: {counts}")
