"""
Preprocess HMDD v4.0 data for use in heterogeneous graph construction.

HMDD v4.0 format: code | PMID | miRNA | disease | description
KHGAT format:     miRNA_name | weight | disease_name

Key tasks:
1. Extract cancer-specific miRNA-disease pairs
2. Normalize miRNA names (precursor → matchable with KHGAT mature names)
3. Map MeSH disease names to circR2Cancer common names
4. Output clean TSV + statistics
"""

import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")
OUTPUT_DIR = os.path.join(DATA_DIR, "external")

# Paths
HMDD_V4_PATH = os.path.join(EXTERNAL_DIR, "hmdd_v4_alldata.txt")
KHGAT_MIRNA_CANCER_PATH = os.path.join(
    DATA_DIR, "CompareExperiment/AE-RF main/KHGAT/dataprocess/miRNA_cancer.txt")
KHGAT_CIRCRNA_MIRNA_PATH = os.path.join(
    DATA_DIR, "CompareExperiment/AE-RF main/KHGAT/dataprocess/circRNA_miRNA.txt")

# Output
HMDD_CANCER_OUT = os.path.join(OUTPUT_DIR, "hmdd_v4_cancer_mirna_disease.txt")
HMDD_ALL_OUT = os.path.join(OUTPUT_DIR, "hmdd_v4_all_mirna_disease.txt")

# =============================================================================
# Disease name mapping: MeSH terms → common cancer names
# =============================================================================

# Map MeSH disease names to simpler cancer names (for matching with circR2Cancer)
MESH_TO_COMMON = {
    # Breast
    "Breast Neoplasms": "breast cancer",
    "Breast Neoplasms, Basal Cell": "breast cancer",
    "Breast Neoplasms, Triple-Negative": "breast cancer",
    "Breast Neoplasms, Inflammatory": "breast cancer",
    "Breast Carcinoma In Situ": "breast cancer",
    "Carcinoma, Ductal, Breast": "breast cancer",
    "Carcinoma, Lobular, Breast": "breast cancer",
    "Breast Neoplasms, Male": "breast cancer",
    "Breast Cancer, Hereditary": "breast cancer",
    "Inflammatory Breast Neoplasms": "breast cancer",
    "Canine Mammary Carcinoma": "breast cancer",

    # Lung
    "Lung Neoplasms": "lung cancer",
    "Carcinoma, Non-Small-Cell Lung": "lung cancer",
    "Carcinoma, Small Cell Lung": "lung cancer",
    "Adenocarcinoma of Lung": "lung cancer",
    "Lung Neoplasms, Small Cell": "lung cancer",
    "Carcinoma, Large Cell Lung": "lung cancer",
    "Adenocarcinoma, Bronchiolo": "lung cancer",
    "Carcinoma, Bronchogenic": "lung cancer",
    "Carcinoma, Lewis Lung": "lung cancer",

    # Colorectal
    "Colorectal Neoplasms": "colorectal cancer",
    "Colon Neoplasms": "colon cancer",
    "Rectal Neoplasms": "rectal cancer",
    "Adenocarcinoma, Colon": "colon cancer",
    "Colorectal Neoplasms, Hereditary Nonpolyposis": "colorectal cancer",

    # Prostate
    "Prostatic Neoplasms": "prostate cancer",
    "Prostatic Neoplasms, Castration-Resistant": "prostate cancer",
    "Carcinoma, Prostatic": "prostate cancer",
    "Adenocarcinoma, Prostate": "prostate cancer",

    # Liver
    "Liver Neoplasms": "liver cancer",
    "Carcinoma, Hepatocellular": "liver cancer",
    "Hepatocellular Carcinoma": "liver cancer",
    "Liver Neoplasms, Experimental": "liver cancer",
    "Adenoma, Liver Cell": "liver cancer",
    "Cholangiocarcinoma": "liver cancer",
    "Bile Duct Neoplasms": "bile duct cancer",
    "Biliary Tract Neoplasms": "bile duct cancer",

    # Gastric/Stomach
    "Stomach Neoplasms": "gastric cancer",
    "Adenocarcinoma, Gastric": "gastric cancer",
    "Gastric Cancer, Hereditary Diffuse": "gastric cancer",
    "Adenocarcinoma, Gastric-Esophageal Junction": "gastric cancer",

    # Esophageal
    "Esophageal Neoplasms": "esophageal cancer",
    "Adenocarcinoma, Esophageal": "esophageal cancer",
    "Esophageal Squamous Cell Carcinoma": "esophageal cancer",
    "Adenocarcinoma Of Esophagus": "esophageal cancer",

    # Pancreatic
    "Pancreatic Neoplasms": "pancreatic cancer",
    "Carcinoma, Pancreatic Ductal": "pancreatic cancer",
    "Adenocarcinoma, Pancreatic Ductal": "pancreatic cancer",
    "Carcinoma, Islet Cell": "pancreatic cancer",

    # Ovarian
    "Ovarian Neoplasms": "ovarian cancer",
    "Carcinoma, Ovarian Epithelial": "ovarian cancer",
    "Ovarian Cancer, Hereditary": "ovarian cancer",

    # Cervical
    "Uterine Cervical Neoplasms": "cervical cancer",
    "Cervical Intraepithelial Neoplasia": "cervical cancer",
    "Adenocarcinoma, Cervical": "cervical cancer",

    # Endometrial/Uterine
    "Endometrial Neoplasms": "endometrial cancer",
    "Uterine Neoplasms": "uterine cancer",
    "Adenocarcinoma, Endometrial": "endometrial cancer",

    # Bladder
    "Urinary Bladder Neoplasms": "bladder cancer",
    "Carcinoma, Bladder": "bladder cancer",
    "Carcinoma, Urothelial": "bladder cancer",

    # Kidney/Renal
    "Kidney Neoplasms": "kidney cancer",
    "Carcinoma, Renal Cell": "renal cell carcinoma",
    "Renal Cell Carcinoma": "renal cell carcinoma",
    "Carcinoma, Renal Cell, Clear Cell": "clear cell renal cell carcinoma",
    "Wilms Tumor": "wilms tumor",

    # Thyroid
    "Thyroid Neoplasms": "thyroid cancer",
    "Carcinoma, Papillary Thyroid": "thyroid cancer",
    "Thyroid Cancer, Papillary": "thyroid cancer",
    "Carcinoma, Thyroid": "thyroid cancer",
    "Carcinoma, Anaplastic Thyroid": "thyroid cancer",
    "Carcinoma, Follicular Thyroid": "thyroid cancer",
    "Carcinoma, Medullary Thyroid": "thyroid cancer",

    # Head and Neck
    "Head and Neck Neoplasms": "head and neck cancer",
    "Squamous Cell Carcinoma of Head and Neck": "head and neck cancer",
    "Mouth Neoplasms": "oral cancer",
    "Tongue Neoplasms": "tongue cancer",
    "Laryngeal Neoplasms": "laryngeal cancer",
    "Nasopharyngeal Neoplasms": "nasopharyngeal cancer",
    "Nasopharyngeal Carcinoma": "nasopharyngeal cancer",
    "Pharyngeal Neoplasms": "pharyngeal cancer",
    "Oropharyngeal Neoplasms": "oropharyngeal cancer",
    "Hypopharyngeal Neoplasms": "hypopharyngeal cancer",
    "Lip Neoplasms": "lip cancer",

    # Brain/CNS
    "Brain Neoplasms": "brain cancer",
    "Glioma": "glioma",
    "Glioblastoma": "glioblastoma",
    "Glioblastoma Multiforme": "glioblastoma",
    "Astrocytoma": "astrocytoma",
    "Medulloblastoma": "medulloblastoma",
    "Neuroblastoma": "neuroblastoma",
    "Retinoblastoma": "retinoblastoma",
    "Meningioma": "meningioma",
    "Brain Stem Neoplasms": "brain cancer",
    "Optic Nerve Glioma": "glioma",

    # Leukemia
    "Leukemia": "leukemia",
    "Leukemia, Myeloid, Acute": "leukemia",
    "Leukemia, Myeloid, Chronic": "leukemia",
    "Leukemia, Lymphocytic, Acute": "leukemia",
    "Leukemia, Lymphocytic, Chronic, B-Cell": "leukemia",
    "Leukemia, T-Cell": "leukemia",
    "Leukemia, B-Cell": "leukemia",
    "Acute Lymphoblastic Leukemia, B-Cell": "leukemia",
    "Acute Lymphoblastic Leukemia, T-Cell": "leukemia",
    "Acute Lymphocytic Leukemia, B-Cell": "leukemia",
    "Leukemia, Myeloid": "leukemia",
    "Leukemia, Myelomonocytic, Chronic": "leukemia",
    "Leukemia, Promyelocytic, Acute": "leukemia",
    "Leukemia, Hairy Cell": "leukemia",
    "Leukemia, Myelogenous, Chronic, BCR-ABL Positive": "leukemia",

    # Lymphoma
    "Lymphoma": "lymphoma",
    "Burkitt Lymphoma": "lymphoma",
    "Lymphoma, B-Cell": "lymphoma",
    "Lymphoma, T-Cell": "lymphoma",
    "Lymphoma, Non-Hodgkin": "lymphoma",
    "Lymphoma, Hodgkin": "lymphoma",
    "Lymphoma, Mantle-Cell": "lymphoma",
    "Lymphoma, Large B-Cell, Diffuse": "lymphoma",
    "Lymphoma, Follicular": "lymphoma",

    # Skin
    "Melanoma": "melanoma",
    "Melanoma, Cutaneous": "melanoma",
    "Melanoma, Experimental": "melanoma",
    "Skin Neoplasms": "skin cancer",
    "Carcinoma, Basal Cell": "skin cancer",
    "Carcinoma, Squamous Cell": "skin cancer",
    "Carcinoma, Merkel Cell": "skin cancer",

    # Bone/Soft tissue
    "Bone Neoplasms": "bone cancer",
    "Osteosarcoma": "osteosarcoma",
    "Sarcoma": "sarcoma",
    "Sarcoma, Ewing": "sarcoma",
    "Sarcoma, Kaposi": "sarcoma",
    "Sarcoma, Synovial": "sarcoma",
    "Chondrosarcoma": "chondrosarcoma",
    "Liposarcoma": "liposarcoma",
    "Rhabdomyosarcoma": "rhabdomyosarcoma",
    "Leiomyosarcoma": "leiomyosarcoma",
    "Soft Tissue Neoplasms": "sarcoma",

    # Myeloma
    "Multiple Myeloma": "multiple myeloma",

    # Other cancers
    "Mesothelioma": "mesothelioma",
    "Mesothelioma, Malignant": "mesothelioma",
    "Thymoma": "thymoma",
    "Thymus Neoplasms": "thymus cancer",
    "Testicular Neoplasms": "testicular cancer",
    "Penile Neoplasms": "penile cancer",
    "Vulvar Neoplasms": "vulvar cancer",
    "Vaginal Neoplasms": "vaginal cancer",
    "Gestational Trophoblastic Disease": "gestational trophoblastic disease",
    "Adrenal Cortex Neoplasms": "adrenal cancer",
    "Adrenocortical Carcinoma": "adrenal cancer",
    "Adrenocortical Carcinoma, Pediatric": "adrenal cancer",
    "Pheochromocytoma": "pheochromocytoma",
    "Paraganglioma": "paraganglioma",
    "Gallbladder Neoplasms": "gallbladder cancer",
    "Ampulla of Vater Neoplasms": "ampullary cancer",
    "Adenocarcinoma, Ampullary": "ampullary cancer",
    "Duodenal Neoplasms": "duodenal cancer",
    "Small Cell Lung Carcinoma": "lung cancer",
    "Gastrointestinal Neoplasms": "gastrointestinal cancer",
    "Gastrointestinal Stromal Tumors": "gist",
    "Peritoneal Neoplasms": "peritoneal cancer",
    "Anus Neoplasms": "anal cancer",
    "Tracheal Neoplasms": "tracheal cancer",
    "Thymic Neoplasms": "thymus cancer",
    "Salivary Gland Neoplasms": "salivary gland cancer",
    "Parathyroid Neoplasms": "parathyroid cancer",
    "Pituitary Neoplasms": "pituitary cancer",
    "Pineal Gland Neoplasms": "pineal cancer",
    "Adrenal Gland Neoplasms": "adrenal cancer",
    "Splenic Neoplasms": "spleen cancer",
    "Heart Neoplasms": "heart cancer",
    "Mediastinal Neoplasms": "mediastinal cancer",
    "Pleural Neoplasms": "pleural cancer",
    "Nose Neoplasms": "nasal cancer",
    "Paranasal Sinus Neoplasms": "sinus cancer",
    "Adenocarcinoma, Sinonasal Intestinal": "sinus cancer",
    "Ear Neoplasms": "ear cancer",
    "Eye Neoplasms": "eye cancer",
    "Orbital Neoplasms": "eye cancer",
    "Conjunctival Neoplasms": "eye cancer",
    "Eyelid Neoplasms": "eye cancer",
    "Adenocarcinoma, Sebaceous, Eyelid": "eye cancer",
    "Adenocarcinoma, Sebaceous, Ocular": "eye cancer",
    "Uveal Neoplasms": "eye cancer",
    "Iris Neoplasms": "eye cancer",
    "Choroid Neoplasms": "eye cancer",
    "Retinal Neoplasms": "eye cancer",
    "Optic Nerve Neoplasms": "eye cancer",
    "Lacrimal Apparatus Neoplasms": "eye cancer",

    # Ameloblastoma (jaw tumor)
    "Ameloblastoma": "ameloblastoma",

    # Carcinoma (general)
    "Carcinoma": "carcinoma",
    "Carcinoma in Situ": "carcinoma",
    "Adenocarcinoma": "adenocarcinoma",
    "Adenocarcinoma in Situ": "adenocarcinoma",
    "Adenocarcinoma, Follicular": "adenocarcinoma",
    "Neoplasms, Unknown Primary": "cancer of unknown primary",
    "Neoplasms, Second Primary": "cancer",

    # Cancer-related conditions (not cancers themselves, but miRNA associations are still valuable)
    "Cancer Pain": "cancer pain",
    "Cancer-Associated Cachexia": "cancer cachexia",
    "Lymphatic Metastasis": "cancer metastasis",
    "Neoplasm Metastasis": "cancer metastasis",
    "Neoplasm Recurrence, Local": "cancer recurrence",
    "Neoplasm, Residual": "cancer recurrence",
    "Neoplastic Cells, Circulating": "circulating tumor cells",
    "Neoplastic Processes": "cancer",
    "Paraneoplastic Syndromes": "cancer",
    "Tumor Lysis Syndrome": "cancer",
    "Precancerous Conditions": "precancerous",
    "Carcinoma, Adenoid Cystic": "adenoid cystic carcinoma",
}


def is_cancer_disease(disease_name: str) -> bool:
    """Check if a disease name is cancer-related."""
    cancer_keywords = [
        'cancer', 'carcinoma', 'neoplasm', 'tumor', 'tumour',
        'melanoma', 'leukemia', 'lymphoma', 'glioma', 'sarcoma',
        'blastoma', 'malignant', 'myeloma', 'mesothelioma',
        'adenoma', 'thymoma', 'pheochromocytoma', 'paraganglioma',
        'meningioma', 'retinoblastoma', 'medulloblastoma',
        'astrocytoma', 'cholangiocarcinoma', 'ameloblastoma',
        'chondrosarcoma', 'liposarcoma', 'rhabdomyosarcoma',
        'leiomyosarcoma', 'osteosarcoma', 'gestational trophoblastic',
        'neoplastic', 'precancerous', 'metastasis',
    ]
    disease_lower = disease_name.lower()
    return any(kw in disease_lower for kw in cancer_keywords)


# =============================================================================
# miRNA name normalization
# =============================================================================

def normalize_mirna_name(name: str) -> str:
    """
    Normalize miRNA names for cross-database matching.

    HMDD v4.0 uses precursor names: hsa-mir-21, hsa-let-7a-1
    KHGAT uses mature names:       hsa-miR-21-5p, hsa-let-7a-5p

    Strategy:
    1. Lowercase
    2. Normalize "mir" → "mir" (already lowercase)
    3. Remove -5p/-3p suffix from mature names
    4. Strip trailing -1/-2 (paralog numbers from precursors)
    5. For let-7 family: strip -1/-2 suffix but keep base name
    """
    name = name.lower().strip()

    # Remove mature arm suffix (-5p, -3p)
    # But be careful: hsa-mir-30c-5p → hsa-mir-30c
    name = re.sub(r'-(5p|3p)$', '', name)

    # Remove star (*) notation
    name = re.sub(r'\*$', '', name)

    return name


def normalize_mirna_name_aggressive(name: str) -> str:
    """
    More aggressive normalization: strip paralog numbers to maximize matching.

    hsa-mir-21 → hsa-mir-21
    hsa-let-7a-1 → hsa-let-7a
    hsa-mir-30c-1 → hsa-mir-30c
    """
    name = normalize_mirna_name(name)

    # For let-7 family: remove trailing paralog number
    # hsa-let-7a-1 → hsa-let-7a
    name = re.sub(r'^(hsa-let-\w+)-\d+$', r'\1', name)

    # For regular mir: remove trailing paralog number
    # hsa-mir-30c-1 → hsa-mir-30c
    # But NOT hsa-mir-100 → hsa-mir-10 (don't strip miRNA number digits)
    name = re.sub(r'^(hsa-mir-\d+[a-z]?)-\d+$', r'\1', name)

    return name


# =============================================================================
# Main preprocessing
# =============================================================================

def load_hmdd_v4(path: str) -> List[Tuple[str, str, str, str, str]]:
    """Load HMDD v4.0 data. Returns list of (code, pmid, mirna, disease, description)."""
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                code, pmid, mirna, disease, desc = parts[0], parts[1], parts[2], parts[3], parts[4]
                entries.append((code, pmid, mirna, disease, desc))
    return entries


def build_disease_mapping(hmdd_entries, target_diseases_lower):
    """Build a mapping from HMDD MeSH disease names to target dataset disease names."""
    mapping = {}

    for _, _, _, disease, _ in hmdd_entries:
        dis_lower = disease.lower().strip()
        if dis_lower in mapping:
            continue

        # Check MeSH_TO_COMMON first
        if disease in MESH_TO_COMMON:
            common = MESH_TO_COMMON[disease]
            if common in target_diseases_lower:
                mapping[dis_lower] = common
                continue

        # Check if the MeSH name contains a target disease name
        for target_dis in target_diseases_lower:
            if target_dis in dis_lower or dis_lower in target_dis:
                mapping[dis_lower] = target_dis
                break

    return mapping


def main():
    print("=" * 70)
    print("HMDD v4.0 Preprocessing for circRNA-Disease Heterogeneous Graph")
    print("=" * 70)

    # Load HMDD v4.0
    print(f"\nLoading HMDD v4.0 from: {HMDD_V4_PATH}")
    entries = load_hmdd_v4(HMDD_V4_PATH)
    print(f"  Total entries: {len(entries)}")

    # Statistics
    all_mirnas = set(e[2].lower().strip() for e in entries)
    all_diseases = set(e[3].lower().strip() for e in entries)
    cancer_diseases = {d for d in all_diseases if is_cancer_disease(d)}

    print(f"  Unique miRNAs: {len(all_mirnas)}")
    print(f"  Unique diseases: {len(all_diseases)}")
    print(f"  Cancer-related diseases: {len(cancer_diseases)}")

    # Separate cancer and non-cancer entries
    cancer_entries = [(c, p, m, d, desc) for c, p, m, d, desc in entries
                      if is_cancer_disease(d)]
    print(f"  Cancer-related entries: {len(cancer_entries)}")

    # Load KHGAT data for comparison
    print(f"\nLoading KHGAT data for comparison...")

    # KHGAT miRNA names (mature)
    khgat_mirna_names = set()
    if os.path.exists(KHGAT_MIRNA_CANCER_PATH):
        with open(KHGAT_MIRNA_CANCER_PATH, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    khgat_mirna_names.add(parts[0].lower().strip())
    print(f"  KHGAT unique miRNAs: {len(khgat_mirna_names)}")

    # KHGAT circRNA-interacting miRNAs
    khgat_circ_mirnas = set()
    if os.path.exists(KHGAT_CIRCRNA_MIRNA_PATH):
        with open(KHGAT_CIRCRNA_MIRNA_PATH, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    khgat_circ_mirnas.add(parts[0].lower().strip())
    print(f"  KHGAT miRNAs with circRNA interactions: {len(khgat_circ_mirnas)}")

    # ─── miRNA Name Normalization Analysis ───
    print(f"\n{'='*50}")
    print("miRNA Name Normalization Analysis")
    print(f"{'='*50}")

    # Build normalization maps
    hmd_mirna_norm = {}  # original → normalized
    hmd_mirna_norm_agg = {}  # original → aggressive normalized
    for m in all_mirnas:
        if m.startswith('hsa-'):
            hmd_mirna_norm[m] = normalize_mirna_name(m)
            hmd_mirna_norm_agg[m] = normalize_mirna_name_aggressive(m)

    khgat_mirna_norm = {}
    khgat_mirna_norm_agg = {}
    for m in khgat_mirna_names:
        khgat_mirna_norm[m] = normalize_mirna_name(m)
        khgat_mirna_norm_agg[m] = normalize_mirna_name_aggressive(m)

    khgat_circ_norm = {}
    khgat_circ_norm_agg = {}
    for m in khgat_circ_mirnas:
        khgat_circ_norm[m] = normalize_mirna_name(m)
        khgat_circ_norm_agg[m] = normalize_mirna_name_aggressive(m)

    # Check overlap
    hmd_norm_set = set(hmd_mirna_norm.values())
    hmd_norm_agg_set = set(hmd_mirna_norm_agg.values())
    khgat_norm_set = set(khgat_mirna_norm.values())
    khgat_norm_agg_set = set(khgat_mirna_norm_agg.values())
    khgat_circ_norm_set = set(khgat_circ_norm.values())
    khgat_circ_norm_agg_set = set(khgat_circ_norm_agg.values())

    print(f"\n  Basic normalization (remove -5p/-3p):")
    print(f"    HMDD unique normalized: {len(hmd_norm_set)}")
    print(f"    KHGAT unique normalized: {len(khgat_norm_set)}")
    overlap = hmd_norm_set & khgat_norm_set
    print(f"    Overlap: {len(overlap)}")
    overlap_circ = hmd_norm_set & khgat_circ_norm_set
    print(f"    Overlap with KHGAT circRNA-interacting: {len(overlap_circ)}")

    print(f"\n  Aggressive normalization (remove paralog numbers):")
    print(f"    HMDD unique normalized: {len(hmd_norm_agg_set)}")
    print(f"    KHGAT unique normalized: {len(khgat_norm_agg_set)}")
    overlap_agg = hmd_norm_agg_set & khgat_norm_agg_set
    print(f"    Overlap: {len(overlap_agg)}")
    overlap_circ_agg = hmd_norm_agg_set & khgat_circ_norm_agg_set
    print(f"    Overlap with KHGAT circRNA-interacting: {len(overlap_circ_agg)}")

    # ─── Build integrated miRNA list ───
    # Strategy: use aggressive normalization to match HMDD miRNAs with KHGAT circRNA miRNAs
    print(f"\n{'='*50}")
    print("Building Integrated miRNA List")
    print(f"{'='*50}")

    # Map: HMDD normalized name → original HMDD name
    hmd_norm_to_orig = defaultdict(set)
    for orig, norm in hmd_mirna_norm_agg.items():
        hmd_norm_to_orig[norm].add(orig)

    # Map: KHGAT normalized name → original KHGAT name
    khgat_norm_to_orig = defaultdict(set)
    for orig, norm in khgat_mirna_norm_agg.items():
        khgat_norm_to_orig[norm].add(orig)

    # For each KHGAT circRNA-interacting miRNA, find matching HMDD miRNA
    matched_mirnas = {}  # KHGAT normalized → (KHGAT original, HMDD original)
    for khg_orig in khgat_circ_mirnas:
        khg_norm = khgat_mirna_norm_agg[khg_orig]
        if khg_norm in hmd_norm_to_orig:
            # Use the first HMDD original name that matches
            hmd_orig = list(hmd_norm_to_orig[khg_norm])[0]
            matched_mirnas[khg_norm] = (khg_orig, hmd_orig)

    print(f"  KHGAT circRNA-interacting miRNAs: {len(khgat_circ_mirnas)}")
    print(f"  Matched to HMDD v4.0: {len(matched_mirnas)}")

    # ─── Count disease associations per matched miRNA ───
    # Use HMDD data to count associations
    hmd_mirna_dis_count = defaultdict(set)  # miRNA_norm → set of disease names
    for _, _, mirna, disease, _ in cancer_entries:
        m_lower = mirna.lower().strip()
        if m_lower in hmd_mirna_norm_agg:
            m_norm = hmd_mirna_norm_agg[m_lower]
            if m_norm in matched_mirnas:
                hmd_mirna_dis_count[m_norm].add(disease.lower().strip())

    # Count total cancer disease associations from HMDD
    print(f"\n  Disease association counts for matched miRNAs (from HMDD v4.0):")
    counts = [len(v) for v in hmd_mirna_dis_count.values()]
    if counts:
        print(f"    Mean: {sum(counts)/len(counts):.1f}")
        print(f"    Median: {sorted(counts)[len(counts)//2]:.0f}")
        print(f"    Min: {min(counts)}, Max: {max(counts)}")

    # Count miRNAs with ≥3, ≥5, ≥10 disease associations
    for threshold in [1, 2, 3, 5, 10, 20]:
        n = sum(1 for c in counts if c >= threshold)
        print(f"    ≥{threshold} disease assoc: {n} miRNAs")

    # ─── Generate output: KHGAT-compatible TSV ───
    print(f"\n{'='*50}")
    print("Generating Output Files")
    print(f"{'='*50}")

    # 1. Cancer-specific miRNA-disease pairs (using HMDD v4.0 cancer entries)
    cancer_output = []
    for _, _, mirna, disease, _ in cancer_entries:
        m_lower = mirna.lower().strip()
        if m_lower in hmd_mirna_norm_agg:
            m_norm = hmd_mirna_norm_agg[m_lower]
            if m_norm in matched_mirnas:
                # Use KHGAT mature miRNA name for compatibility with circRNA_miRNA.txt
                khg_name = matched_mirnas[m_norm][0]
                cancer_output.append((khg_name, "1", disease.lower().strip()))

    # Deduplicate
    cancer_output = list(set(cancer_output))

    with open(HMDD_CANCER_OUT, 'w', encoding='utf-8') as f:
        for mirna, weight, disease in sorted(cancer_output):
            f.write(f"{mirna}\t{weight}\t{disease}\n")

    cancer_mirnas = set(e[0] for e in cancer_output)
    cancer_dis = set(e[2] for e in cancer_output)
    print(f"  Cancer miRNA-disease output: {len(cancer_output)} pairs")
    print(f"    miRNAs: {len(cancer_mirnas)}")
    print(f"    Diseases: {len(cancer_dis)}")
    print(f"  Saved to: {HMDD_CANCER_OUT}")

    # 2. All-disease miRNA-disease pairs (for maximum coverage)
    all_output = []
    for _, _, mirna, disease, _ in entries:
        m_lower = mirna.lower().strip()
        if m_lower in hmd_mirna_norm_agg:
            m_norm = hmd_mirna_norm_agg[m_lower]
            if m_norm in matched_mirnas:
                khg_name = matched_mirnas[m_norm][0]
                all_output.append((khg_name, "1", disease.lower().strip()))

    all_output = list(set(all_output))

    with open(HMDD_ALL_OUT, 'w', encoding='utf-8') as f:
        for mirna, weight, disease in sorted(all_output):
            f.write(f"{mirna}\t{weight}\t{disease}\n")

    all_mirnas_count = len(set(e[0] for e in all_output))
    all_dis_count = len(set(e[2] for e in all_output))
    print(f"\n  All-disease miRNA-disease output: {len(all_output)} pairs")
    print(f"    miRNAs: {all_mirnas_count}")
    print(f"    Diseases: {all_dis_count}")
    print(f"  Saved to: {HMDD_ALL_OUT}")

    # ─── Summary ───
    print(f"\n{'='*50}")
    print("Summary: Coverage Improvement")
    print(f"{'='*50}")

    # Original KHGAT stats
    khgat_orig_pairs = []
    if os.path.exists(KHGAT_MIRNA_CANCER_PATH):
        with open(KHGAT_MIRNA_CANCER_PATH, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    khgat_orig_pairs.append((parts[0].lower().strip(), parts[2].lower().strip()))
    khgat_orig_mirnas = set(e[0] for e in khgat_orig_pairs)

    print(f"\n  KHGAT original:")
    print(f"    miRNA-disease pairs: {len(khgat_orig_pairs)}")
    print(f"    miRNAs: {len(khgat_orig_mirnas)}")
    print(f"    After circRNA filter + ≥3 disease assoc: ~244 miRNAs")

    print(f"\n  HMDD v4.0 (cancer only, matched miRNAs):")
    print(f"    miRNA-disease pairs: {len(cancer_output)}")
    print(f"    miRNAs: {len(cancer_mirnas)}")

    print(f"\n  HMDD v4.0 (all diseases, matched miRNAs):")
    print(f"    miRNA-disease pairs: {len(all_output)}")
    print(f"    miRNAs: {all_mirnas_count}")

    # Expected improvement
    print(f"\n  Expected improvement with HMDD v4.0:")
    print(f"    More disease associations per miRNA → more miRNAs pass ≥3 filter")
    print(f"    More comprehensive cancer coverage (332 vs 42 cancer types)")
    print(f"    Better cross-dataset disease name mapping")

    return {
        'matched_mirnas': matched_mirnas,
        'cancer_output': cancer_output,
        'all_output': all_output,
        'cancer_mirnas': cancer_mirnas,
        'hmd_mirna_dis_count': hmd_mirna_dis_count,
    }


if __name__ == "__main__":
    results = main()
