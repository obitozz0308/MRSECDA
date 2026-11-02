"""
Pathway Data Loader for PathEncoder.

Loads circRNA-miRNA and miRNA-disease association data from the
KHGAT-compatible CSV files, aligns names with the current dataset's
circRNA/disease indices, and produces sparse adjacency matrices
usable by PathEncoder.

Data sources:
- circRNA_miRNA/circRNA_miRNA.csv — rows=circRNAs, cols=miRNAs, binary
- miRNA_cancer/miRNA_cancer.csv — rows=miRNAs, cols=diseases, binary

Author: Claude Code (innovation-2)
Date: 2026-07-23
"""

import os
import re
import numpy as np
from typing import Dict, List, Optional, Tuple

import scipy.sparse as sp


# =============================================================================
# Name Normalization
# =============================================================================

def _normalize(name: str) -> str:
    """Normalize a biological entity name for fuzzy matching."""
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9]', '', name)  # remove non-alphanumeric
    return name


def _match_circrna(
    circ_name: str,
    pathway_names: List[str],
    pathway_norm: List[str],
) -> Optional[int]:
    """
    Match a circRNA name against pathway data circRNA names.
    Returns index in pathway_names or None.
    """
    norm = _normalize(circ_name)

    # Try exact normalized match
    for i, pn in enumerate(pathway_norm):
        if norm == pn:
            return i

    # Try prefix match (hsa_circ_0005232 vs circ_0005232)
    for i, pn in enumerate(pathway_norm):
        if norm.endswith(pn) or pn.endswith(norm):
            return i

    return None


def _match_disease(
    disease_name: str,
    pathway_diseases: List[str],
    pathway_norm: List[str],
) -> Optional[int]:
    """
    Match a disease name against pathway data disease names.
    Uses substring matching since cancer names vary (e.g., 'breast cancer'
    vs 'Breast Carcinoma').
    Returns index in pathway_diseases or None.
    """
    norm = _normalize(disease_name)

    # Try exact normalized match first
    for i, pn in enumerate(pathway_norm):
        if norm == pn:
            return i

    # Try substring match (disease name contains pathway term or vice versa)
    # e.g., "breast cancer" should match "breast carcinoma"
    words = norm.split('carcinoma')
    # Normalize more aggressively: replace 'carcinoma' with 'cancer' for matching
    alt_norm = norm.replace('carcinoma', 'cancer')
    for i, pn in enumerate(pathway_norm):
        pn_alt = pn.replace('carcinoma', 'cancer')
        if alt_norm == pn_alt:
            return i
        # Partial: one contains the other
        if len(alt_norm) > 4 and len(pn_alt) > 4:
            if alt_norm in pn_alt or pn_alt in alt_norm:
                return i

    return None


# =============================================================================
# CSV Loader
# =============================================================================

def _read_matrix_csv(path: str) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Read a CSV file in (rows × cols) format with row names in first column
    and column names in first row.

    Returns:
        row_names: list of row entity names
        col_names: list of column entity names
        matrix: (n_rows, n_cols) binary numpy array
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pathway file not found: {path}")

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError(f"Pathway file too short: {path}")

    # First line: empty first cell, then column names
    header = lines[0].strip().split(',')
    col_names = [h.strip() for h in header[1:] if h.strip()]

    row_names = []
    data_rows = []

    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) < 2:
            continue
        row_name = parts[0].strip()
        if not row_name:
            continue
        row_names.append(row_name)
        # Parse values
        values = []
        for v in parts[1:]:
            try:
                values.append(float(v))
            except (ValueError, IndexError):
                values.append(0.0)
        # Pad or truncate to match col_names
        if len(values) < len(col_names):
            values += [0.0] * (len(col_names) - len(values))
        else:
            values = values[:len(col_names)]
        data_rows.append(values)

    matrix = np.array(data_rows, dtype=np.float32)
    return row_names, col_names, matrix


# =============================================================================
# Main Loading Function
# =============================================================================

def load_pathway_data(
    circrna_names: List[str],
    disease_names: List[str],
    data_dir: str,
) -> Optional[Tuple[sp.spmatrix, sp.spmatrix, int]]:
    """
    Load and align pathway data for a dataset.

    Args:
        circrna_names: circRNA names from the current dataset (length N_circ)
        disease_names: disease names from the current dataset (length N_disease)
        data_dir: root data directory (e.g., 'data/CompareExperiment/Data')

    Returns:
        (A_cm, A_md, N_mirna) if successful, None if pathway data unavailable.
        - A_cm: sparse (N_circ, N_mirna) circRNA→miRNA associations
        - A_md: sparse (N_mirna, N_disease) miRNA→disease associations
        - N_mirna: number of miRNAs in the unified vocabulary
    """
    N_circ = len(circrna_names)
    N_disease = len(disease_names)

    # ── Locate pathway files ──
    cm_path = os.path.join(data_dir, 'circRNA_miRNA', 'circRNA_miRNA.csv')
    md_path = os.path.join(data_dir, 'miRNA_cancer', 'miRNA_cancer.csv')

    if not os.path.exists(cm_path) or not os.path.exists(md_path):
        print(f"  [Pathway] Files not found, PathEncoder disabled")
        return None

    # ── Load CSV files ──
    try:
        cm_circs, cm_mirnas, cm_matrix = _read_matrix_csv(cm_path)
        md_mirnas, md_diseases, md_matrix = _read_matrix_csv(md_path)
    except Exception as e:
        print(f"  [Pathway] Failed to load CSV: {e}")
        return None

    print(f"  [Pathway] Loaded: {len(cm_circs)} circs × {len(cm_mirnas)} miRNAs, "
          f"{len(md_mirnas)} miRNAs × {len(md_diseases)} diseases")

    # ── Build unified miRNA vocabulary ──
    # Use miRNAs that appear in BOTH matrices for maximum signal
    cm_mirna_norm = [_normalize(m) for m in cm_mirnas]
    md_mirna_norm = [_normalize(m) for m in md_mirnas]

    # Map from cm_mirna index to md_mirna index (or -1 if no match)
    cm_to_md_mirna = {}
    for i, cmn in enumerate(cm_mirna_norm):
        for j, mdn in enumerate(md_mirna_norm):
            if cmn == mdn:
                cm_to_md_mirna[i] = j
                break

    # Keep only miRNAs present in both
    shared_cm_indices = sorted(cm_to_md_mirna.keys())
    shared_md_indices = [cm_to_md_mirna[i] for i in shared_cm_indices]
    N_mirna = len(shared_cm_indices)

    if N_mirna < 10:
        print(f"  [Pathway] Only {N_mirna} shared miRNAs (<10), PathEncoder disabled")
        return None

    print(f"  [Pathway] {N_mirna} shared miRNAs between circRNA-miRNA and miRNA-disease")

    # ── Align circRNA names ──
    cm_circ_norm = [_normalize(c) for c in cm_circs]
    circ_matches = 0
    A_cm_data = []  # list of (circ_idx, mirna_idx, value)

    for i, circ_name in enumerate(circrna_names):
        cm_idx = _match_circrna(circ_name, cm_circs, cm_circ_norm)
        if cm_idx is not None:
            circ_matches += 1
            # For this circRNA, get its miRNA associations (only shared miRNAs)
            for local_m, cm_m in enumerate(shared_cm_indices):
                val = cm_matrix[cm_idx, cm_m]
                if val > 0:
                    A_cm_data.append((i, local_m, float(val)))

    circ_coverage = circ_matches / N_circ if N_circ > 0 else 0
    print(f"  [Pathway] circRNA coverage: {circ_matches}/{N_circ} ({circ_coverage:.1%})")

    # ── Align disease names ──
    md_dis_norm = [_normalize(d) for d in md_diseases]
    dis_matches = 0
    A_md_data = []  # list of (mirna_idx, dis_idx, value)

    for j, dis_name in enumerate(disease_names):
        md_idx = _match_disease(dis_name, md_diseases, md_dis_norm)
        if md_idx is not None:
            dis_matches += 1
            for local_m, md_m in enumerate(shared_md_indices):
                val = md_matrix[md_m, md_idx]
                if val > 0:
                    A_md_data.append((local_m, j, float(val)))

    dis_coverage = dis_matches / N_disease if N_disease > 0 else 0
    print(f"  [Pathway] disease coverage: {dis_matches}/{N_disease} ({dis_coverage:.1%})")

    # ── Check minimum coverage ──
    MIN_COVERAGE = 0.03  # 3%
    if circ_coverage < MIN_COVERAGE or dis_coverage < MIN_COVERAGE:
        print(f"  [Pathway] Coverage below {MIN_COVERAGE:.0%}, PathEncoder disabled "
              f"(circ={circ_coverage:.1%}, dis={dis_coverage:.1%})")
        return None

    # ── Build sparse matrices ──
    if A_cm_data:
        cm_rows, cm_cols, cm_vals = zip(*A_cm_data)
        A_cm = sp.coo_matrix((cm_vals, (cm_rows, cm_cols)), shape=(N_circ, N_mirna)).tocsr()
    else:
        A_cm = sp.csr_matrix((N_circ, N_mirna), dtype=np.float32)

    if A_md_data:
        md_rows, md_cols, md_vals = zip(*A_md_data)
        A_md = sp.coo_matrix((md_vals, (md_rows, md_cols)), shape=(N_mirna, N_disease)).tocsr()
    else:
        A_md = sp.csr_matrix((N_mirna, N_disease), dtype=np.float32)

    nnz_cm = A_cm.nnz
    nnz_md = A_md.nnz
    print(f"  [Pathway] A_cm: {A_cm.shape} ({nnz_cm} nnz), "
          f"A_md: {A_md.shape} ({nnz_md} nnz)")

    if nnz_cm == 0 or nnz_md == 0:
        print(f"  [Pathway] Empty matrix after alignment, PathEncoder disabled")
        return None

    return A_cm, A_md, N_mirna
