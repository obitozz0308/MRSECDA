"""
Predict circRNA-miRNA interactions using seed region matching.

This module provides sequence-based prediction of circRNA-miRNA interactions
by scanning circRNA sequences for miRNA seed region binding sites (MREs).
Used to supplement KHGAT/starBase data for datasets with poor miRNA coverage.

Approach:
  1. Extract miRNA seed region (nt 2-8, 7-mer) from miRBase mature sequences
  2. For each circRNA, count reverse-complement seed matches in its sequence
  3. Rank miRNAs by site count, apply min_sites threshold
  4. Output in KHGAT-compatible format for hetero_data.py consumption

Reference: Bartel, D.P. (2009) "MicroRNAs: Target Recognition and Regulatory Functions"
           TargetScan 7.0 seed types: 8mer, 7mer-m8, 7mer-A1, 6mer
"""

from __future__ import annotations
import os, sys, json, re
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_SCRIPT_DIR)
_CACHE_DIR = os.path.join(_PROJ_ROOT, "data", "external")

# ── RNA/DNA complement tables ──
RNA_COMPLEMENT = str.maketrans('AUCGaucg', 'UAGCuagc')
DNA_COMPLEMENT = str.maketrans('ATCGatcg', 'TAGCtagc')


def _reverse_complement_dna(seq: str) -> str:
    """Reverse complement of a DNA sequence."""
    return seq.translate(DNA_COMPLEMENT)[::-1]


def _rna_to_dna(seq: str) -> str:
    """Convert RNA sequence to DNA (U → T)."""
    return seq.replace('U', 'u').replace('T', 'T').replace('u', 'T').upper()


def _extract_seed(mirna_seq_rna: str, seed_type: str = '7mer-m8') -> Optional[str]:
    """
    Extract miRNA seed region and return DNA reverse complement.

    Seed types (from TargetScan convention):
      - '8mer':  nt 2-8 (0-idx: 1-8), 7nt core. Also requires A at position 1.
      - '7mer-m8': nt 2-8 (0-idx: 1-8), 7nt. Most canonical.
      - '7mer-A1': nt 2-7 (0-idx: 1-7), 6nt. Also requires A at position 1.
      - '6mer':   nt 2-7 (0-idx: 1-7), 6nt. Minimum seed.

    Returns DNA reverse complement of the seed (the MRE sequence to search for in circRNA DNA).
    """
    seq = mirna_seq_rna.upper().replace('T', 'U')

    if seed_type == '8mer':
        # Positions 2-8 (1-indexed) = [1:8] (0-indexed)
        if len(seq) < 8:
            return None
        seed_rna = seq[1:8]
        # 8mer also requires A at position 1
        if seq[0] != 'A':
            return None
    elif seed_type == '7mer-m8':
        if len(seq) < 8:
            return None
        seed_rna = seq[1:8]  # nt 2-8
    elif seed_type == '7mer-A1':
        if len(seq) < 7:
            return None
        seed_rna = seq[1:7]  # nt 2-7
        if seq[0] != 'A':
            return None
    elif seed_type == '6mer':
        if len(seq) < 7:
            return None
        seed_rna = seq[1:7]  # nt 2-7
    else:
        raise ValueError(f"Unknown seed type: {seed_type}")

    # Convert seed RNA → DNA, then reverse complement
    seed_dna = _rna_to_dna(seed_rna)
    mre_dna = _reverse_complement_dna(seed_dna)
    return mre_dna


def load_mirbase_sequences(mature_fa_path: str = "") -> Dict[str, str]:
    """Load mature miRNA sequences from miRBase mature.fa.

    Returns: {mirbase_id: rna_sequence}  e.g. {'hsa-miR-21-5p': 'UAGCUU...'}
    Filters to human (hsa-) miRNAs only.
    """
    if not mature_fa_path:
        mature_fa_path = os.path.join(_CACHE_DIR, "mature.fa")
    if not os.path.exists(mature_fa_path):
        print(f"  [SeedMatch] ERROR: miRBase mature.fa not found at {mature_fa_path}")
        return {}

    seqs = {}
    with open(mature_fa_path, 'r') as f:
        current_id = None
        current_seq = []
        for line in f:
            if line.startswith('>'):
                if current_id:
                    seqs[current_id] = ''.join(current_seq)
                header = line[1:].strip().split()[0]
                current_id = header
                current_seq = []
            else:
                current_seq.append(line.strip().upper())
        if current_id:
            seqs[current_id] = ''.join(current_seq)

    # Filter to human miRNAs
    human_seqs = {mid: seq for mid, seq in seqs.items()
                   if mid.lower().startswith('hsa-')}
    print(f"  [SeedMatch] Loaded {len(human_seqs)} human miRNAs from miRBase (total {len(seqs)} entries)")
    return human_seqs


def map_mirna_names_to_mirbase(
    mirna_names: List[str],
    mirbase_seqs: Dict[str, str],
) -> Dict[str, str]:
    """
    Map HMDD/KHGAT miRNA names to miRBase mature IDs.

    Args:
        mirna_names: names in KHGAT/HMDD format (e.g., 'hsa-mir-182', 'hsa-miR-21-5p')
        mirbase_seqs: {mirbase_id: seq} from load_mirbase_sequences()

    Returns: {input_name: mirbase_id} for successfully mapped miRNAs
    """
    name_map = {}
    for name in mirna_names:
        n = name.lower().strip()
        # Normalize: mir → miR
        n = n.replace('mir-', 'miR-')

        matched = None

        # Try exact match
        if n in mirbase_seqs:
            matched = n
        # Try with hsa- prefix
        elif 'hsa-' + n in mirbase_seqs and not n.startswith('hsa-'):
            matched = 'hsa-' + n
        # Try adding -5p
        elif not n.endswith('-5p') and not n.endswith('-3p'):
            variants = [n + '-5p', n + '-3p']
            for v in variants:
                if v in mirbase_seqs:
                    matched = v
                    break
                if not n.startswith('hsa-') and 'hsa-' + v in mirbase_seqs:
                    matched = 'hsa-' + v
                    break

        if matched:
            # Verify human
            if matched.lower().startswith('hsa-'):
                name_map[name] = matched

    return name_map


def predict_circrna_mirna_interactions(
    circrna_seqs: Dict[str, str],
    mirna_names: List[str],
    mirna_name_to_mirbase: Dict[str, str],
    mirbase_seqs: Dict[str, str],
    seed_type: str = '7mer-m8',
    min_sites: int = 2,
    max_mirnas_per_circ: int = 100,
    verbose: bool = True,
) -> List[Tuple[str, str]]:
    """
    Predict circRNA-miRNA interactions using seed region matching.

    For each circRNA, scans its DNA sequence for reverse-complement matches
    of each miRNA's seed region. Returns (mirna_name, circrna_name) pairs
    where the circRNA has >= min_sites seed matches.

    Args:
        circrna_seqs: {circrna_name: dna_sequence}
        mirna_names: list of miRNA names in KHGAT/HMDD format
        mirna_name_to_mirbase: {khgat_name: mirbase_id} mapping
        mirbase_seqs: {mirbase_id: rna_sequence} from miRBase
        seed_type: '8mer', '7mer-m8', '7mer-A1', or '6mer'
        min_sites: minimum seed match count required
        max_mirnas_per_circ: cap miRNAs per circRNA (top by site count)
        verbose: print progress

    Returns:
        List of (mirna_name, circrna_name) pairs
    """
    # ── Step 1: Extract miRNA seed MREs ──
    mirna_mres = {}  # {khgat_name: dna_mre_sequence}
    skipped = 0
    for khgat_name in mirna_names:
        mbid = mirna_name_to_mirbase.get(khgat_name)
        if not mbid or mbid not in mirbase_seqs:
            skipped += 1
            continue
        rna_seq = mirbase_seqs[mbid]
        mre = _extract_seed(rna_seq, seed_type)
        if mre:
            mirna_mres[khgat_name] = mre

    if verbose:
        print(f"  [SeedMatch] {len(mirna_mres)}/{len(mirna_names)} miRNAs have "
              f"valid {seed_type} seeds ({skipped} skipped)")

    if not mirna_mres:
        return []

    # ── Step 2: Scan circRNA sequences for seed matches ──
    pairs = []
    n_circ_with_seqs = 0
    n_circ_no_seqs = 0

    for circ_name, circ_seq in circrna_seqs.items():
        if not circ_seq or len(circ_seq) < 10:
            n_circ_no_seqs += 1
            continue

        n_circ_with_seqs += 1
        circ_seq_upper = circ_seq.upper()

        # Score each miRNA by site count
        mirna_scores = []
        for khgat_name, mre in mirna_mres.items():
            count = circ_seq_upper.count(mre)
            if count >= min_sites:
                mirna_scores.append((khgat_name, count))

        # Sort by site count (descending) and cap
        mirna_scores.sort(key=lambda x: x[1], reverse=True)
        for khgat_name, count in mirna_scores[:max_mirnas_per_circ]:
            pairs.append((khgat_name, circ_name))

    if verbose:
        total_possible = n_circ_with_seqs * len(mirna_mres)
        print(f"  [SeedMatch] circRNAs: {n_circ_with_seqs} with seqs, "
              f"{n_circ_no_seqs} without")
        print(f"  [SeedMatch] Predicted {len(pairs)} interactions "
              f"({len(pairs)/max(total_possible,1)*100:.1f}% of possible)")

        # Coverage stats
        circs_with_mirna = len(set(p[1] for p in pairs))
        mirnas_with_circ = len(set(p[0] for p in pairs))
        print(f"  [SeedMatch] Coverage: {circs_with_mirna}/{n_circ_with_seqs} circRNAs "
              f"({circs_with_mirna/max(n_circ_with_seqs,1)*100:.1f}%), "
              f"{mirnas_with_circ}/{len(mirna_mres)} miRNAs")

    return pairs


def predict_with_fallback(
    circrna_seqs: Dict[str, str],
    mirna_names: List[str],
    mirna_name_to_mirbase: Dict[str, str],
    mirbase_seqs: Dict[str, str],
    min_sites_7mer: int = 2,
    min_sites_6mer: int = 3,
    min_circ_coverage_pct: float = 95.0,
    verbose: bool = True,
) -> List[Tuple[str, str]]:
    """
    Predict circRNA-miRNA interactions with progressive fallback:

    1. Try 7mer-m8 (nt 2-8) with min_sites_7mer
    2. If circRNA coverage < min_circ_coverage_pct, add 6mer (nt 2-7) for uncovered circRNAs
    3. If still insufficient, lower thresholds

    Returns combined list of (mirna_name, circrna_name) pairs.
    """
    all_pairs = set()

    # ── Tier 1: 7mer-m8 (high confidence) ──
    pairs_7mer = predict_circrna_mirna_interactions(
        circrna_seqs, mirna_names, mirna_name_to_mirbase, mirbase_seqs,
        seed_type='7mer-m8', min_sites=min_sites_7mer, verbose=verbose,
    )
    all_pairs.update(pairs_7mer)

    circs_covered = set(p[1] for p in all_pairs)
    circs_with_seqs = [n for n, s in circrna_seqs.items() if s and len(s) >= 10]
    coverage_pct = len(circs_covered) / max(len(circs_with_seqs), 1) * 100

    # ── Tier 2: 6mer for uncovered circRNAs ──
    if coverage_pct < min_circ_coverage_pct:
        uncovered_seqs = {n: s for n, s in circrna_seqs.items()
                          if n not in circs_covered and s and len(s) >= 10}
        if verbose:
            print(f"\n  [SeedMatch] Coverage {coverage_pct:.1f}% < {min_circ_coverage_pct}%, "
                  f"trying 6mer for {len(uncovered_seqs)} uncovered circRNAs...")

        pairs_6mer = predict_circrna_mirna_interactions(
            uncovered_seqs, mirna_names, mirna_name_to_mirbase, mirbase_seqs,
            seed_type='6mer', min_sites=min_sites_6mer, verbose=verbose,
        )
        all_pairs.update(pairs_6mer)

        circs_covered = set(p[1] for p in all_pairs)
        coverage_pct = len(circs_covered) / max(len(circs_with_seqs), 1) * 100

    # ── Tier 3: Relaxed 6mer (min_sites=1) if still low ──
    if coverage_pct < min_circ_coverage_pct:
        uncovered_seqs = {n: s for n, s in circrna_seqs.items()
                          if n not in circs_covered and s and len(s) >= 10}
        if verbose:
            print(f"\n  [SeedMatch] Coverage {coverage_pct:.1f}% < {min_circ_coverage_pct}%, "
                  f"relaxing to min_sites=1 for {len(uncovered_seqs)} uncovered circRNAs...")

        pairs_relaxed = predict_circrna_mirna_interactions(
            uncovered_seqs, mirna_names, mirna_name_to_mirbase, mirbase_seqs,
            seed_type='6mer', min_sites=1, verbose=verbose,
        )
        all_pairs.update(pairs_relaxed)

        circs_covered = set(p[1] for p in all_pairs)
        coverage_pct = len(circs_covered) / max(len(circs_with_seqs), 1) * 100

    if verbose:
        print(f"\n  [SeedMatch] Final coverage: {len(circs_covered)}/{len(circs_with_seqs)} "
              f"circRNAs ({coverage_pct:.1f}%), {len(all_pairs)} total interactions")

    return list(all_pairs)


def build_and_save_interactions(
    dataset_name: str,
    circrna_names: List[str],
    disease_names: List[str],
    circrna_seqs: Dict[str, str],
    khgat_circ_mirna_pairs: List[Tuple[str, str]],
    hmdd_mirna_names: List[str],
    output_dir: str = "",
    min_circ_coverage_pct: float = 95.0,
    verbose: bool = True,
) -> Dict:
    """
    Build comprehensive circRNA-miRNA interactions for a dataset.

    Combines:
      1. Existing KHGAT interactions (filtered to this dataset's circRNAs)
      2. Sequence-predicted interactions (for circRNAs not covered by KHGAT)

    Args:
        dataset_name: e.g., 'circR2Disease'
        circrna_names: list of circRNA names from the dataset
        disease_names: list of disease names from the dataset
        circrna_seqs: {circrna_name: dna_sequence} from circBase
        khgat_circ_mirna_pairs: all known (mirna, circrna) pairs from KHGAT
        hmdd_mirna_names: miRNA names from HMDD v4.0
        output_dir: directory to save output files
        min_circ_coverage_pct: target circRNA coverage percentage

    Returns:
        dict with coverage stats and output file paths
    """
    if not output_dir:
        output_dir = _CACHE_DIR

    # ── Load miRBase sequences ──
    mirbase_seqs = load_mirbase_sequences()
    if not mirbase_seqs:
        return {'error': 'No miRBase sequences available'}

    # ── Build miRNA name mapping ──
    # Combine HMDD miRNA names + KHGAT miRNA names
    all_mirna_names = list(dict.fromkeys(hmdd_mirna_names + [p[0] for p in khgat_circ_mirna_pairs]))
    mirna_name_map = map_mirna_names_to_mirbase(all_mirna_names, mirbase_seqs)

    if verbose:
        n_mapped = len(mirna_name_map)
        print(f"  Mapped {n_mapped}/{len(all_mirna_names)} miRNA names to miRBase")

    # ── Phase 1: KHGAT interactions filtered to this dataset ──
    khgat_set = set()
    circ_name_set = set(str(c).strip() for c in circrna_names)

    # Build normalized lookup for KHGAT circRNA names
    circ_norm_lookup = {}
    for cname in circ_name_set:
        # Multiple normalization variants
        variants = {cname, cname.lower(), cname.upper()}
        # Strip/standardize format
        c_stripped = cname.strip()
        variants.add(c_stripped)
        variants.add(c_stripped.lower())
        variants.add(c_stripped.upper())
        # hsa_circ_XXXXXXXXX variants
        for v in list(variants):
            variants.add(v.replace('hsa_circ_', ''))
            variants.add('hsa_circ_' + v.replace('hsa_circ_', ''))
        for v in variants:
            circ_norm_lookup[v] = cname

    for mirna_name, circ_name in khgat_circ_mirna_pairs:
        c_clean = circ_name.strip()
        if c_clean in circ_norm_lookup:
            canonical = circ_norm_lookup[c_clean]
            khgat_set.add((mirna_name, canonical))
        # Also try direct match
        elif c_clean in circ_name_set:
            khgat_set.add((mirna_name, c_clean))

    khgat_covered_circs = set(p[1] for p in khgat_set)
    if verbose:
        print(f"  KHGAT interactions in {dataset_name}: {len(khgat_set)} pairs, "
              f"{len(khgat_covered_circs)}/{len(circ_name_set)} circRNAs covered "
              f"({len(khgat_covered_circs)/max(len(circ_name_set),1)*100:.1f}%)")

    # ── Phase 2: Sequence-predicted interactions for uncovered circRNAs ──
    uncovered_names = [n for n in circ_name_set if n not in khgat_covered_circs]

    # Filter circrna_seqs to uncovered + name match
    uncovered_seqs = {}
    # Build name normalization for sequence matching
    seq_norm_lookup = {}
    for sname in circrna_seqs:
        seq_norm_lookup[sname.lower().strip()] = sname
        seq_norm_lookup[sname.upper().strip()] = sname
        seq_norm_lookup[sname.strip()] = sname

    for uname in uncovered_names:
        u = uname.strip()
        key = u.lower()
        if u in circrna_seqs:
            uncovered_seqs[uname] = circrna_seqs[u]
        elif key in seq_norm_lookup:
            uncovered_seqs[uname] = circrna_seqs[seq_norm_lookup[key]]

    # Also: ALL circrna_seqs that match dataset circRNAs (for comprehensive coverage)
    all_dataset_seqs = {}
    for cname in circ_name_set:
        c = cname.strip()
        if c in circrna_seqs:
            all_dataset_seqs[cname] = circrna_seqs[c]
        elif c.lower() in seq_norm_lookup:
            all_dataset_seqs[cname] = circrna_seqs[seq_norm_lookup[c.lower()]]

    if verbose:
        print(f"  Sequence coverage: {len(all_dataset_seqs)}/{len(circ_name_set)} "
              f"({len(all_dataset_seqs)/max(len(circ_name_set),1)*100:.1f}%)")

    # Filter HMDD miRNA names to those with miRBase mapping
    valid_mirnas = [m for m in hmdd_mirna_names if m in mirna_name_map]

    predicted_pairs = []
    if all_dataset_seqs:
        predicted_pairs = predict_with_fallback(
            all_dataset_seqs, valid_mirnas, mirna_name_map, mirbase_seqs,
            min_circ_coverage_pct=min_circ_coverage_pct,
            verbose=verbose,
        )

    pred_covered_circs = set(p[1] for p in predicted_pairs)

    # ── Phase 3: Merge ──
    all_pairs = list(khgat_set) + [(m, c) for m, c in predicted_pairs
                                    if (m, c) not in khgat_set]

    total_covered = len(set(p[1] for p in all_pairs))
    coverage_pct = total_covered / max(len(circ_name_set), 1) * 100

    if verbose:
        print(f"\n  === {dataset_name} Final Results ===")
        print(f"  KHGAT:       {len(khgat_set):>8} pairs, {len(khgat_covered_circs):>5} circRNAs")
        print(f"  Predicted:   {len(predicted_pairs):>8} pairs, {len(pred_covered_circs):>5} circRNAs")
        print(f"  Combined:    {len(all_pairs):>8} pairs, {total_covered:>5}/{len(circ_name_set)} "
              f"circRNAs ({coverage_pct:.1f}%)")

    # ── Save output ──
    os.makedirs(output_dir, exist_ok=True)

    # Save in KHGAT-compatible TSV format (miRNA_name \t weight \t circRNA_name)
    out_path = os.path.join(output_dir, f"{dataset_name}_circrna_mirna_pairs.txt")
    with open(out_path, 'w') as f:
        for mirna_name, circ_name in all_pairs:
            f.write(f"{mirna_name}\t1\t{circ_name}\n")

    # Save coverage report
    report_path = os.path.join(output_dir, f"{dataset_name}_mirna_coverage.json")
    report = {
        'dataset': dataset_name,
        'total_circrnas': len(circ_name_set),
        'circrnas_with_seq': len(all_dataset_seqs),
        'circrnas_with_mirna': total_covered,
        'coverage_pct': round(coverage_pct, 2),
        'khgat_pairs': len(khgat_set),
        'predicted_pairs': len(predicted_pairs),
        'combined_pairs': len(all_pairs),
        'unique_mirnas': len(set(p[0] for p in all_pairs)),
        'seed_type': '7mer-m8 + 6mer fallback',
        'min_sites_7mer': 2,
        'min_sites_6mer': 3,
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    if verbose:
        print(f"  Saved: {out_path}")
        print(f"  Saved: {report_path}")

    return report


# =============================================================================
# Main entry point for standalone use
# =============================================================================
if __name__ == "__main__":
    """
    Usage:
      python data/predict_mirna_interactions.py --dataset circR2Disease
      python data/predict_mirna_interactions.py --all
    """
    import argparse
    parser = argparse.ArgumentParser(description="Predict circRNA-miRNA interactions")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--min-coverage", type=float, default=95.0)
    args = parser.parse_args()

    sys.path.insert(0, _PROJ_ROOT)

    # Load all circRNA sequences from cached JSON
    seq_path = os.path.join(_CACHE_DIR, "circrna_sequences.json")
    all_circ_seqs = {}
    if os.path.exists(seq_path):
        with open(seq_path) as f:
            all_circ_seqs = json.load(f)
        print(f"Loaded {len(all_circ_seqs)} circRNA sequences from {seq_path}")
    else:
        print(f"WARNING: circrna_sequences.json not found at {seq_path}")

    # Load KHGAT circRNA-miRNA pairs
    khgat_path = os.path.join(_PROJ_ROOT, "data", "CompareExperiment",
                               "AE-RF main", "KHGAT", "data", "circRNA_miRNA.txt")
    khgat_pairs = []
    if os.path.exists(khgat_path):
        with open(khgat_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    khgat_pairs.append((parts[0].strip(), parts[2].strip()))

    # Load HMDD miRNA names
    hmdd_path = os.path.join(_CACHE_DIR, "hmdd_v4_cancer_mirna_disease.txt")
    hmdd_mirnas = []
    if os.path.exists(hmdd_path):
        with open(hmdd_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    hmdd_mirnas.append(parts[0].strip())
        hmdd_mirnas = list(dict.fromkeys(hmdd_mirnas))  # unique, preserve order

    datasets_to_run = []
    if args.all:
        datasets_to_run = ["circR2Cancer", "circR2Disease", "circRNADisease",
                           "circ2Traits", "circad"]
    elif args.dataset:
        datasets_to_run = [args.dataset]
    else:
        print("Specify --dataset or --all")
        sys.exit(1)

    for ds in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"Processing {ds}...")

        # Import dataset to get circRNA/disease names
        ds_path = os.path.join(_PROJ_ROOT, "data", ds, "dataset.py")
        if not os.path.exists(ds_path):
            print(f"  SKIP: dataset.py not found at {ds_path}")
            continue

        import importlib.util
        spec = importlib.util.spec_from_file_location(f'ds_{ds}', ds_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        circ_names = list(mod.DATASET.get('circrna_names', []))
        disease_names = list(mod.DATASET.get('disease_names', []))

        build_and_save_interactions(
            dataset_name=ds,
            circrna_names=circ_names,
            disease_names=disease_names,
            circrna_seqs=all_circ_seqs,
            khgat_circ_mirna_pairs=khgat_pairs,
            hmdd_mirna_names=hmdd_mirnas,
            min_circ_coverage_pct=args.min_coverage,
            verbose=True,
        )

    print("\nDone!")
