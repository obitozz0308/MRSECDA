"""
External Biological Knowledge for circRNA-Disease Association Prediction

Three knowledge sources:
1. MeSH Disease Semantic Similarity — via NCBI E-utilities API + Wang's method
2. circRNA Sequence k-mer Similarity — via circBase API + 4-mer cosine similarity
3. GO Host-Gene Functional Similarity — via host gene mapping + GO DAG + Wang's method

All sources use lazy loading with local caching (data/external/*.npy).
Fallback: return None if data unavailable → pipeline degrades to GIP-only.
"""

import os
import re
import json
import time
import warnings
from typing import Dict, List, Optional, Set, Tuple, Callable
from collections import defaultdict

import numpy as np
from scipy.io import loadmat

warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external")


def _get_cache_path(key: str) -> str:
    """Return absolute path for a cached file."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"{key}.npy")


def _get_json_cache_path(key: str) -> str:
    """Return absolute path for a cached JSON file."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"{key}.json")


def _load_or_compute_npy(key: str, compute_fn: Callable[[], Optional[np.ndarray]]) -> Optional[np.ndarray]:
    """
    Lazy-load pattern: return cached .npy if exists, else compute, save, and return.
    If compute_fn raises, log warning and return None.
    """
    path = _get_cache_path(key)
    if os.path.exists(path):
        try:
            return np.load(path)
        except Exception:
            pass  # Corrupted cache, recompute
    try:
        result = compute_fn()
        if result is not None:
            np.save(path, result.astype(np.float32))
        return result
    except Exception as e:
        print(f"  [ExternalKnowledge] WARNING: Failed to compute '{key}': {e}")
        return None


def _load_or_compute_json(key: str, compute_fn: Callable[[], Optional[dict]]) -> Optional[dict]:
    """Same pattern for JSON-serializable data."""
    path = _get_json_cache_path(key)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    try:
        result = compute_fn()
        if result is not None:
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
        return result
    except Exception as e:
        print(f"  [ExternalKnowledge] WARNING: Failed to compute '{key}': {e}")
        return None


# =============================================================================
# 1. MeSH Disease Semantic Similarity (Wang's Method)
# =============================================================================

def _fetch_mesh_uid(disease_name: str, timeout: float = 10.0) -> Optional[str]:
    """
    Map a disease name to MeSH Unique ID via NCBI E-utilities esearch.

    Uses progressive search strategies:
    1. Exact match with quotes: "{name}"[mh]
    2. Loose match without quotes: {name}[mh]
    3. Core term only (remove leading qualifiers)
    4. Generic search without [mh] tag

    Returns MeSH UID (e.g., 'D005910') or None.
    """
    import requests
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    # Strategy: try progressively, stop at first match
    strategies = [
        f'"{disease_name}"[mh]',           # exact MeSH heading
        f'{disease_name}[mh]',             # loose
    ]

    # Progressively shorter names (only if multi-word)
    words = disease_name.split()
    if len(words) > 2:
        strategies.append(f'{" ".join(words[1:])}[mh]')    # remove first word
    if len(words) > 1:
        strategies.append(f'{" ".join(words[-2:])}[mh]')   # last two words

    for term in strategies:
        try:
            params = {"db": "mesh", "term": term, "retmax": 1, "retmode": "json"}
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code != 200:
                continue
            data = r.json()
            idlist = data.get("esearchresult", {}).get("idlist", [])
            if idlist:
                return idlist[0]
        except Exception:
            continue

    return None


def _fetch_mesh_tree_numbers(mesh_uid: str, timeout: float = 10.0) -> List[str]:
    """
    Fetch MeSH tree numbers for a given MeSH UID.
    Tree numbers encode the DAG hierarchy, e.g.:
      'C04.557.465.625.600.380' (Glioma)
    The tree number prefix is the parent: 'C04.557.465.625.600' → ancestor.
    """
    import requests
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "mesh", "id": mesh_uid, "retmode": "text"}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            return []
        text = r.text
        # Parse "Tree Number(s): ..." line
        match = re.search(r"Tree Number\(s\):\s*(.+)", text)
        if not match:
            return []
        tree_line = match.group(1)
        # Split by comma, strip whitespace
        tree_nums = [t.strip() for t in tree_line.split(",") if t.strip()]
        return tree_nums
    except Exception:
        return []


def _parse_tree_number(tree_num: str) -> List[str]:
    """
    Decompose a MeSH tree number into its ancestor chain.
    'C04.557.465.625.600.380' → ['C04', 'C04.557', 'C04.557.465', ...]
    Each prefix is an ancestor node in the DAG.
    """
    parts = tree_num.split(".")
    ancestors = []
    for i in range(1, len(parts) + 1):
        ancestors.append(".".join(parts[:i]))
    return ancestors


def _build_mesh_dag(disease_names: List[str], data_dir: str, cache_prefix: str = "") -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """
    Build the MeSH DAG for a list of disease names.

    Steps:
    1. Map each disease name → MeSH UID (with caching)
    2. Fetch tree numbers for each MeSH UID (with caching)
    3. Build ancestor sets from tree numbers

    Returns:
      term_ancestors: Dict[tree_number → Set of ancestor tree_numbers]
      disease_trees: Dict[disease_name → primary tree_number]
    """
    tag = f"_{cache_prefix}" if cache_prefix else ""

    # Load or build MeSH uid mapping
    uid_cache_key = f"mesh_disease_uids{tag}"
    disease_uids = _load_or_compute_json(uid_cache_key, lambda: _build_disease_uid_map(disease_names))

    # Load or build tree number mapping
    tree_cache_key = f"mesh_disease_trees{tag}"
    disease_trees = _load_or_compute_json(tree_cache_key, lambda: _build_disease_tree_map(disease_names, disease_uids))

    if disease_trees is None:
        return {}, {}

    # Build DAG: collect all tree numbers and their ancestors
    dag_cache_key = f"mesh_dag{tag}"
    dag_data = _load_or_compute_json(dag_cache_key, lambda: _construct_dag(disease_trees))

    if dag_data is None:
        return {}, {}

    # Convert JSON sets back to Python sets
    term_ancestors = {k: set(v) for k, v in dag_data.items()}
    return term_ancestors, disease_trees


def _build_disease_uid_map(disease_names: List[str]) -> Optional[Dict[str, str]]:
    """Fetch MeSH UIDs for all disease names."""
    print("  [MeSH] Fetching MeSH UIDs for {} diseases via NCBI E-utilities...".format(len(disease_names)))
    uid_map = {}
    n_found = 0
    for i, name in enumerate(disease_names):
        uid = _fetch_mesh_uid(name)
        if uid:
            uid_map[name] = uid
            n_found += 1
        else:
            print(f"    [{i+1}/{len(disease_names)}] '{name}' → NOT FOUND")
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(disease_names)}] Processed, {n_found} found so far")
        time.sleep(0.35)  # Rate limiting (NCBI allows 3/sec without API key)
    print(f"  [MeSH] Found UIDs for {n_found}/{len(disease_names)} diseases")
    return uid_map if n_found > 0 else None


def _build_disease_tree_map(disease_names: List[str], disease_uids: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Fetch primary tree number for each disease's MeSH UID."""
    if disease_uids is None:
        return None
    print("  [MeSH] Fetching tree numbers for {} MeSH UIDs...".format(len(disease_uids)))
    tree_map = {}
    n_found = 0
    items = list(disease_uids.items())
    for i, (name, uid) in enumerate(items):
        trees = _fetch_mesh_tree_numbers(uid)
        if trees:
            # Use the first (most specific) tree number
            tree_map[name] = trees[0]
            n_found += 1
        else:
            print(f"    [{i+1}/{len(items)}] '{name}' (UID={uid}) → NO TREE NUMBERS")
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(items)}] Processed, {n_found} have tree numbers")
        time.sleep(0.35)
    print(f"  [MeSH] Retrieved tree numbers for {n_found}/{len(items)} diseases")
    return tree_map if n_found > 0 else None


def _construct_dag(disease_trees: Optional[Dict[str, str]]) -> Optional[Dict[str, List[str]]]:
    """Build the MeSH DAG: term → list of ancestor terms."""
    if disease_trees is None:
        return None
    term_ancestors: Dict[str, Set[str]] = {}
    for disease_name, tree_num in disease_trees.items():
        chain = _parse_tree_number(tree_num)
        # The most specific term (full tree number) gets all its prefixes as ancestors
        term = chain[-1]  # full tree number, e.g., 'C04.557.465.625.600.380'
        if term not in term_ancestors:
            term_ancestors[term] = set(chain[:-1])  # all prefixes are ancestors
        else:
            term_ancestors[term].update(chain[:-1])

        # Also register intermediate nodes
        for i, anc in enumerate(chain):
            if anc not in term_ancestors:
                term_ancestors[anc] = set(chain[:i])

    # Convert sets to sorted lists for JSON serialization
    return {k: sorted(v) for k, v in term_ancestors.items()}


def _wang_semantic_contribution(
    term: str,
    dag: Dict[str, Set[str]],
    weight: float = 0.8,
) -> Dict[str, float]:
    """
    Compute semantic contribution of a term to all its ancestors using Wang's method.

    SV(t) = 1 if t == target_term
    SV(t) = max{w * SV(child) for child in children_of_t} otherwise

    where w = 0.8 is the semantic contribution decay factor.
    """
    # Build children map (reverse of DAG)
    children = defaultdict(set)
    for node, ancestors in dag.items():
        for anc in ancestors:
            children[anc].add(node)

    # DFS from the target term
    sv = {}
    stack = [(term, 1.0)]  # (node, contribution_from_parent_chain)
    while stack:
        node, contrib = stack.pop()
        if node not in sv or contrib > sv[node]:
            sv[node] = contrib
        for child in children.get(node, set()):
            child_contrib = contrib * weight
            stack.append((child, child_contrib))

    return sv


def _wang_disease_similarity(
    disease_a: str,
    disease_b: str,
    disease_trees: Dict[str, str],
    dag: Dict[str, Set[str]],
    weight: float = 0.8,
) -> float:
    """
    Compute Wang semantic similarity between two diseases.

    S(A, B) = Σ_{t∈T_A∩T_B} (SV_A(t) + SV_B(t)) / (SV(A) + SV(B))

    where SV_A(t) is the semantic value of term t contributed by disease A,
    and SV(A) = Σ_{t∈T_A} SV_A(t) is the total semantic value of disease A.
    """
    if disease_a not in disease_trees or disease_b not in disease_trees:
        return 0.0

    term_a = disease_trees[disease_a]
    term_b = disease_trees[disease_b]

    if not term_a or not term_b:
        return 0.0

    sv_a = _wang_semantic_contribution(term_a, dag, weight)
    sv_b = _wang_semantic_contribution(term_b, dag, weight)

    # Find common ancestors
    common = set(sv_a.keys()) & set(sv_b.keys())
    if not common:
        return 0.0

    numerator = sum(sv_a.get(t, 0.0) + sv_b.get(t, 0.0) for t in common)
    denominator = sum(sv_a.values()) + sum(sv_b.values())

    if denominator == 0:
        return 0.0

    return numerator / denominator


def load_mesh_similarity(
    disease_names: List[str],
    data_dir: str = "./data/CompareExperiment/Data",
    use_cache: bool = True,
    cache_key: str = "mesh_disease_sim",
    cache_prefix: str = "",
) -> Optional[np.ndarray]:
    """
    Load/compute MeSH disease semantic similarity matrix (N×N, [0,1]).

    Uses Wang's method on the MeSH DAG, with tree numbers fetched
    from NCBI E-utilities and cached locally.
    """
    N = len(disease_names)
    prefix = cache_prefix or cache_key.replace("mesh_disease_sim", "").lstrip("_")

    def compute_fn():
        dag, disease_trees = _build_mesh_dag(disease_names, data_dir, cache_prefix=prefix)
        if not dag or not disease_trees:
            print("  [MeSH] Could not build MeSH DAG — falling back to GIP-only")
            return None

        n_with_trees = sum(1 for d in disease_names if d in disease_trees)
        print(f"  [MeSH] {n_with_trees}/{N} diseases have tree numbers. Computing Wang similarity...")

        sim = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            sim[i, i] = 1.0
            if disease_names[i] not in disease_trees:
                continue
            for j in range(i + 1, N):
                if disease_names[j] not in disease_trees:
                    continue
                s = _wang_disease_similarity(
                    disease_names[i], disease_names[j],
                    disease_trees, dag,
                )
                sim[i, j] = s
                sim[j, i] = s

        n_pairs = int((sim > 0).sum()) - N  # exclude diagonal
        print(f"  [MeSH] Computed similarity for {n_pairs} non-zero disease pairs")
        return sim

    if use_cache:
        return _load_or_compute_npy(cache_key, compute_fn)
    return compute_fn()


# =============================================================================
# 2. circRNA Sequence k-mer Similarity
# =============================================================================

# Shared circBase info cache — used by both k-mer and GO pipelines
def _fetch_all_circbase_info(
    circrna_ids: List[str],
    timeout: float = 5.0,
    max_workers: int = 5,
) -> Optional[Dict[str, Dict[str, str]]]:
    """
    Fetch circBase annotation for all relevant circRNAs (shared cache).

    Now uses ID normalization to try multiple variant forms of each ID,
    dramatically improving coverage for IDs with format variations
    (e.g., 'hsa circ 0031288' → 'hsa_circ_0031288').

    Returns:
        Dict[original_circrna_id → {'gene_symbol': str, 'best_transcript': str, 'position': str, 'strand': str}]
        Cached to circbase_info.json
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import requests as _requests

    # ── Build candidate set: original IDs + normalized variants ──
    # Map each query variant → original ID for result mapping
    variant_to_original: Dict[str, str] = {}
    query_variants: List[str] = []

    for cid in circrna_ids:
        # Determine if this ID could possibly be in circBase
        is_candidate = False
        # Standard hsa_circ_* / hsa_circRNA_*
        if re.search(r'^hsa[_\s]+circ', cid, re.IGNORECASE) and re.search(r'\d+$', cid):
            is_candidate = True
        # circRNA_XXXXX (numeric-only, without hsa prefix)
        elif re.match(r'^circRNA_\d+$', cid, re.IGNORECASE):
            is_candidate = True
        # IDs with format issues that normalization can fix
        elif re.search(r'^hsa[-_\s]+circ', cid, re.IGNORECASE):
            is_candidate = True
        elif re.match(r'^hsa-circRNA\d+', cid, re.IGNORECASE):
            is_candidate = True

        if is_candidate:
            variants = _normalize_circrna_id(cid)
            for v in variants:
                if v not in variant_to_original:
                    variant_to_original[v] = cid
                    query_variants.append(v)

    if not query_variants:
        print("  [circBase] No circBase-compatible IDs found")
        return {}

    # Deduplicate queries (different original IDs may share normalized forms)
    unique_queries = list(dict.fromkeys(query_variants))
    print(f"  [circBase] Fetching info for {len(circrna_ids)} IDs "
          f"({len(unique_queries)} unique query variants)...")

    info_map: Dict[str, Dict[str, str]] = {}
    n_success = 0
    n_new_from_normalization = 0

    def _fetch_one(query_id: str) -> Tuple[str, Optional[Dict[str, str]]]:
        """Fetch circBase info for a single circRNA query variant."""
        url = f"http://circbase.org/cgi-bin/singlerecord.cgi?id={query_id.strip()}"
        try:
            r = _requests.get(url, timeout=timeout)
            if r.status_code != 200:
                return query_id, None
            text = r.text
            info: Dict[str, str] = {}
            gene_match = re.search(r'Gene symbol</td>\s*<td[^>]*><a[^>]*>([^<]+)</a>', text)
            if gene_match:
                info['gene_symbol'] = gene_match.group(1).strip().upper()
            tx_match = re.search(r'Best transcript</td>\s*<td[^>]*><a[^>]*>([^<]+)</a>', text)
            if tx_match:
                info['best_transcript'] = tx_match.group(1).strip()
            pos_match = re.search(r'position \(genome browser link\)</th>.*?<a[^>]*>([^<]+)</a>', text, re.DOTALL)
            if pos_match:
                info['position'] = pos_match.group(1).strip()
            strand_match = re.search(r'<th>strand</th>.*?<td[^>]*>([\+-])</td>', text, re.DOTALL)
            if strand_match:
                info['strand'] = strand_match.group(1).strip()
            return query_id, info if info else None
        except Exception:
            return query_id, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, qid): qid for qid in unique_queries}
        for i, future in enumerate(as_completed(futures)):
            query_id, info = future.result()
            if info:
                # Map result back to original ID
                original_id = variant_to_original.get(query_id, query_id)
                if original_id not in info_map:
                    info_map[original_id] = info
                    n_success += 1
                    # Track whether this came from a normalized variant
                    if query_id != original_id and query_id != original_id.strip():
                        n_new_from_normalization += 1
            if (i + 1) % 50 == 0:
                print(f"    [{i+1}/{len(unique_queries)}] Fetched, {n_success} with data so far")
            # Small delay between submissions to be polite to circBase
            time.sleep(0.05)

    print(f"  [circBase] Got info for {n_success}/{len(circrna_ids)} circRNAs "
          f"({n_new_from_normalization} from ID normalization)")
    return info_map if n_success > 0 else {}


def _extract_gene_name_from_id(circrna_id: str) -> Optional[str]:
    """
    Extract host gene symbol from a circRNA ID when it's gene-name-based.
    Examples:
      'circ-Foxo3' → 'FOXO3'
      'Circ_HIPK3' → 'HIPK3'
      'circZFR' → 'ZFR'
      'cdr1as' → 'CDR1'
      'circMAN2B2/hsa_circRNA_103595' → 'MAN2B2'
      'hsa_circ_0000977' → None (numeric ID, needs circBase)
    """
    # Skip purely numeric / coordinate-based IDs
    if re.match(r"^hsa_circ_\d+$", circrna_id):
        return None
    if re.match(r"^Chr\d+:", circrna_id):
        return None  # chromosome coordinates (needs UCSC)
    if re.match(r"^circRNA-?chr", circrna_id, re.IGNORECASE):
        return None
    if re.match(r"^MCF7_circ", circrna_id):
        return None  # cell-line specific
    if re.match(r"^Fusion-?circ", circrna_id):
        return None  # fusion circRNA, no single gene
    if re.match(r"^Titin\s+circ", circrna_id):
        return "TTN"  # Titin circRNAs

    # Hybrid with gene name suffix: hsa_circRNA_XXXXXX/circGENE
    m = re.match(r"^hsa[\s_]circ\w+\d+/circ([A-Za-z][A-Za-z0-9]*)", circrna_id, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # hsa_circ*-style with numbers — needs circBase (but NOT if has /circGENE suffix)
    if re.match(r"^hsa[\s_]circ\w*\d+", circrna_id, re.IGNORECASE):
        return None

    # Numeric-only circRNA_* (like circRNA_100782, circRNA120) — needs circBase
    if re.match(r"^circRNA[_\s]?\d+$", circrna_id, re.IGNORECASE):
        return None

    # Pattern 0: 'circRNA_GENE' (non-numeric gene name) e.g., circRNA_Atp9b → ATP9B
    m = re.match(r"^circRNA[_\s]?([A-Za-z][A-Za-z0-9]+)$", circrna_id, re.IGNORECASE)
    if m:
        gene = m.group(1).upper()
        if not gene.isdigit() and len(gene) >= 2 and not gene.startswith("CHR"):
            return gene

    # Pattern 1: 'circ-GENE' or 'circGENE' or 'circ_GENE'
    m = re.match(r"^circ[-_\s]?([A-Za-z][A-Za-z0-9]+)$", circrna_id, re.IGNORECASE)
    if m:
        gene = m.group(1).upper()
        if not gene.isdigit() and len(gene) >= 2:
            return gene

    # Pattern 3a: 'circGENE/hsa_circRNA_XXXXX' (hybrid), e.g., circMAN2B2/hsa_circRNA_103595
    m = re.match(r"^circ([A-Za-z][A-Za-z0-9]*)/", circrna_id, re.IGNORECASE)
    if m:
        gene = m.group(1).upper()
        # Filter out non-gene prefixes and numeric-like names
        if gene == "RNA" or gene.isdigit() or re.match(r'^RNA\d+$', gene):
            return None  # not a real gene name
        if not re.search(r'[A-Z]', gene):
            return None
        if len(gene) >= 2:
            return gene

    # Pattern 3b: 'circRNA_GENE/circ_XXXXX' e.g., circRNA_Atp9b/circ_15898
    m = re.match(r"^circRNA[_\s]?([A-Za-z][A-Za-z0-9]*)/", circrna_id, re.IGNORECASE)
    if m:
        gene = m.group(1).upper()
        # Must contain at least one letter (real gene name, not numeric like 0047905)
        if not gene.isdigit() and re.search(r'[A-Z]', gene) and len(gene) >= 2:
            return gene
        return None  # numeric, needs circBase

    # Pattern 3c: 'hsa_circRNA_XXXXXX/circGENE' e.g., hsa_circRNA_100338/circSNX27
    m = re.match(r"^hsa[\s_]circ\w+\d+/circ([A-Za-z][A-Za-z0-9]*)", circrna_id, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern 4a: 'CircSLC8A1-1' → 'SLC8A1'
    m = re.match(r"^circ([A-Za-z0-9]+)-\d+", circrna_id, re.IGNORECASE)
    if m:
        gene = m.group(1).upper()
        if not gene.isdigit():
            return gene

    # Pattern 5: 'cANRIL', 'cdr1as' — well-known named circRNAs
    m = re.match(r"^c([A-Z0-9]+)", circrna_id, re.IGNORECASE)
    if m and len(m.group(1)) >= 3:
        gene = m.group(1).upper()
        if not gene.isdigit():
            # Special cases: well-known circRNAs with unusual names
            SPECIAL = {
                "DR1AS": "CDR1",
            }
            if gene in SPECIAL:
                return SPECIAL[gene]
            # Only return if looks like a gene name (all alpha or alpha+suffix)
            if re.match(r"^[A-Z]+$", gene) and gene not in ("HR", "RNA", "CIRC", "CIRCRNA"):
                return gene
        return None

    # Manual mapping for known special cases
    SPECIAL_GENES = {
        "cdr1as": "CDR1",
        "circhipk3": "HIPK3",
        "circfoxo3": "FOXO3",
        "circzfr": "ZFR",
        "circfat1": "FAT1",
        "cirsmarca5": "SMARCA5",
        "circslc8a1": "SLC8A1",
    }
    lower_id = circrna_id.lower().replace(" ", "")
    if lower_id in SPECIAL_GENES:
        return SPECIAL_GENES[lower_id]

    return None


# =============================================================================
# ID Normalization — fix format variations for better circBase matching
# =============================================================================

def _normalize_circrna_id(circrna_id: str) -> List[str]:
    """
    Generate normalized variant(s) of a circRNA ID for database queries.

    Fixes common format issues:
      - 'hsa circ 0031288'       → 'hsa_circ_0031288' (space → underscore)
      - 'hsa_ circRNA_005086'    → 'hsa_circRNA_005086' (trailing space after prefix)
      - 'hsa_circRNA 102049'     → 'hsa_circRNA_102049' (missing underscore)
      - 'hsa_circRNA_ 100192'    → 'hsa_circRNA_100192' (extra space)
      - 'hsa-circRNA11783-2'     → 'hsa_circRNA_11783' (dash format)
      - 'hsa_circRNA_000167'     → also try 'hsa_circ_000167' (circ → circRNA swap)
      - 'hsa_ circRNA_104387'    → 'hsa_circRNA_104387' (space + underscore mix)

    Returns a list of variants to try, ordered by likelihood.
    """
    variants = []
    cid = circrna_id.strip()

    # Always include the stripped original
    variants.append(cid)

    # Normalize all whitespace
    collapsed = re.sub(r'\s+', '_', cid)
    if collapsed != cid:
        variants.append(collapsed)

    # Remove double underscores
    dedup = re.sub(r'__+', '_', cid)
    if dedup != cid and dedup not in variants:
        variants.append(dedup)
    dedup2 = re.sub(r'__+', '_', collapsed)
    if dedup2 not in variants:
        variants.append(dedup2)

    # Fix hsa_ circRNA → hsa_circRNA (trailing space after prefix)
    fixed_prefix = re.sub(r'^(hsa)_\s+circ', r'\1_circ', cid, flags=re.IGNORECASE)
    if fixed_prefix != cid and fixed_prefix not in variants:
        variants.append(fixed_prefix)

    # Fix hsa circ → hsa_circ (space between prefix and circ)
    fixed_space = re.sub(r'^(hsa)\s+circ', r'\1_circ', cid, flags=re.IGNORECASE)
    if fixed_space != cid and fixed_space not in variants:
        variants.append(fixed_space)

    # Fix hsa-circRNA → hsa_circRNA (dash format)
    fixed_dash = re.sub(r'^(hsa)-circ', r'\1_circ', cid, flags=re.IGNORECASE)
    if fixed_dash != cid and fixed_dash not in variants:
        variants.append(fixed_dash)

    # Fix hsa_circRNA_ → hsa_circRNA (space after last underscore)
    fixed_trail = re.sub(r'^(hsa_circ\w*)_\s+', r'\1_', cid, flags=re.IGNORECASE)
    if fixed_trail != cid and fixed_trail not in variants:
        variants.append(fixed_trail)

    # For hsa_circRNA_NNNNN: also try hsa_circ_NNNNN (circBase uses both prefixes)
    m = re.match(r'^(hsa)_circRNA_(\d+)$', cid, re.IGNORECASE)
    if m:
        alt = f"{m.group(1).lower()}_circ_{m.group(2)}"
        if alt not in variants:
            variants.append(alt)
        # Also try without trailing -N suffix: hsa-circRNA11783-2 → hsa_circRNA_11783
        alt2 = f"hsa_circRNA_{m.group(2)}"
        if alt2 not in variants:
            variants.append(alt2)

    # For hsa-circRNA11783-2: extract the numeric part
    m2 = re.match(r'^hsa-circRNA(\d+)', cid, re.IGNORECASE)
    if m2:
        alt = f"hsa_circRNA_{m2.group(1)}"
        if alt not in variants:
            variants.append(alt)
        alt2 = f"hsa_circ_{m2.group(1)}"
        if alt2 not in variants:
            variants.append(alt2)

    # For hsa_circ_N+ or hsa_circRNA_N+ with <7 digits: pad to 7 digits.
    # circBase uses exactly 7-digit IDs (e.g. hsa_circ_0001891).
    # Old datasets often use 6-digit IDs (e.g. hsa_circ_001891) that aren't
    # directly recognized — zero-padding fixes this.
    for prefix in ['hsa_circ_', 'hsa_circRNA_']:
        m_pad = re.match(rf'^({prefix})(\d+)$', cid, re.IGNORECASE)
        if m_pad:
            digits = m_pad.group(2)
            if len(digits) < 7:
                padded_prefix = m_pad.group(1).lower()
                padded = f"{padded_prefix}{digits:0>7}"
                if padded not in variants:
                    variants.append(padded)
                # Also try circ variant when original is circRNA
                if padded_prefix == 'hsa_circrna_':
                    alt_pad = padded.replace('hsa_circrna_', 'hsa_circ_')
                    if alt_pad not in variants:
                        variants.append(alt_pad)
            break  # only one prefix matches

    # Species prefix normalization (mmu → mouse, rno → rat for Ensembl)
    for species_prefix, species_name in [('mmu', 'mouse'), ('rno', 'rat')]:
        m3 = re.match(rf'^({species_prefix})_circ_(\d+)$', cid, re.IGNORECASE)
        if m3:
            # Standardize: mmu_circ_0001544 → normalized form
            std = f"{species_prefix}_circ_{m3.group(2)}"
            if std not in variants:
                variants.append(std)
        m4 = re.match(rf'^({species_prefix})_circRNA_(\d+)$', cid, re.IGNORECASE)
        if m4:
            std = f"{species_prefix}_circRNA_{m4.group(2)}"
            if std not in variants:
                variants.append(std)
            # Also try circ variant
            alt_circ = f"{species_prefix}_circ_{m4.group(2)}"
            if alt_circ not in variants:
                variants.append(alt_circ)

    # ── circ_N / circ_N+ → hsa_circ_N (add human species prefix) ──
    m_circ = re.match(r'^circ_(\d+)$', cid, re.IGNORECASE)
    if m_circ:
        digits = m_circ.group(1)
        alt = f"hsa_circ_{digits:0>7}"
        if alt not in variants:
            variants.append(alt)
        # Also try without padding
        alt2 = f"hsa_circ_{digits}"
        if alt2 not in variants:
            variants.append(alt2)
        # Also try hsa_circRNA_ variant
        alt3 = f"hsa_circRNA_{digits:0>7}"
        if alt3 not in variants:
            variants.append(alt3)

    # ── circrnaNNNN (no separators) → hsa_circ_XXXXXXXXX ──
    m_crna = re.match(r'^circrna[_\-\s]*(\d+)$', cid, re.IGNORECASE)
    if m_crna:
        digits = m_crna.group(1)
        alt = f"hsa_circ_{digits:0>7}"
        if alt not in variants:
            variants.append(alt)
        alt2 = f"hsa_circRNA_{digits:0>7}"
        if alt2 not in variants:
            variants.append(alt2)

    # ── circrna-NNNNNN → hsa_circ_00NNNNNN (fix hyphen-format circRNA IDs) ──
    m_crna2 = re.match(r'^circrna[_\-\s]+(\d+)$', cid, re.IGNORECASE)
    if m_crna2 and not m_crna:  # don't duplicate the previous pattern
        digits = m_crna2.group(1)
        alt = f"hsa_circ_{digits:0>7}"
        if alt not in variants:
            variants.append(alt)

    # ── hsa_circrna_N+ (no underscore between circrna and number) ──
    m_hcrna = re.match(r'^hsa[_\-\s]*circrna[_\-\s]*(\d+)$', cid, re.IGNORECASE)
    if m_hcrna:
        digits = m_hcrna.group(1)
        alt = f"hsa_circ_{digits:0>7}"
        if alt not in variants:
            variants.append(alt)
        alt2 = f"hsa_circRNA_{digits:0>7}"
        if alt2 not in variants:
            variants.append(alt2)

    # ── has_circ_* → hsa_circ_* (fix common typo: has → hsa) ──
    m_has = re.match(r'^has[_\-\s]+circ[_\-\s]+(\d+)$', cid, re.IGNORECASE)
    if m_has:
        digits = m_has.group(1)
        alt = f"hsa_circ_{digits:0>7}"
        if alt not in variants:
            variants.append(alt)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            result.append(v)

    return result


def _fetch_circbase_info(circrna_id: str, timeout: float = 5.0) -> Optional[Dict[str, str]]:
    """
    Fetch circRNA annotation from circBase singlerecord page.

    Returns dict with keys:
      - 'gene_symbol': host gene symbol (e.g., 'NOL10')
      - 'best_transcript': RefSeq accession (e.g., 'NM_024894')
      - 'position': genomic position (e.g., 'chr2:10784445-10808849')
      - 'strand': '+' or '-'
    """
    import requests
    url = f"http://circbase.org/cgi-bin/singlerecord.cgi?id={circrna_id}"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        text = r.text

        info = {}

        # Gene symbol: <td class="field">GENE</td>
        gene_match = re.search(r'Gene symbol</td>\s*<td[^>]*><a[^>]*>([^<]+)</a>', text)
        if gene_match:
            info['gene_symbol'] = gene_match.group(1).strip().upper()

        # Best transcript: NM_* or NR_*
        tx_match = re.search(r'Best transcript</td>\s*<td[^>]*><a[^>]*>([^<]+)</a>', text)
        if tx_match:
            info['best_transcript'] = tx_match.group(1).strip()

        # Genomic position
        pos_match = re.search(r'position \(genome browser link\)</th>.*?<a[^>]*>([^<]+)</a>', text, re.DOTALL)
        if pos_match:
            info['position'] = pos_match.group(1).strip()

        # Strand
        strand_match = re.search(r'<th>strand</th>.*?<td[^>]*>([\+-])</td>', text, re.DOTALL)
        if not strand_match:
            # Try in the table row
            strand_match = re.search(r'<td[^>]*>([+-])</td>\s*<td[^>]*><a[^>]*>' + re.escape(circrna_id), text)
        if strand_match:
            info['strand'] = strand_match.group(1).strip()

        return info if info else None
    except Exception:
        return None


def _fetch_ncbi_transcript_sequence(accession: str, timeout: float = 10.0) -> Optional[str]:
    """
    Fetch transcript sequence from NCBI Nucleotide database.

    Args:
        accession: RefSeq accession (e.g., 'NM_024894')

    Returns:
        DNA sequence string or None
    """
    import requests
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "nuccore", "id": accession, "rettype": "fasta", "retmode": "text"}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            return None
        lines = r.text.strip().split("\n")
        if len(lines) < 2:
            return None
        seq = "".join(lines[1:]).upper().replace("U", "T")
        # Remove any non-ACGT characters
        seq = re.sub(r"[^ACGT]", "", seq)
        if len(seq) >= 20:
            return seq
        return None
    except Exception:
        return None


def _fetch_circbase_sequence(circrna_id: str, timeout: float = 5.0) -> Optional[str]:
    """
    Fetch circRNA sequence via circBase annotation + NCBI transcript.

    Strategy:
    1. Fetch circBase singlerecord page → get best transcript accession
    2. Fetch transcript sequence from NCBI E-utilities
    3. Fallback: try direct UCSC genome browser DNA fetch
    """
    info = _fetch_circbase_info(circrna_id, timeout)
    if info is None:
        return None

    # Try NCBI transcript first
    if "best_transcript" in info:
        seq = _fetch_ncbi_transcript_sequence(info["best_transcript"])
        if seq:
            return seq

    # Fallback: try to get sequence from UCSC
    if "position" in info:
        # Parse position: 'chr2:10784445-10808849'
        pos_match = re.match(r"chr(\w+):(\d+)-(\d+)", info["position"])
        if pos_match:
            chrom, start, end = pos_match.group(1), int(pos_match.group(2)), int(pos_match.group(3))
            seq = _fetch_ucsc_sequence(chrom, start, end, info.get("strand", "+"))
            if seq:
                return seq

    return None


def _count_kmers(sequence: str, k: int = 4) -> np.ndarray:
    """
    Compute k-mer frequency vector from a nucleotide sequence.

    Args:
        sequence: DNA sequence string (A, C, G, T only)
        k: k-mer size (default 4, giving 4^4 = 256 features)

    Returns:
        Normalized frequency vector of length 4^k
    """
    # Map nucleotides to indices
    nt_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    n_features = 4 ** k

    freq = np.zeros(n_features, dtype=np.float32)

    valid = 0
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if all(c in nt_to_idx for c in kmer):
            idx = 0
            for c in kmer:
                idx = idx * 4 + nt_to_idx[c]
            freq[idx] += 1.0
            valid += 1

    if valid > 0:
        freq /= valid  # Normalize to frequencies

    return freq


def _cosine_similarity_from_freqs(freq_matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity from a frequency matrix.

    Args:
        freq_matrix: shape (N, F) frequency vectors

    Returns:
        Similarity matrix shape (N, N), values in [0, 1]
    """
    # Normalize rows to unit length
    norms = np.linalg.norm(freq_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
    normalized = freq_matrix / norms
    sim = normalized @ normalized.T
    # Clip to [0, 1] (floating point may produce slightly negative values)
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32)


def compute_kmer_similarity(
    circrna_ids: List[str],
    k: int = 4,
    timeout: float = 5.0,
    max_workers: int = 5,
    variance_keep_ratio: float = 0.0,  # 0.0 = use all, 0.3 = keep top 30% high-variance
    cache_prefix: str = "",
) -> Optional[np.ndarray]:
    """
    Compute circRNA sequence k-mer cosine similarity.

    Fetches sequences from circBase with concurrent workers.
    Falls back gracefully if circBase is unavailable.

    Parameters
    ----------
    variance_keep_ratio : float
        If > 0, keep only this fraction of k-mers with highest variance across circRNAs.
        E.g., 0.3 keeps the top 30% most variable k-mers, discarding ubiquitous low-variance ones.
        Default 0.0 = use all 4^k k-mers (no filtering).
    """
    N = len(circrna_ids)

    # Determine cache key (different for pruned vs full)
    prune_tag = f"_var{int(variance_keep_ratio * 100)}" if variance_keep_ratio > 0 else ""
    cache_key = f"{cache_prefix}circrna_kmer_sim_{k}mer{prune_tag}"

    # Early return if the similarity matrix is already cached (avoid re-fetching sequences)
    _sim_path = _get_cache_path(cache_key)
    if os.path.exists(_sim_path):
        try:
            return np.load(_sim_path)
        except Exception:
            pass

    # Load or fetch sequences
    seq_cache_key = "circrna_sequences"
    sequences = _load_or_compute_json(seq_cache_key, lambda: _fetch_all_sequences(
        circrna_ids, timeout, max_workers
    ))

    if sequences is None or len(sequences) == 0:
        print("  [K-mer] No sequences available — falling back")
        return None

    n_with_seq = len(sequences)
    print(f"  [K-mer] {n_with_seq}/{N} circRNAs have sequences")

    if n_with_seq < N * 0.3:
        print("  [K-mer] Too few sequences (<30%) — falling back")
        return None

    def compute_sim_fn():
        # Build frequency matrix for all circRNAs (zeros for missing)
        n_total_features = 4 ** k
        freq_matrix = np.zeros((N, n_total_features), dtype=np.float32)
        for i, cid in enumerate(circrna_ids):
            seq = sequences.get(cid)
            if seq and len(seq) >= k:
                freq_matrix[i] = _count_kmers(seq, k)

        # ── Variance Thresholding ──
        has_seq = np.array([cid in sequences and len(sequences[cid]) >= k for cid in circrna_ids])
        if variance_keep_ratio > 0 and has_seq.any():
            # Compute variance of each k-mer across circRNAs WITH sequences
            freq_with_seq = freq_matrix[has_seq]
            variances = freq_with_seq.var(axis=0)  # shape (256,)
            n_keep = max(1, int(n_total_features * variance_keep_ratio))
            top_indices = np.argsort(variances)[-n_keep:]

            # Report what's being filtered
            kept_var = variances[top_indices]
            discarded_var = variances[np.argsort(variances)[:n_total_features - n_keep]]
            print(f"  [K-mer] Variance threshold: keeping top {n_keep}/{n_total_features} k-mers "
                  f"(var range: {kept_var.min():.6f}–{kept_var.max():.6f}, "
                  f"discarded range: {discarded_var.min():.6f}–{discarded_var.max():.6f})")

            # Filter frequency matrix to high-variance k-mers only
            freq_matrix = freq_matrix[:, top_indices]

        # Compute cosine similarity
        sim = _cosine_similarity_from_freqs(freq_matrix)

        # For circRNAs without sequences, fill with row average from those with sequences
        if not has_seq.all() and has_seq.any():
            avg_sim = sim[has_seq].mean(axis=0)
            for i in range(N):
                if not has_seq[i]:
                    sim[i] = avg_sim
                    sim[:, i] = avg_sim
                    sim[i, i] = 1.0

        n_nonzero = int((sim > 0).sum())
        mean_non_diag = sim[~np.eye(N, dtype=bool)].mean() if N > 1 else 0.0
        print(f"  [K-mer] Computed {k}-mer similarity: {n_nonzero} non-zero entries, "
              f"mean non-diag={mean_non_diag:.4f}")
        return sim

    return _load_or_compute_npy(cache_key, compute_sim_fn)


# =============================================================================
# Direct Gene → Sequence Pipeline (bypasses circBase bottleneck)
# =============================================================================

def _fetch_gene_refseq(gene_symbol: str, timeout: float = 10.0) -> Optional[str]:
    """
    Fetch the best RefSeq transcript accession for a human gene.

    Uses NCBI E-utilities:
      gene_symbol → esearch("gene", "{gene}[gene] + human[orgn]") → GeneID
      GeneID → efetch("gene", rettype="xml") → extract RefSeq RNAs

    Returns the best NM_* (or NR_*) accession, or None.
    """
    import requests
    import xml.etree.ElementTree as ET

    # Step 1: Search for Gene ID
    url_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "gene",
        "term": f"{gene_symbol}[gene] AND human[orgn]",
        "retmax": 3,
        "retmode": "json",
    }
    try:
        r = requests.get(url_search, params=params, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        idlist = data.get("esearchresult", {}).get("idlist", [])
        if not idlist:
            return None
    except Exception:
        return None

    # Step 2: Fetch gene record and extract RefSeq accessions
    gene_id = idlist[0]
    url_fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params2 = {"db": "gene", "id": gene_id, "rettype": "xml", "retmode": "xml"}
    try:
        r2 = requests.get(url_fetch, params=params2, timeout=timeout)
        if r2.status_code != 200:
            return None
        xml_text = r2.text

        # Try XML parsing first
        try:
            root = ET.fromstring(xml_text)
            refseq_accessions = []
            # Search for RefSeq RNA elements
            for elem in root.iter():
                tag = elem.tag.lower()
                if 'rna' in tag or 'refseq' in tag:
                    text = (elem.text or "").strip()
                    if text.startswith("NM_") or text.startswith("NR_"):
                        refseq_accessions.append(text)
                # Also check Gene-ref_mRNA elements
                if 'mrna' in tag or 'genomic' in tag:
                    for child in elem.iter():
                        child_text = (child.text or "").strip()
                        if child_text.startswith("NM_") or child_text.startswith("NR_"):
                            refseq_accessions.append(child_text)

            if refseq_accessions:
                # Prefer NM_ over NR_
                nm = [a for a in refseq_accessions if a.startswith("NM_")]
                if nm:
                    return nm[0]
                return refseq_accessions[0]
        except ET.ParseError:
            pass

        # Fallback: regex extraction
        nm_matches = re.findall(r'(NM_\d+\.?\d*)', xml_text)
        nr_matches = re.findall(r'(NR_\d+\.?\d*)', xml_text)
        if nm_matches:
            # Pick shortest (typically the canonical isoform)
            return sorted(nm_matches, key=len)[0]
        if nr_matches:
            return sorted(nr_matches, key=len)[0]

        return None
    except Exception:
        return None


def _fetch_gene_transcript_sequences(
    gene_symbols: Set[str],
    timeout: float = 10.0,
    max_workers: int = 5,
) -> Dict[str, str]:
    """
    Fetch representative transcript sequences for a set of gene symbols.

    Pipeline: gene_name → NCBI Gene → RefSeq accession → NCBI Nucleotide → sequence
    This bypasses circBase entirely, enabling sequence retrieval for circRNAs
    whose host gene is known but not registered in circBase.

    Uses SEQUENTIAL fetching with NCBI rate limiting (3 req/sec).
    Results are cached in data/external/gene_sequences.json.

    Returns:
        Dict[gene_symbol → DNA_sequence_string]
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import re as _re

    # Filter out obviously non-gene symbols
    real_genes = [g for g in gene_symbols
                  if _re.match(r'^[A-Z][A-Z0-9]+$', g)
                  and not g.startswith('RNA')
                  and len(g) >= 2]
    n_filtered = len(gene_symbols) - len(real_genes)
    if n_filtered > 0:
        print(f"  [Gene→Seq] Filtered {n_filtered} non-gene symbols from {len(gene_symbols)} total")

    print(f"  [Gene→Seq] Fetching RefSeq accessions for {len(real_genes)} genes (sequential, ~0.35s each)...")

    # Step 1: gene → RefSeq accession (SEQUENTIAL with NCBI rate limiting)
    gene_to_accession: Dict[str, str] = {}
    n_gene_found = 0

    for i, gene in enumerate(real_genes):
        time.sleep(0.35)  # NCBI rate limit: 3 req/sec
        acc = _fetch_gene_refseq(gene, timeout=timeout)
        if acc:
            gene_to_accession[gene] = acc
            n_gene_found += 1
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(real_genes)}] Gene→RefSeq: {n_gene_found} found so far")

    print(f"  [Gene→Seq] Found RefSeq accessions for {n_gene_found}/{len(real_genes)} genes")

    if n_gene_found == 0:
        return {}

    # Step 2: RefSeq → sequence (parallel — NCBI Nucleotide has higher limits)
    unique_accessions = list(set(gene_to_accession.values()))
    print(f"  [Gene→Seq] Fetching sequences for {len(unique_accessions)} unique accessions...")

    accession_to_seq: Dict[str, str] = {}
    n_seq_ok = 0

    def _fetch_seq(acc: str) -> Tuple[str, Optional[str]]:
        seq = _fetch_ncbi_transcript_sequence(acc, timeout=timeout)
        if seq and len(seq) >= 100:
            return acc, seq
        return acc, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_seq, acc): acc for acc in unique_accessions}
        for i, future in enumerate(as_completed(futures)):
            acc, seq = future.result()
            if seq:
                accession_to_seq[acc] = seq
                n_seq_ok += 1
            if (i + 1) % 50 == 0:
                print(f"    [{i+1}/{len(unique_accessions)}] NCBI seq fetches, {n_seq_ok} OK so far")

    print(f"  [Gene→Seq] Fetched sequences for {n_seq_ok}/{len(unique_accessions)} unique accessions")

    # Step 3: Map back to gene symbols
    result: Dict[str, str] = {}
    for gene, acc in gene_to_accession.items():
        seq = accession_to_seq.get(acc)
        if seq:
            result[gene] = seq

    print(f"  [Gene→Seq] Total: {len(result)}/{len(real_genes)} genes have sequences")
    return result


def _fetch_sequence_from_coordinates(circrna_id: str) -> Optional[str]:
    """
    Fetch circRNA sequence from genomic coordinates.

    Handles IDs like 'Chr22: 28943661' or 'Chr5: 158368701'.
    Uses UCSC Genome Browser DAS server (hg19).
    """
    import requests

    # Parse coordinate format: 'Chr22: 28943661' or 'Chr5: 158368701'
    # These are typically single-position markers — expand to a window
    m = re.match(r'^[Cc]hr(\w+):\s*(\d+)', circrna_id)
    if not m:
        return None

    chrom = m.group(1)
    pos = int(m.group(2))
    # Expand to a 5kb window around the position (typical circRNA size)
    start = max(0, pos - 2500)
    end = pos + 2500

    # Try UCSC DAS (already in this file)
    seq = _fetch_ucsc_sequence(chrom, start, end)
    if seq and len(seq) >= 100:
        return seq

    # Fallback: try Ensembl REST API (hg38)
    try:
        url = (f"https://rest.ensembl.org/sequence/region/human/"
               f"{chrom}:{start}:{end}:1"
               f"?content-type=text/plain")
        r = requests.get(url, timeout=10.0)
        if r.status_code == 200:
            seq = r.text.strip().upper().replace("U", "T")
            seq = re.sub(r"[^ACGT]", "", seq)
            if len(seq) >= 100:
                return seq
    except Exception:
        pass

    return None


# =============================================================================
# 2b. Unified Sequence Fetching (circBase + Gene Direct + Coordinates)
# =============================================================================

def _fetch_all_sequences(
    circrna_ids: List[str],
    timeout: float = 5.0,
    max_workers: int = 5,
) -> Optional[Dict[str, str]]:
    """
    Fetch sequences for all circRNAs using multiple pipelines:

    1. circBase → best_transcript → NCBI Nucleotide (primary)
    2. Host gene → NCBI Gene → RefSeq → NCBI Nucleotide (fallback)
    3. Genomic coordinates → UCSC/Ensembl (for Chr:pos IDs)

    Uses shared caches: circbase_info.json, host_gene_mapping.json,
    gene_sequences.json (new).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    sequences: Dict[str, str] = {}

    # ── Pipeline 1: circBase → NCBI transcript ──
    print("  [K-mer] Pipeline 1: circBase → NCBI transcript...")
    circbase_info = _load_or_compute_json(
        "circbase_info",
        lambda: _fetch_all_circbase_info(circrna_ids, timeout, max_workers),
    )

    n_from_circbase = 0
    if circbase_info and len(circbase_info) > 0:
        cid_to_accession = {}
        for cid, info in circbase_info.items():
            if "best_transcript" in info:
                cid_to_accession[cid] = info["best_transcript"]

        if cid_to_accession:
            print(f"  [K-mer] {len(cid_to_accession)} circRNAs have best_transcript accessions")
            unique_accessions = list(set(cid_to_accession.values()))
            print(f"  [K-mer] Fetching sequences for {len(unique_accessions)} unique accessions...")

            accession_to_seq: Dict[str, str] = {}
            n_acc_ok = 0

            def _fetch_seq(acc: str) -> Tuple[str, Optional[str]]:
                return acc, _fetch_ncbi_transcript_sequence(acc, timeout=10.0)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_fetch_seq, acc): acc for acc in unique_accessions}
                for i, future in enumerate(as_completed(futures)):
                    acc, seq = future.result()
                    if seq:
                        accession_to_seq[acc] = seq
                        n_acc_ok += 1
                    if (i + 1) % 50 == 0:
                        print(f"    [{i+1}/{len(unique_accessions)}] NCBI fetches, {n_acc_ok} OK")

            print(f"  [K-mer] Pipeline 1: {n_acc_ok}/{len(unique_accessions)} accessions fetched")

            for cid, acc in cid_to_accession.items():
                seq = accession_to_seq.get(acc)
                if seq:
                    sequences[cid] = seq
                    n_from_circbase += 1

    print(f"  [K-mer] Pipeline 1 result: {n_from_circbase}/{len(circrna_ids)} circRNAs with sequences")

    # ── Pipeline 2: Host gene → NCBI Gene → RefSeq → Sequence ──
    n_from_gene = 0
    still_missing = [cid for cid in circrna_ids if cid not in sequences]
    if still_missing:
        print(f"\n  [K-mer] Pipeline 2: Gene direct ({len(still_missing)} still missing)...")

        # Load host gene mapping
        gene_map = _load_or_compute_json(
            "host_gene_mapping",
            lambda: _build_host_gene_mapping(circrna_ids, timeout, max_workers),
        )

        if gene_map and len(gene_map) > 0:
            # Collect unique genes for circRNAs without sequences
            missing_genes: Set[str] = set()
            cid_to_gene: Dict[str, str] = {}
            for cid in still_missing:
                gene = gene_map.get(cid)
                if gene:
                    missing_genes.add(gene)
                    cid_to_gene[cid] = gene

            if missing_genes:
                print(f"  [K-mer] {len(missing_genes)} unique genes for {len(cid_to_gene)} missing circRNAs")

                # Load or compute gene→sequence cache
                gene_seqs = _load_or_compute_json(
                    "gene_sequences",
                    lambda: _fetch_gene_transcript_sequences(missing_genes, timeout, max_workers),
                )

                if gene_seqs and len(gene_seqs) > 0:
                    for cid, gene in cid_to_gene.items():
                        seq = gene_seqs.get(gene)
                        if seq:
                            sequences[cid] = seq
                            n_from_gene += 1

    print(f"  [K-mer] Pipeline 2 result: +{n_from_gene} from gene direct")

    # ── Pipeline 3: Coordinate-based IDs ──
    n_from_coords = 0
    still_missing_2 = [cid for cid in circrna_ids if cid not in sequences]
    coord_ids = [cid for cid in still_missing_2 if re.match(r'^[Cc]hr\d+:', cid)]
    if coord_ids:
        print(f"\n  [K-mer] Pipeline 3: Coordinate-based ({len(coord_ids)} IDs)...")
        for cid in coord_ids:
            seq = _fetch_sequence_from_coordinates(cid)
            if seq:
                sequences[cid] = seq
                n_from_coords += 1
                if n_from_coords <= 3:
                    print(f"    OK: {cid} — {len(seq)}bp")
        print(f"  [K-mer] Pipeline 3 result: +{n_from_coords} from coordinates")

    # ── Summary ──
    total = len(sequences)
    print(f"\n  [K-mer] === Sequence Summary ===")
    print(f"  [K-mer]   circBase → NCBI:   {n_from_circbase}")
    print(f"  [K-mer]   Gene direct:        {n_from_gene}")
    print(f"  [K-mer]   Coordinates:        {n_from_coords}")
    print(f"  [K-mer]   TOTAL:              {total}/{len(circrna_ids)} circRNAs have sequences")
    print(f"  [K-mer]   Still missing:      {len(circrna_ids) - total}")

    return sequences if len(sequences) > 0 else None


# =============================================================================
# 3. GO Host-Gene Functional Similarity
# =============================================================================

def _fetch_ucsc_sequence(chrom: str, start: int, end: int, strand: str = "+") -> Optional[str]:
    """
    Fetch DNA sequence from UCSC Genome Browser DAS server.

    Uses the hg19 assembly (matching circBase coordinates).
    """
    import requests
    # UCSC DAS server
    url = f"http://genome.ucsc.edu/cgi-bin/das/hg19/dna"
    params = {
        "segment": f"chr{chrom}:{start},{end}",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        # Parse DAS XML response for DNA sequence
        dna_match = re.search(r"<DNA[^>]*>([ACGTacgt\n\r]+)</DNA>", r.text, re.DOTALL)
        if dna_match:
            seq = dna_match.group(1).replace("\n", "").replace("\r", "").upper()
            if strand == "-":
                # Reverse complement
                complement = {"A": "T", "C": "G", "G": "C", "T": "A"}
                seq = "".join(complement.get(c, c) for c in reversed(seq))
            return seq
        return None
    except Exception:
        return None


def _fetch_host_gene_from_circbase(circrna_id: str, timeout: float = 5.0) -> Optional[str]:
    """
    Fetch host gene symbol for a circRNA from circBase.

    Uses _fetch_circbase_info to parse the singlerecord page.
    """
    info = _fetch_circbase_info(circrna_id, timeout)
    if info and "gene_symbol" in info:
        return info["gene_symbol"]
    return None


def _build_host_gene_mapping(
    circrna_ids: List[str],
    timeout: float = 5.0,
    max_workers: int = 5,
) -> Optional[Dict[str, str]]:
    """
    Build circRNA → host gene mapping.

    Strategy (multi-source):
    1. Extract gene names from gene-name-based circRNA IDs (circ-Foxo3 → FOXO3)
    2. For hsa_circ_* / hsa_circRNA_* IDs: load shared circbase_info.json cache
       (now with ID normalization for broader matching)
    3. For remaining hsa_circRNA_* IDs: try NCBI Nucleotide direct search
    4. For mmu_* / rno_* IDs: try Ensembl REST API (cross-species)
    5. For hybrids (circGENE/hsa_circ_*): extract gene from the GENE part
    """
    gene_map = {}

    # Step 1: Extract from gene-name-based IDs
    for cid in circrna_ids:
        gene = _extract_gene_name_from_id(cid)
        if gene:
            gene_map[cid] = gene

    n_from_name = len(gene_map)
    print(f"  [GO] Extracted {n_from_name} host genes from circRNA ID names")

    # Step 2: Load shared circBase info cache (with ID normalization)
    remaining = [cid for cid in circrna_ids if cid not in gene_map]
    # Broader matching: any hsa-prefixed ID, plus circRNA_* numeric
    hsa_style = []
    for cid in remaining:
        if re.search(r'^hsa[_\s-]+circ', cid, re.IGNORECASE):
            hsa_style.append(cid)
        elif re.match(r'^circRNA_\d+$', cid, re.IGNORECASE):
            hsa_style.append(cid)
        elif re.match(r'^hsa-circRNA\d+', cid, re.IGNORECASE):
            hsa_style.append(cid)

    if hsa_style:
        circbase_info = _load_or_compute_json(
            "circbase_info",
            lambda: _fetch_all_circbase_info(circrna_ids, timeout, max_workers),
        )

        if circbase_info and len(circbase_info) > 0:
            n_from_circbase = 0
            for cid in hsa_style:
                # Try exact match first, then normalized variants
                info = circbase_info.get(cid) or circbase_info.get(cid.strip())
                if not info:
                    # Try normalized variant matches
                    variants = _normalize_circrna_id(cid)
                    for v in variants:
                        info = circbase_info.get(v)
                        if info:
                            break
                if info and "gene_symbol" in info:
                    gene_map[cid] = info["gene_symbol"]
                    n_from_circbase += 1

            print(f"  [GO] Mapped {n_from_circbase} additional genes from circBase "
                  f"(with ID normalization)")

    # Step 3: Handle remaining "other" IDs — try broader gene name extraction
    remaining2 = [cid for cid in circrna_ids if cid not in gene_map]
    if remaining2:
        n_extra = 0
        for cid in remaining2:
            # circDLGAP4 → DLGAP4, CircDOCK1 → DOCK1
            m = re.match(r'^[Cc]irc([A-Z][A-Z0-9]+)', cid)
            if m:
                gene_map[cid] = m.group(1).upper()
                n_extra += 1
                continue
            # Circ_ADCY1 → ADCY1
            m = re.match(r'^[Cc]irc_([A-Z][A-Z0-9]+)', cid)
            if m:
                gene_map[cid] = m.group(1).upper()
                n_extra += 1
                continue
            # circRNA_Atp9b → ATP9B
            m = re.match(r'^circRNA_([A-Za-z][A-Za-z0-9]+)', cid)
            if m:
                gene_map[cid] = m.group(1).upper()
                n_extra += 1
                continue

        if n_extra > 0:
            print(f"  [GO] Extracted {n_extra} additional genes from 'other' ID patterns")

    # Step 4: Try Ensembl REST API for non-human circRNAs (mmu, rno)
    remaining3 = [cid for cid in circrna_ids if cid not in gene_map]
    nonhuman = [cid for cid in remaining3
                if re.match(r'^(mmu|rno)_circ', cid, re.IGNORECASE)]
    if nonhuman:
        n_ensembl = 0
        for cid in nonhuman[:20]:  # Limit Ensembl queries to avoid abuse
            gene = _fetch_ensembl_gene_for_circrna(cid, timeout=timeout)
            if gene:
                gene_map[cid] = gene
                n_ensembl += 1
        if n_ensembl > 0:
            print(f"  [GO] Mapped {n_ensembl} genes from Ensembl (non-human circRNAs)")

    total = len(gene_map)
    print(f"  [GO] Total host gene mapping: {total}/{len(circrna_ids)} circRNAs")
    return gene_map if total > 0 else None


def _fetch_ensembl_gene_for_circrna(circrna_id: str, timeout: float = 10.0) -> Optional[str]:
    """
    Try to find the host gene for a non-human circRNA using Ensembl REST API.

    For mmu_circ_* and rno_circ_* IDs, queries Ensembl to find
    the gene at the circRNA's genomic coordinates.
    """
    import requests

    # Extract species from prefix
    species_map = {
        'mmu': 'mouse',
        'rno': 'rat',
    }
    m = re.match(r'^(mmu|rno)_circ\w*_(\d+)', circrna_id, re.IGNORECASE)
    if not m:
        return None
    species_prefix = m.group(1).lower()
    species = species_map.get(species_prefix)
    if not species:
        return None

    # Try Ensembl lookup by ID
    # Some circRNA IDs might be in Ensembl as transcript IDs
    try:
        url = f"https://rest.ensembl.org/lookup/id/{circrna_id}"
        params = {"expand": "1", "content-type": "application/json"}
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if "Parent" in data:
                # Get gene from parent transcript
                parent = data["Parent"]
                r2 = requests.get(
                    f"https://rest.ensembl.org/lookup/id/{parent}",
                    params={"expand": "1", "content-type": "application/json"},
                    timeout=timeout,
                )
                if r2.status_code == 200:
                    data2 = r2.json()
                    gene_name = data2.get("gene_display_name") or data2.get("display_name")
                    if gene_name:
                        return gene_name.upper()
            gene_name = data.get("gene_display_name")
            if gene_name:
                return gene_name.upper()
    except Exception:
        pass

    return None


def _load_go_annotations(gene_goa_path: Optional[str] = None) -> Optional[Dict[str, Set[str]]]:
    """
    Load GO annotations (gene symbol → set of GO term IDs).

    Attempts to download goa_human.gaf.gz from geneontology.org if not available locally.
    Decompresses on download and saves as plain text .gaf for faster parsing.
    """
    import requests
    import gzip

    # Early return if parsed annotations are already cached (avoid re-downloading raw GAF)
    _ann_path = _get_json_cache_path("go_annotations")
    if os.path.exists(_ann_path):
        try:
            with open(_ann_path, "r") as f:
                _cached = json.load(f)
            return {k: set(v) for k, v in _cached.items()}
        except Exception:
            pass

    local_path = gene_goa_path or os.path.join(_CACHE_DIR, "goa_human.gaf")

    if not os.path.exists(local_path):
        print("  [GO] Downloading GO annotations from geneontology.org...")
        url = "http://geneontology.org/gene-associations/goa_human.gaf.gz"
        try:
            r = requests.get(url, timeout=300)  # ~260MB download, longer timeout
            if r.status_code == 200:
                # Decompress on the fly and save as plain text
                decompressed = gzip.decompress(r.content)
                with open(local_path, "wb") as f:
                    f.write(decompressed)
                size_mb = len(decompressed) / (1024 * 1024)
                print(f"  [GO] Downloaded and decompressed goa_human.gaf ({size_mb:.0f} MB)")
            else:
                print(f"  [GO] Failed to download GO annotations (HTTP {r.status_code})")
                return None
        except Exception as e:
            print(f"  [GO] Error downloading GO annotations: {e}")
            return None

    # Parse GAF file (plain text now)
    def parse_fn():
        print("  [GO] Parsing GO annotations (this may take a minute)...")
        gene_go: Dict[str, Set[str]] = defaultdict(set)
        try:
            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("!"):
                        continue
                    cols = line.strip().split("\t")
                    if len(cols) < 5:
                        continue
                    # GAF 2.0: col 2 = gene symbol, col 4 = GO term ID
                    gene_symbol = cols[2].strip().upper()
                    go_term = cols[4].strip()
                    if go_term.startswith("GO:"):
                        gene_go[gene_symbol].add(go_term)
        except Exception as e:
            print(f"  [GO] Error parsing GAF: {e}")
            return None

        print(f"  [GO] Parsed {len(gene_go)} genes with GO annotations")
        return {k: sorted(v) for k, v in gene_go.items()}

    result = _load_or_compute_json("go_annotations", parse_fn)
    if result:
        return {k: set(v) for k, v in result.items()}
    return None


def _load_go_dag(go_obo_path: Optional[str] = None) -> Optional[Dict[str, Set[str]]]:
    """
    Load GO DAG from OBO file (term → set of ancestor terms).

    Attempts to download go-basic.obo from geneontology.org if not available locally.
    """
    import requests

    # Early return if parsed DAG is already cached (avoid re-downloading raw OBO)
    _dag_path = _get_json_cache_path("go_dag")
    if os.path.exists(_dag_path):
        try:
            with open(_dag_path, "r") as f:
                _cached = json.load(f)
            return {k: set(v) for k, v in _cached.items()}
        except Exception:
            pass

    local_path = go_obo_path or os.path.join(_CACHE_DIR, "go-basic.obo")

    if not os.path.exists(local_path):
        print("  [GO] Downloading GO ontology from geneontology.org...")
        url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(r.text)
                print("  [GO] Downloaded go-basic.obo")
            else:
                print("  [GO] Failed to download GO ontology")
                return None
        except Exception as e:
            print(f"  [GO] Error downloading GO ontology: {e}")
            return None

    def parse_fn():
        print("  [GO] Parsing GO OBO file...")
        # Parse is_a and relationship: part_of relationships
        parents: Dict[str, Set[str]] = defaultdict(set)
        current_term = None
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line == "[Term]":
                    current_term = None
                elif line.startswith("id: GO:"):
                    current_term = line[4:].strip()
                elif line.startswith("is_a: GO:"):
                    if current_term:
                        parent = line[6:].split("!")[0].strip()
                        parents[current_term].add(parent)
                elif line.startswith("relationship: part_of GO:"):
                    if current_term:
                        parent = line.split("part_of")[1].split("!")[0].strip()
                        parents[current_term].add(parent)

        # Compute transitive closure (ancestors)
        print("  [GO] Computing ancestor closure...")
        ancestors: Dict[str, Set[str]] = {}

        def get_ancestors(term: str, visited: Set[str]) -> Set[str]:
            if term in ancestors:
                return ancestors[term]
            result = set()
            for p in parents.get(term, set()):
                if p not in visited:
                    visited.add(p)
                    result.add(p)
                    result.update(get_ancestors(p, visited))
            ancestors[term] = result
            return result

        all_terms = set(parents.keys())
        for p_terms in parents.values():
            all_terms.update(p_terms)

        for term in all_terms:
            get_ancestors(term, set())

        print(f"  [GO] Parsed DAG with {len(ancestors)} terms")
        return {k: sorted(v) for k, v in ancestors.items()}

    result = _load_or_compute_json("go_dag", parse_fn)
    if result:
        return {k: set(v) for k, v in result.items()}
    return None


def _go_gene_similarity(
    gene_a: str,
    gene_b: str,
    go_annotations: Dict[str, Set[str]],
    go_dag: Dict[str, Set[str]],
) -> float:
    """
    Compute GO-based functional similarity between two genes.

    Uses best-match average (BMA) strategy:
    For each GO term of gene_a, find the best matching GO term of gene_b,
    then average over all terms.

    Individual GO term similarity uses Wang's method.

    Returns: float in [0, 1]
    """
    terms_a = go_annotations.get(gene_a, set())
    terms_b = go_annotations.get(gene_b, set())

    if not terms_a or not terms_b:
        return 0.0

    # For large term sets, use a simpler Jaccard-like measure
    # with ancestor-based semantic expansion
    # Expand: each term → itself + all ancestors
    expanded_a = set(terms_a)
    for t in terms_a:
        expanded_a.update(go_dag.get(t, set()))
    expanded_b = set(terms_b)
    for t in terms_b:
        expanded_b.update(go_dag.get(t, set()))

    # Lin's similarity: 2 * |intersection| / (|A| + |B|), weighted by depth
    if not expanded_a or not expanded_b:
        return 0.0

    intersection = expanded_a & expanded_b
    # Weight by information content: deeper terms (more specific) contribute more
    ic_weight = 0.0
    for t in intersection:
        # Simple depth proxy: count ancestors (more ancestors = deeper = more informative)
        depth = len(go_dag.get(t, set()))
        ic_weight += 1.0 + np.log1p(depth)

    ic_a = sum(1.0 + np.log1p(len(go_dag.get(t, set()))) for t in expanded_a)
    ic_b = sum(1.0 + np.log1p(len(go_dag.get(t, set()))) for t in expanded_b)

    if ic_a == 0 or ic_b == 0:
        return 0.0

    # Lin's similarity with IC weighting
    sim = 2.0 * ic_weight / (ic_a + ic_b)
    return min(1.0, max(0.0, sim))


def _compute_go_term_ic(
    go_annotations: Dict[str, Set[str]],
    go_dag: Dict[str, Set[str]],
) -> Dict[str, float]:
    """
    Compute Information Content for each GO term.

    IC(t) = -log(|genes annotated with t (or its descendants)| / |total genes|)

    Root terms (GO:0008150, GO:0003674, GO:0005575) will have IC ≈ 0
    because they annotate almost all genes. Specific leaf terms have high IC.

    This also propagates annotations UP the DAG: if a gene is annotated with
    term t, it is implicitly annotated with all ancestors of t.

    Returns:
        Dict[GO_term_id → IC_score]
    """
    # Count total genes with ANY GO annotation
    all_annotated_genes = set(go_annotations.keys())
    n_total = len(all_annotated_genes)
    if n_total == 0:
        return {}

    # For each GO term, count how many genes are annotated with it (or its descendants)
    # First, build term → direct gene count
    term_genes_direct: Dict[str, Set[str]] = defaultdict(set)
    for gene, terms in go_annotations.items():
        for t in terms:
            term_genes_direct[t].add(gene)

    # Propagate up the DAG: a gene annotated with a child is also annotated with all ancestors
    term_genes_all: Dict[str, Set[str]] = defaultdict(set)
    for term, genes in term_genes_direct.items():
        term_genes_all[term].update(genes)
        # Also add to all ancestors
        ancestors = go_dag.get(term, set())
        for anc in ancestors:
            term_genes_all[anc].update(genes)

    # Compute IC
    term_ic: Dict[str, float] = {}
    for term, genes in term_genes_all.items():
        freq = len(genes) / n_total
        if freq > 0 and freq < 1.0:
            term_ic[term] = -np.log(freq)
        elif freq >= 1.0:
            term_ic[term] = 0.0  # Ubiquitous term, no information
        else:
            term_ic[term] = 10.0  # Very rare term, high information (cap)

    return term_ic


def compute_go_similarity(
    circrna_ids: List[str],
    go_obo_path: Optional[str] = None,
    gene_goa_path: Optional[str] = None,
    timeout: float = 5.0,
    max_workers: int = 5,
    ic_percentile: float = 0.0,  # 0.0 = use all, 50.0 = keep terms above median IC
    cache_prefix: str = "",
) -> Optional[np.ndarray]:
    """
    Compute GO host-gene functional similarity for circRNAs.

    1. Map circRNA → host gene (from ID name or circBase)
    2. Load GO annotations and DAG
    3. Compute pairwise gene functional similarity

    Parameters
    ----------
    ic_percentile : float
        If > 0, filter GO terms to keep only those with IC above this percentile.
        E.g., 50.0 keeps terms with IC > median (removes overly broad terms).
        This prunes root terms (GO:0008150, GO:0003674, GO:0005575) and other
        low-information terms. Default 0.0 = no pruning.
    """
    N = len(circrna_ids)

    # Determine cache key
    prune_tag = f"_ic{int(ic_percentile)}" if ic_percentile > 0 else ""
    cache_key = f"{cache_prefix}circrna_go_sim{prune_tag}"

    # Step 1: Host gene mapping (dataset-specific cache to avoid cross-contamination)
    _hg_key = f"{cache_prefix}host_gene_mapping"
    gene_map = _load_or_compute_json(_hg_key, lambda: _build_host_gene_mapping(
        circrna_ids, timeout, max_workers
    ))

    if gene_map is None or len(gene_map) == 0:
        print("  [GO] No host gene mapping — falling back")
        return None

    # Step 2: Load GO data
    go_annotations = _load_go_annotations(gene_goa_path)
    if go_annotations is None:
        print("  [GO] No GO annotations available — falling back")
        return None

    go_dag = _load_go_dag(go_obo_path)
    if go_dag is None:
        print("  [GO] No GO DAG available — falling back")
        return None

    # Step 3: Compute pairwise similarity
    def compute_fn():
        # Get genes for each circRNA
        circ_genes = [gene_map.get(cid) for cid in circrna_ids]

        # Count genes with GO annotations
        n_with_go = sum(1 for g in circ_genes if g and g in go_annotations and go_annotations[g])
        print(f"  [GO] {n_with_go}/{N} circRNAs have host genes with GO annotations")

        if n_with_go < 10:
            print("  [GO] Too few circRNAs with GO data — falling back")
            return None

        # ── Compute term IC for pruning ──
        term_ic = None
        ic_threshold = 0.0
        if ic_percentile > 0:
            print("  [GO] Computing GO term Information Content for semantic pruning...")
            term_ic = _compute_go_term_ic(go_annotations, go_dag)
            ic_values = np.array(list(term_ic.values()))
            ic_threshold = np.percentile(ic_values, ic_percentile)
            n_total_terms = len(term_ic)
            n_kept = int((ic_values > ic_threshold).sum())
            print(f"  [GO] IC threshold at {ic_percentile}th percentile = {ic_threshold:.4f}")
            print(f"  [GO] Kept {n_kept}/{n_total_terms} terms (removed {n_total_terms - n_kept} low-IC terms)")

            # Show some pruned root-level terms for diagnostics
            root_like = ["GO:0008150", "GO:0003674", "GO:0005575",
                         "GO:0009987", "GO:0003824", "GO:0005215"]
            for rt in root_like:
                if rt in term_ic:
                    status = "KEPT" if term_ic[rt] > ic_threshold else "PRUNED"
                    print(f"    [{status}] {rt}: IC={term_ic[rt]:.4f}")

        # Pre-compute expanded GO term sets and IC scores per gene
        print("  [GO] Pre-computing gene IC profiles...")
        gene_ic_profiles: Dict[str, Tuple[Set[str], float, float]] = {}
        n_terms_before_prune = 0
        n_terms_after_prune = 0
        seen_genes = set()
        for g in circ_genes:
            if not g or g in seen_genes:
                continue
            seen_genes.add(g)
            if g not in go_annotations:
                continue
            terms = go_annotations[g]
            if not terms:
                continue
            # Expand terms to include ancestors from DAG
            expanded = set(terms)
            for t in terms:
                expanded.update(go_dag.get(t, set()))

            n_terms_before_prune += len(expanded)

            # ── Semantic pruning: filter out low-IC terms ──
            if term_ic is not None and ic_threshold > 0:
                pruned_expanded = {t for t in expanded
                                   if term_ic.get(t, 0.0) > ic_threshold}
                n_terms_after_prune += len(pruned_expanded)
                expanded = pruned_expanded

            # IC-weighted size (depth-based proxy for remaining terms)
            ic_sum = sum(1.0 + np.log1p(len(go_dag.get(t, set()))) for t in expanded)
            gene_ic_profiles[g] = (expanded, float(len(expanded)), ic_sum)

        if term_ic is not None and ic_threshold > 0:
            reduction = (1 - n_terms_after_prune / max(1, n_terms_before_prune)) * 100
            print(f"  [GO] Pruning reduced expanded terms: "
                  f"{n_terms_before_prune} → {n_terms_after_prune} ({reduction:.1f}% reduction)")

        n_profiled = len(gene_ic_profiles)
        print(f"  [GO] Profiled {n_profiled} unique genes")

        # Build index list for genes with IC profiles
        gene_idx = []
        for g in circ_genes:
            if g and g in gene_ic_profiles:
                gene_idx.append(g)
            else:
                gene_idx.append(None)

        sim = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            sim[i, i] = 1.0
            gene_i = gene_idx[i]
            if gene_i is None:
                continue
            expanded_i, _, ic_i = gene_ic_profiles[gene_i]
            for j in range(i + 1, N):
                gene_j = gene_idx[j]
                if gene_j is None:
                    continue
                expanded_j, _, ic_j = gene_ic_profiles[gene_j]
                if ic_i == 0 or ic_j == 0:
                    continue
                intersection = expanded_i & expanded_j
                if not intersection:
                    continue
                ic_weight = sum(1.0 + np.log1p(len(go_dag.get(t, set()))) for t in intersection)
                s = min(1.0, 2.0 * ic_weight / (ic_i + ic_j))
                sim[i, j] = s
                sim[j, i] = s

        # Fill missing rows with average
        has_go = np.array([g is not None for g in gene_idx], dtype=bool)
        if has_go.any() and not has_go.all():
            avg_sim = sim[has_go].mean(axis=0)
            for i in range(N):
                if not has_go[i]:
                    sim[i] = avg_sim
                    sim[:, i] = avg_sim
                    sim[i, i] = 1.0

        n_pairs = int((sim > 0).sum()) - N
        mean_non_diag = sim[~np.eye(N, dtype=bool)].mean() if N > 1 else 0.0
        print(f"  [GO] Computed GO similarity: {n_pairs} non-zero pairs, "
              f"mean non-diag={mean_non_diag:.4f}")
        return sim

    return _load_or_compute_npy(cache_key, compute_fn)


# =============================================================================
# 3.5. External Knowledge Node Features (N×D vectors, NOT similarity matrices)
# =============================================================================
# These functions extract N×D feature vectors from the raw external knowledge
# data. They can be directly concatenated with GIP-derived node features,
# providing non-zero signal even for cold-start nodes (zero GIP).
#
# This is the KEY architectural fix: external knowledge as NODE FEATURES,
# not just graph edges.


def compute_kmer_features(
    circrna_ids: List[str],
    k: int = 4,
    variance_keep_ratio: float = 0.5,
    cache_prefix: str = "",
) -> Optional[np.ndarray]:
    """
    Compute k-mer frequency VECTORS (not similarity!) for each circRNA.

    Returns:
        (N, K) float32 matrix where K is the number of high-variance k-mers
        retained, or None on failure.

    Each row i is the normalized k-mer frequency profile of circRNA i,
    with low-variance k-mers filtered out.

    For circRNAs without sequences, fills with the mean profile.
    """
    N = len(circrna_ids)
    prune_tag = f"_var{int(variance_keep_ratio * 100)}"
    cache_key = f"{cache_prefix}circrna_kmer_features_{k}mer{prune_tag}"

    # Load sequences from existing cache
    sequences = _load_or_compute_json("circrna_sequences", None)
    if sequences is None or len(sequences) == 0:
        print("  [K-mer-Feat] No sequences available — skipping k-mer features")
        return None

    # ── Build sequence lookup with name normalization ──
    # Many circRNA names differ from the canonical circBase format used as
    # keys in circrna_sequences.json. We pre-build a lookup table that maps
    # normalized variants back to canonical keys.
    _seq_norm_lookup = {}  # normalized_name → canonical_key
    for seq_key in sequences.keys():
        _seq_norm_lookup[seq_key.lower().strip()] = seq_key
        # Also index all normalized variants
        for v in _normalize_circrna_id(seq_key):
            v_norm = v.lower().strip()
            if v_norm not in _seq_norm_lookup:
                _seq_norm_lookup[v_norm] = seq_key

    def _lookup_sequence(cid: str) -> Optional[str]:
        """Look up circRNA sequence with name normalization."""
        # Direct match
        if cid in sequences and len(sequences.get(cid, "")) >= k:
            return sequences[cid]
        # Try normalized variants
        for v in _normalize_circrna_id(cid):
            if v in sequences and len(sequences.get(v, "")) >= k:
                return sequences[v]
            # Try via lookup table
            canonical = _seq_norm_lookup.get(v.lower().strip())
            if canonical and len(sequences.get(canonical, "")) >= k:
                return sequences[canonical]
        return None

    n_with_seq = sum(1 for cid in circrna_ids if _lookup_sequence(cid) is not None)
    print(f"  [K-mer-Feat] {n_with_seq}/{N} circRNAs have sequences (k={k})")

    if n_with_seq < N * 0.3:
        print("  [K-mer-Feat] Too few sequences (<30%) — skipping k-mer features")
        return None

    def compute_fn():
        # Build raw frequency matrix
        n_total = 4 ** k
        freq_matrix = np.zeros((N, n_total), dtype=np.float32)
        has_seq = np.zeros(N, dtype=bool)
        for i, cid in enumerate(circrna_ids):
            seq = _lookup_sequence(cid)
            if seq is not None:
                freq_matrix[i] = _count_kmers(seq, k)
                has_seq[i] = True

        # Zero-fill missing nodes (letting model learn "no sequence data")
        # Mean-fill was adding noise — identical mean vectors for all missing
        # nodes are uninformative and dilute real signal.
        if not has_seq.all():
            n_missing = (~has_seq).sum()
            print(f"  [K-mer-Feat] Zero-filled {n_missing}/{N} nodes without sequences")

        # ── Variance Thresholding ──
        n_keep = max(16, int(n_total * variance_keep_ratio))
        variances = freq_matrix.var(axis=0)
        top_indices = np.argsort(variances)[-n_keep:]  # keep top-K
        freq_filtered = freq_matrix[:, top_indices]

        kept_var = variances[top_indices]
        print(f"  [K-mer-Feat] Variance threshold: {n_keep}/{n_total} k-mers retained "
              f"(var range: {kept_var.min():.6f}–{kept_var.max():.6f})")

        # Normalize rows to sum to 1 (L1 norm for probability distribution)
        row_sums = freq_filtered.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        freq_filtered = freq_filtered / row_sums

        print(f"  [K-mer-Feat] Feature matrix: {freq_filtered.shape}")
        return freq_filtered.astype(np.float32)

    return _load_or_compute_npy(cache_key, compute_fn)


def compute_go_features(
    circrna_ids: List[str],
    go_obo_path: Optional[str] = None,
    gene_goa_path: Optional[str] = None,
    n_top_terms: int = 128,
    timeout: float = 5.0,
    max_workers: int = 5,
    cache_prefix: str = "",
) -> Optional[np.ndarray]:
    """
    Compute GO IC-based feature vectors for each circRNA.

    Steps:
    1. Map circRNA → host gene (from ID name or circBase cache)
    2. Compute Information Content for every GO term
    3. Select top-N highest-IC GO terms
    4. For each circRNA: feature[term_i] = IC(term_i) if host gene has it

    Returns:
        (N, n_top_terms) float32 matrix, or None on failure.
    """
    N = len(circrna_ids)
    cache_key = f"{cache_prefix}circrna_go_features_ic{n_top_terms}"

    # Load all GO data (dataset-specific cache to avoid cross-contamination)
    _hg_key2 = f"{cache_prefix}host_gene_mapping"
    gene_map = _load_or_compute_json(_hg_key2, lambda: _build_host_gene_mapping(
        circrna_ids, timeout, max_workers
    ))
    if gene_map is None or len(gene_map) == 0:
        print("  [GO-Feat] No host gene mapping — skipping GO features")
        return None

    go_annotations = _load_go_annotations(gene_goa_path)
    if go_annotations is None:
        print("  [GO-Feat] No GO annotations — skipping GO features")
        return None

    go_dag = _load_go_dag(go_obo_path)
    if go_dag is None:
        print("  [GO-Feat] No GO DAG — skipping GO features")
        return None

    def compute_fn():
        circ_genes = [gene_map.get(cid) for cid in circrna_ids]
        n_mapped = sum(1 for g in circ_genes if g is not None)
        print(f"  [GO-Feat] {n_mapped}/{N} circRNAs have host gene mappings")

        # Compute GO term IC (once, cached in memory but not persisted separately)
        term_ic = _compute_go_term_ic(go_annotations, go_dag)
        if not term_ic:
            print("  [GO-Feat] Could not compute GO term IC")
            return None

        # ── Select top-N highest-IC GO terms ──
        # Only consider terms that appear in expanded annotations of our circRNA host genes
        ic_sorted = sorted(term_ic.items(), key=lambda x: x[1], reverse=True)

        # Filter to terms actually used by our genes
        our_genes = set(g for g in circ_genes if g is not None and g in go_annotations)
        gene_before = len(our_genes)
        relevant_terms: Set[str] = set()
        for g in our_genes:
            for t in go_annotations.get(g, set()):
                relevant_terms.add(t)
                relevant_terms.update(go_dag.get(t, set()))
        print(f"  [GO-Feat] {len(relevant_terms)} unique GO terms from {gene_before} host genes")

        # Top-N by IC, but only from relevant terms (if we have enough)
        top_terms_ic = [(t, ic) for t, ic in ic_sorted
                        if t in relevant_terms and ic > 0.0][:n_top_terms]

        # If not enough relevant terms, pad with any high-IC terms
        if len(top_terms_ic) < n_top_terms:
            extra = [(t, ic) for t, ic in ic_sorted
                     if t not in relevant_terms and ic > 1.0][:n_top_terms - len(top_terms_ic)]
            top_terms_ic.extend(extra)

        if len(top_terms_ic) < 10:
            print("  [GO-Feat] Too few GO terms available — skipping GO features")
            return None

        n_actual = len(top_terms_ic)
        ic_range = (top_terms_ic[-1][1], top_terms_ic[0][1])
        print(f"  [GO-Feat] Selected {n_actual} top-IC GO terms (IC range: {ic_range[0]:.3f}–{ic_range[1]:.3f})")

        # ── Build expanded term sets per host gene ──
        gene_expanded: Dict[str, Set[str]] = {}
        gene_ic_scores: Dict[str, Dict[str, float]] = {}
        for g in our_genes:
            terms = go_annotations.get(g, set())
            expanded = set(terms)
            for t in terms:
                expanded.update(go_dag.get(t, set()))
            gene_expanded[g] = expanded
            # Precompute IC for all terms in this gene's expanded set
            gene_ic_scores[g] = {t: term_ic.get(t, 0.0) for t in expanded if t in term_ic}

        # ── Build feature matrix ──
        selected_terms = [t for t, _ in top_terms_ic]
        feature_matrix = np.zeros((N, n_actual), dtype=np.float32)

        for i, g in enumerate(circ_genes):
            if g is None or g not in gene_expanded:
                continue  # zeros for unknown genes
            expanded = gene_expanded[g]
            ic_map = gene_ic_scores[g]
            for j, term in enumerate(selected_terms):
                if term in expanded:
                    feature_matrix[i, j] = ic_map.get(term, 0.0)

        # Zero-fill missing rows (instead of mean-fill noise)
        has_go = np.array([g is not None and g in gene_expanded for g in circ_genes])
        if not has_go.all():
            n_zero = (~has_go).sum()
            print(f"  [GO-Feat] Zero-filled {n_zero}/{N} nodes without GO data")

        # Normalize each row to unit variance (standard scaling)
        # Use standard normalization per feature for stable gradient
        # Center and scale each column
        col_std = feature_matrix.std(axis=0, keepdims=True)
        col_std = np.where(col_std == 0, 1.0, col_std)
        feature_matrix = (feature_matrix - feature_matrix.mean(axis=0, keepdims=True)) / col_std

        n_nonzero = int((feature_matrix > 0).sum())
        print(f"  [GO-Feat] Feature matrix: {feature_matrix.shape} "
              f"({n_nonzero} non-zero entries, {n_nonzero / (N * n_actual) * 100:.1f}% density)")

        return feature_matrix.astype(np.float32)

    return _load_or_compute_npy(cache_key, compute_fn)


def compute_mesh_features(
    disease_names: List[str],
    data_dir: str = "./data/CompareExperiment/Data",
    cache_key: str = "mesh_disease_sim",
    cache_prefix: str = "",
    n_top_nodes: int = 128,
) -> Optional[np.ndarray]:
    """
    Compute MeSH DAG semantic feature vectors for each disease.

    Uses the MeSH tree number hierarchy to build binary/weighted feature vectors.
    Each dimension corresponds to a tree node (prefix) in the MeSH DAG.

    For a disease with tree number "C04.557.465.625.600.380":
      ancestors = [C04, C04.557, C04.557.465, C04.557.465.625,
                   C04.557.465.625.600, C04.557.465.625.600.380]
      feature[ancestor_i] += 1.0 (weighted by depth/IC)

    Returns:
        (M, F) float32 matrix where M = len(disease_names) and
        F is the number of top informative tree nodes retained.
    """
    M = len(disease_names)
    prefix = cache_prefix or cache_key.replace("mesh_disease_sim", "").lstrip("_")
    feat_cache_key = f"mesh_features{'_' + prefix if prefix else ''}_n{n_top_nodes}"

    # Load the MeSH DAG and disease tree numbers
    dag_path = _get_json_cache_path(f"mesh_dag{'_' + prefix if prefix else ''}")
    trees_path = _get_json_cache_path(f"mesh_disease_trees{'_' + prefix if prefix else ''}")

    if not os.path.exists(dag_path) or not os.path.exists(trees_path):
        print("  [MeSH-Feat] MeSH data not cached — compute mesh similarity first")
        return None

    def compute_fn():
        with open(dag_path) as f:
            dag = json.load(f)
        with open(trees_path) as f:
            disease_trees = json.load(f)

        # Each disease gets one (primary) tree number
        # Build disease → expanded ancestor set
        disease_ancestors: Dict[str, Set[str]] = {}
        n_with_trees = 0
        for d in disease_names:
            tree_num = disease_trees.get(d)
            if tree_num:
                ancestors = set(_parse_tree_number(tree_num))
                disease_ancestors[d] = ancestors
                n_with_trees += 1
            else:
                disease_ancestors[d] = set()

        print(f"  [MeSH-Feat] {n_with_trees}/{M} diseases have MeSH tree numbers")

        # ── Select top informative tree nodes ──
        # Score each node by: depth * inverse_document_frequency
        # depth = number of dots + 1 (root = 1)
        # idf = -log(n_diseases_with_node / M)
        node_docs: Dict[str, int] = defaultdict(int)
        for ancestors in disease_ancestors.values():
            for node in ancestors:
                node_docs[node] += 1

        node_scores = {}
        for node, ndoc in node_docs.items():
            depth = node.count(".") + 1
            idf = np.log((M + 1) / (ndoc + 1)) + 1.0  # smooth IDF
            node_scores[node] = depth * idf

        # Sort by score, keep top nodes
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        top_nodes = sorted_nodes[:n_top_nodes]

        # If DAG has fewer nodes than requested, use all
        if len(top_nodes) < n_top_nodes:
            n_top = len(top_nodes)
            print(f"  [MeSH-Feat] Only {n_top} DAG nodes available (requested {n_top_nodes})")
        else:
            n_top = n_top_nodes

        selected_nodes = [node for node, _ in top_nodes[:n_top]]
        score_range = (top_nodes[n_top-1][1], top_nodes[0][1])
        print(f"  [MeSH-Feat] Selected {n_top} top tree nodes "
              f"(score range: {score_range[0]:.2f}–{score_range[1]:.2f})")

        # ── Build feature matrix ──
        feature_matrix = np.zeros((M, n_top), dtype=np.float32)

        for i, d in enumerate(disease_names):
            ancestors = disease_ancestors.get(d, set())
            if not ancestors:
                continue
            for j, node in enumerate(selected_nodes):
                if node in ancestors:
                    # Weight by node score (depth * IDF)
                    feature_matrix[i, j] = node_scores.get(node, 1.0)

        # Zero-fill missing rows (instead of mean-fill noise)
        has_mesh = np.array([d in disease_trees for d in disease_names])
        if not has_mesh.all():
            n_zero = (~has_mesh).sum()
            print(f"  [MeSH-Feat] Zero-filled {n_zero}/{M} diseases without MeSH data")

        # Standardize
        col_std = feature_matrix.std(axis=0, keepdims=True)
        col_std = np.where(col_std == 0, 1.0, col_std)
        feature_matrix = (feature_matrix - feature_matrix.mean(axis=0, keepdims=True)) / col_std

        n_nonzero = int((feature_matrix > 0).sum())
        print(f"  [MeSH-Feat] Feature matrix: {feature_matrix.shape} "
              f"({n_nonzero} non-zero entries, "
              f"{n_nonzero / (M * n_top) * 100:.1f}% density)")

        return feature_matrix.astype(np.float32)

    return _load_or_compute_npy(feat_cache_key, compute_fn)


# =============================================================================
# 4. High-Level Convenience API
# =============================================================================

def load_all_external_knowledge(
    circrna_ids: List[str],
    disease_names: List[str],
    data_dir: str = "./data/CompareExperiment/Data",
    go_obo_path: Optional[str] = None,
    gene_goa_path: Optional[str] = None,
    kmer_k: int = 4,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Load all three external knowledge similarity matrices.

    Returns dict with keys:
      - 'disease_mesh': MeSH disease semantic similarity (M×M) or None
      - 'circrna_kmer': Sequence k-mer cosine similarity (N×N) or None
      - 'circrna_go': GO host-gene functional similarity (N×N) or None
    """
    print("\n" + "=" * 60)
    print("Loading External Biological Knowledge...")
    print("=" * 60)

    # 1. MeSH
    print("\n[1/3] MeSH Disease Semantic Similarity")
    disease_mesh = load_mesh_similarity(disease_names, data_dir)

    # 2. Sequence k-mer
    print("\n[2/3] circRNA Sequence k-mer Similarity")
    circrna_kmer = compute_kmer_similarity(circrna_ids, k=kmer_k)

    # 3. GO
    print("\n[3/3] GO Host-Gene Functional Similarity")
    circrna_go = compute_go_similarity(circrna_ids, go_obo_path, gene_goa_path)

    print("\n" + "=" * 60)
    results = {
        "disease_mesh": disease_mesh is not None,
        "circrna_kmer": circrna_kmer is not None,
        "circrna_go": circrna_go is not None,
    }
    print(f"External knowledge loaded: {results}")
    print("=" * 60 + "\n")

    return {
        "disease_mesh": disease_mesh,
        "circrna_kmer": circrna_kmer,
        "circrna_go": circrna_go,
    }


# =============================================================================
# 4. miRNA Sequence k-mer Features (miRBase)
# =============================================================================

def _load_mirbase_sequences(mature_fa_path: str = "") -> Dict[str, str]:
    """Load mature miRNA sequences from miRBase mature.fa. Returns {mirbase_id: sequence}."""
    if not mature_fa_path:
        mature_fa_path = os.path.join(_CACHE_DIR, "mature.fa")
    if not os.path.exists(mature_fa_path):
        print("  [miRNA-kmer] miRBase mature.fa not found — download from mirbase.org")
        return {}

    seqs = {}
    with open(mature_fa_path, 'r') as f:
        current_id = None
        current_seq = []
        for line in f:
            if line.startswith('>'):
                if current_id:
                    seqs[current_id] = ''.join(current_seq)
                current_id = line[1:].strip().split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip().upper())
        if current_id:
            seqs[current_id] = ''.join(current_seq)
    return seqs


def _map_mirna_to_mirbase(
    mirna_names: List[str],
    mirbase_seqs: Dict[str, str],
) -> Dict[str, str]:
    """
    Map HMDD/KHGAT miRNA names (precursor: hsa-mir-182) to miRBase mature IDs (hsa-miR-182-5p).
    Returns {input_name: mirbase_id} for matched miRNAs.
    """
    name_map = {}
    for name in mirna_names:
        n = name.lower().strip()
        n = n.replace('mir-', 'miR-')  # Normalize case: mir → miR

        # Try exact, with -5p, with -3p, and paralog variants
        candidates = [n]
        if not n.endswith('-5p') and not n.endswith('-3p'):
            candidates.extend([n + '-5p', n + '-3p'])

        for c in candidates:
            if c in mirbase_seqs:
                name_map[name] = c
                break

        # If still not found, try with different prefix (hsa- vs no prefix)
        if name not in name_map:
            if not n.startswith('hsa-'):
                n2 = 'hsa-' + n
                for c in [n2, n2 + '-5p', n2 + '-3p']:
                    if c in mirbase_seqs:
                        name_map[name] = c
                        break

        # Filter: only keep human miRNAs
        if name in name_map and not name_map[name].startswith('hsa-'):
            del name_map[name]

    return name_map


def compute_mirna_kmer_features(
    mirna_names: List[str],
    k: int = 4,
    cache_key: str = "mirna_kmer_features",
    mature_fa_path: str = "",
) -> Optional[np.ndarray]:
    """
    Compute k-mer frequency vectors for miRNA sequences from miRBase.

    Args:
        mirna_names: miRNA names in HMDD/KHGAT format (e.g., hsa-mir-182)
        k: k-mer size (default 4, giving 4^4=256 features)
        cache_key: cache identifier
        mature_fa_path: path to miRBase mature.fa ("" = use default)

    Returns:
        (N, 4^k) float32 array of normalized k-mer frequencies, or None
    """
    cache_path = _get_cache_path(cache_key)
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass

    mirbase_seqs = _load_mirbase_sequences(mature_fa_path)
    if not mirbase_seqs:
        return None

    name_map = _map_mirna_to_mirbase(mirna_names, mirbase_seqs)
    n_matched = len(name_map)
    print(f"  [miRNA-kmer] Matched {n_matched}/{len(mirna_names)} miRNAs to miRBase sequences")

    if n_matched == 0:
        return None

    # k-mer vocabulary (4^k)
    bases = ['A', 'C', 'G', 'U']
    kmer_to_idx = {}
    from itertools import product
    for idx, kmer in enumerate(product(bases, repeat=k)):
        kmer_to_idx[''.join(kmer)] = idx
    n_kmers = len(kmer_to_idx)  # 256 for k=4

    # Compute k-mer frequencies for each matched miRNA
    N = len(mirna_names)
    features = np.zeros((N, n_kmers), dtype=np.float32)

    for i, name in enumerate(mirna_names):
        if name in name_map:
            seq = mirbase_seqs[name_map[name]]
            # Replace T with U (miRNA sequences use U, but some databases use T)
            seq = seq.replace('T', 'U')
            # Count k-mers
            for j in range(len(seq) - k + 1):
                kmer = seq[j:j + k]
                if kmer in kmer_to_idx:
                    features[i, kmer_to_idx[kmer]] += 1
            # Normalize by sequence length
            seq_len = len(seq) - k + 1
            if seq_len > 0:
                features[i] /= seq_len

    # Save cache
    try:
        np.save(cache_path, features)
    except Exception:
        pass

    n_with_seq = int((features.sum(1) > 0).sum())
    print(f"  [miRNA-kmer] k={k}: {n_with_seq}/{N} miRNAs have k-mer features "
          f"({features.shape[1]} dim)")

    return features


def compute_mirna_kmer_similarity(
    mirna_names: List[str],
    k: int = 4,
    cache_key: str = "mirna_kmer_sim",
    mature_fa_path: str = "",
) -> Optional[np.ndarray]:
    """
    Compute miRNA-miRNA k-mer cosine similarity matrix.

    Returns:
        (N, N) float32 similarity matrix, or None
    """
    cache_path = _get_cache_path(cache_key)
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass

    features = compute_mirna_kmer_features(
        mirna_names, k=k, cache_key=cache_key.replace("_sim", "_features"),
        mature_fa_path=mature_fa_path)
    if features is None:
        return None

    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(features)
    sim = np.nan_to_num(sim, nan=0.0).astype(np.float32)

    try:
        np.save(cache_path, sim)
    except Exception:
        pass

    n_nonzero = int((sim > 0).sum()) - sim.shape[0]
    print(f"  [miRNA-kmer] Similarity matrix: {sim.shape}, {n_nonzero} non-zero pairs")

    return sim
