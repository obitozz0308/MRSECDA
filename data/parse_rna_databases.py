"""
Parse downloaded RNA databases into MRST-CDA project format + ceRNA inference.

Reads raw files under  data/external/raw/  and produces, in  data/external/ :
  - circrna_mirna_real.txt        (miRNA \t 1 \t circRNA)   real CLIP/curated interactions
  - lncrna_mirna_real.txt         (miRNA \t 1 \t lncRNA)    real curated interactions
  - circrna_lncrna_ceRNA.txt      (circRNA \t 1 \t lncRNA)  inferred via shared miRNA (sponge)
  - lncrnadisease_lncrna_disease.txt (lncRNA \t 1 \t disease)  non-cancer associations

Run:  python data/parse_rna_databases.py [--min-shared 2]

Outputs are keyed by canonical (lowercased) entity names; integration in
hetero_data.py does the fuzzy matching against each dataset's own names.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import tarfile
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "external", "raw")
OUT_DIR = os.path.join(ROOT, "data", "external")

# RNAInter RR tarball + inner txt
RNAINTER_TAR = os.path.join(RAW_DIR, "rnainter_RR.tar.gz")
ENCORI_DIR = os.path.join(RAW_DIR, "encori")
LNCRNADISEASE_DIR = os.path.join(RAW_DIR, "lncrnadisease")


def log(msg: str) -> None:
    print(f"[parse] {msg}", flush=True)


def norm(name: str) -> str:
    """Canonical entity name (lowercase, strip)."""
    return name.strip().lower()


def load_tsv_cols(path: str, skip_comment: bool = True):
    """Yield split lines, skipping comments and blank lines."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if skip_comment and line.startswith("#"):
                continue
            if not line.strip():
                continue
            yield line.split("\t")


# ---------------------------------------------------------------------------
# ENCORI parsing (geneType=circRNA -> circID col4; geneType=lncRNA -> geneName col3)
# ---------------------------------------------------------------------------

def parse_encori(encori_dir: str) -> tuple[set, set]:
    """Return (circ_mirna_pairs, lnc_mirna_pairs) as sets of (miRNA, target)."""
    circ_mirna: set = set()
    lnc_mirna: set = set()
    n_files = 0
    for gt in ("circRNA", "lncRNA"):
        gdir = os.path.join(encori_dir, gt)
        if not os.path.isdir(gdir):
            continue
        for fn in os.listdir(gdir):
            if not fn.endswith(".txt"):
                continue
            n_files += 1
            path = os.path.join(gdir, fn)
            header_seen = False
            for parts in load_tsv_cols(path):
                if not header_seen and parts and parts[0] in ("miRNAid", "miRNAname"):
                    header_seen = True
                    continue
                if len(parts) < 6:
                    continue
                mirna = norm(parts[1])            # miRNAname
                if gt == "circRNA":
                    circids = parts[4].strip()    # circID (comma-separated)
                    for c in circids.split(","):
                        c = norm(c)
                        if c and c != "na":
                            circ_mirna.add((mirna, c))
                else:  # lncRNA
                    lnc = norm(parts[3])          # geneName
                    if lnc and lnc != "na":
                        lnc_mirna.add((mirna, lnc))
    log(f"ENCORI: {n_files} files -> circRNA-miRNA {len(circ_mirna)}, "
        f"lncRNA-miRNA {len(lnc_mirna)}")
    return circ_mirna, lnc_mirna


# ---------------------------------------------------------------------------
# RNAInter RR parsing
# ---------------------------------------------------------------------------

def extract_rnainter_txt() -> str | None:
    """Extract the single .txt from the RR tarball into RAW_DIR; return its path."""
    if not os.path.exists(RNAINTER_TAR):
        return None
    out_txt = os.path.join(RAW_DIR, "rnainter_RR.txt")
    if os.path.exists(out_txt) and os.path.getsize(out_txt) > 100_000_000:
        return out_txt
    log(f"Extracting {RNAINTER_TAR} (large, ~2.7 GB uncompressed; may take a while)...")
    with tarfile.open(RNAINTER_TAR, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".txt")]
        # prefer the largest txt
        member = max(members, key=lambda m: m.size) if members else None
        if member is None:
            log("  no .txt inside tarball")
            return None
        tar.extract(member, RAW_DIR)
        return os.path.join(RAW_DIR, member.name)
    return None


def parse_rnainter(txt_path: str) -> tuple[set, set]:
    """Return (circ_mirna, lnc_mirna) from RNAInter RR txt."""
    circ_mirna: set = set()
    lnc_mirna: set = set()
    if not txt_path or not os.path.exists(txt_path):
        log("RNAInter txt not found; skipping")
        return circ_mirna, lnc_mirna

    log(f"Parsing RNAInter RR: {txt_path} (this is the ~9.5M-row file)...")
    with open(txt_path, encoding="utf-8", errors="ignore") as f:
        header = f.readline().rstrip("\n").split("\t")
        # locate columns (robust to column order)
        idx = {}
        for i, h in enumerate(header):
            h_l = h.lower().replace(" ", "")
            if "interactor1.symbol" in h_l or h_l == "interactor1":
                idx["s1"] = i
            elif "interactor2.symbol" in h_l or h_l == "interactor2":
                idx["s2"] = i
            elif "category1" in h_l:
                idx["c1"] = i
            elif "category2" in h_l:
                idx["c2"] = i
            elif "species1" in h_l:
                idx["sp1"] = i
            elif "species2" in h_l:
                idx["sp2"] = i
        log(f"  detected columns: {idx}")

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < max(idx.values(), default=0) + 1:
                continue
            def get(k):
                return parts[idx[k]] if k in idx and idx[k] < len(parts) else ""
            c1 = norm(get("c1")); c2 = norm(get("c2"))
            s1 = norm(get("s1")); s2 = norm(get("s2"))
            # human only (species contains 'homo' or 'human'), if species col present
            if "sp1" in idx and "sp2" in idx:
                sp1 = norm(get("sp1")); sp2 = norm(get("sp2"))
                if ("homo" not in sp1 and "human" not in sp1) or \
                   ("homo" not in sp2 and "human" not in sp2):
                    continue
            # circRNA-miRNA
            if c1 == "circrna" and c2 == "mirna":
                circ_mirna.add((s2, s1))  # (miRNA, circRNA)
            elif c1 == "mirna" and c2 == "circrna":
                circ_mirna.add((s1, s2))
            # lncRNA-miRNA
            elif c1 == "lncrna" and c2 == "mirna":
                lnc_mirna.add((s2, s1))  # (miRNA, lncRNA)
            elif c1 == "mirna" and c2 == "lncrna":
                lnc_mirna.add((s1, s2))
    log(f"RNAInter: circRNA-miRNA {len(circ_mirna)}, lncRNA-miRNA {len(lnc_mirna)}")
    return circ_mirna, lnc_mirna


# ---------------------------------------------------------------------------
# LncRNADisease parsing
# ---------------------------------------------------------------------------

def parse_lncrnadisease(lnc_dir: str) -> set:
    """Return lncRNA-disease pairs (lncRNA, disease) from LncRNADisease v3.0 files."""
    pairs: set = set()
    if not os.path.isdir(lnc_dir):
        log("LncRNADisease dir not found; skipping")
        return pairs
    for fn in os.listdir(lnc_dir):
        path = os.path.join(lnc_dir, fn)
        if fn.endswith((".txt", ".tsv")):
            pairs |= _parse_lncrnadisease_txt(path)
        elif fn.endswith((".xls", ".xlsx")):
            pairs |= _parse_lncrnadisease_xls(path)
    log(f"LncRNADisease: {len(pairs)} lncRNA-disease pairs")
    return pairs


def _parse_lncrnadisease_txt(path: str) -> set:
    pairs: set = set()
    # Expected columns (help page): Symbol, Category, Species, Disease Name, ...
    for parts in load_tsv_cols(path):
        if len(parts) < 4:
            continue
        # Heuristic: find a 'Category' column value in {lncRNA} and a disease col
        symbol = norm(parts[0])
        cat = norm(parts[1]) if len(parts) > 1 else ""
        species = norm(parts[2]) if len(parts) > 2 else ""
        disease = parts[3].strip()
        # tolerate minor header-shift: locate category column by value
        for i in range(1, min(len(parts), 3)):
            if norm(parts[i]) in ("lncrna", "circrna"):
                cat = norm(parts[i])
                break
        if cat != "lncrna":
            continue
        if species and "homo" not in species and "human" not in species and species not in ("", "-"):
            # only keep human or unspecified
            if not any(k in species for k in ("sapiens", "human")):
                continue
        if symbol and disease and disease.lower() not in ("disease name", "disease", "-", "na"):
            pairs.add((symbol, norm(disease)))
    return pairs


def _parse_lncrnadisease_xls(path: str) -> set:
    pairs: set = set()
    try:
        import pandas as pd
        df = pd.read_excel(path, header=None)
    except Exception as e:  # noqa: BLE001
        log(f"  could not read {path}: {e}")
        return pairs
    # Assume first 4 cols: Symbol, Category, Species, Disease Name
    for _, row in df.iterrows():
        if len(row) < 4:
            continue
        symbol = norm(str(row.iloc[0]))
        cat = norm(str(row.iloc[1]))
        species = norm(str(row.iloc[2]))
        disease = str(row.iloc[3]).strip()
        if cat != "lncrna" or symbol == "nan" or disease.lower() in ("nan", "disease name", "disease"):
            continue
        if species not in ("", "-", "nan") and not any(k in species for k in ("homo", "human", "sapiens")):
            continue
        pairs.add((symbol, norm(disease)))
    return pairs


# ---------------------------------------------------------------------------
# ceRNA inference: circRNA-lncRNA via shared miRNA
# ---------------------------------------------------------------------------

def infer_circrna_lncrna(circ_mirna: set, lnc_mirna: set, min_shared: int) -> set:
    """circRNA C and lncRNA L compete if they share >= min_shared miRNAs (sponge)."""
    # build miRNA -> circRNAs, miRNA -> lncRNAs
    mirna_to_circs: dict = defaultdict(set)
    mirna_to_lncs: dict = defaultdict(set)
    for mirna, circ in circ_mirna:
        mirna_to_circs[mirna].add(circ)
    for mirna, lnc in lnc_mirna:
        mirna_to_lncs[mirna].add(lnc)

    # count shared miRNAs per (circ, lnc)
    shared: dict = defaultdict(int)
    for mirna in mirna_to_circs.keys() & mirna_to_lncs.keys():
        for circ in mirna_to_circs[mirna]:
            for lnc in mirna_to_lncs[mirna]:
                shared[(circ, lnc)] += 1

    pairs = {(c, l) for (c, l), n in shared.items() if n >= min_shared}
    log(f"ceRNA inference (min shared miRNA={min_shared}): {len(pairs)} circRNA-lncRNA pairs")
    return pairs


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_pairs(path: str, pairs: set, order: tuple[int, int] = (0, 1)) -> None:
    """Write pairs as  src \t 1 \t tgt  using column order indices."""
    with open(path, "w", encoding="utf-8") as f:
        for p in sorted(pairs):
            a, b = p[order[0]], p[order[1]]
            f.write(f"{a}\t1\t{b}\n")
    log(f"  wrote {len(pairs)} pairs -> {path}")


def write_report(name: str, data: dict) -> None:
    path = os.path.join(OUT_DIR, f"{name}_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log(f"  wrote report -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-shared", type=int, default=2,
                    help="min shared miRNAs for a circRNA-lncRNA ceRNA edge")
    ap.add_argument("--skip-rnainter", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. ENCORI
    circ_mirna: set = set()
    lnc_mirna: set = set()
    if os.path.isdir(ENCORI_DIR):
        cm, lm = parse_encori(ENCORI_DIR)
        circ_mirna |= cm
        lnc_mirna |= lm

    # 2. RNAInter
    if not args.skip_rnainter:
        txt = extract_rnainter_txt()
        cm, lm = parse_rnainter(txt)
        circ_mirna |= cm
        lnc_mirna |= lm

    # 3. LncRNADisease
    lnc_dis = parse_lncrnadisease(LNCRNADISEASE_DIR)

    # 4. ceRNA inference
    circ_lnc = infer_circrna_lncrna(circ_mirna, lnc_mirna, args.min_shared)

    # 5. Save
    save_pairs(os.path.join(OUT_DIR, "circrna_mirna_real.txt"), circ_mirna, order=(0, 1))
    save_pairs(os.path.join(OUT_DIR, "lncrna_mirna_real.txt"), lnc_mirna, order=(0, 1))
    save_pairs(os.path.join(OUT_DIR, "circrna_lncrna_ceRNA.txt"), circ_lnc, order=(0, 1))
    save_pairs(os.path.join(OUT_DIR, "lncrnadisease_lncrna_disease.txt"), lnc_dis, order=(0, 1))

    write_report("circrna_mirna_real",
                 {"pairs": len(circ_mirna), "unique_circrna": len({c for _, c in circ_mirna}),
                  "unique_mirna": len({m for m, _ in circ_mirna})})
    write_report("lncrna_mirna_real",
                 {"pairs": len(lnc_mirna), "unique_lncrna": len({l for _, l in lnc_mirna}),
                  "unique_mirna": len({m for m, _ in lnc_mirna})})
    write_report("circrna_lncrna_ceRNA",
                 {"pairs": len(circ_lnc), "unique_circrna": len({c for c, _ in circ_lnc}),
                  "unique_lncrna": len({l for _, l in circ_lnc}), "min_shared": args.min_shared})
    write_report("lncrnadisease_lncrna_disease",
                 {"pairs": len(lnc_dis), "unique_lncrna": len({l for l, _ in lnc_dis}),
                  "unique_disease": len({d for _, d in lnc_dis})})

    log("All outputs written to data/external/")
    log("Next: re-run training — hetero_data.py will pick up the new files.")


if __name__ == "__main__":
    main()
