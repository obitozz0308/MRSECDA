# Data sources

This file lists the third-party biological databases used by MRSECDA and where to obtain the raw files that are **not** bundled in this repository. The bundled, processed files under `data/` were derived from these sources; only the large raw dumps are excluded and are re-downloaded automatically by the pipeline when needed.

## Bundled inputs

| Input | Location | Source |
|---|---|---|
| Five benchmark datasets (circR2Disease, circR2Cancer, circRNADisease, circ2Traits, circad) | `data/CompareExperiment/Data/` | [Lan et al. 2023, doi:10.1093/bib/bbac613](https://doi.org/10.1093/bib/bbac613) ([lanbiolab/CompareExperiment](https://github.com/lanbiolab/CompareExperiment)) |
| circRNA–miRNA interaction matrix | `data/CompareExperiment/Data/circRNA_miRNA/` | ENCORI / starBase v2.0 (via CompareExperiment) |
| miRNA / lncRNA lists and interactions | `data/CompareExperiment/AE-RF main/KHGAT/data/` | [KHGAT](https://github.com/.../KHGAT) |
| miRNA–disease associations (processed) | `data/external/hmdd_v4_*_mirna_disease.txt` | [HMDD v4.0](https://www.cuilab.cn/hmdd) |
| circRNA–miRNA curated interactions | `data/external/circrna_mirna_real.txt`, `data/external/ENCORI_circRNA_miRNA_hg38.txt` | ENCORI / RNAInter |
| miRBase mature miRNA sequences | `data/external/mature.fa` | [miRBase](https://www.mirbase.org/) |
| MeSH disease semantics | `data/external/mesh_*` | [MeSH](https://www.nlm.nih.gov/mesh/) |
| circBase metadata | `data/external/circbase_info.json` | [circBase](http://www.circbase.org/) |

## Re-downloaded on demand (not bundled)

These files are large and are fetched automatically by `data/external_knowledge.py` when the corresponding feature is computed; they are only needed if you regenerate features from scratch.

| Database | File(s) | Download URL |
|---|---|---|
| Gene Ontology annotations | `goa_human.gaf` | http://geneontology.org/gene-associations/goa_human.gaf.gz |
| Gene Ontology structure | `go-basic.obo` | http://purl.obolibrary.org/obo/go/go-basic.obo |
| circBase circRNA sequences | `circrna_sequences.json` | http://www.circbase.org/ (via `scripts/fetch_missing_circbase_seqs.py`) |
| HMDD v4.0 raw dump | `hmdd_v4_alldata.txt` | https://www.cuilab.cn/hmdd (processed subsets are already bundled) |

## Notes

- The default configuration uses MeSH disease features and GIP; k-mer and GO features are precomputed and cached as `.npy` files but the underlying sequence / GO databases are only needed to recompute them.
- lncRNA was removed from MRSECDA (three node types only), so the LncRNADisease files are intentionally not bundled. The code falls back gracefully when they are absent.
- All third-party data is provided under its original license; please cite the respective databases in any publication that uses them.
- The five benchmark datasets were collected and standardized by Lan et al. (2023) doi:10.1093/bib/bbac613. Two of the underlying sources carry notable terms: circR2Disease is marked "Copyright (c) Shaanxi Normal University, All Rights Reserved" (no open license), and HMDD v4.0 is "free only for academic users". Both are used here for academic research with citation; contact the respective maintainers for commercial use.