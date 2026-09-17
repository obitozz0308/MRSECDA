# MRSECDA

**Pairwise Structural Feature Encoding with Multi-Resolution Heterogeneous Graph Learning for Predicting CircRNA–Disease Associations.**

MRSECDA is a heterogeneous graph neural network that predicts associations between circular RNAs (circRNAs) and diseases. It introduces **PSE** (Pairwise Structural Encoder, a multi-view pairwise structural feature encoder) and a **multi-token heterogeneous encoder** with meta-path reasoning to recover the pairwise structural signal that plain node embeddings cannot capture.

## Method overview

1. **Heterogeneous graph** with three node types — circRNA, disease, and miRNA — connected by circRNA–disease associations, miRNA–disease associations (HMDD v4.0), and circRNA–miRNA interactions (ENCORI/starBase + curated data).
2. **PSE (Pairwise Structural Encoder)**: computes Jaccard and Adamic-Adar structural features over three subgraph views via sparse-matrix operations and injects them into a dual-pathway predictor.
3. **Multi-token heterogeneous encoder**: preserves type-specific tokens and fuses them through cross-type routing (Heterogeneous Graph Transformer), instead of collapsing heterogeneous neighborhoods into a single token.
4. **Meta-path edges (CMC / DMD)**: 2-hop meta-path edge types that let the model reason along the circRNA–miRNA–disease pathway.

## Requirements

- Python 3.9+
- PyTorch 2.0+ and PyTorch Geometric 2.3+

```bash
pip install -r requirements.txt
```

> For `torch-scatter` / `torch-sparse`, install the wheels matching your PyTorch + CUDA version from the [PyTorch Geometric installation page](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Data

The repository ships with the input data needed to reproduce the results:

- `data/CompareExperiment/Data/` — association matrices and node lists for the five benchmark datasets (circR2Disease, circR2Cancer, circRNADisease, circ2Traits, circad) and the circRNA–miRNA interaction matrix.
- `data/CompareExperiment/AE-RF main/KHGAT/data/` — miRNA/lncRNA list and interaction files.
- `data/external/` — precomputed features (GIP, MeSH, k-mer, GO, miRNA k-mer) and the processed HMDD v4.0 / ENCORI / miRBase inputs.

The large third-party raw databases (Gene Ontology, circBase sequences, raw HMDD dump) are **not** bundled. The pipeline downloads them automatically on first use. See [DATA_SOURCES.md](DATA_SOURCES.md) for the exact source URLs and licenses.

## Quick start

Train and evaluate on a single dataset (5-fold CV, seed 42):

```bash
python scripts/run_mrsecda.py --dataset circR2Disease
```

Reproduce the full model on all five datasets with the paper's final configuration (HGT depth L=3, heads H=8, hidden dim 96, lr 5e-4, dropout 0.3, focal loss γ=2.0, CMC/DMD meta-path edges):

```bash
python scripts/run_mrsecda.py \
    --seed 42 --device cpu \
    --hgt_layers 3 --hgt_heads 8 \
    --hidden_dim 96 --lr 5e-4 --dropout 0.3 \
    --use_metapath_edges \
    --datasets circR2Disease circR2Cancer circRNADisease circ2Traits circad
```

Run the ablation study (No_PSE / No_Metapath / No_tokens × 5 datasets):

```bash
python scripts/run_ablation_mrsecda_newparams.py
```

Run the PSE subgraph-view ablation:

```bash
python scripts/run_ablation_pse_views.py
```

Run the parameter sensitivity analysis:

```bash
python scripts/run_param_sensitivity_mrsecda.py
```

Run the top-ranked case study (three cancers):

```bash
python scripts/run_case_study_mrsecda.py
```

Report the six metrics with per-fold optimal threshold:

```bash
python scripts/report_param_6metrics_mrsecda.py
```

Results are written to `results/mrsecda/` as JSON + Markdown. `config.py` defaults to the paper's final configuration (L=3, H=8, dim=96, lr=5e-4, dropout=0.3, `--use_metapath_edges`, seed 42); the explicit flags above are equivalent and shown for transparency.

## Project structure

```
.
├── config.py                 # global configuration (model / data / training)
├── trainer.py                # training, validation, and evaluation utilities
├── models/
│   └── mrsecda_hetero.py   # heterogeneous model (main)
├── data/
│   ├── dataset.py            # dataset loading, splitting, graph construction
│   ├── hetero_data.py        # heterogeneous graph builder
│   ├── structural_features.py# PSE pairwise structural feature encoder
│   ├── metapath_encoder.py   # meta-path neighbor encoding
│   ├── similarity.py         # GIP kernel and similarity fusion
│   ├── external_knowledge.py # MeSH / k-mer / GO / miRNA features
│   └── ...                   # database parsers and preprocessing
├── scripts/
│   ├── run_mrsecda.py               # main training / evaluation entry point
│   ├── run_ablation_mrsecda_newparams.py  # final 3-way ablation (No_PSE/No_Metapath/No_tokens)
│   ├── run_ablation_pse_views.py    # PSE subgraph-view ablation
│   ├── run_case_study_mrsecda.py    # top-ranked case study (3 cancers)
│   ├── run_param_sensitivity_mrsecda.py  # parameter sensitivity
│   ├── run_per_fold_threshold.py    # per-fold optimal threshold
│   └── report_param_6metrics_mrsecda.py  # 6-metric report
└── data/
    ├── CompareExperiment/    # benchmark datasets + KHGAT lists
    └── external/             # precomputed features + processed DB inputs
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{MRSECDA,
  title   = {Pairwise Structural Feature Encoding with Multi-Resolution Heterogeneous Graph Learning for Predicting CircRNA-Disease Associations},
  author  = {Huang, Shuaifu},
  journal = {Bioinformatics},
  year    = {2026},
  note    = {under review}
}
```

## License

[MIT](LICENSE)

## Contact

For questions or comments, please contact the corresponding author (email to be added upon publication).