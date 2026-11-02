"""
Re-evaluate v3.3 Default on all 5 datasets with per-fold optimal threshold.
Only affects F1/ACC/PRE/REC — AUC/AUPR are threshold-invariant.

Usage: python scripts/run_per_fold_threshold.py
"""
import os, sys, json, time
import numpy as np

# Add project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config
from scripts.run_mrsecda import run_dataset_hetero

DEVICE = "cpu"
SEED = 42
DATASETS = ["circR2Cancer", "circR2Disease", "circRNADisease", "circ2Traits", "circad"]

config = Config()
config.train.device = DEVICE
# v3.3 Default config
config.model.use_metapath_edges = True
config.train.finetune_lr = 3e-4
config.model.hidden_dim = 96
config.model.dropout = 0.3
config.model.focal_gamma = 2.0

results = {}

for ds in DATASETS:
    print(f"\n{'='*70}")
    print(f"  Dataset: {ds}")
    print(f"{'='*70}")

    t0 = time.time()
    result = run_dataset_hetero(
        dataset_name=ds, config=config, device=DEVICE,
        ablation=None, seed=SEED, verbose=True,
    )
    elapsed = time.time() - t0

    summary = result.get("summary", {})
    fold_results = result.get("fold_results", [])

    aucs = [r["auc"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]
    accs = [r["accuracy"] for r in fold_results]
    precs = [r["precision"] for r in fold_results]
    recs = [r["recall"] for r in fold_results]

    results[ds] = {
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "aupr_mean": summary.get("aupr", None),
        "aupr_std": summary.get("aupr_std", None),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "prec_mean": float(np.mean(precs)), "prec_std": float(np.std(precs)),
        "rec_mean": float(np.mean(recs)), "rec_std": float(np.std(recs)),
        "per_fold_auc": [float(a) for a in aucs],
        "per_fold_f1": [float(f) for f in f1s],
        "per_fold_acc": [float(a) for a in accs],
        "per_fold_prec": [float(p) for p in precs],
        "per_fold_rec": [float(r) for r in recs],
        "time_s": int(elapsed),
    }

    # Save incrementally
    out_path = os.path.join(ROOT, "results", "mrsecda", "per_fold_threshold_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  {ds} done in {elapsed:.0f}s")
    print(f"  AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

print(f"\n{'='*70}")
print(f"  All done! → {out_path}")
print(f"{'='*70}")
