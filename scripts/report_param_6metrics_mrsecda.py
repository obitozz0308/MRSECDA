"""
Report all 6 metrics (AUC / AUPR / ACC / Precision / Recall / F1) for MRSECDA
parameter sensitivity, by reading the underlying mrsecda_*.json result files
pointed to by the progress file (the progress file only stores 3 of the 6).

Usage:
  python scripts/report_param_6metrics_mrsecda.py --dataset circR2Cancer
  python scripts/report_param_6metrics_mrsecda.py --dataset circR2Cancer --out results/mrsecda/param_6metrics_circR2Cancer.md
"""
import os, sys, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "results", "mrsecda")

PARAM_ORDER = ["lr", "hidden_dim", "dropout", "focal_gamma"]
PARAM_LABELS = {
    "lr": "Learning Rate",
    "hidden_dim": "Hidden Dimension",
    "dropout": "Dropout",
    "focal_gamma": "Focal Gamma",
}
DEFAULT_TAGS = {
    "lr": "lr=5e-4",
    "hidden_dim": "dim=96",
    "dropout": "drop=0.3",
    "focal_gamma": "gamma=2.0",
}
METRICS = ["auc", "aupr", "accuracy", "precision", "recall", "f1"]


def progress_path(dataset):
    suffix = "" if dataset == "circR2Disease" else f"_{dataset}"
    return os.path.join(RESULT_DIR, f"param_sensitivity_mrsecda{suffix}_progress.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="circR2Cancer")
    ap.add_argument("--out", default=None, help="optional markdown output path")
    args = ap.parse_args()

    with open(progress_path(args.dataset), encoding="utf-8") as f:
        progress = json.load(f)

    lines = []
    lines.append(f"# MRSECDA Parameter Sensitivity — 6 Metrics — {args.dataset}")
    lines.append("")
    lines.append("> ★ = default value. Metrics: AUC / AUPR / ACC / Precision / Recall / F1 (threshold=0.5 for ACC/P/R/F1).")
    lines.append("")

    for pk in PARAM_ORDER:
        lines.append(f"## {PARAM_LABELS[pk]}")
        lines.append("")
        header = "| Config | " + " | ".join(m.upper() for m in METRICS) + " |"
        lines.append(header)
        lines.append("|--------|" + "|".join(["----"] * len(METRICS)) + "|")

        for key, val in progress.items():
            if not val or not key.startswith(pk + "|"):
                continue
            tag = key.split("|", 1)[1]
            is_default = tag == DEFAULT_TAGS[pk]

            # Full 6 metrics live in the underlying result file
            full = {}
            rfile = val.get("result_file")
            if rfile:
                rp = os.path.join(RESULT_DIR, rfile)
                if os.path.exists(rp):
                    with open(rp, encoding="utf-8") as f:
                        data = json.load(f)
                    ds_data = next(iter(data.values()), {})
                    full = ds_data.get("summary", {})

            row = [tag + (" ★" if is_default else "")]
            for m in METRICS:
                if m in full:
                    row.append(f"{full[m]:.4f}")
                elif m in val:
                    row.append(f"{val[m]:.4f}")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    text = "\n".join(lines)
    print(text)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
