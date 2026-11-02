"""
MRSECDA Parameter Sensitivity Analysis — 4 parameters × 5 values on circR2Disease

MRSECDA = v3.3 架构 + 默认去 lncRNA + PSE 2 特征（jaccard+AA）。
All experiments: seed=42, 5-fold CV, --use_metapath_edges.

Usage:
  python scripts/run_param_sensitivity_mrsecda.py                    # Run all
  python scripts/run_param_sensitivity_mrsecda.py --resume           # Resume
  python scripts/run_param_sensitivity_mrsecda.py --param lr         # Single param only
"""

from __future__ import annotations
import os, sys, json, time, subprocess, argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULT_DIR = os.path.join(ROOT, "results", "mrsecda")
PROGRESS_FILE = os.path.join(RESULT_DIR, "param_sensitivity_mrsecda_progress.json")
REPORT_FILE = os.path.join(RESULT_DIR, "param_sensitivity_mrsecda_report.md")

DATASET = "circR2Disease"
BASE_ARGS = f"--datasets {DATASET} --seed 42 --use_metapath_edges"

# Default values (marked with ★ in reports)
DEFAULTS = {
    "lr": 3e-4,
    "hidden_dim": 96,
    "dropout": 0.3,
    "focal_gamma": 2.0,
}

# ── Parameter Grids ──
PARAM_GRIDS = {
    "lr": ("Learning Rate", [
        ("lr=5e-5",  "--lr 5e-5"),
        ("lr=1e-4",  "--lr 1e-4"),
        ("lr=3e-4",  "--lr 3e-4"),    # ★ default
        ("lr=5e-4",  "--lr 5e-4"),
        ("lr=1e-3",  "--lr 1e-3"),
    ]),
    "hidden_dim": ("Hidden Dimension", [
        ("dim=64",   "--hidden_dim 64"),
        ("dim=96",   "--hidden_dim 96"),   # ★ default
        ("dim=128",  "--hidden_dim 128"),
        ("dim=192",  "--hidden_dim 192"),
        ("dim=256",  "--hidden_dim 256"),
    ]),
    "dropout": ("Dropout", [
        ("drop=0.1", "--dropout 0.1"),
        ("drop=0.2", "--dropout 0.2"),
        ("drop=0.3", "--dropout 0.3"),    # ★ default
        ("drop=0.4", "--dropout 0.4"),
        ("drop=0.5", "--dropout 0.5"),
    ]),
    "focal_gamma": ("Focal Gamma", [
        ("gamma=0.0", "--focal_gamma 0"),
        ("gamma=0.5", "--focal_gamma 0.5"),
        ("gamma=1.0", "--focal_gamma 1.0"),
        ("gamma=2.0", "--focal_gamma 2.0"),   # ★ default
        ("gamma=3.0", "--focal_gamma 3.0"),
    ]),
}

# Order for report display
PARAM_ORDER = ["lr", "hidden_dim", "dropout", "focal_gamma"]


def run_one(tag: str, extra_args: str) -> dict | None:
    """Run a single parameter config. Returns parsed summary dict or None."""
    cmd = f'python scripts/run_mrsecda.py {BASE_ARGS} {extra_args}'
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {tag}")
    print(f"  CMD: {cmd}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(
            cmd, shell=True, cwd=ROOT, capture_output=True, text=True,
            timeout=7200,
        )
        result_files = sorted(
            [f for f in os.listdir(RESULT_DIR) if f.endswith('.json')],
            key=lambda x: os.path.getmtime(os.path.join(RESULT_DIR, x)),
            reverse=True,
        )
        if result_files:
            latest = os.path.join(RESULT_DIR, result_files[0])
            with open(latest) as f:
                data = json.load(f)
            for ds_name, ds_result in data.items():
                s = ds_result.get("summary", {})
                return {
                    "auc": round(s.get("auc", 0), 4),
                    "auc_std": round(s.get("auc_std", 0), 4),
                    "aupr": round(s.get("aupr", 0), 4),
                    "aupr_std": round(s.get("aupr_std", 0), 4),
                    "f1": round(s.get("f1", 0), 4),
                    "f1_std": round(s.get("f1_std", 0), 4),
                    "result_file": result_files[0],
                }
        return None
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 2h!")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def save_progress(results: dict):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding='utf-8') as f:
            return json.load(f)
    return {}


def generate_report(results: dict):
    """Generate a comprehensive markdown report."""
    lines = []
    lines.append(f"# MRSECDA Parameter Sensitivity Analysis — {DATASET}")
    lines.append("")
    lines.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> All experiments: seed=42, 5-fold CV, `--use_metapath_edges`, no lncRNA, PSE 2-feat (jaccard+AA)")
    lines.append(f"> ★ = current default value")
    lines.append("")

    for param_key in PARAM_ORDER:
        param_label, variants = PARAM_GRIDS[param_key]
        default_val = DEFAULTS[param_key]
        lines.append(f"## {param_label}")
        lines.append("")
        lines.append("| Config | AUC | AUPR | F1 | Δ AUC vs Default |")
        lines.append("|--------|-----|------|----|:---:|")

        default_auc = None
        rows = []

        for tag, _extra_args in variants:
            key = f"{param_key}|{tag}"
            if key in results and results[key]:
                r = results[key]
                auc = r.get("auc", 0)
                aupr = r.get("aupr", 0)
                f1 = r.get("f1", 0)
                try:
                    val = float(_extra_args.split()[-1])
                    is_default = abs(val - default_val) < 1e-9
                except ValueError:
                    is_default = False
                rows.append((tag, auc, aupr, f1, _extra_args))
                if is_default:
                    default_auc = auc
            else:
                rows.append((tag, None, None, None, _extra_args))

        valid_rows = [(t, a, ap, f1, ex) for t, a, ap, f1, ex in rows if a is not None]
        if valid_rows:
            best_auc = max(valid_rows, key=lambda x: x[1])
            best_aupr = max(valid_rows, key=lambda x: x[2])

        for tag, auc, aupr, f1, extra_args in rows:
            if auc is None:
                lines.append(f"| {tag} | — | — | — | — |")
                continue
            try:
                is_def = abs(float(extra_args.split()[-1]) - default_val) < 1e-9
            except ValueError:
                is_def = False
            marker = " ★" if is_def else ""
            delta_str = ""
            if default_auc is not None and not is_def:
                delta = auc - default_auc
                sign = "+" if delta > 0 else ""
                delta_str = f" {sign}{delta:.4f}"
            else:
                delta_str = " —"
            lines.append(f"| {tag}{marker} | {auc:.4f} ± {results.get(f'{param_key}|{tag}', {}).get('auc_std', 0):.4f} | "
                        f"{aupr:.4f} ± {results.get(f'{param_key}|{tag}', {}).get('aupr_std', 0):.4f} | "
                        f"{f1:.4f} |{delta_str} |")

        lines.append("")
        if valid_rows:
            lines.append(f"**Best AUC**: {best_auc[0]} ({best_auc[1]:.4f})")
            lines.append(f"**Best AUPR**: {best_aupr[0]} ({best_aupr[2]:.4f})")
            lines.append("")

    lines.append("## Optimal Configuration Summary")
    lines.append("")
    lines.append("| Parameter | Default | Best | Best AUC | Recommendation |")
    lines.append("|-----------|--------|------|----------|----------------|")

    recommendations = {}
    for param_key in PARAM_ORDER:
        param_label, variants = PARAM_GRIDS[param_key]
        valid_rows = []
        for tag, extra_args in variants:
            key = f"{param_key}|{tag}"
            if key in results and results[key]:
                valid_rows.append((tag, results[key]["auc"], extra_args))
        if valid_rows:
            best = max(valid_rows, key=lambda x: x[1])
            default_tag = [v[0] for v in variants if abs(float(v[1].split()[-1]) - DEFAULTS[param_key]) < 1e-9]
            default_key = f"{param_key}|{default_tag[0]}" if default_tag else None
            default_auc = results.get(default_key, {}).get("auc", 0) if default_key else 0
            delta = best[1] - default_auc
            if delta > 0.002:
                rec = f"⚠️ Change to {best[0]}"
                recommendations[param_key] = best
            elif delta > -0.002:
                rec = "✅ Keep default"
            else:
                rec = f"⚠️ Consider {best[0]}"
                recommendations[param_key] = best
            lines.append(f"| {param_label} | {DEFAULTS[param_key]} | {best[0]} | {best[1]:.4f} | {rec} |")

    lines.append("")
    if recommendations:
        lines.append("### Recommended Changes")
        lines.append("```bash")
        cmd_parts = ["python scripts/run_mrsecda.py --use_metapath_edges"]
        for pk, best in recommendations.items():
            _, _, extra_args = best
            cmd_parts.append(extra_args.split(" ", 1)[1] if " " in extra_args else extra_args)
        lines.append(" \\\n    ".join(cmd_parts))
        lines.append("```")
    else:
        lines.append("**All parameters optimal at defaults. No changes needed.**")
    lines.append("")

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nReport saved to: {REPORT_FILE}")


def main():
    global DATASET, BASE_ARGS, PROGRESS_FILE, REPORT_FILE

    parser = argparse.ArgumentParser(description="MRSECDA Parameter Sensitivity Runner")
    parser.add_argument("--resume", action="store_true", help="Resume from saved progress")
    parser.add_argument("--param", type=str, default=None,
                        choices=list(PARAM_GRIDS.keys()),
                        help="Single parameter to sweep (default: all 4)")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="Dataset to sweep on (default: circR2Disease)")
    args = parser.parse_args()

    DATASET = args.dataset
    BASE_ARGS = f"--datasets {DATASET} --seed 42 --use_metapath_edges"
    _suffix = "" if DATASET == "circR2Disease" else f"_{DATASET}"
    PROGRESS_FILE = os.path.join(RESULT_DIR, f"param_sensitivity_mrsecda{_suffix}_progress.json")
    REPORT_FILE = os.path.join(RESULT_DIR, f"param_sensitivity_mrsecda{_suffix}_report.md")

    params_to_run = [args.param] if args.param else PARAM_ORDER

    if args.resume:
        results = load_progress()
        print(f"Resumed with {len(results)} completed experiments")
    else:
        results = {}
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)

    total = sum(len(PARAM_GRIDS[pk][1]) for pk in params_to_run)
    done = 0
    failed = []

    for param_key in params_to_run:
        param_label, variants = PARAM_GRIDS[param_key]
        print(f"\n{'#'*70}")
        print(f"# PARAMETER: {param_label} ({len(variants)} values)")
        print(f"{'#'*70}")

        for tag, extra_args in variants:
            key = f"{param_key}|{tag}"
            if key in results:
                done += 1
                r = results[key]
                print(f"[{done}/{total}] SKIP {key} (already done): AUC={r.get('auc', '?')}")
                continue

            summary = run_one(tag, extra_args)
            if summary:
                results[key] = summary
                done += 1
                print(f"[{done}/{total}] OK {key}: AUC={summary['auc']:.4f}, AUPR={summary['aupr']:.4f}")
            else:
                failed.append(key)
                done += 1
                print(f"[{done}/{total}] FAIL {key}")

            save_progress(results)

    print(f"\n{'='*70}")
    print(f"Done! {done}/{total} completed, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")
    print(f"{'='*70}")

    generate_report(results)

    print(f"\n{'='*80}")
    print("QUICK SUMMARY — AUC (★ = default)")
    print(f"{'='*80}")
    for param_key in params_to_run:
        param_label, variants = PARAM_GRIDS[param_key]
        print(f"\n{param_label}:")
        for tag, extra_args in variants:
            key = f"{param_key}|{tag}"
            if key in results and results[key]:
                r = results[key]
                is_def = abs(float(extra_args.split()[-1]) - DEFAULTS[param_key]) < 1e-9
                marker = " ★" if is_def else "  "
                print(f"  {tag:<12} {marker}  AUC={r['auc']:.4f}±{r['auc_std']:.4f}  AUPR={r['aupr']:.4f}±{r['aupr_std']:.4f}  F1={r['f1']:.4f}")


if __name__ == "__main__":
    main()
