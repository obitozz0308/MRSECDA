"""
Ablation experiments on MRSECDA with the NEW parameter set.

New params (grid winners): L=3, H=8, hidden_dim=96, lr=5e-4, dropout=0.3
(focal_gamma=2.0 default). Three ablations × 5 datasets = 15 runs:

  No_PSE      --use_metapath_edges --no_struct_feat    drop pairwise structural encoder
  No_Metapath (omit --use_metapath_edges)              drop CMC/DMD metapath edges
  No_tokens   --use_metapath_edges --crt_single_token  drop Token0+Token3, keep Token1 (HGT)

Each run calls `scripts/run_mrsecda.py` as a subprocess; the 6-metric summary is
read from the JSON it saves ("Results saved to: <path>"). Resume via progress.json.
The Full (new-params) baseline is read from
results/mrsecda/new_params_6metrics/progress.json for Δ-vs-Full.

15 runs, 5 workers × 3 threads (16-core CPU).
"""
from __future__ import annotations

import os
import re
import sys
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATASETS = ["circR2Disease", "circR2Cancer", "circRNADisease", "circ2Traits", "circad"]

# New parameter set (grid winners) — shared base for every ablation
NEW_PARAMS = [
    "--seed", "42",
    "--device", "cpu",
    "--hgt_layers", "3",
    "--hgt_heads", "8",
    "--hidden_dim", "96",
    "--lr", "5e-4",
    "--dropout", "0.3",
]

# (tag, extra_args) — extra_args appended to the base command.
ABLATIONS = [
    ("No_PSE",      ["--use_metapath_edges", "--no_struct_feat"]),
    ("No_Metapath", []),                                              # omit metapath edges
    ("No_tokens",   ["--use_metapath_edges", "--crt_single_token"]),
]

MAX_WORKERS = 5
NUM_THREADS = "3"

OUT_DIR = os.path.join(ROOT, "results", "mrsecda", "new_params_ablation")
LOG_DIR = os.path.join(OUT_DIR, "logs")
PROGRESS_JSON = os.path.join(OUT_DIR, "progress.json")
TABLE_MD = os.path.join(OUT_DIR, "table.md")
os.makedirs(LOG_DIR, exist_ok=True)

RUN_SCRIPT = os.path.join(ROOT, "scripts", "run_mrsecda.py")

# Full (new-params) baseline, from the already-completed 6-metric run.
FULL_PROGRESS = os.path.join(ROOT, "results", "mrsecda", "new_params_6metrics", "progress.json")

METRICS = ["auc", "aupr", "accuracy", "precision", "recall", "f1"]


def _extract_summary(log_path: str) -> dict | None:
    """Read the 6-metric summary from the JSON saved by run_mrsecda.py."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, "rb") as f:
        data = f.read()
    paths = re.findall(rb"Results saved to:\s*([^\r\n]+)", data)
    if not paths:
        return None
    rp = paths[-1].decode("utf-8", errors="replace").strip()
    if not os.path.exists(rp):
        return None
    with open(rp, encoding="utf-8") as f:
        all_results = json.load(f)
    for _ds, res in all_results.items():
        s = res.get("summary", {})
        if s and "auc" in s:
            return {m: (s.get(m), s.get(f"{m}_std")) for m in METRICS}
    return None


def _load_full_baseline() -> dict:
    """Return {dataset: auc_mean} for the Full new-params baseline."""
    if not os.path.exists(FULL_PROGRESS):
        return {}
    with open(FULL_PROGRESS, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for ds, res in data.items():
        s = res.get("summary", {})
        auc = s.get("auc")
        if auc is not None:
            out[ds] = auc[0] if isinstance(auc, list) else auc
    return out


def run_one(ablation: str, dataset: str) -> dict:
    tag = f"{ablation}_{dataset}"
    log_path = os.path.join(LOG_DIR, f"{tag}.log")
    cmd = [sys.executable, RUN_SCRIPT, "--dataset", dataset] + NEW_PARAMS + dict(ABLATIONS)[ablation]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = NUM_THREADS
    env["MKL_NUM_THREADS"] = NUM_THREADS
    env["NUMEXPR_NUM_THREADS"] = NUM_THREADS

    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

    summary = _extract_summary(log_path)
    return {"ablation": ablation, "dataset": dataset, "returncode": proc.returncode, "summary": summary}


def _fmt(pair):
    if pair is None or pair[0] is None:
        return "·"
    return f"{pair[0]:.4f}±{pair[1]:.4f}"


def make_table(results: dict, full: dict) -> str:
    lines = ["# MRSECDA ablation (new params L=3 H=8 dim=96 lr=5e-4 drop=0.3) — AUC\n"]
    header = "| Ablation | " + " | ".join(DATASETS) + " | Mean |"
    sep = "|---|" + "|".join(["---"] * (len(DATASETS) + 1)) + "|"
    lines += [header, sep]

    # Full reference row
    full_row = ["| Full"]
    full_means = []
    for ds in DATASETS:
        a = full.get(ds)
        if a is not None:
            full_row.append(f" {a:.4f}")
            full_means.append(a)
        else:
            full_row.append(" ·")
    full_row.append(f" {sum(full_means)/len(full_means):.4f} |" if full_means else " · |")
    lines.append(" | ".join(full_row))

    for ab, _ in ABLATIONS:
        row = [f"| {ab}"]
        means = []
        for ds in DATASETS:
            r = results.get(f"{ab}_{ds}")
            pair = r["summary"]["auc"] if r and r.get("summary") else None
            if pair is None or pair[0] is None:
                row.append(" ·")
            else:
                row.append(f" {pair[0]:.4f}")
                means.append(pair[0])
        row.append(f" {sum(means)/len(means):.4f} |" if means else " · |")
        lines.append(" | ".join(row))

    # Δ vs Full
    lines += ["", "## Δ AUC vs Full (negative = worse)"]
    lines += [header, sep]
    for ab, _ in ABLATIONS:
        row = [f"| {ab}"]
        deltas = []
        for ds in DATASETS:
            r = results.get(f"{ab}_{ds}")
            pair = r["summary"]["auc"] if r and r.get("summary") else None
            fa = full.get(ds)
            if pair is None or pair[0] is None or fa is None:
                row.append(" ·")
            else:
                d = pair[0] - fa
                row.append(f" {d:+.4f}")
                deltas.append(d)
        row.append(f" {sum(deltas)/len(deltas):+.4f} |" if deltas else " · |")
        lines.append(" | ".join(row))
    return "\n".join(lines)


def main():
    results = {}
    lock = threading.Lock()
    full = _load_full_baseline()
    total = len(ABLATIONS) * len(DATASETS)

    # Resume: reuse configs whose log already yields a parseable summary.
    for ab, _ in ABLATIONS:
        for ds in DATASETS:
            tag = f"{ab}_{ds}"
            summary = _extract_summary(os.path.join(LOG_DIR, f"{tag}.log"))
            if summary is not None:
                results[tag] = {"ablation": ab, "dataset": ds, "returncode": 0, "summary": summary}

    to_run = [(ab, ds) for ab, _ in ABLATIONS for ds in DATASETS if f"{ab}_{ds}" not in results]
    print(f"Ablation (new params L=3 H=8 dim=96 lr=5e-4 drop=0.3): "
          f"{len(results)} cached, {len(to_run)} to run, "
          f"{MAX_WORKERS} workers x {NUM_THREADS} threads", flush=True)

    def handle(item):
        ab, ds = item
        r = run_one(ab, ds)
        with lock:
            results[f"{ab}_{ds}"] = r
            with open(PROGRESS_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            with open(TABLE_MD, "w", encoding="utf-8") as f:
                f.write(make_table(results, full))
                done = [v for v in results.values() if v.get("summary")]
                f.write(f"\n\ndone {len(done)}/{total}\n")
        ok = r.get("summary") is not None
        auc = r["summary"]["auc"][0] if ok else None
        print(f"[{'OK' if ok else 'FAIL'}] {ab}_{ds}  AUC={auc if auc is None else f'{auc:.4f}'}  "
              f"rc={r['returncode']}  ({len(results)}/{total} done)", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(handle, item) for item in to_run]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] {e}", flush=True)

    print("\n=== FINAL TABLE ===", flush=True)
    print(make_table(results, full), flush=True)


if __name__ == "__main__":
    main()
