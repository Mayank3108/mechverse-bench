"""
Reads the 6 result files from run_assessment.py and computes:
  - pass@2 per model (overall)
  - pass@2 per model per dimension
  - pass@2 per model per category
  - Comparison table across all 3 models

Output:
    - Prints full results table to terminal
    - Saves results/assessment/pass_at_2_scores.json
    - Saves results/assessment/pass_at_2_report.txt 
"""

import json
import os
import sys
import glob
from collections import defaultdict

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "assessment")



def load_results(model_key: str, run_number: int) -> list:
    pattern = os.path.join(RESULTS_DIR, f"{model_key}_run{run_number}_*.json")
    files   = glob.glob(pattern)

    if not files:
        print(f"  [WARNING] No file found for {model_key} run{run_number}")
        print(f"  Looked for: {pattern}")
        return []

    latest = sorted(files)[-1]
    print(f"  Loading: {os.path.basename(latest)}")

    with open(latest, "r") as f:
        data = json.load(f)

    return data.get("results", [])



def compute_pass_at_2(run1: list, run2: list, prompt_type: str) -> dict:
    """
    Compute pass@2 for a given prompt type (query_wo or query_cot).

    Returns dict keyed by sample_index with pass/fail boolean.
    """
    correct_key = f"{prompt_type}_correct"

    run1_by_idx = {str(r["sample_index"]): r for r in run1}
    run2_by_idx = {str(r["sample_index"]): r for r in run2}

    results = {}
    for idx, r1 in run1_by_idx.items():
        r2          = run2_by_idx.get(idx)
        pass_run1   = r1.get(correct_key, False)
        pass_run2   = r2.get(correct_key, False) if r2 else False
        passed      = pass_run1 or pass_run2

        results[idx] = {
            "passed"     : passed,
            "run1"       : pass_run1,
            "run2"       : pass_run2,
            "dimension"  : r1.get("dimension", "Unknown"),
            "category"   : r1.get("category",  "Unknown"),
            "image"      : r1.get("image",      ""),
        }

    return results


def score_summary(pass_at_2: dict) -> dict:
    """Compute overall, per-dimension, and per-category pass@2 scores."""

    total   = len(pass_at_2)
    passed  = sum(1 for v in pass_at_2.values() if v["passed"])
    overall = round(passed / total * 100, 1) if total > 0 else 0.0

    dim_total  = defaultdict(int)
    dim_passed = defaultdict(int)
    for v in pass_at_2.values():
        d = v["dimension"]
        dim_total[d]  += 1
        dim_passed[d] += int(v["passed"])

    by_dimension = {
        d: {
            "passed" : dim_passed[d],
            "total"  : dim_total[d],
            "pct"    : round(dim_passed[d] / dim_total[d] * 100, 1)
        }
        for d in sorted(dim_total.keys())
    }

    cat_total  = defaultdict(int)
    cat_passed = defaultdict(int)
    for v in pass_at_2.values():
        c = v["category"]
        cat_total[c]  += 1
        cat_passed[c] += int(v["passed"])

    by_category = {
        c: {
            "passed" : cat_passed[c],
            "total"  : cat_total[c],
            "pct"    : round(cat_passed[c] / cat_total[c] * 100, 1)
        }
        for c in sorted(cat_total.keys())
    }

    return {
        "overall"      : {"passed": passed, "total": total, "pct": overall},
        "by_dimension" : by_dimension,
        "by_category"  : by_category,
    }



def format_report(all_scores: dict) -> str:
    lines = []

    lines.append("=" * 70)
    lines.append("  MECHVERSE ASSESSMENT — pass@2 RESULTS")
    lines.append("=" * 70)

    # ── Overall summary ──
    lines.append("\n  OVERALL pass@2 ACCURACY")
    lines.append("-" * 70)
    lines.append(f"  {'Model':<15} {'query_wo':>12} {'query_cot':>12}")
    lines.append("-" * 70)

    for model_key in ["gpt", "opus", "gemini"]:
        if model_key not in all_scores:
            continue
        wo  = all_scores[model_key]["wo"]["overall"]["pct"]
        cot = all_scores[model_key]["cot"]["overall"]["pct"]
        lines.append(f"  {model_key.upper():<15} {wo:>11}% {cot:>11}%")

    lines.append("-" * 70)
    lines.append("  (Random chance baseline: 25.0%)")

    lines.append("\n\n  pass@2 BY DIMENSION")
    lines.append("-" * 70)

    dims = set()
    for model_key in ["gpt", "opus", "gemini"]:
        if model_key in all_scores:
            dims.update(all_scores[model_key]["wo"]["by_dimension"].keys())

    for dim in sorted(dims):
        lines.append(f"\n  {dim}")
        lines.append(f"  {'Model':<15} {'query_wo':>12} {'query_cot':>12}")
        lines.append(f"  {'-'*40}")
        for model_key in ["gpt", "opus", "gemini"]:
            if model_key not in all_scores:
                continue
            wo_dim  = all_scores[model_key]["wo"]["by_dimension"].get(dim, {})
            cot_dim = all_scores[model_key]["cot"]["by_dimension"].get(dim, {})
            wo_pct  = wo_dim.get("pct",  "N/A")
            cot_pct = cot_dim.get("pct", "N/A")
            wo_str  = f"{wo_pct}%" if isinstance(wo_pct,  float) else wo_pct
            cot_str = f"{cot_pct}%" if isinstance(cot_pct, float) else cot_pct
            lines.append(f"  {model_key.upper():<15} {wo_str:>12} {cot_str:>12}")

    lines.append("\n\n  pass@2 BY CATEGORY")
    lines.append("-" * 70)

    cats = set()
    for model_key in ["gpt", "opus", "gemini"]:
        if model_key in all_scores:
            cats.update(all_scores[model_key]["wo"]["by_category"].keys())

    for cat in sorted(cats):
        lines.append(f"\n  {cat}")
        lines.append(f"  {'Model':<15} {'query_wo':>12} {'query_cot':>12} {'n':>5}")
        lines.append(f"  {'-'*48}")
        for model_key in ["gpt", "opus", "gemini"]:
            if model_key not in all_scores:
                continue
            wo_cat  = all_scores[model_key]["wo"]["by_category"].get(cat, {})
            cot_cat = all_scores[model_key]["cot"]["by_category"].get(cat, {})
            wo_pct  = wo_cat.get("pct",   "N/A")
            cot_pct = cot_cat.get("pct",  "N/A")
            n       = wo_cat.get("total", "N/A")
            wo_str  = f"{wo_pct}%" if isinstance(wo_pct,  float) else wo_pct
            cot_str = f"{cot_pct}%" if isinstance(cot_pct, float) else cot_pct
            lines.append(f"  {model_key.upper():<15} {wo_str:>12} {cot_str:>12} {str(n):>5}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)



def compute_all_scores():
    print(f"\n{'='*55}")
    print(f"  COMPUTING pass@2 SCORES")
    print(f"{'='*55}\n")

    model_keys = ["gpt", "opus", "gemini"]
    all_scores = {}

    for model_key in model_keys:
        print(f"\n  [{model_key.upper()}]")
        run1 = load_results(model_key, 1)
        run2 = load_results(model_key, 2)

        if not run1 or not run2:
            print(f"  Skipping {model_key} — missing result files")
            continue

        pass2_wo  = compute_pass_at_2(run1, run2, "query_wo")
        pass2_cot = compute_pass_at_2(run1, run2, "query_cot")

        all_scores[model_key] = {
            "wo"  : score_summary(pass2_wo),
            "cot" : score_summary(pass2_cot),
        }

    report = format_report(all_scores)
    print("\n" + report)

    json_path = os.path.join(RESULTS_DIR, "pass_at_2_scores.json")
    with open(json_path, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"\n  JSON saved to : {json_path}")

    txt_path = os.path.join(RESULTS_DIR, "pass_at_2_report.txt")
    with open(txt_path, "w") as f:
        f.write(report)
    print(f"  TXT  saved to : {txt_path}")


if __name__ == "__main__":
    compute_all_scores()
