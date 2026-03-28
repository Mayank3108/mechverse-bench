"""
Queries all 3 models via OpenRouter with pass@2:
  - GPT-4.5
  - Claude Opus 4.6
  - Gemini 2.5 Pro

For each question, each model is queried TWICE (pass@2).
Both query_wo and query_cot variants are run.

Output files (in results/assessment/):
    gpt_run1_<timestamp>.json
    gpt_run2_<timestamp>.json
    opus_run1_<timestamp>.json
    opus_run2_<timestamp>.json
    gemini_run1_<timestamp>.json
    gemini_run2_<timestamp>.json
"""

import json
import os
import re
import sys
from datetime import datetime
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "models"))
from openrouter_model import OpenRouterModel


NUM_SAMPLES = "1"
MODELS_TO_RUN = ["gpt", "opus", "gemini"]
ANNOTATIONS_PATH = os.path.join(REPO_ROOT, "dataset", "annotations.json")
IMAGES_BASE_DIR  = os.path.join(REPO_ROOT, "dataset")
RESULTS_DIR      = os.path.join(REPO_ROOT, "results", "assessment")
FORMAT_WO  = "\nReply with only a single letter: A, B, C, or D. Nothing else."
FORMAT_COT = "\nFirst explain your reasoning step by step. Then on the very last line write only the answer letter (A, B, C, or D) — nothing else, no punctuation, no colon, no label."



def extract_wo_answer(response: str) -> str:
    if response in ("ERROR", "EMPTY"):
        return response
    if not response or not response.strip():
        return "UNKNOWN"
    cleaned = response.strip().upper()
    if cleaned in ["A", "B", "C", "D"]:
        return cleaned
    match = re.match(r'^([A-D])', cleaned)
    return match.group(1) if match else "UNKNOWN"


def extract_cot_answer(response: str) -> str:
    if response in ("ERROR", "EMPTY"):
        return response
    if not response or not response.strip():
        return "UNKNOWN"
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    if not lines:
        return "UNKNOWN"
    for line in reversed(lines[-3:]):
        clean = line.upper().strip("*:. ")
        if clean in ["A", "B", "C", "D"]:
            return clean
        match = re.search(r'\b([A-D])\b', clean)
        if match:
            return match.group(1)
    return "UNKNOWN"


def write_json(path: str, summary: dict, results: list):
    with open(path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)



def run_single(model_key: str, run_number: int, annotations: list, timestamp_str: str):

    os.makedirs(RESULTS_DIR, exist_ok=True)

    suffix      = f"_n{NUM_SAMPLES}" if NUM_SAMPLES != "all" else ""
    output_path = os.path.join(
        RESULTS_DIR,
        f"{model_key}_run{run_number}{suffix}_{timestamp_str}.json"
    )

    print(f"\n{'='*55}")
    print(f"  Model : {model_key.upper()}  |  Run : {run_number}/2")
    print(f"  Output: {os.path.basename(output_path)}")
    print(f"{'='*55}\n")

    model = OpenRouterModel(model_key=model_key)

    correct_wo  = 0
    correct_cot = 0
    total       = 0
    errors      = 0
    results     = []

    write_json(output_path, {
        "model"             : model_key,
        "run_number"        : run_number,
        "timestamp"         : datetime.now().isoformat(),
        "status"            : "in_progress",
        "total_entries"     : 0,
        "errors_skipped"    : 0,
        "query_wo_accuracy" : 0,
        "query_cot_accuracy": 0,
    }, [])

    for entry in tqdm(annotations, desc=f"{model_key} run{run_number}", unit="sample"):

        if entry.get("is_grid_question", False) and entry.get("image_grid"):
            image_rel_path = entry["image_grid"]
        else:
            image_rel_path = entry["image"]

        image_full_path = os.path.join(IMAGES_BASE_DIR, image_rel_path)

        if not os.path.exists(image_full_path):
            print(f"  [WARNING] Image not found: {image_full_path}")
            errors += 1
            continue

        response_wo  = model.run(image_full_path, entry["query_wo"] + FORMAT_WO)
        predicted_wo = extract_wo_answer(response_wo)
        wo_correct   = predicted_wo == entry["answer"]
        correct_wo  += int(wo_correct)

        response_cot  = model.run(image_full_path, entry["query_cot"] + FORMAT_COT)
        predicted_cot = extract_cot_answer(response_cot)
        cot_correct   = predicted_cot == entry["answer"]
        correct_cot  += int(cot_correct)

        total += 1

        results.append({
            "sample_index"           : entry["sample_index"],
            "problem_index"          : entry["problem_index"],
            "dimension"              : entry["dimension"],
            "category"               : entry["category"],
            "question_type"          : entry["question_type"],
            "is_grid_question"       : entry["is_grid_question"],
            "image"                  : image_rel_path,
            "ground_truth"           : entry["answer"],
            "query_wo_response"      : response_wo,
            "query_wo_predicted"     : predicted_wo,
            "query_wo_correct"       : wo_correct,
            "query_cot_full_response": response_cot,
            "query_cot_predicted"    : predicted_cot,
            "query_cot_correct"      : cot_correct,
        })

        acc_wo  = (correct_wo  / total * 100) if total > 0 else 0
        acc_cot = (correct_cot / total * 100) if total > 0 else 0
        write_json(output_path, {
            "model"             : model_key,
            "run_number"        : run_number,
            "timestamp"         : datetime.now().isoformat(),
            "status"            : "in_progress",
            "total_entries"     : total,
            "errors_skipped"    : errors,
            "query_wo_accuracy" : round(acc_wo,  2),
            "query_cot_accuracy": round(acc_cot, 2),
        }, results)

    acc_wo  = (correct_wo  / total * 100) if total > 0 else 0
    acc_cot = (correct_cot / total * 100) if total > 0 else 0
    write_json(output_path, {
        "model"             : model_key,
        "run_number"        : run_number,
        "timestamp"         : datetime.now().isoformat(),
        "status"            : "complete",
        "total_entries"     : total,
        "errors_skipped"    : errors,
        "query_wo_accuracy" : round(acc_wo,  2),
        "query_cot_accuracy": round(acc_cot, 2),
    }, results)

    print(f"\n  DONE — {model_key.upper()} run {run_number}")
    print(f"  query_wo  accuracy : {acc_wo:.2f}%")
    print(f"  query_cot accuracy : {acc_cot:.2f}%")

    return output_path



def run_assessment():
    with open(ANNOTATIONS_PATH, "r") as f:
        all_annotations = json.load(f)

    if NUM_SAMPLES == "all":
        annotations = all_annotations
    else:
        annotations = all_annotations[:int(NUM_SAMPLES)]

    print(f"\n{'='*55}")
    print(f"  MECHVERSE ASSESSMENT RUNNER")
    print(f"  Entries   : {len(annotations)}")
    print(f"  Models    : {MODELS_TO_RUN}")
    print(f"  Runs/model: 2  (pass@2)")
    print(f"  Total API calls: {len(annotations) * len(MODELS_TO_RUN) * 2 * 2}")
    print(f"  (x2 for query_wo and query_cot)")
    print(f"{'='*55}")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_outputs   = {}

    for model_key in MODELS_TO_RUN:
        all_outputs[model_key] = []
        for run_number in [1, 2]:  
            output_path = run_single(
                model_key    = model_key,
                run_number   = run_number,
                annotations  = annotations,
                timestamp_str= timestamp_str,
            )
            all_outputs[model_key].append(output_path)

    print(f"\n{'='*55}")
    print(f"  ALL RUNS COMPLETE")
    print(f"  Results saved to: {RESULTS_DIR}")
    for model_key, paths in all_outputs.items():
        for p in paths:
            print(f"    {os.path.basename(p)}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_assessment()
