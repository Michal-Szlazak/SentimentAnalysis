"""
Evaluate sentiment classification results against ground truth.

Usage:
    python evaluate.py                  # evaluate all datasets
    python evaluate.py twitter imdb     # evaluate specific datasets
    python evaluate.py --model llama    # evaluate llama outputs
    python evaluate.py --results-dir c:/path/to/results_folder
    python evaluate.py --f1 macro       # choose F1 averaging: macro | weighted | both (default: both)
"""

import json
import argparse
from pathlib import Path

from sklearn.metrics import f1_score, accuracy_score, classification_report

SUBSETS_DIR = Path(__file__).parent / "subsets"

MODEL_DIR_DEFAULTS = {
    "gemini": Path(__file__).parent / "results_gemini",
    "llama": Path(__file__).parent / "results_llama_reindexed",
    "mistral": Path(__file__).parent / "results_mistral",
    "claude": Path(__file__).parent / "results_claude",
}

# Map: stem of result file (before "_gemini_classified_full") -> ground truth file
DATASET_MAP = {
    "twitter":              "twitter_full.jsonl",
    "imdb":                 "imdb_full.jsonl",
    "finance_50agree":      "finance_50agree_full.jsonl",
    "finance_66agree":      "finance_66agree_full.jsonl",
    "finance_75agree":      "finance_75agree_full.jsonl",
    "finance_all_agree":    "finance_all_agree_full.jsonl",
    "yelp_full":            "yelp_full_full.jsonl",
    "tweet_eval_sentiment": "tweet_eval_sentiment_full.jsonl",
    "tweet_eval_irony":     "tweet_eval_irony_full.jsonl",
}

def resolve_results_dir(model: str, custom_dir: str | None) -> Path:
    if custom_dir:
        return Path(custom_dir)

    # Backward compatibility: old Gemini runs were saved in src/results.
    if model == "gemini":
        gemini_dir = MODEL_DIR_DEFAULTS["gemini"]
        legacy_dir = Path(__file__).parent / "results"
        return gemini_dir if gemini_dir.exists() else legacy_dir

    return MODEL_DIR_DEFAULTS[model]


def get_result_suffix(model: str) -> str:
    return f"_{model}_classified_full.jsonl"


def load_ground_truth(path: Path) -> dict[int, int]:
    """Returns {index: label} from a *_full.jsonl file (0-based row order)."""
    labels = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            labels[i] = int(record["label"])
    return labels


def load_predictions(path: Path) -> tuple[dict[int, int], int]:
    """Returns ({original_index: sentiment}, invalid_rows_skipped)."""
    preds = {}
    invalid_rows = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            try:
                idx = int(record["original_index"])
                sentiment = int(record["sentiment"])
            except (KeyError, TypeError, ValueError):
                # Skip malformed rows like sentiment="".
                invalid_rows += 1
                continue
            preds[idx] = sentiment
    return preds, invalid_rows


def evaluate_dataset(dataset_key: str, results_dir: Path, result_suffix: str, f1_avg: str = "both") -> dict:
    gt_file = SUBSETS_DIR / DATASET_MAP[dataset_key]
    result_file = results_dir / f"{dataset_key}{result_suffix}"

    if not gt_file.exists():
        return {"error": f"Ground truth not found: {gt_file}"}
    if not result_file.exists():
        return {"error": f"Results file not found: {result_file}"}

    labels = load_ground_truth(gt_file)
    preds, invalid_rows = load_predictions(result_file)

    # Align: only evaluate records that appear in both
    blocked = sum(1 for v in preds.values() if v == 404)
    common_indices = sorted(set(labels.keys()) & {k for k, v in preds.items() if v != 404})

    y_true = [labels[i] for i in common_indices]
    y_pred = [preds[i] for i in common_indices]

    missing = len(labels) - len(common_indices) - blocked

    if not y_true:
        return {
            "dataset": dataset_key,
            "total_gt": len(labels),
            "evaluated": 0,
            "blocked_404": blocked,
            "missing_predictions": missing,
            "invalid_rows": invalid_rows,
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
        }

    acc = accuracy_score(y_true, y_pred)
    unique_labels = sorted(set(y_true))

    result = {
        "dataset": dataset_key,
        "total_gt": len(labels),
        "evaluated": len(common_indices),
        "blocked_404": blocked,
        "missing_predictions": missing,
        "invalid_rows": invalid_rows,
        "accuracy": acc,
    }

    if f1_avg in ("macro", "both"):
        result["f1_macro"] = f1_score(y_true, y_pred, average="macro", labels=unique_labels)
    if f1_avg in ("weighted", "both"):
        result["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", labels=unique_labels)

    return result


def print_results(results: list[dict], f1_avg: str):
    has_macro    = f1_avg in ("macro", "both")
    has_weighted = f1_avg in ("weighted", "both")

    # Header
    col_dataset = 24
    header = f"{'Dataset':<{col_dataset}}  {'Total':>6}  {'Eval':>6}  {'Blk':>4}  {'Inv':>4}  {'Acc':>7}"
    if has_macro:
        header += f"  {'F1-macro':>9}"
    if has_weighted:
        header += f"  {'F1-wgtd':>9}"
    print(header)
    print("-" * len(header))

    for r in results:
        if "error" in r:
            print(f"{r.get('dataset', '?'):<{col_dataset}}  ERROR: {r['error']}")
            continue
        row = (
            f"{r['dataset']:<{col_dataset}}"
            f"  {r['total_gt']:>6}"
            f"  {r['evaluated']:>6}"
            f"  {r['blocked_404']:>4}"
            f"  {r.get('invalid_rows', 0):>4}"
            f"  {r['accuracy']:>7.4f}"
        )
        if has_macro:
            row += f"  {r['f1_macro']:>9.4f}"
        if has_weighted:
            row += f"  {r['f1_weighted']:>9.4f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment classification results.")
    parser.add_argument(
        "datasets",
        nargs="*",
        default=list(DATASET_MAP.keys()),
        help=f"Datasets to evaluate (default: all). Choices: {', '.join(DATASET_MAP.keys())}",
    )
    parser.add_argument(
        "--f1",
        dest="f1_avg",
        choices=["macro", "weighted", "both"],
        default="both",
        help="F1 averaging strategy (default: both)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print full sklearn classification_report for each dataset",
    )
    parser.add_argument(
        "--model",
        choices=["gemini", "llama", "mistral", "claude"],
        default="gemini",
        help="Model output format to evaluate (default: gemini)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Path to folder with result files. "
            "If omitted, defaults to model-specific folder (e.g. src/results_llama)."
        ),
    )
    args = parser.parse_args()

    invalid = [d for d in args.datasets if d not in DATASET_MAP]
    if invalid:
        parser.error(f"Unknown dataset(s): {invalid}. Choose from: {', '.join(DATASET_MAP.keys())}")

    results_dir = resolve_results_dir(args.model, args.results_dir)
    result_suffix = get_result_suffix(args.model)

    print(f"Model: {args.model}")
    print(f"Results dir: {results_dir}")

    results = []
    for key in args.datasets:
        r = evaluate_dataset(key, results_dir, result_suffix, args.f1_avg)
        results.append(r)

    print_results(results, args.f1_avg)

    if args.report:
        for key in args.datasets:
            gt_file = SUBSETS_DIR / DATASET_MAP[key]
            result_file = results_dir / f"{key}{result_suffix}"
            if not gt_file.exists() or not result_file.exists():
                continue
            labels = load_ground_truth(gt_file)
            preds, _ = load_predictions(result_file)
            common = sorted(set(labels.keys()) & {k for k, v in preds.items() if v != 404})
            y_true = [labels[i] for i in common]
            y_pred = [preds[i] for i in common]
            print(f"\n--- {key} ---")
            print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
