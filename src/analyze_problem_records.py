"""
Analyze problematic records in sentiment result files.

This script compares *_full.jsonl ground truth files with
*_gemini_classified_full.jsonl prediction files and reports:
- missing predictions
- blocked predictions (sentiment == 404)
- duplicate original_index values
- out-of-range indices
- invalid original_index rows
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path


SUBSETS_DIR = Path(__file__).parent / "subsets"
RESULTS_DIR = Path(__file__).parent / "results_mistral" # change it to analyze other model results

DATASET_MAP = {
    "twitter": "twitter_full.jsonl",
    "imdb": "imdb_full.jsonl",
    "finance_50agree": "finance_50agree_full.jsonl",
    "finance_66agree": "finance_66agree_full.jsonl",
    "finance_75agree": "finance_75agree_full.jsonl",
    "finance_all_agree": "finance_all_agree_full.jsonl",
    "yelp_full": "yelp_full_full.jsonl",
    "tweet_eval_sentiment": "tweet_eval_sentiment_full.jsonl",
    "tweet_eval_irony": "tweet_eval_irony_full.jsonl",
}

RESULT_SUFFIX = "_mistral_classified_full.jsonl" # change it to analyze other model results (e.g., "_gemini_classified_full.jsonl")


def load_ground_truth(path: Path) -> dict[int, dict]:
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            gt[i] = rec
    return gt


def load_predictions_raw(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            rec = json.loads(line)
            rows.append(
                {
                    "line_no": line_no,
                    "original_index": rec.get("original_index"),
                    "sentiment": rec.get("sentiment"),
                    "text": rec.get("text", ""),
                }
            )
    return rows


def analyze_dataset(dataset_key: str, max_examples: int = 20) -> dict:
    gt_file = SUBSETS_DIR / DATASET_MAP[dataset_key]
    pred_file = RESULTS_DIR / f"{dataset_key}{RESULT_SUFFIX}"

    if not gt_file.exists():
        return {"dataset": dataset_key, "error": f"Missing GT file: {gt_file}"}
    if not pred_file.exists():
        return {"dataset": dataset_key, "error": f"Missing prediction file: {pred_file}"}

    gt = load_ground_truth(gt_file)
    pred_rows = load_predictions_raw(pred_file)

    pred_by_index = defaultdict(list)
    invalid_index_rows = []
    for row in pred_rows:
        idx = row["original_index"]
        if not isinstance(idx, int):
            invalid_index_rows.append(row)
            continue
        pred_by_index[idx].append(row)

    gt_indices = set(gt.keys())
    pred_indices = set(pred_by_index.keys())

    out_of_range = sorted(i for i in pred_indices if i not in gt_indices)
    missing_indices = sorted(i for i in gt_indices if i not in pred_indices)

    blocked_404_indices = []
    evaluated_indices = []
    duplicate_indices = []

    for idx in sorted(pred_indices & gt_indices):
        rows = pred_by_index[idx]
        if len(rows) > 1:
            duplicate_indices.append(idx)

        # Keep behavior consistent with evaluate.py dictionary overwrite semantics
        final_sentiment = rows[-1]["sentiment"]
        if final_sentiment == 404:
            blocked_404_indices.append(idx)
        else:
            evaluated_indices.append(idx)

    missing_eval_indices = sorted(
        gt_indices - set(evaluated_indices) - set(blocked_404_indices)
    )

    def gt_preview(indices: list[int]) -> list[dict]:
        out = []
        for i in indices[:max_examples]:
            out.append(
                {
                    "index": i,
                    "label": gt[i].get("label"),
                    "text": gt[i].get("text", "")[:200],
                }
            )
        return out

    duplicate_preview = []
    for i in duplicate_indices[:max_examples]:
        duplicate_preview.append(
            {
                "index": i,
                "count": len(pred_by_index[i]),
                "line_numbers": [x["line_no"] for x in pred_by_index[i]],
                "sentiments": [x["sentiment"] for x in pred_by_index[i]],
            }
        )

    blocked_preview = []
    for i in blocked_404_indices[:max_examples]:
        blocked_preview.append(
            {
                "index": i,
                "gt_label": gt[i].get("label"),
                "text": gt[i].get("text", "")[:200],
            }
        )

    return {
        "dataset": dataset_key,
        "total_gt": len(gt_indices),
        "prediction_rows": len(pred_rows),
        "unique_pred_indices": len(pred_indices),
        "evaluated_count": len(evaluated_indices),
        "blocked_404_count": len(blocked_404_indices),
        "missing_pred_count": len(missing_indices),
        "missing_eval_count": len(missing_eval_indices),
        "duplicate_index_count": len(duplicate_indices),
        "out_of_range_count": len(out_of_range),
        "invalid_index_rows_count": len(invalid_index_rows),
        "missing_pred_examples": gt_preview(missing_indices),
        "missing_eval_examples": gt_preview(missing_eval_indices),
        "blocked_404_examples": blocked_preview,
        "duplicate_examples": duplicate_preview,
        "out_of_range_examples": out_of_range[:max_examples],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze problematic records in sentiment result files."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=list(DATASET_MAP.keys()),
        help=f"Datasets (default: all): {', '.join(DATASET_MAP.keys())}",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="How many example records per category to print (default: 20)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save full JSON report",
    )
    args = parser.parse_args()

    invalid = [d for d in args.datasets if d not in DATASET_MAP]
    if invalid:
        raise SystemExit(f"Unknown dataset(s): {invalid}")

    reports = []
    for dataset in args.datasets:
        report = analyze_dataset(dataset, max_examples=args.max_examples)
        reports.append(report)

        print("\n" + "=" * 90)
        print(f"DATASET: {dataset}")
        if "error" in report:
            print(f"ERROR: {report['error']}")
            continue

        print(
            f"Total GT={report['total_gt']} | Pred rows={report['prediction_rows']} | "
            f"Unique idx={report['unique_pred_indices']} | Eval={report['evaluated_count']} | "
            f"404={report['blocked_404_count']} | MissingPred={report['missing_pred_count']} | "
            f"DupIdx={report['duplicate_index_count']} | OutOfRange={report['out_of_range_count']} | "
            f"InvalidIdxRows={report['invalid_index_rows_count']}"
        )

        if report["missing_pred_examples"]:
            print("\nMissing prediction examples:")
            for x in report["missing_pred_examples"]:
                print(f"- idx={x['index']} label={x['label']} text={x['text']}")

        if report["blocked_404_examples"]:
            print("\nBlocked 404 examples:")
            for x in report["blocked_404_examples"]:
                print(f"- idx={x['index']} gt_label={x['gt_label']} text={x['text']}")

        if report["duplicate_examples"]:
            print("\nDuplicate index examples:")
            for x in report["duplicate_examples"]:
                print(
                    f"- idx={x['index']} count={x['count']} "
                    f"lines={x['line_numbers']} sentiments={x['sentiments']}"
                )

        if report["out_of_range_examples"]:
            print("\nOut-of-range indices:")
            print(report["out_of_range_examples"])

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON report to: {out_path}")


if __name__ == "__main__":
    main()
