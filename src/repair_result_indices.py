"""
Repair wrong or duplicated original_index values in model result files.

Strategy:
- Results are written in the same order as the input ground-truth file.
- Failed examples are written with sentiment=404 rather than being skipped.
- Therefore original_index for each row equals its 0-based line number in the file.

Examples:
    python repair_result_indices.py --model llama finance_50agree
    python repair_result_indices.py --model llama --in-place finance_50agree
    python repair_result_indices.py --model llama --in-place
    python repair_result_indices.py --results-dir src/results_llama --suffix _llama_classified_full.jsonl finance_50agree
"""

import argparse
import json
from pathlib import Path

MODEL_DIR_DEFAULTS = {
    "gemini": Path(__file__).parent / "results_gemini",
    "llama": Path(__file__).parent / "results_llama",
    "mistral": Path(__file__).parent / "results_mistral",
    "claude": Path(__file__).parent / "results_claude",
}

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


def resolve_results_dir(model: str, custom_dir: str | None) -> Path:
    if custom_dir:
        return Path(custom_dir)

    if model == "gemini":
        model_dir = MODEL_DIR_DEFAULTS["gemini"]
        legacy_dir = Path(__file__).parent / "results"
        return model_dir if model_dir.exists() else legacy_dir

    return MODEL_DIR_DEFAULTS[model]


def get_result_suffix(model: str, custom_suffix: str | None) -> str:
    if custom_suffix:
        return custom_suffix
    return f"_{model}_classified_full.jsonl"


def repair_dataset(
    dataset_key: str,
    results_dir: Path,
    suffix: str,
    in_place: bool,
) -> dict:
    pred_file = results_dir / f"{dataset_key}{suffix}"

    if not pred_file.exists():
        return {"dataset": dataset_key, "error": f"Prediction file missing: {pred_file}"}

    rows = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    fixed_count = 0
    already_correct_count = 0

    for idx, row in enumerate(rows):
        old = row.get("original_index")
        if old == idx:
            already_correct_count += 1
        else:
            row["original_index"] = idx
            fixed_count += 1

    if in_place:
        output_file = pred_file
    else:
        output_file = pred_file.with_name(pred_file.stem + "_reindexed.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "dataset": dataset_key,
        "input_file": str(pred_file),
        "output_file": str(output_file),
        "rows": len(rows),
        "already_correct": already_correct_count,
        "fixed": fixed_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair original_index values by assigning each row its 0-based line number."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=list(DATASET_MAP.keys()),
        help=f"Datasets to process (default: all). Choices: {', '.join(DATASET_MAP.keys())}",
    )
    parser.add_argument(
        "--model",
        choices=["gemini", "llama", "mistral", "claude"],
        default="llama",
        help="Model folder/suffix preset (default: llama)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Custom path to prediction files directory.",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Custom result suffix, e.g. _llama_classified_full.jsonl",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original files. Default writes *_reindexed.jsonl.",
    )

    args = parser.parse_args()

    invalid = [d for d in args.datasets if d not in DATASET_MAP]
    if invalid:
        parser.error(f"Unknown dataset(s): {invalid}")

    results_dir = resolve_results_dir(args.model, args.results_dir)
    suffix = get_result_suffix(args.model, args.suffix)

    print(f"Results dir: {results_dir}")
    print(f"Suffix: {suffix}")
    print(f"Mode: {'in-place overwrite' if args.in_place else 'write *_reindexed.jsonl'}")

    for dataset in args.datasets:
        report = repair_dataset(dataset, results_dir, suffix, args.in_place)
        print("\n" + "-" * 90)
        print(f"Dataset: {dataset}")
        if "error" in report:
            print(f"Error: {report['error']}")
            continue

        print(f"Input:            {report['input_file']}")
        print(f"Output:           {report['output_file']}")
        print(f"Rows:             {report['rows']}")
        print(f"Already correct:  {report['already_correct']}")
        print(f"Fixed:            {report['fixed']}")


if __name__ == "__main__":
    main()
