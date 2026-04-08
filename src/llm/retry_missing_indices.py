"""
Retry only missing prediction indices (indices present in ground truth but absent in result file).

This is useful when result files have fewer rows than ground truth, e.g. missing 2 records.

Usage:
    cd src/llm
    python retry_missing_indices.py --model gemini twitter
    python retry_missing_indices.py --model gemini twitter imdb --batch-size 10
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).parent.parent
SUBSETS_DIR = BASE_DIR / "subsets"

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

PROMPT_KEYS = {
    "twitter": "twitter",
    "imdb": "imdb",
    "finance_50agree": "finance_50",
    "finance_66agree": "finance_66",
    "finance_75agree": "finance_75",
    "finance_all_agree": "finance_all",
    "yelp_full": "yelp",
    "tweet_eval_sentiment": "tweet_sentiment",
    "tweet_eval_irony": "tweet_irony",
}

MODEL_RESULTS_DIRS = {
    "gemini": BASE_DIR / "results_gemini",
    "mistral": BASE_DIR / "results_mistral",
    "llama": (BASE_DIR / "results_llama_reindexed")
    if (BASE_DIR / "results_llama_reindexed").exists()
    else (BASE_DIR / "results_llama"),
}


def parse_int(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None
        try:
            return int(v)
        except ValueError:
            return None
    return None


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def get_classify_batch_fn(model: str):
    if model == "gemini":
        from gemini_sentiment_classifier import classify_batch

        return classify_batch
    if model == "mistral":
        from mistral_sentiment_classifier import classify_batch

        return classify_batch
    if model == "llama":
        from llama_sentiment_classifier import classify_batch

        return classify_batch
    raise ValueError(f"Unsupported model: {model}")


def map_classifications_to_sentiments(
    classifications: list[dict],
    indices: list[int],
) -> dict[int, int]:
    """Map model response to {original_index: sentiment}."""
    index_set = set(indices)
    sentiment_by_index: dict[int, int] = {}
    ordered_sentiments: list[int] = []

    for item in classifications:
        if not isinstance(item, dict):
            continue
        s = parse_int(item.get("sentiment"))
        raw_idx = parse_int(item.get("index"))

        if s is not None:
            ordered_sentiments.append(s)

        if s is None or raw_idx is None:
            continue

        if raw_idx in index_set:
            sentiment_by_index[raw_idx] = s
        elif 0 <= raw_idx < len(indices):
            sentiment_by_index[indices[raw_idx]] = s

    for offset, orig_idx in enumerate(indices):
        if orig_idx not in sentiment_by_index and offset < len(ordered_sentiments):
            sentiment_by_index[orig_idx] = ordered_sentiments[offset]

    return sentiment_by_index


def get_result_file(dataset: str, model: str, custom_results_dir: str | None) -> Path:
    results_dir = Path(custom_results_dir) if custom_results_dir else MODEL_RESULTS_DIRS[model]
    return results_dir / f"{dataset}_{model}_classified_full.jsonl"


def retry_missing_for_dataset(
    dataset: str,
    model: str,
    batch_size: int,
    max_retries: int,
    retry_base_delay: float,
    custom_results_dir: str | None,
) -> dict:
    gt_file = SUBSETS_DIR / DATASET_MAP[dataset]
    result_file = get_result_file(dataset, model, custom_results_dir)

    if not gt_file.exists():
        return {"dataset": dataset, "error": f"Missing ground truth file: {gt_file}"}
    if not result_file.exists():
        return {"dataset": dataset, "error": f"Missing result file: {result_file}"}

    gt_rows = load_jsonl(gt_file)
    pred_rows = load_jsonl(result_file)

    present_indices = {
        idx
        for idx in (parse_int(r.get("original_index")) for r in pred_rows)
        if idx is not None
    }

    missing_indices = [i for i in range(len(gt_rows)) if i not in present_indices]
    if not missing_indices:
        return {
            "dataset": dataset,
            "result_file": str(result_file),
            "missing_count": 0,
            "filled_count": 0,
            "still_missing": 0,
        }

    classify_batch = get_classify_batch_fn(model)
    from gemini_prompts import get_instruction

    instruction = get_instruction(PROMPT_KEYS[dataset])

    filled_count = 0
    still_missing = 0
    additions: list[dict] = []

    batches = [
        missing_indices[i : i + batch_size]
        for i in range(0, len(missing_indices), batch_size)
    ]

    for b_idx, batch_indices in enumerate(batches, start=1):
        batch_items = [{"text": gt_rows[i].get("text", "")} for i in batch_indices]
        print(f"  Batch {b_idx}/{len(batches)}: missing idx {batch_indices[0]}..{batch_indices[-1]}")

        last_error = None
        success = False
        sentiment_map: dict[int, int] = {}

        for attempt in range(1, max_retries + 1):
            try:
                classifications = classify_batch(batch_items, 0, instruction, batch_indices)
                sentiment_map = map_classifications_to_sentiments(classifications, batch_indices)
                success = True
                break
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    delay = retry_base_delay * attempt
                    print(f"    Retry {attempt}/{max_retries - 1} in {delay:.1f}s: {exc}")
                    time.sleep(delay)

        if not success:
            print(f"    Batch failed after retries: {last_error}")
            for idx in batch_indices:
                additions.append(
                    {
                        "text": gt_rows[idx].get("text", ""),
                        "sentiment": 404,
                        "original_index": idx,
                    }
                )
                still_missing += 1
            continue

        for idx in batch_indices:
            s = sentiment_map.get(idx)
            if isinstance(s, int):
                additions.append(
                    {
                        "text": gt_rows[idx].get("text", ""),
                        "sentiment": s,
                        "original_index": idx,
                    }
                )
                filled_count += 1
            else:
                additions.append(
                    {
                        "text": gt_rows[idx].get("text", ""),
                        "sentiment": 404,
                        "original_index": idx,
                    }
                )
                still_missing += 1

    with open(result_file, "a", encoding="utf-8") as f:
        for row in additions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "dataset": dataset,
        "result_file": str(result_file),
        "missing_count": len(missing_indices),
        "filled_count": filled_count,
        "still_missing": still_missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retry only missing original_index records in model result files."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=list(DATASET_MAP.keys()),
        help=f"Datasets to process (default: all): {', '.join(DATASET_MAP.keys())}",
    )
    parser.add_argument(
        "--model",
        choices=["gemini", "mistral", "llama"],
        default="gemini",
        help="Model to use for retry (default: gemini)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for retry requests (default: 10)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry attempts per batch (default: 3)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Optional custom results directory for the selected model",
    )
    args = parser.parse_args()

    invalid = [d for d in args.datasets if d not in DATASET_MAP]
    if invalid:
        parser.error(f"Unknown dataset(s): {invalid}")

    print(f"Model: {args.model}")
    if args.results_dir:
        print(f"Results dir override: {args.results_dir}")

    for dataset in args.datasets:
        print("\n" + "-" * 90)
        print(f"Dataset: {dataset}")
        report = retry_missing_for_dataset(
            dataset=dataset,
            model=args.model,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            retry_base_delay=2.0,
            custom_results_dir=args.results_dir,
        )

        if "error" in report:
            print(f"Error: {report['error']}")
            continue

        print(f"Result file:     {report['result_file']}")
        print(f"Missing found:   {report['missing_count']}")
        print(f"Filled:          {report['filled_count']}")
        print(f"Fallback 404:    {report['still_missing']}")


if __name__ == "__main__":
    main()
