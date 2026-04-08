"""
Retry classification only for items with sentiment=404 in existing result files.

Reads the result file, finds failed rows (sentiment=404), re-sends them to the model,
and writes the patched result back in-place.

Usage:
    cd src/llm

    # Retry all datasets for a given model
    python retry_404.py --model llama
    python retry_404.py --model mistral

    # Retry specific datasets
    python retry_404.py --model llama twitter_full imdb_full

    # Override batch size
    python retry_404.py --model mistral --batch-size 5 imdb_full
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from run_classifier import DATASETS, PROMPT_KEYS, SUBSETS_DIR

DATASETS_TO_PROCESS = [
    "imdb_full",
    "twitter_full",
    "finance_50_full",
    "finance_66_full",
    "finance_75_full",
    "finance_all_full",
    "yelp_full",
    "tweet_sentiment_full",
    "tweet_irony_full",
]

DEFAULT_BATCH_SIZES = {
    "imdb_full": 10,
    "twitter_full": 30,
    "finance_50_full": 30,
    "finance_66_full": 30,
    "finance_75_full": 30,
    "finance_all_full": 30,
    "yelp_full": 10,
    "tweet_sentiment_full": 30,
    "tweet_irony_full": 30,
}

MODEL_RESULTS_DIRS = {
    "gemini": Path(SUBSETS_DIR).parent / "results_gemini",
    "llama": Path(SUBSETS_DIR).parent / "results_llama_reindexed",
    "mistral": Path(SUBSETS_DIR).parent / "results_mistral",
    "claude": Path(SUBSETS_DIR).parent / "results_claude",
}


def get_classify_batch_fn(model: str):
    if model == "gemini":
        from gemini_sentiment_classifier import classify_batch
        return classify_batch
    if model == "llama":
        from llama_sentiment_classifier import classify_batch
        return classify_batch
    if model == "mistral":
        from mistral_sentiment_classifier import classify_batch
        return classify_batch
    if model == "claude":
        from claude_sentiment_classifier import classify_batch
        return classify_batch
    raise ValueError(f"Unknown model: {model}")


def get_result_file(dataset_key: str, model: str) -> Path:
    results_dir = MODEL_RESULTS_DIRS[model]
    base = DATASETS[dataset_key]
    name = base.replace("_test_stratified.jsonl", f"_{model}_classified.jsonl")
    name = name.replace("_full.jsonl", f"_{model}_classified_full.jsonl")
    return results_dir / name


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def to_int(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        v = value.strip()
        if v == "":
            return None
        try:
            return int(v)
        except ValueError:
            return None
    return None


def map_classifications_to_sentiments(
    classifications: list[dict],
    indices: list[int],
) -> dict[int, int]:
    """Map model response to {original_index: sentiment}.

    Tries index-based first, falls back to positional order.
    """
    index_set = set(indices)
    sentiment_by_index: dict[int, int] = {}
    ordered_sentiments: list[int] = []

    for item in classifications:
        if not isinstance(item, dict):
            continue
        s = to_int(item.get("sentiment"))
        raw_idx = to_int(item.get("index"))

        if s is not None:
            ordered_sentiments.append(s)

        if s is None or raw_idx is None:
            continue

        if raw_idx in index_set:
            sentiment_by_index[raw_idx] = s
        elif 0 <= raw_idx < len(indices):
            sentiment_by_index[indices[raw_idx]] = s

    # Positional fallback for indices still missing.
    for offset, orig_idx in enumerate(indices):
        if orig_idx not in sentiment_by_index and offset < len(ordered_sentiments):
            sentiment_by_index[orig_idx] = ordered_sentiments[offset]

    return sentiment_by_index


def _is_prohibited_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ("prohibited", "blocked", "safety", "content_filter", "content filter"))


def _classify_single(
    classify_batch_fn,
    pos: int,
    item: dict,
    orig_idx: int,
    instruction,
    max_retries: int,
    retry_base_delay: float,
) -> tuple[int, int | None]:
    """Try classifying a single item. Returns (pos, sentiment) or (pos, None) on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            classifications = classify_batch_fn([item], 0, instruction, [orig_idx])
            sentiment_map = map_classifications_to_sentiments(classifications, [orig_idx])
            s = sentiment_map.get(orig_idx)
            if s is not None and s != 404:
                return pos, s
        except Exception as e:
            if _is_prohibited_error(e):
                print(f"    Index {orig_idx}: prohibited content, skipping")
                return pos, None
            if attempt < max_retries:
                time.sleep(retry_base_delay * attempt)
    return pos, None


def retry_dataset(
    dataset_key: str,
    model: str,
    batch_size: int,
    max_workers: int = 1,
    max_retries: int = 3,
    retry_base_delay: float = 2.0,
) -> dict:
    result_file = get_result_file(dataset_key, model)

    if not result_file.exists():
        return {"dataset": dataset_key, "error": f"Result file not found: {result_file}"}

    rows = load_jsonl(result_file)

    # Collect (file_position, row) for all 404 items.
    failed_items: list[tuple[int, dict]] = [
        (pos, row)
        for pos, row in enumerate(rows)
        if to_int(row.get("sentiment")) == 404
    ]

    if not failed_items:
        return {
            "dataset": dataset_key,
            "total_404": 0,
            "fixed": 0,
            "still_failed": 0,
            "output_file": str(result_file),
        }

    print(f"  Found {len(failed_items)} items with sentiment=404")

    instruction = None
    try:
        from gemini_prompts import get_instruction
        instruction = get_instruction(PROMPT_KEYS[dataset_key])
    except Exception:
        pass

    classify_batch_fn = get_classify_batch_fn(model)

    batches = [
        failed_items[i : i + batch_size]
        for i in range(0, len(failed_items), batch_size)
    ]
    total_batches = len(batches)

    fixed_count = 0
    still_failed = 0

    def process_batch(batch_num: int, batch: list[tuple[int, dict]]):
        positions = [pos for pos, _ in batch]
        items = [row for _, row in batch]
        indices = [to_int(row.get("original_index")) or 0 for row in items]

        print(f"  Batch {batch_num}/{total_batches}: indices {indices[0]}..{indices[-1]}")

        last_error = None
        success = False

        for attempt in range(1, max_retries + 1):
            try:
                classifications = classify_batch_fn(items, 0, instruction, indices)
                sentiment_map = map_classifications_to_sentiments(classifications, indices)

                success = True
                return [
                    (pos, sentiment_map.get(idx))
                    for idx, pos in zip(indices, positions)
                ]

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = retry_base_delay * attempt
                    print(f"    Attempt {attempt} failed: {e}. Retry in {delay:.1f}s...")
                    time.sleep(delay)

        if not success:
            print(f"  Batch {batch_num} failed after {max_retries} attempts: {last_error}")
            print(f"  Falling back to per-item classification for {len(batch)} items...")
            updates = []
            for (pos, row), idx in zip(batch, indices):
                result_pos, s = _classify_single(
                    classify_batch_fn, pos, row, idx, instruction, max_retries, retry_base_delay
                )
                updates.append((result_pos, s))
            return updates

        # Should never happen, but keep a safe default.
        return [(pos, None) for pos in positions]

    if max_workers <= 1:
        for batch_num, batch in enumerate(batches, 1):
            updates = process_batch(batch_num, batch)
            for pos, sentiment in updates:
                if sentiment is not None and sentiment != 404:
                    rows[pos]["sentiment"] = sentiment
                    fixed_count += 1
                else:
                    still_failed += 1
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_batch, batch_num, batch): batch_num
                for batch_num, batch in enumerate(batches, 1)
            }
            for future in as_completed(futures):
                batch_num = futures[future]
                try:
                    updates = future.result()
                except Exception as exc:
                    print(f"  Batch {batch_num}/{total_batches} worker error: {exc}")
                    # In case of worker failure, mark the whole batch as still failed.
                    batch = batches[batch_num - 1]
                    still_failed += len(batch)
                    continue

                for pos, sentiment in updates:
                    if sentiment is not None and sentiment != 404:
                        rows[pos]["sentiment"] = sentiment
                        fixed_count += 1
                    else:
                        still_failed += 1

    with open(result_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "dataset": dataset_key,
        "total_404": len(failed_items),
        "fixed": fixed_count,
        "still_failed": still_failed,
        "output_file": str(result_file),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Retry classification for items with sentiment=404."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=DATASETS_TO_PROCESS,
        help=f"Datasets to retry (default: all). Choices: {', '.join(DATASETS_TO_PROCESS)}",
    )
    parser.add_argument(
        "--model",
        choices=["gemini", "llama", "mistral", "claude"],
        default="llama",
        help="Model to use for retry (default: llama)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Texts per API request (default: per-dataset map)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel batch retries per dataset (default: 1)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retry attempts per batch (default: 3)",
    )

    args = parser.parse_args()

    invalid = [d for d in args.datasets if d not in DATASETS_TO_PROCESS]
    if invalid:
        parser.error(f"Unknown dataset(s): {invalid}. Allowed: {', '.join(DATASETS_TO_PROCESS)}")

    print("=" * 80)
    print(f"Retrying 404 items | model={args.model}")
    print(f"Workers: {args.workers}")
    print("=" * 80)

    total_fixed = 0
    total_still_failed = 0

    for dataset_key in args.datasets:
        batch_size = args.batch_size or DEFAULT_BATCH_SIZES[dataset_key]
        print(f"\nDataset: {dataset_key} | batch_size={batch_size}")

        report = retry_dataset(
            dataset_key=dataset_key,
            model=args.model,
            batch_size=batch_size,
            max_workers=args.workers,
            max_retries=args.max_retries,
        )

        if "error" in report:
            print(f"  Error: {report['error']}")
            continue

        if report["total_404"] == 0:
            print("  No 404 items — skipping")
            continue

        total_fixed += report["fixed"]
        total_still_failed += report["still_failed"]

        print(f"  Total 404:    {report['total_404']}")
        print(f"  Fixed:        {report['fixed']}")
        print(f"  Still failed: {report['still_failed']}")
        print(f"  Output:       {report['output_file']}")

    print("\n" + "=" * 80)
    print(f"Summary: fixed={total_fixed}, still_failed={total_still_failed}")
    print("=" * 80)


if __name__ == "__main__":
    main()
