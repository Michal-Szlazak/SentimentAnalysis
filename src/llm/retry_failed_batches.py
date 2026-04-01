"""
Retry only missing classifications based on original_index.

Use case:
- previous full run had failed batches (e.g. 429, JSON parse error)
- output file contains only successful records
- this script classifies only missing samples and merges them back
"""

import argparse
import json
from pathlib import Path

from gemini_prompts import get_dataset_name
from gemini_sentiment_classifier import classify_batch, load_dataset
from run_classifier import DATASETS, PROMPT_KEYS, RESULTS_DIR, SUBSETS_DIR


DEFAULT_DATASETS = ["imdb_full", "yelp_full"]
BATCH_SIZE = 1
WORKERS = 4


def get_input_file(dataset_key: str) -> str:
    return str(Path(SUBSETS_DIR) / DATASETS[dataset_key])


def get_output_file(dataset_key: str) -> str:
    base_name = DATASETS[dataset_key].replace("_test_stratified.jsonl", "_gemini_classified.jsonl")
    base_name = base_name.replace("_full.jsonl", "_gemini_classified_full.jsonl")
    return str(Path(RESULTS_DIR) / base_name)


def load_existing_results(output_file: str) -> dict[int, dict]:
    existing = {}
    output_path = Path(output_file)
    if not output_path.exists():
        return existing

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            idx = row.get("original_index")
            if isinstance(idx, int):
                existing[idx] = row

    return existing


def write_merged_results(output_file: str, merged: dict[int, dict]) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx in sorted(merged.keys()):
            f.write(json.dumps(merged[idx], ensure_ascii=False) + "\n")


def chunked(items: list[int], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def create_404_results(data: list[dict], idx_batch: list[int]) -> dict[int, dict]:
    return {
        idx: {
            "text": data[idx].get("text", ""),
            "sentiment": 404,
            "original_index": idx,
        }
        for idx in idx_batch
    }


def retry_dataset(dataset_key: str) -> None:
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    prompt_key = PROMPT_KEYS[dataset_key]
    input_file = get_input_file(dataset_key)
    output_file = get_output_file(dataset_key)

    print("=" * 80)
    print(f"Dataset: {dataset_key} ({get_dataset_name(prompt_key)})")
    print(f"Input:   {input_file}")
    print(f"Output:  {output_file}")
    print(f"Config:  batch_size={BATCH_SIZE}, workers={WORKERS}")

    data = load_dataset(input_file)
    total = len(data)
    existing = load_existing_results(output_file)

    missing_indices = [i for i in range(total) if i not in existing]

    print(f"Total samples:     {total}")
    print(f"Already classified:{len(existing)}")
    print(f"Missing samples:   {len(missing_indices)}")

    if not missing_indices:
        print("Nothing to retry.\n")
        return

    try:
        from gemini_prompts import get_instruction

        instruction = get_instruction(prompt_key)
    except Exception:
        instruction = None

    recovered = {}
    missing_batches = list(chunked(missing_indices, BATCH_SIZE))
    total_batches = len(missing_batches)

    for batch_no, idx_batch in enumerate(missing_batches, 1):
        batch = [data[i] for i in idx_batch]
        try:
            classifications = classify_batch(
                batch=batch,
                start_index=idx_batch[0],
                instruction=instruction,
                indices=idx_batch,
            )

            if len(classifications) != len(batch):
                raise ValueError(
                    f"Expected {len(batch)} classifications, got {len(classifications)}"
                )

            for item in classifications:
                original_index = item.get("index")
                if isinstance(original_index, int):
                    recovered[original_index] = {
                        "text": item.get("text", ""),
                        "sentiment": item.get("sentiment", ""),
                        "original_index": original_index,
                    }

            print(f"✅ Retry batch {batch_no}/{total_batches} done ({len(recovered)} recovered)")
        except Exception as e:
            print(f"❌ Retry batch {batch_no}/{total_batches} error: {e}")
            recovered.update(create_404_results(data, idx_batch))
            print(f"⚠️ Retry batch {batch_no}/{total_batches} saved with sentiment=404")

    merged = dict(existing)
    merged.update(recovered)
    write_merged_results(output_file, merged)

    still_missing = total - len(merged)
    print(f"Recovered now:     {len(recovered)}")
    print(f"Total after merge: {len(merged)}/{total}")
    if still_missing > 0:
        print(f"Still missing:     {still_missing}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retry only missing sentiment classifications (by original_index)."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Dataset keys to retry (default: imdb_full yelp_full)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for dataset_key in args.datasets:
        retry_dataset(dataset_key)


if __name__ == "__main__":
    main()
