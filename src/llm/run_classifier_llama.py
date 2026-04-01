"""
Runner script for Llama sentiment classification on Vertex AI.
"""

import argparse
import sys
from pathlib import Path

from gemini_prompts import get_dataset_name
from llama_sentiment_classifier import classify_dataset
from run_classifier import DATASETS, PROMPT_KEYS, RESULTS_DIR, SUBSETS_DIR


def get_input_file(dataset_key: str) -> str:
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Available: {', '.join(DATASETS.keys())}")
    return str(Path(SUBSETS_DIR) / DATASETS[dataset_key])


def get_output_file(dataset_key: str) -> str:
    base_name = DATASETS[dataset_key].replace("_test_stratified.jsonl", "_llama_classified.jsonl")
    base_name = base_name.replace("_full.jsonl", "_llama_classified_full.jsonl")
    return str(Path(RESULTS_DIR) / base_name)


def list_datasets():
    print("Available datasets:\n")
    for key, filename in DATASETS.items():
        filepath = Path(SUBSETS_DIR) / filename
        if filepath.exists():
            size = filepath.stat().st_size / 1024
            print(f"  - {key:20} -> {filename:50} ({size:.1f} KB)")
        else:
            print(f"  - {key:20} -> {filename:50} (NOT FOUND)")
    print()


def cli_mode(args):
    input_file = get_input_file(args.dataset)
    output_file = get_output_file(args.dataset)
    prompt_key = PROMPT_KEYS[args.dataset]

    print(f"Dataset: {get_dataset_name(prompt_key)}")
    print(f"Input:   {input_file}")
    print(f"Output:  {output_file}")
    print()

    classify_dataset(
        input_file=input_file,
        output_file=output_file,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        dataset_key=prompt_key,
        max_workers=args.workers,
    )


def main():
    parser = argparse.ArgumentParser(description="Classify sentiment using Llama on Vertex AI")
    parser.add_argument("dataset", nargs="?", help="Dataset key (see --list-datasets for options)")
    parser.add_argument("--batch-size", type=int, default=10, help="Texts per API request")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum texts to process")
    parser.add_argument("--workers", type=int, default=1, help="Parallel API requests")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets and exit")

    args = parser.parse_args()

    if args.list_datasets:
        list_datasets()
        return

    if args.dataset is None:
        print("Provide dataset key or use --list-datasets")
        return

    try:
        cli_mode(args)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
