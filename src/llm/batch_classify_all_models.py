"""
Batch classification runner for all full datasets.

Features:
- Runs datasets sequentially (one after another).
- Supports model switch: gemini or llama.
- Uses per-dataset default batch sizes (can be overridden globally).

Usage:
    cd src/llm

    # Run sentiment classification for all full datasets with selected model
    python batch_classify_all_models.py --model gemini
    python batch_classify_all_models.py --model llama
    python batch_classify_all_models.py --model mistral

    # Optional: process only selected datasets
    python batch_classify_all_models.py --model llama --datasets twitter_full imdb_full

    # Optional: override batch size for all datasets
    python batch_classify_all_models.py --model mistral --batch-size 20
"""

import argparse
from pathlib import Path

from gemini_prompts import get_dataset_name
from gemini_sentiment_classifier import classify_dataset as classify_dataset_gemini
from llama_sentiment_classifier import classify_dataset as classify_dataset_llama
from mistral_sentiment_classifier import classify_dataset as classify_dataset_mistral
from run_classifier import DATASETS, PROMPT_KEYS, SUBSETS_DIR, RESULTS_DIR

# Order of full datasets to process.
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

# Per-dataset default batch size.
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

# Keep model outputs separated to avoid overwriting and simplify evaluation.
MODEL_RESULTS_DIRS = {
    "gemini": Path(SUBSETS_DIR).parent / "results_gemini2",
    "llama": Path(SUBSETS_DIR).parent / "results_llama",
    "mistral": Path(SUBSETS_DIR).parent / "results_mistral",
}


def get_input_file(dataset_key: str) -> str:
    return str(Path(SUBSETS_DIR) / DATASETS[dataset_key])


def get_output_file(dataset_key: str, model: str) -> str:
    model_results_dir = MODEL_RESULTS_DIRS[model]
    base_name = DATASETS[dataset_key].replace("_test_stratified.jsonl", f"_{model}_classified.jsonl")
    base_name = base_name.replace("_full.jsonl", f"_{model}_classified_full.jsonl")
    return str(model_results_dir / base_name)


def resolve_classifier(model: str):
    if model == "gemini":
        return classify_dataset_gemini
    if model == "llama":
        return classify_dataset_llama
    if model == "mistral":
        return classify_dataset_mistral
    raise ValueError(f"Unsupported model: {model}")


def validate_datasets(datasets: list[str]):
    invalid = [d for d in datasets if d not in DATASETS_TO_PROCESS]
    if invalid:
        raise ValueError(
            "Invalid dataset key(s): "
            f"{invalid}. Allowed full datasets: {', '.join(DATASETS_TO_PROCESS)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run sequential sentiment classification for all full datasets."
    )
    parser.add_argument(
        "--model",
        choices=["gemini", "llama", "mistral"],
        default="gemini",
        help="Model backend to use (default: gemini)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel requests per dataset (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for all datasets (default: use per-dataset map)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max samples per dataset (default: all)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DATASETS_TO_PROCESS,
        help=(
            "Optional subset of full datasets to run. "
            f"Default: all ({', '.join(DATASETS_TO_PROCESS)})"
        ),
    )

    args = parser.parse_args()
    validate_datasets(args.datasets)

    classifier = resolve_classifier(args.model)
    total_datasets = len(args.datasets)

    print("=" * 80)
    print("Batch sentiment classification")
    print("=" * 80)
    print(f"Model:   {args.model}")
    print(f"Workers: {args.workers}")
    print(f"Results: {MODEL_RESULTS_DIRS[args.model]}")
    if args.batch_size is None:
        print("Batch:   per-dataset defaults")
    else:
        print(f"Batch:   override={args.batch_size}")
    print(f"Datasets to process: {total_datasets}\n")

    for idx, dataset_key in enumerate(args.datasets, start=1):
        try:
            input_file = get_input_file(dataset_key)
            output_file = get_output_file(dataset_key, args.model)
            prompt_key = PROMPT_KEYS[dataset_key]
            batch_size = args.batch_size or DEFAULT_BATCH_SIZES[dataset_key]

            print("-" * 80)
            print(f"[{idx}/{total_datasets}] Processing: {dataset_key}")
            print(f"Dataset Name: {get_dataset_name(prompt_key)}")
            print(f"Input:      {input_file}")
            print(f"Output:     {output_file}")
            print(f"Batch size: {batch_size}")
            print("-" * 80)

            if not Path(input_file).exists():
                print(f"Input file not found, skipping: {input_file}\n")
                continue

            classifier(
                input_file=input_file,
                output_file=output_file,
                batch_size=batch_size,
                max_samples=args.max_samples,
                dataset_key=prompt_key,
                max_workers=args.workers,
            )

            print(f"Completed: {dataset_key}\n")

        except Exception as exc:
            print(f"Error in {dataset_key}: {exc}\n")
            continue

    print("=" * 80)
    print("All requested datasets processed")
    print("=" * 80)


if __name__ == "__main__":
    main()
