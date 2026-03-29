"""
Batch classification of all required datasets.
Processes each dataset sequentially with fixed parameters: batch_size=30, workers=4
"""

from pathlib import Path
from gemini_sentiment_classifier import classify_dataset
from gemini_prompts import get_dataset_name
from run_classifier import DATASETS, PROMPT_KEYS, SUBSETS_DIR, RESULTS_DIR

# List of datasets to classify (in order)
DATASETS_TO_PROCESS = [
    "tweet_sentiment_full",
    "tweet_irony_full",
    "finance_50_full",
    "finance_66_full",
    "finance_75_full",
    "finance_all_full",
]

# Fixed parameters
BATCH_SIZE = 30
WORKERS = 4

def get_input_file(dataset_key: str) -> str:
    """Get full path to dataset file."""
    return str(Path(SUBSETS_DIR) / DATASETS[dataset_key])

def get_output_file(dataset_key: str) -> str:
    """Get output file path with gemini prefix."""
    base_name = DATASETS[dataset_key].replace("_test_stratified.jsonl", "_gemini_classified.jsonl")
    base_name = base_name.replace("_full.jsonl", "_gemini_classified_full.jsonl")
    return str(Path(RESULTS_DIR) / base_name)

def main():
    print("=" * 80)
    print("🚀 Starting Batch Sentiment Classification")
    print("=" * 80)
    print(f"Configuration: batch_size={BATCH_SIZE}, workers={WORKERS}\n")
    
    total_datasets = len(DATASETS_TO_PROCESS)
    
    for idx, dataset_key in enumerate(DATASETS_TO_PROCESS, 1):
        try:
            input_file = get_input_file(dataset_key)
            output_file = get_output_file(dataset_key)
            prompt_key = PROMPT_KEYS[dataset_key]
            
            print(f"\n{'=' * 80}")
            print(f"[{idx}/{total_datasets}] Processing: {dataset_key}")
            print(f"Dataset Name: {get_dataset_name(prompt_key)}")
            print(f"Input:  {input_file}")
            print(f"Output: {output_file}")
            print(f"{'=' * 80}\n")
            
            # Check if input file exists
            input_path = Path(input_file)
            if not input_path.exists():
                print(f"❌ Input file not found: {input_file}")
                print(f"   Skipping {dataset_key}...\n")
                continue
            
            # Run classification
            classify_dataset(
                input_file=input_file,
                output_file=output_file,
                batch_size=BATCH_SIZE,
                dataset_key=prompt_key,
                max_workers=WORKERS,
            )
            
            print(f"\n✅ Completed: {dataset_key}\n")
            
        except Exception as e:
            print(f"\n❌ Error processing {dataset_key}: {str(e)}\n")
            continue
    
    print("\n" + "=" * 80)
    print("✨ Batch classification complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
