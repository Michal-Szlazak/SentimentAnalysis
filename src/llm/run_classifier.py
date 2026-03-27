"""
Runner script for Gemini Sentiment Classification
Easy-to-use interface for batch or full dataset classification.
"""
from gemini_prompts import get_dataset_name

import sys
import argparse
from pathlib import Path
from gemini_sentiment_classifier import classify_batch, classify_dataset, load_dataset

# Available datasets in the subsets folder
DATASETS = {
    "imdb": "imdb_test_stratified.jsonl",
    "twitter": "twitter_test_stratified.jsonl",
    "finance_50": "finance_50agree_test_stratified.jsonl",
    "finance_66": "finance_66agree_test_stratified.jsonl",
    "finance_75": "finance_75agree_test_stratified.jsonl",
    "finance_all": "finance_all_agree_test_stratified.jsonl",
    "yelp": "yelp_full_test_stratified.jsonl",
    "tweet_sentiment": "tweet_eval_sentiment_test_stratified.jsonl",
    "tweet_irony": "tweet_eval_irony_test_stratified.jsonl",
}

SUBSETS_DIR = r"c:\\Users\\bushi\\Documents\\GitHub\\SentimentAnalysis\\src\\subsets"
RESULTS_DIR = r"c:\\Users\\bushi\\Documents\\GitHub\\SentimentAnalysis\\src\\results"

def get_input_file(dataset_key: str) -> str:
    """Get full path to dataset file."""
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Available: {', '.join(DATASETS.keys())}")
    return str(Path(SUBSETS_DIR) / DATASETS[dataset_key])

def get_output_file(dataset_key: str) -> str:
    """Get output file path with gemini prefix."""
    base_name = DATASETS[dataset_key].replace("_test_stratified.jsonl", "_gemini_classified.jsonl")
    return str(Path(RESULTS_DIR) / base_name)

def list_datasets():
    """Print available datasets."""
    print("📊 Available datasets:\n")
    for key, filename in DATASETS.items():
        filepath = Path(SUBSETS_DIR) / filename
        if filepath.exists():
            size = filepath.stat().st_size / 1024  # KB
            print(f"  • {key:20} → {filename:50} ({size:.1f} KB)")
        else:
            print(f"  • {key:20} → {filename:50} (❌ NOT FOUND)")
    print()

def interactive_mode():
    """Interactive selection menu."""
    print("=" * 70)
    print("🤖 Gemini Sentiment Classifier - Interactive Mode")
    print("=" * 70)
    print()
    
    list_datasets()
    
    # Get dataset choice
    while True:
        dataset = input("📁 Enter dataset key (or 'list' to see all): ").strip().lower()
        if dataset == "list":
            list_datasets()
            continue
        if dataset in DATASETS:
            break
        print(f"❌ Invalid dataset. Try: {', '.join(DATASETS.keys())}\n")
    
    # Get mode choice
    print("\n🔧 Choose mode:")
    print("  1. Classify BATCH (send 10 texts at once)")
    print("  2. Classify FULL DATASET (all texts, batched automatically)")
    
    mode = input("\nEnter mode (1 or 2): ").strip()
    
    input_file = get_input_file(dataset)
    output_file = get_output_file(dataset)
    
    print()
    
    if mode == "1":
        # Batch mode
        batch_size = input("How many texts per request? [default: 10]: ").strip() or "10"
        try:
            batch_size = int(batch_size)
        except ValueError:
            print("❌ Invalid batch size, using 10")
            batch_size = 10
        
        max_samples = input("Max texts to send? [default: 50]: ").strip() or "50"
        try:
            max_samples = int(max_samples)
        except ValueError:
            max_samples = 50
        
        print()
        classify_dataset(
            input_file=input_file,
            output_file=output_file,
            batch_size=batch_size,
            max_samples=max_samples,
            dataset_key=dataset  # DODAJ
        )
    
    elif mode == "2":
        # Full dataset mode
        confirm = input(f"⚠️  This will classify ALL texts from {dataset}. Continue? (y/n): ").strip().lower()
        if confirm == "y":
            print()
            classify_dataset(
                input_file=input_file,
                output_file=output_file,
                batch_size=10
            )
        else:
            print("❌ Cancelled")
    else:
        print("❌ Invalid mode")

def cli_mode(args):
    """Command-line interface with arguments."""
def cli_mode(args):
    input_file = get_input_file(args.dataset)
    output_file = get_output_file(args.dataset)
    
    print(f"📂 Dataset:  {get_dataset_name(args.dataset)}")
    print(f"📂 Input:    {input_file}")
    print(f"📂 Output:   {output_file}")
    print()
    
    classify_dataset(
        input_file=input_file,
        output_file=output_file,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        dataset_key=args.dataset  # DODAJ TĘ LINIĘ
    )

def main():
    parser = argparse.ArgumentParser(
        description="Classify sentiment using Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python run_classifier.py
  
  # CLI mode - classify IMDb with 50 samples
  python run_classifier.py imdb --max-samples 50
  
  # Classify all Twitter data with batch size 5
  python run_classifier.py twitter --batch-size 5
  
  # List available datasets
  python run_classifier.py --list-datasets
        """
    )
    
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset key (see --list-datasets for options)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Texts per API request (default: 10)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum texts to process (default: all)"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit"
    )
    
    args = parser.parse_args()
    
    # Handle special flags
    if args.list_datasets:
        list_datasets()
        return
    
    # Interactive or CLI mode
    if args.dataset is None:
        interactive_mode()
    else:
        try:
            cli_mode(args)
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
