import argparse
import os
import sys

import pandas as pd
from datasets import load_dataset

from deep_learning.transformer_ml import run_transformer_analysis
from utils.dataset_loader import prepare_datasets

# Lift memory ceiling for M4
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    print(f"📦 Loading ONLY {args.dataset}... {args.path}")

    # We load ONLY the specific dataset requested
    raw_dataset = {
        args.dataset: load_dataset("szlazakm/SentimentAnalysis", data_files=args.path)
    }
    prepared_datasets = prepare_datasets(raw_dataset)
    bundle = prepared_datasets[args.dataset]

    # Run the analysis
    result = run_transformer_analysis(args.dataset, bundle, args.model)

    # Append result to a CSV (so we don't lose data if a later one fails)
    results_df = pd.DataFrame([result])
    results_df.to_csv(
        "transformer_results.csv",
        mode="a",
        index=False,
        header=not os.path.exists("transformer_results.csv"),
    )
    print(f"✅ Finished {args.model} on {args.dataset}")


if __name__ == "__main__":
    main()
