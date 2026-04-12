import argparse
import os

import pandas as pd
from datasets import load_dataset

# Note: Using the loader from deep_learning as it's optimized for the transformer lists
from deep_learning.dataset_loader import prepare_datasets
from deep_learning.transformer_ml import run_transformer_analysis

# Lift memory ceiling for M4
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="e.g., yelp_full, imdb, twitter"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to parquet. For yelp, provide train path.",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=False,
        help="Path to test parquet (specifically for yelp_full)",
    )
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    raw_datasets = {}

    if args.dataset == "yelp_full":
        if not args.test_path:
            raise ValueError(
                "For yelp_full, you must provide both --path (train) and --test_path (test)."
            )

        print(f"📦 Loading Yelp Split...")
        raw_datasets["yelp_train"] = load_dataset(
            "szlazakm/SentimentAnalysis", data_files=args.path
        )
        raw_datasets["yelp_test"] = load_dataset(
            "szlazakm/SentimentAnalysis", data_files=args.test_path
        )
    else:
        print(f"📦 Loading {args.dataset}... {args.path}")
        raw_datasets[args.dataset] = load_dataset(
            "szlazakm/SentimentAnalysis", data_files=args.path
        )

    # This calls your logic that merges yelp_train and yelp_test into 'yelp_full'
    prepared_datasets = prepare_datasets(raw_datasets)

    # We always use args.dataset to pull the correct bundle
    bundle = prepared_datasets[args.dataset]

    # Run the analysis
    output = run_transformer_analysis(args.dataset, bundle, args.model)

    # 1. Append summary result to main CSV
    metrics_df = pd.DataFrame([output["metrics"]])
    metrics_df.to_csv(
        "transformer_results.csv",
        mode="a",
        index=False,
        header=not os.path.exists("transformer_results.csv"),
    )

    # 2. Generate specific predictions CSV
    safe_model_name = args.model.replace("/", "_")
    pred_filename = f"{args.dataset}_{safe_model_name}_predictions.csv"

    preds_data = output["predictions"]
    df_preds = pd.DataFrame(
        {
            "rowNumber": range(len(preds_data["predicted_label"])),
            "predicted_label": preds_data["predicted_label"],
            "confidence": preds_data["confidence"],
        }
    )

    df_preds.to_csv(pred_filename, index=False)

    print(f"✅ Summary saved to transformer_results.csv")
    print(f"📊 Detailed predictions saved to: {pred_filename}")


if __name__ == "__main__":
    main()
