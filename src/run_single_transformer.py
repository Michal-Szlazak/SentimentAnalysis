import argparse
import os

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

    # Load dataset
    raw_dataset = {
        args.dataset: load_dataset("szlazakm/SentimentAnalysis", data_files=args.path)
    }
    prepared_datasets = prepare_datasets(raw_dataset)
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
    # Clean model name for file path safety
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
