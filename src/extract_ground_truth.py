import argparse
import os

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_ground_truth(datasets_dict, target_name):
    """
    Extracts the y_test labels using the exact same logic and
    random_state as the training script to ensure row parity.
    """
    le = LabelEncoder()

    # --- YELP SPECIAL CASE ---
    if "yelp_train" in datasets_dict and "yelp_test" in datasets_dict:
        # We only care about the test set for the ground truth CSV
        df_test = pd.DataFrame(datasets_dict["yelp_test"]["train"])

        # We must apply the same sampling used in training so rows match!
        if len(df_test) > 4000:
            df_test = df_test.sample(4000, random_state=42)

        return df_test["label"].tolist()

    # --- OTHER DATASETS ---
    for name, ds_dict in datasets_dict.items():
        if "yelp" in name:
            continue

        df = pd.DataFrame(ds_dict["train"])

        if name.startswith("tweet_eval"):
            # respect the split column
            test_part = df[df["split"] != "train"]
            return test_part["label_int"].tolist()

        elif name in ["twitter", "imdb"] or name.startswith("finance"):
            text_col = (
                "review"
                if name == "imdb"
                else ("content" if name == "twitter" else "sentence")
            )
            label_col = "sentiment"

            df = df.dropna(subset=[text_col, label_col])
            X = df[text_col]
            y = le.fit_transform(df[label_col])

            # Replicate the train_test_split to get the exact same test rows
            _, _, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return y_test.tolist()

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="e.g., yelp_full, imdb, twitter"
    )
    parser.add_argument(
        "--path", type=str, required=True, help="Path to the parquet file"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=False,
        help="Second path specifically for yelp_full test set",
    )
    args = parser.parse_args()

    print(f"📂 Loading data for: {args.dataset}...")

    raw_data = {}
    if args.dataset == "yelp_full":
        if not args.test_path:
            raise ValueError("Yelp requires --test_path to get the test labels.")
        # We load both because some logic depends on the training context,
        # but we primarily process the test set.
        raw_data["yelp_train"] = load_dataset(
            "szlazakm/SentimentAnalysis", data_files=args.path
        )
        raw_data["yelp_test"] = load_dataset(
            "szlazakm/SentimentAnalysis", data_files=args.test_path
        )
    else:
        raw_data[args.dataset] = load_dataset(
            "szlazakm/SentimentAnalysis", data_files=args.path
        )

    true_labels = get_ground_truth(raw_data, args.dataset)

    if true_labels is not None:
        output_filename = f"{args.dataset}_ground_truth.csv"
        df_output = pd.DataFrame(
            {"rowNumber": range(len(true_labels)), "true_label": true_labels}
        )

        df_output.to_csv(output_filename, index=False)
        print(f"✅ Created: {output_filename}")
        print(f"Total rows extracted: {len(true_labels)}")
    else:
        print("❌ Could not extract labels. Check dataset name/path.")


if __name__ == "__main__":
    main()
