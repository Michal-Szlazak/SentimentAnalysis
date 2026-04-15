import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_datasets(datasets_dict):
    prepared_data = {}
    le = LabelEncoder()

    # Yelp handling logic
    if "yelp_test" in datasets_dict:
        df_train = pd.DataFrame(datasets_dict["yelp_train"]["train"])
        df_test = pd.DataFrame(datasets_dict["yelp_test"]["train"])

        # Sampling for performance on Air M4
        if len(df_train) > 20000:
            df_train = df_train.sample(20000, random_state=42)
        if len(df_test) > 4000:
            df_test = df_test.sample(4000, random_state=42)

        prepared_data["yelp_full"] = {
            "X_train": df_train["text"].astype(str).tolist(),
            "X_test": df_test["text"].astype(str).tolist(),
            "y_train": df_train["label"].tolist(),
            "y_test": df_test["label"].tolist(),
        }

    for name, ds_dict in datasets_dict.items():
        if "yelp" in name:
            continue

        df = pd.DataFrame(ds_dict["train"])

        if name.startswith("tweet_eval"):
            train_part = df[df["split"] == "train"]
            test_part = df[df["split"] != "train"]
            X_train, y_train = train_part["text"], train_part["label_int"]
            X_test, y_test = test_part["text"], test_part["label_int"]
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
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            continue

        prepared_data[name] = {
            "X_train": X_train.astype(str).tolist(),
            "X_test": X_test.astype(str).tolist(),
            "y_train": y_train.tolist(),
            "y_test": y_test.tolist(),
        }
    return prepared_data
