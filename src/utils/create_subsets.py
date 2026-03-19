import os
import json
from pathlib import Path
import pandas as pd  # dodaj import

from dataset_loader import prepare_datasets
from subset_sampler import SubsetSampler

def save_jsonl(df, path, text_column="text", label_column="sentiment"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({
                "text": row[text_column],
                "label": row[label_column],
            }, ensure_ascii=False) + "\n")

def save_prompt_examples(df, path, text_column="text", label_column="sentiment", examples_per_class=3):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sampler = SubsetSampler(df, target_column=label_column, samples_per_class=examples_per_class)
    examples = sampler.stratified_sample()

    out = []
    for _, row in examples.iterrows():
        out.append({"text": row[text_column], "label": row[label_column]})

    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def create_subsets_from_prepared(
    prepared_datasets,
    out_dir,
    samples_per_class=500,
    split_name="test",  # tylko dane testowe
):
    for name, data in prepared_datasets.items():
        # Konwertuj na DataFrame
        df = pd.DataFrame({
            'text': data[f'X_{split_name}'],
            'sentiment': data[f'y_{split_name}']
        })
        sampler = SubsetSampler(df, target_column="sentiment", samples_per_class=samples_per_class)
        subset = sampler.stratified_sample()
        out_path = Path(out_dir) / f"{name}_{split_name}_balanced.jsonl"
        save_jsonl(subset, out_path)

def create_prompt_examples_from_prepared(
    prepared_datasets,
    out_dir,
    examples_per_class=3,
    split_name="train",  # przykłady z train
):
    for name, data in prepared_datasets.items():
        # Konwertuj na DataFrame
        df = pd.DataFrame({
            'text': data[f'X_{split_name}'],
            'sentiment': data[f'y_{split_name}']
        })
        out_path = Path(out_dir) / f"{name}_{split_name}_examples.json"
        save_prompt_examples(df, out_path, examples_per_class=examples_per_class)

if __name__ == "__main__":
    from datasets import load_dataset

    files = {
        "tweet_eval_irony": "TweetEvalIrony/tweeteval_irony.parquet",
        "tweet_eval_sentiment": "TweetEvalSentiment/tweeteval_sentiment.parquet",
        "twitter": "Twitter/twitter.parquet",
        "imdb": "IMDB/imdb.parquet",
        "yelp_test": "Yelp/yelp_test.parquet",
        "yelp_train": "Yelp/yelp_train.parquet",
        "finance_50agree": "FinancialPhraseBank/financial_phrasebank_50Agree.parquet",
        "finance_66agree": "FinancialPhraseBank/financial_phrasebank_66Agree.parquet",
        "finance_75agree": "FinancialPhraseBank/financial_phrasebank_75Agree.parquet",
        "finance_all_agree": "FinancialPhraseBank/financial_phrasebank_AllAgree.parquet"
    }
    raw = {n: load_dataset("szlazakm/SentimentAnalysis", data_files=p) for n, p in files.items()}
    prepared = prepare_datasets(raw)

    create_subsets_from_prepared(
        prepared,
        out_dir=r"c:\Users\bushi\Documents\GitHub\SentimentAnalysis\subsets",
        split_name="test",
    )

    create_prompt_examples_from_prepared(
        prepared,
        out_dir=r"c:\Users\bushi\Documents\GitHub\SentimentAnalysis\subsets",
        examples_per_class=3,
        split_name="train",
    )