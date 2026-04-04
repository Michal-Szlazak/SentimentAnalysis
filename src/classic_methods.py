import pandas as pd
from datasets import load_dataset

from classic.dictionary import run_dictionary_analysis
from classic.traditional_ml import run_traditional_ml
from utils.dataset_loader import prepare_datasets

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
    "finance_all_agree": "FinancialPhraseBank/financial_phrasebank_AllAgree.parquet",
}

from concurrent.futures import ProcessPoolExecutor, as_completed


def process_dataset(args):
    name, data_bundle = args

    results = []
    results += run_traditional_ml(name, data_bundle)
    results += run_dictionary_analysis(name, data_bundle)

    return results


if __name__ == "__main__":
    # Load datasets
    raw_datasets = {}
    for name, path in files.items():
        print(f"Pobieranie: {name}...")
        raw_datasets[name] = load_dataset("szlazakm/SentimentAnalysis", data_files=path)

    prepared_datasets = prepare_datasets(raw_datasets)

    # Run in parallel
    all_results = []

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_dataset, (name, data_bundle))
            for name, data_bundle in prepared_datasets.items()
        ]

        for future in as_completed(futures):
            result = future.result()
            all_results.extend(result)

    # Display results
    print("\n\nWyniki końcowe:")
    results_df = pd.DataFrame(all_results)
    print(results_df)
