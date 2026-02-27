from utils.dataset_loader import prepare_datasets
from classic.dictionary import run_dictionary_analysis
from classic.traditional_ml import run_traditional_ml
from utils.dataset_loader import prepare_datasets
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

raw_datasets = {}
for name, path in files.items():
    print(f"Pobieranie: {name}...")
    raw_datasets[name] = load_dataset("szlazakm/SentimentAnalysis", data_files=path)

prepared_datasets = prepare_datasets(raw_datasets)

for name, data_bundle in prepared_datasets.items():
    run_dictionary_analysis(name, data_bundle)
    run_traditional_ml(name, data_bundle)