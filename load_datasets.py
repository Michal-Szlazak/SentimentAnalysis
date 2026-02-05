from datasets import load_dataset

def verify_datasets(datasets_dict):
    print(f"{'Dataset Name':<25} | {'Rows':<8} | {'Columns'}")
    print("-" * 60)
    
    for name, ds in datasets_dict.items():
        
        cols = ", ".join(ds.column_names)
        rows = len(ds)
        
        print(f"{name:<25} | {rows:<8} | {cols}")
        
        print(f"   Sample: {ds[0]}")

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

# to tylko przyklad ladowania z jednego zrodel

all_datasets = {}
for name, path in files.items():
    print(f"Ładowanie: {name}...")
    all_datasets[name] = load_dataset("szlazakm/SentimentAnalysis", data_files=path)
    # mozna dodac split='train' jesli potrzebne lub split=['test', 'train','val'], jezeli chcemy podzielic na czesci


verify_datasets(all_datasets)