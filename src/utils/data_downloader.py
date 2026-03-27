import os
from pathlib import Path
from datasets import load_dataset

# Folder do przechowywania pobranych zbiorów
DATA_CACHE_DIR = Path(__file__).parent.parent.parent / "data_cache"

FILES = {
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

def download_and_cache_datasets():
    """
    Pobiera zbiory danych z HuggingFace i zapisuje je lokalnie jako parquet.
    Jeśli zbiory już istnieją, pomija pobieranie.
    """
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    for name, remote_path in FILES.items():
        local_path = DATA_CACHE_DIR / f"{name}.parquet"
        
        if local_path.exists():
            print(f"✅ {name} już exists ({local_path})")
            continue
        
        print(f"📥 Pobieranie {name}...")
        try:
            dataset = load_dataset("szlazakm/SentimentAnalysis", data_files=remote_path)
            df = dataset['train'].to_pandas()
            df.to_parquet(local_path, index=False)
            print(f"   ✅ Zapisano: {local_path}")
        except Exception as e:
            print(f"   ❌ Błąd: {e}")

def load_cached_datasets():
    """
    Wczytuje wszystkie zapamiętane zbiory danych z cache'u.
    Zwraca słownik {'dataset_name': dataset}
    """
    datasets = {}
    
    if not DATA_CACHE_DIR.exists():
        print("❌ Folder cache nie istnieje. Uruchom najpierw download_and_cache_datasets()")
        return datasets
    
    for name in FILES.keys():
        local_path = DATA_CACHE_DIR / f"{name}.parquet"
        
        if not local_path.exists():
            print(f"⚠️  {name} nie znaleziony w cache ({local_path})")
            continue
        
        import pandas as pd
        df = pd.read_parquet(local_path)
        datasets[name] = {'train': df}
    
    return datasets

if __name__ == "__main__":
    print("🔄 Pobieranie i cache'owanie zbiorów...\n")
    download_and_cache_datasets()
    print("\n✅ Pobieranie zakończone!")
