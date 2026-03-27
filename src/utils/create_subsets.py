import os
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from dataset_loader import prepare_datasets
from data_downloader import load_cached_datasets, download_and_cache_datasets

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

    out = []
    
    # Dla każdej klasy, weź dokładnie examples_per_class próbek
    for class_label in sorted(df[label_column].unique()):
        class_df = df[df[label_column] == class_label]
        
        # Losuj bez zastępowania, max ile przykładów jest dostępnych
        num_samples = min(examples_per_class, len(class_df))
        
        sampled = class_df.sample(n=num_samples, random_state=42)
        
        for _, row in sampled.iterrows():
            out.append({"text": row[text_column], "label": row[label_column]})
    
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def create_subsets_from_prepared(
    prepared_datasets,
    out_dir,
    target_sample_size=2000,
    split_name="test",  # tylko dane testowe
):
    for name, data in prepared_datasets.items():
            df = pd.DataFrame({
                'text': data[f'X_{split_name}'],
                'sentiment': data[f'y_{split_name}']
            })
            
            # Oblicz frakcję na podstawie stałego limitu
            # Jeśli zbiór ma 74k, frakcja będzie mała. 
            # Jeśli zbiór ma 1500, frakcja wyniesie 0.99 (weźmie prawie wszystko).
            test_fraction = min(target_sample_size / len(df), 0.99)
            
            _, subset = train_test_split(
                df,
                test_size=test_fraction,
                stratify=df['sentiment'],
                random_state=42
            )
            
            # Reszta kodu bez zmian...
            out_path = Path(out_dir) / f"{name}_{split_name}_stratified.jsonl"
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
    # Pobierz i cache'uj zbiory (jeśli już nie istnieją)
    print("🔄 Sprawdzanie cache danych...\n")
    download_and_cache_datasets()
    
    print("\n📂 Załadowanie cache'owanych zbiorów...\n")
    raw = load_cached_datasets()
    prepared = prepare_datasets(raw)

    print("\n📝 Tworzenie subsetów...\n")
    create_subsets_from_prepared(
        prepared,
        out_dir=r"c:\Users\bushi\Documents\GitHub\SentimentAnalysis\subsets",
        split_name="test",
    )

    print("\n📝 Tworzenie przykładów do promptów...\n")
    create_prompt_examples_from_prepared(
        prepared,
        out_dir=r"c:\Users\bushi\Documents\GitHub\SentimentAnalysis\subsets",
        examples_per_class=3,
        split_name="train",
    )
    
    print("\n✅ Wszystko gotowe!")