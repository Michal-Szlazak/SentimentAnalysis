import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def prepare_datasets(datasets_dict):
    prepared_data = {}
    le = LabelEncoder()

    if "yelp_train" in datasets_dict and "yelp_test" in datasets_dict:
        print("🔗 Łączenie Yelp: używam dedykowanego zbioru testowego.")
        df_train = pd.DataFrame(datasets_dict['yelp_train']['train'])
        df_test = pd.DataFrame(datasets_dict['yelp_test']['train'])
        
        # Sampling dla wydajności (opcjonalnie)
        if len(df_train) > 30000: df_train = df_train.sample(30000, random_state=42)
        if len(df_test) > 5000: df_test = df_test.sample(5000, random_state=42)
        
        prepared_data["yelp_full"] = {
            'X_train': df_train['text'].astype(str),
            'X_test': df_test['text'].astype(str),
            'y_train': df_train['label'],
            'y_test': df_test['label']
        }

        print(f"✅ Prepared yelp_full: {len(prepared_data['yelp_full']['X_train'])} train / {len(prepared_data['yelp_full']['X_test'])} test samples")
    
    for name, ds_dict in datasets_dict.items():
        
        if "yelp" in name: continue  # Yelp już obsłużony powyżej

        df = pd.DataFrame(ds_dict['train'])
        
        # --- CUSTOM LOGIC PER DATASET ---
        if name.startswith("tweet_eval"):
            # These have a 'split' column we must respect
            train_part = df[df['split'] == 'train']
            test_part = df[df['split'] != 'train']
            
            X_train, y_train = train_part['text'], train_part['label_int']
            X_test, y_test = test_part['text'], test_part['label_int']

        elif name == "twitter":
            # Twitter uses 'content' and 'sentiment'
            df = df.dropna(subset=['content', 'sentiment'])
            X = df['content']
            y = le.fit_transform(df['sentiment'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        elif name == "imdb":
            # IMDB uses 'review' and 'sentiment'
            X = df['review']
            y = le.fit_transform(df['sentiment'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        elif name.startswith("finance"):
            # Finance uses 'sentence' and 'sentiment'
            X = df['sentence']
            y = le.fit_transform(df['sentiment'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        else:
            print(f"Unknown dataset format for {name}, skipping...")
            continue

        prepared_data[name] = {
            'X_train': X_train.astype(str),
            'X_test': X_test.astype(str),
            'y_train': y_train,
            'y_test': y_test
        }
        print(f"✅ Prepared {name}: {len(X_train)} train / {len(X_test)} test samples")

    return prepared_data