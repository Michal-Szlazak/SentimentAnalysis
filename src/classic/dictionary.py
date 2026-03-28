import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score

nltk.download('vader_lexicon', quiet=True)

analyzer = SentimentIntensityAnalyzer()

def map_vader(score, dataset_name):
    if dataset_name == "yelp_full":
        if score <= -0.6: return 0
        elif score <= -0.2: return 1
        elif score <= 0.2: return 2
        elif score <= 0.6: return 3
        else: return 4

    elif dataset_name in ["tweet_eval_sentiment", "twitter"] or "finance" in dataset_name:
        if score > 0.05: return 2
        elif score < -0.05: return 0
        else: return 1

    elif dataset_name in ["imdb", "tweet_eval_irony"]:
        return 1 if score >= 0 else 0

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_dictionary_analysis(name, data_bundle):
    print(f"\n{'-'*40}\nDictionary Analysis (VADER) for: {name}\n{'-'*40}")
    
    X_test = data_bundle['X_test']
    y_test = data_bundle['y_test']
    
    y_pred = [
        map_vader(analyzer.polarity_scores(text or "")['compound'], name)
        for text in X_test
    ]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    return [{
        "Dataset": name,
        "Vectorizer": "-",
        "Model": "VADER",
        "Accuracy": acc,
        "F1": f1
    }]