import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score
import pandas as pd

# Pobranie słownika VADER (wymagane przy pierwszym uruchomieniu)
nltk.download('vader_lexicon', quiet=True)

def run_dictionary_analysis(name, data_bundle):
    print(f"\n{'-'*40}\nAnaliza Słownikowa (VADER) dla: {name}\n{'-'*40}")
    
    X_test = data_bundle['X_test']
    y_test = data_bundle['y_test']
    
    analyzer = SentimentIntensityAnalyzer()
    
    # Mapowanie predykcji VADER na Twoje labele
    # VADER zwraca: neg, neu, pos oraz compound (od -1 do 1)
    
    y_pred = []
    for text in X_test:
        score = analyzer.polarity_scores(text)['compound']
        
        # Logika mapowania zależy od datasetu (różne liczby klas)
        # Przykład dla binarnych (pos/neg jak IMDB) lub 3-klasowych (Twitter)
        num_classes = len(set(y_test))
        
        if num_classes == 2:
            # 0 = neg, 1 = pos
            y_pred.append(1 if score >= 0 else 0)
        elif num_classes == 3:
            # Zakładamy standard: 0=neg, 1=neu, 2=pos (częste w tweet_eval)
            if score > 0.05: y_pred.append(2)
            elif score < -0.05: y_pred.append(0)
            else: y_pred.append(1)
        else:
            # Dla Yelp (5 klas) mapujemy compound (-1 do 1) na (0 do 4)
            # To jest uproszczenie matematyczne:
            y_pred.append(int((score + 1) * 2))

    acc = accuracy_score(y_test, y_pred)
    
    print(pd.DataFrame([{
        "Method": "Lexicon-based",
        "Tool": "VADER",
        "Accuracy": f"{acc:.4f}"
    }]))