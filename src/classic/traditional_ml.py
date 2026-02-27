import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def run_traditional_ml(name, data_bundle):
    print(f"\n{'='*40}\nAnaliza dla: {name}\n{'='*40}")
    
    X_train, X_test = data_bundle['X_train'], data_bundle['X_test']
    y_train, y_test = data_bundle['y_train'], data_bundle['y_test']

    vectorizers = {
        "Bag-of-Words": CountVectorizer(max_features=5000, stop_words='english'),
        "TF-IDF": TfidfVectorizer(max_features=5000, stop_words='english')
    }
    
    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM (Linear)": LinearSVC(max_iter=1000, dual=False)
    }

    results = []
    for v_name, vectorizer in vectorizers.items():
        # Transformacja tekstu na wektory
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        for m_name, model in models.items():
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            acc = accuracy_score(y_test, y_pred)
            
            results.append({
                "Vectorizer": v_name, 
                "Model": m_name, 
                "Accuracy": f"{acc:.4f}"
            })
            
    print(pd.DataFrame(results))