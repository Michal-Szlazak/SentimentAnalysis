import numpy as np
import random
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def run_traditional_ml(name, data_bundle):
    print(f"\n{'='*40}\nAnalysis for: {name}\n{'='*40}")
    
    set_seed(42)

    X_train, X_test = data_bundle['X_train'], data_bundle['X_test']
    y_train, y_test = data_bundle['y_train'], data_bundle['y_test']

    results = []

    # 🔹 Pipelines (vectorizer + model together)
    pipelines = {
        "BoW + NB": Pipeline([
            ("vec", CountVectorizer(ngram_range=(1,2), max_features=10000)),
            ("clf", MultinomialNB())
        ]),
        
        "TF-IDF + NB": Pipeline([
            ("vec", TfidfVectorizer(ngram_range=(1,2), max_features=10000)),
            ("clf", MultinomialNB())
        ]),
        
        "BoW + SVM": Pipeline([
            ("vec", CountVectorizer(ngram_range=(1,2), max_features=10000)),
            ("clf", LinearSVC(dual=False, max_iter=5000, class_weight='balanced'))
        ]),
        
        "TF-IDF + SVM": Pipeline([
            ("vec", TfidfVectorizer(ngram_range=(1,2), max_features=10000)),
            ("clf", LinearSVC(dual=False, max_iter=5000, class_weight='balanced'))
        ])
    }

    # 🔹 Hyperparameter grids (small but meaningful)
    param_grids = {
        "BoW + NB": {"clf__alpha": [0.01, 0.1, 1.0]},
        "TF-IDF + NB": {"clf__alpha": [0.01, 0.1, 1.0]},
        "BoW + SVM": {"clf__C": [0.1, 1, 10]},
        "TF-IDF + SVM": {"clf__C": [0.1, 1, 10]},
    }

    for name_pipe, pipeline in pipelines.items():
        print(f"\nTraining: {name_pipe}")

        grid = GridSearchCV(
            pipeline,
            param_grids[name_pipe],
            scoring="f1_macro",
            cv=5,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        vec_name, model_name = name_pipe.split(" + ")

        results.append({
            "Dataset": name,
            "Vectorizer": vec_name,
            "Model": model_name,
            "Accuracy": acc,
            "F1": f1,
            "Best Params": grid.best_params_
        })

    return results