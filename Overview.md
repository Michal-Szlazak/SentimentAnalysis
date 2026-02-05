

# Projekt: Zastosowanie modeli językowych do analizy sentymentu

**Przegląd literatury, zbiory danych i plan badawczy**

---

## 1. Przegląd Literatury i Stan Badań

    "ABSA.pdf": W artykule tym wprowadzone są autorskie modele Arctic-ABSA (oparte na LLaMA3.1 i ModernBERT), które zestawiono z GPT-4o, Claude 3.5 Sonnet oraz Mistral Large 2,,. 

    "TweetEval" - comperative evaluation for tweet classification.pdf": Badanie ocenia skuteczność modelu RoBERTa (w wariantach base i dotrenowanym na Twitterze) w porównaniu do metod FastText, SVM oraz Bi-LSTM,. 

    "llm-negotiations.pdf": Autorzy wykorzystują modele GPT-3.5 i GPT-4 w rolach negocjujących agentów (generatora i dyskryminatora), porównując je z modelami nadzorowanymi takimi jak RoBERTa, XLNet czy BERTweet,. 

    "arabic-semantic-analysis.pdf": Do klasyfikacji użyto architektur głębokiego uczenia LSTM, GRU i RNN, podczas gdy etykietowanie danych wspomagane było przez LLM takie jak GPT-4o, Claude 3 Sonnet, Gemini 2.5 Pro czy LLaMA 3,. 

    "brain_data.pdf": W eksperymentach wykorzystano model CardiffNLP (oparty na RoBERTa) do etykietowania tekstu oraz sieci MLP i LSTM do przewidywania sentymentu bezpośrednio z sygnałów mózgowych (MEG),. (https://arxiv.org/pdf/2601.18792) 

    "model-uncertainty.pdf": Analiza problemu zmienności opiera się na studiach przypadków z użyciem GPT-4o i Mixtral 8x22B, omawiając także zachowanie modeli takich jak LLaMA czy Falcon,,. 

    "s12652-018-0862-8.pdf": Artykuł porównuje wydajność klasycznych klasyfikatorów uczenia maszynowego, takich jak SVM, Naive Bayes, Regresja Liniowa i Random Forest, w połączeniu z różnymi technikami ekstrakcji cech,. 

    "sa-in-llm-era-reality-check.pdf": Ewaluacja obejmuje duże modele językowe Flan-T5, Flan-UL2, ChatGPT i text-davinci-003, których wyniki zestawiono z mniejszym, wyspecjalizowanym modelem T5-large,. 

    "sentiment-analysis.pdf": Ten przegląd literatury omawia szerokie spektrum architektur głębokiego uczenia stosowanych w analizie sentymentu, w tym CNN, RNN, LSTM, GRU, BERT, GNN oraz LLM,. 

    „emotion_classification.pdf”: Opisuje dwuetapowy proces, w którym zaawansowany model LLaMA-3 generuje wyjaśnienia kontekstowe dla niejednoznacznych wypowiedzi, którymi dostrajany jest klasyfikator RoBERTa. (https://arxiv.org/pdf/2502.19935) 

    „sa-in-llm-era-reality-check.pdf”: Test porównawczy, zestawiający nowoczesne modele generatywne z mniejszymi, wyspecjalizowanymi modelami w różnych zadaniach z dziedziny analizy sentymentu. (https://arxiv.org/pdf/2305.15005) 
 
    “sa-in-age-of-genai.pdf”: Zawiera porównanie skuteczności modeli LLM i modeli transformerowych (Encoder-only) na różnych zbiorach danych oraz przedstawione wyniki przy zastosowaniu różnych technik użycia LLM (np. zero-shot, few-shot). (https://link.springer.com/article/10.1007/s40547-024-00143-4) 
---

## 2. Charakterystyka Zbiorów Danych

* **Twitter (4 klasy):** Krótkie teksty (pozytywne, negatywne, neutralne, nieistotne). Wysoki poziom szumu, emotikony, slang.
* **IMDB:** Długie recenzje filmowe (binarne: pos/neg). Styl narracyjny, rozbudowana argumentacja.
* **SemEval Twitter (Irony):** Detekcja ironii. Kluczowe dla badania zjawisk odwracających sentyment.
* **SemEval Twitter (Sentiment):** Klasyczny benchmark (3 klasy). Czyste dane referencyjne.
* **Yelp Reviews:** Skala 1-5 gwiazdek. Średnia długość, analiza wieloklasowa i aspektowa.
* **Financial PhraseBank:** Sentyment w raportach finansowych. Formalny język, subtelne komunikaty rynkowe.

---

## 3. Modele i Metody

Badanie obejmuje przekrój technologii od metod tradycyjnych po najnowsze modele generatywne:

### A. Metody Klasyczne i Płytkie Sieci

* **Słownikowe:** Analiza na podstawie leksykonów emocji.
* **Tradycyjne ML:** Modele SVM i Naive Bayes oparte na reprezentacjach TF-IDF oraz Bag-of-Words.
* **Deep Learning:** Sieci CNN i LSTM z wykorzystaniem embeddingów statycznych (GloVe).

### B. Architektury Transformerowe (Encoder-only)

* **Baseline:** BERT.
* **Zoptymalizowane:** RoBERTa, SiEBERT (skalibrowany pod analizę sentymentu).

### C. Large & Small Language Models (LLM/SLM)

* **API (Cloud):** Google Gemini 3 (via Google Cloud Vertex AI).
* **Lokalne (SLM):**
* **Llama 3.2** (warianty 1B/3B).
* **T5** (dostosowany do mocy obliczeniowej).



---

## 4. Plan Realizacji Projektu

### Faza I: Przygotowanie i Preprocessing

1. Unifikacja formatów wybranych zbiorów danych.
2. Czyszczenie tekstu (usuwanie szumu) przy zachowaniu elementów istotnych dla sentymentu (np. emotikony).
3. Podział na zbiory treningowe, walidacyjne i testowe.

### Faza II: Ekstrakcja Cech i Reprezentacja

* **Słowniki:** Zliczanie tokenów emocjonalnych.
* **ML:** Generowanie macierzy rzadkich (TF-IDF).
* **DL/Transformers:** Tokenizacja dedykowana dla konkretnych architektur.
* **LLM:** Inżynieria promptów (Prompt Engineering) dla scenariuszy Zero-shot i Few-shot.

### Faza III: Budowa i Trenowanie

* Trening klasyfikatorów bazowych (SVM, NB).
* Fine-tuning modeli BERT/RoBERTa na wybranych zbiorach.
* Konfiguracja lokalnych modeli SLM (Llama 3.2) przy użyciu kwantyzacji.

### Faza IV: Eksperymenty Hybrydowe

Eksperymentowanie z zaawansowanymi strategiami łączenia modeli:

* **Kaskada:** Jeśli pewność (confidence score) modelu mniejszego jest niska  zapytanie do LLM.
* **Voting:** Konsensus między różnymi architekturami.
* **Refinement:** model podstawowy wykonuje wstępną klasyfikację, LLM weryfikuje i podejmuje ostateczną decyzję 
* Model slm „wyjaśnia” emocje zawarte w tekście żeby potem model typu BERT miał prostsze zadanie przy klasyfikacji 

---

## 5. Ewaluacja i Metryki

Celem końcowym jest stworzenie **Macierzy Decyzyjnej**, która pozwoli określić optymalny stosunek kosztów (czas, moc obliczeniowa, cena API) do jakości wyników.

> **Główne metryki:** Accuracy, F1-Score (szczególnie dla klas niezbalansowanych), Latency (opóźnienie), Inference Cost.

---