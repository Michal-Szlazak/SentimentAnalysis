# SentimentAnalysis

Projekt analizy sentymentu z wykorzystaniem modeli językowych (LLM) oraz metod klasycznych ML.

## Instalacja

1. **Sklonuj repozytorium:**
```bash
git clone <repo-url>
cd SentimentAnalysis
```

2. **Zainstaluj zależności:**
```bash
pip install -r requirements.txt
```

3. **Skonfiguruj API klucze:**
   - Utwórz plik `.env` w głównym folderze
   - Dodaj klucz API Gemini: `GEMINI_API_KEY=your_api_key_here`

## Użycie

### 1. Przygotowanie danych

Pobierz i przygotuj podzbiory danych:

```bash
python src/utils/create_subsets.py
```

To polecenie:
- Pobierze zbiory danych z HuggingFace (tylko raz, potem używa cache)
- Przygotuje zrównoważone podzbiory testowe (zachowując proporcje klas)
- Utworzy przykłady do promptów

### 2. Analiza sentymentu z Gemini

#### Opcja A: Szybka analiza (interaktywna)
```bash
python analyze_sentiment.py
```
⚡ **Szybka** - wyniki natychmiastowe
💰 **Droższa** - pełna cena API

#### Opcja B: Batch API (asynchroniczna)
```bash
python analyze_batch_sentiment.py
```
⏰ **Asynchroniczna** - do 24h na przetworzenie
💰 **Taniej** - 50% ceny standardowej API
📊 **Skalowalna** - lepsze dla dużych zbiorów

#### Opcja C: Liczenie tokenów
```bash
python count_tokens.py
```
🔢 **Liczy tokeny** - sprawdź koszty przed analizą
📊 **Statystyki** - średnia, max, min tokenów

### 3. Wyniki

Po uruchomieniu analizy, wyniki będą dostępne w odpowiednich folderach:
- **Interaktywna**: `results/` - natychmiastowe wyniki
- **Batch**: `batch_results/` - wyniki po zakończeniu zadań

## Struktura projektu

```
SentimentAnalysis/
├── data_cache/           # Cache pobranych zbiorów danych
├── subsets/              # Podzbiory do analizy
├── results/              # Wyniki analizy interaktywnej
├── batch_results/        # Wyniki analizy Batch API
├── src/
│   ├── classic/          # Metody klasyczne (VADER, ML)
│   ├── utils/            # Narzędzia pomocnicze
│   │   ├── create_subsets.py           # Przygotowanie podzbiorów
│   │   ├── data_downloader.py          # Pobieranie/cache danych
│   │   ├── dataset_loader.py           # Ładowanie zbiorów
│   │   ├── gemini_api.py               # Konfiguracja Gemini
│   │   ├── gemini_sentiment_analyzer.py    # Analiza interaktywna
│   │   └── gemini_batch_analyzer.py        # Analiza Batch API
│   └── classic_methods.py  # Główny skrypt metod klasycznych
├── analyze_sentiment.py          # Główny skrypt analizy interaktywnej
├── analyze_batch_sentiment.py    # Główny skrypt analizy Batch API
├── requirements.txt      # Zależności Python
└── .env                  # Klucze API (nie commitować!)
```

## Zbiory danych

Projekt używa następujących zbiorów danych:
- **IMDB**: Recenzje filmowe (binarne: neg/pos) - https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data
- **Twitter**: Tweety z sentymentem - https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
- **Yelp**: Recenzje różnego rodzaju biznesów (np. restauracje, gabinety dentystyczne) - https://huggingface.co/datasets/Yelp/yelp_review_full
- **TweetEval**: Benchmark dla analizy sentymentu - https://huggingface.co/datasets/cardiffnlp/tweet_eval/tree/main/sentiment
- **FinancialPhraseBank**: Raporty finansowe - https://huggingface.co/datasets/takala/financial_phrasebank

## Uwagi

- **Batch API**: Asynchroniczne przetwarzanie do 24h, tańsze o 50%
- **Cache**: Dane są cache'owane lokalnie po pierwszym pobraniu
- **Rate limiting**: Interaktywna analiza ma przerwy między requestami