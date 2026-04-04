#!/bin/bash

# List of datasets and their paths
# Format: "name:path"
DATASETS=(
    # "tweet_eval_irony:TweetEvalIrony/tweeteval_irony.parquet"
    # "tweet_eval_sentiment:TweetEvalSentiment/tweeteval_sentiment.parquet"
    "twitter:Twitter/twitter.parquet"
    "imdb:IMDB/imdb.parquet"
    # "yelp_test:Yelp/yelp_test.parquet"
    # "yelp_train:Yelp/yelp_train.parquet"
    # "finance_50agree:FinancialPhraseBank/financial_phrasebank_50Agree.parquet"
    # "finance_66agree:FinancialPhraseBank/financial_phrasebank_66Agree.parquet"
    # "finance_75agree:FinancialPhraseBank/financial_phrasebank_75Agree.parquet"
    # "finance_all_agree:FinancialPhraseBank/financial_phrasebank_AllAgree.parquet"
)

# List of models
MODELS=("distilroberta-base" "distilbert-base-uncased")

for ds_info in "${DATASETS[@]}"; do
    IFS=":" read -r NAME DS_PATH <<< "$ds_info"

    for MODEL in "${MODELS[@]}"; do
        echo "------------------------------------------------"
        echo "STARTING: $MODEL on $NAME"
        echo "------------------------------------------------"

        # We use DS_PATH here
        python3 run_single_transformer.py --dataset "$NAME" --path "$DS_PATH" --model "$MODEL"

        echo "Pausing for 5 seconds to let M4 cool down..."
        sleep 5
    done
done
