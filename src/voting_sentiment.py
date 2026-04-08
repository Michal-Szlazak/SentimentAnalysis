"""
Build final sentiment results with majority voting across 3 LLM models.

Input (per dataset):
- results_gemini/<dataset>_gemini_classified_full.jsonl
- results_mistral/<dataset>_mistral_classified_full.jsonl
- results_llama_reindexed/<dataset>_llama_classified_full.jsonl (fallback: results_llama)

Output:
- results_voting/<dataset>_voting_classified_full.jsonl

Voting rule:
- Use only valid integer sentiments different from 404.
- If one value has strictly more votes than others, use it.
- If there is no strict majority, use tie-breaker priority: Gemini > Mistral > Llama.
- Output 404 only when none of the models has a valid sentiment.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


DATASET_MAP = {
    "twitter": "twitter_full.jsonl",
    "imdb": "imdb_full.jsonl",
    "finance_50agree": "finance_50agree_full.jsonl",
    "finance_66agree": "finance_66agree_full.jsonl",
    "finance_75agree": "finance_75agree_full.jsonl",
    "finance_all_agree": "finance_all_agree_full.jsonl",
    "yelp_full": "yelp_full_full.jsonl",
    "tweet_eval_sentiment": "tweet_eval_sentiment_full.jsonl",
    "tweet_eval_irony": "tweet_eval_irony_full.jsonl",
}

MODELS = ("gemini", "mistral", "llama")
TIE_BREAK_PRIORITY = ("gemini", "llama", "mistral")


def default_llama_dir(base_dir: Path) -> Path:
    reindexed = base_dir / "results_llama_reindexed"
    classic = base_dir / "results_llama"
    return reindexed if reindexed.exists() else classic


def result_file_path(results_dir: Path, dataset_key: str, model: str) -> Path:
    return results_dir / f"{dataset_key}_{model}_classified_full.jsonl"


def parse_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def load_predictions(path: Path) -> dict[int, dict]:
    """Return {original_index: {'sentiment': int|None, 'text': str}}."""
    by_index = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            idx = parse_int(rec.get("original_index"))
            if idx is None:
                continue
            by_index[idx] = {
                "sentiment": parse_int(rec.get("sentiment")),
                "text": rec.get("text", ""),
            }
    return by_index


def vote_sentiment(sentiment_by_model: dict[str, int | None]) -> tuple[int, bool]:
    """Return (final_sentiment, tie_resolved_by_priority)."""
    valid = [
        v for v in sentiment_by_model.values() if isinstance(v, int) and v != 404
    ]
    if not valid:
        return 404, False

    counts = Counter(valid)
    top_value, top_count = counts.most_common(1)[0]
    tied = [label for label, c in counts.items() if c == top_count]

    if len(tied) == 1:
        return top_value, False

    tied_set = set(tied)
    for model in TIE_BREAK_PRIORITY:
        value = sentiment_by_model.get(model)
        if isinstance(value, int) and value != 404 and value in tied_set:
            return value, True

    return 404, False


def choose_text(*candidates: dict | None) -> str:
    for item in candidates:
        if item and item.get("text"):
            return item["text"]
    return ""


def process_dataset(
    dataset_key: str,
    gemini_dir: Path,
    mistral_dir: Path,
    llama_dir: Path,
    output_dir: Path,
) -> dict:
    files = {
        "gemini": result_file_path(gemini_dir, dataset_key, "gemini"),
        "mistral": result_file_path(mistral_dir, dataset_key, "mistral"),
        "llama": result_file_path(llama_dir, dataset_key, "llama"),
    }

    missing_files = [f"{m}:{p}" for m, p in files.items() if not p.exists()]
    if missing_files:
        return {
            "dataset": dataset_key,
            "error": f"Missing model result files: {', '.join(missing_files)}",
        }

    pred = {model: load_predictions(path) for model, path in files.items()}

    all_indices = sorted(
        set(pred["gemini"].keys()) | set(pred["mistral"].keys()) | set(pred["llama"].keys())
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{dataset_key}_voting_classified_full.jsonl"

    tie_break_rows = 0
    blocked_rows = 0
    wrote_rows = 0

    with open(out_file, "w", encoding="utf-8") as f:
        for idx in all_indices:
            g = pred["gemini"].get(idx)
            m = pred["mistral"].get(idx)
            l = pred["llama"].get(idx)

            sentiment_by_model = {
                "gemini": g["sentiment"] if g else None,
                "mistral": m["sentiment"] if m else None,
                "llama": l["sentiment"] if l else None,
            }
            final_sentiment, used_tie_break = vote_sentiment(sentiment_by_model)

            if used_tie_break:
                tie_break_rows += 1
            if final_sentiment == 404:
                blocked_rows += 1

            output_row = {
                "text": choose_text(g, m, l),
                "sentiment": final_sentiment,
                "original_index": idx,
            }
            f.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            wrote_rows += 1

    return {
        "dataset": dataset_key,
        "output_file": str(out_file),
        "rows": wrote_rows,
        "blocked_404": blocked_rows,
        "ties_resolved_by_priority": tie_break_rows,
    }


def main() -> None:
    base_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Create final sentiment via majority voting (gemini/mistral/llama)."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=list(DATASET_MAP.keys()),
        help=f"Datasets to process (default: all). Choices: {', '.join(DATASET_MAP.keys())}",
    )
    parser.add_argument(
        "--results-gemini-dir",
        default=str(base_dir / "results_gemini_reindexed"),
        help="Directory with *_gemini_classified_full.jsonl files",
    )
    parser.add_argument(
        "--results-mistral-dir",
        default=str(base_dir / "results_mistral"),
        help="Directory with *_mistral_classified_full.jsonl files",
    )
    parser.add_argument(
        "--results-llama-dir",
        default=None,
        help="Directory with *_llama_classified_full.jsonl files. Default: results_llama_reindexed if exists, else results_llama.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(base_dir / "results_voting"),
        help="Output directory for *_voting_classified_full.jsonl files",
    )

    args = parser.parse_args()

    invalid = [d for d in args.datasets if d not in DATASET_MAP]
    if invalid:
        parser.error(f"Unknown dataset(s): {invalid}. Choose from: {', '.join(DATASET_MAP.keys())}")

    gemini_dir = Path(args.results_gemini_dir)
    mistral_dir = Path(args.results_mistral_dir)
    llama_dir = Path(args.results_llama_dir) if args.results_llama_dir else default_llama_dir(base_dir)
    output_dir = Path(args.output_dir)

    print(f"Gemini dir:  {gemini_dir}")
    print(f"Mistral dir: {mistral_dir}")
    print(f"Llama dir:   {llama_dir}")
    print(f"Output dir:  {output_dir}")

    for dataset in args.datasets:
        report = process_dataset(dataset, gemini_dir, mistral_dir, llama_dir, output_dir)

        print("\n" + "-" * 90)
        print(f"Dataset: {dataset}")

        if "error" in report:
            print(f"Error: {report['error']}")
            continue

        print(f"Output:             {report['output_file']}")
        print(f"Rows:               {report['rows']}")
        print(f"Blocked final 404:  {report['blocked_404']}")
        print(f"Ties resolved:      {report['ties_resolved_by_priority']}")


if __name__ == "__main__":
    main()
