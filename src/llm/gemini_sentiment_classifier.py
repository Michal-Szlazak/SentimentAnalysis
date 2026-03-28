"""
Gemini API Sentiment Classification with Structured Output
Sends batches of texts to Gemini and saves results with sentiment labels.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai.types import Tool

# Load environment variables
load_dotenv()

# Initialize client
client = genai.Client()
MODEL = "gemini-2.5-flash"

# Define structured output schema
SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "Original index in the batch"},
                    "text": {"type": "string", "description": "The analyzed text"},
                    "sentiment": {"type": "number", "description": "Assigned sentiment label as number"}
                },
                "required": ["index", "text", "sentiment"]
            }
        }
    },
    "required": ["classifications"]
}

def load_jsonl(file_path: str) -> list[dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_json(file_path: str) -> list[dict]:
    """Load data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_dataset(file_path: str) -> list[dict]:
    """Load dataset from either JSONL or JSON file."""
    if file_path.endswith('.jsonl'):
        return load_jsonl(file_path)
    elif file_path.endswith('.json'):
        return load_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def create_batch_prompt(batch: list[dict], start_index: int, instruction: str = None) -> str:
    """Create a prompt for batch sentiment classification."""
    if instruction is None:
        instruction = """Analyze the sentiment of the following texts. For each text, assign one of these sentiment labels:
- positive
- negative
- neutral

Return your classification as JSON with the exact structure specified."""
    
    items_text = ""
    for i, item in enumerate(batch):
        items_text += f"{start_index + i}. {item.get('text', '')}\n\n"
    
    prompt = f"""{instruction}

Texts to analyze:
{items_text}"""
    
    return prompt

def classify_batch(batch: list[dict], start_index: int = 0, instruction: str = None) -> list[dict]:
    """
    Send a batch of texts to Gemini for sentiment classification.
    
    Args:
        batch: List of dicts with 'text' key
        start_index: Starting index for this batch
        
    Returns:
        List of dicts with 'text' and 'sentiment' keys
    """
    prompt = create_batch_prompt(batch, start_index, instruction)

    
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            response_schema=SENTIMENT_SCHEMA,
            response_mime_type="application/json",
            temperature=0.3,
        )
    )
    
    result = json.loads(response.text)
    return result["classifications"]

def classify_dataset(
    input_file: str,
    output_file: str,
    batch_size: int = 10,
    start_index: int = 0,
    max_samples: Optional[int] = None,
    dataset_key: str = None,
    max_workers: int = 1,
):
    instruction = None
    if dataset_key:
        try:
            from gemini_prompts import get_instruction
            instruction = get_instruction(dataset_key)
        except Exception:
            pass

    print(f"📂 Loading dataset from {input_file}...")
    data = load_dataset(input_file)

    if max_samples:
        data = data[:max_samples]
    if start_index > 0:
        data = data[start_index:]

    total = len(data)
    print(f"✅ Loaded {total} samples | batch_size={batch_size} | workers={max_workers}\n")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batches = [
        (batch_idx, data[batch_idx:min(batch_idx + batch_size, total)])
        for batch_idx in range(0, total, batch_size)
    ]
    total_batches = len(batches)
    batch_results = {}

    def process_batch(batch_idx, batch):
        classifications = classify_batch(batch, batch_idx + start_index, instruction)
        return batch_idx, classifications

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_batch, batch_idx, batch): batch_idx
            for batch_idx, batch in batches
        }
        for future in as_completed(futures):
            batch_idx = futures[future]
            batch_num = batch_idx // batch_size + 1
            try:
                _, classifications = future.result()
                batch_results[batch_idx] = classifications
                print(f"✅ Batch {batch_num}/{total_batches} completed")
            except Exception as e:
                print(f"❌ Batch {batch_num}/{total_batches} error: {e}")
                batch_results[batch_idx] = []

    with open(output_path, 'w', encoding='utf-8') as fout:
        for batch_idx in sorted(batch_results.keys()):
            for item in batch_results[batch_idx]:
                output_line = {
                    "text": item.get("text", ""),
                    "sentiment": item.get("sentiment", ""),
                    "original_index": item.get("index")
                }
                fout.write(json.dumps(output_line, ensure_ascii=False) + "\n")

    print(f"\n✨ Classification complete! Results saved to {output_file}")

# if __name__ == "__main__":
#     # Example usage
#     input_file = r"c:\\Users\\bushi\\Documents\\GitHub\\SentimentAnalysis\\src\\subsets\\imdb_test_stratified.jsonl"
#     output_file = r"c:\\Users\\bushi\\Documents\\GitHub\\SentimentAnalysis\\src\\subsets\\imdb_gemini_classified.jsonl"
    
#     classify_dataset(
#         input_file=input_file,
#         output_file=output_file,
#         batch_size=10,
#         max_samples=50  # Test with first 50 samples
#     )
