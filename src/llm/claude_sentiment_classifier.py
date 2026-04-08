"""
Claude Sonnet sentiment classification on Vertex AI.
Compatible workflow with the existing Gemini classifier.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from anthropic import AnthropicVertex
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("ANTHROPIC_VERTEX_LOCATION", "us-east5")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

if not PROJECT_ID:
    raise ValueError("Missing GOOGLE_CLOUD_PROJECT environment variable for Vertex AI")

client = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)


def _normalize_model_name(model_name: str) -> str:
    """Normalize common Anthropic model aliases to Vertex partner model IDs."""
    aliases = {
        "claude-sonnet-4@20250514": "claude-sonnet-4",
        "claude-sonnet-4.5@20250514": "claude-sonnet-4-5",
        "claude-sonnet-4.5": "claude-sonnet-4-5",
        "publishers/anthropic/models/claude-sonnet-4": "claude-sonnet-4",
        "publishers/anthropic/models/claude-sonnet-4-5": "claude-sonnet-4-5",
    }
    return aliases.get(model_name, model_name)


def load_jsonl(file_path: str) -> list[dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_json(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(file_path: str) -> list[dict]:
    if file_path.endswith(".jsonl"):
        return load_jsonl(file_path)
    if file_path.endswith(".json"):
        return load_json(file_path)
    raise ValueError(f"Unsupported file format: {file_path}")


def create_batch_prompt(
    batch: list[dict],
    start_index: int,
    instruction: str = None,
    indices: Optional[list[int]] = None,
) -> str:
    if instruction is None:
        instruction = """Analyze sentiment and return numeric labels."""

    items_text = ""
    for i, item in enumerate(batch):
        item_index = indices[i] if indices is not None else start_index + i
        items_text += f"{item_index}. {item.get('text', '')}\n\n"

    json_contract = """
Return ONLY valid JSON in this exact shape:
{
  "classifications": [
    {"index": 0, "text": "...", "sentiment": 0}
  ]
}
Do not add markdown, comments, or extra keys.
"""

    return f"""{instruction}

{json_contract}
Texts to analyze:
{items_text}"""


def _extract_text_from_message(message) -> str:
    parts = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "\n".join(part for part in parts if part).strip()


def _fix_invalid_escapes(s: str) -> str:
    """Replace backslashes not followed by a valid JSON escape character."""
    return re.sub(r'\\(?!["\\\x2F bfnrtu])', r'\\\\', s)


def _extract_json_object(text: str) -> dict:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try fixing invalid escape sequences and parse again.
    try:
        return json.loads(_fix_invalid_escapes(cleaned))
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Claude response")

    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(_fix_invalid_escapes(candidate))


def classify_batch(
    batch: list[dict],
    start_index: int = 0,
    instruction: str = None,
    indices: Optional[list[int]] = None,
) -> list[dict]:
    prompt = create_batch_prompt(batch, start_index, instruction, indices)
    model_name = _normalize_model_name(MODEL)

    message = client.messages.create(
        model=model_name,
        max_tokens=4096,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    response_text = _extract_text_from_message(message)
    if not response_text:
        raise ValueError("Claude returned empty response text")

    parsed = _extract_json_object(response_text)
    classifications = parsed.get("classifications", [])
    if not isinstance(classifications, list):
        raise ValueError("Invalid response: 'classifications' must be a list")

    return classifications


def classify_dataset(
    input_file: str,
    output_file: str,
    batch_size: int = 10,
    start_index: int = 0,
    max_samples: Optional[int] = None,
    dataset_key: str = None,
    max_workers: int = 1,
    max_retries: int = 3,
    retry_base_delay: float = 2.0,
):
    instruction = None
    if dataset_key:
        try:
            from gemini_prompts import get_instruction

            instruction = get_instruction(dataset_key)
        except Exception:
            pass

    print(f"Loading dataset from {input_file}...")
    data = load_dataset(input_file)

    if max_samples:
        data = data[:max_samples]
    if start_index > 0:
        data = data[start_index:]

    total = len(data)
    print(f"Loaded {total} samples | batch_size={batch_size} | workers={max_workers}\n")
    print(f"Using Claude model: {_normalize_model_name(MODEL)} in region {LOCATION}\n")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batches = [
        (batch_idx, data[batch_idx : min(batch_idx + batch_size, total)])
        for batch_idx in range(0, total, batch_size)
    ]
    total_batches = len(batches)
    batch_results = {}

    def create_404_results(batch_idx, batch):
        return [
            {
                "index": batch_idx + start_index + offset,
                "text": item.get("text", ""),
                "sentiment": 404,
            }
            for offset, item in enumerate(batch)
        ]

    def process_batch(batch_idx, batch):
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                classifications = classify_batch(batch, batch_idx + start_index, instruction)
                if len(classifications) != len(batch):
                    raise ValueError(
                        f"Expected {len(batch)} classifications, got {len(classifications)}"
                    )
                return batch_idx, classifications
            except Exception as e:
                last_error = e
                error_text = str(e)
                if "Error code: 404" in error_text or "status': 'NOT_FOUND'" in error_text:
                    raise RuntimeError(
                        "Claude model not found or no access. "
                        "Use ANTHROPIC_MODEL=claude-sonnet-4-5 (or another valid Vertex Claude model ID) "
                        "and enable the model in Vertex AI Model Garden for this project. "
                        f"Original error: {error_text}"
                    )
                if attempt < max_retries:
                    delay = retry_base_delay * attempt
                    print(
                        f"Batch {batch_idx // batch_size + 1}/{total_batches} retry {attempt}/{max_retries - 1} in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
        raise last_error

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_batch, batch_idx, batch): (batch_idx, batch)
            for batch_idx, batch in batches
        }
        for future in as_completed(futures):
            batch_idx, batch = futures[future]
            batch_num = batch_idx // batch_size + 1
            try:
                _, classifications = future.result()
                batch_results[batch_idx] = classifications
                print(f"Batch {batch_num}/{total_batches} completed")
            except Exception as e:
                print(f"Batch {batch_num}/{total_batches} error: {e}")
                batch_results[batch_idx] = create_404_results(batch_idx, batch)
                print(f"Batch {batch_num}/{total_batches} saved with sentiment=404")

    with open(output_path, "w", encoding="utf-8") as fout:
        for batch_idx in sorted(batch_results.keys()):
            for item in batch_results[batch_idx]:
                output_line = {
                    "text": item.get("text", ""),
                    "sentiment": item.get("sentiment", ""),
                    "original_index": item.get("index"),
                }
                fout.write(json.dumps(output_line, ensure_ascii=False) + "\n")

    print(f"\nClassification complete! Results saved to {output_file}")
