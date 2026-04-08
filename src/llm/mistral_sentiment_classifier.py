"""
Mistral sentiment classification on Vertex AI using rawPredict endpoint.
"""

import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import google.auth
import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
REGION = os.getenv("MISTRAL_VERTEX_LOCATION", "us-central1")
MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-2503")

if not PROJECT_ID:
    raise ValueError("Missing GOOGLE_CLOUD_PROJECT environment variable for Vertex AI")


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
        instruction = "Analyze sentiment and return numeric labels."

    items_text = ""
    for i, item in enumerate(batch):
        item_index = indices[i] if indices is not None else start_index + i
        items_text += f"{item_index}. {item.get('text', '')}\n\n"

        json_contract = """
Return ONLY valid JSON in this exact shape:
{
    "classifications": [
        {"index": 0, "sentiment": 0}
    ]
}
Rules:
- Do not return text.
- index must point to the numbered item above.
- sentiment must be an integer label.
- Do not add markdown, comments, or extra keys.
"""

    return f"""{instruction}

{json_contract}
Texts to analyze:
{items_text}"""


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
        raise ValueError("No JSON object found in model response")

    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(_fix_invalid_escapes(candidate))


def _get_access_token() -> str:
    # Prefer Application Default Credentials to avoid requiring gcloud in PATH.
    try:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(Request())
        if credentials.token:
            return credentials.token
    except Exception:
        pass

    try:
        proc = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            check=True,
            capture_output=True,
            text=True,
        )
        token = proc.stdout.strip()
        if token:
            return token
    except FileNotFoundError as e:
        raise RuntimeError(
            "Cannot get access token: gcloud is not available in PATH and ADC token refresh failed. "
            "Run 'gcloud auth application-default login' or ensure gcloud is installed and available."
        ) from e

    raise RuntimeError(
        "Cannot get access token for Vertex AI. Run 'gcloud auth application-default login' "
        "or 'gcloud auth print-access-token' to verify local auth setup."
    )


def _extract_response_text(resp_json: dict) -> str:
    # Common OpenAI-like schema.
    choices = resp_json.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("content") or ""
                    if text:
                        parts.append(text)
            if parts:
                return "\n".join(parts).strip()

    # Alternate response shapes.
    outputs = resp_json.get("outputs", [])
    if outputs and isinstance(outputs[0], dict):
        maybe_text = outputs[0].get("text") or outputs[0].get("output_text")
        if isinstance(maybe_text, str) and maybe_text.strip():
            return maybe_text

    for key in ("text", "output_text", "response"):
        value = resp_json.get(key)
        if isinstance(value, str) and value.strip():
            return value

    raise ValueError(f"Unable to extract text from response: {resp_json}")


def classify_batch(
    batch: list[dict],
    start_index: int = 0,
    instruction: str = None,
    indices: Optional[list[int]] = None,
) -> list[dict]:
    prompt = create_batch_prompt(batch, start_index, instruction, indices)
    url = (
        f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/"
        f"locations/{REGION}/publishers/mistralai/models/{MODEL}:rawPredict"
    )

    payload = {
        "model": MODEL,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ],
    }

    token = _get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

    response_json = response.json()
    response_text = _extract_response_text(response_json)
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
    print(f"Loaded {total} samples | batch_size={batch_size} | workers={max_workers}")
    print(f"Using Mistral model: {MODEL} in {REGION}\n")

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
                "original_index": batch_idx + start_index + offset,
                "text": item.get("text", ""),
                "sentiment": 404,
            }
            for offset, item in enumerate(batch)
        ]

    def to_int(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            v = value.strip()
            if v == "":
                return None
            try:
                return int(v)
            except ValueError:
                return None
        return None

    def build_output_rows(batch_idx, batch, classifications):
        base_index = batch_idx + start_index
        expected_indices = [base_index + i for i in range(len(batch))]
        expected_set = set(expected_indices)

        sentiment_by_index = {}
        ordered_sentiments = []

        for item in classifications:
            if not isinstance(item, dict):
                continue

            sentiment = to_int(item.get("sentiment"))
            if sentiment is not None:
                ordered_sentiments.append(sentiment)

            raw_index = to_int(item.get("index"))
            if raw_index is None or sentiment is None:
                continue

            # Prefer exact global index; accept local (0..batch-1) as fallback.
            if raw_index in expected_set:
                sentiment_by_index[raw_index] = sentiment
            elif 0 <= raw_index < len(batch):
                sentiment_by_index[base_index + raw_index] = sentiment

        rows = []
        for offset, source_item in enumerate(batch):
            original_index = base_index + offset
            sentiment = sentiment_by_index.get(original_index)
            if sentiment is None and offset < len(ordered_sentiments):
                sentiment = ordered_sentiments[offset]
            if sentiment is None:
                sentiment = 404

            rows.append(
                {
                    "original_index": original_index,
                    "text": source_item.get("text", ""),
                    "sentiment": sentiment,
                }
            )

        return rows

    def process_batch(batch_idx, batch):
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                classifications = classify_batch(batch, batch_idx + start_index, instruction)
                if not isinstance(classifications, list):
                    raise ValueError("Invalid response: classifications is not a list")
                return batch_idx, build_output_rows(batch_idx, batch, classifications)
            except Exception as e:
                last_error = e
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
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nClassification complete! Results saved to {output_file}")
