"""
Convert JSONL files to CSV.

Supports:
- a single input file
- a whole directory of JSONL files

Examples:
    python jsonl_to_csv.py results_gemini/finance_50agree_gemini_classified_full.jsonl
    python jsonl_to_csv.py results_gemini --output-dir csv_results_gemini
    python jsonl_to_csv.py results_mistral --pattern "*_classified_full.jsonl"
"""

import argparse
import csv
import json
from pathlib import Path


def collect_fieldnames(rows: list[dict]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_no}")
            rows.append(record)
    return rows


def convert_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    rows = load_jsonl(input_path)
    fieldnames = collect_fieldnames(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), len(fieldnames)


def resolve_inputs(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob(pattern))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def get_output_path(input_file: Path, source_root: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return input_file.with_suffix(".csv")
    if source_root.is_file():
        return output_dir / input_file.with_suffix(".csv").name
    relative = input_file.relative_to(source_root)
    return output_dir / relative.with_suffix(".csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSONL file(s) to CSV.")
    parser.add_argument("input", help="Input JSONL file or directory containing JSONL files")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Default: write .csv next to each input file.",
    )
    parser.add_argument(
        "--pattern",
        default="*.jsonl",
        help="Glob pattern when input is a directory (default: *.jsonl)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None

    input_files = resolve_inputs(input_path, args.pattern)
    if not input_files:
        raise SystemExit(f"No files matched: {input_path} ({args.pattern})")

    print(f"Input:  {input_path}")
    if output_dir:
        print(f"Output: {output_dir}")
    print(f"Files:  {len(input_files)}")

    for input_file in input_files:
        output_path = get_output_path(input_file, input_path, output_dir)
        row_count, col_count = convert_file(input_file, output_path)
        print(f"Converted: {input_file} -> {output_path} | rows={row_count} cols={col_count}")


if __name__ == "__main__":
    main()