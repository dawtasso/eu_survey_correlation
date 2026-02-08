"""Export validated survey-vote matches to JSON for the website.

Usage:
    python backend/scripts/export_matches_json.py
    python backend/scripts/export_matches_json.py --output custom_path.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
INPUT_CSV = DATA_DIR / "matches" / "survey_vote_matches_judged.csv"
OUTPUT_JSON = DATA_DIR / "matches" / "matches.json"


def transform_matches(input_path: Path, output_path: Path) -> dict:
    """
    Read judged matches CSV, filter to validated pairs, and export as JSON.

    Args:
        input_path: Path to survey_vote_matches_judged.csv
        output_path: Path to write matches.json

    Returns:
        The exported data structure.
    """
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} total matches from {input_path}")

    # Filter to only validated matches (llm_go == True)
    validated = df[df["llm_go"] == True].copy()
    print(f"Filtered to {len(validated)} validated matches (llm_go=True)")

    # Build clean match records
    matches = []
    for _, row in validated.iterrows():
        matches.append(
            {
                "question_id": str(row["question_id"]),
                "question_text": str(row["question_text"]),
                "file_name": str(row["file_name"]),
                "vote_id": str(row["vote_id"]),
                "vote_summary": str(row["vote_summary"]),
                "similarity_score": round(float(row["similarity_score"]), 4),
                "llm_score": int(row["llm_score"]),
                "llm_explanation": str(row["llm_explanation"]),
            }
        )

    # Build output structure with metadata
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(input_path),
        "total_matches": len(matches),
        "matches": matches,
    }

    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(matches)} matches to {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Export matches to JSON")
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_CSV,
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_JSON,
        help="Output JSON path",
    )
    args = parser.parse_args()

    transform_matches(args.input, args.output)


if __name__ == "__main__":
    main()
