"""Rewrite survey questions into clean, readable form using a local LLM.

Strips question IDs, base filters, instruction boilerplate, and reformats
into concise, human-readable questions.

Usage:
    python backend/scripts/simplify_questions.py
    python backend/scripts/simplify_questions.py --input data/surveys/filtered_survey_questions.csv
"""

import argparse
import json
import re
from pathlib import Path

import ollama
import pandas as pd
from loguru import logger
from tqdm import tqdm

DATA_DIR = Path("data")
DEFAULT_INPUT = DATA_DIR / "surveys" / "filtered_survey_questions.csv"
DEFAULT_OUTPUT = DATA_DIR / "surveys" / "filtered_survey_questions_clean.csv"

MODEL = "mistral"

SIMPLIFY_PROMPT = """\
Rewrite this survey question into a short, clear, easy-to-read question.

Rules:
- Remove any question ID prefix (e.g. "QA2.1", "QB3_2", "Q1_1")
- Remove "Base: ..." filters and technical instructions
- Remove "MULTIPLE ANSWERS POSSIBLE" and similar instructions
- Remove "For each of the following..." preambles when possible — keep only the specific item
- Keep the core meaning intact
- Output ONLY the rewritten question, nothing else
- If the question asks about a specific sub-item (after a colon or dash), include both the main topic and the sub-item
- Maximum 1-2 sentences
- Do NOT answer the question, just rewrite it

Examples:
Input: "QA2.1. To what extent do you agree or disagree that each of the following actors is doing enough to ensure that the green transition is fair? :-Your regional or local authorities"
Output: Are your regional or local authorities doing enough to ensure a fair green transition?

Input: "QB6.4. What is your opinion on each of the following statements? Please tell for each statement, whether you are for it or against it. :-A common defence and security policy among EU Member States"
Output: Are you for or against a common defence and security policy among EU Member States?

Input: "Q5_3 To what extent do you agree or disagree with each of the following statements? Everyone should get vaccinated against COVID-19 | Base: All"
Output: Should everyone get vaccinated against COVID-19?

Now rewrite this question:
{question}"""


def simplify_question(question: str) -> str:
    """Call Ollama to simplify a single question."""
    prompt = SIMPLIFY_PROMPT.format(question=question)
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        result = response["message"]["content"].strip()
        # Remove quotes if the LLM wrapped it
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        return result
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return question  # fallback to original


def main():
    parser = argparse.ArgumentParser(description="Simplify survey question text")
    parser.add_argument(
        "--input", type=str, default=str(DEFAULT_INPUT), help="Input CSV"
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT), help="Output CSV"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    logger.info(f"Loading questions from {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} questions")

    # Resume support: if output already exists, skip already-processed rows
    start_idx = 0
    if output_path.exists():
        existing = pd.read_csv(output_path)
        if "question_clean" in existing.columns:
            start_idx = len(existing)
            logger.info(f"Resuming from row {start_idx} ({start_idx} already done)")

    # Process questions
    clean_questions = []
    if start_idx > 0:
        existing = pd.read_csv(output_path)
        clean_questions = existing["question_clean"].tolist()

    for idx in tqdm(range(start_idx, len(df)), desc="Simplifying questions"):
        row = df.iloc[idx]
        original = str(row["question_en"])
        simplified = simplify_question(original)
        clean_questions.append(simplified)

        # Save incrementally every 50 questions
        if (idx + 1) % 50 == 0 or idx == len(df) - 1:
            df_out = df.iloc[: idx + 1].copy()
            df_out["question_clean"] = clean_questions
            df_out.to_csv(output_path, index=False)

    # Final save
    df["question_clean"] = clean_questions
    df.to_csv(output_path, index=False)
    logger.success(f"Saved {len(df)} simplified questions → {output_path}")

    # Show samples
    logger.info("Samples:")
    for _, row in df.sample(min(10, len(df)), random_state=42).iterrows():
        logger.info(f"  ORIGINAL: {str(row.question_en)[:100]}")
        logger.info(f"  CLEAN:    {row.question_clean}")
        logger.info("")


if __name__ == "__main__":
    main()
