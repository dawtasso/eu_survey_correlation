"""Validate clean survey-vote matches using LLM judge.

Filters matches to keep only those where the survey question is genuinely
related to the parliament vote topic.

Usage:
    python backend/scripts/validate_clean_matches.py
"""

import json
import re
from pathlib import Path

import ollama
import pandas as pd
from loguru import logger
from tqdm import tqdm

DATA_DIR = Path("data")
INPUT_CSV = DATA_DIR / "matches" / "survey_vote_matches_clean.csv"
OUTPUT_CSV = DATA_DIR / "matches" / "survey_vote_matches_validated.csv"

MODEL = "mistral"

JUDGE_PROMPT = """\
You are an expert analyst. Determine if this EU survey question and this \
European Parliament vote are about the SAME specific policy topic.

SURVEY QUESTION:
{question}

VOTE:
{vote}

Respond ONLY with valid JSON:
{{"related": true/false, "explanation": "<one sentence>"}}

Rules:
- "related" = true ONLY if both are about the same specific topic \
(e.g. both about migration policy, both about vaccine regulation, \
both about agricultural subsidies, both about corruption, etc.)
- Generic similarity (both mention "EU" or "citizens") is NOT enough
- The survey must ask about something the vote actually legislates on"""


def judge_pair(question: str, vote: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, vote=vote)
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        content = response["message"]["content"].strip()
        # Parse JSON
        json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "related": bool(result.get("related", False)),
                "explanation": str(result.get("explanation", "")),
            }
    except Exception as e:
        logger.error(f"Judge failed: {e}")
    return {"related": False, "explanation": "parse error"}


def main():
    logger.info(f"Loading matches from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(df)} matches to validate")

    # Resume support
    start_idx = 0
    results = []
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        if "llm_related" in existing.columns:
            start_idx = len(existing)
            results = existing.to_dict("records")
            logger.info(f"Resuming from row {start_idx}")

    for idx in tqdm(range(start_idx, len(df)), desc="Validating"):
        row = df.iloc[idx]
        judgment = judge_pair(
            question=str(row["question_clean"]),
            vote=str(row["vote_summary_clean"]),
        )

        row_dict = row.to_dict()
        row_dict["llm_related"] = judgment["related"]
        row_dict["llm_explanation"] = judgment["explanation"]
        results.append(row_dict)

        # Save every 25 rows
        if (idx + 1) % 25 == 0 or idx == len(df) - 1:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False)

    valid = result_df[result_df["llm_related"] == True]
    logger.success(
        f"Validation done: {len(valid)}/{len(result_df)} matches are genuinely related"
    )

    # Show the good ones
    logger.info("\n=== VALIDATED MATCHES ===")
    for _, row in valid.sort_values("similarity_score", ascending=False).head(20).iterrows():
        logger.info(f"  SURVEY ({row.survey_date}): {row.question_clean}")
        logger.info(f"  VOTE   ({row.vote_date}, +{row.days_between}d): {row.vote_summary_clean}")
        logger.info(f"  Why: {row.llm_explanation}")
        logger.info(f"  Similarity: {row.similarity_score:.3f}")
        logger.info("")


if __name__ == "__main__":
    main()
