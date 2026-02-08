"""Validate survey-vote matches using a local LLM (Ollama + Mistral).

Usage:
    python backend/scripts/validate_matches.py
    python backend/scripts/validate_matches.py --limit 20
    python backend/scripts/validate_matches.py --resume
"""

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from eu_survey_correlation.validation import MatchJudge

DATA_DIR = Path("data")
MATCHES_CSV = DATA_DIR / "matches" / "survey_vote_matches.csv"
OUTPUT_CSV = DATA_DIR / "matches" / "survey_vote_matches_judged.csv"


def main():
    parser = argparse.ArgumentParser(description="LLM validation of match pairs")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only judge the first N pairs (useful for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip pairs already judged in the output file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        help="Ollama model name (default: mistral)",
    )
    args = parser.parse_args()

    logger.info(f"Loading matches from {MATCHES_CSV}...")
    df = pd.read_csv(MATCHES_CSV)
    logger.info(f"Loaded {len(df)} match pairs")

    already_judged = pd.DataFrame()
    if args.resume and OUTPUT_CSV.exists():
        already_judged = pd.read_csv(OUTPUT_CSV)
        logger.info(f"Resuming: {len(already_judged)} pairs already judged")

        # Build a set of already-judged (question_id, vote_id) pairs
        judged_keys = set(
            zip(already_judged["question_id"], already_judged["vote_id"])
        )
        df = df[
            ~df.apply(
                lambda r: (r["question_id"], r["vote_id"]) in judged_keys, axis=1
            )
        ]
        logger.info(f"Remaining pairs to judge: {len(df)}")

    if args.limit:
        df = df.head(args.limit)
        logger.info(f"Limited to {len(df)} pairs")

    if df.empty:
        logger.info("Nothing to judge!")
        return

    judge = MatchJudge(model=args.model)
    judged_df = judge.judge_dataframe(df)

    # Concatenate with previously judged pairs if resuming
    if not already_judged.empty:
        judged_df = pd.concat([already_judged, judged_df], ignore_index=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    judged_df.to_csv(OUTPUT_CSV, index=False)
    logger.success(f"Saved {len(judged_df)} judged pairs → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

