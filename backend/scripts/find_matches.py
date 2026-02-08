"""Compute semantic matches between survey questions and vote summaries.

Usage:
    python backend/scripts/find_matches.py
    python backend/scripts/find_matches.py --top-k 10 --threshold 0.4
"""

import argparse
from pathlib import Path

from eu_survey_correlation.embeddings import VoteSurveyMatcher
from loguru import logger

DATA_DIR = Path("data")
SURVEY_EMB = DATA_DIR / "embeddings" / "survey_embeddings.parquet"
VOTE_EMB = DATA_DIR / "embeddings" / "vote_embeddings.parquet"
OUTPUT_PATH = DATA_DIR / "matches" / "survey_vote_matches.csv"


def main():
    parser = argparse.ArgumentParser(description="Find survey-vote semantic matches")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Top-k matches per question"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Minimum similarity"
    )
    args = parser.parse_args()

    logger.info(f"Survey embeddings: {SURVEY_EMB}")
    logger.info(f"Vote embeddings:   {VOTE_EMB}")

    matcher = VoteSurveyMatcher(
        survey_embeddings_path=SURVEY_EMB,
        vote_embeddings_path=VOTE_EMB,
    )

    logger.info(
        f"Matching {len(matcher.survey_df)} questions "
        f"against {len(matcher.vote_df)} votes "
        f"(top_k={args.top_k}, threshold={args.threshold})"
    )

    matcher.save_matches(
        output_path=OUTPUT_PATH,
        top_k=args.top_k,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
