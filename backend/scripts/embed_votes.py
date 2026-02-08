"""Embed vote summaries and save to parquet.

Usage:
    python backend/scripts/embed_votes.py
"""

from pathlib import Path

import pandas as pd
from eu_survey_correlation.embeddings import Embedder
from loguru import logger

DATA_DIR = Path("data")
VOTE_SUMMARIES_CSV = DATA_DIR / "votes" / "vote_summaries.csv"
OUTPUT_PATH = DATA_DIR / "embeddings" / "vote_embeddings.parquet"


def main():
    logger.info(f"Loading vote summaries from {VOTE_SUMMARIES_CSV}...")
    df = pd.read_csv(VOTE_SUMMARIES_CSV)
    logger.info(f"Loaded {len(df)} vote summaries")

    # Drop rows with no summary text
    df = df.dropna(subset=["summary"])
    # Deduplicate on vote_id (keep first occurrence)
    df = df.drop_duplicates(subset=["vote_id"])
    logger.info(f"After cleanup: {len(df)} unique vote summaries to embed")

    embedder = Embedder()
    embedder.embed_dataframe(
        df=df,
        text_column="summary",
        output_path=OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
