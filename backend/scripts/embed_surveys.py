"""Embed survey questions and save to parquet.

Usage:
    python backend/scripts/embed_surveys.py
"""

from pathlib import Path

import pandas as pd
from eu_survey_correlation.embeddings import Embedder
from loguru import logger

DATA_DIR = Path("data")
SURVEY_CSV = DATA_DIR / "surveys" / "all_survey_questions.csv"
OUTPUT_PATH = DATA_DIR / "embeddings" / "survey_embeddings.parquet"


def main():
    logger.info(f"Loading survey questions from {SURVEY_CSV}...")
    df = pd.read_csv(SURVEY_CSV)
    logger.info(f"Loaded {len(df)} survey questions")

    # Drop rows with no English question text
    df = df.dropna(subset=["question_en"])
    logger.info(f"After dropping NaN: {len(df)} questions to embed")

    embedder = Embedder()
    embedder.embed_dataframe(
        df=df,
        text_column="question_en",
        output_path=OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
