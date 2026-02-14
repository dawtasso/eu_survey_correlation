"""Generate labeled training data for the survey-vote relatedness classifier.

Samples positive, hard-negative, and random-negative pairs, then labels each
with a multi-pass LLM evaluation using a strict prompt.

Usage:
    python backend/scripts/generate_training_data.py
    python backend/scripts/generate_training_data.py --n-positive 250 --n-negative 250
    python backend/scripts/generate_training_data.py --passes 3 --resume
"""

import argparse
from pathlib import Path

import pandas as pd
from eu_survey_correlation.training.labeler import (
    label_dataframe,
    sample_training_pairs,
)
from loguru import logger

DATA_DIR = Path("data")
MATCHES_CSV = DATA_DIR / "matches" / "survey_vote_matches.csv"
SURVEY_EMB = DATA_DIR / "embeddings" / "survey_embeddings.parquet"
VOTE_EMB = DATA_DIR / "embeddings" / "vote_embeddings.parquet"
OUTPUT_CSV = DATA_DIR / "training" / "labeled_pairs.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Generate labeled training data for relatedness classifier"
    )
    parser.add_argument(
        "--n-positive", type=int, default=250, help="Number of positive candidates"
    )
    parser.add_argument(
        "--n-negative", type=int, default=250, help="Number of negative candidates"
    )
    parser.add_argument(
        "--passes", type=int, default=1, help="Number of LLM passes per pair"
    )
    parser.add_argument(
        "--model", type=str, default="mistral", help="Ollama model name"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip already-labeled pairs"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    # Step 1: Sample pairs (or load existing if resuming)
    if args.resume and OUTPUT_CSV.exists():
        logger.info(f"Resuming from {OUTPUT_CSV}...")
        already_labeled = pd.read_csv(OUTPUT_CSV)
        logger.info(f"{len(already_labeled)} pairs already labeled")

        # Re-sample to get the full candidate set (deterministic with same seed)
        candidates = sample_training_pairs(
            matches_path=MATCHES_CSV,
            survey_embeddings_path=SURVEY_EMB,
            vote_embeddings_path=VOTE_EMB,
            n_positive=args.n_positive,
            n_negative=args.n_negative,
            seed=args.seed,
        )

        # Filter out already-labeled pairs
        labeled_keys = set(
            zip(
                already_labeled["question_text"], already_labeled["vote_id"].astype(str)
            )
        )
        candidates = candidates[
            ~candidates.apply(
                lambda r: (r["question_text"], str(r["vote_id"])) in labeled_keys,
                axis=1,
            )
        ]
        logger.info(f"Remaining pairs to label: {len(candidates)}")
    else:
        already_labeled = None
        logger.info("Sampling training pairs...")
        candidates = sample_training_pairs(
            matches_path=MATCHES_CSV,
            survey_embeddings_path=SURVEY_EMB,
            vote_embeddings_path=VOTE_EMB,
            n_positive=args.n_positive,
            n_negative=args.n_negative,
            seed=args.seed,
        )

    if candidates.empty:
        logger.info("Nothing to label!")
        return

    # Step 2: Label pairs with multi-pass LLM
    result_df = label_dataframe(
        df=candidates,
        output_path=OUTPUT_CSV,
        model=args.model,
        n_passes=args.passes,
        already_labeled=already_labeled,
    )

    # Step 3: Summary stats
    n_flagged = result_df["llm_flagged"].sum()
    logger.info(f"\nLabel distribution:")
    logger.info(f"{result_df['llm_label'].value_counts().to_string()}")
    logger.info(f"\nFlagged for review: {n_flagged}")
    logger.success(f"Saved {len(result_df)} labeled pairs → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
