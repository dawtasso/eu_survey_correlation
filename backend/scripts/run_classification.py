"""Run the classification pipeline: classify → match → validate.

Classifies ALL surveys and votes independently first, then matches only
relevant subsets (opinion_forward × substantive), then validates pairs.

Usage:
    python backend/scripts/run_classification.py                    # full pipeline
    python backend/scripts/run_classification.py --stage questions  # classify surveys
    python backend/scripts/run_classification.py --stage votes      # classify votes
    python backend/scripts/run_classification.py --stage match      # semantic matching
    python backend/scripts/run_classification.py --stage pairs      # validate pairs
    python backend/scripts/run_classification.py --resume           # resume interrupted
    python backend/scripts/run_classification.py --limit 50         # test subset
"""

import argparse
from pathlib import Path

from eu_survey_correlation.classification import ClassificationPipeline
from loguru import logger

DATA_DIR = Path("data")


def main():
    parser = argparse.ArgumentParser(
        description="Classify surveys & votes, match, then validate pairs"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "questions", "votes", "match", "pairs"],
        default="all",
        help="Which stage to run (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-processed items",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process first N items per stage (for testing)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        help="Ollama model name (default: mistral)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k matches per survey (default: 5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum similarity threshold (default: 0.5)",
    )
    args = parser.parse_args()

    logger.info(
        f"Pipeline: stage={args.stage}, model={args.model}, "
        f"top_k={args.top_k}, threshold={args.threshold}"
    )

    pipeline = ClassificationPipeline(
        output_dir=DATA_DIR / "classification",
        model=args.model,
        surveys_path=DATA_DIR / "surveys" / "all_survey_questions.csv",
        votes_path=DATA_DIR / "votes" / "vote_summaries.csv",
        metadata_path=DATA_DIR / "surveys" / "distributions_metadata.json",
        votes_csv_path=DATA_DIR / "votes" / "votes.csv",
        top_k=args.top_k,
        threshold=args.threshold,
    )

    pipeline.run(resume=args.resume, limit=args.limit, stage=args.stage)


if __name__ == "__main__":
    main()
