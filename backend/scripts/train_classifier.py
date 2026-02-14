"""Train and evaluate the survey-vote relatedness classifier.

Usage:
    python backend/scripts/train_classifier.py
    python backend/scripts/train_classifier.py --labeled-csv data/training/labeled_pairs.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from eu_survey_correlation.training.classifier import build_features, train_model
from eu_survey_correlation.training.evaluator import evaluate_model
from loguru import logger
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
LABELED_CSV = DATA_DIR / "training" / "labeled_pairs.csv"
SURVEY_EMB = DATA_DIR / "embeddings" / "survey_embeddings.parquet"
VOTE_EMB = DATA_DIR / "embeddings" / "vote_embeddings.parquet"
MODEL_PATH = DATA_DIR / "training" / "model.json"
REPORT_PATH = DATA_DIR / "training" / "evaluation_report.txt"


def main():
    parser = argparse.ArgumentParser(
        description="Train survey-vote relatedness classifier"
    )
    parser.add_argument(
        "--labeled-csv",
        type=str,
        default=str(LABELED_CSV),
        help="Path to labeled pairs CSV",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    args = parser.parse_args()

    # Load labeled data
    labeled_df = pd.read_csv(args.labeled_csv)
    logger.info(f"Loaded {len(labeled_df)} labeled pairs")

    # Filter out failed labels
    labeled_df = labeled_df[labeled_df["llm_score"] >= 0].copy()
    logger.info(f"After filtering errors: {len(labeled_df)} pairs")

    # Build features
    X, y = build_features(labeled_df, SURVEY_EMB, VOTE_EMB)

    # Split
    y_binary = (y >= 0.6).astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y_binary
    )
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Train
    model = train_model(X_train, y_train, output_path=MODEL_PATH)

    # Evaluate
    y_pred = model.predict(X_val)
    evaluate_model(
        y_true=y_val,
        y_pred=y_pred,
        output_path=REPORT_PATH,
        feature_importances=model.feature_importances_,
    )


if __name__ == "__main__":
    main()
