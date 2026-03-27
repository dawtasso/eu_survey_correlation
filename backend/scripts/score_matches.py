"""
Score survey<->vote match pairs using the trained classifier.

Usage:
    uv run python backend/scripts/score_matches.py data/matches/simplified_michlou_survey_vote_matches_clean.csv
    uv run python backend/scripts/score_matches.py --unlabelled   # score unlabelled pairs from backup
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from eu_survey_correlation.classifier import (
    DATA,
    MIGRATION_DIR,
    OUTPUT_DIR,
    build_feature_matrix,
    compute_cross_encoder_scores,
    load_embedding_lookup,
)

ROOT = Path(__file__).resolve().parents[2]


def load_model() -> tuple[object, float, str, list[str] | None]:
    """Load trained model, threshold, model type, and optional selected features."""
    model_path = OUTPUT_DIR / "model.joblib"
    threshold_path = OUTPUT_DIR / "threshold.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run train_classifier.py first."
        )

    model = joblib.load(model_path)

    threshold = 0.5
    model_type = "lr"
    selected_features = None
    if threshold_path.exists():
        with open(threshold_path) as f:
            meta = json.load(f)
            threshold = meta["threshold"]
            model_type = meta.get("model_type", "lr")
            selected_features = meta.get("selected_features")

    return model, threshold, model_type, selected_features


def _append_ce_scores(X: np.ndarray, records: list[dict], cache_name: str) -> np.ndarray:
    """Compute cross-encoder scores and append as extra column."""
    cache_path = DATA / "cache" / cache_name
    ce_scores = compute_cross_encoder_scores(records, cache_path)
    return np.column_stack([X, ce_scores])


def _select_features(X: np.ndarray, feature_names: list[str], selected: list[str] | None) -> np.ndarray:
    """Slice feature matrix to selected features if specified."""
    if selected is None:
        return X
    indices = [feature_names.index(f) for f in selected if f in feature_names]
    return X[:, indices]


def score_csv(
    csv_path: Path, model, threshold: float, emb_lookup: dict,
    model_type: str = "lr", selected_features: list[str] | None = None,
) -> pd.DataFrame:
    """Score matches from a CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} matches from {csv_path.name}")

    # Map CSV columns to expected names for feature engineering
    records = []
    for _, row in df.iterrows():
        records.append({
            "question_clean": row.get("question_clean", ""),
            "vote_summary_clean": row.get("summary_clean", ""),
            "similarity_score": row.get("similarity_score", 0),
            "days_between": row.get("time_delta", 0),
        })

    feature_df = build_feature_matrix(records, emb_lookup)
    feature_names = list(feature_df.columns)
    X = feature_df.values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)

    if model_type == "hybrid":
        X = _append_ce_scores(X, records, "cross_encoder_scores_csv.npy")

    X = _select_features(X, feature_names, selected_features)

    probs = model.predict_proba(X)[:, 1]
    df["predicted_quality"] = probs
    df["predicted_accepted"] = probs >= threshold

    return df


def score_unlabelled(
    model, threshold: float, emb_lookup: dict,
    model_type: str = "lr", selected_features: list[str] | None = None,
) -> list[dict]:
    """Score unlabelled pairs from migration backup and rank by uncertainty."""
    backup_files = sorted(MIGRATION_DIR.glob("survey_vote_matches_backup_*.json"))
    if not backup_files:
        raise FileNotFoundError(f"No backup files in {MIGRATION_DIR}")

    with open(backup_files[-1]) as f:
        all_matches = json.load(f)

    unlabelled = [r for r in all_matches if r.get("admin_validated") is None]
    if not unlabelled:
        print("No unlabelled pairs found.")
        return []

    feature_df = build_feature_matrix(unlabelled, emb_lookup)
    feature_names = list(feature_df.columns)
    X = feature_df.values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)

    if model_type == "hybrid":
        X = _append_ce_scores(X, unlabelled, "cross_encoder_scores_unlabelled.npy")

    X = _select_features(X, feature_names, selected_features)

    probs = model.predict_proba(X)[:, 1]

    for r, p in zip(unlabelled, probs):
        r["predicted_probability"] = float(p)
        r["predicted_accepted"] = bool(p >= threshold)
        r["uncertainty"] = float(abs(p - threshold))

    # Sort by uncertainty (closest to boundary = most informative to label)
    unlabelled.sort(key=lambda x: x["uncertainty"])

    return unlabelled


def main() -> None:
    parser = argparse.ArgumentParser(description="Score match pairs with trained classifier")
    parser.add_argument("csv_path", nargs="?", help="Path to matches CSV")
    parser.add_argument("--unlabelled", action="store_true", help="Score unlabelled pairs from backup")
    parser.add_argument("--output", "-o", help="Output CSV path (default: adds _scored suffix)")
    args = parser.parse_args()

    if not args.csv_path and not args.unlabelled:
        parser.error("Provide a CSV path or --unlabelled")

    model, threshold, model_type, selected_features = load_model()
    emb_lookup = load_embedding_lookup()
    print(f"Model type: {model_type}, Threshold: {threshold:.2f}")
    if selected_features:
        print(f"Selected features: {selected_features}")

    if args.csv_path:
        csv_path = Path(args.csv_path)
        if not csv_path.is_absolute():
            csv_path = ROOT / csv_path

        df = score_csv(csv_path, model, threshold, emb_lookup, model_type, selected_features)

        output_path = args.output or str(csv_path).replace(".csv", "_scored.csv")
        df.to_csv(output_path, index=False)

        n_accepted = df["predicted_accepted"].sum()
        print(f"\n── Results ─────────────────────────────────")
        print(f"Predicted accepted: {n_accepted}/{len(df)} ({100*n_accepted/len(df):.1f}%)")
        print(f"Score distribution: mean={df['predicted_quality'].mean():.3f}, "
              f"std={df['predicted_quality'].std():.3f}")
        print(f"Saved to {output_path}")

    if args.unlabelled:
        candidates = score_unlabelled(model, threshold, emb_lookup, model_type, selected_features)
        if candidates:
            print(f"\n── Top 20 pairs to label next (uncertainty sampling) ──")
            for i, c in enumerate(candidates[:20], 1):
                q = str(c.get("question_clean", ""))[:50]
                v = str(c.get("vote_summary_clean", ""))[:50]
                print(f"  {i:2d}. P={c['predicted_probability']:.3f} "
                      f"(unc={c['uncertainty']:.3f}) | Q: {q}... | V: {v}...")

            # Save full results
            output_path = OUTPUT_DIR / "unlabelled_scored.json"
            with open(output_path, "w") as f:
                json.dump(candidates, f, indent=2, default=str)
            print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
