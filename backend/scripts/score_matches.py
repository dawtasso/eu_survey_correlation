"""
Score survey<->vote match pairs using the trained classifier.

Usage:
    make score CSV=data/matches/some_file.csv
    make score-unlabelled
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
    SETFIT_MODEL_DIR,
    build_feature_matrix,
    compute_cross_encoder_scores,
    load_embedding_lookup,
    load_setfit,
    predict_setfit,
)
from eu_survey_correlation.logging import (
    console,
    log,
    print_candidates_table,
    print_kv,
    print_section,
)

ROOT = Path(__file__).resolve().parents[2]


def load_model() -> tuple[object, float, str, list[str] | None]:
    # Check for SetFit model first (if its threshold.json says model_type=setfit)
    setfit_threshold_path = SETFIT_MODEL_DIR / "threshold.json"
    main_threshold_path = OUTPUT_DIR / "threshold.json"

    # Use main threshold.json to determine model type
    model_type = "lr"
    threshold = 0.5
    selected_features = None

    if main_threshold_path.exists():
        with open(main_threshold_path) as f:
            meta = json.load(f)
            threshold = meta["threshold"]
            model_type = meta.get("model_type", "lr")
            selected_features = meta.get("selected_features")

    if model_type == "setfit":
        if not SETFIT_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"threshold.json says model_type=setfit but {SETFIT_MODEL_DIR} not found."
            )
        model = load_setfit()
        # Use setfit-specific threshold if available
        if setfit_threshold_path.exists():
            with open(setfit_threshold_path) as f:
                sf_meta = json.load(f)
                threshold = sf_meta["threshold"]
        return model, threshold, "setfit", None

    model_path = OUTPUT_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run train_classifier.py first."
        )
    model = joblib.load(model_path)
    return model, threshold, model_type, selected_features


def _append_ce_scores(
    X: np.ndarray, records: list[dict], cache_name: str
) -> np.ndarray:
    cache_path = DATA / "cache" / cache_name
    ce_scores = compute_cross_encoder_scores(records, cache_path)
    return np.column_stack([X, ce_scores])


def _select_features(
    X: np.ndarray, feature_names: list[str], selected: list[str] | None
) -> np.ndarray:
    if selected is None:
        return X
    indices = [feature_names.index(f) for f in selected if f in feature_names]
    return X[:, indices]


def score_csv(
    csv_path: Path,
    model,
    threshold: float,
    emb_lookup: dict,
    model_type: str = "lr",
    selected_features: list[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    log.info(
        f"Loaded [bold]{len(df)}[/] matches from {csv_path.name}",
        extra={"markup": True},
    )

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "question_clean": row.get("question_clean", ""),
                "vote_summary_clean": row.get("summary_clean", ""),
                "similarity_score": row.get("similarity_score", 0),
                "days_between": row.get("time_delta", 0),
            }
        )

    if model_type == "setfit":
        probs = predict_setfit(model, records)
    else:
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
    model,
    threshold: float,
    emb_lookup: dict,
    model_type: str = "lr",
    selected_features: list[str] | None = None,
) -> list[dict]:
    backup_files = sorted(MIGRATION_DIR.glob("survey_vote_matches_backup_*.json"))
    if not backup_files:
        raise FileNotFoundError(f"No backup files in {MIGRATION_DIR}")

    with open(backup_files[-1]) as f:
        all_matches = json.load(f)

    unlabelled = [r for r in all_matches if r.get("admin_validated") is None]
    if not unlabelled:
        log.info("No unlabelled pairs found.")
        return []

    if model_type == "setfit":
        probs = predict_setfit(model, unlabelled)
    else:
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

    unlabelled.sort(key=lambda x: x["uncertainty"])

    return unlabelled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score match pairs with trained classifier"
    )
    parser.add_argument("csv_path", nargs="?", help="Path to matches CSV")
    parser.add_argument(
        "--unlabelled", action="store_true", help="Score unlabelled pairs from backup"
    )
    parser.add_argument(
        "--output", "-o", help="Output CSV path (default: adds _scored suffix)"
    )
    args = parser.parse_args()

    if not args.csv_path and not args.unlabelled:
        parser.error("Provide a CSV path or --unlabelled")

    model, threshold, model_type, selected_features = load_model()
    emb_lookup = load_embedding_lookup()
    print_kv("Model", model_type)
    print_kv("Threshold", f"{threshold:.2f}")
    if selected_features:
        print_kv("Features", ", ".join(selected_features))

    if args.csv_path:
        csv_path = Path(args.csv_path)
        if not csv_path.is_absolute():
            csv_path = ROOT / csv_path

        df = score_csv(
            csv_path, model, threshold, emb_lookup, model_type, selected_features
        )

        output_path = args.output or str(csv_path).replace(".csv", "_scored.csv")
        df.to_csv(output_path, index=False)

        n_accepted = int(df["predicted_accepted"].sum())
        print_section("Results")
        print_kv(
            "Predicted accepted",
            f"{n_accepted}/{len(df)} ({100*n_accepted/len(df):.1f}%)",
        )
        print_kv(
            "Score distribution",
            f"mean={df['predicted_quality'].mean():.3f}  "
            f"std={df['predicted_quality'].std():.3f}",
        )
        print_kv("Saved to", output_path)

    if args.unlabelled:
        candidates = score_unlabelled(
            model, threshold, emb_lookup, model_type, selected_features
        )
        if candidates:
            print_section("Top pairs to label (uncertainty sampling)")
            print_candidates_table(candidates, threshold=threshold, max_rows=3)

            output_path = OUTPUT_DIR / "unlabelled_scored.json"
            with open(output_path, "w") as f:
                json.dump(candidates, f, indent=2, default=str)
            print_kv("Full results", output_path)


if __name__ == "__main__":
    main()
