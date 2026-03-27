"""
Train a match-quality classifier on human-labelled survey<->vote pairs.

Usage:
    uv run python backend/scripts/train_classifier.py              # default: LR
    uv run python backend/scripts/train_classifier.py --model lr    # logistic regression
    uv run python backend/scripts/train_classifier.py --model hybrid  # XGBoost + cross-encoder score
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np

from eu_survey_correlation.classifier import (
    OUTPUT_DIR,
    build_feature_matrix,
    compute_cross_encoder_scores,
    generate_report,
    load_embedding_lookup,
    load_labelled_data,
    train_and_evaluate,
    train_and_evaluate_hybrid,
)
from eu_survey_correlation.classifier.constants import DATA, MIGRATION_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Train match-quality classifier")
    parser.add_argument(
        "--model", choices=["lr", "hybrid"], default="lr",
        help="Model type: 'lr' (LogisticRegression) or 'hybrid' (XGBoost + cross-encoder score)",
    )
    args = parser.parse_args()

    # Load data
    labelled = load_labelled_data()
    emb_lookup = load_embedding_lookup()

    # Split labelled / unlabelled
    accepted = [r for r in labelled if r["admin_validated"] is True]
    refused = [r for r in labelled if r["admin_validated"] is False]
    print(f"Accepted: {len(accepted)}, Refused: {len(refused)}")

    # Build features
    feature_df = build_feature_matrix(labelled, emb_lookup)
    y = np.array([1 if r["admin_validated"] else 0 for r in labelled])
    feature_names = list(feature_df.columns)
    X = feature_df.values.astype(np.float64)

    # Check for NaN
    nan_mask = np.isnan(X)
    if nan_mask.any():
        print(f"Warning: {nan_mask.sum()} NaN values found, filling with 0")
        X = np.nan_to_num(X, nan=0.0)

    print(f"\nFeature matrix: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Class balance: {y.sum()} accepted / {len(y) - y.sum()} refused\n")

    # Train and evaluate
    model_type = args.model
    if model_type == "hybrid":
        print("Training hybrid model (XGBoost + cross-encoder score)...")
        ce_scores = compute_cross_encoder_scores(labelled)
        results = train_and_evaluate_hybrid(X, ce_scores, y, feature_names)
    else:
        print("Training LR baseline...")
        results = train_and_evaluate(X, y, feature_names)

    # Print results
    cv = results["cv_metrics"]
    print(f"\n── Cross-Validation Results ({model_type}) ─────────────────")
    print(f"F1:        {cv['f1']['mean']:.3f} ± {cv['f1']['std']:.3f}")
    print(f"Precision: {cv['precision']['mean']:.3f} ± {cv['precision']['std']:.3f}")
    print(f"Recall:    {cv['recall']['mean']:.3f} ± {cv['recall']['std']:.3f}")
    print(f"PR-AUC:    {cv['pr_auc']['mean']:.3f} ± {cv['pr_auc']['std']:.3f}")
    print(f"Threshold: {results['calibrated_threshold']:.2f}")

    print("\n── Feature Importances ─────────────────────")
    for name, coef in sorted(results["feature_importances"].items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name:25s} {coef:+.4f}")

    # Score unlabelled pairs for active learning
    #   Sources: (a) backup JSON unlabelled + (b) all_candidates.csv if it exists
    import pandas as pd

    unlabelled: list[dict] = []

    backup_files = sorted(MIGRATION_DIR.glob("survey_vote_matches_backup_*.json"))
    if backup_files:
        with open(backup_files[-1]) as f:
            all_matches = json.load(f)
        unlabelled.extend(r for r in all_matches if r.get("admin_validated") is None)

    candidates_csv = DATA / "matches" / "all_candidates.csv"
    if candidates_csv.exists():
        cand_df = pd.read_csv(candidates_csv)
        existing_ids = {r.get("match_id") for r in unlabelled}
        for _, row in cand_df.iterrows():
            mid = str(row.get("match_id", ""))
            if mid in existing_ids:
                continue
            existing_ids.add(mid)
            unlabelled.append({
                "match_id": mid,
                "question_clean": str(row.get("question_clean", "")),
                "vote_summary_clean": str(row.get("vote_summary_clean", "")),
                "similarity_score": float(row.get("similarity_score", 0)),
                "days_between": float(row.get("days_between", 0)) if pd.notna(row.get("days_between")) else 0,
            })
        print(f"Loaded {len(cand_df)} rows from {candidates_csv.name}")

    unlabelled_scored = None
    if unlabelled:
        X_unlab = build_feature_matrix(unlabelled, emb_lookup).values.astype(np.float64)
        X_unlab = np.nan_to_num(X_unlab, nan=0.0)

        if model_type == "hybrid":
            ce_unlab = compute_cross_encoder_scores(unlabelled, DATA / "cache" / "cross_encoder_scores_unlabelled.npy")
            X_unlab = np.column_stack([X_unlab, ce_unlab])

        probs = results["model"].predict_proba(X_unlab)[:, 1]
        for r, p in zip(unlabelled, probs):
            r["predicted_probability"] = float(p)
        unlabelled_scored = unlabelled
        print(f"\nScored {len(unlabelled)} unlabelled pairs (backup + candidates)")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(results["model"], OUTPUT_DIR / "model.joblib")

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(results["cv_metrics"], f, indent=2)

    with open(OUTPUT_DIR / "feature_importances.json", "w") as f:
        json.dump(results["feature_importances"], f, indent=2)

    threshold_meta = {"threshold": results["calibrated_threshold"], "model_type": model_type}
    if model_type == "lr":
        threshold_meta["best_C"] = results.get("best_C")
    elif model_type == "hybrid":
        threshold_meta["best_params"] = results.get("best_params")
    with open(OUTPUT_DIR / "threshold.json", "w") as f:
        json.dump(threshold_meta, f, indent=2)

    print(f"\nModel saved to {OUTPUT_DIR / 'model.joblib'}")

    # Generate report
    generate_report(
        results, X, y, feature_names,
        n_accepted=len(accepted), n_refused=len(refused),
        feature_df=feature_df, unlabelled=unlabelled_scored,
    )


if __name__ == "__main__":
    main()
