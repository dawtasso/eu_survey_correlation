"""
Train a match-quality classifier on human-labelled survey<->vote pairs.

Usage:
    make train                    # default: LR
    make train MODEL=hybrid       # XGBoost + cross-encoder score
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
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
from eu_survey_correlation.logging import (
    console,
    log,
    print_candidates_table,
    print_kv,
    print_metrics,
    print_section,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train match-quality classifier")
    parser.add_argument(
        "--model",
        choices=["lr", "hybrid"],
        default="lr",
        help="Model type: 'lr' (LogisticRegression) or 'hybrid' (XGBoost + cross-encoder score)",
    )
    args = parser.parse_args()

    # Load data
    print_section("Loading data")
    labelled = load_labelled_data()
    emb_lookup = load_embedding_lookup()

    accepted = [r for r in labelled if r["admin_validated"] is True]
    refused = [r for r in labelled if r["admin_validated"] is False]
    print_kv("Accepted", len(accepted), "green")
    print_kv("Refused", len(refused), "red")

    # Build features
    feature_df = build_feature_matrix(labelled, emb_lookup)
    y = np.array([1 if r["admin_validated"] else 0 for r in labelled])
    feature_names = list(feature_df.columns)
    X = feature_df.values.astype(np.float64)

    nan_mask = np.isnan(X)
    if nan_mask.any():
        log.warning(f"{nan_mask.sum()} NaN values found, filling with 0")
        X = np.nan_to_num(X, nan=0.0)

    print_kv("Feature matrix", f"{X.shape[0]} x {X.shape[1]}")
    print_kv("Features", ", ".join(feature_names))

    # Train
    model_type = args.model
    if model_type == "hybrid":
        print_section("Training hybrid (XGBoost + cross-encoder)")
        ce_scores = compute_cross_encoder_scores(labelled)
        results = train_and_evaluate_hybrid(X, ce_scores, y, feature_names)
    else:
        print_section("Training LR baseline")
        results = train_and_evaluate(X, y, feature_names)

    # Results
    print_metrics(results["cv_metrics"])
    print_kv("Threshold", f"{results['calibrated_threshold']:.2f}")

    print_section("Feature importances")
    for name, coef in sorted(
        results["feature_importances"].items(), key=lambda x: abs(x[1]), reverse=True
    ):
        color = "green" if coef > 0 else "red"
        console.print(f"  {name:25s} [{color}]{coef:+.4f}[/]")

    # Score unlabelled pairs
    print_section("Active learning pool")
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
            unlabelled.append(
                {
                    "match_id": mid,
                    "question_clean": str(row.get("question_clean", "")),
                    "vote_summary_clean": str(row.get("vote_summary_clean", "")),
                    "similarity_score": float(row.get("similarity_score", 0)),
                    "days_between": (
                        float(row.get("days_between", 0))
                        if pd.notna(row.get("days_between"))
                        else 0
                    ),
                }
            )
        log.info(f"Loaded {len(cand_df)} rows from {candidates_csv.name}")

    unlabelled_scored = None
    if unlabelled:
        X_unlab = build_feature_matrix(unlabelled, emb_lookup).values.astype(np.float64)
        X_unlab = np.nan_to_num(X_unlab, nan=0.0)

        if model_type == "hybrid":
            ce_unlab = compute_cross_encoder_scores(
                unlabelled, DATA / "cache" / "cross_encoder_scores_unlabelled.npy"
            )
            X_unlab = np.column_stack([X_unlab, ce_unlab])

        probs = results["model"].predict_proba(X_unlab)[:, 1]
        for r, p in zip(unlabelled, probs):
            r["predicted_probability"] = float(p)
        unlabelled_scored = unlabelled
        log.info(
            f"Scored [bold]{len(unlabelled)}[/] unlabelled pairs",
            extra={"markup": True},
        )

        sorted_by_unc = sorted(
            unlabelled,
            key=lambda x: abs(
                x["predicted_probability"] - results["calibrated_threshold"]
            ),
        )
        print_candidates_table(
            sorted_by_unc, threshold=results["calibrated_threshold"], max_rows=2
        )
    else:
        log.warning(
            "No unlabelled pairs — run [bold]make generate-candidates[/] first",
            extra={"markup": True},
        )

    # Save outputs
    print_section("Saving")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(results["model"], OUTPUT_DIR / "model.joblib")

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(results["cv_metrics"], f, indent=2)

    with open(OUTPUT_DIR / "feature_importances.json", "w") as f:
        json.dump(results["feature_importances"], f, indent=2)

    threshold_meta = {
        "threshold": results["calibrated_threshold"],
        "model_type": model_type,
    }
    if model_type == "lr":
        threshold_meta["best_C"] = results.get("best_C")
    elif model_type == "hybrid":
        threshold_meta["best_params"] = results.get("best_params")
    with open(OUTPUT_DIR / "threshold.json", "w") as f:
        json.dump(threshold_meta, f, indent=2)

    print_kv("Model", OUTPUT_DIR / "model.joblib")

    # Generate report
    generate_report(
        results,
        X,
        y,
        feature_names,
        n_accepted=len(accepted),
        n_refused=len(refused),
        feature_df=feature_df,
        unlabelled=unlabelled_scored,
    )

    console.print("\n[bold green]Done![/]")


if __name__ == "__main__":
    main()
