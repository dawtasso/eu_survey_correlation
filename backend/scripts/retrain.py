"""
One-command retrain: Optuna HPO + train final model + generate report.

Usage:
    uv run python backend/scripts/retrain.py                          # full retrain
    uv run python backend/scripts/retrain.py --n-trials 200           # more Optuna trials
    uv run python backend/scripts/retrain.py --skip-optuna            # reuse cached study
    uv run python backend/scripts/retrain.py --use-cross-encoder      # include CE scores
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np

from eu_survey_correlation.classifier import (
    DATA,
    MIGRATION_DIR,
    OUTPUT_DIR,
    build_feature_matrix,
    compute_cross_encoder_scores,
    evaluate_cv,
    generate_report,
    load_embedding_lookup,
    load_labelled_data,
    train_final_model,
)
from eu_survey_correlation.classifier.optuna_search import (
    _make_estimator_factory,
    best_trial_to_config,
    run_optuna_study,
)

OPTUNA_DB = f"sqlite:///{DATA / 'classifier' / 'optuna.db'}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain classifier with Optuna HPO")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Max seconds for Optuna search")
    parser.add_argument("--skip-optuna", action="store_true", help="Reuse cached Optuna study")
    parser.add_argument("--use-cross-encoder", action="store_true", help="Include cross-encoder scores")
    args = parser.parse_args()

    # 1. Load data
    print("Loading data...")
    labelled = load_labelled_data()
    emb_lookup = load_embedding_lookup()

    accepted = [r for r in labelled if r["admin_validated"] is True]
    refused = [r for r in labelled if r["admin_validated"] is False]
    print(f"Accepted: {len(accepted)}, Refused: {len(refused)}")

    # 2. Build features
    feature_df = build_feature_matrix(labelled, emb_lookup)
    y = np.array([1 if r["admin_validated"] else 0 for r in labelled])
    feature_names = list(feature_df.columns)
    X = feature_df.values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)

    # Optional: add cross-encoder scores
    if args.use_cross_encoder:
        print("Computing cross-encoder scores...")
        ce_scores = compute_cross_encoder_scores(labelled)
        X = np.column_stack([X, ce_scores])
        feature_names = feature_names + ["cross_encoder_score"]

    print(f"Feature matrix: {X.shape}, Features: {feature_names}")
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    # 3. Run Optuna (or load cached study)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_optuna:
        import optuna
        print("Loading cached Optuna study...")
        study = optuna.load_study(study_name="classifier_hpo", storage=OPTUNA_DB)
        print(f"Loaded study with {len(study.trials)} trials, best F1={study.best_value:.3f}")
    else:
        print(f"Running Optuna HPO ({args.n_trials} trials)...")
        study = run_optuna_study(
            X, y, feature_names,
            n_trials=args.n_trials,
            timeout=args.timeout,
            storage=OPTUNA_DB,
            include_cross_encoder=args.use_cross_encoder,
        )
        print(f"Best trial #{study.best_trial.number}: F1={study.best_value:.3f}")

    # 4. Extract best config
    config = best_trial_to_config(study, feature_names)
    print(f"\nBest config:")
    print(f"  Model: {config['model_type']}")
    print(f"  Features: {config['selected_features']}")
    print(f"  Hyperparameters: {config['hyperparameters']}")

    # 5. Select features
    selected = config["selected_features"]
    selected_indices = [feature_names.index(f) for f in selected]
    X_sel = X[:, selected_indices]

    # 6. Full outer CV for robust metrics
    print("\nRunning full outer CV (5x10 repeats)...")
    make_estimator = _make_estimator_factory(
        config["model_type"], config["hyperparameters"], scale_pos_weight
    )
    cv_results = evaluate_cv(X_sel, y, make_estimator, n_splits=5, n_repeats=10)

    cv = cv_results["cv_metrics"]
    print(f"\n── Cross-Validation Results ─────────────────")
    print(f"F1:        {cv['f1']['mean']:.3f} +- {cv['f1']['std']:.3f}")
    print(f"Precision: {cv['precision']['mean']:.3f} +- {cv['precision']['std']:.3f}")
    print(f"Recall:    {cv['recall']['mean']:.3f} +- {cv['recall']['std']:.3f}")
    print(f"PR-AUC:    {cv['pr_auc']['mean']:.3f} +- {cv['pr_auc']['std']:.3f}")
    print(f"Threshold: {cv_results['calibrated_threshold']:.2f}")

    # 7. Train final model on all data
    print("\nTraining final model...")
    final_results = train_final_model(X_sel, y, make_estimator, selected)

    # Merge cv_results into final_results
    final_results.update(cv_results)
    final_results["model_type"] = config["model_type"]
    final_results["selected_features"] = selected
    final_results["optuna_best_trial"] = config["optuna_best_trial"]
    final_results["optuna_best_value"] = config["optuna_best_value"]

    # For LR, include best_C for report compatibility
    if config["model_type"] == "logistic_regression":
        final_results["best_C"] = config["hyperparameters"].get("C")

    # 8. Save outputs
    joblib.dump(final_results["model"], OUTPUT_DIR / "model.joblib")

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(cv_results["cv_metrics"], f, indent=2)

    with open(OUTPUT_DIR / "feature_importances.json", "w") as f:
        json.dump(final_results["feature_importances"], f, indent=2)

    threshold_meta = {
        "threshold": cv_results["calibrated_threshold"],
        "model_type": config["model_type"],
        "selected_features": selected,
        "optuna_best_trial": config["optuna_best_trial"],
        "optuna_best_value": config["optuna_best_value"],
    }
    if config["model_type"] == "logistic_regression":
        threshold_meta["best_C"] = config["hyperparameters"].get("C")
    else:
        threshold_meta["hyperparameters"] = config["hyperparameters"]

    with open(OUTPUT_DIR / "threshold.json", "w") as f:
        json.dump(threshold_meta, f, indent=2)

    print(f"\nModel saved to {OUTPUT_DIR / 'model.joblib'}")

    # 9. Score unlabelled pairs for active learning
    #    Sources: (a) backup JSON unlabelled rows + (b) all_candidates.csv if it exists
    unlabelled: list[dict] = []

    # (a) Backup unlabelled
    backup_files = sorted(MIGRATION_DIR.glob("survey_vote_matches_backup_*.json"))
    if backup_files:
        with open(backup_files[-1]) as f:
            all_matches = json.load(f)
        backup_unlabelled = [r for r in all_matches if r.get("admin_validated") is None]
        unlabelled.extend(backup_unlabelled)
        print(f"Backup unlabelled: {len(backup_unlabelled)} pairs")

    # (b) all_candidates.csv (from generate_candidates.py)
    candidates_csv = DATA / "matches" / "all_candidates.csv"
    if candidates_csv.exists():
        import pandas as pd

        cand_df = pd.read_csv(candidates_csv)
        # Deduplicate: skip match_ids already in backup
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
        print(f"Candidates CSV: {len(cand_df)} rows, {len(unlabelled) - len(backup_unlabelled if backup_files else [])} new")

    unlabelled_scored = None
    if unlabelled:
        X_unlab = build_feature_matrix(unlabelled, emb_lookup).values.astype(np.float64)
        X_unlab = np.nan_to_num(X_unlab, nan=0.0)

        if args.use_cross_encoder:
            ce_unlab = compute_cross_encoder_scores(
                unlabelled, DATA / "cache" / "cross_encoder_scores_unlabelled.npy"
            )
            X_unlab = np.column_stack([X_unlab, ce_unlab])

        # Select same features
        all_feat_names = list(build_feature_matrix([unlabelled[0]], emb_lookup).columns)
        if args.use_cross_encoder:
            all_feat_names.append("cross_encoder_score")
        sel_idx = [all_feat_names.index(f) for f in selected if f in all_feat_names]
        X_unlab_sel = X_unlab[:, sel_idx]

        probs = final_results["model"].predict_proba(X_unlab_sel)[:, 1]
        for r, p in zip(unlabelled, probs):
            r["predicted_probability"] = float(p)
        unlabelled_scored = unlabelled
        print(f"Scored {len(unlabelled)} total unlabelled pairs for active learning")

    # 10. Generate report (pass full feature_df — plots need all features like similarity_score)
    generate_report(
        final_results, X_sel, y, selected,
        n_accepted=len(accepted), n_refused=len(refused),
        feature_df=feature_df, unlabelled=unlabelled_scored,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
