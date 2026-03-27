"""
One-command retrain: Optuna HPO + train final model + generate report.

Usage:
    make retrain                          # full retrain (100 Optuna trials)
    make retrain TRIALS=200               # more trials
    make retrain-quick                    # reuse cached study
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
from eu_survey_correlation.logging import (
    console,
    log,
    print_candidates_table,
    print_kv,
    print_metrics,
    print_section,
)

OPTUNA_DB = f"sqlite:///{DATA / 'classifier' / 'optuna.db'}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain classifier with Optuna HPO")
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="Max seconds for Optuna search"
    )
    parser.add_argument(
        "--skip-optuna", action="store_true", help="Reuse cached Optuna study"
    )
    parser.add_argument(
        "--use-cross-encoder", action="store_true", help="Include cross-encoder scores"
    )
    args = parser.parse_args()

    # 1. Load data
    print_section("Loading data")
    labelled = load_labelled_data()
    emb_lookup = load_embedding_lookup()

    accepted = [r for r in labelled if r["admin_validated"] is True]
    refused = [r for r in labelled if r["admin_validated"] is False]
    print_kv("Accepted", len(accepted), "green")
    print_kv("Refused", len(refused), "red")

    # 2. Build features
    feature_df = build_feature_matrix(labelled, emb_lookup)
    y = np.array([1 if r["admin_validated"] else 0 for r in labelled])
    feature_names = list(feature_df.columns)
    X = feature_df.values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)

    if args.use_cross_encoder:
        log.info("Computing cross-encoder scores...")
        ce_scores = compute_cross_encoder_scores(labelled)
        X = np.column_stack([X, ce_scores])
        feature_names = feature_names + ["cross_encoder_score"]

    print_kv("Feature matrix", f"{X.shape[0]} x {X.shape[1]}")
    print_kv("Features", ", ".join(feature_names))

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    # 3. Run Optuna
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print_section("Optuna HPO")

    if args.skip_optuna:
        import optuna

        study = optuna.load_study(study_name="classifier_hpo", storage=OPTUNA_DB)
        log.info(
            f"Loaded cached study: {len(study.trials)} trials, best F1={study.best_value:.3f}"
        )
    else:
        log.info(f"Running {args.n_trials} trials...")
        study = run_optuna_study(
            X,
            y,
            feature_names,
            n_trials=args.n_trials,
            timeout=args.timeout,
            storage=OPTUNA_DB,
            include_cross_encoder=args.use_cross_encoder,
        )
        log.info(
            f"Best trial #{study.best_trial.number}: [bold green]F1={study.best_value:.3f}[/]",
            extra={"markup": True},
        )

    # 4. Extract best config
    config = best_trial_to_config(study, feature_names)
    print_section("Best config")
    print_kv("Model", config["model_type"], "bold")
    print_kv("Features", ", ".join(config["selected_features"]))
    print_kv("Hyperparameters", config["hyperparameters"])

    # 5. Select features
    selected = config["selected_features"]
    selected_indices = [feature_names.index(f) for f in selected]
    X_sel = X[:, selected_indices]

    # 6. Full outer CV
    print_section("Outer CV (5x10 repeats)")
    make_estimator = _make_estimator_factory(
        config["model_type"], config["hyperparameters"], scale_pos_weight
    )
    cv_results = evaluate_cv(X_sel, y, make_estimator, n_splits=5, n_repeats=10)
    print_metrics(cv_results["cv_metrics"])
    print_kv("Calibrated threshold", f"{cv_results['calibrated_threshold']:.2f}")

    # 7. Train final model
    print_section("Final model")
    final_results = train_final_model(X_sel, y, make_estimator, selected)

    final_results.update(cv_results)
    final_results["model_type"] = config["model_type"]
    final_results["selected_features"] = selected
    final_results["optuna_best_trial"] = config["optuna_best_trial"]
    final_results["optuna_best_value"] = config["optuna_best_value"]

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

    log.info(
        f"Model saved to [bold]{OUTPUT_DIR / 'model.joblib'}[/]", extra={"markup": True}
    )

    # 9. Score unlabelled pairs for active learning
    print_section("Active learning pool")
    unlabelled: list[dict] = []

    backup_files = sorted(MIGRATION_DIR.glob("survey_vote_matches_backup_*.json"))
    if backup_files:
        with open(backup_files[-1]) as f:
            all_matches = json.load(f)
        backup_unlabelled = [r for r in all_matches if r.get("admin_validated") is None]
        unlabelled.extend(backup_unlabelled)
        log.info(f"Backup unlabelled: {len(backup_unlabelled)} pairs")

    candidates_csv = DATA / "matches" / "all_candidates.csv"
    if candidates_csv.exists():
        import pandas as pd

        cand_df = pd.read_csv(candidates_csv)
        existing_ids = {r.get("match_id") for r in unlabelled}
        n_before = len(unlabelled)
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
        log.info(f"Candidates CSV: +{len(unlabelled) - n_before} new pairs")

    unlabelled_scored = None
    if unlabelled:
        X_unlab = build_feature_matrix(unlabelled, emb_lookup).values.astype(np.float64)
        X_unlab = np.nan_to_num(X_unlab, nan=0.0)

        if args.use_cross_encoder:
            ce_unlab = compute_cross_encoder_scores(
                unlabelled, DATA / "cache" / "cross_encoder_scores_unlabelled.npy"
            )
            X_unlab = np.column_stack([X_unlab, ce_unlab])

        all_feat_names = list(build_feature_matrix([unlabelled[0]], emb_lookup).columns)
        if args.use_cross_encoder:
            all_feat_names.append("cross_encoder_score")
        sel_idx = [all_feat_names.index(f) for f in selected if f in all_feat_names]
        X_unlab_sel = X_unlab[:, sel_idx]

        probs = final_results["model"].predict_proba(X_unlab_sel)[:, 1]
        for r, p in zip(unlabelled, probs):
            r["predicted_probability"] = float(p)
        unlabelled_scored = unlabelled
        log.info(
            f"Scored [bold]{len(unlabelled)}[/] unlabelled pairs",
            extra={"markup": True},
        )

        # Show top candidates
        sorted_by_unc = sorted(
            unlabelled,
            key=lambda x: abs(
                x["predicted_probability"] - cv_results["calibrated_threshold"]
            ),
        )
        print_candidates_table(
            sorted_by_unc, threshold=cv_results["calibrated_threshold"], max_rows=2
        )
    else:
        log.warning(
            "No unlabelled pairs found — run [bold]make generate-candidates[/] first",
            extra={"markup": True},
        )

    # 10. Generate report
    print_section("Report")
    generate_report(
        final_results,
        X_sel,
        y,
        selected,
        n_accepted=len(accepted),
        n_refused=len(refused),
        feature_df=feature_df,
        unlabelled=unlabelled_scored,
    )

    console.print("\n[bold green]Done![/]")


if __name__ == "__main__":
    main()
