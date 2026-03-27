"""Optuna hyperparameter optimisation for the classifier pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .training import evaluate_cv

# Silence Optuna INFO logs (keep WARNING+)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_estimator_factory(
    model_type: str, params: dict[str, Any], scale_pos_weight: float
) -> callable:
    """Return a zero-arg callable that builds a fresh estimator."""

    def _factory():
        if model_type == "logistic_regression":
            clf = LogisticRegression(
                C=params["C"],
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )
            return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        elif model_type == "xgboost":
            from xgboost import XGBClassifier

            clf = XGBClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )
            return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        elif model_type == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params.get("min_samples_split", 2),
                class_weight="balanced",
                random_state=42,
            )
            return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        elif model_type == "svm":
            clf = SVC(
                C=params["C"],
                kernel=params.get("kernel", "rbf"),
                probability=True,
                class_weight="balanced",
                random_state=42,
            )
            return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    return _factory


def _suggest_params(trial: optuna.Trial, model_type: str) -> dict[str, Any]:
    """Suggest hyperparameters for a given model type."""
    if model_type == "logistic_regression":
        return {"C": trial.suggest_float("lr_C", 0.001, 100, log=True)}

    elif model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 300),
            "max_depth": trial.suggest_int("xgb_max_depth", 2, 8),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
        }

    elif model_type == "random_forest":
        return {
            "n_estimators": trial.suggest_int("rf_n_estimators", 50, 300),
            "max_depth": trial.suggest_int("rf_max_depth", 2, 10),
            "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
        }

    elif model_type == "svm":
        return {
            "C": trial.suggest_float("svm_C", 0.01, 100, log=True),
            "kernel": trial.suggest_categorical("svm_kernel", ["rbf", "linear"]),
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_optuna_study(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_trials: int = 100,
    timeout: int | None = None,
    storage: str | None = None,
    study_name: str = "classifier_hpo",
    include_cross_encoder: bool = False,
) -> optuna.Study:
    """Run Optuna HPO over feature selection + model selection + per-model HPO.

    Parameters
    ----------
    X : Feature matrix (all features).
    y : Binary labels.
    feature_names : Names of all features (must match X columns).
    n_trials : Number of Optuna trials.
    timeout : Max seconds for the study (None = no limit).
    storage : SQLite URL for persistence (e.g. "sqlite:///data/classifier/optuna.db").
    study_name : Name for the Optuna study.
    include_cross_encoder : If True, include cross_encoder_score in feature selection.

    Returns
    -------
    optuna.Study with best trial accessible via study.best_trial.
    """
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    # Build list of selectable features
    selectable = list(feature_names)
    if include_cross_encoder and "cross_encoder_score" not in selectable:
        selectable.append("cross_encoder_score")

    def objective(trial: optuna.Trial) -> float:
        # 1. Feature selection (boolean mask, min 2 features)
        selected_mask = [
            trial.suggest_categorical(f"feat_{name}", [True, False])
            for name in selectable
        ]

        selected_indices = [i for i, sel in enumerate(selected_mask) if sel]
        if len(selected_indices) < 2:
            return 0.0  # Prune: need at least 2 features

        X_sel = X[:, selected_indices] if all(i < X.shape[1] for i in selected_indices) else X[:, [i for i in selected_indices if i < X.shape[1]]]

        # 2. Model selection
        model_type = trial.suggest_categorical(
            "model_type",
            ["logistic_regression", "xgboost", "random_forest", "svm"],
        )

        # 3. Per-model HPO
        params = _suggest_params(trial, model_type)

        # 4. Evaluate via CV (reduced repeats for speed)
        make_estimator = _make_estimator_factory(model_type, params, scale_pos_weight)
        try:
            results = evaluate_cv(
                X_sel, y, make_estimator,
                n_splits=5, n_repeats=3, random_state=42,
            )
        except Exception:
            return 0.0

        return results["cv_metrics"]["f1"]["mean"]

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return study


def best_trial_to_config(
    study: optuna.Study, feature_names: list[str]
) -> dict:
    """Extract the best trial's config into a serialisable dict.

    Returns
    -------
    dict with keys: model_type, hyperparameters, selected_features, threshold,
                    optuna_best_trial, optuna_best_value
    """
    best = study.best_trial

    model_type = best.params["model_type"]
    selected_features = [
        name for name in feature_names
        if best.params.get(f"feat_{name}", False)
    ]

    # Extract model-specific hyperparameters
    prefix_map = {
        "logistic_regression": "lr_",
        "xgboost": "xgb_",
        "random_forest": "rf_",
        "svm": "svm_",
    }
    prefix = prefix_map[model_type]
    hyperparameters = {
        k.removeprefix(prefix): v
        for k, v in best.params.items()
        if k.startswith(prefix)
    }

    return {
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "selected_features": selected_features,
        "optuna_best_trial": best.number,
        "optuna_best_value": best.value,
    }
