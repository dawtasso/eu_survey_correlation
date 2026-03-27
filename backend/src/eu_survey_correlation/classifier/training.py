"""Training and evaluation routines for the classifier."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes F1."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.1, 0.91, 0.01):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def evaluate_cv(
    X: np.ndarray,
    y: np.ndarray,
    make_estimator: Callable[[], object],
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict:
    """Generic CV runner — takes a factory callable that returns an sklearn estimator.

    The estimator must implement fit(X, y) and predict_proba(X).
    Returns dict with cv_metrics, all_y_true, all_y_prob, thresholds.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    fold_metrics = []
    all_y_true, all_y_prob = [], []
    thresholds = []

    for train_idx, val_idx in cv.split(X, y):
        estimator = make_estimator()
        estimator.fit(X[train_idx], y[train_idx])
        y_prob = estimator.predict_proba(X[val_idx])[:, 1]
        threshold = find_best_threshold(y[val_idx], y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        fold_metrics.append({
            "f1": f1_score(y[val_idx], y_pred, zero_division=0),
            "precision": precision_score(y[val_idx], y_pred, zero_division=0),
            "recall": recall_score(y[val_idx], y_pred, zero_division=0),
            "pr_auc": average_precision_score(y[val_idx], y_prob),
            "threshold": threshold,
        })
        all_y_true.extend(y[val_idx])
        all_y_prob.extend(y_prob)
        thresholds.append(threshold)

    metrics_df = pd.DataFrame(fold_metrics)
    calibrated_threshold = float(np.median(thresholds))

    return {
        "cv_metrics": {
            col: {"mean": float(metrics_df[col].mean()), "std": float(metrics_df[col].std())}
            for col in ["f1", "precision", "recall", "pr_auc", "threshold"]
        },
        "all_y_true": np.array(all_y_true),
        "all_y_prob": np.array(all_y_prob),
        "calibrated_threshold": calibrated_threshold,
    }


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    make_estimator: Callable[[], object],
    feature_names: list[str],
) -> dict:
    """Train a final model on all data and return model + feature importances."""
    estimator = make_estimator()
    estimator.fit(X, y)

    # Extract feature importances
    importance = _extract_importances(estimator, feature_names)

    return {
        "model": estimator,
        "feature_importances": importance,
    }


def _extract_importances(estimator: object, feature_names: list[str]) -> dict[str, float]:
    """Extract feature importances from various estimator types."""
    # Handle Pipeline
    model = estimator
    if hasattr(model, "named_steps"):
        # Get the last step of the pipeline
        last_step_name = list(model.named_steps.keys())[-1]
        model = model.named_steps[last_step_name]

    if hasattr(model, "coef_"):
        # Linear model (LR, SVM)
        coefs = model.coef_[0]
        return dict(zip(feature_names, coefs.tolist()))
    elif hasattr(model, "feature_importances_"):
        # Tree-based (XGBoost, RandomForest)
        return dict(zip(feature_names, model.feature_importances_.tolist()))
    else:
        return {name: 0.0 for name in feature_names}


def train_and_evaluate(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """Train LogisticRegression with RepeatedStratifiedKFold, calibrate threshold.

    Legacy wrapper that preserves the original API.
    """
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    # Store per-fold metrics
    fold_metrics = []
    all_y_true, all_y_prob = [], []
    thresholds = []

    # Tune C via inner logic: try a few values, pick best mean F1
    best_C, best_C_f1 = 1.0, 0.0
    for C_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
        f1s = []
        inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
        for train_idx, val_idx in inner_cv.split(X, y):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(C=C_val, class_weight="balanced", max_iter=1000, random_state=42)),
            ])
            pipe.fit(X[train_idx], y[train_idx])
            pred = pipe.predict(X[val_idx])
            f1s.append(f1_score(y[val_idx], pred, zero_division=0))
        mean_f1 = np.mean(f1s)
        if mean_f1 > best_C_f1:
            best_C, best_C_f1 = C_val, mean_f1

    print(f"Best C={best_C} (inner CV F1={best_C_f1:.3f})")

    # Outer CV with best C
    for train_idx, val_idx in cv.split(X, y):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=best_C, class_weight="balanced", max_iter=1000, random_state=42)),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        y_prob = pipe.predict_proba(X[val_idx])[:, 1]
        threshold = find_best_threshold(y[val_idx], y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        fold_metrics.append({
            "f1": f1_score(y[val_idx], y_pred, zero_division=0),
            "precision": precision_score(y[val_idx], y_pred, zero_division=0),
            "recall": recall_score(y[val_idx], y_pred, zero_division=0),
            "pr_auc": average_precision_score(y[val_idx], y_prob),
            "threshold": threshold,
        })
        all_y_true.extend(y[val_idx])
        all_y_prob.extend(y_prob)
        thresholds.append(threshold)

    # Train final model on all data
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=best_C, class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    final_pipe.fit(X, y)

    # Feature importances from LR coefficients
    coefs = final_pipe.named_steps["lr"].coef_[0]
    importance = dict(zip(feature_names, coefs.tolist()))

    metrics_df = pd.DataFrame(fold_metrics)
    calibrated_threshold = float(np.median(thresholds))

    results = {
        "model": final_pipe,
        "best_C": best_C,
        "calibrated_threshold": calibrated_threshold,
        "cv_metrics": {
            col: {"mean": float(metrics_df[col].mean()), "std": float(metrics_df[col].std())}
            for col in ["f1", "precision", "recall", "pr_auc", "threshold"]
        },
        "feature_importances": importance,
        "all_y_true": np.array(all_y_true),
        "all_y_prob": np.array(all_y_prob),
    }

    return results


def train_and_evaluate_hybrid(
    X: np.ndarray, ce_scores: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> dict:
    """Train XGBoost on features + cross-encoder score with RepeatedStratifiedKFold."""
    from xgboost import XGBClassifier

    X_hybrid = np.column_stack([X, ce_scores])
    hybrid_feature_names = feature_names + ["cross_encoder_score"]

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos = n_neg / max(n_pos, 1)

    # Inner CV for hyperparameter tuning
    best_params, best_f1 = {}, 0.0
    inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    for n_est in [50, 100, 200]:
        for max_d in [3, 5, 7]:
            for lr in [0.05, 0.1, 0.2]:
                f1s = []
                for tr, va in inner_cv.split(X_hybrid, y):
                    xgb = XGBClassifier(
                        n_estimators=n_est, max_depth=max_d, learning_rate=lr,
                        scale_pos_weight=scale_pos, random_state=42,
                        eval_metric="logloss", verbosity=0,
                    )
                    xgb.fit(X_hybrid[tr], y[tr])
                    f1s.append(f1_score(y[va], xgb.predict(X_hybrid[va]), zero_division=0))
                mean_f1 = np.mean(f1s)
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best_params = {"n_estimators": n_est, "max_depth": max_d, "learning_rate": lr}

    print(f"Hybrid best params: {best_params} (inner F1={best_f1:.3f})")

    # Outer CV
    fold_metrics = []
    all_y_true, all_y_prob = [], []
    thresholds = []

    for train_idx, val_idx in cv.split(X_hybrid, y):
        xgb = XGBClassifier(
            **best_params, scale_pos_weight=scale_pos, random_state=42,
            eval_metric="logloss", verbosity=0,
        )
        xgb.fit(X_hybrid[train_idx], y[train_idx])
        y_prob = xgb.predict_proba(X_hybrid[val_idx])[:, 1]
        threshold = find_best_threshold(y[val_idx], y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        fold_metrics.append({
            "f1": f1_score(y[val_idx], y_pred, zero_division=0),
            "precision": precision_score(y[val_idx], y_pred, zero_division=0),
            "recall": recall_score(y[val_idx], y_pred, zero_division=0),
            "pr_auc": average_precision_score(y[val_idx], y_prob),
            "threshold": threshold,
        })
        all_y_true.extend(y[val_idx])
        all_y_prob.extend(y_prob)
        thresholds.append(threshold)

    # Train final model on all data
    final_model = XGBClassifier(
        **best_params, scale_pos_weight=scale_pos, random_state=42,
        eval_metric="logloss", verbosity=0,
    )
    final_model.fit(X_hybrid, y)

    # Feature importances from XGBoost
    importance = dict(zip(hybrid_feature_names, final_model.feature_importances_.tolist()))

    metrics_df = pd.DataFrame(fold_metrics)
    calibrated_threshold = float(np.median(thresholds))

    return {
        "model": final_model,
        "model_type": "hybrid",
        "best_params": best_params,
        "calibrated_threshold": calibrated_threshold,
        "cv_metrics": {
            col: {"mean": float(metrics_df[col].mean()), "std": float(metrics_df[col].std())}
            for col in ["f1", "precision", "recall", "pr_auc", "threshold"]
        },
        "feature_importances": importance,
        "all_y_true": np.array(all_y_true),
        "all_y_prob": np.array(all_y_prob),
    }
