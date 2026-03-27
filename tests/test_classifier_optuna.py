"""Tests for the Optuna HPO classifier pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eu_survey_correlation.classifier import build_feature_matrix, evaluate_cv
from eu_survey_correlation.classifier.optuna_search import (
    _make_estimator_factory,
    best_trial_to_config,
    run_optuna_study,
)


@pytest.fixture
def sample_records():
    """A small set of labelled records for testing."""
    rng = np.random.RandomState(42)
    records = []
    for i in range(50):
        accepted = i < 15
        records.append({
            "question_clean": f"Question about topic {i} in EU policy?",
            "vote_summary_clean": f"The Parliament voted on topic {i} regarding EU policy." if accepted
                                  else f"Unrelated vote summary about procedure {i*7}.",
            "similarity_score": 0.5 + (0.15 if accepted else -0.05) + rng.uniform(-0.1, 0.1),
            "days_between": rng.randint(10, 1000),
            "admin_validated": accepted,
        })
    return records


@pytest.fixture
def feature_data(sample_records):
    """Build X, y, feature_names from sample records."""
    df = build_feature_matrix(sample_records)
    y = np.array([1 if r["admin_validated"] else 0 for r in sample_records])
    X = df.values.astype(np.float64)
    feature_names = list(df.columns)
    return X, y, feature_names


class TestEstimatorFactory:
    """Test that all model type factories produce valid estimators."""

    @pytest.mark.parametrize("model_type,params", [
        ("logistic_regression", {"C": 1.0}),
        ("random_forest", {"n_estimators": 50, "max_depth": 3, "min_samples_split": 2}),
        ("svm", {"C": 1.0, "kernel": "rbf"}),
    ])
    def test_model_supports_predict_proba(self, model_type, params, feature_data):
        X, y, _ = feature_data
        factory = _make_estimator_factory(model_type, params, scale_pos_weight=2.0)
        estimator = factory()
        estimator.fit(X, y)
        probs = estimator.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_xgboost_supports_predict_proba(self, feature_data):
        X, y, _ = feature_data
        params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1,
                  "subsample": 0.8, "colsample_bytree": 0.8}
        factory = _make_estimator_factory("xgboost", params, scale_pos_weight=2.0)
        estimator = factory()
        estimator.fit(X, y)
        probs = estimator.predict_proba(X)
        assert probs.shape == (len(X), 2)


class TestEvaluateCV:
    """Test the generic CV runner."""

    def test_evaluate_cv_returns_valid_metrics(self, feature_data):
        X, y, _ = feature_data

        def make_lr():
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)),
            ])

        results = evaluate_cv(X, y, make_lr, n_splits=3, n_repeats=2)

        assert "cv_metrics" in results
        assert "calibrated_threshold" in results
        cv = results["cv_metrics"]
        for metric in ["f1", "precision", "recall", "pr_auc"]:
            assert 0 <= cv[metric]["mean"] <= 1
            assert cv[metric]["std"] >= 0


class TestOptunaStudy:
    """Test Optuna HPO (small number of trials)."""

    def test_single_trial_returns_valid_score(self, feature_data):
        X, y, feature_names = feature_data
        study = run_optuna_study(
            X, y, feature_names,
            n_trials=3,
            storage=None,  # in-memory
        )
        assert len(study.trials) == 3
        assert study.best_value >= 0.0

    def test_feature_selection_min_2(self, feature_data):
        X, y, feature_names = feature_data
        study = run_optuna_study(X, y, feature_names, n_trials=5, storage=None)
        config = best_trial_to_config(study, feature_names)
        assert len(config["selected_features"]) >= 2

    def test_best_trial_to_config_keys(self, feature_data):
        X, y, feature_names = feature_data
        study = run_optuna_study(X, y, feature_names, n_trials=5, storage=None)
        config = best_trial_to_config(study, feature_names)

        assert "model_type" in config
        assert config["model_type"] in ["logistic_regression", "xgboost", "random_forest", "svm"]
        assert "hyperparameters" in config
        assert isinstance(config["hyperparameters"], dict)
        assert "selected_features" in config
        assert isinstance(config["selected_features"], list)
        assert "optuna_best_trial" in config
        assert isinstance(config["optuna_best_trial"], int)
        assert "optuna_best_value" in config
        assert isinstance(config["optuna_best_value"], float)
