"""Tests for the match quality classifier pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from eu_survey_correlation.classifier import (
    build_feature_matrix,
    compute_features,
    find_best_threshold,
    train_and_evaluate,
)


# ── Fixtures ──────────────────────────────────────────────────────────
@pytest.fixture
def sample_record():
    return {
        "question_clean": "Should the EU invest more in renewable energy sources?",
        "vote_summary_clean": "The European Parliament voted to increase investment in renewable energy by 30%.",
        "similarity_score": 0.65,
        "days_between": 120,
        "admin_validated": True,
    }


@pytest.fixture
def sample_records():
    """A small set of labelled records for testing."""
    records = []
    for i in range(50):
        accepted = i < 15  # ~30% accepted
        records.append({
            "question_clean": f"Question about topic {i} in EU policy?",
            "vote_summary_clean": f"The Parliament voted on topic {i} regarding EU policy." if accepted
                                  else f"Unrelated vote summary about procedure {i*7}.",
            "similarity_score": 0.5 + (0.15 if accepted else -0.05) + np.random.uniform(-0.1, 0.1),
            "days_between": np.random.randint(10, 1000),
            "admin_validated": accepted,
        })
    return records


@pytest.fixture
def embedding_lookup():
    """Mock embedding lookup with 384-dim vectors."""
    rng = np.random.RandomState(42)
    return {
        "Should the EU invest more in renewable energy sources?": rng.randn(384).astype(np.float32),
        "The European Parliament voted to increase investment in renewable energy by 30%.": rng.randn(384).astype(np.float32),
    }


# ── Test feature engineering ──────────────────────────────────────────
class TestFeatureEngineering:
    def test_compute_features_returns_correct_keys(self, sample_record):
        features = compute_features(sample_record)
        expected_keys = {
            "similarity_score", "days_between", "len_question", "len_vote_summary",
            "len_ratio", "word_overlap", "question_has_number", "emb_cosine",
            "emb_abs_diff_mean", "emb_abs_diff_std",
        }
        assert set(features.keys()) == expected_keys

    def test_compute_features_no_nan(self, sample_record):
        features = compute_features(sample_record)
        for k, v in features.items():
            assert not np.isnan(v), f"Feature {k} is NaN"

    def test_compute_features_with_embeddings(self, sample_record, embedding_lookup):
        features = compute_features(sample_record, emb_lookup=embedding_lookup)
        # emb_cosine should differ from similarity_score when embeddings are provided
        assert "emb_cosine" in features
        assert isinstance(features["emb_cosine"], float)
        assert -1.0 <= features["emb_cosine"] <= 1.0

    def test_compute_features_empty_text(self):
        record = {
            "question_clean": "",
            "vote_summary_clean": "",
            "similarity_score": 0.0,
            "days_between": 0,
        }
        features = compute_features(record)
        assert features["len_question"] == 0
        assert features["word_overlap"] == 0.0

    def test_build_feature_matrix_shape(self, sample_records):
        df = build_feature_matrix(sample_records)
        assert df.shape[0] == len(sample_records)
        assert df.shape[1] == 10  # 10 features
        assert not df.isnull().any().any(), "Feature matrix contains NaN"

    def test_word_overlap_identical(self):
        record = {
            "question_clean": "renewable energy EU",
            "vote_summary_clean": "renewable energy EU",
            "similarity_score": 0.9,
            "days_between": 0,
        }
        features = compute_features(record)
        assert features["word_overlap"] == 1.0

    def test_word_overlap_disjoint(self):
        record = {
            "question_clean": "alpha beta gamma",
            "vote_summary_clean": "delta epsilon zeta",
            "similarity_score": 0.1,
            "days_between": 0,
        }
        features = compute_features(record)
        assert features["word_overlap"] == 0.0

    def test_question_has_number(self):
        record_with = {
            "question_clean": "Should 25% of budget go to energy?",
            "vote_summary_clean": "Vote on budget.",
            "similarity_score": 0.5,
            "days_between": 0,
        }
        record_without = {
            "question_clean": "Should budget go to energy?",
            "vote_summary_clean": "Vote on budget.",
            "similarity_score": 0.5,
            "days_between": 0,
        }
        assert compute_features(record_with)["question_has_number"] == 1
        assert compute_features(record_without)["question_has_number"] == 0


# ── Test threshold calibration ────────────────────────────────────────
class TestThresholdCalibration:
    def test_finds_reasonable_threshold(self):
        rng = np.random.RandomState(42)
        y_true = np.array([1]*30 + [0]*70)
        y_prob = np.where(y_true == 1, rng.uniform(0.5, 0.9, 100), rng.uniform(0.1, 0.6, 100))
        threshold = find_best_threshold(y_true, y_prob)
        assert 0.1 <= threshold <= 0.9

    def test_threshold_maximizes_f1(self):
        # Perfect separation case
        y_true = np.array([0]*50 + [1]*50)
        y_prob = np.array([0.1]*50 + [0.9]*50)
        threshold = find_best_threshold(y_true, y_prob)
        # Threshold should be between 0.1 and 0.9
        assert 0.1 < threshold < 0.9


# ── Test training pipeline ───────────────────────────────────────────
class TestTraining:
    def test_train_runs_without_error(self, sample_records):
        df = build_feature_matrix(sample_records)
        y = np.array([1 if r["admin_validated"] else 0 for r in sample_records])
        X = df.values.astype(np.float64)
        feature_names = list(df.columns)

        results = train_and_evaluate(X, y, feature_names)

        assert "model" in results
        assert "cv_metrics" in results
        assert "feature_importances" in results
        assert "calibrated_threshold" in results

    def test_cv_metrics_structure(self, sample_records):
        df = build_feature_matrix(sample_records)
        y = np.array([1 if r["admin_validated"] else 0 for r in sample_records])
        X = df.values.astype(np.float64)

        results = train_and_evaluate(X, y, list(df.columns))
        cv = results["cv_metrics"]

        for metric in ["f1", "precision", "recall", "pr_auc"]:
            assert metric in cv
            assert "mean" in cv[metric]
            assert "std" in cv[metric]
            assert 0 <= cv[metric]["mean"] <= 1

    def test_model_predicts_probabilities(self, sample_records):
        df = build_feature_matrix(sample_records)
        y = np.array([1 if r["admin_validated"] else 0 for r in sample_records])
        X = df.values.astype(np.float64)

        results = train_and_evaluate(X, y, list(df.columns))
        model = results["model"]

        probs = model.predict_proba(X)[:, 1]
        assert len(probs) == len(X)
        assert all(0 <= p <= 1 for p in probs)

    def test_feature_importances_match_features(self, sample_records):
        df = build_feature_matrix(sample_records)
        y = np.array([1 if r["admin_validated"] else 0 for r in sample_records])
        X = df.values.astype(np.float64)
        feature_names = list(df.columns)

        results = train_and_evaluate(X, y, feature_names)
        assert set(results["feature_importances"].keys()) == set(feature_names)
