"""SetFit few-shot classifier for survey-vote match quality.

Fine-tunes `all-MiniLM-L6-v2` with contrastive learning + logistic head.
Designed as an alternative to the hand-crafted-feature LR/XGBoost pipeline.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

from .constants import OUTPUT_DIR
from .training import find_best_threshold

SETFIT_MODEL_DIR = OUTPUT_DIR / "setfit_model"
SETFIT_LOGS_DIR = OUTPUT_DIR / "setfit_logs"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _make_texts(records: list[dict]) -> list[str]:
    """Build input texts: 'question [SEP] vote_summary'."""
    return [
        f"{r.get('question_clean', '')} [SEP] {r.get('vote_summary_clean', '')}"
        for r in records
    ]


def _train_one(
    texts: list[str],
    labels: list[int],
) -> object:
    """Train a single SetFit model on given texts/labels."""
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    train_ds = Dataset.from_dict({"text": texts, "label": labels})

    model = SetFitModel.from_pretrained(BASE_MODEL)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = str(SETFIT_LOGS_DIR / run_name)
    training_args = TrainingArguments(
        num_epochs=2,
        batch_size=16,
        num_iterations=20,
        logging_dir=log_dir,
        run_name=run_name,
        report_to="tensorboard",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )
    trainer.train()
    return model


def _compute_fold_metrics(y_true, y_prob):
    """Compute metrics for a single fold given true labels and predicted probs."""
    threshold = find_best_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob),
        "threshold": threshold,
    }


def train_setfit_eval(
    labelled: list[dict],
    kfold: bool = False,
    n_splits: int = 5,
    n_repeats: int = 3,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Evaluate SetFit and return metrics + calibrated threshold.

    Default: single stratified 80/20 split (fast).
    With kfold=True: RepeatedStratifiedKFold (slow but more robust).
    """
    from eu_survey_correlation.logging import log

    # Clean previous TensorBoard logs
    if SETFIT_LOGS_DIR.exists():
        shutil.rmtree(SETFIT_LOGS_DIR)

    texts = _make_texts(labelled)
    y = np.array([1 if r["admin_validated"] else 0 for r in labelled])

    if kfold:
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        total_folds = n_splits * n_repeats
    else:
        cv = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        total_folds = 1

    fold_metrics = []
    all_y_true, all_y_prob = [], []
    thresholds = []

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(texts, y)):
        log.info(f"SetFit fold {fold_i + 1}/{total_folds}")

        train_texts = [texts[i] for i in train_idx]
        train_labels = y[train_idx].tolist()
        val_texts = [texts[i] for i in val_idx]

        model = _train_one(train_texts, train_labels)
        y_prob = predict_setfit(model, val_texts)
        metrics = _compute_fold_metrics(y[val_idx], y_prob)

        fold_metrics.append(metrics)
        all_y_true.extend(y[val_idx])
        all_y_prob.extend(y_prob)
        thresholds.append(metrics["threshold"])

    import pandas as pd

    metrics_df = pd.DataFrame(fold_metrics)
    calibrated_threshold = float(np.median(thresholds))

    return {
        "cv_metrics": {
            col: {
                "mean": float(metrics_df[col].mean()),
                "std": float(metrics_df[col].std()) if len(fold_metrics) > 1 else 0.0,
            }
            for col in ["f1", "precision", "recall", "pr_auc", "threshold"]
        },
        "all_y_true": np.array(all_y_true),
        "all_y_prob": np.array(all_y_prob),
        "calibrated_threshold": calibrated_threshold,
    }


def train_setfit_final(labelled: list[dict]) -> object:
    """Train a final SetFit model on all labelled data."""
    texts = _make_texts(labelled)
    labels = [1 if r["admin_validated"] else 0 for r in labelled]
    return _train_one(texts, labels)


def predict_setfit(model: object, texts_or_records: list) -> np.ndarray:
    """Return P(accept) for a list of texts or record dicts.

    Accepts either raw text strings or record dicts (with question_clean / vote_summary_clean).
    """
    if not texts_or_records:
        return np.array([])

    # If first element is a dict, convert to texts
    if isinstance(texts_or_records[0], dict):
        texts = _make_texts(texts_or_records)
    else:
        texts = texts_or_records

    probs = model.predict_proba(texts)
    # predict_proba returns (n, 2) — take column 1 for P(accept)
    probs = np.array(probs)
    if probs.ndim == 2:
        return probs[:, 1]
    return probs


def save_setfit(model: object, path: Path | None = None) -> Path:
    """Save a trained SetFit model to disk."""
    path = path or SETFIT_MODEL_DIR
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    return path


def load_setfit(path: Path | None = None) -> object:
    """Load a saved SetFit model from disk."""
    from setfit import SetFitModel

    path = path or SETFIT_MODEL_DIR
    return SetFitModel.from_pretrained(str(path))
