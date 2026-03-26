"""
Benchmark multiple classifiers for survey↔vote pair matching.

Compares: LR baseline, XGBoost, Cross-encoder zero-shot, Hybrid, Cross-encoder fine-tuned.
All models evaluated on identical CV folds for fair comparison.

Usage:
    uv run python backend/scripts/benchmark_classifiers.py
    uv run python backend/scripts/benchmark_classifiers.py --skip-finetune
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Import shared code from train_classifier
sys.path.insert(0, str(Path(__file__).parent))
from train_classifier import (
    DATA,
    FIGURES_DIR,
    OUTPUT_DIR,
    build_feature_matrix,
    find_best_threshold,
    load_embedding_lookup,
    load_labelled_data,
)

CACHE_DIR = DATA / "cache"
CROSS_ENCODER_CACHE = CACHE_DIR / "cross_encoder_scores.npy"

# Plot style
STYLE = "seaborn-v0_8-whitegrid"
COLORS = {
    "LR baseline": "#4C72B0",
    "XGBoost": "#DD8452",
    "Cross-encoder zero-shot": "#55A868",
    "Hybrid (CE + XGB)": "#C44E52",
    "Cross-encoder fine-tuned": "#8172B3",
}
DPI = 180


# ── Cross-encoder scoring ─────────────────────────────────────────────
def compute_cross_encoder_scores(
    records: list[dict], cache_path: Path = CROSS_ENCODER_CACHE
) -> np.ndarray:
    """Score all pairs with a multilingual cross-encoder. Cache results."""
    if cache_path.exists():
        scores = np.load(cache_path)
        if len(scores) == len(records):
            print(f"Loaded {len(scores)} cached cross-encoder scores from {cache_path.name}")
            return scores
        print(f"Cache size mismatch ({len(scores)} vs {len(records)}), recomputing...")

    from sentence_transformers import CrossEncoder

    model_name = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    print(f"Loading cross-encoder: {model_name}")
    ce_model = CrossEncoder(model_name)

    pairs = []
    for r in records:
        q = str(r.get("question_clean") or "")
        v = str(r.get("vote_summary_clean") or r.get("summary_clean") or "")
        pairs.append((q, v))

    print(f"Scoring {len(pairs)} pairs with cross-encoder...")
    scores = ce_model.predict(pairs, show_progress_bar=True)
    scores = np.array(scores, dtype=np.float64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, scores)
    print(f"Cached cross-encoder scores to {cache_path.name}")
    return scores


# ── Model evaluation helpers ──────────────────────────────────────────
def evaluate_folds_with_proba(
    y: np.ndarray,
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
    train_and_predict_fn,
) -> dict:
    """Evaluate a model that produces probabilities, with per-fold threshold calibration.

    train_and_predict_fn(train_idx, val_idx) -> y_prob for validation set
    """
    fold_metrics = []
    all_y_true, all_y_prob = [], []

    for train_idx, val_idx in fold_indices:
        y_prob = train_and_predict_fn(train_idx, val_idx)
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

    metrics_df = pd.DataFrame(fold_metrics)
    return {
        "cv_metrics": {
            col: {"mean": float(metrics_df[col].mean()), "std": float(metrics_df[col].std())}
            for col in ["f1", "precision", "recall", "pr_auc", "threshold"]
        },
        "all_y_true": np.array(all_y_true),
        "all_y_prob": np.array(all_y_prob),
    }


# ── Model A: LR baseline ─────────────────────────────────────────────
def benchmark_lr(
    X: np.ndarray, y: np.ndarray, fold_indices: list
) -> dict:
    """Logistic Regression baseline — reproduces existing pipeline."""
    # Inner CV for C tuning
    best_C, best_f1 = 1.0, 0.0
    for C_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
        f1s = []
        inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
        for tr, va in inner_cv.split(X, y):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(C=C_val, class_weight="balanced", max_iter=1000, random_state=42)),
            ])
            pipe.fit(X[tr], y[tr])
            f1s.append(f1_score(y[va], pipe.predict(X[va]), zero_division=0))
        if np.mean(f1s) > best_f1:
            best_C, best_f1 = C_val, np.mean(f1s)

    print(f"  LR best C={best_C} (inner F1={best_f1:.3f})")

    def train_predict(train_idx, val_idx):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=best_C, class_weight="balanced", max_iter=1000, random_state=42)),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        return pipe.predict_proba(X[val_idx])[:, 1]

    return evaluate_folds_with_proba(y, fold_indices, train_predict)


# ── Model B: XGBoost ──────────────────────────────────────────────────
def benchmark_xgboost(
    X: np.ndarray, y: np.ndarray, fold_indices: list
) -> dict:
    """XGBoost on the same 10 features."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos = n_neg / max(n_pos, 1)

    # Grid search over a few configs using inner CV
    best_params, best_f1 = {}, 0.0
    inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)

    for n_est in [50, 100, 200]:
        for max_d in [3, 5, 7]:
            for lr in [0.05, 0.1, 0.2]:
                f1s = []
                for tr, va in inner_cv.split(X, y):
                    xgb = XGBClassifier(
                        n_estimators=n_est, max_depth=max_d, learning_rate=lr,
                        scale_pos_weight=scale_pos, random_state=42,
                        eval_metric="logloss", verbosity=0,
                    )
                    xgb.fit(X[tr], y[tr])
                    f1s.append(f1_score(y[va], xgb.predict(X[va]), zero_division=0))
                mean_f1 = np.mean(f1s)
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best_params = {"n_estimators": n_est, "max_depth": max_d, "learning_rate": lr}

    print(f"  XGBoost best params: {best_params} (inner F1={best_f1:.3f})")

    def train_predict(train_idx, val_idx):
        xgb = XGBClassifier(
            **best_params, scale_pos_weight=scale_pos, random_state=42,
            eval_metric="logloss", verbosity=0,
        )
        xgb.fit(X[train_idx], y[train_idx])
        return xgb.predict_proba(X[val_idx])[:, 1]

    return evaluate_folds_with_proba(y, fold_indices, train_predict)


# ── Model C: Cross-encoder zero-shot ──────────────────────────────────
def benchmark_cross_encoder_zeroshot(
    ce_scores: np.ndarray, y: np.ndarray, fold_indices: list
) -> dict:
    """Cross-encoder scores used directly — CV only calibrates threshold."""

    def train_predict(train_idx, val_idx):
        # No training; just return the pre-computed scores for the val set
        return ce_scores[val_idx]

    return evaluate_folds_with_proba(y, fold_indices, train_predict)


# ── Model D: Hybrid (CE score + XGBoost) ──────────────────────────────
def benchmark_hybrid(
    X: np.ndarray, ce_scores: np.ndarray, y: np.ndarray, fold_indices: list
) -> dict:
    """Cross-encoder score as 11th feature + XGBoost."""
    X_hybrid = np.column_stack([X, ce_scores])

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos = n_neg / max(n_pos, 1)

    # Simpler grid search for the hybrid
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

    print(f"  Hybrid best params: {best_params} (inner F1={best_f1:.3f})")

    def train_predict(train_idx, val_idx):
        xgb = XGBClassifier(
            **best_params, scale_pos_weight=scale_pos, random_state=42,
            eval_metric="logloss", verbosity=0,
        )
        xgb.fit(X_hybrid[train_idx], y[train_idx])
        return xgb.predict_proba(X_hybrid[val_idx])[:, 1]

    return evaluate_folds_with_proba(y, fold_indices, train_predict)


# ── Model E: Cross-encoder fine-tuned ─────────────────────────────────
def benchmark_cross_encoder_finetuned(
    records: list[dict], y: np.ndarray, fold_indices_reduced: list
) -> dict:
    """Fine-tune cross-encoder within each CV fold (reduced folds for speed)."""
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
    from datasets import Dataset
    import tempfile

    model_name = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # Prepare text pairs
    pairs_q = []
    pairs_v = []
    for r in records:
        pairs_q.append(str(r.get("question_clean") or ""))
        pairs_v.append(str(r.get("vote_summary_clean") or r.get("summary_clean") or ""))

    fold_metrics = []
    all_y_true, all_y_prob = [], []

    for fold_i, (train_idx, val_idx) in enumerate(fold_indices_reduced):
        print(f"    Fine-tune fold {fold_i + 1}/{len(fold_indices_reduced)}...")

        # Build training dataset
        train_data = {
            "sentence1": [pairs_q[i] for i in train_idx],
            "sentence2": [pairs_v[i] for i in train_idx],
            "label": [float(y[i]) for i in train_idx],
        }
        train_dataset = Dataset.from_dict(train_data)

        # Build eval dataset
        eval_data = {
            "sentence1": [pairs_q[i] for i in val_idx],
            "sentence2": [pairs_v[i] for i in val_idx],
            "label": [float(y[i]) for i in val_idx],
        }
        eval_dataset = Dataset.from_dict(eval_data)

        # Fresh model each fold
        ce_model = CrossEncoder(model_name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = CrossEncoderTrainingArguments(
                output_dir=tmp_dir,
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                learning_rate=2e-5,
                weight_decay=0.01,
                warmup_ratio=0.1,
                logging_steps=50,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )

            trainer = CrossEncoderTrainer(
                model=ce_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
            trainer.train()

        # Predict on validation set
        val_pairs = [(pairs_q[i], pairs_v[i]) for i in val_idx]
        y_prob = ce_model.predict(val_pairs)
        y_prob = np.array(y_prob, dtype=np.float64)

        # Sigmoid if raw logits (values outside [0,1])
        if y_prob.min() < 0 or y_prob.max() > 1:
            y_prob = 1 / (1 + np.exp(-y_prob))

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

    metrics_df = pd.DataFrame(fold_metrics)
    return {
        "cv_metrics": {
            col: {"mean": float(metrics_df[col].mean()), "std": float(metrics_df[col].std())}
            for col in ["f1", "precision", "recall", "pr_auc", "threshold"]
        },
        "all_y_true": np.array(all_y_true),
        "all_y_prob": np.array(all_y_prob),
    }


# ── Reporting ─────────────────────────────────────────────────────────
def print_comparison_table(results: dict[str, dict]) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 90)
    print(f"{'Model':<30} {'F1':>10} {'Precision':>12} {'Recall':>10} {'PR-AUC':>10}")
    print("-" * 90)
    for name, res in results.items():
        cv = res["cv_metrics"]
        print(
            f"{name:<30} "
            f"{cv['f1']['mean']:.3f}±{cv['f1']['std']:.3f} "
            f"{cv['precision']['mean']:.3f}±{cv['precision']['std']:.3f} "
            f"{cv['recall']['mean']:.3f}±{cv['recall']['std']:.3f} "
            f"{cv['pr_auc']['mean']:.3f}±{cv['pr_auc']['std']:.3f}"
        )
    print("=" * 90)

    # Highlight winner
    best_name = max(results, key=lambda n: results[n]["cv_metrics"]["f1"]["mean"])
    best_f1 = results[best_name]["cv_metrics"]["f1"]["mean"]
    print(f"\nWinner: {best_name} (F1={best_f1:.3f})")


def plot_benchmark_comparison(results: dict[str, dict], output_path: Path) -> None:
    """Bar chart comparing F1, Precision, Recall across models."""
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(12, 6))

    models = list(results.keys())
    metrics = ["f1", "precision", "recall"]
    x = np.arange(len(models))
    width = 0.25

    for i, metric in enumerate(metrics):
        means = [results[m]["cv_metrics"][metric]["mean"] for m in models]
        stds = [results[m]["cv_metrics"][metric]["std"] for m in models]
        bars = ax.bar(x + i * width, means, width, yerr=stds, label=metric.upper(),
                      capsize=3, alpha=0.85)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Benchmark Comparison — F1 / Precision / Recall")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved comparison chart to {output_path}")


def plot_pr_curves(results: dict[str, dict], output_path: Path) -> None:
    """Overlaid PR curves for all models."""
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, res in results.items():
        y_true = res["all_y_true"]
        y_prob = res["all_y_prob"]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        color = COLORS.get(name, "#333333")
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})", color=color, linewidth=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend(fontsize=9, loc="best")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved PR curves to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark classifiers for pair matching")
    parser.add_argument("--skip-finetune", action="store_true",
                        help="Skip the cross-encoder fine-tuning (slow on CPU)")
    args = parser.parse_args()

    # Load data
    labelled = load_labelled_data()
    emb_lookup = load_embedding_lookup()

    # Build features
    feature_df = build_feature_matrix(labelled, emb_lookup)
    y = np.array([1 if r["admin_validated"] else 0 for r in labelled])
    X = feature_df.values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)

    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    print(f"\nDataset: {len(y)} pairs ({n_pos} pos / {n_neg} neg)")
    print(f"Features: {list(feature_df.columns)}")
    print(f"Feature matrix shape: {X.shape}\n")

    # Generate shared CV folds
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    fold_indices = [(train_idx, val_idx) for train_idx, val_idx in cv.split(X, y)]
    print(f"Generated {len(fold_indices)} CV folds (5×10)\n")

    # Reduced folds for fine-tuning
    cv_reduced = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    fold_indices_reduced = [(train_idx, val_idx) for train_idx, val_idx in cv_reduced.split(X, y)]

    results: dict[str, dict] = {}

    # ── Model A: LR baseline ──────────────────────────────────────────
    print("Running Model A: LR baseline...")
    t0 = time.time()
    results["LR baseline"] = benchmark_lr(X, y, fold_indices)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # ── Model B: XGBoost ──────────────────────────────────────────────
    print("Running Model B: XGBoost...")
    t0 = time.time()
    results["XGBoost"] = benchmark_xgboost(X, y, fold_indices)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # ── Compute cross-encoder scores (shared by C and D) ──────────────
    print("Computing cross-encoder scores...")
    t0 = time.time()
    ce_scores = compute_cross_encoder_scores(labelled)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # ── Model C: Cross-encoder zero-shot ──────────────────────────────
    print("Running Model C: Cross-encoder zero-shot...")
    t0 = time.time()
    results["Cross-encoder zero-shot"] = benchmark_cross_encoder_zeroshot(ce_scores, y, fold_indices)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # ── Model D: Hybrid ───────────────────────────────────────────────
    print("Running Model D: Hybrid (CE + XGB)...")
    t0 = time.time()
    results["Hybrid (CE + XGB)"] = benchmark_hybrid(X, ce_scores, y, fold_indices)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # ── Model E: Cross-encoder fine-tuned ─────────────────────────────
    if not args.skip_finetune:
        print("Running Model E: Cross-encoder fine-tuned (reduced folds)...")
        t0 = time.time()
        results["Cross-encoder fine-tuned"] = benchmark_cross_encoder_finetuned(
            labelled, y, fold_indices_reduced
        )
        print(f"  Done in {time.time() - t0:.1f}s\n")
    else:
        print("Skipping Model E: Cross-encoder fine-tuned (--skip-finetune)\n")

    # ── Print comparison ──────────────────────────────────────────────
    print_comparison_table(results)

    # ── Save results ──────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # JSON results (without numpy arrays)
    json_results = {}
    for name, res in results.items():
        json_results[name] = res["cv_metrics"]
    with open(OUTPUT_DIR / "benchmark_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'benchmark_results.json'}")

    # Figures
    plot_benchmark_comparison(results, FIGURES_DIR / "benchmark_comparison.png")
    plot_pr_curves(results, FIGURES_DIR / "benchmark_pr_curves.png")


if __name__ == "__main__":
    main()
