"""
Train a match-quality classifier on human-labelled survey↔vote pairs.

Usage:
    uv run python backend/scripts/train_classifier.py              # default: LR
    uv run python backend/scripts/train_classifier.py --model lr    # logistic regression
    uv run python backend/scripts/train_classifier.py --model hybrid  # XGBoost + cross-encoder score
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MIGRATION_DIR = DATA / "migration"
EMBEDDING_CACHE = DATA / "cache" / "embeddings.parquet"
OUTPUT_DIR = DATA / "classifier"
FIGURES_DIR = OUTPUT_DIR / "figures"


# ── Feature engineering ───────────────────────────────────────────────
def _word_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def compute_features(row: dict, emb_lookup: dict[str, np.ndarray] | None = None) -> dict:
    """Compute features for a single match pair."""
    q = str(row.get("question_clean") or "")
    v = str(row.get("vote_summary_clean") or row.get("summary_clean") or "")

    features: dict = {
        "similarity_score": float(row.get("similarity_score", 0)),
        "days_between": abs(float(row.get("days_between") or row.get("time_delta") or 0)),
        "len_question": len(q),
        "len_vote_summary": len(v),
        "len_ratio": len(q) / max(len(v), 1),
        "question_has_number": int(bool(re.search(r"\d", q))),
    }

    # Word overlap (Jaccard)
    q_words, v_words = _word_set(q), _word_set(v)
    if q_words or v_words:
        features["word_overlap"] = len(q_words & v_words) / max(len(q_words | v_words), 1)
    else:
        features["word_overlap"] = 0.0

    # Embedding-based features
    if emb_lookup is not None:
        emb_q = emb_lookup.get(q)
        emb_v = emb_lookup.get(v)
        if emb_q is not None and emb_v is not None:
            cos_sim = np.dot(emb_q, emb_v) / (
                np.linalg.norm(emb_q) * np.linalg.norm(emb_v) + 1e-9
            )
            abs_diff = np.abs(emb_q - emb_v)
            features["emb_cosine"] = float(cos_sim)
            features["emb_abs_diff_mean"] = float(abs_diff.mean())
            features["emb_abs_diff_std"] = float(abs_diff.std())
        else:
            features["emb_cosine"] = features["similarity_score"]
            features["emb_abs_diff_mean"] = 0.0
            features["emb_abs_diff_std"] = 0.0
    else:
        features["emb_cosine"] = features["similarity_score"]
        features["emb_abs_diff_mean"] = 0.0
        features["emb_abs_diff_std"] = 0.0

    return features


def build_feature_matrix(
    records: list[dict], emb_lookup: dict[str, np.ndarray] | None = None
) -> pd.DataFrame:
    """Build feature matrix from a list of match records."""
    rows = [compute_features(r, emb_lookup) for r in records]
    return pd.DataFrame(rows)


# ── Data loading ──────────────────────────────────────────────────────
def load_labelled_data() -> list[dict]:
    """Load human-labelled matches from the most recent migration backup."""
    backup_files = sorted(MIGRATION_DIR.glob("survey_vote_matches_backup_*.json"))
    if not backup_files:
        raise FileNotFoundError(f"No backup files in {MIGRATION_DIR}")
    with open(backup_files[-1]) as f:
        all_matches = json.load(f)
    labelled = [r for r in all_matches if r.get("admin_validated") is not None]
    print(f"Loaded {len(labelled)} labelled matches from {backup_files[-1].name}")
    return labelled


def load_embedding_lookup() -> dict[str, np.ndarray]:
    """Load embeddings parquet and build text→vector lookup."""
    if not EMBEDDING_CACHE.exists():
        print(f"Warning: {EMBEDDING_CACHE} not found, embedding features will be zero")
        return {}
    df = pd.read_parquet(EMBEDDING_CACHE)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    lookup = {}
    for _, row in df.iterrows():
        text = str(row["text"])
        lookup[text] = row[emb_cols].values.astype(np.float32)
    print(f"Loaded {len(lookup)} embeddings ({len(emb_cols)} dims)")
    return lookup


# ── Training ──────────────────────────────────────────────────────────
def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes F1."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.1, 0.91, 0.01):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def train_and_evaluate(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """Train LogisticRegression with RepeatedStratifiedKFold, calibrate threshold."""
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


# ── Cross-encoder scoring ─────────────────────────────────────────────
CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CROSS_ENCODER_CACHE = DATA / "cache" / "cross_encoder_scores.npy"


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

    print(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
    ce_model = CrossEncoder(CROSS_ENCODER_MODEL)

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


def score_single_pair_cross_encoder(question: str, vote_summary: str) -> float:
    """Score a single pair with the cross-encoder (no caching)."""
    from sentence_transformers import CrossEncoder

    ce_model = CrossEncoder(CROSS_ENCODER_MODEL)
    return float(ce_model.predict([(question, vote_summary)])[0])


# ── Hybrid training (XGBoost + cross-encoder score) ───────────────────
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


# ── Plot style ────────────────────────────────────────────────────────
STYLE = "seaborn-v0_8-whitegrid"
PAL = {"accepted": "#4C72B0", "refused": "#DD8452", "neutral": "#8C8C8C", "highlight": "#C44E52"}
FONT_TITLE = 14
FONT_LABEL = 12
DPI = 180


def _apply_style() -> None:
    plt.style.use(STYLE)
    plt.rcParams.update({"axes.titlesize": FONT_TITLE, "axes.labelsize": FONT_LABEL})


# ── Reporting ─────────────────────────────────────────────────────────
def generate_report(
    results: dict,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_accepted: int,
    n_refused: int,
    feature_df: pd.DataFrame,
    unlabelled: list[dict] | None = None,
) -> None:
    """Generate figures and report.md."""
    _apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    threshold = results["calibrated_threshold"]
    y_true = results["all_y_true"]
    y_prob = results["all_y_prob"]
    importances = results["feature_importances"]

    # ── Figure 1: Score distributions with KDE ────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    accepted_scores = feature_df.loc[y == 1, "similarity_score"]
    refused_scores = feature_df.loc[y == 0, "similarity_score"]
    bins = np.linspace(
        min(accepted_scores.min(), refused_scores.min()),
        max(accepted_scores.max(), refused_scores.max()),
        30,
    )
    ax.hist(refused_scores, bins=bins, alpha=0.45, label=f"Refused (n={n_refused})",
            color=PAL["refused"], edgecolor="white", linewidth=0.5)
    ax.hist(accepted_scores, bins=bins, alpha=0.45, label=f"Accepted (n={n_accepted})",
            color=PAL["accepted"], edgecolor="white", linewidth=0.5)
    # KDE overlays
    if len(accepted_scores) > 2:
        sns.kdeplot(accepted_scores, ax=ax, color=PAL["accepted"], linewidth=2)
    if len(refused_scores) > 2:
        sns.kdeplot(refused_scores, ax=ax, color=PAL["refused"], linewidth=2)
    ax.axvline(accepted_scores.mean(), color=PAL["accepted"], ls="--", lw=1.5,
               label=f"Accepted mean = {accepted_scores.mean():.3f}")
    ax.axvline(refused_scores.mean(), color=PAL["refused"], ls="--", lw=1.5,
               label=f"Refused mean = {refused_scores.mean():.3f}")
    ax.axvline(threshold, color=PAL["highlight"], ls=":", lw=2,
               label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("Similarity Score")
    ax.set_ylabel("Count")
    ax.set_title("Similarity Score Distribution by Label")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "score_distributions.png", dpi=DPI)
    plt.close(fig)

    # ── Figure 2: Feature importances (abs values + direction markers) ─
    sorted_imp = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    names, vals = zip(*sorted_imp)
    abs_vals = [abs(v) for v in vals]
    colors = [PAL["accepted"] if v > 0 else PAL["refused"] for v in vals]
    bars = ax.barh(names, abs_vals, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        sign = "+" if v > 0 else "-"
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{sign}{abs(v):.3f}", va="center", fontsize=9)
    ax.set_xlabel("|Logistic Regression Coefficient|")
    ax.set_title("Feature Importances (blue = predicts accepted, orange = predicts refused)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importances.png", dpi=DPI)
    plt.close(fig)

    # ── Figure 3: PR curve with operating point ───────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    pr_display = PrecisionRecallDisplay.from_predictions(
        y_true, y_prob, ax=ax, name="LogisticRegression", color=PAL["accepted"],
    )
    # Mark operating point at calibrated threshold
    y_pred_cal = (y_prob >= threshold).astype(int)
    op_prec = precision_score(y_true, y_pred_cal, zero_division=0)
    op_rec = recall_score(y_true, y_pred_cal, zero_division=0)
    ax.plot(op_rec, op_prec, "o", color=PAL["highlight"], markersize=10, zorder=5,
            label=f"Operating point (t={threshold:.2f})")
    ax.annotate(f"P={op_prec:.2f}, R={op_rec:.2f}", (op_rec, op_prec),
                textcoords="offset points", xytext=(10, -15), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=PAL["highlight"]))
    ax.set_title("Precision-Recall Curve")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=DPI)
    plt.close(fig)

    # ── Figure 4: Confusion matrix with row-normalized percentages ────
    y_pred_cal = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_cal)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Refused", "Accepted"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"Counts (threshold={threshold:.2f})")
    ConfusionMatrixDisplay(cm_norm, display_labels=["Refused", "Accepted"]).plot(
        ax=axes[1], cmap="Blues", colorbar=False, values_format=".1%")
    axes[1].set_title("Row-Normalized (%)")
    fig.suptitle("Confusion Matrix", fontsize=FONT_TITLE, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 5: Calibration plot with histogram ─────────────────────
    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    ax_cal.plot(prob_pred, prob_true, "o-", color=PAL["accepted"], label="LR", linewidth=2)
    ax_cal.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax_cal.fill_between(prob_pred, prob_true, prob_pred, alpha=0.15, color=PAL["accepted"])
    ax_cal.set_ylabel("Fraction of positives")
    ax_cal.set_title("Calibration Plot")
    ax_cal.legend(fontsize=9)
    ax_hist.hist(y_prob, bins=20, color=PAL["neutral"], alpha=0.7, edgecolor="white")
    ax_hist.axvline(threshold, color=PAL["highlight"], ls=":", lw=2, label=f"Threshold = {threshold:.2f}")
    ax_hist.set_xlabel("Mean predicted probability")
    ax_hist.set_ylabel("Count")
    ax_hist.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_plot.png", dpi=DPI)
    plt.close(fig)

    # ── Figure 6: Threshold sensitivity ───────────────────────────────
    thresholds_sweep = np.arange(0.10, 0.91, 0.01)
    f1s, precs, recs = [], [], []
    for t in thresholds_sweep:
        yp = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, yp, zero_division=0))
        precs.append(precision_score(y_true, yp, zero_division=0))
        recs.append(recall_score(y_true, yp, zero_division=0))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds_sweep, f1s, label="F1", linewidth=2, color=PAL["accepted"])
    ax.plot(thresholds_sweep, precs, label="Precision", linewidth=2, color=PAL["refused"])
    ax.plot(thresholds_sweep, recs, label="Recall", linewidth=2, color="#55A868")
    ax.axvline(threshold, color=PAL["highlight"], ls=":", lw=2,
               label=f"Calibrated threshold = {threshold:.2f}")
    best_f1_idx = int(np.argmax(f1s))
    ax.annotate(f"Best F1 = {f1s[best_f1_idx]:.3f}",
                (thresholds_sweep[best_f1_idx], f1s[best_f1_idx]),
                textcoords="offset points", xytext=(15, -10), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=PAL["accepted"]))
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sensitivity — F1 / Precision / Recall")
    ax.legend(fontsize=10)
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_sensitivity.png", dpi=DPI)
    plt.close(fig)

    # ── Figure 7: Pipeline architecture diagram ───────────────────────
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.3, "Raw Pair\n(question, vote)", PAL["neutral"]),
        (2.6, f"10 Features\n(compute_features)", PAL["accepted"]),
        (4.9, "StandardScaler\n(zero mean, unit var)", PAL["accepted"]),
        (7.2, f"LogisticRegression\n(C={results['best_C']}, balanced)", PAL["accepted"]),
        (9.5, f"P(accepted)\nthreshold={threshold:.2f}", PAL["refused"]),
        (11.8, "Decision\naccept / refuse", PAL["highlight"]),
    ]
    box_w, box_h = 1.9, 1.6
    for x, label, color in boxes:
        rect = matplotlib.patches.FancyBboxPatch(
            (x, 0.7), box_w, box_h, linewidth=2,
            edgecolor=color, facecolor=color, alpha=0.15,
            boxstyle="round,pad=0.05", zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, 0.7 + box_h / 2, label,
                ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=3)
    # Arrows between boxes
    for i in range(len(boxes) - 1):
        x_start = boxes[i][0] + box_w
        x_end = boxes[i + 1][0]
        ax.annotate("", xy=(x_end, 1.5), xytext=(x_start, 1.5),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))
    ax.set_title("Pipeline Architecture", fontsize=FONT_TITLE, pad=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pipeline_architecture.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 8: Active learning candidates ──────────────────────────
    if unlabelled:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        probs = np.array([u["predicted_probability"] for u in unlabelled])
        ax.hist(probs, bins=25, color=PAL["neutral"], alpha=0.7, edgecolor="white")
        ax.axvline(threshold, color=PAL["highlight"], ls=":", lw=2,
                   label=f"Threshold = {threshold:.2f}")
        # Highlight the uncertainty zone
        ax.axvspan(threshold - 0.1, threshold + 0.1, alpha=0.1, color=PAL["highlight"],
                   label="Uncertainty zone (±0.1)")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")
        ax.set_title(f"Unlabelled Pairs (n={len(unlabelled)}) — Predicted Probability")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "active_learning_candidates.png", dpi=DPI)
        plt.close(fig)

    # ── Generate report.md ────────────────────────────────────────────
    cv = results["cv_metrics"]
    report = f"""# Match Quality Classifier — Report

## Dataset Summary
- **Total labelled pairs**: {n_accepted + n_refused}
- **Accepted**: {n_accepted} ({100*n_accepted/(n_accepted+n_refused):.1f}%)
- **Refused**: {n_refused} ({100*n_refused/(n_accepted+n_refused):.1f}%)
- **Class ratio** (refused:accepted): {n_refused/max(n_accepted,1):.1f}:1
- **Features**: {len(feature_names)}

### Feature Statistics
{feature_df.describe().round(3).to_string()}

## Model: Logistic Regression (C={results['best_C']})

### Cross-Validation Results (5-fold, 10 repeats)

| Metric | Mean | Std |
|--------|------|-----|
| F1 | {cv['f1']['mean']:.3f} | {cv['f1']['std']:.3f} |
| Precision | {cv['precision']['mean']:.3f} | {cv['precision']['std']:.3f} |
| Recall | {cv['recall']['mean']:.3f} | {cv['recall']['std']:.3f} |
| PR-AUC | {cv['pr_auc']['mean']:.3f} | {cv['pr_auc']['std']:.3f} |

**Calibrated threshold**: {threshold:.2f}

## Figures

### Pipeline Architecture
![Pipeline](figures/pipeline_architecture.png)

### Score Distributions
![Score distributions](figures/score_distributions.png)

### Feature Importances
![Feature importances](figures/feature_importances.png)

### Threshold Sensitivity
![Threshold sensitivity](figures/threshold_sensitivity.png)

### Precision-Recall Curve
![PR curve](figures/precision_recall_curve.png)

### Confusion Matrix
![Confusion matrix](figures/confusion_matrix.png)

### Calibration Plot
![Calibration](figures/calibration_plot.png)
"""

    if unlabelled:
        top_candidates = sorted(unlabelled, key=lambda x: abs(x["predicted_probability"] - threshold))[:20]
        report += """
### Active Learning Candidates
![Active learning](figures/active_learning_candidates.png)

**Top 20 pairs to label next** (closest to decision boundary):

| # | Question (truncated) | Vote (truncated) | P(accepted) |
|---|---------------------|-------------------|-------------|
"""
        for i, c in enumerate(top_candidates, 1):
            q = str(c.get("question_clean", ""))[:60]
            v = str(c.get("vote_summary_clean", ""))[:60]
            report += f"| {i} | {q} | {v} | {c['predicted_probability']:.3f} |\n"

    report += """
## Recommendations

1. **If F1 < 0.65**: Add more features (LLM-based semantic relatedness, topic overlap)
2. **If F1 0.65-0.75**: Try SetFit with sentence-transformers for a potential boost
3. **If F1 > 0.75**: Deploy as-is, use predictions in frontend
4. **Active learning**: Label the top uncertainty candidates to improve the model
"""

    with open(OUTPUT_DIR / "report.md", "w") as f:
        f.write(report)
    print(f"Report saved to {OUTPUT_DIR / 'report.md'}")


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    import joblib

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

    # Score unlabelled pairs
    with open(sorted(MIGRATION_DIR.glob("survey_vote_matches_backup_*.json"))[-1]) as f:
        all_matches = json.load(f)
    unlabelled = [r for r in all_matches if r.get("admin_validated") is None]
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
        print(f"\nScored {len(unlabelled)} unlabelled pairs")

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
