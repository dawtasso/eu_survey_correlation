"""Report and figure generation for the classifier pipeline."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .constants import DPI, FIGURES_DIR, FONT_LABEL, FONT_TITLE, OUTPUT_DIR, PAL, STYLE


def _apply_style() -> None:
    plt.style.use(STYLE)
    plt.rcParams.update({"axes.titlesize": FONT_TITLE, "axes.labelsize": FONT_LABEL})


def _plot_score_distributions(
    feature_df: pd.DataFrame, y: np.ndarray, threshold: float,
    n_accepted: int, n_refused: int,
) -> None:
    """Figure 1: Score distributions with KDE."""
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


def _plot_feature_importances(importances: dict[str, float], model_type: str = "lr") -> None:
    """Figure 2: Feature importances."""
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
    label = "Coefficient" if model_type == "lr" else "Importance"
    ax.set_xlabel(f"|{label}|")
    ax.set_title(f"Feature Importances (blue = predicts accepted, orange = predicts refused)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importances.png", dpi=DPI)
    plt.close(fig)


def _plot_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, prefix: str = "",
) -> None:
    """Figure 3: PR curve with operating point."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    PrecisionRecallDisplay.from_predictions(
        y_true, y_prob, ax=ax, name="Classifier", color=PAL["accepted"],
    )
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
    fig.savefig(FIGURES_DIR / f"{prefix}precision_recall_curve.png", dpi=DPI)
    plt.close(fig)


def _plot_confusion_matrix(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, prefix: str = "",
) -> None:
    """Figure 4: Confusion matrix with row-normalized percentages."""
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
    fig.savefig(FIGURES_DIR / f"{prefix}confusion_matrix.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, prefix: str = "",
) -> None:
    """Figure 5: Calibration plot with histogram."""
    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    ax_cal.plot(prob_pred, prob_true, "o-", color=PAL["accepted"], label="Classifier", linewidth=2)
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
    fig.savefig(FIGURES_DIR / f"{prefix}calibration_plot.png", dpi=DPI)
    plt.close(fig)


def _plot_threshold_sensitivity(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, prefix: str = "",
) -> None:
    """Figure 6: Threshold sensitivity."""
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
    fig.savefig(FIGURES_DIR / f"{prefix}threshold_sensitivity.png", dpi=DPI)
    plt.close(fig)


def _plot_pipeline_architecture(results: dict, threshold: float) -> None:
    """Figure 7: Pipeline architecture diagram."""
    model_type = results.get("model_type", "lr")
    if model_type == "lr":
        model_label = f"LogisticRegression\n(C={results.get('best_C', '?')}, balanced)"
    else:
        model_label = f"{model_type}\n(optimized)"

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.3, "Raw Pair\n(question, vote)", PAL["neutral"]),
        (2.6, f"Features\n(compute_features)", PAL["accepted"]),
        (4.9, "StandardScaler\n(zero mean, unit var)", PAL["accepted"]),
        (7.2, model_label, PAL["accepted"]),
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
    for i in range(len(boxes) - 1):
        x_start = boxes[i][0] + box_w
        x_end = boxes[i + 1][0]
        ax.annotate("", xy=(x_end, 1.5), xytext=(x_start, 1.5),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))
    ax.set_title("Pipeline Architecture", fontsize=FONT_TITLE, pad=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pipeline_architecture.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_active_learning(unlabelled: list[dict], threshold: float) -> None:
    """Figure 8: Active learning candidates."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    probs = np.array([u["predicted_probability"] for u in unlabelled])
    ax.hist(probs, bins=25, color=PAL["neutral"], alpha=0.7, edgecolor="white")
    ax.axvline(threshold, color=PAL["highlight"], ls=":", lw=2,
               label=f"Threshold = {threshold:.2f}")
    ax.axvspan(threshold - 0.1, threshold + 0.1, alpha=0.1, color=PAL["highlight"],
               label="Uncertainty zone (+-0.1)")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_title(f"Unlabelled Pairs (n={len(unlabelled)}) — Predicted Probability")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "active_learning_candidates.png", dpi=DPI)
    plt.close(fig)


def generate_report(
    results: dict,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_accepted: int,
    n_refused: int,
    feature_df: pd.DataFrame,
    unlabelled: list[dict] | None = None,
    setfit_results: dict | None = None,
) -> None:
    """Generate figures and report.md."""
    _apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    threshold = results["calibrated_threshold"]
    y_true = results["all_y_true"]
    y_prob = results["all_y_prob"]
    importances = results["feature_importances"]
    model_type = results.get("model_type", "lr")

    _plot_score_distributions(feature_df, y, threshold, n_accepted, n_refused)
    _plot_feature_importances(importances, model_type)
    _plot_pr_curve(y_true, y_prob, threshold)
    _plot_confusion_matrix(y_true, y_prob, threshold)
    _plot_calibration(y_true, y_prob, threshold)
    _plot_threshold_sensitivity(y_true, y_prob, threshold)
    _plot_pipeline_architecture(results, threshold)

    if unlabelled:
        _plot_active_learning(unlabelled, threshold)

    # ── Generate report.md ────────────────────────────────────────────
    cv = results["cv_metrics"]

    # Model description
    if model_type == "lr":
        model_desc = f"Logistic Regression (C={results.get('best_C', '?')})"
    else:
        model_desc = f"{model_type} (Optuna-optimized)" if results.get("optuna_best_trial") else model_type

    selected = results.get("selected_features")
    features_line = f"{len(selected)} (selected by Optuna)" if selected else f"{len(feature_names)}"

    report = f"""# Match Quality Classifier — Report

## Dataset Summary
- **Total labelled pairs**: {n_accepted + n_refused}
- **Accepted**: {n_accepted} ({100*n_accepted/(n_accepted+n_refused):.1f}%)
- **Refused**: {n_refused} ({100*n_refused/(n_accepted+n_refused):.1f}%)
- **Class ratio** (refused:accepted): {n_refused/max(n_accepted,1):.1f}:1
- **Features**: {features_line}

### Feature Statistics
{feature_df.describe().round(3).to_string()}

## Model: {model_desc}

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

    if setfit_results:
        sf_cv = setfit_results["cv_metrics"]
        sf_threshold = setfit_results["calibrated_threshold"]
        report += f"""
## SetFit Comparison

| Metric | Optuna ({model_type}) | SetFit |
|--------|----------------------|--------|
| F1 | {cv['f1']['mean']:.3f} +/- {cv['f1']['std']:.3f} | {sf_cv['f1']['mean']:.3f} +/- {sf_cv['f1']['std']:.3f} |
| Precision | {cv['precision']['mean']:.3f} +/- {cv['precision']['std']:.3f} | {sf_cv['precision']['mean']:.3f} +/- {sf_cv['precision']['std']:.3f} |
| Recall | {cv['recall']['mean']:.3f} +/- {cv['recall']['std']:.3f} | {sf_cv['recall']['mean']:.3f} +/- {sf_cv['recall']['std']:.3f} |
| PR-AUC | {cv['pr_auc']['mean']:.3f} +/- {cv['pr_auc']['std']:.3f} | {sf_cv['pr_auc']['mean']:.3f} +/- {sf_cv['pr_auc']['std']:.3f} |
| Threshold | {threshold:.2f} | {sf_threshold:.2f} |
"""

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


def generate_setfit_report(
    eval_results: dict,
    n_accepted: int,
    n_refused: int,
) -> None:
    """Generate SetFit-specific figures and setfit_report.md.

    Called by both `make setfit` and `make retrain-setfit` so the latest
    run always produces the same artifacts.
    """
    _apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cv = eval_results["cv_metrics"]
    threshold = eval_results["calibrated_threshold"]
    y_true = eval_results["all_y_true"]
    y_prob = eval_results["all_y_prob"]

    prefix = "setfit_"
    _plot_pr_curve(y_true, y_prob, threshold, prefix=prefix)
    _plot_confusion_matrix(y_true, y_prob, threshold, prefix=prefix)
    _plot_calibration(y_true, y_prob, threshold, prefix=prefix)
    _plot_threshold_sensitivity(y_true, y_prob, threshold, prefix=prefix)

    report = f"""# SetFit Classifier — Report

## Dataset Summary
- **Total labelled pairs**: {n_accepted + n_refused}
- **Accepted**: {n_accepted} ({100*n_accepted/(n_accepted+n_refused):.1f}%)
- **Refused**: {n_refused} ({100*n_refused/(n_accepted+n_refused):.1f}%)
- **Class ratio** (refused:accepted): {n_refused/max(n_accepted,1):.1f}:1

## Model: SetFit (all-MiniLM-L6-v2)

### Evaluation Results

| Metric | Mean | Std |
|--------|------|-----|
| F1 | {cv['f1']['mean']:.3f} | {cv['f1']['std']:.3f} |
| Precision | {cv['precision']['mean']:.3f} | {cv['precision']['std']:.3f} |
| Recall | {cv['recall']['mean']:.3f} | {cv['recall']['std']:.3f} |
| PR-AUC | {cv['pr_auc']['mean']:.3f} | {cv['pr_auc']['std']:.3f} |

**Calibrated threshold**: {threshold:.2f}

## Figures

### Precision-Recall Curve
![PR curve](figures/{prefix}precision_recall_curve.png)

### Confusion Matrix
![Confusion matrix](figures/{prefix}confusion_matrix.png)

### Threshold Sensitivity
![Threshold sensitivity](figures/{prefix}threshold_sensitivity.png)

### Calibration Plot
![Calibration](figures/{prefix}calibration_plot.png)
"""

    with open(OUTPUT_DIR / "setfit_report.md", "w") as f:
        f.write(report)
    print(f"SetFit report saved to {OUTPUT_DIR / 'setfit_report.md'}")
