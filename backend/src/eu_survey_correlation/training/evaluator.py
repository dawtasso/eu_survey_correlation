"""Evaluation metrics for the survey-vote relatedness classifier."""

from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.6,
    output_path: Path | None = None,
    feature_importances: np.ndarray | None = None,
) -> dict:
    """
    Evaluate model performance on regression and classification metrics.

    Args:
        y_true: Ground truth scores (0-1 continuous).
        y_pred: Predicted scores (0-1 continuous).
        threshold: Binarization threshold for classification metrics.
        output_path: If provided, save the report to this file.
        feature_importances: Optional feature importance array from the model.

    Returns:
        Dict with all computed metrics.
    """
    # Clamp predictions to [0, 1]
    y_pred = np.clip(y_pred, 0.0, 1.0)

    # --- Regression metrics ---
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # --- Classification metrics ---
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    cls_report = classification_report(
        y_true_binary, y_pred_binary, target_names=["unrelated", "related"]
    )

    metrics = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # --- Build report ---
    lines = [
        "=" * 60,
        "SURVEY-VOTE RELATEDNESS CLASSIFIER — EVALUATION REPORT",
        "=" * 60,
        "",
        "REGRESSION METRICS",
        f"  MSE:  {mse:.4f}",
        f"  MAE:  {mae:.4f}",
        f"  R²:   {r2:.4f}",
        "",
        f"CLASSIFICATION METRICS (threshold={threshold})",
        f"  Precision: {precision:.4f}",
        f"  Recall:    {recall:.4f}",
        f"  F1:        {f1:.4f}",
        "",
        "Confusion Matrix:",
        f"  TN={cm[0][0]:>4}  FP={cm[0][1]:>4}",
        f"  FN={cm[1][0]:>4}  TP={cm[1][1]:>4}",
        "",
        "Classification Report:",
        cls_report,
    ]

    # --- Feature importance ---
    if feature_importances is not None:
        top_k = 20
        top_indices = np.argsort(feature_importances)[::-1][:top_k]
        lines.append(f"TOP-{top_k} FEATURE IMPORTANCES")

        feature_names = ["cosine_sim"] + [
            f"prod_{i}" for i in range(384)
        ] + [
            f"diff_{i}" for i in range(384)
        ]

        for rank, idx in enumerate(top_indices, 1):
            name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
            lines.append(f"  {rank:>2}. {name:<12} {feature_importances[idx]:.4f}")
        lines.append("")

    report = "\n".join(lines)

    # Print to stdout
    logger.info(f"\n{report}")

    # Save to file
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        logger.success(f"Report saved → {output_path}")

    return metrics
