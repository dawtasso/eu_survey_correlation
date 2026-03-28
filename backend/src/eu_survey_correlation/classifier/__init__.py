"""Classifier package for survey-vote match quality prediction."""

from .constants import (
    DATA,
    DPI,
    EMBEDDING_CACHE,
    FIGURES_DIR,
    MIGRATION_DIR,
    OUTPUT_DIR,
    PAL,
    ROOT,
)
from .cross_encoder import (
    CROSS_ENCODER_CACHE,
    CROSS_ENCODER_MODEL,
    compute_cross_encoder_scores,
    score_single_pair_cross_encoder,
)
from .data import load_embedding_lookup, load_labelled_data
from .features import ALL_FEATURE_NAMES, build_feature_matrix, compute_features
from .reporting import generate_report, generate_setfit_report
from .setfit_model import (
    SETFIT_MODEL_DIR,
    load_setfit,
    predict_setfit,
    save_setfit,
    train_setfit_eval,
    train_setfit_final,
)
from .training import (
    evaluate_cv,
    find_best_threshold,
    train_and_evaluate,
    train_and_evaluate_hybrid,
    train_final_model,
)

__all__ = [
    # constants
    "ROOT",
    "DATA",
    "MIGRATION_DIR",
    "EMBEDDING_CACHE",
    "OUTPUT_DIR",
    "FIGURES_DIR",
    "PAL",
    "DPI",
    # features
    "ALL_FEATURE_NAMES",
    "compute_features",
    "build_feature_matrix",
    # data
    "load_labelled_data",
    "load_embedding_lookup",
    # training
    "find_best_threshold",
    "evaluate_cv",
    "train_final_model",
    "train_and_evaluate",
    "train_and_evaluate_hybrid",
    # cross_encoder
    "CROSS_ENCODER_MODEL",
    "CROSS_ENCODER_CACHE",
    "compute_cross_encoder_scores",
    "score_single_pair_cross_encoder",
    # reporting
    "generate_report",
    "generate_setfit_report",
    # setfit
    "SETFIT_MODEL_DIR",
    "train_setfit_eval",
    "train_setfit_final",
    "predict_setfit",
    "save_setfit",
    "load_setfit",
]
