"""Feature engineering for survey-vote match pairs."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

ALL_FEATURE_NAMES = [
    "similarity_score",
    "days_between",
    "len_question",
    "len_vote_summary",
    "len_ratio",
    "question_has_number",
    "word_overlap",
    "emb_cosine",
    "emb_abs_diff_mean",
    "emb_abs_diff_std",
]


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
