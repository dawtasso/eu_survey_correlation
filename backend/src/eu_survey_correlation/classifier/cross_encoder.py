"""Cross-encoder scoring for survey-vote pairs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import DATA

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
