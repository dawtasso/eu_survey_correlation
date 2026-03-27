"""Data loading utilities for the classifier pipeline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .constants import EMBEDDING_CACHE, MIGRATION_DIR


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
    """Load embeddings parquet and build text->vector lookup."""
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
