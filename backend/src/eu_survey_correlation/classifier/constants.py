"""Shared paths and plot style constants for the classifier pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[4]  # eu_survey_correlation/
DATA = ROOT / "data"
MIGRATION_DIR = DATA / "migration"
EMBEDDING_CACHE = DATA / "cache" / "embeddings.parquet"
OUTPUT_DIR = DATA / "classifier"
FIGURES_DIR = OUTPUT_DIR / "figures"

# ── Plot style ────────────────────────────────────────────────────────
STYLE = "seaborn-v0_8-whitegrid"
PAL = {"accepted": "#4C72B0", "refused": "#DD8452", "neutral": "#8C8C8C", "highlight": "#C44E52"}
FONT_TITLE = 14
FONT_LABEL = 12
DPI = 180
