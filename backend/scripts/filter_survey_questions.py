"""Filter survey questions to keep only those that could relate to a parliament vote.

Removes:
- Demographic questions (age, gender, occupation, education, region, etc.)
- Personal behavior questions (buying habits, what you do personally)
- Metadata fields (fieldwork country, region codes, weighting variables)
- Questions with no meaningful policy content

Uses two approaches:
1. Heuristic keyword filtering for obvious non-policy questions
2. Semantic similarity: keeps questions whose max cosine similarity to any
   vote summary exceeds a threshold

Usage:
    python backend/scripts/filter_survey_questions.py
    python backend/scripts/filter_survey_questions.py --threshold 0.45
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

DATA_DIR = Path("data")
SURVEY_EMB = DATA_DIR / "embeddings" / "survey_embeddings.parquet"
VOTE_EMB = DATA_DIR / "embeddings" / "vote_embeddings.parquet"
SURVEY_CSV = DATA_DIR / "surveys" / "all_survey_questions.csv"
OUTPUT_CSV = DATA_DIR / "surveys" / "filtered_survey_questions.csv"


# Patterns that indicate demographic / metadata / personal behavior questions
EXCLUDE_PATTERNS = [
    # Demographics
    r"\bhow old are you\b",
    r"\bage\b.*\bcategories\b",
    r"\bgender\b",
    r"\bsex\b",
    r"\bmarital status\b",
    r"\beducation\b.*\bstopped\b",
    r"\bfull.?time education\b",
    r"\bcurrent occupation\b",
    r"\blast occupation\b",
    r"\bpaid work\b",
    r"\bhousehold\b.*\bchildren\b",
    r"\bchildren\b.*\bhousehold\b",
    r"\bchildren aged\b",
    r"\bincome\b.*\bhousehold\b",
    r"\bhousehold\b.*\bincome\b",
    r"\burbanisation\b",
    r"\btype of community\b",
    # Metadata / fieldwork
    r"^B\b.*fieldwork",
    r"^Region$",
    r"\bfieldwork\b",
    r"\bturnover\b.*\bbase\b",
    r"\bnumber of employees\b",
    r"\bcompany size\b",
    r"\bScreening\b",
    # Personal behavior (not policy)
    r"\bhave you (ever )?(bought|purchased|used)\b",
    r"\bhow often do you\b.*\bbuy\b",
    r"\bwhere do you\b.*\bbuy\b",
    r"\bwhich of the following products have you bought\b",
    r"\bhave you seen this logo\b",
    r"\bdo you recogni[sz]e\b",
]

# sheet_id patterns that are typically demographics/metadata
EXCLUDE_SHEET_PATTERNS = [
    r"^D\d+r?\d*$",  # D1, D4, D7, D4r2, etc. (demographic codes)
    r"^SD\d+",  # SD1, SD2, SD6, SD8c, etc. (socio-demographic)
    r"^T\d+$",  # T36, T95, T96, T144, T162 (metadata/region codes when standalone)
    r"^B$",  # B = fieldwork country
    r"^brk_SCR",  # Screening/company variables
]


def _extract_embeddings(df: pd.DataFrame) -> np.ndarray:
    emb_cols = sorted(
        [c for c in df.columns if c.startswith("emb_")],
        key=lambda c: int(c.split("_")[1]),
    )
    return df[emb_cols].values.astype(np.float32)


def heuristic_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Remove questions matching demographic/metadata/personal patterns."""
    original_count = len(df)

    # Filter by question text patterns
    text_mask = pd.Series(True, index=df.index)
    for pattern in EXCLUDE_PATTERNS:
        matches = df["question_en"].str.contains(pattern, case=False, na=False)
        text_mask &= ~matches

    # Filter by sheet_id patterns
    id_mask = pd.Series(True, index=df.index)
    for pattern in EXCLUDE_SHEET_PATTERNS:
        matches = df["sheet_id"].str.match(pattern, case=False, na=False)
        id_mask &= ~matches

    combined_mask = text_mask & id_mask
    filtered = df[combined_mask].copy()

    removed = original_count - len(filtered)
    logger.info(
        f"Heuristic filter: removed {removed} questions "
        f"({removed / original_count:.1%}), kept {len(filtered)}"
    )
    return filtered


def semantic_filter(
    df: pd.DataFrame,
    survey_emb_path: Path,
    vote_emb_path: Path,
    threshold: float = 0.45,
) -> pd.DataFrame:
    """Keep only questions whose max cosine similarity to any vote >= threshold."""
    logger.info("Loading embeddings for semantic filtering...")
    survey_emb_df = pd.read_parquet(survey_emb_path)
    vote_emb_df = pd.read_parquet(vote_emb_path)

    survey_emb = _extract_embeddings(survey_emb_df)
    vote_emb = _extract_embeddings(vote_emb_df)

    # Normalize
    survey_emb = survey_emb / (np.linalg.norm(survey_emb, axis=1, keepdims=True) + 1e-9)
    vote_emb = vote_emb / (np.linalg.norm(vote_emb, axis=1, keepdims=True) + 1e-9)

    # Build mapping from sheet_id to embedding index
    emb_sheet_ids = survey_emb_df["sheet_id"].values
    id_to_emb_idx = {sid: idx for idx, sid in enumerate(emb_sheet_ids)}

    # Compute max similarity per survey question (in batches for memory)
    batch_size = 256
    max_sims = {}

    for start in range(0, len(survey_emb), batch_size):
        end = min(start + batch_size, len(survey_emb))
        batch = survey_emb[start:end]
        sims = batch @ vote_emb.T  # (batch, n_votes)
        batch_max = sims.max(axis=1)
        for i, idx in enumerate(range(start, end)):
            max_sims[emb_sheet_ids[idx]] = float(batch_max[i])

    # Filter: keep questions in df that have max_sim >= threshold
    df = df.copy()
    df["max_vote_similarity"] = df["sheet_id"].map(max_sims)

    # Questions without embeddings (shouldn't happen, but be safe)
    no_emb = df["max_vote_similarity"].isna()
    if no_emb.any():
        logger.warning(f"{no_emb.sum()} questions have no embeddings, dropping them")

    above = df["max_vote_similarity"] >= threshold
    filtered = df[above & ~no_emb].copy()

    logger.info(
        f"Semantic filter (threshold={threshold}): "
        f"kept {len(filtered)}/{len(df)} questions"
    )
    logger.info(
        f"Max similarity stats: "
        f"mean={df['max_vote_similarity'].mean():.3f}, "
        f"median={df['max_vote_similarity'].median():.3f}, "
        f"min={df['max_vote_similarity'].min():.3f}, "
        f"max={df['max_vote_similarity'].max():.3f}"
    )

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Filter survey questions to keep only policy-relevant ones"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Min cosine similarity to any vote summary (default: 0.45)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_CSV),
        help=f"Output path (default: {OUTPUT_CSV})",
    )
    args = parser.parse_args()

    logger.info(f"Loading survey questions from {SURVEY_CSV}...")
    df = pd.read_csv(SURVEY_CSV)
    df = df.dropna(subset=["question_en"])
    logger.info(f"Loaded {len(df)} questions (after dropping NaN)")

    # Step 1: heuristic filter
    df = heuristic_filter(df)

    # Step 2: semantic filter
    df = semantic_filter(
        df,
        survey_emb_path=SURVEY_EMB,
        vote_emb_path=VOTE_EMB,
        threshold=args.threshold,
    )

    # Drop helper column before saving
    df = df.drop(columns=["max_vote_similarity"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Saved {len(df)} filtered questions → {output_path}")

    # Show some examples of kept questions
    logger.info("Sample kept questions:")
    for _, row in df.sample(min(10, len(df)), random_state=0).iterrows():
        logger.info(f"  [{row.sheet_id}] {str(row.question_en)[:120]}")


if __name__ == "__main__":
    main()
