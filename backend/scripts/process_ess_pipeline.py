"""Full pipeline for ESS questions: filter → embed → simplify → match → validate.

Runs the same cleaning steps as the Eurobarometer pipeline but for ESS data.

Usage:
    python backend/scripts/process_ess_pipeline.py
    python backend/scripts/process_ess_pipeline.py --skip-simplify  # skip LLM step
"""

import argparse
import re
from pathlib import Path

import numpy as np
import ollama
import pandas as pd
from eu_survey_correlation.embeddings import Embedder
from loguru import logger
from tqdm import tqdm

DATA_DIR = Path("data")
ESS_CSV = DATA_DIR / "surveys" / "ess_survey_questions.csv"
VOTE_EMB = DATA_DIR / "embeddings" / "vote_embeddings.parquet"
VOTE_SUMMARIES_CSV = DATA_DIR / "votes" / "vote_summaries.csv"
VOTES_CSV = DATA_DIR / "votes" / "votes.csv"

# Output paths
ESS_FILTERED = DATA_DIR / "surveys" / "ess_filtered.csv"
ESS_EMB = DATA_DIR / "embeddings" / "ess_embeddings.parquet"
ESS_CLEAN = DATA_DIR / "surveys" / "ess_filtered_clean.csv"

MODEL = "mistral"

# Heuristic exclusions for ESS questions
EXCLUDE_PATTERNS = [
    r"\bparty voted for\b",
    r"\bhighest level of education\b",
    r"\byear of birth\b",
    r"\bgender\b",
    r"\bmarital status\b",
    r"\bhousehold\b.*\bmembers?\b",
    r"\brelationship.*respondent\b",
    r"\bcountry.*birth\b",
    r"\bfather.*born\b",
    r"\bmother.*born\b",
    r"\bcitizenship\b",
    r"\blanguage.*home\b",
    r"\binterview\b.*\bdate\b",
    r"\bweight\b.*\bdesign\b",
    r"\boccupation\b.*\bISCO\b",
    r"\bNACE\b",
    r"\bindustry\b.*\bcode\b",
    r"\btechnical problems\b.*\bvideo\b",
    r"\bCARD \d+\s*$",  # just a card reference, no real question
    r"^\s*ASK IF\b",
    r"\bWRITE IN\b",
    r"\bperson in household\b",
]

SIMPLIFY_PROMPT = """\
Rewrite this survey question into a short, clear, easy-to-read question.

Rules:
- Remove any variable ID prefix
- Remove CARD references (e.g. "CARD 5", "STILL CARD 8")
- Remove interviewer instructions (e.g. "READ OUT", "ASK ALL", "ASK IF")
- Remove scale descriptions (e.g. "using a scale from 0 to 10")
- Keep the core meaning intact
- Maximum 1-2 sentences
- Output ONLY the rewritten question, nothing else

Examples:
Input: "Most people can be trusted or you can't be too careful: CARD 3Using this card, generally speaking, would you say that most people can be trusted, or that you can't be too careful in dealing with people?"
Output: Can most people be trusted, or can't you be too careful in dealing with people?

Input: "Trust in country's parliament: CARD 8Using this card, please tell me on a score of 0-10 how much you personally trust each of the institutions I read out. 0 means you do not trust an institution at all, and 10 means you have complete trust. Country's parliament"
Output: How much do you trust your country's parliament?

Now rewrite this question:
{question}"""


def _extract_embeddings(df: pd.DataFrame) -> np.ndarray:
    emb_cols = sorted(
        [c for c in df.columns if c.startswith("emb_")],
        key=lambda c: int(c.split("_")[1]),
    )
    return df[emb_cols].values.astype(np.float32)


def step_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic + semantic filtering of ESS questions."""
    logger.info(f"Step 1: Filtering {len(df)} ESS questions...")
    original = len(df)

    # Heuristic filter
    mask = pd.Series(True, index=df.index)
    for pattern in EXCLUDE_PATTERNS:
        mask &= ~df["question_en"].str.contains(pattern, case=False, na=False, regex=True)
    df = df[mask].copy()
    logger.info(f"  Heuristic filter: {original} → {len(df)}")

    return df


def step_embed(df: pd.DataFrame) -> pd.DataFrame:
    """Embed ESS questions."""
    logger.info(f"Step 2: Embedding {len(df)} questions...")
    embedder = Embedder()
    embedder.embed_dataframe(
        df=df,
        text_column="question_en",
        output_path=ESS_EMB,
    )
    return pd.read_parquet(ESS_EMB)


def step_semantic_filter(df: pd.DataFrame, emb_df: pd.DataFrame, threshold: float = 0.45) -> pd.DataFrame:
    """Filter by semantic similarity to votes."""
    logger.info("Step 2b: Semantic filtering...")
    vote_emb_df = pd.read_parquet(VOTE_EMB)

    survey_emb = _extract_embeddings(emb_df)
    vote_emb = _extract_embeddings(vote_emb_df)

    survey_emb = survey_emb / (np.linalg.norm(survey_emb, axis=1, keepdims=True) + 1e-9)
    vote_emb = vote_emb / (np.linalg.norm(vote_emb, axis=1, keepdims=True) + 1e-9)

    emb_ids = emb_df["sheet_id"].values
    max_sims = {}
    batch_size = 256
    for start in range(0, len(survey_emb), batch_size):
        end = min(start + batch_size, len(survey_emb))
        batch = survey_emb[start:end]
        sims = batch @ vote_emb.T
        batch_max = sims.max(axis=1)
        for i, idx in enumerate(range(start, end)):
            max_sims[emb_ids[idx]] = float(batch_max[i])

    df = df.copy()
    df["max_vote_similarity"] = df["sheet_id"].map(max_sims)
    before = len(df)
    df = df[df["max_vote_similarity"] >= threshold].copy()
    df = df.drop(columns=["max_vote_similarity"])
    logger.info(f"  Semantic filter (threshold={threshold}): {before} → {len(df)}")
    return df


def step_simplify(df: pd.DataFrame) -> pd.DataFrame:
    """Simplify question text via LLM."""
    logger.info(f"Step 3: Simplifying {len(df)} questions via LLM...")

    # Resume support
    if ESS_CLEAN.exists():
        existing = pd.read_csv(ESS_CLEAN)
        if "question_clean" in existing.columns and len(existing) == len(df):
            logger.info("  Already simplified, skipping.")
            return existing

    clean_questions = []
    start_idx = 0
    if ESS_CLEAN.exists():
        existing = pd.read_csv(ESS_CLEAN)
        if "question_clean" in existing.columns:
            start_idx = len(existing)
            clean_questions = existing["question_clean"].tolist()
            logger.info(f"  Resuming from row {start_idx}")

    for idx in tqdm(range(start_idx, len(df)), desc="Simplifying ESS"):
        row = df.iloc[idx]
        original = str(row["question_en"])
        prompt = SIMPLIFY_PROMPT.format(question=original[:500])
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            result = response["message"]["content"].strip()
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            clean_questions.append(result)
        except Exception as e:
            logger.error(f"LLM failed: {e}")
            clean_questions.append(original[:200])

        # Save every 50
        if (idx + 1) % 50 == 0 or idx == len(df) - 1:
            df_out = df.iloc[: idx + 1].copy()
            df_out["question_clean"] = clean_questions
            df_out.to_csv(ESS_CLEAN, index=False)

    df = df.copy()
    df["question_clean"] = clean_questions
    df.to_csv(ESS_CLEAN, index=False)
    logger.success(f"  Saved simplified questions → {ESS_CLEAN}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Process ESS questions pipeline")
    parser.add_argument("--skip-simplify", action="store_true", help="Skip LLM simplification")
    parser.add_argument("--threshold", type=float, default=0.45, help="Semantic filter threshold")
    args = parser.parse_args()

    # Load ESS questions
    df = pd.read_csv(ESS_CSV)
    df = df.dropna(subset=["question_en"])
    logger.info(f"Loaded {len(df)} ESS questions")

    # Step 1: Heuristic filter
    df = step_filter(df)

    # Save filtered
    df.to_csv(ESS_FILTERED, index=False)

    # Step 2: Embed
    emb_df = step_embed(df)

    # Step 2b: Semantic filter
    df = step_semantic_filter(df, emb_df, threshold=args.threshold)
    df.to_csv(ESS_FILTERED, index=False)
    logger.success(f"Saved {len(df)} filtered ESS questions → {ESS_FILTERED}")

    # Step 3: Simplify
    if not args.skip_simplify:
        df = step_simplify(df)

    logger.success(f"Pipeline complete: {len(df)} ESS questions ready")


if __name__ == "__main__":
    main()
