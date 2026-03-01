"""Match ESS survey questions to parliament votes with temporal filtering.

Mirrors match_surveys_to_votes.py but for ESS data. Produces ESS-specific
matches, then merges with existing Eurobarometer matches into a combined file
with a 'source' column.

Usage:
    python backend/scripts/match_ess_to_votes.py
    python backend/scripts/match_ess_to_votes.py --top-k 3 --threshold 0.50
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import ollama
import pandas as pd
from loguru import logger
from tqdm import tqdm

DATA_DIR = Path("data")
ESS_CLEAN_CSV = DATA_DIR / "surveys" / "ess_filtered_clean.csv"
ESS_EMB = DATA_DIR / "embeddings" / "ess_embeddings.parquet"
VOTE_EMB = DATA_DIR / "embeddings" / "vote_embeddings.parquet"
VOTES_CSV = DATA_DIR / "votes" / "votes.csv"
VOTE_SUMMARIES_CSV = DATA_DIR / "votes" / "vote_summaries.csv"

EB_MATCHES_CSV = DATA_DIR / "matches" / "survey_vote_matches_clean.csv"
ESS_MATCHES_CSV = DATA_DIR / "matches" / "ess_vote_matches_clean.csv"
COMBINED_OUTPUT = DATA_DIR / "matches" / "survey_vote_matches_clean.csv"

MODEL = "mistral"

SIMPLIFY_VOTE_PROMPT = """\
Rewrite this European Parliament vote summary into a short, clear description.

Rules:
- Maximum 2 sentences
- Focus on WHAT was voted on and the outcome (passed/rejected)
- Remove procedural details (vote counts, amendment numbers, article references)
- Remove group names (EPP, S&D, etc.) unless essential
- Keep the core policy topic clear
- Output ONLY the rewritten summary, nothing else

Vote summary:
{summary}"""


def _extract_embeddings(df: pd.DataFrame) -> np.ndarray:
    emb_cols = sorted(
        [c for c in df.columns if c.startswith("emb_")],
        key=lambda c: int(c.split("_")[1]),
    )
    return df[emb_cols].values.astype(np.float32)


def simplify_vote_summary(summary: str, cache: dict) -> str:
    if not summary or summary == "nan" or len(summary.strip()) < 10:
        return summary
    if summary in cache:
        return cache[summary]

    prompt = SIMPLIFY_VOTE_PROMPT.format(summary=summary[:1500])
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        result = response["message"]["content"].strip()
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        cache[summary] = result
        return result
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return summary[:200]


def main():
    parser = argparse.ArgumentParser(description="Match ESS questions to votes")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    # 1. Load ESS clean questions
    logger.info("Loading ESS filtered clean questions...")
    ess = pd.read_csv(ESS_CLEAN_CSV)
    ess = ess.dropna(subset=["question_clean"])
    logger.info(f"Loaded {len(ess)} ESS questions")

    # ESS already has survey_date
    ess["survey_date"] = pd.to_datetime(ess["survey_date"], errors="coerce")
    ess = ess[ess["survey_date"].notna()].copy()
    logger.info(f"With valid dates: {len(ess)}")

    # 2. Load vote data
    logger.info("Loading votes...")
    votes_full = pd.read_csv(VOTES_CSV)
    vote_summaries = pd.read_csv(VOTE_SUMMARIES_CSV)
    votes_full["vote_date"] = pd.to_datetime(
        votes_full["timestamp"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    vote_date_map = dict(zip(votes_full["id"], votes_full["vote_date"]))
    vote_id_to_summary = dict(zip(vote_summaries["vote_id"], vote_summaries["summary"]))

    # 3. Load embeddings
    logger.info("Loading embeddings...")
    ess_emb_df = pd.read_parquet(ESS_EMB)
    vote_emb_df = pd.read_parquet(VOTE_EMB)

    ess_emb = _extract_embeddings(ess_emb_df)
    vote_emb = _extract_embeddings(vote_emb_df)

    ess_emb = ess_emb / (np.linalg.norm(ess_emb, axis=1, keepdims=True) + 1e-9)
    vote_emb = vote_emb / (np.linalg.norm(vote_emb, axis=1, keepdims=True) + 1e-9)

    # Build index maps
    ess_id_to_emb_idx = {
        sid: idx for idx, sid in enumerate(ess_emb_df["sheet_id"].values)
    }

    # 4. Compute matches
    valid_ess_ids = set(ess["sheet_id"].values)
    ess_lookup = ess.set_index("sheet_id")

    logger.info(f"Computing matches (top_k={args.top_k}, threshold={args.threshold})...")
    rows = []
    batch_size = 256

    for start in tqdm(range(0, len(ess_emb), batch_size), desc="Matching ESS→Votes"):
        end = min(start + batch_size, len(ess_emb))
        batch = ess_emb[start:end]
        sims = batch @ vote_emb.T

        for i in range(sims.shape[0]):
            emb_idx = start + i
            sheet_id = ess_emb_df.iloc[emb_idx]["sheet_id"]

            if sheet_id not in valid_ess_ids:
                continue

            survey_row = ess_lookup.loc[sheet_id]
            if isinstance(survey_row, pd.DataFrame):
                survey_row = survey_row.iloc[0]

            survey_date = survey_row["survey_date"]
            scores = sims[i]

            top_k = args.top_k
            if top_k < len(scores):
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
            else:
                top_indices = np.arange(len(scores))

            top_indices = top_indices[scores[top_indices] >= args.threshold]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            for vote_idx in top_indices:
                vote_id = int(vote_emb_df.iloc[vote_idx]["vote_id"])
                vote_date = vote_date_map.get(vote_id)

                if vote_date is None or pd.isna(vote_date):
                    continue
                if survey_date >= vote_date:
                    continue

                rows.append({
                    "question_id": sheet_id,
                    "question_clean": survey_row.get("question_clean", ""),
                    "question_original": survey_row.get("question_en", ""),
                    "survey_file": survey_row.get("file_name", ""),
                    "survey_date": survey_date.strftime("%Y-%m-%d"),
                    "vote_id": vote_id,
                    "vote_summary_original": str(vote_id_to_summary.get(vote_id, ""))[:500],
                    "vote_date": vote_date.strftime("%Y-%m-%d"),
                    "days_between": (vote_date - survey_date).days,
                    "similarity_score": float(scores[vote_idx]),
                    "source": "ESS",
                })

    if not rows:
        logger.warning("No ESS matches found!")
        return

    ess_matches = pd.DataFrame(rows)
    ess_matches = ess_matches.sort_values("similarity_score", ascending=False)
    logger.info(
        f"Found {len(ess_matches)} ESS matches "
        f"({ess_matches['question_id'].nunique()} questions, "
        f"{ess_matches['vote_id'].nunique()} votes)"
    )

    # 5. Simplify vote summaries
    unique_vote_ids = ess_matches["vote_id"].unique()
    logger.info(f"Simplifying {len(unique_vote_ids)} vote summaries...")

    vote_clean_cache_path = DATA_DIR / "votes" / "vote_summaries_clean_cache.json"
    vote_clean_cache = {}
    if vote_clean_cache_path.exists():
        with open(vote_clean_cache_path) as f:
            vote_clean_cache = json.load(f)
        logger.info(f"Loaded {len(vote_clean_cache)} cached simplifications")

    for vid in tqdm(unique_vote_ids, desc="Simplifying votes"):
        vid_str = str(vid)
        if vid_str in vote_clean_cache:
            continue
        original = str(vote_id_to_summary.get(vid, ""))
        simplified = simplify_vote_summary(original, {})
        vote_clean_cache[vid_str] = simplified

        if len(vote_clean_cache) % 20 == 0:
            with open(vote_clean_cache_path, "w") as f:
                json.dump(vote_clean_cache, f, indent=2, ensure_ascii=False)

    with open(vote_clean_cache_path, "w") as f:
        json.dump(vote_clean_cache, f, indent=2, ensure_ascii=False)

    ess_matches["vote_summary_clean"] = ess_matches["vote_id"].apply(
        lambda vid: vote_clean_cache.get(str(vid), "")
    )

    # 6. Save ESS matches
    ESS_MATCHES_CSV.parent.mkdir(parents=True, exist_ok=True)
    ess_matches.to_csv(ESS_MATCHES_CSV, index=False)
    logger.success(f"Saved {len(ess_matches)} ESS matches → {ESS_MATCHES_CSV}")

    # 7. Merge with EB matches
    logger.info("Merging with Eurobarometer matches...")
    eb_matches = pd.read_csv(EB_MATCHES_CSV)
    eb_matches["source"] = "Eurobarometer"

    combined = pd.concat([eb_matches, ess_matches], ignore_index=True)
    combined = combined.sort_values("similarity_score", ascending=False)
    combined.to_csv(COMBINED_OUTPUT, index=False)
    logger.success(
        f"Combined: {len(combined)} matches "
        f"({len(eb_matches)} EB + {len(ess_matches)} ESS) → {COMBINED_OUTPUT}"
    )


if __name__ == "__main__":
    main()
