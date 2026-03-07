"""Match survey questions to parliament votes with temporal filtering.

Only keeps matches where the survey was published BEFORE the vote.
Also simplifies vote summaries via LLM for readability.

Usage:
    python backend/scripts/match_surveys_to_votes.py
    python backend/scripts/match_surveys_to_votes.py --top-k 3 --threshold 0.50
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import ollama
import pandas as pd
from loguru import logger
from tqdm import tqdm

DATA_DIR = Path("data")
SURVEY_CLEAN_CSV = DATA_DIR / "surveys" / "simplified_michlou_survey_tri.csv"
SURVEY_EMB = DATA_DIR / "embeddings" / "survey_embeddings.parquet"
VOTE_EMB = DATA_DIR / "embeddings" / "vote_embeddings.parquet"
VOTES_CSV = DATA_DIR / "votes" / "votes.csv"
VOTE_SUMMARIES_CSV = DATA_DIR / "votes" / "vote_summaries.csv"
DISTRIBUTIONS_META = DATA_DIR / "surveys" / "distributions_metadata.json"
OUTPUT_CSV = DATA_DIR / "matches" / "simplified_michlou_survey_vote_matches_clean.csv"

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


def build_survey_date_mapping() -> dict[str, str]:
    """Build mapping from survey file_name -> publication date (YYYY-MM-DD)."""
    with open(DISTRIBUTIONS_META) as f:
        data = json.load(f)

    mapping = {}
    for item in data:
        title_en = item.get("title", {}).get("en", "") or item.get("title", {}).get(
            "fr", ""
        )
        issued = item.get("issued", "")
        if not issued or not title_en:
            continue

        # Only volume_B (not BP)
        if "volume_B" not in title_en or "volume_BP" in title_en:
            continue

        date_str = issued[:10]

        # Normalize title to match file_name in our CSV
        # Remove "Link to " / "Lien vers " prefix
        name = title_en
        for prefix in ["Link to ", "Lien vers ", "Lien_vers_"]:
            name = name.replace(prefix, "")
        # Remove extension
        name = re.sub(r"\.(zip|xlsx|xls)$", "", name, flags=re.I)

        # Try to match against our file names by extracting key identifiers
        # (EB number, name patterns)
        mapping[name.strip()] = date_str

    return mapping


def resolve_survey_date(file_name: str, date_mapping: dict[str, str]) -> str | None:
    """Resolve the publication date for a survey file."""
    # Strip extension from file_name
    base = re.sub(r"\.(xlsx|xls|zip)$", "", file_name, flags=re.I)

    # Direct match
    if base in date_mapping:
        return date_mapping[base]

    # Try matching by EB/FL number
    file_nums = re.findall(r"(?:fl|ebs|eb|sp)_?(\d{3})", file_name, re.I)
    if not file_nums:
        # Try Parlemeter pattern
        parlm = re.search(r"Parlemeter_(\d+)[\._](\d+)_(\d{4})", file_name)
        if parlm:
            for key, date in date_mapping.items():
                if f"Parlemeter_{parlm.group(1)}" in key:
                    return date

        # Try standard EB pattern (eb98, eb_94, etc.)
        std_match = re.search(r"eb_?(\d{2,3})", file_name, re.I)
        if std_match:
            num = std_match.group(1)
            for key, date in date_mapping.items():
                if f"eb_{num}" in key.lower() or f"eb{num}" in key.lower():
                    return date

        # Try extracting year from filename
        year_match = re.search(r"(20\d{2})", file_name)
        if year_match:
            return f"{year_match.group(1)}-06-01"  # approximate mid-year

        return None

    target_num = file_nums[0]
    for key, date in date_mapping.items():
        key_nums = re.findall(r"(?:fl|ebs|eb|sp)_?(\d{3})", key, re.I)
        if target_num in key_nums:
            return date

    return None


def _extract_embeddings(df: pd.DataFrame) -> np.ndarray:
    emb_cols = sorted(
        [c for c in df.columns if c.startswith("emb_")],
        key=lambda c: int(c.split("_")[1]),
    )
    return df[emb_cols].values.astype(np.float32)


def simplify_vote_summary(summary: str) -> str:
    """Use LLM to simplify a vote summary."""
    if not summary or summary == "nan" or len(summary.strip()) < 10:
        return summary

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
        return result
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return summary[:200]


def main():
    parser = argparse.ArgumentParser(
        description="Match survey questions to parliament votes"
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Top-k matches per question"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.50, help="Min cosine similarity"
    )
    args = parser.parse_args()

    # 1. Load survey questions (with clean text)
    logger.info("Loading clean survey questions...")
    surveys = pd.read_csv(SURVEY_CLEAN_CSV)
    logger.info(f"Loaded {len(surveys)} survey questions")

    # 2. Build survey date mapping
    logger.info("Building survey date mapping...")
    date_mapping = build_survey_date_mapping()
    surveys["survey_date"] = surveys["file_name"].apply(
        lambda f: resolve_survey_date(f, date_mapping)
    )
    dated = surveys["survey_date"].notna().sum()
    logger.info(f"Resolved dates for {dated}/{len(surveys)} surveys")
    undated_files = surveys[surveys["survey_date"].isna()]["file_name"].unique()
    if len(undated_files) > 0:
        logger.warning(f"No date for files: {list(undated_files)}")

    # Drop surveys without dates (can't check temporal ordering)
    surveys = surveys[surveys["survey_date"].notna()].copy()
    surveys["survey_date"] = pd.to_datetime(surveys["survey_date"])

    # 3. Load vote data with dates
    logger.info("Loading votes...")
    votes_full = pd.read_csv(VOTES_CSV)
    vote_summaries = pd.read_csv(VOTE_SUMMARIES_CSV)

    # Parse vote dates
    votes_full["vote_date"] = pd.to_datetime(
        votes_full["timestamp"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    vote_date_map = dict(zip(votes_full["id"], votes_full["vote_date"]))

    # 4. Load embeddings and compute matches
    logger.info("Loading embeddings...")
    survey_emb_df = pd.read_parquet(SURVEY_EMB)
    vote_emb_df = pd.read_parquet(VOTE_EMB)

    survey_emb = _extract_embeddings(survey_emb_df)
    vote_emb = _extract_embeddings(vote_emb_df)

    # Normalize
    survey_emb = survey_emb / (np.linalg.norm(survey_emb, axis=1, keepdims=True) + 1e-9)
    vote_emb = vote_emb / (np.linalg.norm(vote_emb, axis=1, keepdims=True) + 1e-9)

    # Build index maps
    survey_id_to_emb_idx = {
        sid: idx for idx, sid in enumerate(survey_emb_df["sheet_id"].values)
    }
    vote_id_to_emb_idx = {
        vid: idx for idx, vid in enumerate(vote_emb_df["vote_id"].values)
    }
    vote_id_to_summary = dict(zip(vote_summaries["vote_id"], vote_summaries["summary"]))

    # 5. Compute matches with temporal filtering
    logger.info(
        f"Computing matches (top_k={args.top_k}, threshold={args.threshold})..."
    )
    rows = []
    batch_size = 256

    # Get the set of survey sheet_ids we're working with
    valid_survey_ids = set(surveys["sheet_id"].values)
    survey_lookup = surveys.set_index("sheet_id")

    for start in tqdm(
        range(0, len(survey_emb), batch_size), desc="Computing similarities"
    ):
        end = min(start + batch_size, len(survey_emb))
        batch = survey_emb[start:end]
        sims = batch @ vote_emb.T

        for i in range(sims.shape[0]):
            emb_idx = start + i
            sheet_id = survey_emb_df.iloc[emb_idx]["sheet_id"]

            if sheet_id not in valid_survey_ids:
                continue

            survey_row = survey_lookup.loc[sheet_id]
            # Handle duplicate sheet_ids (take first)
            if isinstance(survey_row, pd.DataFrame):
                survey_row = survey_row.iloc[0]

            survey_date = survey_row["survey_date"]
            scores = sims[i]

            # Get top-k indices above threshold
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

                # Temporal filter: survey must be published BEFORE the vote
                if vote_date is None or pd.isna(vote_date):
                    continue
                if survey_date >= vote_date:
                    continue

                rows.append(
                    {
                        "question_id": sheet_id,
                        "question_clean": survey_row.get("question_clean", ""),
                        "question_original": survey_row.get("question_en", ""),
                        "survey_file": survey_row.get("file_name", ""),
                        "survey_date": survey_date.strftime("%Y-%m-%d"),
                        "vote_id": vote_id,
                        "vote_summary_original": str(
                            vote_id_to_summary.get(vote_id, "")
                        )[:500],
                        "vote_date": vote_date.strftime("%Y-%m-%d"),
                        "days_between": (vote_date - survey_date).days,
                        "similarity_score": float(scores[vote_idx]),
                    }
                )

    matches_df = pd.DataFrame(rows)
    if matches_df.empty:
        logger.warning("No matches found!")
        return

    matches_df = matches_df.sort_values("similarity_score", ascending=False)
    logger.info(
        f"Found {len(matches_df)} temporal matches "
        f"({matches_df['question_id'].nunique()} questions, "
        f"{matches_df['vote_id'].nunique()} votes)"
    )

    # 6. Simplify vote summaries via LLM (deduplicated)
    unique_vote_ids = matches_df["vote_id"].unique()
    logger.info(f"Simplifying {len(unique_vote_ids)} unique vote summaries...")

    # Check for existing simplified summaries to resume
    vote_clean_cache_path = DATA_DIR / "votes" / "vote_summaries_clean_cache.json"
    vote_clean_cache = {}
    if vote_clean_cache_path.exists():
        with open(vote_clean_cache_path) as f:
            vote_clean_cache = json.load(f)
        logger.info(f"Loaded {len(vote_clean_cache)} cached vote simplifications")

    for vid in tqdm(unique_vote_ids, desc="Simplifying votes"):
        vid_str = str(vid)
        if vid_str in vote_clean_cache:
            continue
        original = str(vote_id_to_summary.get(vid, ""))
        simplified = simplify_vote_summary(original)
        vote_clean_cache[vid_str] = simplified

        # Save cache incrementally
        if len(vote_clean_cache) % 20 == 0:
            vote_clean_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(vote_clean_cache_path, "w") as f:
                json.dump(vote_clean_cache, f, indent=2, ensure_ascii=False)

    # Final cache save
    with open(vote_clean_cache_path, "w") as f:
        json.dump(vote_clean_cache, f, indent=2, ensure_ascii=False)

    # Add clean summaries to matches
    matches_df["vote_summary_clean"] = matches_df["vote_id"].apply(
        lambda vid: vote_clean_cache.get(str(vid), "")
    )

    # 7. Save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(OUTPUT_CSV, index=False)
    logger.success(f"Saved {len(matches_df)} matches → {OUTPUT_CSV}")

    # Show top matches
    logger.info("\n=== TOP MATCHES ===")
    for _, row in matches_df.head(15).iterrows():
        logger.info(f"  Survey ({row.survey_date}): {row.question_clean}")
        logger.info(
            f"  Vote   ({row.vote_date}, +{row.days_between}d): {row.vote_summary_clean}"
        )
        logger.info(f"  Similarity: {row.similarity_score:.3f}")
        logger.info("")


if __name__ == "__main__":
    main()
