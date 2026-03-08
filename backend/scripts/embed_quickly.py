from pathlib import Path

import pandas as pd
from eu_survey_correlation.simplifier import Simplifier

# ── Config ────────────────────────────────────────────────────────────
DATA_DIR = Path("data")

SURVEY_CSV = DATA_DIR / "surveys" / "michlou_survey_tri.csv"
VOTE_SUMMARIES_CSV = DATA_DIR / "votes" / "vote_summaries.csv"
VOTES_CSV = DATA_DIR / "votes" / "votes.csv"

EMBEDDING_CACHE = DATA_DIR / "cache" / "embeddings.parquet"

SIMPLIFIER_CACHE = DATA_DIR / "cache" / "simplified_text.json"
OUTPUT_CSV = DATA_DIR / "matches" / "simplified_michlou_survey_vote_matches_clean.csv"

TOP_K = 3
THRESHOLD = 0.0
## 1. Load data
surveys = (
    pd.read_csv(SURVEY_CSV)
    .dropna(subset=["question_en"])
    .drop_duplicates(subset=["question_en"])
)
vote_summaries = (
    pd.read_csv(VOTE_SUMMARIES_CSV)
    .dropna(subset=["summary"])
    .drop_duplicates(subset=["vote_id"])
)
print(f"Surveys: {len(surveys)} | Vote summaries: {len(vote_summaries)}")
## 2. Simplify questions and votes

simplifier = Simplifier(cache_path=SIMPLIFIER_CACHE)

# Skip if question_clean already present in input CSV
surveys = simplifier.simplify_dataframe(
    surveys,
    text_column="question_en",
    output_column="question_clean",
    prompt_template=Simplifier.SURVEY_QUESTION_PROMPT,
    save_path=SURVEY_CSV,
)
print(len(surveys))
surveys = surveys.drop_duplicates(subset=["question_clean"])
print(len(surveys))
# Skip if question_clean already present in input CSV
vote_summaries = simplifier.simplify_dataframe(
    vote_summaries,
    text_column="summary",
    output_column="summary_clean",
    prompt_template=Simplifier.VOTE_SUMMARY_PROMPT,
    save_path=VOTE_SUMMARIES_CSV,
)
print(len(vote_summaries))
vote_summaries = vote_summaries.drop_duplicates(subset=["summary_clean"])
print(len(vote_summaries))
