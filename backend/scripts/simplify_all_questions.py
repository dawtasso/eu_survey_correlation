import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from eu_survey_correlation.simplifier import Simplifier
from loguru import logger
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = Path("data")

SURVEY_QUESTIONS_FILE = "/Users/ugo/Documents/MH2D_projets/dawta/eu_survey_correlation/data/surveys/all_survey_questions.csv"
FINAL_OUTPUT_FILE = "/Users/ugo/Documents/MH2D_projets/dawta/eu_survey_correlation/data/surveys/all_survey_questions_simplified.csv"
MAX_WORKERS = 2  # Increased slightly for M1 Pro performance
SIMPLIFIER_CACHE = DATA_DIR / "cache" / "simplified_text.json"


def process_and_simplify():
    # 1. Load the JSONL into a list of dicts
    df = pd.read_csv(SURVEY_QUESTIONS_FILE)[:5]
    # 2. Initialize your Simplifier
    # M1 Pro hint: Mistral is great, but 'phi3' or 'llama3' are also very fast on Apple Silicon
    simplifier = Simplifier(model="mistral", cache_path=SIMPLIFIER_CACHE)

    # 3. Run the simplification
    # We use your existing 'VOTE_SUMMARY_PROMPT'
    df_simplified = simplifier.simplify_dataframe(
        df=df,
        text_column="question_en",
        output_column="question_clean",
        prompt_template=Simplifier.SURVEY_QUESTION_PROMPT,
        concurrency=2,  # Keep it low for local LLMs so your Mac doesn't lag
    )

    # 4. Save the final consolidated result
    # This will have: reference, url, summary_text, simplified_summary
    df_simplified.to_csv(FINAL_OUTPUT_FILE, index=False)
    logger.success(f"Success! Processed {len(df_simplified)} summaries.")


if __name__ == "__main__":
    process_and_simplify()
