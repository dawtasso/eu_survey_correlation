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

SCRAPED_FILE = "/Users/ugo/Documents/MH2D_projets/dawta/eu_survey_correlation/data/votes/vote_procedure_summaries.json"
FINAL_OUTPUT_FILE = "/Users/ugo/Documents/MH2D_projets/dawta/eu_survey_correlation/data/votes/procedure_summaries_simplified.csv"
MAX_WORKERS = 15  # Increased slightly for M1 Pro performance
BASE_URL = "https://oeil.europarl.europa.eu"
SIMPLIFIER_CACHE = DATA_DIR / "cache" / "simplified_text.json"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def extract_clean_summary(html_content):
    """Parses the summary page and returns formatted text."""
    soup = BeautifulSoup(html_content, "html.parser")
    content_div = soup.find("div", class_="es_product-content")

    if not content_div:
        return ""

    paragraphs = content_div.find_all("p")
    clean_paragraphs = [
        p.get_text(" ", strip=True) for p in paragraphs if p.get_text(strip=True)
    ]

    return "\n\n".join(clean_paragraphs)


def scrape_procedure(procedure_ref):
    """Finds the summary URL and extracts the text content."""
    try:
        # 1. Get the main procedure page
        main_url = f"{BASE_URL}/oeil/en/procedure-file?reference={procedure_ref}"
        response = requests.get(main_url, headers=headers, timeout=15)

        if response.status_code != 200:
            return procedure_ref, {"url": None, "summary_text": None}

        soup = BeautifulSoup(response.content, "html.parser")
        summary_url = None
        summary_text = None

        # 2. Find the Summary URL in Section 3
        section = soup.find(id="section3")
        if section:
            for row in section.find_all("tr"):
                cols = row.find_all("td")
                if len(cols) >= 4 and "Decision by Parliament" in cols[1].get_text():
                    link = cols[3].find("a", href=True)
                    if link:
                        href = link["href"]
                        summary_url = (
                            f"{BASE_URL}{href}" if href.startswith("/") else href
                        )
                        break

        # 3. If a summary URL was found, fetch and clean the text
        if summary_url:
            sum_resp = requests.get(summary_url, headers=headers, timeout=15)
            if sum_resp.status_code == 200:
                summary_text = extract_clean_summary(sum_resp.content)

        return procedure_ref, {"url": summary_url, "summary_text": summary_text}

    except Exception as e:
        return procedure_ref, {"url": None, "summary_text": f"Error: {str(e)}"}


def process_and_simplify():
    # 1. Load the JSONL into a list of dicts
    data = []
    with open(SCRAPED_FILE, "r") as f:
        for line in f:
            item = json.loads(line)
            # Flatten the structure for the DataFrame
            data.append(
                {
                    "reference": item["reference"],
                    "url": item["data"]["url"],
                    "summary_text": item["data"]["summary_text"],
                }
            )

    df = pd.DataFrame(data)
    # 2. Initialize your Simplifier
    # M1 Pro hint: Mistral is great, but 'phi3' or 'llama3' are also very fast on Apple Silicon
    simplifier = Simplifier(model="mistral", cache_path=SIMPLIFIER_CACHE)

    # 3. Run the simplification
    # We use your existing 'VOTE_SUMMARY_PROMPT'
    df_simplified = simplifier.simplify_dataframe(
        df=df,
        text_column="summary_text",
        output_column="simplified_summary",
        prompt_template=Simplifier.VOTE_SUMMARY_PROMPT,
        concurrency=2,  # Keep it low for local LLMs so your Mac doesn't lag
    )

    # 4. Save the final consolidated result
    # This will have: reference, url, summary_text, simplified_summary
    df_simplified.to_csv(FINAL_OUTPUT_FILE, index=False)
    logger.success(f"Success! Processed {len(df_simplified)} summaries.")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    votes_df = pd.read_csv(
        "/Users/ugo/Documents/MH2D_projets/dawta/eu_survey_correlation/data/votes/votes.csv"
    )
    procedure_references = votes_df["procedure_reference"].dropna().unique().tolist()

    # Load existing progress (using .jsonl to allow easy appending)
    finished_refs = set()
    try:
        with open(SCRAPED_FILE, "r") as f:
            for line in f:
                finished_refs.add(json.loads(line)["reference"])
    except FileNotFoundError:
        pass

    to_process = [r for r in procedure_references if r not in finished_refs]

    # Open in 'a' (append) mode
    with open(SCRAPED_FILE, "a") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_ref = {
                executor.submit(scrape_procedure, ref): ref for ref in to_process
            }

            for future in tqdm(
                as_completed(future_to_ref), total=len(to_process), desc="Scraping"
            ):
                ref, data = future.result()

                # Each line is a standalone JSON object
                entry = {"reference": ref, "data": data}

                f_out.write(json.dumps(entry) + "\n")
                f_out.flush()

    ## Simplify
    process_and_simplify()
