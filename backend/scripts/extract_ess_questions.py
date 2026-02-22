"""Extract policy-relevant questions from ESS codebooks and add to the pipeline.

Parses all 11 ESS rounds, filters out demographics/metadata, deduplicates
across rounds, and produces a CSV compatible with the survey pipeline.

Usage:
    python backend/scripts/extract_ess_questions.py
"""

import re
from pathlib import Path

import pandas as pd
from eu_survey_correlation.surveys.ess_scraper import ESSCodebookParser
from loguru import logger

DATA_DIR = Path("data")
ESS_DIR = DATA_DIR / "surveys" / "ess"
OUTPUT_CSV = DATA_DIR / "surveys" / "ess_survey_questions.csv"

# ESS round approximate fieldwork years (for temporal matching later)
ESS_ROUND_YEARS = {
    "ESS1": 2002,
    "ESS2": 2004,
    "ESS3": 2006,
    "ESS4": 2008,
    "ESS5": 2010,
    "ESS6": 2012,
    "ESS7": 2014,
    "ESS8": 2016,
    "ESS9": 2018,
    "ESS10": 2020,
    "ESS11": 2023,
}

# Variable ID patterns to exclude (demographics, metadata, household, country-specific education)
EXCLUDE_ID_PATTERNS = [
    r"^name$",
    r"^essround$",
    r"^edition$",
    r"^proddate$",
    r"^idno$",
    r"^cntry$",
    r"^dweight$",
    r"^pspwght$",
    r"^pweight$",
    r"^anweight$",
    r"^inwyr$",  # interview year
    r"^inwmm$",  # interview month
    r"^inwdd$",  # interview day
    r"^inwyys$",  # interview start
    r"^supqmm$",
    r"^supqyr$",
    r"^yrbrn\d*$",  # year of birth of household members
    r"^gndr\d+$",  # gender of household members (gndr2, gndr3, etc.)
    r"^rshipa?\d+$",  # relationship to respondent
    r"^agea$",  # age
    r"^hhmmb$",  # household members
    r"^dngnapp$",  # not applicable
    r"^edlv[a-z]{2,4}$",  # country-specific education (edlvahu, edlvfdse, etc.)
    r"^edlvf[a-z]{2,3}$",  # father's education country-specific
    r"^edlvm[a-z]{2,3}$",  # mother's education country-specific
    r"^edlvp[a-z]{2,3}$",  # partner's education country-specific
    r"^region$",
    r"^regunit$",
    r"^intewde$",
    r"^litession$",
    r"^lnghom",  # language at home
    r"^ctzcntr$",  # citizen of country
    r"^ctzship",  # citizenship
    r"^brncntr$",  # born in country
    r"^livecnta$",
    r"^blgetmg$",
    r"^facntr$",  # father born in country
    r"^mocntr$",  # mother born in country
    r"^emplno[fp]?$",  # number of employees
    r"^njbspv$",
    r"^wkhtot$",
    r"^esession$",
]

# Description patterns to exclude
EXCLUDE_DESC_PATTERNS = [
    r"person in household",
    r"^CODE SEX",
    r"^ASK IF.*CARD \d+$",
    r"in what year was he/she born",
    r"highest level of education.*\([A-Z][a-z]+\)",  # country-specific education
    r"^Not applicable",
    r"^WRITE IN",
]


def extract_ess_questions() -> pd.DataFrame:
    """Parse all ESS codebooks and return policy-relevant questions."""
    all_questions = []

    for ess_dir in sorted(ESS_DIR.iterdir()):
        if not ess_dir.is_dir():
            continue

        codebooks = list(ess_dir.glob("*.html"))
        if not codebooks:
            continue

        round_name = ess_dir.name
        round_key = re.match(r"(ESS\d+)", round_name).group(1) if re.match(r"(ESS\d+)", round_name) else round_name
        year = ESS_ROUND_YEARS.get(round_key, 0)

        logger.info(f"Parsing {round_name} ({year})...")
        parser = ESSCodebookParser(codebooks[0])
        df = parser.parse()

        # Deduplicate to one row per variable
        uniq = df[["variable_id", "variable_name", "variable_description"]].drop_duplicates(
            "variable_id"
        )

        # Filter out excluded IDs
        mask = pd.Series(True, index=uniq.index)
        for pattern in EXCLUDE_ID_PATTERNS:
            mask &= ~uniq["variable_id"].str.match(pattern, case=False, na=False)

        # Filter out excluded descriptions
        for pattern in EXCLUDE_DESC_PATTERNS:
            mask &= ~uniq["variable_description"].str.contains(
                pattern, case=False, na=False, regex=True
            )

        # Must have a description longer than 20 chars
        mask &= uniq["variable_description"].str.len() > 20

        filtered = uniq[mask].copy()
        filtered["ess_round"] = round_name
        filtered["ess_year"] = year

        logger.info(f"  {len(filtered)}/{len(uniq)} variables kept")
        all_questions.append(filtered)

    combined = pd.concat(all_questions, ignore_index=True)

    # Deduplicate across rounds: same variable_id appearing in multiple rounds
    # Keep the latest round's version
    combined = combined.sort_values("ess_year", ascending=False)
    deduped = combined.drop_duplicates(subset="variable_id", keep="first")
    deduped = deduped.sort_values("variable_id")

    logger.info(
        f"Total: {len(deduped)} unique questions "
        f"(from {len(combined)} across {combined.ess_round.nunique()} rounds)"
    )

    # Build question text: combine variable_name + description
    deduped = deduped.copy()
    deduped["question_en"] = deduped.apply(
        lambda r: f"{r['variable_name']}: {r['variable_description']}", axis=1
    )

    # Format for the pipeline (compatible with all_survey_questions.csv)
    output = pd.DataFrame(
        {
            "sheet_id": deduped["variable_id"],
            "question_en": deduped["question_en"],
            "question_fr": "",
            "file_name": deduped["ess_round"],
            "survey_date": deduped["ess_year"].apply(lambda y: f"{y}-06-01"),
        }
    )

    return output


def main():
    questions = extract_ess_questions()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    questions.to_csv(OUTPUT_CSV, index=False)
    logger.success(f"Saved {len(questions)} ESS questions → {OUTPUT_CSV}")

    # Show samples
    logger.info("Sample questions:")
    for _, row in questions.sample(min(15, len(questions)), random_state=42).iterrows():
        logger.info(f"  [{row.sheet_id}] {str(row.question_en)[:120]}")


if __name__ == "__main__":
    main()
