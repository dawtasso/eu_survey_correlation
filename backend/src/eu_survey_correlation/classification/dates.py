"""Date enrichment for survey-vote pairs.

Extracts Eurobarometer edition numbers from filenames, maps them to publication
dates via metadata, and joins vote dates from the votes CSV.
"""

import json
import re
from pathlib import Path

import pandas as pd
from loguru import logger


def build_survey_date_map(
    metadata_path: Path = Path("data/surveys/distributions_metadata.json"),
) -> dict[str, pd.Timestamp]:
    """Build a mapping from Eurobarometer edition number to publication date.

    Parses edition numbers (FL/EB/EBS/SP + digits) from the metadata JSON
    descriptions and maps each to its latest ``issued`` date.
    """
    with open(metadata_path) as f:
        meta = json.load(f)

    edition_dates: dict[str, str] = {}
    for m in meta:
        if "issued" not in m:
            continue
        desc = m.get("description", {}).get("en", "")
        issued = m["issued"][:10]
        for match in re.finditer(
            r"(?:FLASH|FL|EB|EBS|SP)\D?(\d{2,4})", desc, re.IGNORECASE
        ):
            num = match.group(1)
            if num not in edition_dates or issued > edition_dates[num]:
                edition_dates[num] = issued

    logger.debug(f"Survey date map: {len(edition_dates)} editions")
    return {k: pd.Timestamp(v) for k, v in edition_dates.items()}


def get_survey_date(
    file_name: str, edition_dates: dict[str, pd.Timestamp]
) -> pd.Timestamp:
    """Extract survey date from a filename by matching its edition number."""
    if not isinstance(file_name, str):
        return pd.NaT
    for match in re.finditer(
        r"(?:fl|eb|ebs|SP)\D?(\d{2,4})", file_name, re.IGNORECASE
    ):
        num = match.group(1)
        if num in edition_dates:
            return edition_dates[num]
    return pd.NaT


def build_vote_date_map(
    votes_csv_path: Path = Path("data/votes/votes.csv"),
) -> dict[str, pd.Timestamp]:
    """Build a mapping from vote_id (str) to vote datetime."""
    votes = pd.read_csv(votes_csv_path)
    votes["date"] = pd.to_datetime(
        votes["timestamp"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    logger.debug(f"Vote date map: {len(votes)} votes")
    return dict(zip(votes["id"].astype(str), votes["date"]))


def enrich_with_dates(
    df: pd.DataFrame,
    metadata_path: Path = Path("data/surveys/distributions_metadata.json"),
    votes_csv_path: Path = Path("data/votes/votes.csv"),
) -> pd.DataFrame:
    """Add ``survey_date``, ``vote_date``, and ``survey_before_vote`` columns.

    Parameters
    ----------
    df : DataFrame
        Must contain ``file_name`` and ``vote_id`` columns.

    Returns
    -------
    DataFrame with three new columns added.
    """
    edition_dates = build_survey_date_map(metadata_path)
    vote_dates = build_vote_date_map(votes_csv_path)

    df = df.copy()
    df["survey_date"] = df["file_name"].apply(
        lambda fn: get_survey_date(fn, edition_dates)
    )
    df["vote_date"] = df["vote_id"].astype(str).map(vote_dates)
    df["survey_before_vote"] = (
        df["survey_date"].notna()
        & df["vote_date"].notna()
        & (df["survey_date"] < df["vote_date"])
    )

    n_survey = df["survey_date"].notna().sum()
    n_vote = df["vote_date"].notna().sum()
    logger.info(
        f"Date enrichment: {n_survey}/{len(df)} surveys dated, "
        f"{n_vote}/{len(df)} votes dated"
    )
    return df
