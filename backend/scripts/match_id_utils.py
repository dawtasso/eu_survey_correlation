"""Pure helper functions for deterministic match_id generation and migration.

match_id = sha256("{question_id}::{survey_file}::{procedure_reference}")[:16]
match_key = "{question_id}::{survey_file}::{procedure_reference}"
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


SEPARATOR = "::"


def make_match_key(
    question_id: str, survey_file: str, procedure_reference: str
) -> str:
    """Build the human-readable match key."""
    return f"{question_id}{SEPARATOR}{survey_file}{SEPARATOR}{procedure_reference}"


def make_match_id(
    question_id: str, survey_file: str, procedure_reference: str
) -> str:
    """Build a deterministic match_id (first 16 chars of sha256 hex digest)."""
    key = make_match_key(question_id, survey_file, procedure_reference)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def decode_match_key(match_key: str) -> tuple[str, str, str]:
    """Split a match_key back into (question_id, survey_file, procedure_reference)."""
    parts = match_key.split(SEPARATOR)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid match_key (expected 3 parts separated by '{SEPARATOR}'): {match_key!r}"
        )
    return parts[0], parts[1], parts[2]


# ── Vote-mapping helpers ──────────────────────────────────────────────


def build_vote_to_procedure_map(votes_csv_path: str | Path) -> dict[int, str]:
    """Map vote_id → procedure_reference from the full votes.csv.

    Every row in votes.csv that has a non-empty procedure_reference is included.
    """
    df = pd.read_csv(votes_csv_path)
    df = df.dropna(subset=["procedure_reference"])
    return dict(zip(df["id"].astype(int), df["procedure_reference"].astype(str)))


def build_main_vote_map(votes_csv_path: str | Path) -> dict[str, int]:
    """Map procedure_reference → vote_id, keeping only is_main=True rows.

    When a procedure has multiple is_main=True votes (common in real data),
    the one with the highest vote id is kept (typically the final resolution vote).
    """
    df = pd.read_csv(votes_csv_path)
    main = df[df["is_main"] == True].dropna(subset=["procedure_reference"])  # noqa: E712

    # Keep highest id per procedure (last/final vote)
    main = main.sort_values("id").drop_duplicates(
        subset="procedure_reference", keep="last"
    )

    return dict(
        zip(
            main["procedure_reference"].astype(str),
            main["id"].astype(int),
        )
    )


def _validation_priority(row: dict) -> tuple[int, float]:
    """Score a row for collision resolution.

    Priority: admin_validated=True > False > None, then highest similarity_score.
    Returns a tuple suitable for max().
    """
    av = row.get("admin_validated")
    if av is True:
        rank = 2
    elif av is False:
        rank = 1
    else:
        rank = 0
    score = float(row.get("similarity_score") or 0)
    return (rank, score)


def build_migration_mapping(
    old_rows: list[dict],
    vote_to_proc: dict[int, str],
) -> tuple[dict[str, str], dict[str, str], list[dict], list[dict]]:
    """Compute old_match_id → new_match_id mapping for every existing row.

    When multiple old rows map to the same new_match_id (duplicates from
    different vote_ids on the same procedure), the best row is kept
    (admin_validated=True > False > None, then highest similarity_score).

    Parameters
    ----------
    old_rows : list[dict]
        Rows from the survey_vote_matches table (must have match_id, question_id,
        survey_file, vote_id).
    vote_to_proc : dict[int, str]
        Mapping of vote_id → procedure_reference.

    Returns
    -------
    id_mapping : dict[str, str]
        old_match_id → new_match_id (winner rows only)
    key_mapping : dict[str, str]
        old_match_id → new_match_key (winner rows only)
    missing : list[dict]
        Rows where vote_id had no procedure_reference (cannot migrate).
    duplicates : list[dict]
        Rows that lost collision resolution (should be deleted).
    """
    # First pass: compute new_id for every mappable row
    candidates: dict[str, list[dict]] = {}  # new_id → list of (old_row + new_id info)
    missing: list[dict] = []

    for row in old_rows:
        old_id = row["match_id"]
        question_id = row.get("question_id", "")
        survey_file = row.get("survey_file", "")
        vote_id = row.get("vote_id")

        if vote_id is None or int(vote_id) not in vote_to_proc:
            missing.append(row)
            continue

        proc_ref = vote_to_proc[int(vote_id)]
        new_id = make_match_id(question_id, survey_file, proc_ref)
        new_key = make_match_key(question_id, survey_file, proc_ref)

        entry = {**row, "_new_id": new_id, "_new_key": new_key}
        candidates.setdefault(new_id, []).append(entry)

    # Second pass: resolve collisions
    id_mapping: dict[str, str] = {}
    key_mapping: dict[str, str] = {}
    duplicates: list[dict] = []

    for new_id, entries in candidates.items():
        if len(entries) == 1:
            winner = entries[0]
        else:
            # Pick the best row
            winner = max(entries, key=_validation_priority)
            for e in entries:
                if e["match_id"] != winner["match_id"]:
                    duplicates.append(e)

        id_mapping[winner["match_id"]] = winner["_new_id"]
        key_mapping[winner["match_id"]] = winner["_new_key"]

    return id_mapping, key_mapping, missing, duplicates
