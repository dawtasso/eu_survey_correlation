"""Migrate match_ids in Supabase from old format to deterministic hashes.

Old format:  {question_id}_{vote_id}_{csv_row_index}
New format:  sha256("{question_id}::{survey_file}::{procedure_reference}")[:16]

Usage:
    uv run python backend/scripts/migrate_match_ids.py           # dry-run (default)
    uv run python backend/scripts/migrate_match_ids.py --apply   # actually update Supabase
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from match_id_utils import (
    build_migration_mapping,
    build_vote_to_procedure_map,
)

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
VOTES_CSV = ROOT / "data" / "votes" / "votes.csv"
MIGRATION_DIR = ROOT / "data" / "migration"


def get_supabase():
    import os
    from supabase import create_client

    url = os.environ["SUPABASE_URL"]
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def fetch_all_rows(supabase, table: str) -> list[dict]:
    """Fetch all rows from a Supabase table (handles pagination)."""
    rows = []
    page_size = 1000
    offset = 0
    while True:
        resp = (
            supabase.table(table)
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows.extend(resp.data)
        if len(resp.data) < page_size:
            break
        offset += page_size
    return rows


def save_backup(data: list[dict], name: str, migration_dir: Path) -> Path:
    migration_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = migration_dir / f"{name}_backup_{ts}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Backup saved: {path} ({len(data)} rows)")
    return path


def dry_run(supabase) -> dict:
    """Run all checks without writing to Supabase. Returns summary dict."""
    logger.info("=== DRY RUN ===")

    # 1. Download tables
    matches = fetch_all_rows(supabase, "survey_vote_matches")
    alignments = fetch_all_rows(supabase, "survey_answer_alignments")
    logger.info(f"Downloaded {len(matches)} matches, {len(alignments)} alignments")

    # 2. Save backups
    save_backup(matches, "survey_vote_matches", MIGRATION_DIR)
    save_backup(alignments, "survey_answer_alignments", MIGRATION_DIR)

    # 3. Build mapping (with collision resolution)
    vote_to_proc = build_vote_to_procedure_map(VOTES_CSV)
    id_map, key_map, missing, duplicates = build_migration_mapping(matches, vote_to_proc)

    # 4. Count admin_validated
    validated_true = sum(1 for r in matches if r.get("admin_validated") is True)
    validated_false = sum(1 for r in matches if r.get("admin_validated") is False)
    validated_none = sum(1 for r in matches if r.get("admin_validated") is None)

    # 5. Check what we lose from duplicates
    dup_validated_true = sum(1 for d in duplicates if d.get("admin_validated") is True)

    # 6. Count alignment rows affected by duplicates
    dup_match_ids = {d["match_id"] for d in duplicates}
    alignment_match_ids = {a["match_id"] for a in alignments}
    alignments_on_duplicates = alignment_match_ids & dup_match_ids
    alignments_with_mapping = alignment_match_ids & set(id_map.keys())
    alignments_without_mapping = alignment_match_ids - set(id_map.keys()) - dup_match_ids

    summary = {
        "total_matches": len(matches),
        "to_migrate": len(id_map),
        "duplicates_to_delete": len(duplicates),
        "missing_procedure": len(missing),
        "admin_validated_true": validated_true,
        "admin_validated_false": validated_false,
        "admin_validated_none": validated_none,
        "dup_validated_true_lost": dup_validated_true,
        "total_alignments": len(alignments),
        "alignments_with_mapping": len(alignments_with_mapping),
        "alignments_on_duplicates": len(alignments_on_duplicates),
        "alignments_without_mapping": len(alignments_without_mapping),
        "final_match_count": len(id_map) + len(missing),
    }

    logger.info("--- Summary ---")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")

    if duplicates:
        logger.warning(f"{len(duplicates)} duplicate rows will be deleted (collision resolution):")
        for d in duplicates:
            logger.warning(
                f"  DELETE {d['match_id']}  vote_id={d.get('vote_id')}  "
                f"admin_validated={d.get('admin_validated')}  "
                f"score={d.get('similarity_score', '?')}"
            )

    if dup_validated_true > 0:
        logger.warning(
            f"  ⚠ {dup_validated_true} of the deleted duplicates have admin_validated=True!"
        )

    if missing:
        logger.warning(f"Missing procedure_reference for {len(missing)} rows:")
        for row in missing[:5]:
            logger.warning(f"  match_id={row['match_id']}  vote_id={row.get('vote_id')}")

    if alignments_on_duplicates:
        logger.warning(
            f"{len(alignments_on_duplicates)} alignment match_ids belong to duplicate rows "
            "(their alignments will be reassigned to the winner row)"
        )

    if alignments_without_mapping:
        logger.warning(
            f"{len(alignments_without_mapping)} alignment match_ids have no mapping "
            "(their match rows may be in the 'missing' set)"
        )

    # Save mapping for review
    mapping_path = MIGRATION_DIR / "id_mapping.json"
    MIGRATION_DIR.mkdir(parents=True, exist_ok=True)
    mapping_data = {
        old: {"new_id": new, "match_key": key_map[old]}
        for old, new in id_map.items()
    }
    if duplicates:
        mapping_data["_duplicates_to_delete"] = [
            {"match_id": d["match_id"], "vote_id": d.get("vote_id"),
             "admin_validated": d.get("admin_validated")}
            for d in duplicates
        ]
    with open(mapping_path, "w") as f:
        json.dump(mapping_data, f, indent=2)
    logger.info(f"Mapping saved: {mapping_path}")

    return summary


def apply(supabase) -> None:
    """Run migration: update match_ids in both tables."""
    summary = dry_run(supabase)

    if summary["missing_procedure"] > 0:
        logger.warning(
            f"{summary['missing_procedure']} rows cannot be migrated (no procedure_reference). "
            "They will be left unchanged."
        )

    logger.info("=== APPLYING MIGRATION ===")

    # Re-build mapping
    matches = fetch_all_rows(supabase, "survey_vote_matches")
    alignments = fetch_all_rows(supabase, "survey_answer_alignments")
    vote_to_proc = build_vote_to_procedure_map(VOTES_CSV)
    id_map, key_map, _, duplicates = build_migration_mapping(matches, vote_to_proc)

    # Build a lookup: for each duplicate, find the winner's new_id
    # (the winner is the row whose old_id is in id_map)
    dup_to_winner_new_id: dict[str, str] = {}
    for d in duplicates:
        # The duplicate's new_id would be the same as the winner's
        # Find it via vote_to_proc
        vote_id = d.get("vote_id")
        if vote_id and int(vote_id) in vote_to_proc:
            from match_id_utils import make_match_id
            proc_ref = vote_to_proc[int(vote_id)]
            q = d.get("question_id", "")
            sf = d.get("survey_file", "")
            dup_to_winner_new_id[d["match_id"]] = make_match_id(q, sf, proc_ref)

    # Step 1: Reassign alignments from duplicate rows to winner rows
    reassigned = 0
    for a in alignments:
        old_id = a["match_id"]
        if old_id in dup_to_winner_new_id:
            new_id = dup_to_winner_new_id[old_id]
            try:
                supabase.table("survey_answer_alignments").update(
                    {"match_id": new_id}
                ).eq("match_id", old_id).eq(
                    "answer_label", a["answer_label"]
                ).execute()
                reassigned += 1
            except Exception as e:
                logger.error(f"Failed to reassign alignment {old_id}: {e}")
    if reassigned:
        logger.info(f"Reassigned {reassigned} alignments from duplicate rows")

    # Step 2: Delete duplicate match rows
    deleted = 0
    for d in duplicates:
        try:
            supabase.table("survey_vote_matches").delete().eq(
                "match_id", d["match_id"]
            ).execute()
            deleted += 1
        except Exception as e:
            logger.error(f"Failed to delete duplicate {d['match_id']}: {e}")
    if deleted:
        logger.info(f"Deleted {deleted}/{len(duplicates)} duplicate match rows")

    # Step 3: Update remaining match rows with new IDs
    updated_matches = 0
    for old_id, new_id in id_map.items():
        row = next(r for r in matches if r["match_id"] == old_id)
        proc_ref = vote_to_proc.get(int(row["vote_id"])) if row.get("vote_id") else None
        match_key = key_map[old_id]
        try:
            supabase.table("survey_vote_matches").update(
                {
                    "match_id": new_id,
                    "match_key": match_key,
                    "procedure_reference": proc_ref,
                }
            ).eq("match_id", old_id).execute()
            updated_matches += 1
        except Exception as e:
            logger.error(f"Failed to update match {old_id} → {new_id}: {e}")

    logger.info(f"Updated {updated_matches}/{len(id_map)} match rows")

    # Step 4: Update alignment rows with new match IDs
    updated_alignments = 0
    for a in alignments:
        old_id = a["match_id"]
        if old_id in id_map:
            new_id = id_map[old_id]
            try:
                supabase.table("survey_answer_alignments").update(
                    {"match_id": new_id}
                ).eq("match_id", old_id).eq(
                    "answer_label", a["answer_label"]
                ).execute()
                updated_alignments += 1
            except Exception as e:
                logger.error(f"Failed to update alignment {old_id}: {e}")

    logger.info(f"Updated {updated_alignments} alignment rows")

    # Verify counts
    final_matches = fetch_all_rows(supabase, "survey_vote_matches")
    final_alignments = fetch_all_rows(supabase, "survey_answer_alignments")
    final_validated = sum(1 for r in final_matches if r.get("admin_validated") is True)

    expected_matches = len(matches) - len(duplicates)
    expected_validated = summary["admin_validated_true"] - summary["dup_validated_true_lost"]

    assert len(final_matches) == expected_matches, (
        f"Match count: expected {expected_matches}, got {len(final_matches)}"
    )
    assert len(final_alignments) == len(alignments), (
        f"Alignment count changed: {len(alignments)} → {len(final_alignments)}"
    )
    assert final_validated >= expected_validated, (
        f"Validated count: expected >={expected_validated}, got {final_validated}"
    )

    logger.success(
        f"Migration complete. {len(final_matches)} matches, "
        f"{len(final_alignments)} alignments, {final_validated} validated."
    )


def main():
    parser = argparse.ArgumentParser(description="Migrate match_ids to deterministic hashes")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply the migration (default is dry-run)",
    )
    args = parser.parse_args()

    supabase = get_supabase()

    if args.apply:
        apply(supabase)
    else:
        dry_run(supabase)


if __name__ == "__main__":
    main()
