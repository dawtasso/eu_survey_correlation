"""
Push classifier predictions to Supabase for frontend display.

1. Scores ALL existing matches in Supabase with the trained model
2. Updates predicted_probability column (NEVER touches admin_validated)
3. Finds new candidate pairs from CSV and inserts them

Usage:
    uv run python backend/scripts/push_predictions_to_supabase.py                 # update existing only
    uv run python backend/scripts/push_predictions_to_supabase.py --insert-new    # also insert new candidates
    uv run python backend/scripts/push_predictions_to_supabase.py --insert-new --csv data/matches/all_candidates.csv
    uv run python backend/scripts/push_predictions_to_supabase.py --insert-new --active-learning --limit 50
    uv run python backend/scripts/push_predictions_to_supabase.py --dry-run       # preview without writing
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from eu_survey_correlation.classifier import (
    DATA,
    OUTPUT_DIR,
    build_feature_matrix,
    compute_cross_encoder_scores,
    load_embedding_lookup,
)

import sys

sys.path.insert(0, str(Path(__file__).parent))
from match_id_utils import build_main_vote_map, build_vote_to_procedure_map, make_match_id, make_match_key

ROOT = Path(__file__).resolve().parents[2]
VOTES_CSV = ROOT / "data" / "votes" / "votes.csv"
DEFAULT_CANDIDATES_CSV = ROOT / "data" / "matches" / "all_candidates.csv"
LEGACY_MATCHES_CSV = ROOT / "data" / "matches" / "simplified_michlou_survey_vote_matches_clean.csv"
BATCH_SIZE = 100

# Minimum predicted probability to insert a new candidate pair
NEW_CANDIDATE_THRESHOLD = 0.4


def get_supabase():
    from supabase import create_client

    url = os.environ["SUPABASE_URL"]
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def fetch_all_supabase_matches(supabase) -> list[dict]:
    """Fetch all rows from survey_vote_matches (handles pagination)."""
    rows = []
    page_size = 1000
    offset = 0
    while True:
        resp = (
            supabase.table("survey_vote_matches")
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows.extend(resp.data)
        if len(resp.data) < page_size:
            break
        offset += page_size
    return rows


def load_model_and_config() -> tuple:
    """Load trained model, threshold, model type, and selected features."""
    import joblib

    model_path = OUTPUT_DIR / "model.joblib"
    threshold_path = OUTPUT_DIR / "threshold.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run train_classifier.py first."
        )

    model = joblib.load(model_path)

    model_type = "lr"
    threshold = 0.5
    selected_features = None
    if threshold_path.exists():
        with open(threshold_path) as f:
            meta = json.load(f)
            threshold = meta["threshold"]
            model_type = meta.get("model_type", "lr")
            selected_features = meta.get("selected_features")

    return model, threshold, model_type, selected_features


def _select_features(X: np.ndarray, feature_names: list[str], selected: list[str] | None) -> np.ndarray:
    """Slice feature matrix to selected features if specified."""
    if selected is None:
        return X
    indices = [feature_names.index(f) for f in selected if f in feature_names]
    return X[:, indices]


def score_records(
    records: list[dict],
    model,
    model_type: str,
    emb_lookup: dict,
    selected_features: list[str] | None = None,
    cache_name: str = "cross_encoder_scores_supabase.npy",
) -> np.ndarray:
    """Score a list of records and return predicted probabilities."""
    feature_df = build_feature_matrix(records, emb_lookup)
    feature_names = list(feature_df.columns)
    X = feature_df.values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)

    if model_type == "hybrid":
        cache_path = DATA / "cache" / cache_name
        ce_scores = compute_cross_encoder_scores(records, cache_path)
        X = np.column_stack([X, ce_scores])

    X = _select_features(X, feature_names, selected_features)

    return model.predict_proba(X)[:, 1]


# ── Step 1: Update existing Supabase matches ─────────────────────────
def update_existing_matches(supabase, dry_run: bool = False) -> int:
    """Score all existing matches and update predicted_probability."""
    logger.info("Fetching existing matches from Supabase...")
    rows = fetch_all_supabase_matches(supabase)
    if not rows:
        logger.warning("No matches found in Supabase")
        return 0

    logger.info(f"Found {len(rows)} existing matches")

    model, threshold, model_type, selected_features = load_model_and_config()
    emb_lookup = load_embedding_lookup()
    logger.info(f"Model type: {model_type}, threshold: {threshold:.2f}")

    # Prepare records for scoring (map Supabase column names)
    records = []
    for r in rows:
        records.append({
            "question_clean": r.get("question_clean") or "",
            "vote_summary_clean": r.get("vote_summary_clean") or "",
            "similarity_score": r.get("similarity_score") or 0,
            "days_between": r.get("days_between") or 0,
        })

    probs = score_records(records, model, model_type, emb_lookup, selected_features, "cross_encoder_scores_supabase.npy")

    logger.info(f"Score distribution: mean={probs.mean():.3f}, std={probs.std():.3f}")
    logger.info(f"Predicted accepted (>={threshold:.2f}): {(probs >= threshold).sum()}/{len(probs)}")

    if dry_run:
        logger.info("[DRY RUN] Would update predicted_probability for all existing matches")
        for i, (r, p) in enumerate(zip(rows, probs)):
            if i < 5:
                logger.info(f"  {r['match_id']}: {p:.4f} {'[ACCEPT]' if p >= threshold else '[REFUSE]'}")
        return len(rows)

    # Batch update — only update predicted_probability, never touch admin_validated
    updated = 0
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        batch_probs = probs[start : start + BATCH_SIZE]

        for row, prob in zip(batch, batch_probs):
            supabase.table("survey_vote_matches").update(
                {"predicted_probability": round(float(prob), 4)}
            ).eq("match_id", row["match_id"]).execute()
            updated += 1

        logger.info(f"  Updated {updated}/{len(rows)} matches")

    logger.info(f"Updated predicted_probability for {updated} existing matches")
    return updated


# ── Step 2: Insert new candidate pairs ────────────────────────────────
def insert_new_candidates(
    supabase,
    csv_path: Path,
    dry_run: bool = False,
    min_threshold: float = NEW_CANDIDATE_THRESHOLD,
    active_learning: bool = False,
    limit: int | None = None,
) -> int:
    """Find CSV pairs not in Supabase, score them, insert high-probability ones."""
    if not csv_path.exists():
        logger.warning(f"Candidates CSV not found: {csv_path}")
        return 0

    if not VOTES_CSV.exists():
        logger.warning(f"Votes CSV not found: {VOTES_CSV}")
        return 0

    # Load vote_id → procedure_reference mapping (all votes, for resolution)
    vote_to_proc = build_vote_to_procedure_map(VOTES_CSV)
    # Also load main vote_ids to filter: only main votes have the correct summary
    main_vote_ids = set(build_main_vote_map(VOTES_CSV).values())
    logger.info(f"Loaded {len(vote_to_proc)} vote->procedure mappings ({len(main_vote_ids)} main votes)")

    # Load existing match_ids from Supabase
    existing_rows = fetch_all_supabase_matches(supabase)
    existing_ids = {r["match_id"] for r in existing_rows}
    logger.info(f"Existing Supabase match_ids: {len(existing_ids)}")

    # Parse CSV and find new candidates
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path.name}")

    new_candidates = []
    skipped_no_proc = 0

    for _, row in df.iterrows():
        question_id = str(row.get("sheet_id", row.get("question_id", ""))).strip()
        survey_file = str(row.get("file_name", row.get("survey_file", ""))).strip()

        # Handle vote_id → procedure_reference resolution
        vote_id = None
        procedure_reference = None

        if pd.notna(row.get("vote_id")):
            vote_id = int(float(row["vote_id"]))
            # Only accept main votes (non-main votes don't have the right summary)
            if vote_id in vote_to_proc and vote_id in main_vote_ids:
                procedure_reference = vote_to_proc[vote_id]

        # If CSV already has procedure_reference (e.g. from generate_candidates.py)
        if procedure_reference is None and pd.notna(row.get("procedure_reference")):
            procedure_reference = str(row["procedure_reference"]).strip()

        if procedure_reference is None:
            skipped_no_proc += 1
            continue

        match_id = make_match_id(question_id, survey_file, procedure_reference)

        if match_id in existing_ids:
            continue  # Already in Supabase

        # Handle column name variations between CSV sources
        vote_summary_clean = str(
            row.get("summary_clean", row.get("vote_summary_clean", row.get("simplified_summary", "")))
        ).strip()

        new_candidates.append({
            "match_id": match_id,
            "match_key": make_match_key(question_id, survey_file, procedure_reference),
            "question_id": question_id,
            "question_clean": str(row.get("question_clean", "")).strip(),
            "question_original": str(row.get("question_en", row.get("question_original", ""))).strip(),
            "survey_file": survey_file,
            "survey_date": str(row.get("survey_date", "")).strip() or None,
            "vote_id": vote_id,
            "procedure_reference": procedure_reference,
            "vote_summary_original": str(row.get("summary", row.get("vote_summary_original", ""))).strip(),
            "vote_summary_clean": vote_summary_clean,
            "vote_date": str(row.get("vote_date", "")).strip() or None,
            "days_between": int(float(row["time_delta"])) if pd.notna(row.get("time_delta")) else
                            int(float(row["days_between"])) if pd.notna(row.get("days_between")) else None,
            "similarity_score": float(row["similarity_score"]) if pd.notna(row.get("similarity_score")) else None,
            "source": str(row.get("source", "Eurobarometer")),
            "admin_validated": None,  # Always null for new candidates
        })

    if skipped_no_proc:
        logger.info(f"Skipped {skipped_no_proc} CSV rows without procedure_reference mapping")

    if not new_candidates:
        logger.info("No new candidates to insert")
        return 0

    logger.info(f"Found {len(new_candidates)} new candidate pairs not in Supabase")

    # Score the new candidates
    model, threshold, model_type, selected_features = load_model_and_config()
    emb_lookup = load_embedding_lookup()

    score_records_input = []
    for c in new_candidates:
        score_records_input.append({
            "question_clean": c["question_clean"],
            "vote_summary_clean": c["vote_summary_clean"],
            "similarity_score": c["similarity_score"] or 0,
            "days_between": c["days_between"] or 0,
        })

    probs = score_records(
        score_records_input, model, model_type, emb_lookup, selected_features,
        "cross_encoder_scores_new_candidates.npy",
    )

    # Add predicted_probability to each candidate
    for c, p in zip(new_candidates, probs):
        c["predicted_probability"] = round(float(p), 4)

    if active_learning:
        # Sort by uncertainty (closest to decision boundary = most informative)
        new_candidates.sort(key=lambda c: abs(c["predicted_probability"] - threshold))
        if limit:
            new_candidates = new_candidates[:limit]
        logger.info(
            f"Active learning mode: selected {len(new_candidates)} pairs "
            f"closest to threshold {threshold:.2f}"
        )
    else:
        # Filter to high-probability candidates only
        new_candidates = [c for c in new_candidates if c["predicted_probability"] >= min_threshold]
        logger.info(
            f"Filtered to {len(new_candidates)} candidates with P >= {min_threshold}"
        )
        if limit:
            new_candidates.sort(key=lambda c: c["predicted_probability"], reverse=True)
            new_candidates = new_candidates[:limit]

    if not new_candidates:
        return 0

    if dry_run:
        logger.info("[DRY RUN] Would insert new candidates:")
        for c in sorted(new_candidates, key=lambda x: x["predicted_probability"], reverse=True)[:10]:
            logger.info(
                f"  P={c['predicted_probability']:.3f} | "
                f"Q: {c['question_clean'][:60]}... | "
                f"V: {c['vote_summary_clean'][:60]}..."
            )
        return len(new_candidates)

    # Batch upsert — NEVER overwrite existing admin_validated
    inserted = 0
    for start in range(0, len(new_candidates), BATCH_SIZE):
        batch = new_candidates[start : start + BATCH_SIZE]
        supabase.table("survey_vote_matches").upsert(
            batch, on_conflict="match_id"
        ).execute()
        inserted += len(batch)
        logger.info(f"  Inserted {inserted}/{len(new_candidates)} new candidates")

    logger.info(f"Inserted {inserted} new candidate pairs")
    return inserted


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Push predictions to Supabase")
    parser.add_argument("--insert-new", action="store_true",
                        help="Also insert new candidate pairs from CSV")
    parser.add_argument("--csv", type=str, default=None,
                        help=f"CSV file with candidates (default: {DEFAULT_CANDIDATES_CSV.relative_to(ROOT)})")
    parser.add_argument("--active-learning", action="store_true",
                        help="Insert pairs nearest to decision boundary (most informative for labelling)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of new candidates to insert")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing to Supabase")
    parser.add_argument("--threshold", type=float, default=NEW_CANDIDATE_THRESHOLD,
                        help=f"Min predicted probability for new candidates (default: {NEW_CANDIDATE_THRESHOLD})")
    args = parser.parse_args()

    # Resolve CSV path
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = ROOT / csv_path
    else:
        csv_path = DEFAULT_CANDIDATES_CSV if DEFAULT_CANDIDATES_CSV.exists() else LEGACY_MATCHES_CSV

    supabase = get_supabase()

    # Step 1: Update existing matches
    logger.info("=" * 60)
    logger.info("Step 1: Scoring existing Supabase matches")
    logger.info("=" * 60)
    n_updated = update_existing_matches(supabase, dry_run=args.dry_run)

    # Step 2: Insert new candidates (optional)
    if args.insert_new:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Step 2: Inserting new candidate pairs")
        logger.info("=" * 60)
        logger.info(f"Source CSV: {csv_path}")
        n_inserted = insert_new_candidates(
            supabase,
            csv_path=csv_path,
            dry_run=args.dry_run,
            min_threshold=args.threshold,
            active_learning=args.active_learning,
            limit=args.limit,
        )
    else:
        n_inserted = 0
        logger.info("\nSkipping new candidate insertion (use --insert-new to enable)")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"  Updated: {n_updated} existing matches")
    logger.info(f"  Inserted: {n_inserted} new candidates")
    if args.dry_run:
        logger.info("  [DRY RUN — no changes written to Supabase]")


if __name__ == "__main__":
    main()
