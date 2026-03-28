"""
Push classifier predictions to Supabase for frontend display.

Usage:
    make push-predictions                              # update existing only
    make add-candidates LIMIT=10                       # push 10 most uncertain
    make add-candidates-dry LIMIT=10                   # preview without writing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from eu_survey_correlation.classifier import (
    DATA,
    OUTPUT_DIR,
    SETFIT_MODEL_DIR,
    build_feature_matrix,
    compute_cross_encoder_scores,
    load_embedding_lookup,
    load_setfit,
    predict_setfit,
)
from eu_survey_correlation.logging import (
    console,
    log,
    print_candidates_table,
    print_kv,
    print_match,
    print_section,
)

sys.path.insert(0, str(Path(__file__).parent))
from match_id_utils import (
    build_main_vote_map,
    build_vote_to_procedure_map,
    make_match_id,
    make_match_key,
)

ROOT = Path(__file__).resolve().parents[2]
VOTES_CSV = ROOT / "data" / "votes" / "votes.csv"
DEFAULT_CANDIDATES_CSV = ROOT / "data" / "matches" / "all_candidates.csv"
LEGACY_MATCHES_CSV = (
    ROOT / "data" / "matches" / "simplified_michlou_survey_vote_matches_clean.csv"
)
BATCH_SIZE = 100

NEW_CANDIDATE_THRESHOLD = 0.4


def get_supabase():
    from supabase import create_client

    url = os.environ["SUPABASE_URL"]
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def fetch_all_supabase_matches(supabase) -> list[dict]:
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
    import joblib

    threshold_path = OUTPUT_DIR / "threshold.json"
    setfit_threshold_path = SETFIT_MODEL_DIR / "threshold.json"

    model_type = "lr"
    threshold = 0.5
    selected_features = None

    if threshold_path.exists():
        with open(threshold_path) as f:
            meta = json.load(f)
            threshold = meta["threshold"]
            model_type = meta.get("model_type", "lr")
            selected_features = meta.get("selected_features")

    if model_type == "setfit":
        if not SETFIT_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"threshold.json says model_type=setfit but {SETFIT_MODEL_DIR} not found."
            )
        model = load_setfit()
        if setfit_threshold_path.exists():
            with open(setfit_threshold_path) as f:
                sf_meta = json.load(f)
                threshold = sf_meta["threshold"]
        return model, threshold, "setfit", None

    model_path = OUTPUT_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run train_classifier.py first."
        )
    model = joblib.load(model_path)
    return model, threshold, model_type, selected_features


def _select_features(
    X: np.ndarray, feature_names: list[str], selected: list[str] | None
) -> np.ndarray:
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
    if model_type == "setfit":
        return predict_setfit(model, records)

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
    log.info("Fetching existing matches from Supabase...")
    rows = fetch_all_supabase_matches(supabase)
    if not rows:
        log.warning("No matches found in Supabase")
        return 0

    log.info(f"Found [bold]{len(rows)}[/] existing matches", extra={"markup": True})

    model, threshold, model_type, selected_features = load_model_and_config()
    emb_lookup = load_embedding_lookup()
    print_kv("Model", model_type)
    print_kv("Threshold", f"{threshold:.2f}")

    records = []
    for r in rows:
        records.append(
            {
                "question_clean": r.get("question_clean") or "",
                "vote_summary_clean": r.get("vote_summary_clean") or "",
                "similarity_score": r.get("similarity_score") or 0,
                "days_between": r.get("days_between") or 0,
            }
        )

    probs = score_records(
        records,
        model,
        model_type,
        emb_lookup,
        selected_features,
        "cross_encoder_scores_supabase.npy",
    )

    n_accept = int((probs >= threshold).sum())
    print_kv("Score distribution", f"mean={probs.mean():.3f}  std={probs.std():.3f}")
    print_kv("Predicted accepted", f"{n_accept}/{len(probs)}")

    if dry_run:
        log.info(
            "[bold yellow]DRY RUN[/] — would update all existing matches",
            extra={"markup": True},
        )
        for i, (r, p) in enumerate(zip(rows, probs)):
            if i < 3:
                decision = "ACCEPT" if p >= threshold else "REFUSE"
                print_match(
                    r["match_id"][:8],
                    r.get("question_clean", ""),
                    r.get("vote_summary_clean", ""),
                    probability=p,
                    threshold=threshold,
                )
        return len(rows)

    updated = 0
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        batch_probs = probs[start : start + BATCH_SIZE]

        for row, prob in zip(batch, batch_probs):
            supabase.table("survey_vote_matches").update(
                {"predicted_probability": round(float(prob), 4)}
            ).eq("match_id", row["match_id"]).execute()
            updated += 1

        log.info(f"Updated {updated}/{len(rows)} matches")

    log.info(
        f"Updated predicted_probability for [bold]{updated}[/] matches",
        extra={"markup": True},
    )
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
    if not csv_path.exists():
        log.warning(f"Candidates CSV not found: {csv_path}")
        return 0

    if not VOTES_CSV.exists():
        log.warning(f"Votes CSV not found: {VOTES_CSV}")
        return 0

    vote_to_proc = build_vote_to_procedure_map(VOTES_CSV)
    main_vote_ids = set(build_main_vote_map(VOTES_CSV).values())
    log.info(
        f"Loaded {len(vote_to_proc)} vote->procedure mappings ({len(main_vote_ids)} main)"
    )

    existing_rows = fetch_all_supabase_matches(supabase)
    existing_ids = {r["match_id"] for r in existing_rows}
    log.info(f"Existing Supabase match_ids: {len(existing_ids)}")

    df = pd.read_csv(csv_path)
    log.info(
        f"Loaded [bold]{len(df)}[/] rows from {csv_path.name}", extra={"markup": True}
    )

    new_candidates = []
    skipped_no_proc = 0

    for _, row in df.iterrows():
        question_id = str(row.get("sheet_id", row.get("question_id", ""))).strip()
        survey_file = str(row.get("file_name", row.get("survey_file", ""))).strip()

        vote_id = None
        procedure_reference = None

        if pd.notna(row.get("vote_id")):
            vote_id = int(float(row["vote_id"]))
            if vote_id in vote_to_proc and vote_id in main_vote_ids:
                procedure_reference = vote_to_proc[vote_id]

        if procedure_reference is None and pd.notna(row.get("procedure_reference")):
            procedure_reference = str(row["procedure_reference"]).strip()

        if procedure_reference is None:
            skipped_no_proc += 1
            continue

        match_id = make_match_id(question_id, survey_file, procedure_reference)

        if match_id in existing_ids:
            continue

        vote_summary_clean = str(
            row.get(
                "summary_clean",
                row.get("vote_summary_clean", row.get("simplified_summary", "")),
            )
        ).strip()

        new_candidates.append(
            {
                "match_id": match_id,
                "match_key": make_match_key(
                    question_id, survey_file, procedure_reference
                ),
                "question_id": question_id,
                "question_clean": str(row.get("question_clean", "")).strip(),
                "question_original": str(
                    row.get("question_en", row.get("question_original", ""))
                ).strip(),
                "survey_file": survey_file,
                "survey_date": str(row.get("survey_date", "")).strip() or None,
                "vote_id": vote_id,
                "procedure_reference": procedure_reference,
                "vote_summary_original": str(
                    row.get("summary", row.get("vote_summary_original", ""))
                ).strip(),
                "vote_summary_clean": vote_summary_clean,
                "vote_date": str(row.get("vote_date", "")).strip() or None,
                "days_between": (
                    int(float(row["time_delta"]))
                    if pd.notna(row.get("time_delta"))
                    else (
                        int(float(row["days_between"]))
                        if pd.notna(row.get("days_between"))
                        else None
                    )
                ),
                "similarity_score": (
                    float(row["similarity_score"])
                    if pd.notna(row.get("similarity_score"))
                    else None
                ),
                "source": str(row.get("source", "Eurobarometer")),
                "admin_validated": None,
            }
        )

    if skipped_no_proc:
        log.info(f"Skipped {skipped_no_proc} rows without procedure_reference")

    if not new_candidates:
        log.info("No new candidates to insert")
        return 0

    log.info(
        f"Found [bold]{len(new_candidates)}[/] new candidate pairs",
        extra={"markup": True},
    )

    # Score
    model, threshold, model_type, selected_features = load_model_and_config()
    emb_lookup = load_embedding_lookup()

    score_records_input = []
    for c in new_candidates:
        score_records_input.append(
            {
                "question_clean": c["question_clean"],
                "vote_summary_clean": c["vote_summary_clean"],
                "similarity_score": c["similarity_score"] or 0,
                "days_between": c["days_between"] or 0,
            }
        )

    probs = score_records(
        score_records_input,
        model,
        model_type,
        emb_lookup,
        selected_features,
        "cross_encoder_scores_new_candidates.npy",
    )

    for c, p in zip(new_candidates, probs):
        c["predicted_probability"] = round(float(p), 4)

    if active_learning:
        new_candidates.sort(key=lambda c: abs(c["predicted_probability"] - threshold))
        if limit:
            new_candidates = new_candidates[:limit]
        log.info(
            f"Active learning: selected [bold]{len(new_candidates)}[/] pairs closest to threshold {threshold:.2f}",
            extra={"markup": True},
        )
    else:
        new_candidates = [
            c for c in new_candidates if c["predicted_probability"] >= min_threshold
        ]
        log.info(
            f"Filtered to [bold]{len(new_candidates)}[/] candidates with P >= {min_threshold}",
            extra={"markup": True},
        )
        if limit:
            new_candidates.sort(key=lambda c: c["predicted_probability"], reverse=True)
            new_candidates = new_candidates[:limit]

    if not new_candidates:
        return 0

    # Show preview
    print_candidates_table(
        new_candidates, threshold=threshold, max_rows=min(3, len(new_candidates))
    )

    if dry_run:
        log.info(
            f"[bold yellow]DRY RUN[/] — would insert {len(new_candidates)} candidates",
            extra={"markup": True},
        )
        # Show a few full-text comparisons
        for c in new_candidates[:3]:
            print_match(
                c["match_id"][:8],
                c["question_clean"],
                c["vote_summary_clean"],
                score=c.get("similarity_score"),
                probability=c["predicted_probability"],
                threshold=threshold,
            )
        return len(new_candidates)

    inserted = 0
    for start in range(0, len(new_candidates), BATCH_SIZE):
        batch = new_candidates[start : start + BATCH_SIZE]
        supabase.table("survey_vote_matches").upsert(
            batch, on_conflict="match_id"
        ).execute()
        inserted += len(batch)
        log.info(f"Inserted {inserted}/{len(new_candidates)} candidates")

    log.info(
        f"Inserted [bold green]{inserted}[/] new candidate pairs",
        extra={"markup": True},
    )
    return inserted


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Push predictions to Supabase")
    parser.add_argument(
        "--insert-new",
        action="store_true",
        help="Also insert new candidate pairs from CSV",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help=f"CSV file with candidates (default: {DEFAULT_CANDIDATES_CSV.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--active-learning",
        action="store_true",
        help="Insert pairs nearest to decision boundary",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of new candidates to insert",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing to Supabase"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=NEW_CANDIDATE_THRESHOLD,
        help=f"Min predicted probability for new candidates (default: {NEW_CANDIDATE_THRESHOLD})",
    )
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = ROOT / csv_path
    else:
        csv_path = (
            DEFAULT_CANDIDATES_CSV
            if DEFAULT_CANDIDATES_CSV.exists()
            else LEGACY_MATCHES_CSV
        )

    supabase = get_supabase()

    # Step 1: Update existing
    print_section("Scoring existing Supabase matches")
    n_updated = update_existing_matches(supabase, dry_run=args.dry_run)

    # Step 2: Insert new
    if args.insert_new:
        print_section("Inserting new candidates")
        print_kv("Source CSV", csv_path)
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
        log.info("Skipping insertion (use --insert-new to enable)")

    # Summary
    print_section("Summary")
    print_kv("Updated", f"{n_updated} existing matches")
    print_kv("Inserted", f"{n_inserted} new candidates")
    if args.dry_run:
        console.print("  [bold yellow]DRY RUN — no changes written[/]")


if __name__ == "__main__":
    main()
