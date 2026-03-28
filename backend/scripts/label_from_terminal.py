"""
Interactive terminal labelling tool for survey-vote match pairs.

Shows pairs sorted by model uncertainty (active learning), writes labels
to Supabase immediately, and can pull fresh backups for retraining.

Usage:
    make label                    # label 20 most uncertain pairs
    make label LIMIT=50           # label up to 50
    make label LOCAL=1            # offline mode (from local backup)
    make sync                     # pull fresh Supabase backup
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from eu_survey_correlation.classifier import (
    MIGRATION_DIR,
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
    print_kv,
    print_match,
    print_section,
)

ROOT = Path(__file__).resolve().parents[2]


# ── Supabase helpers ──────────────────────────────────────────────────

def get_supabase():
    from supabase import create_client

    url = os.environ["SUPABASE_URL"]
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def fetch_unlabelled_from_supabase(supabase) -> list[dict]:
    """Fetch all rows where admin_validated IS NULL."""
    rows = []
    page_size = 1000
    offset = 0
    while True:
        resp = (
            supabase.table("survey_vote_matches")
            .select("*")
            .is_("admin_validated", "null")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows.extend(resp.data)
        if len(resp.data) < page_size:
            break
        offset += page_size
    return rows


def fetch_all_from_supabase(supabase) -> list[dict]:
    """Fetch all rows for backup."""
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


def fetch_unlabelled_from_local() -> list[dict]:
    """Load unlabelled pairs from the most recent local backup."""
    backup_files = sorted(MIGRATION_DIR.glob("survey_vote_matches_backup_*.json"))
    if not backup_files:
        raise FileNotFoundError(f"No backup files in {MIGRATION_DIR}")

    path = backup_files[-1]
    log.info(f"Loading from local backup: {path.name}")
    with open(path) as f:
        all_matches = json.load(f)

    return [r for r in all_matches if r.get("admin_validated") is None]


# ── Model scoring ─────────────────────────────────────────────────────

def load_model_and_config():
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
        raise FileNotFoundError(f"No trained model at {model_path}. Run train_classifier.py first.")
    model = joblib.load(model_path)
    return model, threshold, model_type, selected_features


def _select_features(X: np.ndarray, feature_names: list[str], selected: list[str] | None) -> np.ndarray:
    if selected is None:
        return X
    indices = [feature_names.index(f) for f in selected if f in feature_names]
    return X[:, indices]


def score_pairs(
    records: list[dict],
    model,
    model_type: str,
    emb_lookup: dict,
    selected_features: list[str] | None = None,
) -> np.ndarray:
    """Score records and return P(accept) array."""
    if model_type == "setfit":
        return predict_setfit(model, records)

    from eu_survey_correlation.classifier import DATA

    feature_df = build_feature_matrix(records, emb_lookup)
    feature_names = list(feature_df.columns)
    X = feature_df.values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)

    if model_type == "hybrid":
        cache_path = DATA / "cache" / "cross_encoder_scores_labelling.npy"
        ce_scores = compute_cross_encoder_scores(records, cache_path)
        X = np.column_stack([X, ce_scores])

    X = _select_features(X, feature_names, selected_features)
    return model.predict_proba(X)[:, 1]


# ── Backup ────────────────────────────────────────────────────────────

def pull_backup(supabase) -> Path:
    """Download all Supabase rows to a timestamped JSON backup."""
    MIGRATION_DIR.mkdir(parents=True, exist_ok=True)

    rows = fetch_all_from_supabase(supabase)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = MIGRATION_DIR / f"survey_vote_matches_backup_{timestamp}.json"

    with open(path, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    log.info(f"Backup saved: {path.name} ({len(rows)} rows)")
    return path


# ── Interactive labelling loop ────────────────────────────────────────

def label_loop(pairs: list[dict], threshold: float, supabase=None, local: bool = False) -> dict:
    """Run the interactive labelling loop. Returns session stats."""
    undo_stack: list[tuple[str, bool | None]] = []  # (match_id, previous_value)
    n_labelled = 0
    n_yes = 0
    n_no = 0
    total = len(pairs)

    for i, pair in enumerate(pairs):
        # Clear screen and show progress
        console.clear()
        print_section(f"Labelling — {n_labelled} done, {i+1}/{total}")

        match_id = pair["match_id"]
        question = pair.get("question_clean") or pair.get("question_original") or ""
        vote = pair.get("vote_summary_clean") or pair.get("vote_summary_original") or ""
        sim = pair.get("similarity_score")
        prob = pair.get("predicted_probability")
        days = pair.get("days_between")

        print_match(
            match_id[:12],
            question,
            vote,
            score=float(sim) if sim is not None else None,
            probability=float(prob) if prob is not None else None,
            days=float(days) if days is not None else None,
            threshold=threshold,
        )

        # Extra context
        if pair.get("match_key"):
            print_kv("Match key", pair["match_key"])
        if pair.get("source"):
            print_kv("Source", pair["source"])
        if pair.get("survey_file"):
            print_kv("Survey", pair["survey_file"])
        if pair.get("procedure_reference"):
            print_kv("Procedure", pair["procedure_reference"])
        console.print()

        # Prompt
        while True:
            console.print("[bold green]y[/]=yes  [bold red]n[/]=no  [bold yellow]s[/]=skip  [bold blue]u[/]=undo  [bold]q[/]=quit")
            choice = console.input("[bold]> [/]").strip().lower()

            if choice in ("y", "yes"):
                _write_label(supabase, match_id, True, local)
                undo_stack.append((match_id, None))  # previous was None (unlabelled)
                n_labelled += 1
                n_yes += 1
                console.print("[green]✓ Labelled YES[/]")
                break

            elif choice in ("n", "no"):
                _write_label(supabase, match_id, False, local)
                undo_stack.append((match_id, None))
                n_labelled += 1
                n_no += 1
                console.print("[red]✗ Labelled NO[/]")
                break

            elif choice in ("s", "skip"):
                break

            elif choice in ("u", "undo"):
                if not undo_stack:
                    console.print("[yellow]Nothing to undo[/]")
                    continue
                prev_id, prev_val = undo_stack.pop()
                _write_label(supabase, prev_id, prev_val, local)
                n_labelled -= 1
                # Adjust counts
                # We don't track which was yes/no on undo, so approximate
                if n_yes > 0 and n_no > 0:
                    # Can't know for sure, but it's just for display
                    pass
                console.print(f"[yellow]↩ Reverted {prev_id[:12]}[/]")
                continue

            elif choice in ("q", "quit"):
                return {"labelled": n_labelled, "yes": n_yes, "no": n_no, "total": total}

            else:
                console.print("[dim]Invalid input. Use y/n/s/u/q[/]")

    return {"labelled": n_labelled, "yes": n_yes, "no": n_no, "total": total}


def _write_label(supabase, match_id: str, value: bool | None, local: bool) -> None:
    """Write a label to Supabase (or log for local mode)."""
    if local:
        label_str = "NULL" if value is None else str(value)
        log.info(f"[dim]Local mode: {match_id[:12]} → admin_validated={label_str}[/]",
                 extra={"markup": True})
        return

    supabase.table("survey_vote_matches").update(
        {"admin_validated": value}
    ).eq("match_id", match_id).execute()


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive terminal labelling for survey-vote pairs")
    parser.add_argument("--limit", type=int, default=20, help="Max pairs to show (default: 20)")
    parser.add_argument("--all", action="store_true", help="Show all unlabelled pairs (no limit)")
    parser.add_argument("--local", action="store_true", help="Offline mode: read from local backup, don't write to Supabase")
    parser.add_argument("--sync-only", action="store_true", help="Only pull a fresh backup from Supabase, then exit")
    args = parser.parse_args()

    # Sync-only mode
    if args.sync_only:
        print_section("Pulling fresh backup from Supabase")
        supabase = get_supabase()
        path = pull_backup(supabase)
        print_kv("Saved to", str(path))
        return

    # Load model
    print_section("Loading model")
    model, threshold, model_type, selected_features = load_model_and_config()
    emb_lookup = load_embedding_lookup()
    print_kv("Model", model_type)
    print_kv("Threshold", f"{threshold:.2f}")

    # Fetch unlabelled pairs
    supabase = None
    if args.local:
        print_section("Loading unlabelled pairs (local)")
        pairs = fetch_unlabelled_from_local()
    else:
        print_section("Fetching unlabelled pairs from Supabase")
        supabase = get_supabase()
        pairs = fetch_unlabelled_from_supabase(supabase)

    if not pairs:
        log.info("No unlabelled pairs found. Nothing to label!")
        return

    log.info(f"Found [bold]{len(pairs)}[/] unlabelled pairs", extra={"markup": True})

    # Score pairs for uncertainty sorting
    print_section("Scoring pairs")
    score_input = [
        {
            "question_clean": p.get("question_clean") or "",
            "vote_summary_clean": p.get("vote_summary_clean") or "",
            "similarity_score": p.get("similarity_score") or 0,
            "days_between": p.get("days_between") or 0,
        }
        for p in pairs
    ]
    probs = score_pairs(score_input, model, model_type, emb_lookup, selected_features)

    for p, prob in zip(pairs, probs):
        p["predicted_probability"] = float(prob)
        p["uncertainty"] = float(abs(prob - threshold))

    # Sort by uncertainty (most informative first)
    pairs.sort(key=lambda x: x["uncertainty"])

    # Apply limit
    if not args.all:
        pairs = pairs[: args.limit]

    log.info(f"Presenting [bold]{len(pairs)}[/] pairs (sorted by uncertainty)", extra={"markup": True})
    console.print()

    # Run labelling loop
    stats = label_loop(pairs, threshold, supabase=supabase, local=args.local)

    # Session summary
    console.print()
    print_section("Session Summary")
    print_kv("Labelled", f"{stats['labelled']} / {stats['total']} presented")
    print_kv("Yes", str(stats["yes"]))
    print_kv("No", str(stats["no"]))
    if stats["labelled"] > 0:
        rate = stats["yes"] / stats["labelled"] * 100
        print_kv("Accept rate", f"{rate:.0f}%")

    # Offer backup
    if not args.local and stats["labelled"] > 0:
        console.print()
        choice = console.input("[bold]Pull fresh backup for retraining? [Y/n] [/]").strip().lower()
        if choice in ("", "y", "yes"):
            path = pull_backup(supabase)
            print_kv("Backup saved", str(path))
            console.print("[dim]Run 'make retrain-quick' to retrain with new labels[/]")


if __name__ == "__main__":
    main()
