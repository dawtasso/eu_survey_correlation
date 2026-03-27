"""
Generate new survey<->vote candidate pairs from the full survey + procedure data.

Usage:
    make generate-candidates
    make generate-candidates TOP_K=3 THRESHOLD=0.45
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

SURVEYS_CSV = DATA / "surveys" / "all_survey_questions_simplified.csv"
PROCEDURES_CSV = DATA / "votes" / "procedure_summaries_simplified.csv"
VOTES_CSV = DATA / "votes" / "votes.csv"
OUTPUT_CSV = DATA / "matches" / "all_candidates.csv"

SURVEY_EMB_CACHE = DATA / "cache" / "survey_questions_embeddings.parquet"
PROCEDURE_EMB_CACHE = DATA / "cache" / "procedure_summaries_embeddings.parquet"

sys.path.insert(0, str(Path(__file__).parent))
from match_id_utils import build_main_vote_map, make_match_id, make_match_key

from eu_survey_correlation.logging import console, log, print_kv, print_section


def load_surveys() -> pd.DataFrame:
    df = pd.read_csv(SURVEYS_CSV)
    df = df.dropna(subset=["question_clean"])
    df["question_clean"] = df["question_clean"].astype(str).str.strip()
    df = df[df["question_clean"].str.len() > 10]
    log.info(f"Loaded {len(df)} survey questions from {SURVEYS_CSV.name}")
    return df.reset_index(drop=True)


def load_procedures_with_votes() -> pd.DataFrame:
    procs = pd.read_csv(PROCEDURES_CSV)
    procs = procs.dropna(subset=["simplified_summary"])
    procs["simplified_summary"] = procs["simplified_summary"].astype(str).str.strip()
    procs = procs[procs["simplified_summary"].str.len() > 10]
    log.info(f"Loaded {len(procs)} procedures with summaries")

    proc_to_vote = build_main_vote_map(VOTES_CSV)
    log.info(f"Mapped {len(proc_to_vote)} procedures to main vote_id")

    votes_df = pd.read_csv(VOTES_CSV, usecols=["id", "timestamp", "procedure_reference"])
    vote_info = votes_df.set_index("id")[["timestamp"]].to_dict()["timestamp"]

    procs["vote_id"] = procs["reference"].map(proc_to_vote)
    procs = procs.dropna(subset=["vote_id"])
    procs["vote_id"] = procs["vote_id"].astype(int)
    procs["vote_date"] = procs["vote_id"].map(vote_info)

    log.info(f"After join: {len(procs)} procedures with vote_id")
    return procs.reset_index(drop=True)


def embed_texts(texts: list[str], cache_path: Path, label: str) -> np.ndarray:
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        emb_cols = sorted(
            [c for c in cached.columns if c.startswith("emb_")],
            key=lambda c: int(c.split("_")[1]),
        )
        if len(cached) == len(texts) and "text" in cached.columns:
            cached_texts = cached["text"].tolist()
            if cached_texts == texts:
                log.info(f"Loaded cached {label} embeddings ({len(cached)} x {len(emb_cols)})")
                return cached[emb_cols].values.astype(np.float32)
        log.info(f"Cache mismatch for {label}, recomputing...")

    from eu_survey_correlation.embeddings.embedder import Embedder

    embedder = Embedder()
    embeddings = embedder.embed_texts(texts, batch_size=256)

    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    cache_df = pd.DataFrame({"text": texts} | {col: embeddings[:, i] for i, col in enumerate(emb_cols)})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_parquet(cache_path, index=False)
    log.info(f"Cached {label} embeddings to {cache_path.name}")

    return embeddings


def compute_matches(
    survey_df: pd.DataFrame,
    survey_emb: np.ndarray,
    proc_df: pd.DataFrame,
    proc_emb: np.ndarray,
    top_k: int,
    threshold: float,
    batch_size: int = 512,
) -> pd.DataFrame:
    from tqdm import tqdm

    n_surveys = len(survey_df)
    n_procs = len(proc_df)
    log.info(f"Computing {n_surveys} x {n_procs} = {n_surveys * n_procs:,} similarity pairs...")

    rows: list[dict] = []

    for start in tqdm(range(0, n_surveys, batch_size), desc="Matching"):
        end = min(start + batch_size, n_surveys)
        batch_emb = survey_emb[start:end]

        sims = batch_emb @ proc_emb.T

        for i in range(sims.shape[0]):
            survey_idx = start + i
            scores = sims[i]

            above = np.where(scores >= threshold)[0]
            if len(above) == 0:
                continue

            if len(above) > top_k:
                top_scores = scores[above]
                top_proc_idx = above[np.argpartition(top_scores, -top_k)[-top_k:]]
            else:
                top_proc_idx = above

            top_proc_idx = top_proc_idx[np.argsort(scores[top_proc_idx])[::-1]]

            s_row = survey_df.iloc[survey_idx]
            for proc_idx in top_proc_idx:
                p_row = proc_df.iloc[proc_idx]
                question_id = str(s_row["sheet_id"])
                survey_file = str(s_row["file_name"])
                procedure_ref = str(p_row["reference"])

                rows.append({
                    "match_id": make_match_id(question_id, survey_file, procedure_ref),
                    "match_key": make_match_key(question_id, survey_file, procedure_ref),
                    "question_id": question_id,
                    "question_clean": s_row["question_clean"],
                    "question_original": str(s_row.get("question_en", "")),
                    "survey_file": survey_file,
                    "vote_id": int(p_row["vote_id"]),
                    "procedure_reference": procedure_ref,
                    "vote_summary_clean": p_row["simplified_summary"],
                    "vote_summary_original": str(p_row.get("summary_text", "")),
                    "vote_date": str(p_row.get("vote_date", "")),
                    "similarity_score": float(scores[proc_idx]),
                    "source": "Eurobarometer",
                })

    matches = pd.DataFrame(rows)
    if not matches.empty:
        matches = matches.sort_values("similarity_score", ascending=False)
        matches = matches.drop_duplicates(subset="match_id", keep="first")

    log.info(f"Generated {len(matches)} candidate pairs")
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate survey-vote candidate pairs")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k matches per question (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.40, help="Min cosine similarity (default: 0.40)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for similarity computation")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help=f"Output CSV (default: {OUTPUT_CSV.relative_to(ROOT)})")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else OUTPUT_CSV

    # 1. Load data
    print_section("Loading data")
    survey_df = load_surveys()
    proc_df = load_procedures_with_votes()

    # 2. Embed
    print_section("Embedding")
    survey_emb = embed_texts(survey_df["question_clean"].tolist(), SURVEY_EMB_CACHE, "survey")
    proc_emb = embed_texts(proc_df["simplified_summary"].tolist(), PROCEDURE_EMB_CACHE, "procedure")

    # 3. Match
    print_section(f"Matching (top_k={args.top_k}, threshold={args.threshold})")
    matches = compute_matches(
        survey_df, survey_emb, proc_df, proc_emb,
        top_k=args.top_k, threshold=args.threshold, batch_size=args.batch_size,
    )

    if matches.empty:
        log.warning("No matches found above threshold!")
        return

    # 4. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output_path, index=False)

    print_section("Results")
    print_kv("Saved", f"{len(matches)} candidates to {output_path}")
    print_kv("Score", f"mean={matches['similarity_score'].mean():.3f}  "
                      f"std={matches['similarity_score'].std():.3f}  "
                      f"min={matches['similarity_score'].min():.3f}  "
                      f"max={matches['similarity_score'].max():.3f}")
    print_kv("Unique questions", matches["question_id"].nunique())
    print_kv("Unique procedures", matches["procedure_reference"].nunique())
    console.print("\n[bold green]Done![/]")


if __name__ == "__main__":
    main()
