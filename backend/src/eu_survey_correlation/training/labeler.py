"""Robust LLM-based labeling for survey-vote pair relatedness.

Uses a strict prompt with few-shot examples and multi-pass validation
to produce high-quality training labels.
"""

import json
import re
import statistics
from pathlib import Path

import numpy as np
import ollama
import pandas as pd
from loguru import logger
from tqdm import tqdm

from eu_survey_correlation.classification.dates import (
    build_survey_date_map as _build_survey_date_map,
)
from eu_survey_correlation.classification.dates import (
    build_vote_date_map as _build_vote_date_map,
)
from eu_survey_correlation.classification.dates import (
    get_survey_date as _get_survey_date,
)

DEFAULT_MODEL = "mistral"

LABELING_PROMPT = """\
You are evaluating whether a European Parliament vote matches a Eurobarometer \
survey question in a way that lets us measure if parliament followed public opinion.

A pair is a VALID MATCH only if ALL of these are true:
1. The survey asks for citizens' OPINION or PREFERENCE about what should be done \
(NOT asking about something already decided or already in place)
2. The vote ACTS ON the same specific topic the survey asked about
3. A citizen's answer to the survey would be directly relevant to whether \
the vote should pass or not

VALID — opinion before action:
- Survey: "Should the EU provide financial support to member states affected by COVID-19?"
  Vote: "Recovery plan of 750 billion euros for member states" → VALID
  (citizens express preference on what should be done → parliament later acts on it)
- Survey: "Do you think the EU should do more to fight climate change?"
  Vote: "Regulation to reduce CO2 emissions by 55% by 2030" → VALID
  (citizens express preference → parliament legislates)

INVALID — retrospective (survey references something already decided):
- Survey: "What are your thoughts about the recovery plan of 750 billion euros \
supporting all Member States?"
  Vote: "Recovery plan of 750 billion euros" → INVALID
  (the survey MENTIONS the plan as already existing — it is asking about a past \
decision, not expressing a preference before the vote)
- Survey: "The European Year of Skills is taking place. Tell me if it applies to you."
  Vote: "Decision on a European Year of Skills 2023" → INVALID
  (the survey references the Year of Skills as already happening)

INVALID — different specific topic within same domain:
- Survey: "Are you concerned about energy prices?"
  Vote: "Regulation on energy labelling of appliances" → INVALID
  (prices ≠ labelling, even though both are about energy)

INVALID — no actionable opinion:
- Survey: "Do you trust EU institutions?"
  Vote: "EU allocates 100 million euros to Ukraine" → INVALID
  (trust is too abstract to connect to a specific legislative action)

SURVEY QUESTION:
{question_text}

VOTE SUMMARY:
{vote_summary}

Respond ONLY with valid JSON (no markdown, no extra text):
{{"score": <float 0.0 to 1.0>, "label": "<valid or retrospective or indirect or \
unrelated>", "explanation": "<one sentence>"}}

Scoring guide:
- 0.0-0.2: unrelated (different topics entirely)
- 0.2-0.4: indirect (same broad domain, but different specific concern)
- 0.4-0.6: retrospective (same topic, but the survey references an already-decided \
policy — not a forward-looking opinion)
- 0.6-1.0: valid (citizens express opinion/preference → vote acts on it)\
"""


def _parse_response(content: str) -> dict:
    """Parse LLM JSON response with regex fallback."""
    # Try direct JSON parse
    try:
        result = json.loads(content)
        return _validate_result(result)
    except json.JSONDecodeError:
        pass

    # Fallback: extract JSON from surrounding text
    json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return _validate_result(result)
        except json.JSONDecodeError:
            pass

    logger.warning(f"Could not parse LLM response: {content[:200]}")
    return {"score": -1.0, "label": "error", "explanation": "parse error"}


def _validate_result(result: dict) -> dict:
    """Ensure result has expected keys and types."""
    score = float(result.get("score", -1.0))
    score = max(0.0, min(1.0, score))  # clamp to [0, 1]
    label = str(result.get("label", "error"))
    explanation = str(result.get("explanation", ""))
    return {"score": score, "label": label, "explanation": explanation}


def label_pair(
    question_text: str,
    vote_summary: str,
    model: str = DEFAULT_MODEL,
    n_passes: int = 3,
) -> dict:
    """
    Label a single pair with multi-pass validation.

    Returns:
        dict with: score (median), label, explanation (from median pass),
        scores (list of all pass scores), flagged (bool, True if disagreement > 0.3).
    """
    prompt = LABELING_PROMPT.format(
        question_text=question_text,
        vote_summary=vote_summary,
    )

    pass_results = []
    for _ in range(n_passes):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            content = response["message"]["content"].strip()
            pass_results.append(_parse_response(content))
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            pass_results.append(
                {"score": -1.0, "label": "error", "explanation": f"llm error: {e}"}
            )

    # Filter out failed passes
    valid_scores = [r["score"] for r in pass_results if r["score"] >= 0]

    if not valid_scores:
        return {
            "score": -1.0,
            "label": "error",
            "explanation": "all passes failed",
            "scores": [r["score"] for r in pass_results],
            "flagged": True,
        }

    median_score = statistics.median(valid_scores)
    score_range = max(valid_scores) - min(valid_scores)
    flagged = score_range > 0.3

    # Pick the pass result closest to the median for label/explanation
    best_pass = min(
        [r for r in pass_results if r["score"] >= 0],
        key=lambda r: abs(r["score"] - median_score),
    )

    return {
        "score": round(median_score, 3),
        "label": best_pass["label"],
        "explanation": best_pass["explanation"],
        "scores": [round(r["score"], 3) for r in pass_results],
        "flagged": flagged,
    }


def sample_training_pairs(
    matches_path: Path,
    survey_embeddings_path: Path,
    vote_embeddings_path: Path,
    n_positive: int = 250,
    n_negative: int = 250,
    seed: int = 42,
    metadata_path: Path | None = None,
    votes_csv_path: Path | None = None,
) -> pd.DataFrame:
    """
    Sample a balanced set of candidate pairs for labeling.

    Filters out pairs where the survey was published after the vote (retrospective).
    Positives: highest-similarity matches where survey predates vote.
    Hard negatives: pairs near the similarity boundary.
    Random negatives: random survey x vote pairings.

    Returns:
        DataFrame with columns: question_text, vote_summary, vote_id, sample_type,
        similarity_score.
    """
    rng = np.random.default_rng(seed)

    # --- Build date maps for temporal filtering ---
    if metadata_path is None:
        metadata_path = Path("data/surveys/distributions_metadata.json")
    if votes_csv_path is None:
        votes_csv_path = Path("data/votes/votes.csv")

    edition_dates = _build_survey_date_map(metadata_path)
    vote_dates = _build_vote_date_map(votes_csv_path)
    logger.info(
        f"Date maps: {len(edition_dates)} survey editions, "
        f"{len(vote_dates)} votes with dates"
    )

    # Load matches for positive sampling
    matches_df = pd.read_csv(matches_path)
    logger.info(f"Loaded {len(matches_df)} candidate matches")

    # --- Temporal filtering: keep only survey_date < vote_date ---
    matches_df["survey_date"] = matches_df["file_name"].apply(
        lambda fn: _get_survey_date(fn, edition_dates)
    )
    matches_df["vote_date"] = matches_df["vote_id"].astype(str).map(vote_dates)

    has_dates = matches_df["survey_date"].notna() & matches_df["vote_date"].notna()
    survey_before_vote = matches_df["survey_date"] < matches_df["vote_date"]
    temporal_ok = has_dates & survey_before_vote

    n_before = len(matches_df)
    matches_df = matches_df[temporal_ok].copy()
    logger.info(
        f"Temporal filter: {n_before} -> {len(matches_df)} pairs "
        f"(kept only survey before vote)"
    )

    # --- Opinion filter: keep only forward-looking opinion questions ---
    opinion_keywords = [
        "should",
        "do you think",
        "do you agree",
        "in favour",
        "in favor",
        "would you",
        "do you support",
        "are you in favour",
        "to what extent",
        "how important",
        "how effective",
    ]
    opinion_pattern = "|".join(opinion_keywords)
    is_opinion = (
        matches_df["question_text"].str.lower().str.contains(opinion_pattern, na=False)
    )
    n_before_opinion = len(matches_df)
    matches_df = matches_df[is_opinion].copy()
    logger.info(
        f"Opinion filter: {n_before_opinion} -> {len(matches_df)} pairs "
        f"(kept only forward-looking opinion questions)"
    )

    # Sample positives — take highest similarity pairs
    n_pos = min(n_positive, len(matches_df))
    pos_sample = matches_df.nlargest(n_pos, "similarity_score")
    pos_sample = pos_sample[
        ["question_text", "vote_summary", "vote_id", "similarity_score"]
    ].copy()
    pos_sample["sample_type"] = "positive"

    # Load embeddings for negative sampling
    survey_df = pd.read_parquet(survey_embeddings_path)
    vote_df = pd.read_parquet(vote_embeddings_path)

    survey_emb_cols = sorted(
        [c for c in survey_df.columns if c.startswith("emb_")],
        key=lambda c: int(c.split("_")[1]),
    )
    vote_emb_cols = sorted(
        [c for c in vote_df.columns if c.startswith("emb_")],
        key=lambda c: int(c.split("_")[1]),
    )
    survey_embs = survey_df[survey_emb_cols].values.astype(np.float32)
    vote_embs = vote_df[vote_emb_cols].values.astype(np.float32)

    # Normalize
    survey_embs = survey_embs / (
        np.linalg.norm(survey_embs, axis=1, keepdims=True) + 1e-9
    )
    vote_embs = vote_embs / (np.linalg.norm(vote_embs, axis=1, keepdims=True) + 1e-9)

    # --- Hard negatives: pairs with similarity between 0.35 and 0.50 ---
    n_hard = n_negative // 2
    hard_neg_rows = []
    survey_indices = rng.choice(
        len(survey_df), size=min(200, len(survey_df)), replace=False
    )

    for s_idx in survey_indices:
        if len(hard_neg_rows) >= n_hard:
            break
        sims = survey_embs[s_idx] @ vote_embs.T
        # Find votes in the 0.35-0.50 similarity range
        candidates = np.where((sims >= 0.35) & (sims < 0.50))[0]
        if len(candidates) > 0:
            v_idx = rng.choice(candidates)
            hard_neg_rows.append(
                {
                    "question_text": survey_df.iloc[s_idx].get("question_en", ""),
                    "vote_summary": str(vote_df.iloc[v_idx].get("summary", "")),
                    "vote_id": vote_df.iloc[v_idx].get("vote_id", ""),
                    "similarity_score": float(sims[v_idx]),
                    "sample_type": "hard_negative",
                }
            )

    hard_neg_df = pd.DataFrame(hard_neg_rows)
    logger.info(f"Sampled {len(hard_neg_df)} hard negatives")

    # --- Random negatives: fully random pairings ---
    n_random = n_negative - len(hard_neg_rows)
    rand_s_indices = rng.choice(len(survey_df), size=n_random)
    rand_v_indices = rng.choice(len(vote_df), size=n_random)

    random_neg_rows = []
    for s_idx, v_idx in zip(rand_s_indices, rand_v_indices):
        sim = float(survey_embs[s_idx] @ vote_embs[v_idx])
        random_neg_rows.append(
            {
                "question_text": survey_df.iloc[s_idx].get("question_en", ""),
                "vote_summary": str(vote_df.iloc[v_idx].get("summary", "")),
                "vote_id": vote_df.iloc[v_idx].get("vote_id", ""),
                "similarity_score": sim,
                "sample_type": "random_negative",
            }
        )

    random_neg_df = pd.DataFrame(random_neg_rows)
    logger.info(f"Sampled {len(random_neg_df)} random negatives")

    # Combine all
    result = pd.concat([pos_sample, hard_neg_df, random_neg_df], ignore_index=True)
    result = result.drop_duplicates(subset=["question_text", "vote_id"], keep="first")
    logger.info(f"Total sampled pairs: {len(result)}")
    return result


def label_dataframe(
    df: pd.DataFrame,
    output_path: Path,
    model: str = DEFAULT_MODEL,
    n_passes: int = 3,
    already_labeled: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Label all pairs in a DataFrame using multi-pass LLM evaluation.

    Saves incrementally after each pair. Supports resume via already_labeled.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labeled_rows = []
    if already_labeled is not None and not already_labeled.empty:
        labeled_rows = already_labeled.to_dict("records")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling pairs"):
        result = label_pair(
            question_text=str(row["question_text"]),
            vote_summary=str(row["vote_summary"]),
            model=model,
            n_passes=n_passes,
        )

        row_dict = row.to_dict()
        row_dict["llm_score"] = result["score"]
        row_dict["llm_label"] = result["label"]
        row_dict["llm_explanation"] = result["explanation"]
        row_dict["llm_scores"] = str(result["scores"])
        row_dict["llm_flagged"] = result["flagged"]
        labeled_rows.append(row_dict)

        # Save incrementally
        pd.DataFrame(labeled_rows).to_csv(output_path, index=False)

    result_df = pd.DataFrame(labeled_rows)
    n_new = len(df)
    new_rows = result_df.tail(n_new)
    n_flagged = new_rows["llm_flagged"].sum()
    logger.success(
        f"Labeling complete: {n_new} pairs labeled, "
        f"{n_flagged} flagged for review, "
        f"avg score: {new_rows['llm_score'].mean():.3f}"
    )
    return result_df
