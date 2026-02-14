"""OOP classification engine for survey-vote pairs.

Provides LLMClient for Ollama interaction, and three classifier classes:
- SurveyClassifier: classifies questions as opinion_forward / not_forward
- VoteClassifier: classifies votes as substantive / procedural
- PairValidator: validates matched pairs as valid / not_valid
"""

import json
import re
from pathlib import Path

import ollama
import pandas as pd
from loguru import logger
from tqdm import tqdm

from .prompts import (
    PAIR_VALIDATE_PROMPT,
    QUESTION_CLASSIFY_PROMPT,
    VOTE_CLASSIFY_PROMPT,
)

DEFAULT_MODEL = "mistral"


class LLMClient:
    """Shared Ollama LLM client with JSON parsing and retry logic."""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._validate_model()

    def _validate_model(self) -> None:
        try:
            ollama.show(self.model)
            logger.success(f"Model '{self.model}' is available")
        except ollama.ResponseError:
            logger.error(
                f"Model '{self.model}' not found. Run: ollama pull {self.model}"
            )
            raise

    def call(self, prompt: str) -> dict:
        """Call Ollama and return parsed JSON dict. Returns error dict on failure."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature},
            )
            content = response["message"]["content"].strip()
            return self._parse_json(content)
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return {"error": str(e)}

    @staticmethod
    def _parse_json(content: str) -> dict:
        """Parse JSON from LLM output with regex fallback."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse LLM response: {content[:200]}")
        return {"error": "parse_error", "raw": content[:200]}


class SurveyClassifier:
    """Classifies survey questions as opinion_forward or not_forward.

    Deduplicates by question text before calling the LLM.
    Saves incrementally to support resume on interruption.
    """

    OUTPUT_FILE = "questions_classified.csv"

    def __init__(self, llm: LLMClient, output_dir: Path):
        self.llm = llm
        self.output_dir = Path(output_dir)
        self.output_path = self.output_dir / self.OUTPUT_FILE

    def classify(
        self,
        surveys_df: pd.DataFrame,
        *,
        resume: bool = False,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Classify unique survey questions.

        Parameters
        ----------
        surveys_df : DataFrame with ``question_en`` and ``file_name`` columns.
        resume : Skip already-classified questions.
        limit : Process at most N questions (for testing).

        Returns
        -------
        DataFrame with ``question_type`` and ``question_type_explanation`` columns.
        """
        # Deduplicate by question text
        unique = surveys_df.drop_duplicates(subset=["question_en"]).copy()
        logger.info(
            f"SurveyClassifier: {len(unique)} unique questions "
            f"(from {len(surveys_df)} total)"
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resume support
        done_texts: set[str] = set()
        rows: list[dict] = []
        if resume and self.output_path.exists():
            existing = pd.read_csv(self.output_path)
            done_texts = set(existing["question_en"])
            rows = existing.to_dict("records")
            logger.info(f"Resuming: {len(done_texts)} already classified")

        remaining = unique[~unique["question_en"].isin(done_texts)]
        if limit is not None:
            remaining = remaining.head(limit)
        logger.info(f"Remaining: {len(remaining)} questions to classify")

        for _, row in tqdm(
            remaining.iterrows(), total=len(remaining), desc="Classifying questions"
        ):
            prompt = QUESTION_CLASSIFY_PROMPT.format(question_text=row["question_en"])
            result = self.llm.call(prompt)

            rows.append(
                {
                    "question_en": row["question_en"],
                    "question_type": result.get("type", "error"),
                    "question_type_explanation": result.get(
                        "explanation", result.get("error", "")
                    ),
                }
            )
            pd.DataFrame(rows).to_csv(self.output_path, index=False)

        result_df = pd.DataFrame(rows)
        n_forward = (result_df["question_type"] == "opinion_forward").sum()
        logger.success(
            f"SurveyClassifier done: {n_forward}/{len(result_df)} opinion_forward "
            f"({n_forward / max(len(result_df), 1) * 100:.0f}%)"
        )
        return result_df

    def load_results(self) -> pd.DataFrame:
        """Load previously saved classification results."""
        if not self.output_path.exists():
            raise FileNotFoundError(
                f"No results at {self.output_path}. Run classify() first."
            )
        return pd.read_csv(self.output_path)

    def get_forward_looking(self) -> pd.DataFrame:
        """Load results and filter to opinion_forward questions only."""
        df = self.load_results()
        return df[df["question_type"] == "opinion_forward"].copy()


class VoteClassifier:
    """Classifies EP votes as substantive or procedural.

    Deduplicates by vote_id before calling the LLM.
    Saves incrementally to support resume on interruption.
    """

    OUTPUT_FILE = "votes_classified.csv"

    def __init__(self, llm: LLMClient, output_dir: Path):
        self.llm = llm
        self.output_dir = Path(output_dir)
        self.output_path = self.output_dir / self.OUTPUT_FILE

    def classify(
        self,
        votes_df: pd.DataFrame,
        *,
        resume: bool = False,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Classify unique votes.

        Parameters
        ----------
        votes_df : DataFrame with ``vote_id`` and ``summary`` columns.
        resume : Skip already-classified votes.
        limit : Process at most N votes (for testing).

        Returns
        -------
        DataFrame with ``vote_type`` and ``vote_type_explanation`` columns.
        """
        unique = votes_df.drop_duplicates(subset=["vote_id"]).copy()
        logger.info(
            f"VoteClassifier: {len(unique)} unique votes "
            f"(from {len(votes_df)} total)"
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resume support
        done_ids: set[str] = set()
        rows: list[dict] = []
        if resume and self.output_path.exists():
            existing = pd.read_csv(self.output_path)
            done_ids = set(existing["vote_id"].astype(str))
            rows = existing.to_dict("records")
            logger.info(f"Resuming: {len(done_ids)} already classified")

        remaining = unique[~unique["vote_id"].astype(str).isin(done_ids)]
        if limit is not None:
            remaining = remaining.head(limit)
        logger.info(f"Remaining: {len(remaining)} votes to classify")

        for _, row in tqdm(
            remaining.iterrows(), total=len(remaining), desc="Classifying votes"
        ):
            prompt = VOTE_CLASSIFY_PROMPT.format(vote_summary=row["summary"])
            result = self.llm.call(prompt)

            rows.append(
                {
                    "vote_id": row["vote_id"],
                    "summary": row["summary"],
                    "vote_type": result.get("type", "error"),
                    "vote_type_explanation": result.get(
                        "explanation", result.get("error", "")
                    ),
                }
            )
            pd.DataFrame(rows).to_csv(self.output_path, index=False)

        result_df = pd.DataFrame(rows)
        n_subst = (result_df["vote_type"] == "substantive").sum()
        logger.success(
            f"VoteClassifier done: {n_subst}/{len(result_df)} substantive "
            f"({n_subst / max(len(result_df), 1) * 100:.0f}%)"
        )
        return result_df

    def load_results(self) -> pd.DataFrame:
        """Load previously saved classification results."""
        if not self.output_path.exists():
            raise FileNotFoundError(
                f"No results at {self.output_path}. Run classify() first."
            )
        return pd.read_csv(self.output_path)

    def get_substantive(self) -> pd.DataFrame:
        """Load results and filter to substantive votes only."""
        df = self.load_results()
        return df[df["vote_type"] == "substantive"].copy()


class PairValidator:
    """Validates survey-vote pairs as valid or not_valid.

    Expects pairs that already passed question + vote classification filters.
    Saves incrementally to support resume on interruption.
    """

    OUTPUT_FILE = "pairs_validated.csv"

    def __init__(self, llm: LLMClient, output_dir: Path):
        self.llm = llm
        self.output_dir = Path(output_dir)
        self.output_path = self.output_dir / self.OUTPUT_FILE

    def validate(
        self,
        pairs_df: pd.DataFrame,
        *,
        resume: bool = False,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Validate candidate pairs.

        Parameters
        ----------
        pairs_df : DataFrame with ``question_en`` (or ``question_text``)
                   and ``summary`` (or ``vote_summary``) columns.
        resume : Skip already-validated pairs.
        limit : Process at most N pairs (for testing).

        Returns
        -------
        DataFrame with ``pair_valid`` (bool) and ``pair_explanation`` columns added.
        """
        logger.info(f"PairValidator: {len(pairs_df)} pairs to validate")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve column names (handle both naming conventions)
        q_col = "question_en" if "question_en" in pairs_df.columns else "question_text"
        v_col = "summary" if "summary" in pairs_df.columns else "vote_summary"

        # Resume support
        done_keys: set[tuple[str, str]] = set()
        rows: list[dict] = []
        if resume and self.output_path.exists():
            existing = pd.read_csv(self.output_path)
            done_keys = set(
                zip(existing[q_col], existing["vote_id"].astype(str))
            )
            rows = existing.to_dict("records")
            logger.info(f"Resuming: {len(done_keys)} already validated")

        remaining = pairs_df[
            ~pairs_df.apply(
                lambda r: (r[q_col], str(r["vote_id"])) in done_keys, axis=1
            )
        ]
        if limit is not None:
            remaining = remaining.head(limit)
        logger.info(f"Remaining: {len(remaining)} pairs to validate")

        for _, row in tqdm(
            remaining.iterrows(), total=len(remaining), desc="Validating pairs"
        ):
            prompt = PAIR_VALIDATE_PROMPT.format(
                question_text=row[q_col],
                vote_summary=row[v_col],
            )
            result = self.llm.call(prompt)

            row_dict = row.to_dict()
            row_dict["pair_valid"] = bool(result.get("valid", False))
            row_dict["pair_explanation"] = result.get(
                "explanation", result.get("error", "")
            )
            rows.append(row_dict)

            pd.DataFrame(rows).to_csv(self.output_path, index=False)

        result_df = pd.DataFrame(rows)
        n_valid = result_df["pair_valid"].sum() if not result_df.empty else 0
        logger.success(
            f"PairValidator done: {n_valid}/{len(result_df)} valid "
            f"({n_valid / max(len(result_df), 1) * 100:.0f}%)"
        )
        return result_df

    def load_results(self) -> pd.DataFrame:
        """Load previously saved validation results."""
        if not self.output_path.exists():
            raise FileNotFoundError(
                f"No results at {self.output_path}. Run validate() first."
            )
        return pd.read_csv(self.output_path)
