"""Semantic matching on filtered survey-vote subsets.

Wraps the existing Embedder for embedding and implements batched cosine
similarity with top-k extraction, operating only on pre-filtered DataFrames.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from eu_survey_correlation.embeddings.embedder import Embedder


class SemanticMatcher:
    """Embed and match filtered surveys against filtered votes.

    Unlike the original VoteSurveyMatcher (which loads pre-computed parquet
    files for the full dataset), this class embeds on-the-fly from filtered
    DataFrames — so only opinion_forward questions and substantive votes
    get embedded and compared.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 256,
        top_k: int = 5,
        threshold: float = 0.5,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.top_k = top_k
        self.threshold = threshold
        self._embedder: Embedder | None = None

    @property
    def embedder(self) -> Embedder:
        """Lazy-load the SentenceTransformer model."""
        if self._embedder is None:
            self._embedder = Embedder(self.model_name)
        return self._embedder

    def match(
        self,
        surveys_df: pd.DataFrame,
        votes_df: pd.DataFrame,
        survey_text_col: str = "question_en",
        vote_text_col: str = "summary",
    ) -> pd.DataFrame:
        """Embed both sides and find top-k matches per survey question.

        Parameters
        ----------
        surveys_df : Filtered surveys with at least ``survey_text_col`` and
                     ``file_name`` columns.
        votes_df : Filtered votes with at least ``vote_text_col`` and
                   ``vote_id`` columns.
        survey_text_col : Column containing survey question text.
        vote_text_col : Column containing vote summary text.

        Returns
        -------
        DataFrame of candidate pairs with columns:
            question_en, file_name, vote_id, summary, similarity_score
        """
        logger.info(
            f"SemanticMatcher: embedding {len(surveys_df)} surveys "
            f"× {len(votes_df)} votes"
        )

        # Embed both sides
        survey_texts = surveys_df[survey_text_col].fillna("").tolist()
        vote_texts = votes_df[vote_text_col].fillna("").tolist()

        survey_embs = self.embedder.embed_texts(survey_texts)
        vote_embs = self.embedder.embed_texts(vote_texts)

        # Normalize (embed_texts already normalizes, but be safe)
        survey_embs = survey_embs / (
            np.linalg.norm(survey_embs, axis=1, keepdims=True) + 1e-9
        )
        vote_embs = vote_embs / (
            np.linalg.norm(vote_embs, axis=1, keepdims=True) + 1e-9
        )

        # Batched cosine similarity + top-k extraction
        rows: list[dict] = []
        n_surveys = survey_embs.shape[0]

        for start in tqdm(
            range(0, n_surveys, self.batch_size), desc="Computing matches"
        ):
            end = min(start + self.batch_size, n_surveys)
            batch_emb = survey_embs[start:end]
            sims = batch_emb @ vote_embs.T

            for i in range(sims.shape[0]):
                survey_idx = start + i
                scores = sims[i]

                # Top-k via argpartition (O(n) instead of O(n log n))
                if self.top_k < len(scores):
                    top_indices = np.argpartition(scores, -self.top_k)[
                        -self.top_k :
                    ]
                else:
                    top_indices = np.arange(len(scores))

                top_indices = top_indices[scores[top_indices] >= self.threshold]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

                survey_row = surveys_df.iloc[survey_idx]
                for vote_idx in top_indices:
                    vote_row = votes_df.iloc[vote_idx]
                    rows.append(
                        {
                            "question_en": survey_row[survey_text_col],
                            "file_name": survey_row.get("file_name", ""),
                            "vote_id": vote_row["vote_id"],
                            "summary": str(vote_row[vote_text_col]),
                            "similarity_score": float(scores[vote_idx]),
                        }
                    )

        matches_df = pd.DataFrame(rows)
        if matches_df.empty:
            logger.warning("No matches found above threshold")
            return matches_df

        # Sort and deduplicate
        matches_df = matches_df.sort_values("similarity_score", ascending=False)
        matches_df["match_id"] = matches_df.apply(
            lambda r: hashlib.sha256(
                f"{r['question_en']}_{r['vote_id']}".encode()
            ).hexdigest(),
            axis=1,
        )
        matches_df = matches_df.drop_duplicates(subset=["match_id"], keep="first")

        logger.success(
            f"SemanticMatcher: {len(matches_df)} matches "
            f"(top_k={self.top_k}, threshold={self.threshold})"
        )
        return matches_df

    def save_matches(
        self,
        matches_df: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """Save matches DataFrame to CSV (writes header even if empty)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if matches_df.empty:
            # Write header-only CSV so read_csv won't crash
            pd.DataFrame(
                columns=[
                    "question_en", "file_name", "vote_id",
                    "summary", "similarity_score", "match_id",
                ]
            ).to_csv(output_path, index=False)
        else:
            matches_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(matches_df)} matches → {output_path}")
