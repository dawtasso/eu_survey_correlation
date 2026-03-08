"""Semantic pair matching between survey and vote embeddings."""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm


def _extract_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Extract embedding columns (emb_0, emb_1, ...) and metadata column names."""
    emb_cols = sorted(
        [c for c in df.columns if c.startswith("emb_")],
        key=lambda c: int(c.split("_")[1]),
    )
    meta_cols = [c for c in df.columns if not c.startswith("emb_")]
    return df[emb_cols].values.astype(np.float32), meta_cols


class PairMatcher:
    """Find survey questions semantically related to parliament votes.

    No date filtering is performed inside this class — dates and
    ``time_delta`` should be joined externally (e.g. in the notebook).
    """

    def __init__(
        self,
        survey_embeddings: pd.DataFrame | Path,
        vote_embeddings: pd.DataFrame | Path,
    ):
        if isinstance(survey_embeddings, (str, Path)):
            logger.info(f"Loading survey embeddings from {survey_embeddings}...")
            self.survey_df = pd.read_parquet(survey_embeddings)
        else:
            self.survey_df = survey_embeddings

        if isinstance(vote_embeddings, (str, Path)):
            logger.info(f"Loading vote embeddings from {vote_embeddings}...")
            self.vote_df = pd.read_parquet(vote_embeddings)
        else:
            self.vote_df = vote_embeddings

        self.survey_emb, self.survey_meta_cols = _extract_embeddings(self.survey_df)
        self.vote_emb, self.vote_meta_cols = _extract_embeddings(self.vote_df)

        logger.success(
            f"Loaded {len(self.survey_df)} survey and {len(self.vote_df)} vote embeddings "
            f"(dim={self.survey_emb.shape[1]})"
        )

    def _is_vote_after_survey(self, survey_row: pd.Series, vote_row: pd.Series) -> bool:
        """Return True if vote_date is strictly after survey_date."""
        if "survey_date" not in self.survey_meta_cols or "vote_date" not in self.vote_meta_cols:
            return True
        survey_date = survey_row.get("survey_date")
        vote_date = vote_row.get("vote_date")
        if survey_date is None or vote_date is None or pd.isna(survey_date) or pd.isna(vote_date):
            return False
        return vote_date > survey_date

    def match(
        self,
        top_k: int = 5,
        threshold: float = 0.5,
        batch_size: int = 256,
        vote_after_survey: bool = True,
    ) -> pd.DataFrame:
        """For each survey question, find the top-k most similar votes.

        When ``vote_after_survey=True``, only votes dated after the survey are
        considered, but the full ``top_k`` candidates are still selected from
        that filtered pool.

        Returns a DataFrame containing all survey metadata columns, all vote
        metadata columns (prefixed with ``vote_`` if collision), and a
        ``similarity_score`` column.
        """
        logger.info(
            f"Computing matches (top_k={top_k}, threshold={threshold}, "
            f"batch_size={batch_size}, vote_after_survey={vote_after_survey})..."
        )
        rows: list[dict] = []
        n_surveys = self.survey_emb.shape[0]

        for start in tqdm(range(0, n_surveys, batch_size), desc="Computing matches"):
            end = min(start + batch_size, n_surveys)
            batch_emb = self.survey_emb[start:end]

            # Cosine similarity (embeddings are already normalized)
            sims = batch_emb @ self.vote_emb.T

            for i in range(sims.shape[0]):
                survey_idx = start + i
                scores = sims[i]
                survey_row = self.survey_df.iloc[survey_idx]

                # Build candidate mask: threshold + optional date filter
                candidate_mask = scores >= threshold
                if vote_after_survey:
                    date_mask = np.array([
                        self._is_vote_after_survey(survey_row, self.vote_df.iloc[j])
                        for j in range(len(scores))
                    ])
                    candidate_mask = candidate_mask & date_mask

                candidate_indices = np.where(candidate_mask)[0]

                # Keep top_k by score from the filtered candidates
                if len(candidate_indices) > top_k:
                    top_scores = scores[candidate_indices]
                    top_indices = candidate_indices[np.argpartition(top_scores, -top_k)[-top_k:]]
                else:
                    top_indices = candidate_indices

                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

                for vote_idx in top_indices:
                    vote_row = self.vote_df.iloc[vote_idx]
                    row: dict = {}

                    # Survey metadata
                    for col in self.survey_meta_cols:
                        row[col] = survey_row[col]

                    # Vote metadata (prefix with vote_ to avoid collisions)
                    for col in self.vote_meta_cols:
                        key = f"vote_{col}" if col in self.survey_meta_cols else col
                        row[key] = vote_row[col]

                    row["similarity_score"] = float(scores[vote_idx])
                    rows.append(row)

        matches_df = pd.DataFrame(rows)
        if not matches_df.empty:
            matches_df = matches_df.sort_values("similarity_score", ascending=False)

        logger.info(f"Found {len(matches_df)} matches above threshold {threshold}")
        return matches_df
