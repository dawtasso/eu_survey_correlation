"""Compute cosine similarity between survey and vote embeddings and extract top-k matches."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm


def _extract_embeddings(df: pd.DataFrame) -> np.ndarray:
    """Extract embedding columns (emb_0, emb_1, ...) as a numpy array."""
    emb_cols = sorted(
        [c for c in df.columns if c.startswith("emb_")],
        key=lambda c: int(c.split("_")[1]),
    )
    return df[emb_cols].values.astype(np.float32)


class VoteSurveyMatcher:
    """Find survey questions semantically related to parliament votes."""

    def __init__(
        self,
        survey_embeddings_path: Path,
        vote_embeddings_path: Path,
    ):
        logger.info(f"Loading survey embeddings from {survey_embeddings_path}...")
        self.survey_df = pd.read_parquet(survey_embeddings_path)
        logger.info(f"Loading vote embeddings from {vote_embeddings_path}...")
        self.vote_df = pd.read_parquet(vote_embeddings_path)

        self.survey_emb = _extract_embeddings(self.survey_df)
        self.vote_emb = _extract_embeddings(self.vote_df)

        # Normalize for cosine similarity (dot product on unit vectors)
        self.survey_emb = self.survey_emb / (
            np.linalg.norm(self.survey_emb, axis=1, keepdims=True) + 1e-9
        )
        self.vote_emb = self.vote_emb / (
            np.linalg.norm(self.vote_emb, axis=1, keepdims=True) + 1e-9
        )

        logger.success(
            f"Loaded {len(self.survey_df)} survey embeddings "
            f"and {len(self.vote_df)} vote embeddings "
            f"(dim={self.survey_emb.shape[1]})"
        )

    def compute_matches(
        self,
        top_k: int = 5,
        threshold: float = 0.5,
        batch_size: int = 256,
    ) -> pd.DataFrame:
        """
        For each survey question, find the top-k most similar vote summaries.

        Computes similarities in batches to avoid memory issues.

        Args:
            top_k: Number of best matches to keep per survey question.
            threshold: Minimum cosine similarity to include a match.
            batch_size: Number of survey questions to process at once.

        Returns:
            DataFrame with columns: question_id, question_text, file_name,
            vote_id, vote_summary, similarity_score.
        """
        logger.info(
            f"Computing matches (top_k={top_k}, threshold={threshold}, "
            f"batch_size={batch_size})..."
        )
        rows = []
        n_surveys = self.survey_emb.shape[0]

        for start in tqdm(range(0, n_surveys, batch_size), desc="Computing matches"):
            end = min(start + batch_size, n_surveys)
            batch_emb = self.survey_emb[start:end]

            # Cosine similarity: (batch, votes)
            sims = batch_emb @ self.vote_emb.T

            for i in range(sims.shape[0]):
                survey_idx = start + i
                scores = sims[i]

                # Get top-k indices
                if top_k < len(scores):
                    top_indices = np.argpartition(scores, -top_k)[-top_k:]
                else:
                    top_indices = np.arange(len(scores))

                top_indices = top_indices[scores[top_indices] >= threshold]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

                for vote_idx in top_indices:
                    survey_row = self.survey_df.iloc[survey_idx]
                    vote_row = self.vote_df.iloc[vote_idx]

                    rows.append(
                        {
                            "question_id": survey_row.get("sheet_id", ""),
                            "question_text": survey_row.get("question_en", ""),
                            "file_name": survey_row.get("file_name", ""),
                            "vote_id": vote_row.get("vote_id", ""),
                            "vote_summary": str(vote_row.get("summary", "")),
                            "similarity_score": float(scores[vote_idx]),
                        }
                    )

        matches_df = pd.DataFrame(rows)
        if not matches_df.empty:
            matches_df = matches_df.sort_values("similarity_score", ascending=False)

        matches_df = matches_df.rename(columns={"question_id": "question_index"})
        matches_df["question_id"] = matches_df["question_text"].apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()
        )
        matches_df["match_id"] = (
            matches_df["question_id"] + "_" + matches_df["vote_id"].astype(str)
        )
        # Drop duplicate matches (same question_text + vote_id pair)
        matches_df = matches_df.drop_duplicates(subset=["match_id"], keep="first")
        logger.info(f"Found {len(matches_df)} matches above threshold {threshold}")
        return matches_df

    def save_matches(
        self,
        output_path: Path,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Compute matches and save to CSV."""
        matches_df = self.compute_matches(top_k=top_k, threshold=threshold)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        matches_df.to_csv(output_path, index=False)
        logger.success(
            f"Saved {len(matches_df)} matches "
            f"({matches_df['question_id'].nunique()} questions matched) "
            f"→ {output_path}"
        )
        return matches_df
