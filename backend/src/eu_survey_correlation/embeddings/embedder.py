"""Wrapper around SentenceTransformer for batch embedding with progress tracking."""

import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """Embeds text using a SentenceTransformer model and stores results as parquet."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        logger.info(f"Loading SentenceTransformer model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        logger.success(
            f"Model '{model_name}' loaded (dim={self.model.get_sentence_embedding_dimension()})"
        )

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 128,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of texts into dense vectors.

        Args:
            texts: List of strings to embed.
            batch_size: Number of texts to encode per batch.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        logger.info(f"Encoding {len(texts)} texts (batch_size={batch_size})...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        logger.success(f"Encoded {len(texts)} texts → shape {embeddings.shape}")
        return embeddings

    def embed_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        output_path: Path,
        batch_size: int = 128,
    ) -> pd.DataFrame:
        """
        Embed a text column from a DataFrame and save results to parquet.

        The output parquet contains all original columns plus one column per
        embedding dimension (emb_0, emb_1, ...).

        Args:
            df: Source DataFrame.
            text_column: Name of the column containing text to embed.
            output_path: Path to write the parquet file.
            batch_size: Batch size for encoding.

        Returns:
            DataFrame with original columns + embedding columns.
        """
        texts = df[text_column].fillna("").tolist()
        embeddings = self.embed_texts(texts, batch_size=batch_size)

        # Create embedding columns
        emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
        emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)

        result_df = pd.concat(
            [df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(output_path, index=False)

        logger.success(
            f"Saved {len(result_df)} embeddings ({embeddings.shape[1]} dims) → {output_path}"
        )
        return result_df
