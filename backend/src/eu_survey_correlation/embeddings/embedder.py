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
    """Embeds text using a SentenceTransformer model with a shared parquet cache.

    The cache file (``cache_path``) stores ``text`` + ``emb_0 … emb_N`` columns.
    When ``embed_dataframe`` is called, only texts missing from the cache are
    encoded; the cache is then updated with the new rows.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_path: Path | None = None,
    ):
        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache_df: pd.DataFrame | None = None

        logger.info(f"Loading SentenceTransformer model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        logger.success(
            f"Model '{model_name}' loaded (dim={self.model.get_sentence_embedding_dimension()})"
        )

        if self.cache_path and self.cache_path.exists():
            self._cache_df = pd.read_parquet(self.cache_path)
            logger.info(
                f"Loaded embedding cache: {len(self._cache_df)} rows from {self.cache_path}"
            )

    def _save_cache(self) -> None:
        if self.cache_path and self._cache_df is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_df.to_parquet(self.cache_path, index=False)

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 128,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed a list of texts into dense vectors (no caching)."""
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
        batch_size: int = 128,
    ) -> pd.DataFrame:
        """Embed a text column, using the shared cache for already-seen texts.

        Returns a DataFrame aligned with *df* containing the original columns
        plus ``emb_0, emb_1, …`` columns.  The cache file is updated with any
        newly computed embeddings.
        """
        texts = df[text_column].fillna("").astype(str)
        unique_texts = set(texts)

        # Determine which texts are already cached
        emb_cols: list[str] = []
        cached_map: dict[str, np.ndarray] = {}

        if self._cache_df is not None and "text" in self._cache_df.columns:
            emb_cols = sorted(
                [c for c in self._cache_df.columns if c.startswith("emb_")],
                key=lambda c: int(c.split("_")[1]),
            )
            cached_texts = set(self._cache_df["text"].values)
            hits = unique_texts & cached_texts
            if hits:
                hit_df = self._cache_df[self._cache_df["text"].isin(hits)]
                for _, row in hit_df.iterrows():
                    cached_map[row["text"]] = row[emb_cols].values.astype(np.float32)

        to_compute = sorted(unique_texts - set(cached_map))
        logger.info(
            f"Embedding: {len(cached_map)} cached, {len(to_compute)} to compute"
        )

        # Compute missing embeddings
        if to_compute:
            new_embs = self.embed_texts(to_compute, batch_size=batch_size)
            if not emb_cols:
                emb_cols = [f"emb_{i}" for i in range(new_embs.shape[1])]

            for text, emb in zip(to_compute, new_embs):
                cached_map[text] = emb

            # Update cache
            new_cache_rows = pd.DataFrame(
                {"text": to_compute}
                | {col: new_embs[:, i] for i, col in enumerate(emb_cols)}
            )
            if self._cache_df is not None:
                self._cache_df = pd.concat(
                    [self._cache_df, new_cache_rows], ignore_index=True
                )
            else:
                self._cache_df = new_cache_rows
            self._save_cache()

        # Build result aligned with input df
        all_embs = np.stack([cached_map[t] for t in texts])
        emb_df = pd.DataFrame(all_embs, columns=emb_cols, index=df.index)

        result_df = pd.concat(
            [df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1
        )

        logger.success(f"Result: {len(result_df)} rows × {len(emb_cols)} dims")
        return result_df
