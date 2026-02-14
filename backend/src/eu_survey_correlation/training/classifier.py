"""XGBoost-based classifier for survey-vote relatedness.

Uses embedding-derived features (cosine similarity, element-wise product,
absolute difference) to predict a continuous relatedness score (0-1).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def _extract_embeddings(df: pd.DataFrame) -> np.ndarray:
    """Extract embedding columns (emb_0, emb_1, ...) as a numpy array."""
    emb_cols = sorted(
        [c for c in df.columns if c.startswith("emb_")],
        key=lambda c: int(c.split("_")[1]),
    )
    return df[emb_cols].values.astype(np.float32)


def build_features(
    labeled_df: pd.DataFrame,
    survey_embeddings_path: Path,
    vote_embeddings_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix from labeled pairs and their embeddings.

    Features per pair (769 total):
    - cosine similarity (1)
    - element-wise product of embeddings (384)
    - absolute difference of embeddings (384)

    Returns:
        (X, y) where X is (n_pairs, 769) and y is (n_pairs,) continuous scores.
    """
    survey_df = pd.read_parquet(survey_embeddings_path)
    vote_df = pd.read_parquet(vote_embeddings_path)

    survey_embs = _extract_embeddings(survey_df)
    vote_embs = _extract_embeddings(vote_df)

    # Normalize
    survey_embs = survey_embs / (
        np.linalg.norm(survey_embs, axis=1, keepdims=True) + 1e-9
    )
    vote_embs = vote_embs / (
        np.linalg.norm(vote_embs, axis=1, keepdims=True) + 1e-9
    )

    # Build lookup indices for fast matching
    survey_text_to_idx = {}
    for i, row in survey_df.iterrows():
        survey_text_to_idx[row.get("question_en", "")] = i

    vote_id_to_idx = {}
    for i, row in vote_df.iterrows():
        vote_id_to_idx[str(row.get("vote_id", ""))] = i

    features_list = []
    scores = []
    skipped = 0

    for _, row in labeled_df.iterrows():
        s_idx = survey_text_to_idx.get(row["question_text"])
        v_idx = vote_id_to_idx.get(str(row["vote_id"]))

        if s_idx is None or v_idx is None:
            skipped += 1
            continue

        s_emb = survey_embs[s_idx]
        v_emb = vote_embs[v_idx]

        cosine_sim = np.dot(s_emb, v_emb)
        element_product = s_emb * v_emb
        abs_diff = np.abs(s_emb - v_emb)

        feature_vec = np.concatenate([[cosine_sim], element_product, abs_diff])
        features_list.append(feature_vec)
        scores.append(row["llm_score"])

    if skipped > 0:
        logger.warning(f"Skipped {skipped} pairs (embedding not found)")

    X = np.array(features_list, dtype=np.float32)
    y = np.array(scores, dtype=np.float32)
    logger.info(f"Built feature matrix: X={X.shape}, y={y.shape}")
    return X, y


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    output_path: Path | None = None,
) -> xgb.XGBRegressor:
    """
    Train an XGBRegressor with grid search over hyperparameters.

    Uses stratified k-fold based on binarized labels (score >= 0.6).
    """
    # Binarize for stratified splitting
    y_binary = (y >= 0.6).astype(int)

    param_grid = {
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.1],
    }

    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logger.info("Running grid search over hyperparameters...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    logger.success(
        f"Best params: {grid_search.best_params_}, "
        f"CV MSE: {-grid_search.best_score_:.4f}"
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        best_model.save_model(str(output_path))
        logger.success(f"Model saved → {output_path}")

    return best_model


def load_model(model_path: Path) -> xgb.XGBRegressor:
    """Load a trained XGBoost model from file."""
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    return model
