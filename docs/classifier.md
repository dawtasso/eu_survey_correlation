# Match Quality Classifier

Logistic regression classifier that predicts whether a survey question / EP vote pair is a genuine thematic match or a false positive from the similarity search.

## Overview

The EU Survey Correlation pipeline finds candidate matches between Eurobarometer survey questions and European Parliament votes using embedding similarity. Many of these candidates are false positives — the classifier learns to separate real matches from noise using 10 hand-crafted features and ~200 human labels.

**When to use it:**
- After the similarity search produces candidate pairs
- To score unlabelled pairs (`score_matches.py`)
- To prioritize which pairs a human should review next (active learning)

## Architecture

```
Raw Pair (question, vote)
        │
        ▼
┌───────────────────┐
│  compute_features  │  10 features: similarity, time, text length,
│                    │  word overlap, embeddings
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  StandardScaler    │  zero-mean, unit-variance normalization
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ LogisticRegression │  class_weight="balanced", regularized (C tuned)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  P(accepted)       │  calibrated probability
│  ≥ threshold?      │  threshold chosen to maximize F1 on CV folds
└────────┬──────────┘
         │
    ┌────┴────┐
    ▼         ▼
 ACCEPT    REFUSE
```

See also: `data/classifier/figures/pipeline_architecture.png` (generated).

## Features

| # | Feature | Description | Source |
|---|---------|-------------|--------|
| 1 | `similarity_score` | Cosine similarity from initial embedding search | Matching pipeline |
| 2 | `days_between` | Absolute time gap between survey fieldwork and vote date | Metadata |
| 3 | `len_question` | Character length of the cleaned survey question | Text |
| 4 | `len_vote_summary` | Character length of the cleaned vote summary | Text |
| 5 | `len_ratio` | `len_question / len_vote_summary` | Text |
| 6 | `question_has_number` | Whether the question contains a digit (0/1) | Text |
| 7 | `word_overlap` | Jaccard similarity of word sets | Text |
| 8 | `emb_cosine` | Cosine similarity of cached embeddings (falls back to `similarity_score` if missing) | Embeddings |
| 9 | `emb_abs_diff_mean` | Mean of element-wise absolute embedding difference | Embeddings |
| 10 | `emb_abs_diff_std` | Std of element-wise absolute embedding difference | Embeddings |

Feature importance rankings are shown in `data/classifier/figures/feature_importances.png`.

## Model

**Why Logistic Regression?**
- Interpretable coefficients (each feature has a signed weight)
- Well-calibrated probabilities out of the box
- Fast to train, no GPU needed
- Works well with small datasets (~200 labels) — avoids overfitting
- `class_weight="balanced"` upweights the minority class (accepted) automatically

**Key components:**
- `StandardScaler` — normalizes features to zero mean, unit variance. Important because LR is sensitive to feature scale (e.g., `days_between` can be ~1000 while `word_overlap` is 0–1).
- `LogisticRegression(C=..., class_weight="balanced")` — L2-regularized logistic regression. Lower C = stronger regularization.

## Training

**Cross-validation strategy:**
- Outer CV: `RepeatedStratifiedKFold(n_splits=5, n_repeats=10)` — 50 folds total, stratified to preserve class balance
- Inner CV: `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)` — used to select the best `C` value
- Per-fold threshold calibration: for each outer fold, the threshold that maximizes F1 is found by sweeping 0.10–0.90
- Final threshold: median of all 50 per-fold thresholds

**Class imbalance handling:**
- `class_weight="balanced"` adjusts loss weights inversely proportional to class frequency
- Threshold calibration further adapts the decision boundary

## Parameters to Tune

| Parameter | Current | What it controls | Suggested range | Notes |
|-----------|---------|-----------------|----------------|-------|
| `C` | Auto-tuned (inner CV) | Regularization strength (inverse). Lower = more regularization | `[0.01, 0.1, 1, 10, 100]` | Already tuned automatically |
| `class_weight` | `"balanced"` | How much to upweight minority class | `"balanced"` or `{0: 1, 1: w}` | Try manual weights if recall is too low |
| `max_iter` | `1000` | Max optimization iterations | Usually fine | Increase if convergence warning appears |
| `n_splits` (outer) | `5` | CV fold count | 3–10 | More = better estimate, slower |
| `n_repeats` (outer) | `10` | Times to repeat CV | 5–20 | More = more stable estimates |
| Threshold | Median of per-fold best | Decision boundary on P(accepted) | 0.1–0.9 | See threshold sensitivity plot |
| Similarity search `k` | (upstream) | Number of candidates per question | 5–20 | More candidates = more noise for classifier |

The **threshold sensitivity plot** (`data/classifier/figures/threshold_sensitivity.png`) shows how F1, precision, and recall change as you move the threshold — use it to pick a threshold that matches your precision/recall preference.

## Performance

Current results are in `data/classifier/report.md`. Key figures:

| Figure | Path | Shows |
|--------|------|-------|
| Score distributions | `figures/score_distributions.png` | How well similarity alone separates classes |
| Feature importances | `figures/feature_importances.png` | Which features drive predictions |
| Threshold sensitivity | `figures/threshold_sensitivity.png` | F1/precision/recall vs. threshold |
| PR curve | `figures/precision_recall_curve.png` | Precision-recall tradeoff with operating point |
| Confusion matrix | `figures/confusion_matrix.png` | Counts + row-normalized percentages |
| Calibration | `figures/calibration_plot.png` | How well predicted probabilities match reality |
| Pipeline | `figures/pipeline_architecture.png` | Data flow diagram |
| Active learning | `figures/active_learning_candidates.png` | Unlabelled pairs near decision boundary |

All figures are in `data/classifier/figures/`.

## Usage

### Train the classifier

```bash
uv run python backend/scripts/train_classifier.py
```

**Outputs** (all in `data/classifier/`):
- `model.joblib` — trained sklearn pipeline
- `metrics.json` — CV metrics
- `feature_importances.json` — coefficient values
- `threshold.json` — calibrated threshold and best C
- `report.md` — auto-generated report with embedded figures
- `figures/` — all PNG plots

### Score unlabelled matches

```bash
uv run python backend/scripts/score_matches.py
```

Uses the saved model + threshold to predict on all unlabelled pairs.

### Run tests

```bash
uv run pytest tests/test_classifier.py -v
```

## Improvement Ideas

1. **More labels** — active learning: label the 20 pairs closest to the decision boundary (listed in `report.md`)
2. **More features** — LLM-based semantic relatedness score, shared named entities, topic/committee overlap
3. **SetFit** — few-shot fine-tuning of sentence-transformers; works well with <500 labels
4. **XGBoost/LightGBM** — gradient boosting may capture non-linear feature interactions
5. **Hard negative mining** — ensure training set includes difficult near-misses, not just random negatives
6. **Ensemble** — combine LR + SetFit predictions for robustness
