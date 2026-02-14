# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EU Survey-Vote Correlation: a semantic matching pipeline that correlates Eurobarometer survey questions (~4,100) with European Parliament vote summaries (~5,600) using NLP embeddings and LLM validation.

## Commands

```bash
# Install dependencies
uv sync

# Run the 3-stage pipeline:
# 1. Generate embeddings
python backend/scripts/embed_surveys.py
python backend/scripts/embed_votes.py

# 2. Find semantic matches (cosine similarity)
python backend/scripts/find_matches.py --top-k 5 --threshold 0.5

# 3. LLM validation via local Ollama
ollama serve &                                          # must be running
python backend/scripts/validate_matches.py              # full run
python backend/scripts/validate_matches.py --limit 50   # test subset
python backend/scripts/validate_matches.py --resume     # resume interrupted

# 4. Export validated matches to JSON for website
python backend/scripts/export_matches_json.py

# 5. Train relatedness classifier (requires labeled data)
python backend/scripts/generate_training_data.py --n-positive 250 --n-negative 250 --passes 3 --resume
python backend/scripts/train_classifier.py

# Web scraping (data collection)
python backend/scripts/run_eurobarometer_scraper.py --start 1 --end 5000 --workers 4
```

No test suite or linter is configured.

## Architecture

### Three-Stage Pipeline

```
Surveys CSV + Votes CSV
        │
   ┌────▼────┐
   │ EMBED   │  all-MiniLM-L6-v2 (384-dim, L2-normalized)
   │         │  → parquet files with emb_0..emb_383 columns
   └────┬────┘
   ┌────▼────┐
   │ MATCH   │  Batched cosine similarity (dot product on normalized vecs)
   │         │  top_k=5, threshold=0.5, SHA-256 dedup
   │         │  → 9,732 candidate pairs CSV
   └────┬────┘
   ┌────▼────┐
   │ VALIDATE│  Mistral via local Ollama, score 1-10 (go if ≥7)
   │         │  Incremental save + resume support
   │         │  → judged pairs CSV → matches.json for website
   └─────────┘
```

### Source Layout

- `backend/src/eu_survey_correlation/` — Python package
  - `embeddings/embedder.py` — SentenceTransformer wrapper with L2 normalization and parquet output
  - `embeddings/matcher.py` — Batched cosine similarity matching with top-k extraction (`np.argpartition`)
  - `validation/llm_judge.py` — Ollama-based LLM scoring with structured prompt, JSON parsing with regex fallback
  - `training/labeler.py` — Strict LLM labeling with multi-pass validation and negative sampling
  - `training/classifier.py` — XGBoost regressor on embedding-derived features (769-dim)
  - `training/evaluator.py` — Regression + classification metrics and reporting
  - `surveys/eurobarometer_scraper.py` — Selenium parallel scraper (ThreadPoolExecutor, per-worker Chrome driver)
- `backend/scripts/` — CLI entry points for each pipeline stage
- `backend/notebooks/` — Exploratory notebooks (numbered 1-5, following the pipeline stages)
- `data/` — All pipeline artifacts: surveys/, votes/, embeddings/ (parquet), matches/ (CSV + JSON), training/ (labeled data + model)

### Key Design Choices

- **Embeddings stored as parquet columns** (emb_0..emb_383) alongside original DataFrame columns
- **Memory-efficient batching**: 256 surveys per batch in matcher to avoid OOM
- **Deduplication**: SHA-256 hash of `question_text + vote_id` for match_id
- **Incremental persistence**: Both LLM validation and web scraper save after each item and support resume

### Deployment

GitHub Actions workflow (`sync-matches.yml`) auto-syncs `survey_vote_matches_judged.csv` to a website repo on push. Requires `WEBSITE_REPO` and `WEBSITE_REPO_PAT` secrets.

## Dependencies

Python 3.12+, managed with `uv`. Key libraries: sentence-transformers, pandas, numpy, pyarrow, ollama, xgboost, scikit-learn, beautifulsoup4, selenium, google-genai, loguru, rich.
