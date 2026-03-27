# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EU Survey-Vote Correlation — semantic matching pipeline between **European survey questions** (Eurobarometer + ESS) and **European Parliament vote summaries**. Determines how citizens' opinions align with legislative outcomes.

## Commands

```bash
# Install dependencies
uv sync                          # production deps
uv sync --extra dev              # + pytest, scikit-learn, xgboost, datasets

# Run any script
uv run python backend/scripts/<script>.py

# Tests
uv run pytest                    # all tests
uv run pytest tests/test_classifier.py -v   # single test file

# LLM dependency (required for simplify/validate steps)
ollama serve &
ollama pull mistral
```

## Architecture

### Pipeline (sequential, each step reads previous output)

1. **Embed** — `embed_surveys.py`, `embed_votes.py` → 384-dim vectors via `all-MiniLM-L6-v2`
2. **Filter** — `filter_survey_questions.py` → heuristic regex + semantic cosine threshold (≥0.45)
3. **Simplify** — `simplify_questions.py` → Mistral rewrites interviewer-style questions to readable text
4. **ESS** — `extract_ess_questions.py` + `process_ess_pipeline.py` → parse HTML codebooks, filter, embed, simplify
5. **Match** — `match_surveys_to_votes.py` → cosine similarity + temporal ordering (survey before vote)
6. **Validate** — `validate_clean_matches.py` → LLM judge filters false positives (~11% pass rate)
7. **Classify** — `train_classifier.py` → logistic regression on 10 hand-crafted features (~200 human labels)
8. **Score** — `score_matches.py` → apply trained model to unlabelled pairs

LLM steps (simplify, validate) are **resumable** — they save progress incrementally.

### Source code layout

- `backend/scripts/` — runnable pipeline scripts (entry points)
- `backend/src/eu_survey_correlation/` — importable library:
  - `embeddings/embedder.py` — SentenceTransformer wrapper
  - `embeddings/pair_matcher.py` — VoteSurveyMatcher (cosine + temporal matching)
  - `embeddings/date_utils.py` — date parsing utilities
  - `surveys/ess_scraper.py` — ESSCodebookParser (HTML → DataFrame)
  - `surveys/eurobarometer_scraper.py` — Eurobarometer scraping
  - `surveys/volume_b_parser.py` — Volume B document parser
  - `simplifier.py` — LLM question simplification
- `backend/notebooks/` — exploration notebooks (numbered pipeline steps + MAIN_pipeline.ipynb)
- `data/` — all intermediate and output data (surveys, votes, embeddings, matches, classifier artifacts)

### Key data files

- `data/surveys/all_survey_questions.csv` — raw Eurobarometer questions
- `data/votes/vote_summaries.csv` — EP vote summaries
- `data/matches/survey_vote_matches_validated.csv` — final validated pairs
- `data/classifier/model.joblib` — trained sklearn pipeline
- `data/classifier/threshold.json` — calibrated decision threshold

### External dependencies

- **Ollama + Mistral** — local LLM for text simplification and match validation
- **Supabase** — remote database (push predictions via `push_predictions_to_supabase.py`)
- **sentence-transformers** (`all-MiniLM-L6-v2`) — embedding model
- **Google GenAI** — alternative LLM provider (via `google-genai` package)

## Conventions

- Package manager: `uv` (always use `uv run` to execute scripts)
- Python ≥ 3.12
- Logging: `loguru`
- Data format: CSV for tabular data, Parquet for embeddings, JSON for caches/configs
- Build system: hatchling (package path: `backend/src/eu_survey_correlation`)
