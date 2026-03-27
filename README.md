# EU Survey-Vote Correlation

This project creates semantic matches between **European survey questions** (Eurobarometer + ESS) and **European Parliament vote summaries**, enabling analysis of how citizens' opinions align with legislative outcomes.

A trained classifier scores each pair, and an active-learning loop lets you label uncertain pairs from the terminal, push labels to Supabase, and retrain in minutes.

---

## Pipeline Overview

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 1. EMBED     │──▶│ 2. FILTER    │──▶│ 3. SIMPLIFY  │──▶│ 4. MATCH     │──▶│ 5. VALIDATE  │
│              │   │              │   │              │   │              │   │              │
│ Sentence     │   │ Remove non-  │   │ LLM rewrites │   │ Cosine sim + │   │ LLM judge    │
│ transformer  │   │ policy Qs    │   │ to readable  │   │ temporal     │   │ strict topic │
│ vectors      │   │              │   │ text         │   │ ordering     │   │ matching     │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
        │                                                                           │
        ▼                                                                           ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 6. GENERATE  │──▶│ 7. CLASSIFY  │──▶│ 8. SCORE     │──▶│ 9. LABEL     │──▶│ 10. RETRAIN  │
│              │   │              │   │              │   │              │   │              │
│ Candidate    │   │ Train LR on  │   │ Score all    │   │ Terminal     │   │ Optuna HPO + │
│ pairs from   │   │ hand-crafted │   │ unlabelled   │   │ active       │   │ new labels   │
│ full data    │   │ features     │   │ pairs        │   │ learning     │   │              │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

---

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai/) with the Mistral model (for LLM steps)
- Supabase project (for labelling sync + frontend)

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync                     # production deps
uv sync --extra dev         # + pytest, scikit-learn, xgboost, datasets

# Install and start Ollama, pull Mistral
ollama serve &
ollama pull mistral
```

### Environment variables

Create a `.env` file at the project root:

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key   # optional, for admin ops
```

---

## Makefile Commands

```bash
make help                    # show all available commands
```

### Training & scoring

| Command | Description |
|---------|-------------|
| `make retrain` | Full retrain with Optuna HPO (TRIALS=100) |
| `make retrain-quick` | Retrain reusing cached Optuna study |
| `make train` | Train classifier (legacy LR baseline) |
| `make score CSV=path` | Score a CSV file |
| `make score-unlabelled` | Score unlabelled pairs from backup |

### Candidates & Supabase

| Command | Description |
|---------|-------------|
| `make generate-candidates` | Generate candidate pairs (TOP_K=5, THRESHOLD=0.40) |
| `make add-candidates LIMIT=10` | Push active-learning candidates to Supabase |
| `make add-candidates-dry LIMIT=10` | Preview without writing |
| `make push-predictions` | Score & update all existing Supabase matches |

### Labelling

| Command | Description |
|---------|-------------|
| `make label` | Label 20 most uncertain pairs interactively |
| `make label LIMIT=50` | Label up to 50 pairs |
| `make sync` | Pull fresh Supabase backup to `data/migration/` |

### Tests

| Command | Description |
|---------|-------------|
| `make test` | Run all tests |

---

## Quick Start — Full Pipeline

Run the scripts in order. Each step reads the output of the previous one.

### Step 1: Embed raw data

Encode survey questions and vote summaries into 384-dim vectors using `all-MiniLM-L6-v2`.

```bash
uv run python backend/scripts/embed_surveys.py
uv run python backend/scripts/embed_votes.py
```

| Script | Reads | Produces |
| --- | --- | --- |
| `embed_surveys.py` | `data/surveys/all_survey_questions.csv` | `data/embeddings/survey_embeddings.parquet` |
| `embed_votes.py` | `data/votes/vote_summaries.csv` | `data/embeddings/vote_embeddings.parquet` |

### Step 2: Filter Eurobarometer questions

Two-pass filter to keep only policy-relevant questions:
1. **Heuristic** — regex patterns remove demographics, personal behavior, metadata
2. **Semantic** — keeps questions with cosine similarity >= 0.45 to at least one vote embedding

```bash
uv run python backend/scripts/filter_survey_questions.py
```

Result: 4,118 -> 2,388 (heuristic) -> 1,278 (semantic). Options: `--threshold 0.45`

### Step 3: Simplify Eurobarometer questions

Survey questions are written for interviewers, not readers. Mistral rewrites them into short, clear text.

```bash
uv run python backend/scripts/simplify_questions.py
```

**Before:** `QB4.5 On a scale of 1 to 10, how much confidence do you have in each of the following? Base: All respondents - MULTIPLE ANSWERS POSSIBLE - The European Central Bank`

**After:** `How much confidence do you have in the European Central Bank?`

Resumable — saves every 50 rows. ~25 min for 1,278 questions.

### Step 4: Extract and process ESS questions

Parse all 11 ESS rounds (2002-2023) from HTML codebooks, then filter and simplify.

```bash
uv run python backend/scripts/extract_ess_questions.py
uv run python backend/scripts/process_ess_pipeline.py
```

Result: 1,951 -> 1,556 (heuristic) -> 290 (semantic) -> 290 with clean text. Options: `--threshold 0.45`, `--skip-simplify`

### Step 5: Match surveys to votes

For each cleaned survey question, find the top-k most similar votes where the **survey was published before the vote** (temporal ordering). Also simplifies vote summaries via LLM (cached).

```bash
uv run python backend/scripts/match_surveys_to_votes.py
```

Result: 1,036 temporal matches (273 questions -> 232 votes). Options: `--top-k 3`, `--threshold 0.50`

### Step 6: Validate matches with LLM

Cosine similarity produces false positives. A strict LLM judge keeps only pairs about the **same specific policy topic**.

```bash
uv run python backend/scripts/validate_clean_matches.py
```

Result: 115 / 1,036 validated as genuinely related (11% pass rate). Resumable — saves every 25 rows.

### Step 7: Generate candidates

Generate candidate pairs from the full survey + procedure data for the classifier.

```bash
make generate-candidates                    # TOP_K=5, THRESHOLD=0.40
```

### Step 8: Train classifier

Train a logistic regression on ~10 hand-crafted features (~200 human labels). Uses Optuna for hyperparameter search and threshold calibration.

```bash
make retrain                                # full Optuna HPO
make retrain-quick                          # reuse cached study
```

Produces `data/classifier/model.joblib`, `data/classifier/threshold.json`, and `data/classifier/report.md`.

### Step 9: Score & push to Supabase

Score all pairs and push predictions to Supabase for the frontend.

```bash
make push-predictions                       # update existing matches
make add-candidates LIMIT=10                # push uncertain pairs for labelling
```

### Step 10: Label & retrain loop

Interactive terminal tool for active learning: shows the most uncertain pairs, lets you label them, and syncs to Supabase instantly.

```bash
make label                                  # label 20 most uncertain pairs
make label LIMIT=50                         # label more
make sync                                   # pull fresh backup
make retrain-quick                          # retrain with new labels
```

The labelling interface displays side-by-side panels (question vs vote summary) with model probability, similarity score, and metadata. Keys: `y`=yes, `n`=no, `s`=skip, `u`=undo, `q`=quit.

All LLM scripts are **resumable** — they save progress incrementally and can be restarted without losing work.

---

## Data Flow

```
INPUT DATA
├── data/surveys/all_survey_questions.csv          (4,118 Eurobarometer questions)
├── data/surveys/ess/ESS1..ESS11/*.html            (ESS codebooks, 11 rounds)
├── data/votes/vote_summaries.csv                  (5,581 vote summaries)
├── data/votes/votes.csv                           (vote records with timestamps)
└── data/surveys/distributions_metadata.json       (survey publication dates)

STEP 1 — EMBED
├── → data/embeddings/survey_embeddings.parquet    (4,118 × 384-dim vectors)
└── → data/embeddings/vote_embeddings.parquet      (5,581 × 384-dim vectors)

STEP 2 — FILTER EUROBAROMETER
└── → data/surveys/filtered_survey_questions.csv   (1,278 policy-relevant questions)

STEP 3 — SIMPLIFY EUROBAROMETER
└── → data/surveys/filtered_survey_questions_clean.csv  (+ question_clean column)

STEP 4 — ESS
├── → data/surveys/ess_survey_questions.csv        (1,951 extracted questions)
├── → data/surveys/ess_filtered.csv                (290 after filtering)
├── → data/surveys/ess_filtered_clean.csv          (290 + question_clean column)
└── → data/embeddings/ess_embeddings.parquet       (290 × 384-dim vectors)

STEP 5 — MATCH
├── → data/matches/survey_vote_matches_clean.csv   (1,036 temporal matches)
└── → data/votes/vote_summaries_clean_cache.json   (232 simplified vote summaries)

STEP 6 — VALIDATE
└── → data/matches/survey_vote_matches_validated.csv  (115 validated pairs)

STEP 7 — GENERATE CANDIDATES
└── → data/matches/all_candidates.csv              (full candidate pool)

STEP 8 — CLASSIFY
├── → data/classifier/model.joblib                 (trained sklearn pipeline)
├── → data/classifier/threshold.json               (calibrated threshold + config)
└── → data/classifier/report.md                    (performance report)

STEP 9 — SUPABASE
└── → survey_vote_matches table                    (predicted_probability updated)

STEP 10 — LABEL
└── → data/migration/survey_vote_matches_backup_*.json  (timestamped backups)
```

---

## Project Structure

```
eu_survey_correlation/
├── backend/
│   ├── scripts/
│   │   ├── embed_surveys.py              # Step 1: embed survey questions
│   │   ├── embed_votes.py                # Step 1: embed vote summaries
│   │   ├── filter_survey_questions.py     # Step 2: heuristic + semantic filter
│   │   ├── simplify_questions.py          # Step 3: LLM simplification
│   │   ├── extract_ess_questions.py       # Step 4: parse ESS codebooks → CSV
│   │   ├── process_ess_pipeline.py        # Step 4: ESS filter → embed → simplify
│   │   ├── match_surveys_to_votes.py      # Step 5: temporal matching
│   │   ├── validate_clean_matches.py      # Step 6: LLM validation
│   │   ├── generate_candidates.py         # Step 7: candidate pair generation
│   │   ├── train_classifier.py            # Step 8: train classifier
│   │   ├── retrain.py                     # Step 8: Optuna HPO + retrain
│   │   ├── score_matches.py               # Step 9: score pairs
│   │   ├── push_predictions_to_supabase.py # Step 9: push to Supabase
│   │   ├── label_from_terminal.py         # Step 10: interactive terminal labelling
│   │   └── match_id_utils.py              # Shared: deterministic match ID generation
│   ├── src/eu_survey_correlation/
│   │   ├── classifier/                    # Feature engineering, model training, evaluation
│   │   ├── embeddings/
│   │   │   ├── embedder.py               # SentenceTransformer wrapper
│   │   │   ├── pair_matcher.py           # VoteSurveyMatcher (cosine + temporal)
│   │   │   └── date_utils.py            # Date parsing utilities
│   │   ├── surveys/
│   │   │   ├── ess_scraper.py            # ESSCodebookParser (HTML → DataFrame)
│   │   │   ├── eurobarometer_scraper.py  # Eurobarometer scraping
│   │   │   └── volume_b_parser.py        # Volume B document parser
│   │   ├── simplifier.py                 # LLM question simplification
│   │   └── logging.py                    # Rich-based logging (print_match, tables)
│   └── notebooks/                         # Exploration notebooks
├── data/
│   ├── surveys/                           # Raw + processed survey data
│   ├── votes/                             # Vote records + summaries
│   ├── embeddings/                        # Parquet embedding files
│   ├── matches/                           # Match CSVs (candidates, validated)
│   ├── classifier/                        # Model artifacts (model.joblib, threshold.json)
│   ├── migration/                         # Supabase backup JSONs
│   └── cache/                             # Cross-encoder score caches
├── tests/                                 # pytest test suite
├── Makefile                               # All workflow commands
├── pyproject.toml                         # Project config (hatchling)
└── CLAUDE.md                              # AI coding assistant instructions
```

---

## External Dependencies

| Service | Purpose |
|---------|---------|
| **Ollama + Mistral** | Local LLM for text simplification and match validation |
| **Supabase** | Remote database for labels, predictions, and frontend display |
| **sentence-transformers** (`all-MiniLM-L6-v2`) | 384-dim embedding model |
| **Google GenAI** | Alternative LLM provider (via `google-genai` package) |

---

## Example Validated Match

| Survey (2019) | Vote (2019) |
| --- | --- |
| *How satisfied are you with the measures taken by the EU to fight terrorism?* | *Prevention of the dissemination of terrorist content online* |
| **Similarity: 0.68 — LLM: related** | |

---

## Author

Dawta
