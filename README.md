# EU Survey-Vote Correlation

This project creates semantic matches between **European survey questions** (Eurobarometer + ESS) and **European Parliament vote summaries**, enabling analysis of how citizens' opinions align with legislative outcomes.

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
```

---

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai/) with the Mistral model (for LLM steps)

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync

# Install and start Ollama, pull Mistral
ollama serve &
ollama pull mistral
```

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
1. **Heuristic** — regex patterns remove demographics, personal behavior, metadata (age, gender, occupation, household, region, etc.)
2. **Semantic** — keeps questions with cosine similarity ≥ 0.45 to at least one vote embedding

```bash
uv run python backend/scripts/filter_survey_questions.py
```

| Reads | Produces |
| --- | --- |
| `data/surveys/all_survey_questions.csv`, `data/embeddings/survey_embeddings.parquet`, `data/embeddings/vote_embeddings.parquet` | `data/surveys/filtered_survey_questions.csv` |

Result: 4,118 → 2,388 (heuristic) → 1,278 (semantic)

Options: `--threshold 0.45`

### Step 3: Simplify Eurobarometer questions

Survey questions are written for interviewers, not readers. Mistral rewrites them into short, clear text.

```bash
uv run python backend/scripts/simplify_questions.py
```

| Reads | Produces |
| --- | --- |
| `data/surveys/filtered_survey_questions.csv` | `data/surveys/filtered_survey_questions_clean.csv` |

**Before:** `QB4.5 On a scale of 1 to 10, how much confidence do you have in each of the following? Base: All respondents - MULTIPLE ANSWERS POSSIBLE - The European Central Bank`

**After:** `How much confidence do you have in the European Central Bank?`

Resumable — saves every 50 rows. ~25 min for 1,278 questions.

### Step 4: Extract and process ESS questions

Parse all 11 ESS rounds (2002–2023) from HTML codebooks, then filter and simplify.

```bash
# Extract from codebooks
uv run python backend/scripts/extract_ess_questions.py

# Filter + embed + simplify
uv run python backend/scripts/process_ess_pipeline.py
```

| Script | Reads | Produces |
| --- | --- | --- |
| `extract_ess_questions.py` | `data/surveys/ess/ESS1/` … `data/surveys/ess/ESS11/` (HTML codebooks) | `data/surveys/ess_survey_questions.csv` |
| `process_ess_pipeline.py` | `data/surveys/ess_survey_questions.csv`, `data/embeddings/vote_embeddings.parquet` | `data/surveys/ess_filtered.csv`, `data/surveys/ess_filtered_clean.csv`, `data/embeddings/ess_embeddings.parquet` |

Result: 1,951 → 1,556 (heuristic) → 290 (semantic) → 290 with clean text

Options: `--threshold 0.45`, `--skip-simplify`

### Step 5: Match surveys to votes

For each cleaned survey question, find the top-k most similar votes where the **survey was published before the vote** (temporal ordering). Also simplifies vote summaries via LLM (cached).

```bash
uv run python backend/scripts/match_surveys_to_votes.py
```

| Reads | Produces |
| --- | --- |
| `data/surveys/filtered_survey_questions_clean.csv`, `data/embeddings/survey_embeddings.parquet`, `data/embeddings/vote_embeddings.parquet`, `data/votes/votes.csv`, `data/votes/vote_summaries.csv`, `data/surveys/distributions_metadata.json` | `data/matches/survey_vote_matches_clean.csv`, `data/votes/vote_summaries_clean_cache.json` |

Result: 1,036 temporal matches (273 questions → 232 votes)

Options: `--top-k 3`, `--threshold 0.50`

### Step 6: Validate matches with LLM

Cosine similarity produces false positives. A strict LLM judge keeps only pairs about the **same specific policy topic**.

```bash
uv run python backend/scripts/validate_clean_matches.py
```

| Reads | Produces |
| --- | --- |
| `data/matches/survey_vote_matches_clean.csv` | `data/matches/survey_vote_matches_validated.csv` |

Result: 115 / 1,036 validated as genuinely related (11% pass rate)

Resumable — saves every 25 rows.

---

## Complete command sequence

```bash
# Setup
uv sync
ollama serve &
ollama pull mistral

# 1. Embed
uv run python backend/scripts/embed_surveys.py
uv run python backend/scripts/embed_votes.py

# 2. Filter Eurobarometer
uv run python backend/scripts/filter_survey_questions.py

# 3. Simplify Eurobarometer
uv run python backend/scripts/simplify_questions.py

# 4. Extract and process ESS
uv run python backend/scripts/extract_ess_questions.py
uv run python backend/scripts/process_ess_pipeline.py

# 5. Match
uv run python backend/scripts/match_surveys_to_votes.py

# 6. Validate
uv run python backend/scripts/validate_clean_matches.py
```

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
│   │   ├── find_matches.py               # (legacy) basic cosine matching
│   │   └── validate_matches.py           # (legacy) basic LLM validation
│   ├── src/eu_survey_correlation/
│   │   ├── embeddings/
│   │   │   ├── embedder.py               # SentenceTransformer wrapper
│   │   │   └── matcher.py                # VoteSurveyMatcher class
│   │   ├── surveys/
│   │   │   └── ess_scraper.py            # ESSCodebookParser (HTML → DataFrame)
│   │   └── validation/
│   │       └── llm_judge.py              # MatchJudge class
│   └── notebooks/
├── data/
│   ├── surveys/
│   ├── votes/
│   ├── embeddings/
│   └── matches/
└── pyproject.toml
```

---

## Example Validated Match

| Survey (2019) | Vote (2019) |
| --- | --- |
| *How satisfied are you with the measures taken by the EU to fight terrorism?* | *Prevention of the dissemination of terrorist content online* |
| **Similarity: 0.68 — LLM: related** | |

---

## Author

Dawta
