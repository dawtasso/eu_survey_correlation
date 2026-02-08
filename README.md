# EU Survey-Vote Correlation

This project creates semantic matches between **Eurobarometer survey questions** and **European Parliament vote summaries**, enabling analysis of how citizens' opinions align with legislative outcomes.

---

## Methodology Overview

The matching pipeline consists of **3 stages**:

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  1. EMBEDDING   │ ───▶ │  2. MATCHING    │ ───▶ │ 3. VALIDATION   │
│                 │      │                 │      │                 │
│ Transform texts │      │ Cosine          │      │ LLM-based       │
│ into vectors    │      │ similarity      │      │ relevance check │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

---

## Stage 1: Embedding

Both survey questions and vote summaries are encoded into dense vector representations using a **Sentence Transformer** model.

| Property      | Value                        |
| ------------- | ---------------------------- |
| Model         | `all-MiniLM-L6-v2`         |
| Dimension     | 384                          |
| Normalization | L2-normalized (unit vectors) |

### Inputs

- **Survey questions** (`data/surveys/all_survey_questions.csv`)Contains `question_en` — the English text of each Eurobarometer question.
- **Vote summaries** (`data/votes/vote_summaries.csv`)
  Contains `summary` — legislative context for each European Parliament vote.

### Outputs

- `data/embeddings/survey_embeddings.parquet`
- `data/embeddings/vote_embeddings.parquet`

### Scripts

```bash
# Embed survey questions
python backend/scripts/embed_surveys.py

# Embed vote summaries
python backend/scripts/embed_votes.py
```

---

## Stage 2: Semantic Matching

For each survey question, we compute the **cosine similarity** with all vote summaries and retrieve the top-k most similar votes.

### Algorithm

1. Load pre-computed embeddings (already L2-normalized)
2. Compute similarity matrix: `S = survey_embeddings @ vote_embeddings.T`
3. For each survey question:
   - Extract top-k vote indices with highest similarity
   - Filter by minimum threshold (default: 0.5)
   - Store matches with metadata

### Parameters

| Parameter     | Default | Description                     |
| ------------- | ------- | ------------------------------- |
| `top_k`     | 5       | Max matches per survey question |
| `threshold` | 0.5     | Minimum cosine similarity       |

### Output

`data/matches/survey_vote_matches.csv`

| Column               | Description                          |
| -------------------- | ------------------------------------ |
| `question_id`      | Survey question identifier           |
| `question_text`    | Full English question text           |
| `file_name`        | Source Eurobarometer file            |
| `vote_id`          | European Parliament vote ID          |
| `vote_summary`     | Truncated vote summary (≤500 chars) |
| `similarity_score` | Cosine similarity [0, 1]             |

### Script

```bash
python backend/scripts/find_matches.py --top-k 5 --threshold 0.5
```

---

## Stage 3: LLM Validation

Semantic similarity alone can produce false positives (e.g., matching on generic political terms). We use a local LLM to validate each candidate pair.

### Model

| Property    | Value                                       |
| ----------- | ------------------------------------------- |
| Backend     | [Ollama](https://ollama.ai/) (local inference) |
| Model       | `mistral`                                 |
| Temperature | 0.0 (deterministic)                         |

### Evaluation Prompt

The LLM receives each (survey question, vote summary) pair and outputs:

```json
{
  "score": 1-10,
  "explanation": "one sentence reasoning",
  "go": true/false
}
```

### Scoring Rubric

| Score | Interpretation                                                        |
| ----- | --------------------------------------------------------------------- |
| 1–3  | **Unrelated** — different topics                               |
| 4–6  | **Loosely related** — same broad domain but different focus    |
| 7–10 | **Clearly related** — vote directly addresses the survey topic |

A match is marked `go: true` only if `score ≥ 7`.

### Output

`data/matches/survey_vote_matches_judged.csv`

Adds columns:

- `llm_score` — relevance score (1-10)
- `llm_explanation` — LLM reasoning
- `llm_go` — boolean: recommended for final dataset

### Script

```bash
# Validate all matches
python backend/scripts/validate_matches.py

# Validate a subset for testing
python backend/scripts/validate_matches.py --limit 50

# Resume interrupted validation
python backend/scripts/validate_matches.py --resume
```

---

## Data Flow Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                               │
├──────────────────────────────────────────────────────────────────┤
│  Eurobarometer surveys        European Parliament votes          │
│  (Excel files)                (CSV with vote_id + summary)       │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    STAGE 1: EMBEDDING                            │
├──────────────────────────────────────────────────────────────────┤
│  all_survey_questions.csv  ──▶  survey_embeddings.parquet        │
│  vote_summaries.csv        ──▶  vote_embeddings.parquet          │
│                                                                  │
│  Model: all-MiniLM-L6-v2 (384-dim sentence embeddings)           │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    STAGE 2: MATCHING                             │
├──────────────────────────────────────────────────────────────────┤
│  For each survey question, find top-k votes by cosine similarity │
│                                                                  │
│  Output: survey_vote_matches.csv                                 │
│          (~10k candidate pairs at threshold=0.5)                 │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                   STAGE 3: VALIDATION                            │
├──────────────────────────────────────────────────────────────────┤
│  LLM (Mistral via Ollama) scores each pair for thematic match    │
│                                                                  │
│  Output: survey_vote_matches_judged.csv                          │
│          (pairs with llm_go=True are validated matches)          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
eu_survey_correlation/
├── backend/
│   ├── scripts/
│   │   ├── embed_surveys.py      # Stage 1: embed survey questions
│   │   ├── embed_votes.py        # Stage 1: embed vote summaries
│   │   ├── find_matches.py       # Stage 2: cosine similarity matching
│   │   └── validate_matches.py   # Stage 3: LLM validation
│   ├── src/eu_survey_correlation/
│   │   ├── embeddings/
│   │   │   ├── embedder.py       # SentenceTransformer wrapper
│   │   │   └── matcher.py        # VoteSurveyMatcher class
│   │   └── validation/
│   │       └── llm_judge.py      # MatchJudge class (Ollama/Mistral)
│   └── notebooks/                # Exploration and review notebooks
├── data/
│   ├── surveys/
│   │   └── all_survey_questions.csv
│   ├── votes/
│   │   └── vote_summaries.csv
│   ├── embeddings/
│   │   ├── survey_embeddings.parquet
│   │   └── vote_embeddings.parquet
│   └── matches/
│       ├── survey_vote_matches.csv
│       └── survey_vote_matches_judged.csv
└── pyproject.toml
```

---

## Requirements

- Python 3.10+
- Dependencies: `sentence-transformers`, `pandas`, `numpy`, `ollama`, `loguru`, `tqdm`
- Ollama with Mistral model for validation:
  ```bash
  ollama pull mistral
  ```

---

## Quick Start

```bash
# 1. Install dependencies
uv sync  # or pip install -e .

# 2. Generate embeddings
python backend/scripts/embed_surveys.py
python backend/scripts/embed_votes.py

# 3. Find semantic matches
python backend/scripts/find_matches.py --top-k 5 --threshold 0.5

# 4. Validate with LLM (requires Ollama running)
ollama serve &
python backend/scripts/validate_matches.py
```

---

## Example Match

| Survey Question                                                                             | Vote Summary                                                                                                                                                           | Score                                              |
| ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| *"How effective do you think the EU's response would be in managing health emergencies?"* | *"Evaluation of the effectiveness of EU and national measures... Recommendations to improve EU crisis management and preparedness for future health emergencies..."* | **0.74** (similarity) / **9/10** (LLM) |

---

## Author

Dawta