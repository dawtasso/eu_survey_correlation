---
name: LLM Match Validation
overview: Add an Ollama/Mistral-powered validation step that evaluates each survey-vote match pair with a relevance score, short explanation, and go/no-go verdict. Results are saved to an enriched CSV and explored in a notebook.
todos:
  - id: add-ollama-dep
    content: Add ollama to pyproject.toml dependencies
    status: completed
  - id: llm-judge-module
    content: Create validation/llm_judge.py with MatchJudge class using Ollama Mistral
    status: completed
    dependencies:
      - add-ollama-dep
  - id: validate-script
    content: Create validate_matches.py CLI script with --limit and --resume support
    status: completed
    dependencies:
      - llm-judge-module
  - id: review-notebook
    content: Create 5_review_judgments.ipynb for browsing/filtering LLM judgments
    status: in_progress
    dependencies:
      - validate-script
---

# LLM Match Validation with Ollama + Mistral

## Approach

Simple and direct: for each match pair, send the survey question + vote summary to Mistral running locally via Ollama. The LLM returns a structured judgment (score, explanation, go/no-go). No LangChain/LangGraph needed -- just HTTP calls to the local Ollama API via the `ollama` Python package.

## Prerequisites

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Run `ollama pull mistral` to download the model
3. Add `ollama` to project dependencies

## What gets created

```
backend/
├── src/eu_survey_correlation/
│   └── validation/
│       ├── __init__.py
│       └── llm_judge.py          # MatchJudge class: calls Ollama/Mistral
├── scripts/
│   └── validate_matches.py       # CLI: reads matches CSV, runs LLM, saves enriched CSV
└── notebooks/
    └── 5_review_judgments.ipynb   # Browse/filter LLM judgments
data/
└── matches/
    └── survey_vote_matches_judged.csv   # Original + llm_score, llm_explanation, llm_go
```

## Implementation

### 1. `llm_judge.py` -- MatchJudge class

- Uses `ollama` Python package to call local Mistral
- Prompt: given a survey question and vote summary, return JSON with `score` (1-10), `explanation` (1 sentence), `go` (boolean)
- Processes pairs in sequence with progress bar
- Handles JSON parsing errors gracefully (retry or mark as failed)
- Batches with configurable rate (pairs per minute) to avoid overloading

### 2. `validate_matches.py` -- CLI script

- Reads `data/matches/survey_vote_matches.csv`
- Runs `MatchJudge` on each pair
- Saves enriched CSV to `data/matches/survey_vote_matches_judged.csv`
- Supports `--limit N` to test on first N pairs before full run
- Supports `--resume` to skip already-judged pairs

### 3. `5_review_judgments.ipynb` -- Review notebook

- Load judged CSV
- Summary stats: distribution of scores, go/no-go ratio
- Filter and display go-only matches side-by-side
- Show worst-scored matches to verify the LLM is catching bad pairs