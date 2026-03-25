# Plan: Deterministic match_id & Supabase Migration

## Context

We have two Supabase tables:
- **`survey_vote_matches`** (~1322 rows, PK = `match_id`) — candidate pairs with `admin_validated` (True/False/null)
- **`survey_answer_alignments`** (178 rows, PK = `(match_id, answer_label)`) — labelled answers (aligned, opposed, neutral, unknown, unrelated)

**Problem**: current `match_id = {question_id}_{vote_id}_{csv_row_index}` is fragile — re-generating the CSV with more rows shifts indices and breaks links to already-labelled data.

**Goal**: make `match_id` deterministic so we can add more pairs without losing any existing labels.

---

## 1. Add `procedure_reference` to the data model

### Vote hierarchy (votes.csv)
```
procedure_reference (2287 unique) → reference (2425) → vote_id/id (22920)
```
- `procedure_reference`: the legislative procedure (e.g. `2021/0200(COD)`)
- `reference`: the specific document (e.g. `A10-0003/2024`), many per procedure
- `vote_id` (`id` column): individual vote (main + amendments), many per reference
- `is_main`: flags the main vote within a reference

### ⚠️ Critical: vote_id must point to the main vote

**The problem**: summaries describe the overall procedure outcome ("Parliament adopted the regulation"), but `votes.csv` contains many rows per procedure (main vote + amendments). If we pick the wrong `vote_id`, the for/against/abstention counts won't match the summary.

**Current state**:
- `vote_summaries.csv` (old, 1611 rows): keyed directly by `vote_id`, **all 1611 are `is_main=True`** → stats are correct today
- `procedure_summaries_simplified.csv` (new, ~2213 procedures): keyed by `procedure_reference`, needs explicit mapping to main vote
- The pipeline's `dict(zip(votes['procedure_reference'], votes['id']))` picks the **last CSV row** per procedure — **376/1938 times this is NOT the main vote** (it's an amendment)

**Fix**: when mapping `procedure_reference → vote_id`, always filter to `is_main=True` first:
```python
main_votes = votes_full[votes_full['is_main'] == True]
proc_to_vote_id = dict(zip(main_votes['procedure_reference'], main_votes['id']))
proc_to_vote_date = dict(zip(main_votes['procedure_reference'], main_votes['vote_date']))
```

### What to do
- The vote summaries in `procedure_summaries_simplified.csv` are keyed by `procedure_reference` (= `reference` column in that CSV)
- The current matches CSV uses `vote_id` but it's really matching at the procedure level
- **Add `procedure_reference` to the matches output** so we have the natural key
- **Always resolve `vote_id` via `is_main=True`** to keep summary ↔ vote result consistent

---

## 2. New deterministic `match_id` format

**Stored ID**: short sha256 hash (first 16 chars) for URL-safety and compactness
```
match_id = sha256("{question_id}::{survey_file}::{procedure_reference}")[:16]
```
Example: `a3f7b2c91e04d8f1`

**Human-readable key** stored alongside in a new `match_key` column:
```
{question_id}::{survey_file}::{procedure_reference}
```
Example: `T13::ebs_509_volume_B.xlsx::2021/0200(COD)`

---

## Execution checklist

1. [x] **Create `backend/scripts/match_id_utils.py`** — pure helper functions (hash ID, readable key, vote mapping)
2. [x] **Create `tests/test_migration.py`** — 25 tests (all passing)
3. [x] **Create `backend/scripts/migrate_match_ids.py`** — migration script with dry-run/apply modes
4. [x] **Update `dawta-website/.../import_matches.py`** — remove DELETE, use new match_id, add procedure_reference, skip validated rows
5. [x] **Fix `MAIN_pipeline.ipynb` cell 9** — filter is_main=True, add procedure_reference to output
6. [x] **Update `dawta-website/.../models.py`** — add match_key + procedure_reference to SurveyVoteMatch
7. [x] **Add pytest to `pyproject.toml`** — dev dependency

### Remaining steps (manual):
8. [ ] **Run migration dry-run**: `uv run python backend/scripts/migrate_match_ids.py`
9. [ ] **Review dry-run output**, then run `--apply`
10. [ ] **Add `match_key` and `procedure_reference` columns** to Supabase table schema
11. [ ] **Re-run MAIN_pipeline.ipynb** to regenerate CSV with `procedure_reference` column
12. [ ] **Re-import** with updated `import_matches.py`
13. [ ] **Verify** frontend still loads matches and labels correctly

---

## Future work

- Expand matching dataset (4299 survey questions instead of 107)
- Train classifier from labelled data (~97 admin_validated + 178 answer alignments)
