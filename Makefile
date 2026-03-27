.PHONY: retrain generate-candidates add-candidates push-predictions score test

# ── Training ──────────────────────────────────────────────────────────
retrain: ## Retrain classifier with Optuna HPO (TRIALS=100)
	uv run python backend/scripts/retrain.py --n-trials $(or $(TRIALS),100)

retrain-quick: ## Retrain reusing cached Optuna study
	uv run python backend/scripts/retrain.py --skip-optuna

train: ## Train classifier (legacy LR baseline)
	uv run python backend/scripts/train_classifier.py

# ── Candidates ────────────────────────────────────────────────────────
generate-candidates: ## Generate candidate pairs from full data (TOP_K=5, THRESHOLD=0.40)
	uv run python backend/scripts/generate_candidates.py --top-k $(or $(TOP_K),5) --threshold $(or $(THRESHOLD),0.40)

add-candidates: ## Push active-learning candidates to Supabase (LIMIT=10)
	uv run python backend/scripts/push_predictions_to_supabase.py --insert-new --active-learning --limit $(or $(LIMIT),10)

add-candidates-dry: ## Preview active-learning candidates without writing
	uv run python backend/scripts/push_predictions_to_supabase.py --insert-new --active-learning --limit $(or $(LIMIT),10) --dry-run

push-predictions: ## Score & update all existing Supabase matches
	uv run python backend/scripts/push_predictions_to_supabase.py

# ── Scoring ───────────────────────────────────────────────────────────
score: ## Score a CSV file (CSV=path/to/file.csv)
	uv run python backend/scripts/score_matches.py $(CSV)

score-unlabelled: ## Score unlabelled pairs from backup
	uv run python backend/scripts/score_matches.py --unlabelled

# ── Tests ─────────────────────────────────────────────────────────────
test: ## Run all tests
	uv run pytest tests/ -v

# ── Help ──────────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
