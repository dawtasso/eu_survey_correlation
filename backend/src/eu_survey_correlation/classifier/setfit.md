Plan: Add SetFit deep learning classifier                                                                                                                                                  │
│                                                                                                                                                                                            │
│ Context                                                                                                                                                                                    │
│                                                                                                                                                                                            │
│ Current classifier is LR on 10 hand-crafted features (~200 labels). Add SetFit as alternative: fine-tunes all-MiniLM-L6-v2 (already used) with contrastive learning, then logistic head.   │
│ Designed for few-shot.                                                                                                                                                                     │
│                                                                                                                                                                                            │
│ Files                                                                                                                                                                                      │
│                                                                                                                                                                                            │
│ ┌──────────────────────────────────────────────────────────────┬───────────────────────────────────────────┐                                                                               │
│ │                             File                             │                  Action                   │                                                                               │
│ ├──────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                               │
│ │ pyproject.toml                                               │ Add setfit to dev deps                    │                                                                               │
│ ├──────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                               │
│ │ backend/src/eu_survey_correlation/classifier/setfit_model.py │ NEW — train, evaluate, predict, save/load │                                                                               │
│ ├──────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                               │
│ │ backend/src/eu_survey_correlation/classifier/__init__.py     │ Export new functions                      │                                                                               │
│ ├──────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                               │
│ │ backend/scripts/retrain.py                                   │ Add --setfit flag                         │                                                                               │
│ ├──────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                               │
│ │ backend/scripts/push_predictions_to_supabase.py              │ Support setfit in score_records()         │                                                                               │
│ ├──────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                               │
│ │ backend/scripts/label_from_terminal.py                       │ Support setfit in score_pairs()           │                                                                               │
│ ├──────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                               │
│ │ backend/scripts/score_matches.py                             │ Support setfit in scoring                 │                                                                               │
│ ├──────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤                                                                               │
│ │ Makefile                                                     │ Add retrain-setfit target                 │                                                                               │
│ └──────────────────────────────────────────────────────────────┴───────────────────────────────────────────┘                                                                               │
│                                                                                                                                                                                            │
│ New module: classifier/setfit_model.py                                                                                                                                                     │
│                                                                                                                                                                                            │
│ Key functions:                                                                                                                                                                             │
│ - train_setfit_cv(labelled, n_splits=5, n_repeats=3) → CV metrics + calibrated threshold                                                                                                   │
│ - train_setfit_final(labelled) → trained SetFitModel                                                                                                                                       │
│ - predict_setfit(model, records) → P(accept) array                                                                                                                                         │
│ - save_setfit(model, path) / load_setfit(path) → persistence                                                                                                                               │
│                                                                                                                                                                                            │
│ Input text: f"{question_clean} [SEP] {vote_summary_clean}" (SetFit takes single text).                                                                                                     │
│ Base model: sentence-transformers/all-MiniLM-L6-v2.                                                                                                                                        │
│ Training: num_epochs=3, batch_size=16, num_iterations=20.                                                                                                                                  │
│ CV: RepeatedStratifiedKFold(5, 3) — lighter than LR since training is slower.                                                                                                              │
│ Threshold: reuse find_best_threshold().                                                                                                                                                    │
│ Save dir: OUTPUT_DIR / "setfit_model/".                                                                                                                                                    │
│                                                                                                                                                                                            │
│ Integration                                                                                                                                                                                │
│                                                                                                                                                                                            │
│ - retrain.py --setfit: trains SetFit after Optuna LR, saves model + threshold.json with model_type: "setfit", report compares both                                                         │
│ - Scoring scripts: check model_type in threshold.json → if "setfit", load from setfit_model/ dir and use predict_setfit()                                                                  │
│ - Existing LR flow unchanged when model_type != "setfit"                                                                                                                                   │
│                                                                                                                                                                                            │
│ Makefile                                                                                                                                                                                   │
│                                                                                                                                                                                            │
│ retrain-setfit: ## Train SetFit classifier                                                                                                                                                 │
│       uv run python backend/scripts/retrain.py --setfit --skip-optuna                                                                                                                      │
│                                                                                                                                                                                            │
│ Verification                                                                                                                                                                               │
│                                                                                                                                                                                            │
│ uv sync --extra dev                                                                                                                                                                        │
│ make retrain-setfit                                                                                                                                                                        │
│ make score-unlabelled                                                                                                                                                                      │
│ make label LIMIT=3