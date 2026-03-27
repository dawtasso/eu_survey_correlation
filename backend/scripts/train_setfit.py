"""
Standalone SetFit training: load data → eval → train final → save.

Usage:
    make setfit                # fast single-split eval + train
    make setfit KFOLD=1        # full k-fold eval + train
"""

from __future__ import annotations

import json

from eu_survey_correlation.classifier import (
    OUTPUT_DIR,
    load_labelled_data,
    predict_setfit,
    save_setfit,
    train_setfit_eval,
    train_setfit_final,
)
from eu_survey_correlation.logging import (
    console,
    log,
    print_kv,
    print_metrics,
    print_section,
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train SetFit classifier")
    parser.add_argument("--kfold", action="store_true", help="Full k-fold CV (slow)")
    args = parser.parse_args()

    # 1. Load data
    print_section("Loading data")
    labelled = load_labelled_data()
    accepted = [r for r in labelled if r["admin_validated"] is True]
    refused = [r for r in labelled if r["admin_validated"] is False]
    print_kv("Accepted", len(accepted), "green")
    print_kv("Refused", len(refused), "red")

    # 2. Evaluate
    print_section("SetFit evaluation")
    mode = "k-fold (5x3)" if args.kfold else "single split (80/20)"
    log.info(f"Mode: {mode}")
    setfit_cv = train_setfit_eval(labelled, kfold=args.kfold)
    print_metrics(setfit_cv["cv_metrics"])
    print_kv("Threshold", f"{setfit_cv['calibrated_threshold']:.2f}")

    # 3. Train final on all data
    print_section("Training final model")
    model = train_setfit_final(labelled)
    path = save_setfit(model)
    log.info(f"Model saved to [bold]{path}[/]", extra={"markup": True})

    # 4. Save threshold
    threshold_meta = {
        "threshold": setfit_cv["calibrated_threshold"],
        "model_type": "setfit",
        "cv_metrics": setfit_cv["cv_metrics"],
    }
    with open(path / "threshold.json", "w") as f:
        json.dump(threshold_meta, f, indent=2)

    console.print("\n[bold green]Done![/]")


if __name__ == "__main__":
    main()
