#!/usr/bin/env python3
"""
Train all classifiers and run cross-validation.

Usage:
    python scripts/train.py
    python scripts/train.py --skip-embeddings
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config.config import PROCESSED_FILE, INTERIM_FILE, MODELS_DIR, FIGURES_DIR, METRICS_DIR
from src.training.trainer import train_all
from src.evaluation.evaluator import (
    run_temporal_validation_multi,
    cv_confusion_matrix,
    plot_class_distribution,
    plot_articles_over_time,
    plot_token_length_distribution,
    plot_model_comparison,
)
from src.utils.logging_utils import get_logger

log = get_logger("train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train news classifiers")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip sentence-transformers models (faster)")
    parser.add_argument("--data", type=Path, default=PROCESSED_FILE)
    args = parser.parse_args()

    if not args.data.exists():
        log.error("Dataset not found: %s — run parse_html.py first", args.data)
        sys.exit(1)

    df = pd.read_parquet(args.data)
    log.info("Loaded %d articles from %s", len(df), args.data)

    if len(df) < 50:
        log.error("Dataset too small (%d articles). Scrape more data first.", len(df))
        sys.exit(1)

    # EDA plots
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(df)
    plot_articles_over_time(df)
    plot_token_length_distribution(df)

    # Train all models
    results = train_all(df)

    # Plots
    plot_model_comparison(results)

    # Best model by macro F1 (prefer a tfidf/bow pipeline that takes raw text,
    # so confusion-matrix CV and temporal refit work on the text column directly)
    text_models = {k: v for k, v in results.items()
                   if any(t in k for t in ("tfidf", "bow", "ensemble"))}
    ranking = text_models or results
    best_model = max(ranking, key=lambda k: ranking[k]["macro_f1_mean"])
    log.info("Best model by macro F1: %s", best_model)

    import joblib
    import json

    # CV confusion matrix of the best model (answers: which classes confuse?)
    best_path = MODELS_DIR / f"{best_model}.joblib"
    if best_path.exists():
        from src.config.config import CATEGORY_DISPLAY
        le = joblib.load(MODELS_DIR / "label_encoder.joblib")
        labels = [CATEGORY_DISPLAY.get(c, c) for c in le.classes_]
        model = joblib.load(best_path)
        X_text = df["body"].tolist()
        y = le.transform(df["category"])
        cv_confusion_matrix(model, X_text, y, labels, model_name=best_model)

    # Temporal validation with MULTIPLE cuts (3, 6, 9 months)
    temporal_res = run_temporal_validation_multi(df, model_name=best_model)
    if temporal_res:
        temporal_path = METRICS_DIR / "temporal_validation.json"
        temporal_path.write_text(json.dumps(temporal_res, indent=2))
        log.info("Temporal validation (multi-cut) → %s", temporal_path)
        print(f"\nValidación temporal — {best_model} (varios cortes):")
        for tag, c in temporal_res.get("cuts", {}).items():
            print(f"  {tag:>4} (corte {c['cutoff_date']}, test={c['test_size']}): "
                  f"acc={c['accuracy']:.3f}  macroF1={c['macro_f1']:.3f}")
        s = temporal_res.get("summary", {})
        if s:
            print(f"  Promedio: acc={s['accuracy_mean']:.3f}±{s['accuracy_std']:.3f}  "
                  f"macroF1={s['macro_f1_mean']:.3f}±{s['macro_f1_std']:.3f}")

    print(f"\nAll models saved to: {MODELS_DIR}")
    print(f"Plots saved to:       {FIGURES_DIR}")
    print(f"Metrics saved to:     {METRICS_DIR}")


if __name__ == "__main__":
    main()
