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
    run_temporal_validation,
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

    # Temporal validation on best model
    best_model = max(results, key=lambda k: results[k]["macro_f1_mean"])
    log.info("Best model by macro F1: %s", best_model)

    if "tfidf" in best_model:
        temporal_res = run_temporal_validation(df, model_name=best_model)
        if temporal_res:
            import json
            temporal_path = METRICS_DIR / "temporal_validation.json"
            temporal_path.write_text(json.dumps(temporal_res, indent=2))
            log.info("Temporal validation → %s", temporal_path)
            print(f"\nTemporal Validation ({best_model}):")
            print(f"  Accuracy: {temporal_res['accuracy']:.3f}")
            print(f"  Macro F1: {temporal_res['macro_f1']:.3f}")

    print(f"\nAll models saved to: {MODELS_DIR}")
    print(f"Plots saved to:       {FIGURES_DIR}")
    print(f"Metrics saved to:     {METRICS_DIR}")


if __name__ == "__main__":
    main()
