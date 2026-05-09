#!/usr/bin/env python3
"""
Preprocess the scraped dataset (NLP cleaning).

The Arc API scraper already extracts structured text, so this step
only applies NLP preprocessing (lowercase, stopwords, stemming).

Usage:
    python scripts/parse_html.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing.text_cleaner import preprocess_dataframe
from src.config.config import INTERIM_FILE, PROCESSED_FILE
from src.utils.logging_utils import get_logger

log = get_logger("parse_html")


def main() -> None:
    parser = argparse.ArgumentParser(description="NLP preprocessing")
    parser.add_argument("--interim", type=Path, default=INTERIM_FILE)
    parser.add_argument("--out",     type=Path, default=PROCESSED_FILE)
    args = parser.parse_args()

    if not args.interim.exists():
        log.error("Interim file not found: %s — run run_scraper.py first", args.interim)
        sys.exit(1)

    df = pd.read_parquet(args.interim)
    log.info("Loaded %d articles", len(df))

    df_clean = preprocess_dataframe(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(args.out, index=False)
    log.info("Saved → %s  (%d rows)", args.out, len(df_clean))

    print("\n=== Dataset Summary ===")
    print(df_clean["category"].value_counts().to_string())
    print(f"\nDate range:    {df_clean['date'].min().date()} → {df_clean['date'].max().date()}")
    print(f"Median tokens: {df_clean['n_tokens'].median():.0f}")
    print(f"\nNext step: python scripts/train.py")


if __name__ == "__main__":
    main()
