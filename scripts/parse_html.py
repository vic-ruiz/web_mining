#!/usr/bin/env python3
"""
Parse raw HTML files → structured dataset.

Usage:
    python scripts/parse_html.py
    python scripts/parse_html.py --raw-dir data/raw --out data/interim/articles.parquet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.parsing.html_parser import parse_all
from src.preprocessing.text_cleaner import preprocess_dataframe
from src.config.config import DATA_RAW, INTERIM_FILE, PROCESSED_FILE
from src.utils.logging_utils import get_logger

log = get_logger("parse_html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse HTML → parquet dataset")
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW)
    parser.add_argument("--interim-out", type=Path, default=INTERIM_FILE)
    parser.add_argument("--processed-out", type=Path, default=PROCESSED_FILE)
    args = parser.parse_args()

    # Step 1: Parse HTML
    log.info("Parsing HTML from %s …", args.raw_dir)
    df = parse_all(args.raw_dir)
    if df.empty:
        log.error("No articles parsed. Did you run the scraper first?")
        sys.exit(1)

    args.interim_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.interim_out, index=False)
    log.info("Interim dataset → %s  (%d rows)", args.interim_out, len(df))

    # Step 2: Preprocess text
    log.info("Preprocessing text …")
    df_clean = preprocess_dataframe(df)

    args.processed_out.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(args.processed_out, index=False)
    log.info("Processed dataset → %s  (%d rows)", args.processed_out, len(df_clean))

    # Summary
    print("\n=== Dataset Summary ===")
    print(df_clean["category"].value_counts().to_string())
    print(f"\nDate range: {df_clean['date'].min()} → {df_clean['date'].max()}")
    print(f"Median tokens: {df_clean['n_tokens'].median():.0f}")
    print(f"Total articles: {len(df_clean)}")


if __name__ == "__main__":
    main()
