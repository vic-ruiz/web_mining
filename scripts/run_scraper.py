#!/usr/bin/env python3
"""
Scrape Página 12 via Arc Publishing JSON API.

Usage:
    python scripts/run_scraper.py
    python scripts/run_scraper.py --articles-per-class 1000 --delay 1.0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.scraping.arc_api_scraper import scrape_all
from src.config.config import INTERIM_FILE
from src.utils.logging_utils import get_logger

log = get_logger("run_scraper")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Página 12 via Arc API")
    parser.add_argument("--articles-per-class", type=int, default=1000,
                        help="Target number of articles per category (default: 1000)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between API requests in seconds (default: 1.0)")
    parser.add_argument("--out", type=Path, default=INTERIM_FILE,
                        help="Output parquet path")
    args = parser.parse_args()

    log.info("Target: %d articles/class × 4 classes = ~%d total",
             args.articles_per_class, args.articles_per_class * 4)
    log.info("Estimated time: ~%.0f minutes",
             args.articles_per_class / 15 * 4 * args.delay / 60)

    df = scrape_all(
        target_per_class=args.articles_per_class,
        delay=args.delay,
    )

    if df.empty:
        log.error("No articles collected.")
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    log.info("Saved → %s  (%d rows)", args.out, len(df))

    print("\n=== Scraping Summary ===")
    print(df["category"].value_counts().to_string())
    print(f"\nDate range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\nNext step: python scripts/parse_html.py")


if __name__ == "__main__":
    main()
