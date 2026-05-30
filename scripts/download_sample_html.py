#!/usr/bin/env python3
"""
Download a small, *correctly labelled* sample of raw article HTML per category.

The assignment requires delivering the HTML pages used (or, if too large, "a set
with some examples of each section is enough"). The Arc Publishing scraper works
with the JSON API and never stores HTML, and the leftover files under data/raw/
came from the old broken Scrapy crawler and were mislabelled. This script builds
a clean deliverable: it takes the URLs actually scraped per category and saves
the real article HTML into muestra_html/<categoria>/<slug>.html (UTF-8).

Usage:
    python scripts/download_sample_html.py --per-class 12
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config.config import INTERIM_FILE, ROOT, CATEGORY_DISPLAY
from src.utils.logging_utils import get_logger

log = get_logger("sample_html")

OUT_DIR = ROOT / "muestra_html"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "es-AR,es;q=0.9",
}


def _slug(url: str) -> str:
    tail = url.rstrip("/").split("/")[-1]
    return (tail or "articulo")[:120]


def pick_spread(group: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pick n rows evenly spread across the date range for temporal diversity."""
    g = group.dropna(subset=["url"]).sort_values("date")
    if len(g) <= n:
        return g
    idx = [round(i * (len(g) - 1) / (n - 1)) for i in range(n)]
    return g.iloc[sorted(set(idx))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--interim", type=Path, default=INTERIM_FILE)
    args = parser.parse_args()

    df = pd.read_parquet(args.interim)
    session = requests.Session()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for category, group in df.groupby("category"):
        cat_dir = OUT_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        sample = pick_spread(group, args.per_class)
        log.info("[%s] downloading %d HTML examples", category, len(sample))

        for _, row in sample.iterrows():
            url = row["url"]
            try:
                r = session.get(url, headers=HEADERS, timeout=20)
                r.raise_for_status()
                r.encoding = "utf-8"
                fname = cat_dir / f"{_slug(url)}.html"
                fname.write_text(r.text, encoding="utf-8")
                total += 1
                time.sleep(args.delay)
            except Exception as e:
                log.warning("  failed %s: %s", url, e)

    print(f"\nSaved {total} HTML examples to {OUT_DIR}")
    for category in sorted(df["category"].unique()):
        d = OUT_DIR / category
        n = len(list(d.glob("*.html"))) if d.exists() else 0
        print(f"  {CATEGORY_DISPLAY.get(category, category):<10} {n} archivos")


if __name__ == "__main__":
    main()
