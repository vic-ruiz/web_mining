"""
Página 12 scraper via Arc Publishing content API.

The site exposes a JSON API at:
  /pf/api/v3/content/fetch/p12-section?query={"page":N,...}&_website=pagina12

This is far more reliable than HTML scraping:
  - Proper pagination (667+ pages per section)
  - Full article body included in the response
  - No JavaScript rendering needed
  - Clean structured data
"""

import json
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.config import INTERIM_FILE, CATEGORIES
from src.utils.logging_utils import get_logger

log = get_logger("arc_scraper")

BASE_URL   = "https://www.pagina12.com.ar"
API_PATH   = "/pf/api/v3/content/fetch/p12-section"
PAGE_SIZE  = 15
HEADERS    = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "es-AR,es;q=0.9",
}

# Map from arc section path → our category label
SECTION_MAP = {
    "/economia": "economia",
    "/el-pais":  "elpais",
    "/sociedad": "sociedad",
    "/el-mundo": "elmundo",
}


def _build_url(section: str, page: int, size: int = PAGE_SIZE) -> str:
    query = json.dumps({
        "offset": 0,
        "size":   size,
        "page":   page,
        "primarySection": section,
        "arc-site": "pagina12",
    })
    params = urllib.parse.urlencode({
        "query":    query,
        "d":        "100",
        "_website": "pagina12",
    })
    return f"{BASE_URL}{API_PATH}?{params}"


def _fetch_page(
    session: requests.Session,
    section: str,
    page: int,
    retries: int = 3,
) -> Optional[dict]:
    url = _build_url(section, page)
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                log.warning("Failed %s page=%d: %s", section, page, e)
    return None


def _extract_body(content_elements: list) -> str:
    parts = []
    for el in content_elements:
        t = el.get("type", "")
        if t in ("text", "header"):
            raw = el.get("content") or el.get("text") or ""
            # Strip embedded HTML tags
            import re
            cleaned = re.sub(r"<[^>]+>", " ", raw).strip()
            if cleaned:
                parts.append(cleaned)
        elif t == "list":
            for item in el.get("items", []):
                raw = item.get("content") or item.get("text") or ""
                import re
                cleaned = re.sub(r"<[^>]+>", " ", raw).strip()
                if cleaned:
                    parts.append(cleaned)
    return " ".join(parts)


def _parse_article(el: dict, category: str) -> Optional[dict]:
    headline  = el.get("headlines", {}).get("basic", "").strip()
    subhead   = el.get("subheadlines", {}).get("basic", "") or \
                el.get("description", {}).get("basic", "")
    subhead   = subhead.strip()

    body = _extract_body(el.get("content_elements", []))
    if len(body) < 150:
        return None

    raw_date = (
        el.get("display_date")
        or el.get("first_publish_date")
        or ""
    )
    date = None
    if raw_date:
        try:
            date = datetime.fromisoformat(raw_date.replace("Z", "+00:00")) \
                          .astimezone(timezone.utc).replace(tzinfo=None)
        except ValueError:
            pass

    url = BASE_URL + el.get("website_url", "")

    credits = el.get("credits", {}).get("by", [])
    author = credits[0].get("name", "") if credits else ""

    return {
        "title":    headline,
        "subtitle": subhead,
        "body":     body,
        "date":     date,
        "url":      url,
        "author":   author,
        "category": category,
    }


def scrape_section(
    section: str,
    category: str,
    target_articles: int,
    delay: float,
    session: requests.Session,
) -> list[dict]:
    """Fetch pages spread across the full temporal range of a section."""

    # First, get total count
    data = _fetch_page(session, section, page=1)
    if not data:
        log.error("Could not fetch %s page 1", section)
        return []

    total  = data.get("count", 0)
    max_p  = max(1, total // PAGE_SIZE)
    needed = max(1, -(-target_articles // PAGE_SIZE))   # ceil division
    needed = min(needed, max_p)

    # Space pages evenly across full history for good temporal coverage
    if needed >= max_p:
        pages = list(range(1, max_p + 1))
    else:
        step = max(1, max_p // needed)
        pages = list(range(1, max_p + 1, step))[:needed]

    log.info(
        "[%s] total=%d  max_pages=%d  fetching=%d pages (step=%d)",
        category, total, max_p, len(pages), max_p // needed if needed else 1,
    )

    records: list[dict] = []
    seen_urls: set[str] = set()

    # Page 1 is already fetched
    for i, page in enumerate(pages):
        raw = data if page == 1 else _fetch_page(session, section, page)
        if raw is None:
            continue

        for el in raw.get("content_elements", []):
            url = BASE_URL + el.get("website_url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            record = _parse_article(el, category)
            if record:
                records.append(record)

        if len(records) >= target_articles:
            break

        if i < len(pages) - 1 and page != 1:
            time.sleep(delay)

    log.info("[%s] Collected %d articles", category, len(records))
    return records[:target_articles]


def scrape_all(
    target_per_class: int = 1000,
    delay: float = 1.0,
) -> pd.DataFrame:
    session = requests.Session()
    all_records: list[dict] = []

    for section, category in SECTION_MAP.items():
        records = scrape_section(
            section, category, target_per_class, delay, session
        )
        all_records.extend(records)
        log.info("[%s] Running total: %d articles", category, len(all_records))

    df = pd.DataFrame(all_records)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    log.info(
        "Scraping complete. Total: %d  |  %s → %s",
        len(df),
        df["date"].min().date() if df["date"].notna().any() else "?",
        df["date"].max().date() if df["date"].notna().any() else "?",
    )
    return df


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles-per-class", type=int, default=1000)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=INTERIM_FILE)
    args = parser.parse_args()

    df = scrape_all(
        target_per_class=args.articles_per_class,
        delay=args.delay,
    )
    if df.empty:
        log.error("No articles collected.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    log.info("Saved → %s  (%d rows)", args.out, len(df))

    print("\nDistribution:")
    print(df["category"].value_counts().to_string())
    print(f"\nDate range: {df['date'].min().date()} → {df['date'].max().date()}")


if __name__ == "__main__":
    main()
