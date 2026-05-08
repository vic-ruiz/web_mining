"""
Parse Página 12 HTML files → structured DataFrame.

Página 12 uses Arc Publishing CMS. Article content is embedded in the page
as a JSON blob: Fusion.globalContent = {...};
This is far more reliable than CSS selectors for this site.
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.config import DATA_RAW, INTERIM_FILE, MIN_BODY_CHARS, CATEGORIES
from src.utils.logging_utils import get_logger

log = get_logger("parser")

_FUSION_RE = re.compile(r"Fusion\.globalContent\s*=\s*(\{.+?\});", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub(" ", text).strip()


def _extract_fusion(html: str) -> Optional[dict]:
    m = _FUSION_RE.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def parse_article(html_path: Path, category: str) -> Optional[dict]:
    try:
        html = html_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    data = _extract_fusion(html)

    if data:
        return _parse_from_fusion(data, html_path, category)
    # Fallback: parse with BeautifulSoup (less reliable, but better than nothing)
    return _parse_from_bs(html, html_path, category)


def _parse_from_fusion(data: dict, path: Path, category: str) -> Optional[dict]:
    headline = data.get("headlines", {}).get("basic", "").strip()
    subhead  = data.get("subheadlines", {}).get("basic", "").strip()

    body_parts: list[str] = []
    for el in data.get("content_elements", []):
        el_type = el.get("type", "")
        if el_type in ("text", "header"):
            raw = el.get("content") or el.get("text") or ""
            cleaned = _strip_html(raw).strip()
            if cleaned:
                body_parts.append(cleaned)
        elif el_type == "list":
            for item in el.get("items", []):
                raw = item.get("content") or item.get("text") or ""
                cleaned = _strip_html(raw).strip()
                if cleaned:
                    body_parts.append(cleaned)

    body = " ".join(body_parts)

    raw_date = data.get("display_date") or data.get("first_publish_date") or ""
    date = _parse_date(raw_date)

    url = "https://www.pagina12.com.ar" + data.get("website_url", "")

    credits = data.get("credits", {}).get("by", [])
    author = credits[0].get("name", "") if credits else ""

    if len(body) < MIN_BODY_CHARS:
        log.debug("Skipping %s — body too short (%d chars)", path.name, len(body))
        return None

    return {
        "title":    headline,
        "subtitle": subhead,
        "body":     body,
        "date":     date,
        "url":      url,
        "author":   author,
        "category": category,
        "source":   str(path),
        "method":   "fusion",
    }


def _parse_from_bs(html: str, path: Path, category: str) -> Optional[dict]:
    """BeautifulSoup fallback for articles without Fusion.globalContent."""
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.find("h1")
    headline = title_tag.get_text(strip=True) if title_tag else ""

    # Collect all long text nodes
    body_parts = [
        t.strip()
        for t in soup.stripped_strings
        if len(t.strip()) > 60
    ]
    body = " ".join(body_parts[:50])  # cap to avoid nav text

    time_tag = soup.find("time")
    raw_date = time_tag.get("datetime", "") if time_tag else ""
    date = _parse_date(raw_date)

    if len(body) < MIN_BODY_CHARS:
        return None

    return {
        "title":    headline,
        "subtitle": "",
        "body":     body,
        "date":     date,
        "url":      "",
        "author":   "",
        "category": category,
        "source":   str(path),
        "method":   "bs4_fallback",
    }


def _parse_date(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    except ValueError:
        return None


def parse_all(raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    records: list[dict] = []
    seen_urls: set[str] = set()

    for category in CATEGORIES:
        section_dir = raw_dir / category
        if not section_dir.exists():
            log.warning("Directory not found: %s", section_dir)
            continue

        html_files = list(section_dir.glob("*.html"))
        log.info("[%s] Processing %d HTML files", category, len(html_files))

        ok = skipped = dup = 0
        for html_path in html_files:
            record = parse_article(html_path, category)
            if record is None:
                skipped += 1
                continue
            url = record["url"]
            if url and url in seen_urls:
                dup += 1
                continue
            if url:
                seen_urls.add(url)
            records.append(record)
            ok += 1

        log.info("[%s] OK=%d  short/invalid=%d  duplicates=%d", category, ok, skipped, dup)

    df = pd.DataFrame(records)
    if df.empty:
        log.warning("No articles parsed — check data/raw/")
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["body"]).reset_index(drop=True)

    log.info("Total articles: %d  |  date range: %s → %s",
             len(df),
             df["date"].min().date() if df["date"].notna().any() else "?",
             df["date"].max().date() if df["date"].notna().any() else "?")
    return df


def main() -> None:
    INTERIM_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = parse_all()
    if df.empty:
        log.error("Nothing to save.")
        return
    df.to_parquet(INTERIM_FILE, index=False)
    log.info("Saved → %s  (%d rows)", INTERIM_FILE, len(df))
    print("\nDistribution:")
    print(df["category"].value_counts().to_string())
    print("\nDate range by category:")
    print(df.groupby("category")["date"].agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main()
