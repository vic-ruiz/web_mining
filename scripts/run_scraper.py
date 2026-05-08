#!/usr/bin/env python3
"""
Run the Página 12 scraper.

Usage:
    python scripts/run_scraper.py
    python scripts/run_scraper.py --blocks 6 --delay 2.0
"""

import argparse
import multiprocessing
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config.config import (
    DATA_RAW,
    SECTIONS,
    SCRAPER_PAGES_PER_BLOCK,
    SCRAPER_PAGES_TO_SKIP,
    SCRAPER_NUM_BLOCKS,
    SCRAPER_DOWNLOAD_DELAY,
)
from src.utils.logging_utils import get_logger

log = get_logger("run_scraper")


def generar_paginas(pages_per_block: int, pages_to_skip: int, num_blocks: int) -> list[int]:
    paginas = []
    page = 1
    for _ in range(num_blocks):
        for offset in range(pages_per_block):
            paginas.append(page + offset)
        page += pages_per_block + pages_to_skip
    return paginas


def run_spider(base_dir: Path, num_blocks: int, delay: float) -> None:
    import scrapy
    from scrapy.crawler import CrawlerProcess
    from scrapy.http import HtmlResponse
    import re

    ARTICLE_REGEX = re.compile(r"^https://www\.pagina12\.com\.ar/20\d\d/\d{2}/\d{2}/.+")
    paginas = generar_paginas(SCRAPER_PAGES_PER_BLOCK, SCRAPER_PAGES_TO_SKIP, num_blocks)

    class Pagina12Spider(scrapy.Spider):
        name = "crawler_pagina12"
        allowed_domains = ["www.pagina12.com.ar", "pagina12.com.ar"]

        custom_settings = {
            "USER_AGENT": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "DEFAULT_REQUEST_HEADERS": {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "es-AR,es;q=0.9",
            },
            "LOG_ENABLED":              True,
            "LOG_LEVEL":                "INFO",
            "ROBOTSTXT_OBEY":           False,
            "DOWNLOAD_DELAY":           delay,
            "RANDOMIZE_DOWNLOAD_DELAY": True,
            "RETRY_TIMES":              2,
            "DEPTH_LIMIT":              0,
        }

        def start_requests(self):
            for seccion, carpeta in SECTIONS.items():
                for nro in paginas:
                    url = f"https://www.pagina12.com.ar/secciones/{seccion}?page={nro}"
                    yield scrapy.Request(url, callback=self.parse_indice,
                                        cb_kwargs={"carpeta": carpeta})

        def parse_indice(self, response: HtmlResponse, carpeta: str):
            found = 0
            for href in response.css("a::attr(href)").getall():
                url_abs = response.urljoin(href)
                if ARTICLE_REGEX.match(url_abs):
                    found += 1
                    yield scrapy.Request(url_abs, callback=self.parse_noticia,
                                         cb_kwargs={"carpeta": carpeta})
            self.logger.info("[%s] %s → %d articles", carpeta, response.url, found)

        def parse_noticia(self, response: HtmlResponse, carpeta: str):
            slug = response.url.rstrip("/").split("/")[-1]
            if not slug.endswith(".html"):
                slug += ".html"
            dest = base_dir / carpeta / slug
            dest.write_text(response.text, encoding="utf-8")
            self.logger.info("[%s] Saved: %s", carpeta, slug)

    process = CrawlerProcess()
    process.crawl(Pagina12Spider)
    process.start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Página 12 news articles")
    parser.add_argument("--blocks", type=int, default=SCRAPER_NUM_BLOCKS,
                        help=f"Number of page blocks per section (default: {SCRAPER_NUM_BLOCKS})")
    parser.add_argument("--delay", type=float, default=SCRAPER_DOWNLOAD_DELAY,
                        help=f"Download delay in seconds (default: {SCRAPER_DOWNLOAD_DELAY})")
    parser.add_argument("--outdir", type=Path, default=DATA_RAW,
                        help="Output directory for raw HTML")
    args = parser.parse_args()

    # Create section directories
    for carpeta in SECTIONS.values():
        (args.outdir / carpeta).mkdir(parents=True, exist_ok=True)

    paginas = generar_paginas(SCRAPER_PAGES_PER_BLOCK, SCRAPER_PAGES_TO_SKIP, args.blocks)
    log.info("Scraping %d sections × %d pages = %d index requests",
             len(SECTIONS), len(paginas), len(SECTIONS) * len(paginas))
    log.info("Output → %s", args.outdir)

    p = multiprocessing.Process(target=run_spider,
                                args=(args.outdir, args.blocks, args.delay))
    p.start()
    p.join()

    # Count results
    total = sum(len(list((args.outdir / c).glob("*.html"))) for c in SECTIONS.values())
    log.info("Done. Total HTML files: %d", total)
    for carpeta in SECTIONS.values():
        n = len(list((args.outdir / carpeta).glob("*.html")))
        log.info("  %-10s %d files", carpeta, n)


if __name__ == "__main__":
    main()
