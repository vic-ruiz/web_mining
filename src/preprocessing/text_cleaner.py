"""
NLP preprocessing pipeline for Spanish news text.

Configurable steps:
  - HTML tag stripping
  - Unicode normalization
  - Lowercasing
  - Accent removal (optional)
  - Punctuation & number removal
  - Stopword removal
  - Stemming (Snowball / Spanish)
"""

import re
import sys
import unicodedata
from pathlib import Path
from typing import Callable

import pandas as pd
from nltk.stem.snowball import SnowballStemmer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.config import STOPWORDS_PATH, PROCESSED_FILE, INTERIM_FILE, TEXT_COL
from src.utils.logging_utils import get_logger

log = get_logger("preprocessor")

_stemmer = SnowballStemmer("spanish")
_html_re = re.compile(r"<[^>]+>")
_nonalpha_re = re.compile(r"[^a-záéíóúüñ\s]", re.UNICODE)
_spaces_re = re.compile(r"\s+")


def _load_stopwords(path: Path = STOPWORDS_PATH) -> set[str]:
    if not path.exists():
        log.warning("Stopwords file not found at %s — using empty set", path)
        return set()
    with path.open(encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


_STOPWORDS: set[str] = _load_stopwords()


def strip_html(text: str) -> str:
    return _html_re.sub(" ", text)


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def remove_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_html_tags: bool = True,
    remove_punct: bool = True,
    drop_stopwords: bool = True,
    stem: bool = False,
    strip_accents: bool = False,
) -> str:
    if strip_html_tags:
        text = strip_html(text)
    text = normalize_unicode(text)
    if lowercase:
        text = text.lower()
    if strip_accents:
        text = remove_accents(text)
    if remove_punct:
        text = _nonalpha_re.sub(" ", text)
    tokens = _spaces_re.split(text.strip())
    if drop_stopwords:
        tokens = [t for t in tokens if t and t not in _STOPWORDS]
    else:
        tokens = [t for t in tokens if t]
    if stem:
        tokens = [_stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


def make_cleaner(
    lowercase: bool = True,
    strip_html_tags: bool = True,
    remove_punct: bool = True,
    drop_stopwords: bool = True,
    stem: bool = False,
    strip_accents: bool = False,
) -> Callable[[str], str]:
    """Return a configured cleaning function (usable in sklearn pipeline)."""
    def _clean(text: str) -> str:
        return clean_text(
            text,
            lowercase=lowercase,
            strip_html_tags=strip_html_tags,
            remove_punct=remove_punct,
            drop_stopwords=drop_stopwords,
            stem=stem,
            strip_accents=strip_accents,
        )
    return _clean


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = TEXT_COL,
    **cleaner_kwargs,
) -> pd.DataFrame:
    df = df.copy()
    cleaner = make_cleaner(**cleaner_kwargs)
    df["text_clean"] = df[text_col].fillna("").map(cleaner)
    df["n_tokens"] = df["text_clean"].str.split().str.len()
    df["n_chars"]  = df["text_clean"].str.len()
    log.info(
        "Preprocessing done. Median tokens: %.0f  |  empty docs: %d",
        df["n_tokens"].median(),
        (df["n_tokens"] == 0).sum(),
    )
    return df


def main() -> None:
    if not INTERIM_FILE.exists():
        log.error("Interim file not found: %s — run parse_html.py first", INTERIM_FILE)
        return

    df = pd.read_parquet(INTERIM_FILE)
    log.info("Loaded %d articles from %s", len(df), INTERIM_FILE)

    df = preprocess_dataframe(df)

    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_FILE, index=False)
    log.info("Saved → %s  (%d rows)", PROCESSED_FILE, len(df))

    print("\nToken stats by category:")
    print(df.groupby("category")["n_tokens"].describe()[["mean","50%","min","max"]].round(1).to_string())


if __name__ == "__main__":
    main()
