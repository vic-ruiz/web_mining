"""Central configuration for the Web Mining TP1 pipeline."""

from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

DATA_RAW       = ROOT / "data" / "raw"
DATA_INTERIM   = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SPLITS    = ROOT / "data" / "splits"
MODELS_DIR     = ROOT / "models"
REPORTS_DIR    = ROOT / "reports"
FIGURES_DIR    = REPORTS_DIR / "figures"
METRICS_DIR    = REPORTS_DIR / "metrics"

# ── Scraping ──────────────────────────────────────────────────────────────────
SECTIONS: dict[str, str] = {
    "economia": "economia",
    "el-pais":  "elpais",
    "sociedad": "sociedad",
    "el-mundo": "elmundo",
}

SCRAPER_PAGES_PER_BLOCK = 5
SCRAPER_PAGES_TO_SKIP   = 20
SCRAPER_NUM_BLOCKS      = 8   # 40 pages × 4 sections = 160 index requests
SCRAPER_DOWNLOAD_DELAY  = 1.5

# ── Parsing ───────────────────────────────────────────────────────────────────
INTERIM_FILE   = DATA_INTERIM / "articles.parquet"
MIN_BODY_CHARS = 150   # discard stubs shorter than this

# ── Preprocessing ─────────────────────────────────────────────────────────────
PROCESSED_FILE = DATA_PROCESSED / "dataset.parquet"

STOPWORDS_PATH = ROOT / "text_mining_python" / "text_mining_python" / "stopwords_es.txt"

# ── Features ─────────────────────────────────────────────────────────────────
TFIDF_CONFIG = {
    "min_df":      3,
    "max_df":      0.90,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "max_features": 50_000,
}

# sentence-transformers model (runs on CPU, ~420 MB)
SENTENCE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDINGS_CACHE = DATA_INTERIM / "embeddings.npy"

# ── Training ──────────────────────────────────────────────────────────────────
LABEL_COL    = "category"
TEXT_COL     = "body"
DATE_COL     = "date"

CV_FOLDS     = 5
RANDOM_STATE = 42

# ── Temporal split ────────────────────────────────────────────────────────────
# Train on articles before cutoff, test on articles after
TEMPORAL_CUTOFF_MONTHS = 3   # last N months → test set

# ── Category labels ───────────────────────────────────────────────────────────
CATEGORIES = ["economia", "elpais", "sociedad", "elmundo"]
CATEGORY_DISPLAY = {
    "economia": "Economía",
    "elpais":   "El País",
    "sociedad": "Sociedad",
    "elmundo":  "El Mundo",
}
