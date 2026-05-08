"""
Sentence-transformers embeddings feature extractor.

Uses paraphrase-multilingual-MiniLM-L12-v2 (runs on CPU, ~420 MB).
Embeddings are cached to disk to avoid recomputing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.config import SENTENCE_MODEL, EMBEDDINGS_CACHE
from src.utils.logging_utils import get_logger

log = get_logger("embeddings")


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer that converts texts to dense sentence
    embeddings using sentence-transformers.
    """

    def __init__(
        self,
        model_name: str = SENTENCE_MODEL,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress = show_progress
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            log.info("Loading sentence-transformers model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        model = self._load_model()
        texts = list(X)
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings


def compute_and_cache(
    texts: list[str],
    cache_path: Path = EMBEDDINGS_CACHE,
    model_name: str = SENTENCE_MODEL,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute embeddings and save to disk; reload from cache if available."""
    if cache_path.exists():
        log.info("Loading cached embeddings from %s", cache_path)
        return np.load(str(cache_path))

    log.info("Computing embeddings for %d texts (this may take a while on CPU)...", len(texts))
    embedder = SentenceEmbedder(model_name=model_name, batch_size=batch_size)
    embeddings = embedder.transform(texts)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), embeddings)
    log.info("Embeddings cached → %s  shape=%s", cache_path, embeddings.shape)
    return embeddings
