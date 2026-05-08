"""TF-IDF feature extraction with sklearn FunctionTransformer wrapper."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.config import TFIDF_CONFIG
from src.preprocessing.text_cleaner import make_cleaner
from src.utils.logging_utils import get_logger

log = get_logger("tfidf")


def build_tfidf_pipeline(
    classifier: Any,
    cleaner_kwargs: dict | None = None,
    tfidf_kwargs: dict | None = None,
) -> Pipeline:
    """
    Returns a full sklearn Pipeline:
        raw text → clean → TF-IDF → classifier
    """
    ck = cleaner_kwargs or {}
    tk = {**TFIDF_CONFIG, **(tfidf_kwargs or {})}

    cleaner_fn = make_cleaner(**ck)

    steps = [
        ("cleaner", FunctionTransformer(
            lambda X: [cleaner_fn(t) for t in X],
            validate=False,
        )),
        ("tfidf", TfidfVectorizer(**tk)),
        ("clf", classifier),
    ]
    return Pipeline(steps)


def get_top_features(pipeline: Pipeline, category_names: list[str], n: int = 20) -> dict:
    """Return top TF-IDF features per class for a fitted LogReg or SVC pipeline."""
    clf = pipeline.named_steps["clf"]
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    feature_names = np.array(vectorizer.get_feature_names_out())

    result = {}
    if hasattr(clf, "coef_"):
        coef = clf.coef_
        for i, cat in enumerate(category_names):
            top_idx = np.argsort(coef[i])[-n:][::-1]
            result[cat] = list(feature_names[top_idx])
    return result
