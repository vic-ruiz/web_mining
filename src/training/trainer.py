"""
Model training: BoW baseline, TF-IDF models, sentence embeddings, ensemble.

Execution order:
  1. BoW + Naive Bayes          — explicit baseline
  2. TF-IDF + LR                — strong baseline
  3. TF-IDF + LinearSVC         — best linear model
  4. Sentence embeddings + LR   — modern alternative (title+subtitle, fits 128-token context)
  5. Soft voting ensemble        — top-2 TF-IDF models with probability averaging
  6. Stacking ensemble           — LR meta-learner over all base models

Key design decision on embeddings:
  MiniLM-L12 has max 128 tokens. Full articles have median ~382 tokens.
  Embedding the full article loses 70% of text. Instead we embed
  title + subtitle + first 200 chars of body, which fits the context
  and is the most information-dense part of a news article.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import config as cfg
from src.features.tfidf_features import build_tfidf_pipeline, TextCleaner
from src.utils.logging_utils import get_logger

log = get_logger("trainer")


# ── Embedding text preparation ────────────────────────────────────────────────

def _make_embed_text(df: pd.DataFrame) -> list[str]:
    """
    Concatenate title + subtitle + first 200 chars of body.
    This fits within the 128-token context of MiniLM and uses the
    most semantically rich parts of the article.
    """
    texts = []
    for _, row in df.iterrows():
        title    = str(row.get("title",    "") or "")
        subtitle = str(row.get("subtitle", "") or "")
        body     = str(row.get("body",     "") or "")[:200]
        texts.append(f"{title}. {subtitle}. {body}".strip())
    return texts


# ── Model definitions ─────────────────────────────────────────────────────────

def _get_tfidf_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        # Explicit BoW baseline (CountVectorizer, no TF weighting)
        "bow_nb": Pipeline([
            ("cleaner", TextCleaner()),
            ("bow",     CountVectorizer(
                min_df=3, max_df=0.90,
                ngram_range=(1, 1),
                max_features=50_000,
            )),
            ("clf",     MultinomialNB(alpha=0.1)),
        ]),
        "bow_lr": Pipeline([
            ("cleaner", TextCleaner()),
            ("bow",     CountVectorizer(
                min_df=3, max_df=0.90,
                ngram_range=(1, 2),
                max_features=50_000,
            )),
            ("clf",     LogisticRegression(
                C=5.0, max_iter=1000, solver="lbfgs",
                random_state=cfg.RANDOM_STATE,
            )),
        ]),
        "tfidf_lr": build_tfidf_pipeline(
            LogisticRegression(
                C=5.0, max_iter=1000, solver="lbfgs",
                random_state=cfg.RANDOM_STATE,
            )
        ),
        "tfidf_svc": build_tfidf_pipeline(
            LinearSVC(C=1.0, max_iter=2000, random_state=cfg.RANDOM_STATE),
            tfidf_kwargs={"ngram_range": (1, 2)},
        ),
    }
    return models


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_model(
    model: Any,
    X: Any,
    y: np.ndarray,
    n_folds: int = cfg.CV_FOLDS,
    n_jobs: int = -1,
) -> dict:
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.RANDOM_STATE)
    scoring = {
        "accuracy":    "accuracy",
        "macro_f1":    "f1_macro",
        "weighted_f1": "f1_weighted",
    }
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
    return {
        "accuracy_mean":    round(float(results["test_accuracy"].mean()),    4),
        "accuracy_std":     round(float(results["test_accuracy"].std()),     4),
        "macro_f1_mean":    round(float(results["test_macro_f1"].mean()),    4),
        "macro_f1_std":     round(float(results["test_macro_f1"].std()),     4),
        "weighted_f1_mean": round(float(results["test_weighted_f1"].mean()), 4),
        "weighted_f1_std":  round(float(results["test_weighted_f1"].std()),  4),
    }


# ── Ensemble builders ─────────────────────────────────────────────────────────

def _build_soft_voting(X_text: list[str], y: np.ndarray) -> tuple[Any, dict]:
    """
    Soft voting between TF-IDF+LR and BoW+LR.
    LinearSVC has no predict_proba, so we use the two LR models.
    """
    lr_tfidf = build_tfidf_pipeline(
        LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs",
                           random_state=cfg.RANDOM_STATE)
    )
    lr_bow = Pipeline([
        ("cleaner", TextCleaner()),
        ("bow",     CountVectorizer(min_df=3, max_df=0.90,
                                   ngram_range=(1, 2), max_features=50_000)),
        ("clf",     LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs",
                                       random_state=cfg.RANDOM_STATE)),
    ])

    ensemble = VotingClassifier(
        estimators=[("tfidf_lr", lr_tfidf), ("bow_lr", lr_bow)],
        voting="soft",
        n_jobs=-1,
    )
    t0 = time.time()
    cv_res = cross_validate_model(ensemble, X_text, y, n_jobs=1)
    cv_res["train_time_s"] = round(time.time() - t0, 1)
    ensemble.fit(X_text, y)
    return ensemble, cv_res


def _build_stacking(X_text: list[str], y: np.ndarray) -> tuple[Any, dict]:
    """
    Stacking: TF-IDF+LR, BoW+LR, BoW+NB → meta LR.
    """
    base = [
        ("tfidf_lr", build_tfidf_pipeline(
            LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs",
                               random_state=cfg.RANDOM_STATE)
        )),
        ("bow_lr", Pipeline([
            ("cleaner", TextCleaner()),
            ("bow",     CountVectorizer(min_df=3, max_df=0.90,
                                       ngram_range=(1, 2), max_features=50_000)),
            ("clf",     LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs",
                                           random_state=cfg.RANDOM_STATE)),
        ])),
        ("bow_nb", Pipeline([
            ("cleaner", TextCleaner()),
            ("bow",     CountVectorizer(min_df=3, max_df=0.90,
                                       ngram_range=(1, 1), max_features=50_000)),
            ("clf",     MultinomialNB(alpha=0.1)),
        ])),
    ]
    meta = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                              random_state=cfg.RANDOM_STATE)
    stacking = StackingClassifier(
        estimators=base,
        final_estimator=meta,
        cv=3,
        n_jobs=1,
        passthrough=False,
    )
    t0 = time.time()
    cv_res = cross_validate_model(stacking, X_text, y, n_jobs=1)
    cv_res["train_time_s"] = round(time.time() - t0, 1)
    stacking.fit(X_text, y)
    return stacking, cv_res


# ── Main training entry ───────────────────────────────────────────────────────

def train_all(df: pd.DataFrame) -> dict:
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y       = le.fit_transform(df[cfg.LABEL_COL])
    X_text  = df[cfg.TEXT_COL].tolist()
    X_embed = _make_embed_text(df)

    log.info("Classes: %s", list(le.classes_))
    log.info("Class distribution: %s",
             dict(zip(le.classes_, np.bincount(y).tolist())))

    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}

    # ── 1. BoW and TF-IDF models ───────────────────────────────────────────────
    for name, model in _get_tfidf_models().items():
        log.info("Training %s …", name)
        t0     = time.time()
        cv_res = cross_validate_model(model, X_text, y)
        cv_res["train_time_s"] = round(time.time() - t0, 1)
        all_results[name] = cv_res
        log.info("  %s  acc=%.3f±%.3f  macro_f1=%.3f±%.3f  (%.0fs)",
                 name,
                 cv_res["accuracy_mean"], cv_res["accuracy_std"],
                 cv_res["macro_f1_mean"], cv_res["macro_f1_std"],
                 cv_res["train_time_s"])
        model.fit(X_text, y)
        joblib.dump(model, cfg.MODELS_DIR / f"{name}.joblib")

    # ── 2. Sentence embeddings on title+subtitle+body[:200] ───────────────────
    try:
        from src.features.embedding_features import compute_and_cache

        embed_cache = cfg.DATA_INTERIM / "embeddings_short.npy"
        X_emb = compute_and_cache(X_embed, cache_path=embed_cache)

        emb_lr = LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs",
                                    random_state=cfg.RANDOM_STATE)
        log.info("Training emb_lr (title+subtitle+body excerpt) …")
        t0     = time.time()
        cv_res = cross_validate_model(emb_lr, X_emb, y)
        cv_res["train_time_s"] = round(time.time() - t0, 1)
        all_results["emb_lr"] = cv_res
        log.info("  emb_lr  acc=%.3f±%.3f  macro_f1=%.3f±%.3f  (%.0fs)",
                 cv_res["accuracy_mean"], cv_res["accuracy_std"],
                 cv_res["macro_f1_mean"], cv_res["macro_f1_std"],
                 cv_res["train_time_s"])
        emb_lr.fit(X_emb, y)
        joblib.dump(emb_lr, cfg.MODELS_DIR / "emb_lr.joblib")

    except Exception as e:
        log.warning("Embedding model skipped: %s", e)

    # ── 3. Soft voting ensemble ────────────────────────────────────────────────
    log.info("Training ensemble_voting …")
    ens_voting, cv_voting = _build_soft_voting(X_text, y)
    all_results["ensemble_voting"] = cv_voting
    joblib.dump(ens_voting, cfg.MODELS_DIR / "ensemble_voting.joblib")
    log.info("  ensemble_voting  acc=%.3f±%.3f  macro_f1=%.3f±%.3f  (%.0fs)",
             cv_voting["accuracy_mean"], cv_voting["accuracy_std"],
             cv_voting["macro_f1_mean"], cv_voting["macro_f1_std"],
             cv_voting["train_time_s"])

    # ── 4. Stacking ensemble ───────────────────────────────────────────────────
    log.info("Training ensemble_stacking …")
    ens_stack, cv_stack = _build_stacking(X_text, y)
    all_results["ensemble_stacking"] = cv_stack
    joblib.dump(ens_stack, cfg.MODELS_DIR / "ensemble_stacking.joblib")
    log.info("  ensemble_stacking  acc=%.3f±%.3f  macro_f1=%.3f±%.3f  (%.0fs)",
             cv_stack["accuracy_mean"], cv_stack["accuracy_std"],
             cv_stack["macro_f1_mean"], cv_stack["macro_f1_std"],
             cv_stack["train_time_s"])

    # ── Save label encoder and results ────────────────────────────────────────
    joblib.dump(le, cfg.MODELS_DIR / "label_encoder.joblib")
    results_path = cfg.METRICS_DIR / "cv_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(all_results, indent=2))
    log.info("CV results → %s", results_path)

    _print_summary(all_results)
    return all_results


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'Accuracy':>12} {'Macro F1':>12} {'Time':>8}")
    print("-" * 70)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["macro_f1_mean"]):
        print(f"{name:<25}"
              f" {r['accuracy_mean']:>8.3f}±{r['accuracy_std']:.3f}"
              f" {r['macro_f1_mean']:>8.3f}±{r['macro_f1_std']:.3f}"
              f" {r.get('train_time_s', 0):>6.0f}s")
    print("=" * 70)
