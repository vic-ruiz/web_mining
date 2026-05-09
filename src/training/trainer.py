"""
Model training: multiple classifiers, cross-validation, model persistence.

Models trained:
  1. TF-IDF + Logistic Regression   (fast baseline)
  2. TF-IDF + Linear SVC            (usually best linear model)
  3. TF-IDF + LightGBM              (strong ensemble)
  4. Sentence embeddings + LR       (semantic baseline)
  5. Sentence embeddings + LightGBM (semantic ensemble)
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import config as cfg
from src.features.tfidf_features import build_tfidf_pipeline
from src.features.embedding_features import SentenceEmbedder
from src.utils.logging_utils import get_logger

log = get_logger("trainer")


# ── Model definitions ─────────────────────────────────────────────────────────

def _get_models() -> dict[str, Any]:
    try:
        import lightgbm as lgb
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            random_state=cfg.RANDOM_STATE,
            verbose=-1,
        )
        lgb_available = True
    except ImportError:
        log.warning("LightGBM not installed — skipping LGB models")
        lgb_available = False

    models: dict[str, Any] = {
        "tfidf_lr": build_tfidf_pipeline(
            LogisticRegression(
                C=5.0, max_iter=1000,
                solver="lbfgs",
                random_state=cfg.RANDOM_STATE,
            )
        ),
        "tfidf_svc": build_tfidf_pipeline(
            LinearSVC(C=1.0, max_iter=2000, random_state=cfg.RANDOM_STATE),
            tfidf_kwargs={"ngram_range": (1, 2)},
        ),
    }

    if lgb_available:
        models["tfidf_lgb"] = build_tfidf_pipeline(lgb_clf)

    return models


def _get_embedding_models(X_emb: np.ndarray) -> dict[str, Any]:
    """Models that use pre-computed embeddings (no pipeline needed)."""
    try:
        import lightgbm as lgb
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            random_state=cfg.RANDOM_STATE,
            verbose=-1,
        )
        lgb_available = True
    except ImportError:
        lgb_available = False

    models: dict[str, Any] = {
        "emb_lr": LogisticRegression(
            C=5.0, max_iter=1000,
            solver="lbfgs",
            random_state=cfg.RANDOM_STATE,
        ),
    }
    if lgb_available:
        models["emb_lgb"] = lgb_clf

    return models


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_model(
    model: Any,
    X: Any,
    y: np.ndarray,
    n_folds: int = cfg.CV_FOLDS,
) -> dict:
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
    }
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "accuracy_mean":    round(float(results["test_accuracy"].mean()), 4),
        "accuracy_std":     round(float(results["test_accuracy"].std()),  4),
        "macro_f1_mean":    round(float(results["test_macro_f1"].mean()), 4),
        "macro_f1_std":     round(float(results["test_macro_f1"].std()),  4),
        "weighted_f1_mean": round(float(results["test_weighted_f1"].mean()), 4),
        "weighted_f1_std":  round(float(results["test_weighted_f1"].std()),  4),
    }


# ── Main training entry ───────────────────────────────────────────────────────

def train_all(df: pd.DataFrame) -> dict:
    """
    Train all models, run CV, save best model.
    Returns dict of results keyed by model name.
    """
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(df[cfg.LABEL_COL])
    X_text = df[cfg.TEXT_COL].tolist()

    log.info("Classes: %s", list(le.classes_))
    log.info("Class distribution: %s", dict(zip(le.classes_, np.bincount(y))))

    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}

    # ── TF-IDF based models ────────────────────────────────────────────────────
    for name, model in _get_models().items():
        log.info("Training %s …", name)
        t0 = time.time()
        cv_res = cross_validate_model(model, X_text, y)
        elapsed = round(time.time() - t0, 1)
        cv_res["train_time_s"] = elapsed
        all_results[name] = cv_res
        log.info("  %s  acc=%.3f±%.3f  macro_f1=%.3f±%.3f  (%.0fs)",
                 name,
                 cv_res["accuracy_mean"], cv_res["accuracy_std"],
                 cv_res["macro_f1_mean"], cv_res["macro_f1_std"],
                 elapsed)

        # Fit on full data and save
        model.fit(X_text, y)
        joblib.dump(model, cfg.MODELS_DIR / f"{name}.joblib")
        log.info("  Saved → models/%s.joblib", name)

    # ── Embedding based models ─────────────────────────────────────────────────
    try:
        from src.features.embedding_features import compute_and_cache
        X_emb = compute_and_cache(X_text)

        for name, model in _get_embedding_models(X_emb).items():
            log.info("Training %s …", name)
            t0 = time.time()
            cv_res = cross_validate_model(model, X_emb, y)
            elapsed = round(time.time() - t0, 1)
            cv_res["train_time_s"] = elapsed
            all_results[name] = cv_res
            log.info("  %s  acc=%.3f±%.3f  macro_f1=%.3f±%.3f  (%.0fs)",
                     name,
                     cv_res["accuracy_mean"], cv_res["accuracy_std"],
                     cv_res["macro_f1_mean"], cv_res["macro_f1_std"],
                     elapsed)

            model.fit(X_emb, y)
            joblib.dump(model, cfg.MODELS_DIR / f"{name}.joblib")
            log.info("  Saved → models/%s.joblib", name)

    except Exception as e:
        log.warning("Embedding models skipped: %s", e)

    # Save label encoder and results
    joblib.dump(le, cfg.MODELS_DIR / "label_encoder.joblib")
    results_path = cfg.METRICS_DIR / "cv_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(all_results, indent=2))
    log.info("CV results saved → %s", results_path)

    _print_summary(all_results)
    return all_results


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 65)
    print(f"{'Model':<20} {'Accuracy':>10} {'Macro F1':>10} {'Time(s)':>8}")
    print("-" * 65)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["macro_f1_mean"]):
        print(f"{name:<20} {r['accuracy_mean']:>9.3f}±{r['accuracy_std']:.3f}"
              f" {r['macro_f1_mean']:>9.3f}±{r['macro_f1_std']:.3f}"
              f" {r.get('train_time_s', 0):>7.0f}s")
    print("=" * 65)
