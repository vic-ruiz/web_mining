"""
Evaluation: classification metrics, confusion matrices, temporal validation.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import config as cfg
from src.utils.logging_utils import get_logger

log = get_logger("evaluator")

sns.set_theme(style="whitegrid", font_scale=1.1)


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f",
        xticklabels=labels, yticklabels=labels,
        cmap="Blues", ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        log.info("Confusion matrix saved → %s", save_path)
    plt.close(fig)


# ── Temporal validation ───────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    cutoff_months: int = cfg.TEMPORAL_CUTOFF_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame by date. Oldest articles → train, newest → test.
    Ensures NO future data leaks into training.
    """
    df = df.dropna(subset=[cfg.DATE_COL]).copy()
    max_date = df[cfg.DATE_COL].max()
    cutoff = max_date - pd.DateOffset(months=cutoff_months)

    train = df[df[cfg.DATE_COL] <= cutoff]
    test  = df[df[cfg.DATE_COL] >  cutoff]

    log.info(
        "Temporal split: train=%d (up to %s)  test=%d (after %s)",
        len(train), cutoff.date(), len(test), cutoff.date()
    )
    return train, test


def run_temporal_validation(
    df: pd.DataFrame,
    model_name: str = "tfidf_lr",
) -> dict:
    """
    Train on oldest articles, evaluate on newest.
    Returns metrics dict.
    """
    from sklearn.preprocessing import LabelEncoder

    model_path = cfg.MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        log.error("Model not found: %s", model_path)
        return {}

    train_df, test_df = temporal_split(df)
    if len(test_df) < 10:
        log.warning("Test set too small for temporal validation (%d)", len(test_df))
        return {}

    le = joblib.load(cfg.MODELS_DIR / "label_encoder.joblib")

    is_pipeline = hasattr(joblib.load(model_path), "predict")
    model = joblib.load(model_path)

    train_X = train_df[cfg.TEXT_COL].tolist()
    test_X  = test_df[cfg.TEXT_COL].tolist()
    train_y = le.transform(train_df[cfg.LABEL_COL])
    test_y  = le.transform(test_df[cfg.LABEL_COL])

    model.fit(train_X, train_y)
    pred = model.predict(test_X)

    report = classification_report(test_y, pred, target_names=le.classes_, output_dict=True)
    result = {
        "model":         model_name,
        "train_size":    len(train_df),
        "test_size":     len(test_df),
        "accuracy":      round(accuracy_score(test_y, pred), 4),
        "macro_f1":      round(f1_score(test_y, pred, average="macro"), 4),
        "weighted_f1":   round(f1_score(test_y, pred, average="weighted"), 4),
        "per_class":     {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"], 4),
                "f1":        round(report[cls]["f1-score"], 4),
            }
            for cls in le.classes_
        },
    }

    # Confusion matrix
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        test_y, pred, list(le.classes_),
        title=f"Temporal Validation — {model_name}",
        save_path=cfg.FIGURES_DIR / f"cm_temporal_{model_name}.png",
    )

    log.info(
        "[Temporal] %s  acc=%.3f  macro_f1=%.3f",
        model_name, result["accuracy"], result["macro_f1"]
    )
    return result


# ── Cross-validation confusion matrix ────────────────────────────────────────

def cv_confusion_matrix(
    model,
    X: list[str],
    y: np.ndarray,
    labels: list[str],
    model_name: str = "model",
    n_folds: int = cfg.CV_FOLDS,
) -> None:
    """Plot averaged confusion matrix from stratified cross-validation."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.RANDOM_STATE)
    cms = []
    for train_idx, test_idx in cv.split(X, y):
        X_train = [X[i] for i in train_idx]
        X_test  = [X[i] for i in test_idx]
        model.fit(X_train, y[train_idx])
        pred = model.predict(X_test)
        cms.append(confusion_matrix(y[test_idx], pred, normalize="true",
                                    labels=range(len(labels))))
    avg_cm = np.mean(cms, axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(avg_cm, annot=True, fmt=".2f",
                xticklabels=labels, yticklabels=labels,
                cmap="Blues", ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"CV Confusion Matrix ({n_folds}-fold) — {model_name}")
    fig.tight_layout()

    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = cfg.FIGURES_DIR / f"cm_cv_{model_name}.png"
    fig.savefig(save_path, dpi=120)
    log.info("CV confusion matrix saved → %s", save_path)
    plt.close(fig)


# ── EDA plots ─────────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["category"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax, color=sns.color_palette("muted", len(counts)))
    ax.set_title("Article Distribution by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    fig.tight_layout()
    save_path = cfg.FIGURES_DIR / "class_distribution.png"
    fig.savefig(save_path, dpi=120)
    log.info("Class distribution plot → %s", save_path)
    plt.close(fig)


def plot_articles_over_time(df: pd.DataFrame) -> None:
    df = df.dropna(subset=[cfg.DATE_COL]).copy()
    df["month"] = df[cfg.DATE_COL].dt.to_period("M")
    pivot = df.groupby(["month", "category"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(ax=ax, marker="o", markersize=3)
    ax.set_title("Articles per Month by Category")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    fig.tight_layout()
    save_path = cfg.FIGURES_DIR / "articles_over_time.png"
    fig.savefig(save_path, dpi=120)
    log.info("Timeline plot → %s", save_path)
    plt.close(fig)


def plot_token_length_distribution(df: pd.DataFrame) -> None:
    if "n_tokens" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for cat, grp in df.groupby("category"):
        sns.kdeplot(grp["n_tokens"], ax=ax, label=cat, fill=True, alpha=0.3)
    ax.set_title("Token Length Distribution by Category")
    ax.set_xlabel("Tokens")
    ax.legend()
    fig.tight_layout()
    save_path = cfg.FIGURES_DIR / "token_distribution.png"
    fig.savefig(save_path, dpi=120)
    log.info("Token distribution plot → %s", save_path)
    plt.close(fig)


def plot_model_comparison(results: dict) -> None:
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "Accuracy": r["accuracy_mean"],
            "Macro F1": r["macro_f1_mean"],
        })
    comp = pd.DataFrame(rows).sort_values("Macro F1", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comp))
    w = 0.35
    ax.barh(x - w/2, comp["Accuracy"], w, label="Accuracy", color="#4C72B0")
    ax.barh(x + w/2, comp["Macro F1"], w, label="Macro F1",  color="#DD8452")
    ax.set_yticks(x)
    ax.set_yticklabels(comp["Model"])
    ax.set_xlim(0, 1)
    ax.set_title("Model Comparison (5-fold CV)")
    ax.legend()
    fig.tight_layout()
    save_path = cfg.FIGURES_DIR / "model_comparison.png"
    fig.savefig(save_path, dpi=120)
    log.info("Model comparison plot → %s", save_path)
    plt.close(fig)
