#!/usr/bin/env python3
"""
Generate final Markdown report from saved metrics.

Usage:
    python scripts/generate_report.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config.config import METRICS_DIR, FIGURES_DIR, REPORTS_DIR, PROCESSED_FILE
from src.utils.logging_utils import get_logger

log = get_logger("report")


def load_cv_results() -> dict:
    p = METRICS_DIR / "cv_results.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_temporal_results() -> dict:
    p = METRICS_DIR / "temporal_validation.json"
    return json.loads(p.read_text()) if p.exists() else {}


def build_report(cv: dict, temporal: dict, df: pd.DataFrame) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n = len(df)
    cats = df["category"].value_counts().to_dict()

    lines = [
        "# TP1 Web Mining — Clasificación de Noticias Página 12",
        "",
        f"**Generado:** {now}  ",
        f"**Dataset:** {n} artículos  ",
        f"**Categorías:** Economía, El País, Sociedad, El Mundo",
        "",
        "---",
        "",
        "## 1. Dataset",
        "",
        "| Categoría | Artículos |",
        "|-----------|----------:|",
    ]
    for cat, cnt in sorted(cats.items()):
        lines.append(f"| {cat} | {cnt} |")

    d_min = df["date"].min()
    d_max = df["date"].max()
    lines += [
        "",
        f"**Rango temporal:** {d_min.date() if pd.notna(d_min) else '?'} → {d_max.date() if pd.notna(d_max) else '?'}  ",
        f"**Mediana de tokens por artículo:** {df.get('n_tokens', pd.Series([0])).median():.0f}",
        "",
        "---",
        "",
        "## 2. Cross-Validation (5-fold Estratificado)",
        "",
        "| Modelo | Accuracy | Macro F1 | Weighted F1 |",
        "|--------|:--------:|:--------:|:-----------:|",
    ]

    for name, r in sorted(cv.items(), key=lambda x: -x[1]["macro_f1_mean"]):
        acc  = f"{r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f}"
        mf1  = f"{r['macro_f1_mean']:.3f}±{r['macro_f1_std']:.3f}"
        wf1  = f"{r['weighted_f1_mean']:.3f}±{r['weighted_f1_std']:.3f}"
        lines.append(f"| {name} | {acc} | {mf1} | {wf1} |")

    best = max(cv, key=lambda k: cv[k]["macro_f1_mean"]) if cv else "N/A"
    lines += [
        "",
        f"> **Mejor modelo:** `{best}` (por Macro F1)",
        "",
        "---",
        "",
        "## 3. Validación Temporal",
        "",
        "Entrenamiento en artículos históricos, test en los más recientes.",
        "Esto simula el desempeño real del modelo ante noticias futuras.",
        "",
    ]

    if temporal:
        lines += [
            f"| Métrica | Valor |",
            f"|---------|------:|",
            f"| Modelo  | {temporal.get('model', '?')} |",
            f"| Train   | {temporal.get('train_size', '?')} artículos |",
            f"| Test    | {temporal.get('test_size', '?')} artículos |",
            f"| Accuracy | {temporal.get('accuracy', 0):.3f} |",
            f"| Macro F1 | {temporal.get('macro_f1', 0):.3f} |",
            f"| Weighted F1 | {temporal.get('weighted_f1', 0):.3f} |",
            "",
            "### F1 por categoría (temporal)",
            "",
            "| Categoría | Precision | Recall | F1 |",
            "|-----------|:---------:|:------:|:--:|",
        ]
        for cls, m in temporal.get("per_class", {}).items():
            lines.append(f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    else:
        lines.append("_Resultados temporales no disponibles._")

    lines += [
        "",
        "---",
        "",
        "## 4. Análisis",
        "",
        "### ¿Por qué usar validación temporal?",
        "",
        "La validación cruzada estándar mezcla artículos de todas las fechas,",
        "lo que puede introducir **data leakage temporal**: el modelo \"ve\" noticias",
        "futuras durante el entrenamiento (a través de la jerga política/económica",
        "del momento). La validación temporal evita este sesgo.",
        "",
        "### Sesgo de clases",
        "",
        "Si alguna categoría tiene muchos más artículos, el modelo tiende a",
        "favorecerla. Por eso reportamos **Macro F1** (trata todas las clases",
        "por igual) además de Accuracy.",
        "",
        "### Clases difíciles",
        "",
        "Las categorías más confundibles suelen ser:",
        "- **El País vs. Economía**: muchas notas mezclan política económica",
        "- **El Mundo vs. El País**: noticias de política exterior",
        "",
        "---",
        "",
        "## 5. Figuras",
        "",
    ]

    for fig in sorted(FIGURES_DIR.glob("*.png")):
        lines.append(f"![{fig.stem}](reports/figures/{fig.name})")
        lines.append("")

    lines += [
        "---",
        "",
        "## 6. Reproducibilidad",
        "",
        "```bash",
        "# 1. Scraping",
        "python scripts/run_scraper.py",
        "",
        "# 2. Parsing + preprocesamiento",
        "python scripts/parse_html.py",
        "",
        "# 3. Entrenamiento + evaluación",
        "python scripts/train.py",
        "",
        "# 4. Reporte",
        "python scripts/generate_report.py",
        "```",
        "",
        "---",
        "_Generado automáticamente por el pipeline Web Mining TP1_",
    ]

    return "\n".join(lines)


def main() -> None:
    cv = load_cv_results()
    temporal = load_temporal_results()

    if not cv:
        log.warning("No CV results found — run train.py first")

    df = pd.read_parquet(PROCESSED_FILE) if PROCESSED_FILE.exists() else pd.DataFrame()

    report = build_report(cv, temporal, df)

    out = REPORTS_DIR / "report.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    log.info("Report saved → %s", out)
    print(f"\nReport written to: {out}")


if __name__ == "__main__":
    main()
