#!/usr/bin/env python3
"""
Generate the LaTeX result tables (\\input-ed by informe/informe_tp1.tex) from the
metrics JSON produced by train.py. Keeps the report data-driven and reproducible.

Usage:
    python scripts/make_latex_tables.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config.config import METRICS_DIR, ROOT

INFORME_DIR = ROOT / "informe"

DISPLAY = {
    "tfidf_svc":         "TF-IDF + LinearSVC",
    "tfidf_lr":          "TF-IDF + Reg. Logística",
    "bow_lr":            "BoW + Reg. Logística",
    "bow_nb":            "BoW + Naive Bayes",
    "emb_lr":            "Sentence embeddings + LR",
    "emb_lgb":           "Sentence embeddings + LightGBM",
    "tfidf_lgb":         "TF-IDF + LightGBM",
    "ensemble_voting":   "Ensamble (soft voting)",
    "ensemble_stacking": "Ensamble (stacking)",
}


def latex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("_", r"\_")


def cv_table(cv: dict) -> str:
    rows = sorted(cv.items(), key=lambda x: -x[1]["macro_f1_mean"])
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Modelo} & \textbf{Accuracy} & \textbf{Macro F1} \\",
        r"\midrule",
    ]
    for i, (name, r) in enumerate(rows):
        disp = latex_escape(DISPLAY.get(name, name))
        acc = f"{r['accuracy_mean']:.3f} $\\pm$ {r['accuracy_std']:.3f}"
        f1  = f"{r['macro_f1_mean']:.3f} $\\pm$ {r['macro_f1_std']:.3f}"
        if i == 0:  # best model in bold
            disp, acc, f1 = f"\\textbf{{{disp}}}", f"\\textbf{{{acc}}}", f"\\textbf{{{f1}}}"
        lines.append(f"{disp} & {acc} & {f1} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def temporal_table(temp: dict) -> str:
    cuts = temp.get("cuts", {})
    lines = [
        r"\begin{tabular}{rlrrcc}",
        r"\toprule",
        r"\textbf{Corte} & \textbf{Fecha} & \textbf{Train} & \textbf{Test} & \textbf{Accuracy} & \textbf{Macro F1} \\",
        r"\midrule",
    ]
    for tag, c in cuts.items():
        lines.append(
            f"{tag} & {c['cutoff_date']} & {c['train_size']} & {c['test_size']} "
            f"& {c['accuracy']:.3f} & {c['macro_f1']:.3f} \\\\"
        )
    s = temp.get("summary", {})
    if s:
        lines += [
            r"\midrule",
            f"\\multicolumn{{4}}{{r}}{{\\textbf{{Promedio}}}} "
            f"& \\textbf{{{s['accuracy_mean']:.3f} $\\pm$ {s['accuracy_std']:.3f}}} "
            f"& \\textbf{{{s['macro_f1_mean']:.3f} $\\pm$ {s['macro_f1_std']:.3f}}} \\\\",
        ]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    INFORME_DIR.mkdir(parents=True, exist_ok=True)
    cv = json.loads((METRICS_DIR / "cv_results.json").read_text())
    temp = json.loads((METRICS_DIR / "temporal_validation.json").read_text())

    (INFORME_DIR / "tabla_cv.tex").write_text(cv_table(cv) + "\n", encoding="utf-8")
    (INFORME_DIR / "tabla_temporal.tex").write_text(temporal_table(temp) + "\n", encoding="utf-8")
    print("Tablas LaTeX generadas en", INFORME_DIR)


if __name__ == "__main__":
    main()
