# TP1 Web Mining — Clasificación de Noticias Página 12

**Generado:** 2026-05-09 11:13  
**Dataset:** 3941 artículos  
**Categorías:** Economía, El País, Sociedad, El Mundo

---

## 1. Dataset

| Categoría | Artículos |
|-----------|----------:|
| economia | 954 |
| elmundo | 1000 |
| elpais | 987 |
| sociedad | 1000 |

**Rango temporal:** 2022-09-30 → 2026-05-09  
**Mediana de tokens por artículo:** 382

---

## 2. Cross-Validation (5-fold Estratificado)

| Modelo | Accuracy | Macro F1 | Weighted F1 |
|--------|:--------:|:--------:|:-----------:|
| tfidf_svc | 0.908±0.011 | 0.908±0.011 | 0.908±0.011 |
| ensemble_stacking | 0.902±0.007 | 0.902±0.007 | 0.902±0.007 |
| tfidf_lr | 0.901±0.010 | 0.902±0.010 | 0.901±0.010 |
| ensemble_voting | 0.896±0.011 | 0.896±0.011 | 0.896±0.011 |
| bow_lr | 0.892±0.010 | 0.893±0.011 | 0.892±0.010 |
| bow_nb | 0.887±0.011 | 0.887±0.011 | 0.886±0.011 |
| emb_lr | 0.808±0.011 | 0.809±0.011 | 0.808±0.011 |

> **Mejor modelo:** `tfidf_svc` (por Macro F1)

---

## 3. Validación Temporal

Entrenamiento en artículos históricos, test en los más recientes.
Esto simula el desempeño real del modelo ante noticias futuras.

| Métrica | Valor |
|---------|------:|
| Modelo  | tfidf_svc |
| Train   | 3481 artículos |
| Test    | 460 artículos |
| Accuracy | 0.874 |
| Macro F1 | 0.878 |
| Weighted F1 | 0.873 |

### F1 por categoría (temporal)

| Categoría | Precision | Recall | F1 |
|-----------|:---------:|:------:|:--:|
| economia | 0.805 | 0.986 | 0.886 |
| elmundo | 0.855 | 0.947 | 0.899 |
| elpais | 0.904 | 0.805 | 0.852 |
| sociedad | 0.896 | 0.860 | 0.878 |

---

## 4. Análisis

### ¿Por qué usar validación temporal?

La validación cruzada estándar mezcla artículos de todas las fechas,
lo que puede introducir **data leakage temporal**: el modelo "ve" noticias
futuras durante el entrenamiento (a través de la jerga política/económica
del momento). La validación temporal evita este sesgo.

### Sesgo de clases

Si alguna categoría tiene muchos más artículos, el modelo tiende a
favorecerla. Por eso reportamos **Macro F1** (trata todas las clases
por igual) además de Accuracy.

### Clases difíciles

Las categorías más confundibles suelen ser:
- **El País vs. Economía**: muchas notas mezclan política económica
- **El Mundo vs. El País**: noticias de política exterior

---

## 5. Figuras

![articles_over_time](reports/figures/articles_over_time.png)

![class_distribution](reports/figures/class_distribution.png)

![cm_temporal_tfidf_svc](reports/figures/cm_temporal_tfidf_svc.png)

![model_comparison](reports/figures/model_comparison.png)

![token_distribution](reports/figures/token_distribution.png)

---

## 6. Reproducibilidad

```bash
# 1. Scraping
python scripts/run_scraper.py

# 2. Parsing + preprocesamiento
python scripts/parse_html.py

# 3. Entrenamiento + evaluación
python scripts/train.py

# 4. Reporte
python scripts/generate_report.py
```

---
_Generado automáticamente por el pipeline Web Mining TP1_