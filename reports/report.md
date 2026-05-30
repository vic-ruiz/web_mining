# TP1 Web Mining — Clasificación de Noticias Página 12

**Generado:** 2026-05-29 21:18  
**Dataset:** 5999 artículos  
**Categorías:** Economía, El País, Sociedad, El Mundo

---

## 1. Dataset

| Categoría | Artículos |
|-----------|----------:|
| economia | 1500 |
| elmundo | 1499 |
| elpais | 1500 |
| sociedad | 1500 |

**Rango temporal:** 2023-05-23 → 2026-05-29  
**Mediana de tokens por artículo:** 389

---

## 2. Cross-Validation (5-fold Estratificado)

| Modelo | Accuracy | Macro F1 | Weighted F1 |
|--------|:--------:|:--------:|:-----------:|
| tfidf_svc | 0.904±0.011 | 0.903±0.011 | 0.903±0.011 |
| ensemble_stacking | 0.901±0.009 | 0.901±0.009 | 0.901±0.009 |
| tfidf_lr | 0.900±0.010 | 0.900±0.010 | 0.900±0.010 |
| ensemble_voting | 0.890±0.009 | 0.890±0.009 | 0.890±0.009 |
| bow_lr | 0.885±0.007 | 0.885±0.007 | 0.885±0.007 |
| bow_nb | 0.883±0.006 | 0.882±0.006 | 0.882±0.006 |
| emb_lr | 0.818±0.005 | 0.818±0.005 | 0.818±0.005 |

> **Mejor modelo:** `tfidf_svc` (por Macro F1)

---

## 3. Validación Temporal (varios cortes)

Entrenamiento en artículos históricos, test en los más recientes.
Esto simula el desempeño real del modelo ante noticias futuras.
Se evalúan **3 cortes** (3, 6 y 9 meses) para descartar que un
buen resultado sea un accidente de un período particular.

**Modelo:** `tfidf_svc`

| Corte | Fecha corte | Train | Test | Accuracy | Macro F1 | Weighted F1 |
|------:|-------------|------:|-----:|:--------:|:--------:|:-----------:|
| 3m | 2026-02-28 | 5185 | 814 | 0.853 | 0.858 | 0.852 |
| 6m | 2025-11-29 | 4403 | 1596 | 0.848 | 0.853 | 0.847 |
| 9m | 2025-08-29 | 3570 | 2429 | 0.846 | 0.849 | 0.845 |

> **Promedio entre cortes:** Macro F1 = 0.853 ± 0.004  ·  Accuracy = 0.849 ± 0.003

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

![cm_cv_tfidf_svc](reports/figures/cm_cv_tfidf_svc.png)

![cm_temporal_tfidf_svc](reports/figures/cm_temporal_tfidf_svc.png)

![cm_temporal_tfidf_svc_3m](reports/figures/cm_temporal_tfidf_svc_3m.png)

![cm_temporal_tfidf_svc_6m](reports/figures/cm_temporal_tfidf_svc_6m.png)

![cm_temporal_tfidf_svc_9m](reports/figures/cm_temporal_tfidf_svc_9m.png)

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