# Muestra de páginas HTML por sección

Esta carpeta es el **entregable de datos** que pide la consigna del TP1:

> *"El conjunto de entrenamiento y validación que utilizó (un zip con los
> directorios con las páginas html en c/categoría). Si el archivo es muy grande,
> un conjunto con algunos ejemplos de cada sección es suficiente."*

El corpus completo son ~6.000 artículos (~1.500 por sección), demasiado grande para
versionar como HTML crudo. Por eso aquí se incluyen **12 ejemplos reales por
sección**, descargados desde las URLs efectivamente usadas y guardados en UTF-8:

| Carpeta      | Sección  | Ejemplos |
|--------------|----------|---------:|
| `economia/`  | Economía | 12 |
| `elpais/`    | El País  | 12 |
| `sociedad/`  | Sociedad | 12 |
| `elmundo/`   | El Mundo | 12 |

Los ejemplos se eligieron espaciados a lo largo del rango temporal del corpus
(mayo 2023 → mayo 2026) para reflejar la diversidad temporal del dataset.

> **Nota:** el corpus real no se descargó como HTML sino vía la API JSON de Arc
> Publishing (ver informe, sección 2). Estos HTML se reconstruyen con
> `scripts/download_sample_html.py` a partir de las mismas URLs, únicamente como
> entregable de muestra. El dataset estructurado completo está en
> `data/interim/articles.parquet`.

Para regenerar la muestra:

```bash
python scripts/download_sample_html.py --per-class 12
```
