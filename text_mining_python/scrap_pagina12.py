# -*- coding: utf-8 -*-
"""
Spider de Scrapy para descargar noticias de Página 12.
Secciones: Economía, Sociedad, El Mundo, El País.

Ejecutar con:
    python scrap_pagina12.py

Estructura de salida:
    noticias/
    ├── economia/    → .html de cada noticia
    ├── sociedad/
    ├── elmundo/
    └── elpais/
"""

import os
import re
import multiprocessing

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.http import HtmlResponse


# ── Configuración de secciones ────────────────────────────────────────────────

# Directorio base donde se crearán las subcarpetas (relativo a este script)
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "noticias")

# Clave = segmento en la URL de Página 12 / Valor = carpeta local de destino
SECCIONES = {
    "economia": "economia",
    "sociedad": "sociedad",
    "el-mundo":  "elmundo",
    "el-pais":   "elpais",
}


# ── Estrategia de salto de páginas ────────────────────────────────────────────

def generar_paginas(pages_per_block: int = 5,
                    pages_to_skip: int = 20,
                    num_blocks: int = 6) -> list:
    """
    Genera los números de página a descargar usando una estrategia de saltos
    para cubrir un arco temporal amplio sin descargar miles de páginas.

    Ejemplo con valores por defecto:
        Bloque 1 →  páginas  1-5
        Bloque 2 →  páginas 26-30   (salto de 20)
        Bloque 3 →  páginas 51-55
        Bloque 4 →  páginas 76-80
        Bloque 5 →  páginas 101-105
        Bloque 6 →  páginas 126-130
    Total: 30 páginas × 4 secciones = 120 requests de índice.
    """
    paginas = []
    page = 1
    for _ in range(num_blocks):
        for offset in range(pages_per_block):
            paginas.append(page + offset)
        page += pages_per_block + pages_to_skip
    return paginas


# ── Spider principal ──────────────────────────────────────────────────────────

class Pagina12Spider(scrapy.Spider):

    name = "crawler_pagina12"
    allowed_domains = ["www.pagina12.com.ar", "pagina12.com.ar"]

    # Regex: artículos con ID numérico de ≥6 dígitos seguido de guion.
    # Ejemplos válidos:
    #   https://www.pagina12.com.ar/289430-nombre-de-la-nota
    #   https://www.pagina12.com.ar/1234567-otra-nota-larga
    ARTICLE_REGEX = re.compile(r"^https://www\.pagina12\.com\.ar/\d{6,}-")

    custom_settings = {
        "USER_AGENT": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "LOG_ENABLED": True,
        "LOG_LEVEL": "INFO",
        "ROBOTSTXT_OBEY": False,
        # Espera base entre requests; Scrapy la multiplica aleatoriamente
        # entre 0.5× y 1.5× para no parecer un bot predecible.
        "DOWNLOAD_DELAY": 1.5,
        "RANDOMIZE_DOWNLOAD_DELAY": True,
        # Reintentos mínimos para no saturar el servidor ante errores 5xx
        "RETRY_TIMES": 2,
        # Scrapy no sigue links por cuenta propia; lo controlamos nosotros
        "DEPTH_LIMIT": 0,
    }

    def __init__(self, base_dir: str = BASE_DIR, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_dir = base_dir
        self._crear_directorios()

    def _crear_directorios(self):
        """Crea las 4 carpetas de destino si no existen."""
        for carpeta in SECCIONES.values():
            ruta = os.path.join(self.base_dir, carpeta)
            os.makedirs(ruta, exist_ok=True)
            self.logger.info(f"Directorio listo: {ruta}")

    # ── Generación de requests iniciales ─────────────────────────────────────

    def start_requests(self):
        """
        Genera los requests al índice de cada sección usando la estrategia
        de saltos, en lugar de descargar todas las páginas en secuencia.
        """
        paginas = generar_paginas(
            pages_per_block=5,
            pages_to_skip=20,
            num_blocks=6,
        )
        self.logger.info(f"Páginas a descargar por sección: {paginas}")

        for seccion, carpeta in SECCIONES.items():
            for nro in paginas:
                url = (
                    f"https://www.pagina12.com.ar/secciones/{seccion}"
                    f"?page={nro}"
                )
                yield scrapy.Request(
                    url,
                    callback=self.parse_indice,
                    cb_kwargs={"carpeta": carpeta},
                )

    # ── Parseo del índice de sección ─────────────────────────────────────────

    def parse_indice(self, response: HtmlResponse, carpeta: str):
        """
        Lee una página de índice de sección y genera un request
        por cada link de noticia que cumpla con el patrón regex.
        """
        links_encontrados = 0

        for href in response.css("a::attr(href)").getall():
            url_absoluta = response.urljoin(href)
            if self.ARTICLE_REGEX.match(url_absoluta):
                links_encontrados += 1
                yield scrapy.Request(
                    url_absoluta,
                    callback=self.parse_noticia,
                    cb_kwargs={"carpeta": carpeta},
                )

        self.logger.info(
            f"[{carpeta}] {response.url} "
            f"→ {links_encontrados} noticias encontradas"
        )

    # ── Guardado de la noticia ────────────────────────────────────────────────

    def parse_noticia(self, response: HtmlResponse, carpeta: str):
        """
        Guarda el HTML completo de la noticia en la carpeta correspondiente.
        El nombre del archivo es el slug final de la URL.
        Encoding: UTF-8 (Página 12 usa este encoding de forma consistente).
        """
        # Extraer el slug de la URL, ej. "289430-nombre-de-la-nota"
        slug = response.url.rstrip("/").split("/")[-1]
        if not slug.endswith(".html"):
            slug += ".html"

        ruta_archivo = os.path.join(self.base_dir, carpeta, slug)

        with open(ruta_archivo, "wt", encoding="utf-8") as f:
            f.write(response.text)

        self.logger.info(f"[{carpeta}] Guardada: {slug}")


# ── Punto de entrada ──────────────────────────────────────────────────────────

def run_spider(base_dir: str):
    """Lanza el spider dentro de un CrawlerProcess de Scrapy."""
    process = CrawlerProcess()
    process.crawl(Pagina12Spider, base_dir=base_dir)
    process.start()


if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)

    # El reactor de Twisted (que usa Scrapy internamente) no puede reiniciarse
    # en el mismo proceso. Correr el spider en un proceso separado permite
    # lanzarlo sin restricciones y es el patrón recomendado para scripts locales.
    p = multiprocessing.Process(target=run_spider, args=(BASE_DIR,))
    p.start()
    p.join()

    print(f"\nListo. Noticias guardadas en: {BASE_DIR}")
