"""Configuración del backend: rutas del proyecto y conexión a la base de datos.

Los valores se leen del archivo .env de la raíz del proyecto (ver .env.example).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import URL

RAIZ = Path(__file__).resolve().parents[2]
load_dotenv(RAIZ / ".env")

DATOS = Path(os.getenv("DATOS_DIR", RAIZ / "datos"))  # DATOS_DIR solo se usa en pruebas
CRUDOS_VIDEOS = DATOS / "crudos" / "videos"
CRUDOS_FOTOS = DATOS / "crudos" / "fotos"
IMAGENES = DATOS / "imagenes"   # imágenes candidatas al dataset (fotogramas y fotos procesadas)
DATASET = DATOS / "dataset"     # dataset exportado en formato YOLO
EVIDENCIAS = DATOS / "evidencias"  # fotos de la vitrina con las que se actualizó el inventario

ANCHO_MAXIMO = 1280  # ancho con el que se guardan las imágenes del dataset


def url_base_datos():
    # DB_URL permite usar otra base (p. ej. SQLite en pruebas) sin tocar el código.
    if os.getenv("DB_URL"):
        return os.environ["DB_URL"]
    # URL.create escapa caracteres especiales de la contraseña (@, :, /).
    return URL.create(
        "postgresql+psycopg",
        username=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "inventario_cafeteria"),
    )
