"""Procesamiento de los archivos subidos: guardado, fotos y extracción de fotogramas."""

import re
import shutil
import unicodedata
import uuid
from pathlib import Path

import cv2
import numpy as np

from vision.entrenamiento.extraer_fotogramas import PARAMETROS, procesar_video

from .config import ANCHO_MAXIMO, DATOS, IMAGENES
from .db import SesionBD
from .models import ArchivoMedia, Imagen

EXT_VIDEO = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
EXT_FOTO = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EXT_HEIC = {".heic", ".heif"}


def relativa(ruta: Path) -> str:
    """Ruta relativa a datos/ con barras '/', que es como se guarda en la BD y se sirve por HTTP."""
    return ruta.relative_to(DATOS).as_posix()


def nombre_seguro(nombre: str) -> str:
    # Solo ASCII sin espacios: OpenCV en Windows falla con rutas que tienen tildes o ñ.
    base = unicodedata.normalize("NFKD", Path(nombre).name).encode("ascii", "ignore").decode()
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "archivo"
    return f"{uuid.uuid4().hex[:8]}_{base}"


def guardar_subida(archivo_subido, carpeta: Path) -> Path:
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / nombre_seguro(archivo_subido.filename)
    with destino.open("wb") as f:
        shutil.copyfileobj(archivo_subido.file, f, length=1024 * 1024)
    return destino


def borrar_archivo(ruta_relativa: str):
    (DATOS / ruta_relativa).unlink(missing_ok=True)


def procesar_foto(archivo: ArchivoMedia, codigo_sesion: str) -> Imagen:
    """Convierte una foto subida en una imagen del dataset (JPG, ancho máximo 1280 px)."""
    datos = np.fromfile(str(DATOS / archivo.ruta), dtype=np.uint8)
    foto = cv2.imdecode(datos, cv2.IMREAD_COLOR)  # aplica la orientación EXIF del iPhone
    if foto is None:
        raise ValueError("No se pudo leer la imagen")
    alto, ancho = foto.shape[:2]
    if ancho > ANCHO_MAXIMO:
        foto = cv2.resize(foto, (ANCHO_MAXIMO, round(alto * ANCHO_MAXIMO / ancho)),
                          interpolation=cv2.INTER_AREA)
    carpeta = IMAGENES / codigo_sesion
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / f"{codigo_sesion}_a{archivo.id}_foto.jpg"
    cv2.imwrite(str(destino), foto, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return Imagen(sesion_id=archivo.sesion_id, archivo_id=archivo.id, ruta=relativa(destino),
                  ancho=foto.shape[1], alto=foto.shape[0])


def extraer_video(archivo_id: int, parametros: dict | None = None):
    """Tarea en segundo plano: extrae los fotogramas de un video y los registra en la BD.

    Si el video ya se había procesado, sus imágenes anteriores (y sus etiquetas)
    se reemplazan.
    """
    parametros = {**PARAMETROS, **(parametros or {})}
    with SesionBD() as bd:
        archivo = bd.get(ArchivoMedia, archivo_id)
        if archivo is None:
            return
        archivo.estado, archivo.detalle = "procesando", None
        for img in list(archivo.imagenes):
            borrar_archivo(img.ruta)
            bd.delete(img)
        bd.commit()

        codigo = archivo.sesion.codigo
        try:
            r = procesar_video(DATOS / archivo.ruta, IMAGENES / codigo,
                               f"{codigo}_a{archivo.id}", **parametros)
            if r["guardados"]:
                # Todos los fotogramas de un video tienen el mismo tamaño.
                alto, ancho = cv2.imread(str(r["guardados"][0])).shape[:2]
            for ruta in r["guardados"]:
                bd.add(Imagen(sesion_id=archivo.sesion_id, archivo_id=archivo.id,
                              ruta=relativa(ruta), ancho=ancho, alto=alto))
            archivo.estado = "procesado"
            archivo.detalle = (f"{len(r['guardados'])} fotogramas guardados; descartados: "
                               f"{r['movimiento']} en movimiento, {r['desenfocados']} desenfocados, "
                               f"{r['repetidos']} repetidos")
        except Exception as e:  # el error queda visible en la interfaz
            archivo.estado, archivo.detalle = "error", str(e)[:300]
        bd.commit()
