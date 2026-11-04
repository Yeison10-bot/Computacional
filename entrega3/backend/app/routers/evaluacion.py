"""Videos de prueba con sus eventos reales y evaluación del sistema de retiros.

Cada video vive en datos/videos_prueba/<id>/ con:
  video.<ext>  → el archivo subido
  meta.json    → nombre, eventos reales marcados por el usuario y último resultado
"""

import json
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..config import DATOS
from ..evaluacion import analizar_video, comparar
from ..monitor import cargar_parametros
from ..vision import obtener_detector
from . import entrenamiento

router = APIRouter(prefix="/api/evaluacion", tags=["Evaluación de retiros"])

VIDEOS = DATOS / "videos_prueba"
EXTENSIONES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

_candado = threading.Lock()
_en_curso = {}   # id → {"progreso": 0-1}


class EventoReal(BaseModel):
    t: float = Field(ge=0, description="segundo del video en que ocurre")
    categoria: str
    tipo: str = Field("venta", pattern="^(venta|reposicion)$")
    cantidad: int = Field(1, ge=1, le=20)
    nota: str = ""


class Eventos(BaseModel):
    eventos: list[EventoReal]
    tolerancia_s: float = Field(15, ge=2, le=60)


def carpeta(id_: str) -> Path:
    if not id_ or any(c in id_ for c in "/\\.") or not (VIDEOS / id_ / "meta.json").exists():
        raise HTTPException(404, "No existe ese video de prueba")
    return VIDEOS / id_


def leer(id_: str) -> dict:
    meta = json.loads((carpeta(id_) / "meta.json").read_text(encoding="utf-8"))
    meta["id"] = id_
    meta["url"] = f"/archivos/videos_prueba/{id_}/{meta['archivo']}"
    meta["evaluando"] = id_ in _en_curso
    meta["progreso"] = _en_curso.get(id_, {}).get("progreso")
    return meta


def guardar(id_: str, meta: dict):
    meta = {k: v for k, v in meta.items() if k not in ("id", "url", "evaluando", "progreso")}
    destino = VIDEOS / id_ / "meta.json"
    temporal = destino.with_suffix(".tmp")
    temporal.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    temporal.replace(destino)


@router.get("/videos")
def listar():
    if not VIDEOS.exists():
        return []
    salida = []
    for c in sorted(VIDEOS.iterdir(), reverse=True):
        if (c / "meta.json").exists():
            m = leer(c.name)
            if m.get("resultado"):
                m["resultado"] = {k: v for k, v in m["resultado"].items() if k not in ("filas", "eventos_detectados")}
            salida.append(m)
    return salida


@router.post("/videos")
def subir(archivo: UploadFile = File(...), nombre: str = Form("")):
    extension = Path(archivo.filename or "").suffix.lower()
    if extension not in EXTENSIONES:
        raise HTTPException(400, f"Formato no admitido ({extension or 'sin extensión'}): usa MP4 o MOV")
    id_ = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4]
    destino = VIDEOS / id_
    destino.mkdir(parents=True)
    with open(destino / f"video{extension}", "wb") as f:
        shutil.copyfileobj(archivo.file, f, length=1024 * 1024)
    captura = cv2_info(destino / f"video{extension}")
    guardar(id_, {"nombre": nombre.strip() or Path(archivo.filename).stem, "archivo": f"video{extension}",
                  "nombre_original": archivo.filename, "subido_en": datetime.now().astimezone().isoformat(),
                  **captura, "eventos_reales": [], "tolerancia_s": 15, "resultado": None})
    return leer(id_)


def cv2_info(ruta):
    import cv2
    c = cv2.VideoCapture(str(ruta))
    try:
        if not c.isOpened():
            shutil.rmtree(ruta.parent, ignore_errors=True)
            raise HTTPException(400, "No se pudo leer el video (si es HEVC del iPhone, grábalo en «Más compatible»)")
        fps = c.get(cv2.CAP_PROP_FPS) or 30
        cuadros = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        return {"duracion_s": round(cuadros / fps, 1), "fps": round(fps, 1),
                "resolucion": f"{int(c.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))}"}
    finally:
        c.release()


@router.get("/videos/{id_}")
def detalle(id_: str):
    return leer(id_)


@router.put("/videos/{id_}/eventos")
def guardar_eventos(id_: str, datos: Eventos):
    meta = leer(id_)
    meta["eventos_reales"] = sorted((e.model_dump() for e in datos.eventos), key=lambda e: e["t"])
    meta["tolerancia_s"] = datos.tolerancia_s
    # Si ya había un análisis, se vuelve a comparar con los eventos corregidos (sin reprocesar el video).
    if meta.get("resultado"):
        r = meta["resultado"]
        meta["resultado"] = {**r, **comparar(meta["eventos_reales"], r["eventos_detectados"], r["categorias"],
                                             r["inicio_s"], datos.tolerancia_s)}
    guardar(id_, meta)
    return leer(id_)


@router.delete("/videos/{id_}", status_code=204)
def borrar(id_: str):
    if id_ in _en_curso:
        raise HTTPException(409, "Ese video se está evaluando")
    shutil.rmtree(carpeta(id_))


@router.post("/videos/{id_}/evaluar")
def evaluar(id_: str):
    meta = leer(id_)
    if entrenamiento.en_curso():
        raise HTTPException(409, "Hay un entrenamiento en curso: la GPU está ocupada")
    with _candado:
        if _en_curso:
            raise HTTPException(409, "Ya se está evaluando otro video; espera a que termine")
        _en_curso[id_] = {"progreso": 0.0}

    def trabajo():
        try:
            detector = obtener_detector()
            analisis = analizar_video(carpeta(id_) / meta["archivo"], detector, cargar_parametros(),
                                      lambda f: _en_curso[id_].update(progreso=round(f, 3)))
            actual = leer(id_)  # los eventos reales pudieron cambiar mientras tanto
            actual["resultado"] = {
                **{k: v for k, v in analisis.items() if k != "eventos"},
                "eventos_detectados": analisis["eventos"], "modelo": detector.info(),
                "evaluado_en": datetime.now().astimezone().isoformat(), "error": None,
                **comparar(actual["eventos_reales"], analisis["eventos"], analisis["categorias"],
                           analisis["inicio_s"], actual.get("tolerancia_s", 15)),
            }
            guardar(id_, actual)
        except Exception as e:
            actual = leer(id_)
            actual["resultado"] = {"error": f"{type(e).__name__}: {e}",
                                   "evaluado_en": datetime.now().astimezone().isoformat()}
            guardar(id_, actual)
        finally:
            _en_curso.pop(id_, None)

    threading.Thread(target=trabajo, daemon=True).start()
    return leer(id_)


@router.get("/resumen")
def resumen():
    """Totales de todos los videos evaluados: las métricas del sistema para el informe."""
    vp = fn = fp = dup = 0
    latencias, videos = [], 0
    for m in listar():
        r = m.get("resultado") or {}
        if r.get("error") or "aciertos" not in r:
            continue
        videos += 1
        vp, fn, fp, dup = vp + r["aciertos"], fn + r["omitidos"], fp + r["falsos_positivos"], dup + r["duplicados"]
        if r.get("latencia_media_s") is not None:
            latencias += [r["latencia_media_s"]] * r["aciertos"]
    return {
        "videos": videos, "aciertos": vp, "omitidos": fn, "falsos_positivos": fp, "duplicados": dup,
        "precision": round(vp / (vp + fp), 4) if vp + fp else None,
        "recall": round(vp / (vp + fn), 4) if vp + fn else None,
        "f1": round(2 * vp / (2 * vp + fp + fn), 4) if vp + fp + fn else None,
        "latencia_media_s": round(sum(latencias) / len(latencias), 2) if latencias else None,
        "parametros_actuales": cargar_parametros(),
    }
