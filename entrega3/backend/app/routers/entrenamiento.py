"""Entrenamiento del modelo desde la interfaz.

El entrenamiento corre en un proceso aparte (vision/entrenamiento/entrenar.py):
tarda minutos u horas y ocupa la GPU, y así la API sigue respondiendo. Cada
entrenamiento tiene su carpeta en datos/entrenamientos/<id>/ con:
  progreso.json  → avance por época y resultado final (lo escribe el script)
  registro.txt   → salida de consola del script, para diagnosticar errores
  train/, test/  → pesos, curvas y matrices de confusión de Ultralytics
"""

import json
import os
import subprocess
import sys
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import DATASET, DATOS, RAIZ
from ..monitor import monitor
from ..vision import recargar_detector

router = APIRouter(prefix="/api/entrenamiento", tags=["Entrenamiento"])

ENTRENAMIENTOS = DATOS / "entrenamientos"
SCRIPT = RAIZ / "vision" / "entrenamiento" / "entrenar.py"
MODELO_PROPIO = RAIZ / "vision" / "modelos" / "best.pt"
FINALES = ("terminado", "error", "detenido")

_proceso = None          # subprocess.Popen del entrenamiento en curso
_proceso_id = None       # id (carpeta) de ese entrenamiento
_recargados = set()      # entrenamientos cuyo modelo ya se cargó en la API


class Parametros(BaseModel):
    epocas: int = Field(50, ge=1, le=300)
    imgsz: int = Field(640, ge=320, le=1280, multiple_of=32)
    lote: int = Field(8, ge=1, le=64)
    paciencia: int = Field(15, ge=0, le=100)
    base: str = Field("coco", pattern="^(coco|propio)$")


def leer(id_: str):
    try:
        return json.loads((ENTRENAMIENTOS / id_ / "progreso.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def en_curso():
    return _proceso is not None and _proceso.poll() is None


def resumen(id_: str):
    datos = leer(id_)
    if datos is None:
        return None
    if id_ == _proceso_id and _proceso is not None and _proceso.poll() is not None and datos["fase"] not in FINALES:
        # El proceso murió sin escribir su estado final (cerrado a la fuerza o error grave).
        datos.update(fase="error", error="El proceso de entrenamiento se cerró inesperadamente; revisa registro.txt.")
    elif id_ != _proceso_id and datos["fase"] not in FINALES:
        # Quedó a medias de una ejecución anterior de la API (p. ej. se reinició el servidor).
        datos.update(fase="interrumpido", mensaje="Este entrenamiento no terminó (se reinició el servidor).")
    # Un modelo nuevo terminado: la API lo empieza a usar (una sola vez).
    if datos["fase"] == "terminado" and id_ not in _recargados:
        _recargados.add(id_)
        recargar_detector()
    datos["id"] = id_
    datos["activo"] = id_ == _proceso_id and en_curso()
    base = f"/archivos/entrenamientos/{id_}"
    datos["graficas"] = [{"nombre": g, "url": f"{base}/{g}"} for g in datos.get("graficas", [])]
    return datos


@router.get("")
def listar():
    """Entrenamientos realizados, del más reciente al más antiguo (sin el historial por época)."""
    if not ENTRENAMIENTOS.exists():
        return []
    salida = []
    for carpeta in sorted(ENTRENAMIENTOS.iterdir(), reverse=True):
        datos = resumen(carpeta.name) if carpeta.is_dir() else None
        if datos:
            datos.pop("historial", None)
            salida.append(datos)
    return salida


@router.get("/requisitos")
def requisitos():
    """Lo que hace falta antes de poder entrenar."""
    def imagenes(p):
        carpeta = DATASET / "images" / p
        extensiones = (".jpg", ".jpeg", ".png")  # sin contar el .gitkeep
        return sum(1 for a in carpeta.iterdir() if a.suffix.lower() in extensiones) if carpeta.exists() else 0
    return {
        "dataset_exportado": (DATASET / "data.yaml").exists(),
        "imagenes": {p: imagenes(p) for p in ("train", "val", "test")},
        "modelo_propio": MODELO_PROPIO.exists(),
        "monitor_activo": monitor.activo,
        "entrenando": en_curso(),
    }


@router.get("/{id_}")
def detalle(id_: str):
    if "/" in id_ or "\\" in id_ or ".." in id_:
        raise HTTPException(400, "Identificador inválido")
    datos = resumen(id_)
    if datos is None:
        raise HTTPException(404, "No existe ese entrenamiento")
    return datos


@router.post("/iniciar")
def iniciar(p: Parametros):
    global _proceso, _proceso_id
    if en_curso():
        raise HTTPException(409, "Ya hay un entrenamiento en curso")
    if monitor.activo:
        # La GPU de 4 GB no alcanza para entrenar y vigilar la vitrina a la vez.
        raise HTTPException(409, "Detén el monitoreo de la vitrina antes de entrenar (comparten la GPU)")
    if not (DATASET / "data.yaml").exists():
        raise HTTPException(400, "Primero exporta el dataset en «Exportar dataset»")
    if p.base == "propio" and not MODELO_PROPIO.exists():
        raise HTTPException(400, "Todavía no hay un modelo propio del cual continuar")

    id_ = datetime.now().strftime("%Y%m%d_%H%M%S")
    while (ENTRENAMIENTOS / id_).exists():  # dos inicios en el mismo segundo
        id_ += "b"
    carpeta = ENTRENAMIENTOS / id_
    carpeta.mkdir(parents=True)
    # Estado inicial escrito aquí: el script tarda unos segundos en arrancar y la
    # interfaz consulta el avance de inmediato.
    (carpeta / "progreso.json").write_text(json.dumps({
        "fase": "preparando", "mensaje": "Iniciando el proceso de entrenamiento…", "parametros": p.model_dump(),
        "iniciado_en": datetime.now().astimezone().isoformat(), "epoca": 0, "epocas": p.epocas,
        "historial": [], "graficas": [], "resultado": None, "error": None}, ensure_ascii=False), encoding="utf-8")
    comando = [sys.executable, str(SCRIPT), "--data", str(DATASET / "data.yaml"), "--salida", str(carpeta), "--epocas", str(p.epocas),
               "--imgsz", str(p.imgsz), "--lote", str(p.lote), "--paciencia", str(p.paciencia), "--base", p.base]
    registro = open(carpeta / "registro.txt", "w", encoding="utf-8")
    entorno = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    opciones = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}
    _proceso = subprocess.Popen(comando, cwd=RAIZ, stdout=registro, stderr=subprocess.STDOUT, env=entorno, **opciones)
    registro.close()  # el proceso hijo conserva su propia copia del archivo
    _proceso_id = id_
    return {"id": id_}


@router.post("/detener")
def detener():
    if not en_curso():
        raise HTTPException(409, "No hay un entrenamiento en curso")
    if os.name == "nt":
        # /T cierra también los procesos hijos (los workers que cargan las imágenes).
        subprocess.run(["taskkill", "/PID", str(_proceso.pid), "/T", "/F"], capture_output=True)
    else:
        _proceso.terminate()
    _proceso.wait(timeout=15)
    archivo = ENTRENAMIENTOS / _proceso_id / "progreso.json"
    datos = leer(_proceso_id) or {}
    datos.update(fase="detenido", mensaje="Detenido por el usuario. El modelo en uso no cambió.",
                 terminado_en=datetime.now().astimezone().isoformat())
    archivo.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    return resumen(_proceso_id)
