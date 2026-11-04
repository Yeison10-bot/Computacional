"""Control del monitoreo continuo y transmisión del video anotado."""

import asyncio
import os

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..monitor import monitor
from . import entrenamiento

router = APIRouter(prefix="/api/monitor", tags=["Monitoreo continuo"])


def camara_servidor_permitida():
    """En el servidor desplegado no hay cámara conectada: solo se permite la del
    dispositivo que abre la página (fuente "navegador"). Además así nadie puede
    pedirle al servidor que abra archivos arbitrarios como "video"."""
    return os.getenv("CAMARA_SERVIDOR", "true").lower() == "true"


class Iniciar(BaseModel):
    fuente: str = Field("0", min_length=1,
                        description='índice de cámara del servidor (0, 1, ...), ruta de un video o "navegador"')


@router.get("/estado")
def estado():
    return monitor.estado()


@router.post("/iniciar")
def iniciar(datos: Iniciar):
    if entrenamiento.en_curso():
        raise HTTPException(409, "Hay un entrenamiento en curso: la GPU está ocupada")
    if datos.fuente.strip() != "navegador" and not camara_servidor_permitida():
        raise HTTPException(400, "En este servidor solo se puede usar la cámara del dispositivo que abre la página")
    try:
        monitor.iniciar(datos.fuente.strip().strip('"'))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return monitor.estado()


@router.post("/fotograma")
async def fotograma(request: Request):
    """Recibe un fotograma JPEG de la cámara del navegador (cuerpo binario image/jpeg).

    Responde con las detecciones del último fotograma analizado, para que la
    página las dibuje sobre su propio video en vivo.
    """
    datos = np.frombuffer(await request.body(), dtype=np.uint8)
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR) if datos.size else None
    if imagen is None:
        raise HTTPException(400, "El fotograma no es una imagen JPEG válida")
    try:
        monitor.recibir_fotograma(imagen)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    e = monitor.estado()
    return {"fase": e["fase"], "detecciones": e["detecciones"], "analizados": e["fotogramas"]}


class Ajustes(BaseModel):
    fps_analisis: float | None = Field(None, ge=1, le=10)
    confirmacion_s: float | None = Field(None, ge=1, le=15)
    quietud_s: float | None = Field(None, ge=0.5, le=5)
    ventana_s: float | None = Field(None, ge=0.6, le=5)


@router.patch("/parametros")
def ajustar(datos: Ajustes):
    """Cambia la velocidad de análisis y los tiempos de confirmación (se aplican al instante)."""
    return monitor.ajustar(datos.model_dump(exclude_none=True))


@router.post("/detener")
def detener():
    monitor.detener()
    return monitor.estado()


@router.get("/camaras")
def camaras():
    """Cámaras conectadas (índices 0 a 4). Con el monitoreo activo no se prueban:
    abrir la misma cámara dos veces falla en Windows."""
    if not camara_servidor_permitida():
        return []
    if monitor.activo:
        raise HTTPException(409, "Detén el monitoreo para buscar cámaras")
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    encontradas = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        ok, fotograma = cap.read() if cap.isOpened() else (False, None)
        if ok:
            encontradas.append({"indice": i, "resolucion": f"{fotograma.shape[1]}x{fotograma.shape[0]}"})
        cap.release()
    return encontradas


@router.get("/video")
async def video(request: Request):
    """Video en vivo como MJPEG (multipart/x-mixed-replace): el navegador lo muestra en un <img>."""
    async def fotogramas():
        anterior = None
        # La transmisión termina al detenerse el monitoreo: una conexión abierta
        # para siempre impediría que el servidor se reinicie o se apague.
        while monitor.activo and not await request.is_disconnected():
            jpeg = monitor.jpeg
            if jpeg is not None and jpeg is not anterior:
                anterior = jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(0.05)

    return StreamingResponse(fotogramas(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})
