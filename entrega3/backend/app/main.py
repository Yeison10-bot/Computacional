"""Punto de entrada de la API.

Ejecutar desde la raíz del proyecto:
    .venv\\Scripts\\python.exe -m uvicorn backend.app.main:app --reload

Documentación interactiva: http://localhost:8000/docs
"""

import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import DATOS
from .monitor import monitor
from .routers import catalogo, dataset, entrenamiento, estadisticas, evaluacion, exportar, inventario
from .routers import monitor as rutas_monitor
from .vision import obtener_detector


@asynccontextmanager
async def ciclo_de_vida(_app):
    # El modelo YOLO se carga en segundo plano al arrancar: así el primer
    # monitoreo o conteo no tiene que esperar a PyTorch y a la GPU.
    threading.Thread(target=obtener_detector, daemon=True).start()
    # Con MONITOR_AUTOINICIO=true en el .env, la cámara empieza a vigilar la
    # vitrina apenas arranca el servidor, sin entrar a la interfaz.
    if os.getenv("MONITOR_AUTOINICIO", "false").lower() == "true":
        try:
            monitor.iniciar(os.getenv("VIDEO_FUENTE", "0"))
        except (ValueError, RuntimeError) as e:
            print(f"No se pudo iniciar el monitoreo automático: {e}")
    yield
    monitor.detener()  # libera la cámara al apagar el servidor


app = FastAPI(
    title="Inventario con visión computacional",
    description="API del proyecto de Inteligencia Computacional II — Tercer corte",
    version="0.2.0",
    lifespan=ciclo_de_vida,
)

# El frontend en desarrollo corre en otro puerto (Vite, 5180).
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5180"],
                   allow_methods=["*"], allow_headers=["*"])

app.include_router(catalogo.router)
app.include_router(inventario.router)
app.include_router(rutas_monitor.router)
app.include_router(dataset.router)
app.include_router(exportar.router)
app.include_router(entrenamiento.router)
app.include_router(estadisticas.router)
app.include_router(evaluacion.router)

# Sirve las imágenes y videos guardados en datos/ (p. ej. /archivos/imagenes/S01/...).
DATOS.mkdir(exist_ok=True)
app.mount("/archivos", StaticFiles(directory=DATOS), name="archivos")


@app.get("/api/salud", tags=["Sistema"])
def salud():
    return {"estado": "ok"}
