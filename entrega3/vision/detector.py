"""Detector de productos con YOLO, compartido por el backend y el módulo de video.

Usa el modelo propio entrenado (vision/modelos/best.pt) si existe. Si todavía no
se ha entrenado, usa el modelo preentrenado en COCO como PROVISIONAL, traduciendo
las clases de COCO que equivalen a nuestras categorías. COCO no tiene latas ni
paquetes de snacks, así que con el modelo provisional esas categorías no se
pueden contar: `categorias_detectables` lo informa para que nadie interprete
"no detecté latas" como "hay 0 latas".
"""

import threading
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

MODELOS = Path(__file__).resolve().parent / "modelos"
MODELO_PROPIO = MODELOS / "best.pt"
MODELO_BASE = MODELOS / "yolo11n.pt"

# Clase de COCO → categoría del proyecto (solo para el modelo provisional).
MAPEO_COCO = {
    "bottle": "botella",
    "cup": "vaso",
    "wine glass": "vaso",
    "banana": "fruta",
    "apple": "fruta",
    "orange": "fruta",
}


class Detector:
    def __init__(self, ruta_modelo=None, confianza=0.4):
        ruta = Path(ruta_modelo) if ruta_modelo else (MODELO_PROPIO if MODELO_PROPIO.exists() else MODELO_BASE)
        self.ruta = ruta
        self.propio = ruta.resolve() != MODELO_BASE.resolve()
        self.modelo = YOLO(str(ruta))
        self.confianza = confianza
        self.dispositivo = 0 if torch.cuda.is_available() else "cpu"
        # El modelo de PyTorch no es seguro para usarlo desde varios hilos a la vez
        # (FastAPI atiende peticiones en paralelo): se serializa la inferencia.
        self._candado = threading.Lock()
        # Calentamiento: la primera inferencia inicializa CUDA y tarda segundos;
        # se hace aquí para que no se mida como tiempo de respuesta del sistema.
        self.modelo(np.zeros((640, 640, 3), dtype=np.uint8), device=self.dispositivo, verbose=False)

    def categoria(self, nombre_clase):
        return nombre_clase if self.propio else MAPEO_COCO.get(nombre_clase)

    @property
    def categorias_detectables(self):
        nombres = self.modelo.names.values()
        return sorted({c for n in nombres if (c := self.categoria(n))})

    def info(self):
        return {
            "modelo": self.ruta.name,
            "tipo": "propio" if self.propio else "provisional (COCO)",
            "categorias_detectables": self.categorias_detectables,
            "dispositivo": "GPU" if self.dispositivo == 0 else "CPU",
            "confianza_minima": self.confianza,
        }

    def detectar(self, imagen_bgr):
        """Devuelve las detecciones de nuestras categorías, con la caja normalizada (0 a 1)."""
        with self._candado:
            r = self.modelo(imagen_bgr, conf=self.confianza, device=self.dispositivo, verbose=False)[0]
        detecciones = []
        for clase, conf, (xc, yc, w, h) in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                               r.boxes.xywhn.tolist()):
            nombre = self.modelo.names[int(clase)]
            categoria = self.categoria(nombre)
            if categoria is None:
                continue  # objeto que no es un producto (persona, silla, ...)
            detecciones.append({"categoria": categoria, "clase_original": nombre, "confianza": conf,
                                "x_centro": xc, "y_centro": yc, "ancho": w, "alto": h})
        return detecciones

    @staticmethod
    def contar(detecciones):
        return Counter(d["categoria"] for d in detecciones)
