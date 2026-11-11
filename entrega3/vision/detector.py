"""Detector de productos con YOLO, compartido por el backend y el módulo de video.

Usa el modelo propio entrenado (vision/modelos/best.pt) si existe. Si todavía no
se ha entrenado, usa el modelo preentrenado en COCO como PROVISIONAL, traduciendo
las clases de COCO que equivalen a nuestras categorías. COCO no tiene latas ni
paquetes de snacks, así que con el modelo provisional esas categorías no se
pueden contar: `categorias_detectables` lo informa para que nadie interprete
"no detecté latas" como "hay 0 latas".

Tres reglas para que el conteo sea estable:

  1. NMS agnóstico a la clase (solo con el modelo propio): un mismo objeto no
     puede salir con dos cajas de clases distintas; queda la más confiable.
  2. Umbral de confianza por categoría: el entrenamiento guarda en best.json el
     umbral que mejor le fue a cada clase en validación. Sin ese archivo se usa
     el mismo umbral para todas.
  3. Histéresis: para APARECER, un producto debe superar el umbral de su clase;
     para SEGUIR ahí le basta una fracción de ese umbral, si coincide con una
     detección reciente de la misma categoría en el mismo sitio. Así un producto
     cuya confianza ronda el umbral no aparece y desaparece (ventas y
     reposiciones falsas en cadena).
"""

import hashlib
import json
import threading
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

MODELOS = Path(__file__).resolve().parent / "modelos"
MODELO_PROPIO = MODELOS / "best.pt"
MODELO_BASE = MODELOS / "yolo11n.pt"

FACTOR_PERMANENCIA = 0.6  # fracción del umbral que basta para conservar un producto ya visto
IOU_MISMO_OBJETO = 0.5    # solapamiento mínimo con la detección anterior para ser "el mismo"

# Clase de COCO → categoría del proyecto (solo para el modelo provisional).
MAPEO_COCO = {
    "bottle": "botella",
    "cup": "vaso",
    "wine glass": "vaso",
    "banana": "fruta",
    "apple": "fruta",
    "orange": "fruta",
}


def huella(ruta):
    """SHA-256 del archivo de pesos: liga best.json con el best.pt del que salió."""
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def iou(a, b):
    """Solapamiento (0 a 1) entre dos detecciones con cajas normalizadas."""
    ax1, ay1 = a["x_centro"] - a["ancho"] / 2, a["y_centro"] - a["alto"] / 2
    ax2, ay2 = a["x_centro"] + a["ancho"] / 2, a["y_centro"] + a["alto"] / 2
    bx1, by1 = b["x_centro"] - b["ancho"] / 2, b["y_centro"] - b["alto"] / 2
    bx2, by2 = b["x_centro"] + b["ancho"] / 2, b["y_centro"] + b["alto"] / 2
    comun = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = a["ancho"] * a["alto"] + b["ancho"] * b["alto"] - comun
    return comun / union if union > 0 else 0.0


class MemoriaDetecciones:
    """Detecciones aceptadas en los últimos instantes, para la histéresis.

    Guarda un poco más que el fotograma anterior: si un producto se pierde un
    fotograma suelto, al volver todavía cuenta como "ya estaba".
    """

    def __init__(self, duracion_s=1.0):
        self.duracion_s = duracion_s
        self._historia = deque()  # (tiempo, detecciones)

    def previas(self, ahora):
        while self._historia and ahora - self._historia[0][0] > self.duracion_s:
            self._historia.popleft()
        return [d for _, detecciones in self._historia for d in detecciones]

    def actualizar(self, ahora, detecciones):
        self._historia.append((ahora, detecciones))


class Detector:
    def __init__(self, ruta_modelo=None, confianza=0.4):
        ruta = Path(ruta_modelo) if ruta_modelo else (MODELO_PROPIO if MODELO_PROPIO.exists() else MODELO_BASE)
        self.ruta = ruta
        self.propio = ruta.resolve() != MODELO_BASE.resolve()
        self.modelo = YOLO(str(ruta))
        self.confianza = confianza
        self.umbrales = self._cargar_umbrales()
        # El modelo se consulta con el umbral más bajo que puede hacer falta (el de
        # permanencia); después cada detección se filtra con el de su categoría.
        self._piso = min([self.umbral(c) for c in self.categorias_detectables] or [confianza]) * FACTOR_PERMANENCIA
        self.dispositivo = 0 if torch.cuda.is_available() else "cpu"
        # El modelo de PyTorch no es seguro para usarlo desde varios hilos a la vez
        # (FastAPI atiende peticiones en paralelo): se serializa la inferencia.
        self._candado = threading.Lock()
        # Calentamiento: la primera inferencia inicializa CUDA y tarda segundos;
        # se hace aquí para que no se mida como tiempo de respuesta del sistema.
        self.modelo(np.zeros((640, 640, 3), dtype=np.uint8), device=self.dispositivo, verbose=False)

    def _cargar_umbrales(self):
        """Umbrales por categoría de best.json, solo si corresponden a ESTOS pesos
        (si se copió un best.pt nuevo sin su .json, los umbrales viejos no valen)."""
        try:
            datos = json.loads(self.ruta.with_suffix(".json").read_text(encoding="utf-8"))
            if datos.get("sha256") != huella(self.ruta):
                return {}
            conocidas = set(self.modelo.names.values())
            return {c: float(u) for c, u in datos.get("umbrales", {}).items() if c in conocidas}
        except (OSError, ValueError, TypeError, AttributeError):
            return {}

    def categoria(self, nombre_clase):
        return nombre_clase if self.propio else MAPEO_COCO.get(nombre_clase)

    def umbral(self, categoria):
        """Confianza mínima para que un producto de esta categoría aparezca."""
        return self.umbrales.get(categoria, self.confianza)

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
            "umbrales": {c: round(self.umbral(c), 2) for c in self.categorias_detectables},
            "factor_permanencia": FACTOR_PERMANENCIA,
        }

    def detectar(self, imagen_bgr, previas=None):
        """Devuelve las detecciones de nuestras categorías, con la caja normalizada (0 a 1).

        `previas`: detecciones aceptadas hace un instante (MemoriaDetecciones). Con
        ellas se aplica la histéresis; sin ellas (una foto suelta) solo cuenta el
        umbral de cada categoría. Las conservadas por histéresis llevan
        "sostenida": True.
        """
        with self._candado:
            r = self.modelo(imagen_bgr, conf=self._piso, agnostic_nms=self.propio,
                            device=self.dispositivo, verbose=False)[0]
        detecciones = []
        for clase, conf, (xc, yc, w, h) in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                               r.boxes.xywhn.tolist()):
            nombre = self.modelo.names[int(clase)]
            categoria = self.categoria(nombre)
            if categoria is None:
                continue  # objeto que no es un producto (persona, silla, ...)
            d = {"categoria": categoria, "clase_original": nombre, "confianza": conf,
                 "x_centro": xc, "y_centro": yc, "ancho": w, "alto": h, "sostenida": False}
            umbral = self.umbral(categoria)
            if conf < umbral:
                ya_estaba = conf >= umbral * FACTOR_PERMANENCIA and any(
                    p["categoria"] == categoria and iou(d, p) >= IOU_MISMO_OBJETO for p in previas or ())
                if not ya_estaba:
                    continue
                d["sostenida"] = True
            detecciones.append(d)
        return detecciones

    @staticmethod
    def contar(detecciones):
        return Counter(d["categoria"] for d in detecciones)
