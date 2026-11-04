"""Acceso al detector YOLO desde el backend.

El modelo se carga la primera vez que se necesita (no al arrancar la API), porque
importar PyTorch y cargar los pesos tarda unos segundos.
"""

import threading

_detector = None
_candado = threading.Lock()


def obtener_detector():
    global _detector
    with _candado:
        if _detector is None:
            from vision.detector import Detector  # importación diferida: carga PyTorch
            _detector = Detector()
        return _detector


def recargar_detector():
    """Descarta el modelo cargado; el siguiente uso carga el más reciente (p. ej. un best.pt nuevo)."""
    global _detector
    with _candado:
        _detector = None
