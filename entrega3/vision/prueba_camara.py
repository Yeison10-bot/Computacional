"""Prueba del flujo cámara → OpenCV → YOLO con un modelo preentrenado (COCO).

Sirve para comprobar, ANTES de entrenar nuestro modelo, que:
  1. OpenCV abre la fuente de video (webcam, iPhone con Iriun o archivo).
  2. YOLO corre sobre la GPU y dibuja las detecciones.
  3. La velocidad (FPS) es suficiente para tiempo real.

El modelo COCO ya conoce clases como "bottle", "cup", "banana", "apple",
así que varios de nuestros objetos deberían detectarse aunque no esté entrenado
para nuestras categorías.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python.exe vision\\prueba_camara.py                 # cámara 0
    .venv\\Scripts\\python.exe vision\\prueba_camara.py --fuente 1      # otra cámara (p. ej. Iriun)
    .venv\\Scripts\\python.exe vision\\prueba_camara.py --fuente video.mp4
    .venv\\Scripts\\python.exe vision\\prueba_camara.py --listar        # qué índices de cámara existen
Tecla Q para salir.
"""

import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent
MODELO_BASE = RAIZ / "vision" / "modelos" / "yolo11n.pt"


def abrir_fuente(fuente):
    # Un número es un índice de cámara; cualquier otra cosa es ruta o URL.
    # Se usa el backend por defecto (Media Foundation): en OpenCV 5 DirectShow
    # ya no permite abrir cámaras por índice.
    if fuente.isdigit():
        return cv2.VideoCapture(int(fuente))
    return cv2.VideoCapture(fuente)


def listar_camaras(maximo=5):
    # OpenCV imprime errores al probar índices sin cámara; aquí son esperables.
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    for i in range(maximo):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if ok:
            print(f"  Cámara {i}: disponible ({frame.shape[1]}x{frame.shape[0]})")
        cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fuente", default="0", help="índice de cámara, ruta de video o URL")
    parser.add_argument("--conf", type=float, default=0.4, help="confianza mínima")
    parser.add_argument("--listar", action="store_true", help="lista las cámaras disponibles")
    args = parser.parse_args()

    if args.listar:
        listar_camaras()
        return

    # Si el archivo no existe, Ultralytics lo descarga (≈ 5 MB) en esa ruta.
    modelo = YOLO(str(MODELO_BASE))
    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    print(f"Inferencia en: {'GPU' if dispositivo == 0 else 'CPU'}")
    cap = abrir_fuente(args.fuente)
    if not cap.isOpened():
        print(f"No se pudo abrir la fuente: {args.fuente}")
        return

    anterior = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break  # fin del video o cámara desconectada

        resultado = modelo(frame, conf=args.conf, device=dispositivo, verbose=False)[0]
        anotado = resultado.plot()  # dibuja cajas, clase y confianza

        ahora = time.perf_counter()
        fps = 1 / (ahora - anterior)
        anterior = ahora
        cv2.putText(anotado, f"FPS: {fps:.1f}  objetos: {len(resultado.boxes)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Prueba YOLO - Q para salir", anotado)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
