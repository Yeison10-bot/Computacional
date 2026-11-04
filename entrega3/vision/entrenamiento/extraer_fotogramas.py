"""Extrae fotogramas útiles de los videos de captura para construir el dataset.

Recorre cada video y evalúa un fotograma cada `intervalo` segundos. Lo guarda
solo si aporta información nueva y nítida:
  - En movimiento: se compara con el fotograma INMEDIATAMENTE anterior del
    video (diferencia absoluta media). Si cambió mucho, algo se está moviendo
    (la mano, un objeto, la cámara) y la imagen sale movida o con la mano
    tapando productos. Este criterio no depende de cuántos objetos haya en la
    escena, a diferencia de una medida de nitidez.
  - Desenfocado: varianza del Laplaciano muy baja (casi sin bordes).
  - Repetido: si la diferencia con el último fotograma GUARDADO es pequeña, la
    escena no cambió y la imagen sería un duplicado.

La función `procesar_video` la usa también el backend cuando se sube un video
desde la interfaz web.

Uso por consola (desde la raíz del proyecto):
    .venv\\Scripts\\python.exe vision\\entrenamiento\\extraer_fotogramas.py datos\\crudos\\videos\\S01

Los fotogramas quedan en datos\\imagenes\\<nombre de la carpeta>\\.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
SALIDA_BASE = RAIZ / "datos" / "imagenes"
EXTENSIONES = {".mov", ".mp4", ".avi", ".mkv"}

PARAMETROS = {"intervalo": 0.5, "ancho": 1280, "max_movimiento": 2.0,
              "min_nitidez": 5.0, "min_cambio": 4.0}
TAMANO_COMPARACION = (160, 90)  # baja resolución: más rápido y menos sensible al ruido del sensor


def nitidez(gris):
    return cv2.Laplacian(gris, cv2.CV_64F).var()


def miniatura(frame):
    return cv2.cvtColor(cv2.resize(frame, TAMANO_COMPARACION, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)


def diferencia(mini_a, mini_b):
    return float(np.mean(cv2.absdiff(mini_a, mini_b)))


def redimensionar(frame, ancho_max):
    alto, ancho = frame.shape[:2]
    if ancho <= ancho_max:
        return frame
    escala = ancho_max / ancho
    return cv2.resize(frame, (ancho_max, round(alto * escala)), interpolation=cv2.INTER_AREA)


def procesar_video(ruta, carpeta_salida, prefijo, intervalo=0.5, ancho=1280,
                   max_movimiento=2.0, min_nitidez=5.0, min_cambio=4.0):
    """Extrae los fotogramas de un video.

    Devuelve las rutas guardadas y cuántos fotogramas se descartaron por
    movimiento, desenfoque y repetición. Lanza ValueError si el video no se
    puede abrir.
    """
    ruta, carpeta_salida = Path(ruta), Path(carpeta_salida)
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video {ruta.name}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    paso = max(1, round(fps * intervalo))
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    guardados = []
    descartes = {"movimiento": 0, "desenfocados": 0, "repetidos": 0}
    mini_anterior = mini_guardada = None
    indice = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mini = miniatura(frame)
        # El primer fotograma no tiene anterior; se evalúa desde el segundo.
        if indice % paso == 0 and mini_anterior is not None:
            if diferencia(mini, mini_anterior) > max_movimiento:
                descartes["movimiento"] += 1
            elif mini_guardada is not None and diferencia(mini, mini_guardada) < min_cambio:
                descartes["repetidos"] += 1
            else:
                frame = redimensionar(frame, ancho)
                if nitidez(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < min_nitidez:
                    descartes["desenfocados"] += 1
                else:
                    destino = carpeta_salida / f"{prefijo}_f{indice:06d}.jpg"
                    cv2.imwrite(str(destino), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    mini_guardada = mini
                    guardados.append(destino)
        mini_anterior = mini
        indice += 1
    cap.release()
    return {"guardados": guardados, **descartes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("carpeta", type=Path, help="carpeta de una sesión con videos")
    parser.add_argument("--intervalo", type=float, default=PARAMETROS["intervalo"],
                        help="segundos entre fotogramas evaluados")
    parser.add_argument("--ancho", type=int, default=PARAMETROS["ancho"],
                        help="ancho máximo de las imágenes guardadas")
    parser.add_argument("--max-movimiento", type=float, default=PARAMETROS["max_movimiento"],
                        help="diferencia media máxima con el fotograma anterior (0-255)")
    parser.add_argument("--min-nitidez", type=float, default=PARAMETROS["min_nitidez"],
                        help="varianza del Laplaciano mínima")
    parser.add_argument("--min-cambio", type=float, default=PARAMETROS["min_cambio"],
                        help="diferencia media mínima con el último guardado (0-255)")
    args = parser.parse_args()

    carpeta = args.carpeta.resolve()
    videos = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES)
    if not videos:
        print(f"No hay videos en {carpeta}")
        return

    # Sesión = prefijo de la carpeta (S01_dia_supA → S01); se usa luego para dividir el dataset.
    sesion = carpeta.name.split("_")[0]
    salida = SALIDA_BASE / carpeta.name
    print(f"Sesión {sesion}: {len(videos)} video(s) → {salida}")
    total = 0
    for video in videos:
        try:
            r = procesar_video(video, salida, f"{sesion}_{video.stem}", args.intervalo, args.ancho,
                               args.max_movimiento, args.min_nitidez, args.min_cambio)
        except ValueError as e:
            print(f"  [X] {e}")
            continue
        print(f"  {video.name}: {len(r['guardados'])} guardados; descartados: "
              f"{r['movimiento']} en movimiento, {r['desenfocados']} desenfocados, "
              f"{r['repetidos']} repetidos")
        total += len(r["guardados"])
    print(f"Total: {total} fotogramas")


if __name__ == "__main__":
    main()
