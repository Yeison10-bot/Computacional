"""Evaluación del sistema de retiros con videos de prueba (etapa 11).

Para cada video el usuario marca en la interfaz los eventos REALES (segundo,
categoría, venta o reposición, cantidad). Aquí el video se pasa por el mismo
algoritmo del monitoreo en vivo (LogicaRetiros), usando el tiempo del video en
vez del reloj, sin tocar la base de datos. Luego se emparejan los eventos
detectados con los reales y se calculan:

  - aciertos (verdaderos positivos), retiros omitidos (falsos negativos),
    eventos falsos o duplicados (falsos positivos);
  - precisión, recall y F1;
  - latencia: segundos desde el evento real hasta que el sistema lo registra.

Emparejamiento (por unidad): un evento detectado corresponde a uno real si es de
la misma categoría y tipo, y ocurre entre 1 s antes y `tolerancia_s` después del
real. Cada detección se usa una sola vez; si sobran, son falsos positivos (ahí
caen los registros duplicados).
"""

import time
from collections import Counter

import cv2

from .monitor import LogicaRetiros, cargar_parametros

ANTICIPO_S = 1.0  # el usuario puede marcar el evento un poco tarde


def analizar_video(ruta, detector, parametros=None, al_progresar=None):
    """Recorre el video a `fps_analisis` y devuelve los eventos que registraría el sistema."""
    parametros = dict(parametros or cargar_parametros())
    captura = cv2.VideoCapture(str(ruta))
    if not captura.isOpened():
        raise ValueError("No se pudo abrir el video (si es HEVC del iPhone, grábalo en «Más compatible»)")
    fps_video = captura.get(cv2.CAP_PROP_FPS) or 30
    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    salto = max(1, round(fps_video / parametros["fps_analisis"]))
    categorias = list(detector.categorias_detectables)
    logica = LogicaRetiros(categorias, parametros)
    from vision.detector import MemoriaDetecciones  # importación diferida: carga PyTorch
    memoria = MemoriaDetecciones()  # misma histéresis que el monitoreo en vivo
    eventos, inicio_s, inicial = [], None, None
    tiempos_inferencia = []
    indice = 0
    reloj = time.monotonic()
    try:
        while True:
            ok = captura.grab()
            if not ok:
                break
            if indice % salto == 0:
                ok, fotograma = captura.retrieve()
                if not ok:
                    break
                t = indice / fps_video
                t0 = time.perf_counter()
                detecciones = detector.detectar(fotograma, memoria.previas(t))
                memoria.actualizar(t, detecciones)
                conteo = Counter(d["categoria"] for d in detecciones)
                tiempos_inferencia.append((time.perf_counter() - t0) * 1000)
                previo = dict(logica.estable) if logica.estable else None
                ini, confirmados, _ = logica.paso(t, fotograma, conteo)
                if ini is not None:
                    inicio_s, inicial = round(t, 2), ini
                for c, v in confirmados.items():
                    diferencia = v - previo[c]
                    eventos.append({"t": round(t, 2), "categoria": c, "cantidad": abs(diferencia),
                                    "tipo": "venta" if diferencia < 0 else "reposicion"})
                if al_progresar and total:
                    al_progresar(min(1.0, indice / total))
            indice += 1
    finally:
        captura.release()
    duracion = indice / fps_video
    return {
        "eventos": eventos, "inicio_s": inicio_s, "estado_inicial": inicial, "categorias": categorias,
        "duracion_s": round(duracion, 1), "parametros": parametros,
        "inferencia_ms": round(sum(tiempos_inferencia) / len(tiempos_inferencia), 1) if tiempos_inferencia else None,
        "procesado_en_s": round(time.monotonic() - reloj, 1),
    }


def unidades(eventos):
    """Un evento de 2 unidades cuenta como 2 retiros."""
    return [dict(e, cantidad=1) for e in eventos for _ in range(int(e["cantidad"]))]


def comparar(reales, detectados, categorias, inicio_s, tolerancia_s=15.0):
    """Empareja eventos reales con detectados y calcula las métricas."""
    reales_u = sorted(unidades(reales), key=lambda e: e["t"])
    detect_u = sorted(unidades(detectados), key=lambda e: e["t"])
    usados = set()
    filas = []
    excluidos = []
    for r in reales_u:
        if inicio_s is None or r["t"] < inicio_s:
            # Ocurrió antes de que el sistema terminara de iniciar: no se puede exigir.
            excluidos.append(r)
            continue
        candidato = None
        for i, d in enumerate(detect_u):
            if i in usados or d["categoria"] != r["categoria"] or d["tipo"] != r["tipo"]:
                continue
            if r["t"] - ANTICIPO_S <= d["t"] <= r["t"] + tolerancia_s:
                candidato = i
                break
        if candidato is None:
            motivo = "categoría no detectable con el modelo actual" if r["categoria"] not in categorias else "no detectado"
            filas.append({"real": r, "detectado": None, "resultado": "omitido", "motivo": motivo})
        else:
            usados.add(candidato)
            d = detect_u[candidato]
            filas.append({"real": r, "detectado": d, "resultado": "acierto", "latencia_s": round(d["t"] - r["t"], 2)})
    falsos = [d for i, d in enumerate(detect_u) if i not in usados]
    for d in falsos:
        # ¿Es un duplicado? (misma categoría y tipo que un acierto cercano)
        duplicado = any(f["resultado"] == "acierto" and f["real"]["categoria"] == d["categoria"]
                        and f["real"]["tipo"] == d["tipo"] and abs(f["detectado"]["t"] - d["t"]) <= tolerancia_s
                        for f in filas)
        filas.append({"real": None, "detectado": d, "resultado": "duplicado" if duplicado else "falso"})

    vp = sum(f["resultado"] == "acierto" for f in filas)
    fn = sum(f["resultado"] == "omitido" for f in filas)
    fp = len(falsos)
    latencias = [f["latencia_s"] for f in filas if f["resultado"] == "acierto"]
    return {
        "aciertos": vp, "omitidos": fn, "falsos": sum(f["resultado"] == "falso" for f in filas),
        "duplicados": sum(f["resultado"] == "duplicado" for f in filas), "falsos_positivos": fp,
        "precision": round(vp / (vp + fp), 4) if vp + fp else None,
        "recall": round(vp / (vp + fn), 4) if vp + fn else None,
        "f1": round(2 * vp / (2 * vp + fp + fn), 4) if vp + fp + fn else None,
        "latencia_media_s": round(sum(latencias) / len(latencias), 2) if latencias else None,
        "latencia_max_s": max(latencias) if latencias else None,
        "excluidos": excluidos, "tolerancia_s": tolerancia_s,
        "filas": sorted(filas, key=lambda f: (f["real"] or f["detectado"])["t"]),
    }
