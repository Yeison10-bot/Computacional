"""Monitoreo continuo de la vitrina (etapa 7).

La cámara queda fija y este módulo analiza su imagen sin parar. YOLO se equivoca
de vez en cuando entre un fotograma y otro, y una mano tapa productos al tomar
uno, así que NO se registra un evento por lo que se ve en un solo fotograma.
El algoritmo (docs/02_arquitectura.md, sección 4) es:

  1. Detección: YOLO cuenta los productos de cada categoría en el fotograma.
  2. Suavizado temporal: se guardan los conteos de los últimos segundos y se usa
     la MODA (valor más frecuente); así un parpadeo de 1-2 fotogramas no cuenta.
     Un cambio solo avanza si la moda tiene MAYORÍA (≥ 60 % de la ventana): si el
     modelo duda (p. ej. ve 0, 1 y 2 frutas en fotogramas seguidos) se espera en
     vez de adivinar, para no registrar ventas y reposiciones falsas en cadena.
  3. Detección de movimiento (procesamiento clásico): diferencia absoluta con el
     fotograma anterior, umbral por píxel y porcentaje de píxeles que cambiaron.
     Si hay movimiento (una mano), no se confirma nada.
  4. Confirmación: si la moda difiere del estado estable y la escena sigue
     QUIETA durante T segundos, se registra el evento: si bajó, venta; si subió,
     reposición. Después el estado estable pasa a ser el valor nuevo, por eso un
     producto que se queda fuera de la vitrina no genera ventas duplicadas.

Hilos: un lector toma los fotogramas de la cámara (o los recibe del navegador),
el analizador aplica el algoritmo a la velocidad configurada, y un publicador
dibuja las últimas detecciones sobre el video para verlo en vivo con fluidez,
independiente de la velocidad del análisis.
"""

import json
import threading
import time
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np

from .config import DATOS, EVIDENCIAS
from .db import SesionBD
from .inventario_logica import confianza_media, registrar_conteo
from .vision import obtener_detector

PARAMETROS = {
    "fps_analisis": 5,          # fotogramas analizados por segundo
    "ventana_s": 1.4,           # segundos de historia para la moda
    "mayoria": 0.6,             # fracción mínima de la ventana que debe coincidir con la moda
    "confirmacion_s": 4.0,      # tiempo que un cambio debe mantenerse con la escena quieta
    "quietud_s": 1.0,           # tiempo sin movimiento para considerar la escena quieta
    "umbral_pixel": 25,         # diferencia de gris (0-255) para que un píxel "cambie"
    "fraccion_movimiento": 0.01,  # porcentaje de píxeles cambiados que cuenta como movimiento
}
# Ajustables desde la interfaz, con sus límites.
AJUSTABLES = {"fps_analisis": (1, 10), "confirmacion_s": (1, 15), "quietud_s": (0.5, 5), "ventana_s": (0.6, 5)}
ARCHIVO_PARAMETROS = DATOS / "monitor_parametros.json"
FPS_VIDEO = 12  # fotogramas por segundo de la transmisión en vivo (solo visualización)

# Colores de las categorías en BGR (OpenCV), los mismos de la interfaz web.
COLORES = {"botella": (214, 120, 42), "lata": (52, 104, 235), "vaso": (122, 175, 27),
           "paquete": (0, 161, 237), "fruta": (164, 123, 232)}


def moda(valores):
    """Valor más frecuente y la fracción de la ventana que lo tiene.
    En empate gana el más reciente (refleja antes un cambio real)."""
    frecuencias = Counter(valores)
    maximo = max(frecuencias.values())
    valor = next(v for v in reversed(valores) if frecuencias[v] == maximo)
    return valor, maximo / len(valores)


def cargar_parametros():
    parametros = dict(PARAMETROS)
    try:
        guardados = json.loads(ARCHIVO_PARAMETROS.read_text(encoding="utf-8"))
        parametros.update({k: v for k, v in guardados.items() if k in AJUSTABLES})
    except (OSError, ValueError):
        pass  # sin archivo o dañado: valores por defecto
    return parametros


class LogicaRetiros:
    """El algoritmo de los pasos 2 a 4, sin cámara ni base de datos.

    Recibe el tiempo, el fotograma y el conteo de YOLO, y dice qué cambios se
    confirman. Lo usan el monitoreo en vivo (con el reloj real) y la evaluación
    con videos de prueba (con el tiempo del video), así ambos se comportan igual.
    """

    def __init__(self, categorias, parametros):
        self.categorias = list(categorias)
        self.p = parametros
        self.ventana = deque()
        self.estable = None          # lo que el sistema cree que hay en la vitrina
        self.pendientes = {}         # categoría → {"valor", "desde", "certeza"}
        self.ultimo_movimiento = None
        self.gris_anterior = None

    def paso(self, ahora, fotograma, conteo):
        """Devuelve (inicial, confirmados, info): `inicial` es el primer estado estable
        (o None) y `confirmados` los cambios confirmados en este paso {categoría: valor}."""
        p = self.p
        if self.ultimo_movimiento is None:
            self.ultimo_movimiento = ahora  # al iniciar se exige un periodo de quietud

        # Paso 3: movimiento por diferencia de fotogramas (gris, suavizado, baja resolución).
        gris = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(fotograma, (320, 180)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        fraccion = 0.0
        if self.gris_anterior is not None:
            cambio = cv2.absdiff(gris, self.gris_anterior) > p["umbral_pixel"]
            fraccion = float(np.count_nonzero(cambio)) / cambio.size
        self.gris_anterior = gris
        if fraccion > p["fraccion_movimiento"]:
            self.ultimo_movimiento = ahora
        quieto = ahora - self.ultimo_movimiento >= p["quietud_s"]

        # Paso 2: moda en la ventana (su largo en fotogramas depende de la velocidad).
        largo = max(3, round(p["ventana_s"] * p["fps_analisis"]))
        if self.ventana.maxlen != largo:
            self.ventana = deque(self.ventana, maxlen=largo)
        self.ventana.append({c: conteo.get(c, 0) for c in self.categorias})
        resultado = {c: moda([v[c] for v in self.ventana]) for c in self.categorias}
        modas = {c: r[0] for c, r in resultado.items()}
        certeza = {c: r[1] for c, r in resultado.items()}

        # Paso 4: inicialización y confirmación de cambios.
        inicial, confirmados = None, {}
        if self.estable is None:
            if len(self.ventana) == self.ventana.maxlen and quieto:
                self.estable = dict(modas)
                inicial = dict(modas)
        else:
            for c in self.categorias:
                v = modas[c]
                if v == self.estable[c]:
                    self.pendientes.pop(c, None)  # era ruido u oclusión: volvió al valor estable
                    continue
                pend = self.pendientes.get(c)
                if pend is None or pend["valor"] != v:
                    pend = self.pendientes[c] = {"valor": v, "desde": None}
                pend["certeza"] = certeza[c]
                if not quieto or certeza[c] < p["mayoria"]:
                    # Con movimiento o con una detección dudosa el temporizador se reinicia.
                    pend["desde"] = None
                elif pend["desde"] is None:
                    pend["desde"] = ahora
                elif ahora - pend["desde"] >= p["confirmacion_s"]:
                    confirmados[c] = v
            self.estable.update(confirmados)
            for c in confirmados:
                self.pendientes.pop(c, None)

        if self.estable is None:
            fase = "iniciando"
        elif not quieto:
            fase = "movimiento"
        elif any(x["desde"] is not None for x in self.pendientes.values()):
            fase = "confirmando"
        else:
            fase = "estable"
        return inicial, confirmados, {"modas": modas, "quieto": quieto, "fraccion": fraccion, "fase": fase}


class LectorVideo(threading.Thread):
    """Lee fotogramas de una cámara o archivo continuamente y conserva solo el último."""

    def __init__(self, captura, es_archivo):
        super().__init__(daemon=True)
        self.captura = captura
        self.es_archivo = es_archivo
        # Un archivo se reproduce a su velocidad real, como si fuera una cámara.
        self.periodo = 1 / (captura.get(cv2.CAP_PROP_FPS) or 30) if es_archivo else 0
        self.ultimo = None      # (numero, fotograma)
        self.terminado = False
        self._parar = threading.Event()

    def run(self):
        numero = 0
        while not self._parar.is_set():
            inicio = time.monotonic()
            ok, fotograma = self.captura.read()
            if not ok:
                if self.es_archivo:
                    break  # fin del video
                time.sleep(0.05)  # la cámara no entregó imagen; se reintenta
                continue
            numero += 1
            self.ultimo = (numero, fotograma)
            if self.periodo:
                time.sleep(max(0, self.periodo - (time.monotonic() - inicio)))
        self.terminado = True
        self.captura.release()

    def parar(self):
        self._parar.set()


class LectorExterno:
    """Fuente "navegador": la página web abre la cámara del computador y envía los
    fotogramas por HTTP (POST /api/monitor/fotograma). Se analizan igual que los
    de una cámara conectada al servidor."""

    es_archivo = False
    ESPERA_MAXIMA_S = 8  # sin fotogramas durante este tiempo → la pestaña se cerró

    def __init__(self):
        self.ultimo = None
        self.terminado = False
        self.ultimo_recibido = time.monotonic()
        self.recibidos = 0

    def start(self):
        pass  # no hay hilo: los fotogramas llegan por la API

    def recibir(self, fotograma):
        self.recibidos += 1
        self.ultimo = (self.recibidos, fotograma)
        self.ultimo_recibido = time.monotonic()

    def sin_senal(self):
        return time.monotonic() - self.ultimo_recibido > self.ESPERA_MAXIMA_S

    def parar(self):
        self.terminado = True


class Monitor:
    def __init__(self):
        self._candado = threading.Lock()
        self._hilo = None
        self._hilo_video = None
        self._lector = None
        self._parar = threading.Event()
        self.parametros = cargar_parametros()
        self.jpeg = None             # último fotograma anotado, para la transmisión en vivo
        self._detecciones = []       # detecciones del último fotograma analizado
        self._estado = self._estado_inicial()

    def _estado_inicial(self, **extra):
        return {"activo": False, "fase": "detenido", "fuente": None, "iniciado_en": None, "error": None,
                "fps": 0.0, "inferencia_ms": 0.0, "movimiento": False, "fraccion_movimiento": 0.0,
                "categorias": [], "conteo_actual": {}, "moda": {}, "estable": None, "pendientes": {},
                "eventos": [], "fotogramas": 0, "recibidos": 0, "detecciones": [], "resolucion": None,
                "modelo": None, "parametros": self.parametros, **extra}

    @property
    def activo(self):
        return self._hilo is not None and self._hilo.is_alive()

    def estado(self):
        with self._candado:
            recibidos = self._lector.recibidos if isinstance(self._lector, LectorExterno) else None
            return {**self._estado, "eventos": list(self._estado["eventos"]), "recibidos": recibidos,
                    "detecciones": list(self._detecciones), "parametros": dict(self.parametros)}

    def ajustar(self, cambios: dict):
        """Cambia parámetros en caliente (se aplican en el siguiente fotograma) y los guarda."""
        for clave, valor in cambios.items():
            minimo, maximo = AJUSTABLES[clave]
            self.parametros[clave] = min(maximo, max(minimo, float(valor)))
        ARCHIVO_PARAMETROS.parent.mkdir(parents=True, exist_ok=True)
        ARCHIVO_PARAMETROS.write_text(json.dumps({k: self.parametros[k] for k in AJUSTABLES}, indent=2),
                                      encoding="utf-8")
        return dict(self.parametros)

    def iniciar(self, fuente: str, detector=None):
        """fuente: índice de cámara del servidor ("0", "1"...), ruta de un archivo
        de video, o "navegador" (la página web envía los fotogramas)."""
        if self.activo:
            raise RuntimeError("El monitoreo ya está activo")
        if fuente == "navegador":
            self._lector = LectorExterno()
        else:
            es_archivo = not str(fuente).isdigit()
            captura = cv2.VideoCapture(str(fuente) if es_archivo else int(fuente))
            if not captura.isOpened():
                raise ValueError(f"No se pudo abrir la fuente de video «{fuente}»")
            if not es_archivo:
                captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._lector = LectorVideo(captura, es_archivo)
        self._parar.clear()
        self.jpeg = None
        self._detecciones = []
        with self._candado:
            self._estado = self._estado_inicial(activo=True, fase="cargando", fuente=str(fuente),
                                                iniciado_en=datetime.now().astimezone().isoformat())
        self._lector.start()
        self._hilo = threading.Thread(target=self._bucle, args=(detector,), daemon=True)
        self._hilo.start()
        if fuente != "navegador":  # con el navegador, la página muestra su propio video
            self._hilo_video = threading.Thread(target=self._publicar_video, daemon=True)
            self._hilo_video.start()

    def recibir_fotograma(self, fotograma):
        """Entrega un fotograma enviado por el navegador (solo con la fuente "navegador")."""
        if not self.activo or not isinstance(self._lector, LectorExterno):
            raise RuntimeError("El monitoreo con la cámara del navegador no está activo")
        self._lector.recibir(fotograma)

    def detener(self):
        self._parar.set()
        if self._lector:
            self._lector.parar()
        for hilo in (self._hilo, self._hilo_video):
            if hilo:
                hilo.join(timeout=5)
        with self._candado:
            self._estado["activo"] = False
            if self._estado["fase"] not in ("finalizado", "error"):
                self._estado["fase"] = "detenido"

    # ------------------------------------------------------------------ análisis

    def _bucle(self, detector):
        try:
            self._analizar(detector or obtener_detector())
        except Exception as e:  # el error queda visible en la interfaz
            with self._candado:
                self._estado.update(error=f"{type(e).__name__}: {e}", fase="error")
        finally:
            self._lector.parar()
            with self._candado:
                self._estado["activo"] = False

    def _analizar(self, detector):
        categorias = list(detector.categorias_detectables)
        with self._candado:
            self._estado.update(categorias=categorias, modelo=detector.info(), fase="iniciando")

        logica = LogicaRetiros(categorias, self.parametros)
        ultimo_numero = 0
        tiempos = deque(maxlen=10)

        while not self._parar.is_set():
            p = self.parametros  # se lee en cada vuelta: los ajustes se aplican en caliente
            inicio = time.monotonic()
            actual = self._lector.ultimo
            if actual is None or actual[0] == ultimo_numero:
                if self._lector.terminado:
                    with self._candado:
                        self._estado["fase"] = "finalizado"
                    return
                if isinstance(self._lector, LectorExterno) and self._lector.sin_senal():
                    with self._candado:
                        self._estado.update(fase="detenido", error="La cámara del navegador dejó de enviar "
                                            "imágenes (¿se cerró la pestaña?). Vuelve a iniciar el monitoreo.")
                    return
                time.sleep(0.01)
                continue
            ultimo_numero, fotograma = actual
            ahora = time.monotonic()

            # Paso 1: detección.
            t0 = time.perf_counter()
            detecciones = detector.detectar(fotograma)
            inferencia_ms = (time.perf_counter() - t0) * 1000
            self._detecciones = detecciones
            conteo = Counter(d["categoria"] for d in detecciones)

            # Pasos 2 a 4 (LogicaRetiros).
            inicial, confirmados, info = logica.paso(ahora, fotograma, conteo)
            if inicial is not None:
                self._registrar(inicial, categorias, "ajuste", fotograma, detecciones,
                                "Monitoreo iniciado: inventario igualado a lo que ve la cámara")
            if confirmados:
                self._registrar(confirmados, list(confirmados), "movimientos", fotograma, detecciones)

            tiempos.append(ahora)
            fps = (len(tiempos) - 1) / (tiempos[-1] - tiempos[0]) if len(tiempos) > 1 else 0.0
            with self._candado:
                self._estado.update(
                    fase=info["fase"], fps=round(fps, 1), inferencia_ms=round(inferencia_ms, 1),
                    movimiento=not info["quieto"], fraccion_movimiento=round(info["fraccion"], 4),
                    resolucion=f"{fotograma.shape[1]}x{fotograma.shape[0]}",
                    conteo_actual={c: conteo.get(c, 0) for c in categorias}, moda=info["modas"],
                    estable=dict(logica.estable) if logica.estable else None,
                    fotogramas=self._estado["fotogramas"] + 1,
                    pendientes={c: {"valor": x["valor"], "certeza": round(x.get("certeza", 1.0), 2),
                                    "dudosa": x.get("certeza", 1.0) < p["mayoria"],
                                    "restante_s": None if x["desde"] is None else
                                    round(max(0.0, p["confirmacion_s"] - (ahora - x["desde"])), 1)}
                                for c, x in logica.pendientes.items()})

            time.sleep(max(0.0, 1 / p["fps_analisis"] - (time.monotonic() - inicio)))

    def _registrar(self, conteos, categorias, modo, fotograma, detecciones, nota=None):
        """Guarda la foto de evidencia y registra los movimientos en la base de datos."""
        EVIDENCIAS.mkdir(parents=True, exist_ok=True)
        nombre = f"{datetime.now():%Y%m%d_%H%M%S_%f}_monitor.jpg"
        # La evidencia se guarda con los recuadros dibujados: muestra qué vio el modelo.
        (EVIDENCIAS / nombre).write_bytes(self._anotar(fotograma, detecciones, None))
        evidencia = (EVIDENCIAS / nombre).relative_to(DATOS).as_posix()
        with SesionBD() as bd:
            movimientos = registrar_conteo(bd, conteos, categorias, modo, evidencia,
                                           {c: confianza_media(detecciones, c) for c in categorias})
            nuevos = [{"tipo": m.tipo, "categoria": m.producto.categoria_yolo, "nombre": m.producto.nombre,
                       "cantidad": m.cantidad, "total": float(m.precio_unitario * m.cantidad) if m.tipo == "venta" else 0.0,
                       "fecha_hora": m.fecha_hora.isoformat(), "evidencia_url": f"/archivos/{evidencia}"}
                      for m in movimientos]
        with self._candado:
            if nota:
                self._estado["eventos"].insert(0, {"tipo": "info", "texto": nota,
                                                   "fecha_hora": datetime.now().astimezone().isoformat()})
            self._estado["eventos"][:0] = nuevos
            del self._estado["eventos"][30:]

    # ------------------------------------------------------------------ video en vivo

    def _publicar_video(self):
        """Dibuja las últimas detecciones sobre el fotograma más reciente ~12 veces por
        segundo. Así el video se ve fluido aunque el análisis vaya más lento."""
        ultimo = 0
        while not self._parar.is_set() and not self._lector.terminado:
            inicio = time.monotonic()
            actual = self._lector.ultimo
            if actual is not None and actual[0] != ultimo:
                ultimo, fotograma = actual
                with self._candado:
                    fase = self._estado["fase"]
                self.jpeg = self._anotar(fotograma, self._detecciones, fase)
            time.sleep(max(0.0, 1 / FPS_VIDEO - (time.monotonic() - inicio)))

    @staticmethod
    def _anotar(fotograma, detecciones, fase):
        img = fotograma.copy()
        alto, ancho = img.shape[:2]
        grosor = max(2, ancho // 400)
        escala = ancho / 1600
        for d in detecciones:
            x1, y1 = int((d["x_centro"] - d["ancho"] / 2) * ancho), int((d["y_centro"] - d["alto"] / 2) * alto)
            x2, y2 = int((d["x_centro"] + d["ancho"] / 2) * ancho), int((d["y_centro"] + d["alto"] / 2) * alto)
            color = COLORES.get(d["categoria"], (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, grosor)
            texto = f"{d['categoria']} {d['confianza']:.0%}"
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, 2)
            cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(img, texto, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), 2)
        if fase == "movimiento":
            # Borde ámbar: hay movimiento y no se está confirmando ningún cambio.
            cv2.rectangle(img, (0, 0), (ancho - 1, alto - 1), (0, 170, 255), grosor * 3)
        ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return jpeg.tobytes() if ok else None


monitor = Monitor()
