"""Entrenamiento del modelo propio (etapa 6): YOLO11n con transfer learning.

Lo lanza la interfaz web (página "Entrenar modelo") como un proceso aparte, para
que la API siga respondiendo mientras la GPU entrena. También se puede usar a mano:

    .venv\\Scripts\\python.exe vision\\entrenamiento\\entrenar.py --salida datos\\entrenamientos\\prueba

Durante el entrenamiento escribe <salida>/progreso.json después de cada época
(la interfaz lo lee para mostrar el avance). Al terminar:
  1. Evalúa el mejor modelo en la partición de validación y en la de prueba
     (mAP50, mAP50-95, precisión y recall, global y por categoría).
  2. Calcula el umbral de confianza de cada categoría (curva F1 de validación).
  3. Copia el mejor modelo a vision/modelos/best.pt (el anterior queda como
     best_anterior.pt) junto con best.json (los umbrales), y la API lo empieza a usar.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
MODELOS = RAIZ / "vision" / "modelos"
DATA_YAML = RAIZ / "datos" / "dataset" / "data.yaml"

# Gráficas que genera Ultralytics y que sirven para el informe.
GRAFICAS = ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
            "BoxPR_curve.png", "BoxF1_curve.png", "labels.jpg", "val_batch0_pred.jpg", "val_batch0_labels.jpg"]


class Progreso:
    """Guarda el estado en un JSON que se reemplaza de forma atómica (la API nunca lee uno a medias)."""

    def __init__(self, carpeta: Path, datos: dict):
        self.archivo = carpeta / "progreso.json"
        self.datos = datos
        self.guardar()

    def guardar(self, **cambios):
        self.datos.update(cambios, actualizado_en=datetime.now().astimezone().isoformat())
        temporal = self.archivo.with_suffix(".tmp")
        temporal.write_text(json.dumps(self.datos, indent=2, ensure_ascii=False), encoding="utf-8")
        temporal.replace(self.archivo)


def num(valor):
    try:
        return round(float(valor), 4)
    except (TypeError, ValueError):
        return None


def metricas(resultado):
    """Métricas globales y por categoría de un model.val()."""
    caja = resultado.box
    nombres = resultado.names
    por_clase = []
    for i, clase in enumerate(caja.ap_class_index):
        por_clase.append({"categoria": nombres[int(clase)], "precision": num(caja.p[i]), "recall": num(caja.r[i]),
                          "mAP50": num(caja.ap50[i]), "mAP50_95": num(caja.ap[i])})
    return {"precision": num(caja.mp), "recall": num(caja.mr), "mAP50": num(caja.map50),
            "mAP50_95": num(caja.map), "por_clase": por_clase,
            "inferencia_ms": num(resultado.speed.get("inference"))}


# Aumentos para una cámara que mira la vitrina desde arriba: un producto puede
# estar en cualquier orientación, así que voltear la imagen en vertical es tan
# válido como en horizontal (juntos equivalen a girarla 180°). El giro libre se
# deja moderado: YOLO recalcula la caja como el rectángulo recto que envuelve a
# la caja girada, y con giros grandes las cajas de empaques alargados quedan
# holgadas, lo que empeora la separación de productos pegados. Las orientaciones
# intermedias deben venir de las fotos reales.
AUMENTOS = {"flipud": 0.5, "fliplr": 0.5, "degrees": 20.0}

UMBRAL_MIN, UMBRAL_MAX = 0.25, 0.60


def umbrales_por_clase(resultado):
    """Umbral de confianza de cada categoría a partir de la curva F1 de validación.

    Se toma la confianza MÁS BAJA cuyo F1 queda a menos del 2 % del máximo: la
    curva suele ser una meseta, y en la vitrina real las confianzas son menores
    que en validación, así que conviene el extremo que favorece no perder productos.
    Se limita a un rango razonable porque validación se parece a entrenamiento.
    """
    caja = resultado.box
    umbrales = {}
    for i, clase in enumerate(caja.ap_class_index):
        curva = caja.f1_curve[i]
        if curva.max() <= 0:
            continue  # la clase no se detectó nunca: se queda con el umbral general
        indice = int((curva >= curva.max() * 0.98).argmax())
        umbrales[resultado.names[int(clase)]] = round(min(UMBRAL_MAX, max(UMBRAL_MIN, float(caja.px[indice]))), 2)
    return umbrales


def contar_imagenes(particion):
    carpeta = DATA_YAML.parent / "images" / particion
    return sum(1 for p in carpeta.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if carpeta.exists() else 0


def main():
    global DATA_YAML
    ap = argparse.ArgumentParser(description="Entrena YOLO con el dataset exportado")
    ap.add_argument("--salida", required=True, help="carpeta de este entrenamiento")
    ap.add_argument("--epocas", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lote", type=int, default=8, help="imágenes por lote (8 cabe en 4 GB de GPU)")
    ap.add_argument("--paciencia", type=int, default=15, help="épocas sin mejorar antes de parar")
    ap.add_argument("--base", choices=["coco", "propio"], default="coco",
                    help="coco: parte de yolo11n.pt; propio: continúa desde el best.pt actual")
    ap.add_argument("--data", default=str(DATA_YAML), help="data.yaml del dataset (por defecto, el exportado)")
    args = ap.parse_args()
    DATA_YAML = Path(args.data).resolve()

    salida = Path(args.salida).resolve()
    salida.mkdir(parents=True, exist_ok=True)
    base = MODELOS / ("best.pt" if args.base == "propio" else "yolo11n.pt")
    progreso = Progreso(salida, {
        "fase": "preparando", "mensaje": "Cargando PyTorch y el modelo base…",
        "parametros": vars(args) | {"modelo_base": base.name},
        "iniciado_en": datetime.now().astimezone().isoformat(),
        "epoca": 0, "epocas": args.epocas, "historial": [], "graficas": [], "resultado": None, "error": None,
    })

    try:
        if not DATA_YAML.exists():
            raise RuntimeError("No existe datos/dataset/data.yaml: exporta el dataset primero.")
        if contar_imagenes("train") == 0 or contar_imagenes("val") == 0:
            raise RuntimeError("Las particiones train y val no pueden estar vacías: etiqueta más imágenes y vuelve a exportar.")
        if not base.exists():
            raise RuntimeError(f"No se encontró el modelo base {base.name}.")

        import torch
        from ultralytics import YOLO

        dispositivo = 0 if torch.cuda.is_available() else "cpu"
        modelo = YOLO(str(base))
        inicio = time.monotonic()

        epoca_entrenada = {"pendiente": False}

        def al_entrenar_epoca(_trainer):
            epoca_entrenada["pendiente"] = True

        def al_terminar_epoca(trainer):
            # Ultralytics vuelve a llamar este evento en su validación final (sin una
            # época de entrenamiento antes): no es una época nueva y se ignora.
            if not epoca_entrenada["pendiente"]:
                return
            epoca_entrenada["pendiente"] = False
            m = trainer.metrics or {}
            perdidas = trainer.label_loss_items(trainer.tloss, prefix="train")
            fila = {
                "epoca": trainer.epoch + 1,
                "box_loss": num(perdidas.get("train/box_loss")), "cls_loss": num(perdidas.get("train/cls_loss")),
                "val_box_loss": num(m.get("val/box_loss")), "val_cls_loss": num(m.get("val/cls_loss")),
                "precision": num(m.get("metrics/precision(B)")), "recall": num(m.get("metrics/recall(B)")),
                "mAP50": num(m.get("metrics/mAP50(B)")), "mAP50_95": num(m.get("metrics/mAP50-95(B)")),
            }
            historial = progreso.datos["historial"] + [fila]
            transcurrido = time.monotonic() - inicio
            restante = transcurrido / fila["epoca"] * (trainer.epochs - fila["epoca"])
            progreso.guardar(fase="entrenando", epoca=fila["epoca"], epocas=trainer.epochs, historial=historial,
                             mensaje=f"Época {fila['epoca']} de {trainer.epochs}",
                             restante_s=round(restante))

        modelo.add_callback("on_train_epoch_end", al_entrenar_epoca)
        modelo.add_callback("on_fit_epoch_end", al_terminar_epoca)
        progreso.guardar(fase="entrenando", mensaje="Primera época (la primera tarda más: prepara el dataset)…")
        modelo.train(data=str(DATA_YAML), epochs=args.epocas, imgsz=args.imgsz, batch=args.lote,
                     patience=args.paciencia, device=dispositivo, project=str(salida), name="train",
                     exist_ok=True, workers=2, seed=42, verbose=False, plots=True, **AUMENTOS)

        carpeta_train = salida / "train"
        mejor = carpeta_train / "weights" / "best.pt"
        if not mejor.exists():
            raise RuntimeError("El entrenamiento terminó sin generar best.pt.")

        progreso.guardar(fase="evaluando", mensaje="Evaluando el mejor modelo en validación y prueba…",
                         restante_s=None)
        final = YOLO(str(mejor))
        resultado = {"duracion_s": round(time.monotonic() - inicio), "dispositivo": "GPU" if dispositivo == 0 else "CPU",
                     "epocas_completadas": progreso.datos["epoca"]}
        validacion = final.val(data=str(DATA_YAML), split="val", device=dispositivo,
                               project=str(salida), name="val", exist_ok=True, verbose=False)
        resultado["val"] = metricas(validacion)
        resultado["umbrales"] = umbrales_por_clase(validacion)
        if contar_imagenes("test"):
            resultado["test"] = metricas(final.val(data=str(DATA_YAML), split="test", device=dispositivo,
                                                   project=str(salida), name="test", exist_ok=True, verbose=False))

        # El modelo nuevo pasa a ser el de la API; el anterior se conserva por si hay que volver.
        destino = MODELOS / "best.pt"
        umbrales_json = destino.with_suffix(".json")
        if destino.exists():
            shutil.copy2(destino, MODELOS / "best_anterior.pt")
            if umbrales_json.exists():
                shutil.copy2(umbrales_json, MODELOS / "best_anterior.json")
        shutil.copy2(mejor, destino)
        # El sha256 liga los umbrales con estos pesos: el detector los ignora si no coinciden.
        umbrales_json.write_text(json.dumps(
            {"sha256": hashlib.sha256(destino.read_bytes()).hexdigest(), "entrenamiento": salida.name,
             "umbrales": resultado["umbrales"]}, indent=2, ensure_ascii=False), encoding="utf-8")
        resultado["modelo"] = str(destino.relative_to(RAIZ))

        graficas = [f"train/{g}" for g in GRAFICAS if (carpeta_train / g).exists()]
        graficas += [f"test/{g}" for g in ("confusion_matrix.png",) if (salida / "test" / g).exists()]
        progreso.guardar(fase="terminado", mensaje="Entrenamiento terminado: el modelo nuevo ya está en uso.",
                         resultado=resultado, graficas=graficas,
                         terminado_en=datetime.now().astimezone().isoformat())
    except Exception as e:
        traceback.print_exc()
        mensaje = str(e)
        if "out of memory" in mensaje.lower():
            mensaje = "La GPU se quedó sin memoria: baja el tamaño de lote o de imagen."
        progreso.guardar(fase="error", error=f"{type(e).__name__}: {mensaje}", mensaje="El entrenamiento falló",
                         terminado_en=datetime.now().astimezone().isoformat())
        sys.exit(1)


if __name__ == "__main__":  # necesario en Windows: los workers del DataLoader reimportan este archivo
    main()
