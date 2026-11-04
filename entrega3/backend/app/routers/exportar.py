"""Resumen del dataset y exportación a formato YOLO."""

import random
import shutil
from collections import Counter, defaultdict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from ..config import DATASET, DATOS
from ..db import obtener_bd
from ..models import Anotacion, ArchivoMedia, Imagen, Producto
from ..schemas import ExportarParametros

router = APIRouter(prefix="/api/dataset", tags=["Exportación"])
PARTICIONES = ("train", "val", "test")


@router.get("/resumen")
def resumen(bd: Session = Depends(obtener_bd)):
    """Cuántas cajas hay por categoría y cuántas imágenes por estado y partición."""
    productos = bd.scalars(select(Producto).order_by(Producto.clase_id)).all()
    cajas = dict(bd.execute(
        select(Anotacion.producto_id, func.count()).join(Anotacion.imagen)
        .where(Imagen.estado == "etiquetada").group_by(Anotacion.producto_id)).all())
    estados = dict(bd.execute(select(Imagen.estado, func.count()).group_by(Imagen.estado)).all())
    particiones = dict(bd.execute(
        select(Imagen.particion, func.count()).where(Imagen.particion.is_not(None))
        .group_by(Imagen.particion)).all())
    return {
        "clases": [{"clase_id": p.clase_id, "categoria": p.categoria_yolo, "nombre": p.nombre,
                    "cajas": cajas.get(p.id, 0)} for p in productos],
        "imagenes": {e: estados.get(e, 0) for e in ("pendiente", "etiquetada", "descartada")},
        "particiones": {p: particiones.get(p, 0) for p in PARTICIONES},
    }


def asignar_particiones(imagenes, val, test, semilla):
    """Reparte las imágenes en train/val/test por GRUPOS, no por imagen.

    Grupo = el video del que salió el fotograma (las fotos sueltas se agrupan por
    sesión). Fotogramas del mismo video son casi idénticos: si quedaran unos en
    train y otros en val, la métrica de validación saldría inflada (fuga de datos).
    Las sesiones de prueba de generalización van completas a test.
    """
    advertencias = []
    asignacion = {}
    grupos = defaultdict(list)
    for img in imagenes:
        if img.sesion.prueba_generalizacion:
            asignacion[img.id] = "test"
        elif img.archivo.tipo == "video":
            grupos[("video", img.archivo_id)].append(img)
        else:
            grupos[("fotos", img.sesion_id)].append(img)

    claves = sorted(grupos)
    random.Random(semilla).shuffle(claves)
    if len(claves) < 3:
        advertencias.append(
            f"Solo hay {len(claves)} grupo(s) (videos o sesiones de fotos): se dividió por imagen. "
            "Las métricas de validación pueden salir optimistas; sube más videos.")
        sueltas = [img for c in claves for img in grupos[c]]
        random.Random(semilla).shuffle(sueltas)
        grupos = {("img", i.id): [i] for i in sueltas}
        claves = list(grupos)

    total = sum(len(grupos[c]) for c in claves)
    cupo = {"val": val * total, "test": test * total}
    conteo = Counter()
    # Se llenan primero val y test hasta su cupo, el resto va a train; se deja
    # siempre al menos un grupo para train.
    for i, clave in enumerate(claves):
        quedan = len(claves) - i
        if conteo["val"] < cupo["val"] and quedan > 1:
            destino = "val"
        elif conteo["test"] < cupo["test"] and quedan > 1:
            destino = "test"
        else:
            destino = "train"
        for img in grupos[clave]:
            asignacion[img.id] = destino
        conteo[destino] += len(grupos[clave])
    return asignacion, advertencias


@router.post("/exportar")
def exportar(param: ExportarParametros, bd: Session = Depends(obtener_bd)):
    """Escribe datos/dataset/ en formato YOLO (images/, labels/, data.yaml)."""
    imagenes = bd.scalars(
        select(Imagen).where(Imagen.estado == "etiquetada")
        .options(selectinload(Imagen.anotaciones).selectinload(Anotacion.producto),
                 selectinload(Imagen.sesion), selectinload(Imagen.archivo))
        .order_by(Imagen.id)).all()
    if len(imagenes) < 3:
        raise HTTPException(400, "Se necesitan al menos 3 imágenes etiquetadas para exportar")

    asignacion, advertencias = asignar_particiones(imagenes, param.val, param.test, param.semilla)

    # Se reescribe el dataset completo para que no queden imágenes de exportaciones anteriores.
    for sub in ("images", "labels"):
        for p in PARTICIONES:
            carpeta = DATASET / sub / p
            shutil.rmtree(carpeta, ignore_errors=True)
            carpeta.mkdir(parents=True)

    bd.execute(Imagen.__table__.update().values(particion=None))
    resumen_part = {p: {"imagenes": 0, "cajas": Counter()} for p in PARTICIONES}
    for img in imagenes:
        part = asignacion[img.id]
        img.particion = part
        origen = DATOS / img.ruta
        nombre = origen.name
        shutil.copy2(origen, DATASET / "images" / part / nombre)
        lineas = [f"{a.producto.clase_id} {a.x_centro:.6f} {a.y_centro:.6f} {a.ancho:.6f} {a.alto:.6f}"
                  for a in img.anotaciones]
        # Una imagen sin cajas lleva un .txt vacío: YOLO la usa como ejemplo negativo.
        (DATASET / "labels" / part / f"{origen.stem}.txt").write_text(
            "\n".join(lineas) + ("\n" if lineas else ""), encoding="utf-8")
        resumen_part[part]["imagenes"] += 1
        resumen_part[part]["cajas"].update(a.producto.categoria_yolo for a in img.anotaciones)
    bd.commit()

    productos = bd.scalars(select(Producto).order_by(Producto.clase_id)).all()
    nombres = "\n".join(f"  {p.clase_id}: {p.categoria_yolo}" for p in productos)
    (DATASET / "data.yaml").write_text(
        f"# Generado por la interfaz web (POST /api/dataset/exportar)\n"
        f"path: {DATASET.as_posix()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{nombres}\n", encoding="utf-8")

    for p in PARTICIONES:
        if resumen_part[p]["imagenes"] == 0 and p != "test":
            advertencias.append(f"La partición '{p}' quedó vacía; el entrenamiento la necesita.")
    return {
        "particiones": {p: {"imagenes": r["imagenes"], "cajas": dict(r["cajas"])}
                        for p, r in resumen_part.items()},
        "advertencias": advertencias,
        "ruta": str(DATASET),
    }
