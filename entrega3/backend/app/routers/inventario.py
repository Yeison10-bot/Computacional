"""Inventario actualizado por visión: conteo desde una foto, movimientos y panel."""

import time
import uuid
from datetime import datetime
from decimal import Decimal

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from ..config import ANCHO_MAXIMO, DATOS, EVIDENCIAS
from ..db import obtener_bd
from ..inventario_logica import confianza_media, registrar_conteo
from ..models import Inventario, Movimiento, Producto
from ..schemas import ConteoFoto, Deteccion, FilaConteo, MovimientoOut, Sincronizar
from ..vision import obtener_detector, recargar_detector

router = APIRouter(prefix="/api", tags=["Inventario por visión"])


def productos_con_inventario(bd: Session):
    return bd.scalars(select(Producto).options(joinedload(Producto.inventario))
                      .order_by(Producto.clase_id)).all()


def analizar(bd: Session, ruta_relativa: str):
    """Corre el detector sobre una evidencia guardada y compara con el inventario."""
    imagen = cv2.imread(str(DATOS / ruta_relativa))
    if imagen is None:
        raise HTTPException(404, "No se encontró la foto de evidencia")
    detector = obtener_detector()
    inicio = time.perf_counter()
    detecciones = detector.detectar(imagen)
    tiempo_ms = (time.perf_counter() - inicio) * 1000
    conteos = detector.contar(detecciones)
    detectables = set(detector.categorias_detectables)
    productos = productos_con_inventario(bd)
    return imagen, detecciones, conteos, detectables, productos, detector.info(), tiempo_ms


@router.get("/vision/modelo")
def modelo():
    return obtener_detector().info()


@router.post("/vision/recargar")
def recargar():
    """Vuelve a cargar el modelo (después de entrenar un best.pt nuevo)."""
    recargar_detector()
    return obtener_detector().info()


@router.post("/inventario/conteo", response_model=ConteoFoto)
def contar_desde_foto(foto: UploadFile = File(...), bd: Session = Depends(obtener_bd)):
    """Recibe una foto de la vitrina, detecta los productos y los compara con el inventario.

    No modifica nada: el inventario se actualiza al confirmar con /inventario/sincronizar.
    """
    datos = np.frombuffer(foto.file.read(), dtype=np.uint8)
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if imagen is None:
        raise HTTPException(400, "No se pudo leer la imagen (si es HEIC del iPhone, usa JPG)")
    alto, ancho = imagen.shape[:2]
    if ancho > ANCHO_MAXIMO:
        imagen = cv2.resize(imagen, (ANCHO_MAXIMO, round(alto * ANCHO_MAXIMO / ancho)),
                            interpolation=cv2.INTER_AREA)
    EVIDENCIAS.mkdir(parents=True, exist_ok=True)
    nombre = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.jpg"
    cv2.imwrite(str(EVIDENCIAS / nombre), imagen, [cv2.IMWRITE_JPEG_QUALITY, 92])
    ruta = f"evidencias/{nombre}"

    imagen, detecciones, conteos, detectables, productos, info, tiempo_ms = analizar(bd, ruta)
    clase_de = {p.categoria_yolo: p.clase_id for p in productos}
    filas = []
    for p in productos:
        detectable = p.categoria_yolo in detectables
        detectado = conteos.get(p.categoria_yolo, 0) if detectable else None
        filas.append(FilaConteo(
            producto_id=p.id, clase_id=p.clase_id, categoria=p.categoria_yolo, nombre=p.nombre,
            detectable=detectable, en_sistema=p.inventario.cantidad, detectado=detectado,
            diferencia=None if detectado is None else detectado - p.inventario.cantidad))
    return ConteoFoto(
        evidencia=ruta, url=f"/archivos/{ruta}", ancho=imagen.shape[1], alto=imagen.shape[0],
        detecciones=[Deteccion(clase_id=clase_de[d["categoria"]], **d)
                     for d in detecciones if d["categoria"] in clase_de],
        filas=filas, modelo=info, tiempo_ms=round(tiempo_ms, 1))


@router.post("/inventario/sincronizar", response_model=list[MovimientoOut])
def sincronizar(datos: Sincronizar, bd: Session = Depends(obtener_bd)):
    """Iguala el inventario a lo que se ve en la foto y registra cada diferencia.

    El conteo se recalcula aquí sobre la misma foto (no se confía en números
    enviados por el navegador). Solo se tocan las categorías que el modelo puede
    detectar. Todo se guarda en una sola transacción.
    """
    _, detecciones, conteos, detectables, _, _, _ = analizar(bd, datos.evidencia)
    creados = registrar_conteo(bd, conteos, detectables, datos.modo, datos.evidencia,
                               {c: confianza_media(detecciones, c) for c in detectables})
    return [movimiento_salida(m, m.producto) for m in creados]


def movimiento_salida(m: Movimiento, p: Producto) -> MovimientoOut:
    total = m.precio_unitario * m.cantidad if m.tipo == "venta" else Decimal(0)
    return MovimientoOut(
        id=m.id, producto_id=p.id, categoria=p.categoria_yolo, nombre=p.nombre, tipo=m.tipo,
        cantidad=m.cantidad, precio_unitario=float(m.precio_unitario), total=float(total),
        fecha_hora=m.fecha_hora, origen=m.origen, confianza=m.confianza,
        evidencia_url=f"/archivos/{m.evidencia}" if m.evidencia else None)


@router.get("/movimientos")
def listar_movimientos(tipo: str | None = None, producto_id: int | None = None,
                       desde: int = 0, limite: int = 50, bd: Session = Depends(obtener_bd)):
    filtro = []
    if tipo:
        filtro.append(Movimiento.tipo == tipo)
    if producto_id:
        filtro.append(Movimiento.producto_id == producto_id)
    total = bd.scalar(select(func.count()).select_from(Movimiento).where(*filtro))
    filas = bd.scalars(select(Movimiento).options(joinedload(Movimiento.producto)).where(*filtro)
                       .order_by(Movimiento.fecha_hora.desc(), Movimiento.id.desc())
                       .offset(desde).limit(min(limite, 200)))
    return {"total": total, "items": [movimiento_salida(m, m.producto) for m in filas]}


@router.get("/panel")
def panel(bd: Session = Depends(obtener_bd)):
    """Datos del panel principal: indicadores, existencias y últimos movimientos."""
    productos = productos_con_inventario(bd)
    hoy = func.date(Movimiento.fecha_hora) == func.current_date()
    ventas_hoy = dict(bd.execute(
        select(Movimiento.producto_id, func.sum(Movimiento.cantidad))
        .where(Movimiento.tipo == "venta", hoy).group_by(Movimiento.producto_id)).all())
    ingresos_hoy = bd.scalar(select(func.coalesce(func.sum(Movimiento.cantidad * Movimiento.precio_unitario), 0))
                             .where(Movimiento.tipo == "venta", hoy))
    ingresos_total = bd.scalar(select(func.coalesce(func.sum(Movimiento.cantidad * Movimiento.precio_unitario), 0))
                               .where(Movimiento.tipo == "venta"))
    ultima_lectura = bd.scalar(select(func.max(Movimiento.fecha_hora)).where(Movimiento.origen == "vision"))
    verificado = bd.scalar(select(func.max(Inventario.actualizado_en)))
    ultimos = bd.scalars(select(Movimiento).options(joinedload(Movimiento.producto))
                         .order_by(Movimiento.fecha_hora.desc(), Movimiento.id.desc()).limit(8))
    return {
        "valor_inventario": float(sum(p.precio * p.inventario.cantidad for p in productos)),
        "unidades": sum(p.inventario.cantidad for p in productos),
        "ventas_hoy": {"unidades": int(sum(ventas_hoy.values())), "ingresos": float(ingresos_hoy)},
        "ingresos_total": float(ingresos_total),
        "ultimo_movimiento_vision": ultima_lectura,
        "ultima_actualizacion": verificado,
        "productos": [{
            "id": p.id, "clase_id": p.clase_id, "categoria": p.categoria_yolo, "nombre": p.nombre,
            "precio": float(p.precio), "cantidad": p.inventario.cantidad,
            "stock_minimo": p.inventario.stock_minimo, "actualizado_en": p.inventario.actualizado_en,
            "vendidas_hoy": int(ventas_hoy.get(p.id, 0)),
        } for p in productos],
        "ultimos_movimientos": [movimiento_salida(m, m.producto) for m in ultimos],
    }
