"""Regla de negocio única para pasar de "lo que ve la cámara" a movimientos de inventario.

La usan el conteo manual por foto (/api/inventario/sincronizar) y el monitoreo
continuo (monitor.py), así ambos registran ventas y reposiciones igual.
"""

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from .models import Movimiento, Producto


def registrar_conteo(bd: Session, conteos: dict, categorias, modo: str, evidencia: str | None,
                     confianzas: dict | None = None) -> list[Movimiento]:
    """Iguala el inventario de `categorias` a `conteos` y registra cada diferencia.

    modo "movimientos": bajada → venta (un retiro = una venta), subida → reposición.
    modo "ajuste": la diferencia se registra como ajuste (conteo inicial o corrección).
    La comparación es contra la base de datos, así una diferencia ya registrada
    (por otra foto o por el monitoreo) no se vuelve a contar.
    Hace commit y devuelve los movimientos creados.
    """
    confianzas = confianzas or {}
    productos = bd.scalars(select(Producto).options(joinedload(Producto.inventario))).all()
    creados = []
    for p in productos:
        if p.categoria_yolo not in categorias:
            continue
        visto = conteos.get(p.categoria_yolo, 0)
        diferencia = visto - p.inventario.cantidad
        p.inventario.actualizado_en = func.now()  # marca la verificación aunque no haya cambios
        if diferencia == 0:
            continue
        if modo == "ajuste":
            tipo, cantidad = "ajuste", diferencia
        elif diferencia < 0:
            tipo, cantidad = "venta", -diferencia
        else:
            tipo, cantidad = "reposicion", diferencia
        mov = Movimiento(producto_id=p.id, tipo=tipo, cantidad=cantidad, precio_unitario=p.precio,
                         origen="vision", evidencia=evidencia, confianza=confianzas.get(p.categoria_yolo))
        mov.producto = p
        bd.add(mov)
        p.inventario.cantidad = visto
        creados.append(mov)
    bd.commit()
    for m in creados:
        bd.refresh(m)  # trae la fecha_hora asignada por la base de datos
    return creados


def confianza_media(detecciones, categoria):
    valores = [d["confianza"] for d in detecciones if d["categoria"] == categoria]
    return round(sum(valores) / len(valores), 3) if valores else None
