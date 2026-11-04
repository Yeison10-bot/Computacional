"""Catálogo de productos e inventario."""

from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from ..db import obtener_bd
from ..models import Movimiento, Producto
from ..schemas import ProductoActualizar, ProductoOut

router = APIRouter(prefix="/api/productos", tags=["Catálogo"])


def a_salida(p: Producto) -> ProductoOut:
    return ProductoOut(
        id=p.id, clase_id=p.clase_id, categoria_yolo=p.categoria_yolo, nombre=p.nombre,
        precio=float(p.precio), activo=p.activo,
        cantidad=p.inventario.cantidad, stock_minimo=p.inventario.stock_minimo,
    )


@router.get("", response_model=list[ProductoOut])
def listar(bd: Session = Depends(obtener_bd)):
    productos = bd.scalars(
        select(Producto).options(joinedload(Producto.inventario)).order_by(Producto.clase_id))
    return [a_salida(p) for p in productos]


@router.patch("/{producto_id}", response_model=ProductoOut)
def actualizar(producto_id: int, datos: ProductoActualizar, bd: Session = Depends(obtener_bd)):
    p = bd.get(Producto, producto_id)
    if p is None:
        raise HTTPException(404, "Producto no encontrado")

    if datos.nombre is not None:
        p.nombre = datos.nombre.strip()
    if datos.precio is not None:
        p.precio = Decimal(str(datos.precio))
    if datos.stock_minimo is not None:
        p.inventario.stock_minimo = datos.stock_minimo
    if datos.cantidad is not None and datos.cantidad != p.inventario.cantidad:
        # Un cambio manual de existencias queda en el historial como ajuste,
        # así el inventario siempre se puede reconstruir a partir de los movimientos.
        bd.add(Movimiento(
            producto_id=p.id, tipo="ajuste", cantidad=datos.cantidad - p.inventario.cantidad,
            precio_unitario=p.precio, origen="manual",
        ))
        p.inventario.cantidad = datos.cantidad

    bd.commit()
    bd.refresh(p)
    return a_salida(p)
