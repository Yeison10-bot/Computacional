"""Catálogo de productos e inventario: crear, modificar y eliminar productos.

Cada producto es una CLASE del modelo YOLO: `categoria_yolo` es el nombre de la
clase y `clase_id` su número (0, 1, 2...). YOLO exige números seguidos, así que al
eliminar un producto los demás se renumeran. Las etiquetas del dataset apuntan al
producto (no al número), por eso renumerar no las daña.

Cambiar las clases (agregar, renombrar o eliminar) obliga a volver a exportar el
dataset y a reentrenar: el modelo entrenado solo conoce las clases que tenía.
"""

from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, joinedload

from ..db import obtener_bd
from ..models import Anotacion, Imagen, Inventario, Movimiento, Producto
from ..monitor import monitor
from ..schemas import ProductoActualizar, ProductoCrear, ProductoOut

router = APIRouter(prefix="/api/productos", tags=["Catálogo"])

MAXIMO_PRODUCTOS = 8  # más clases exigen muchas más fotos etiquetadas para entrenar bien


def a_salida(p: Producto, bd: Session | None = None) -> ProductoOut:
    usos = {}
    if bd is not None:
        usos = {"cajas": bd.scalar(select(func.count()).select_from(Anotacion).where(Anotacion.producto_id == p.id)),
                "movimientos": bd.scalar(select(func.count()).select_from(Movimiento).where(Movimiento.producto_id == p.id))}
    return ProductoOut(
        id=p.id, clase_id=p.clase_id, categoria_yolo=p.categoria_yolo, nombre=p.nombre,
        precio=float(p.precio), activo=p.activo,
        cantidad=p.inventario.cantidad, stock_minimo=p.inventario.stock_minimo, **usos,
    )


def sin_monitoreo():
    if monitor.activo:
        raise HTTPException(409, "Detén el monitoreo de la vitrina antes de cambiar los productos")


def verificar_categoria_libre(bd: Session, categoria: str, excepto: int | None = None):
    otro = bd.scalar(select(Producto).where(Producto.categoria_yolo == categoria))
    if otro is not None and otro.id != excepto:
        raise HTTPException(409, f"Ya existe un producto con la clase «{categoria}»")


def renumerar(bd: Session):
    """Deja los clase_id seguidos (0, 1, 2...) conservando el orden."""
    productos = bd.scalars(select(Producto).order_by(Producto.clase_id)).all()
    # Primero a negativos para no chocar con la restricción UNIQUE mientras se reasignan.
    for p in productos:
        p.clase_id = -1 - p.clase_id
    bd.flush()
    for i, p in enumerate(productos):
        p.clase_id = i
    bd.flush()


@router.get("", response_model=list[ProductoOut])
def listar(bd: Session = Depends(obtener_bd)):
    productos = bd.scalars(
        select(Producto).options(joinedload(Producto.inventario)).order_by(Producto.clase_id)).all()
    return [a_salida(p, bd) for p in productos]


@router.post("", response_model=ProductoOut, status_code=201)
def crear(datos: ProductoCrear, bd: Session = Depends(obtener_bd)):
    sin_monitoreo()
    total = bd.scalar(select(func.count()).select_from(Producto))
    if total >= MAXIMO_PRODUCTOS:
        raise HTTPException(400, f"Máximo {MAXIMO_PRODUCTOS} productos: con más clases el modelo necesita "
                                 "muchas más fotos etiquetadas para distinguirlas")
    verificar_categoria_libre(bd, datos.categoria_yolo)
    maximo = bd.scalar(select(func.max(Producto.clase_id)))
    p = Producto(clase_id=0 if maximo is None else maximo + 1, categoria_yolo=datos.categoria_yolo,
                 nombre=datos.nombre.strip(), precio=Decimal(str(datos.precio)))
    p.inventario = Inventario(cantidad=0, stock_minimo=datos.stock_minimo)
    bd.add(p)
    bd.commit()
    bd.refresh(p)
    return a_salida(p, bd)


@router.patch("/{producto_id}", response_model=ProductoOut)
def actualizar(producto_id: int, datos: ProductoActualizar, bd: Session = Depends(obtener_bd)):
    p = bd.get(Producto, producto_id)
    if p is None:
        raise HTTPException(404, "Producto no encontrado")

    if datos.categoria_yolo is not None and datos.categoria_yolo != p.categoria_yolo:
        sin_monitoreo()
        verificar_categoria_libre(bd, datos.categoria_yolo, excepto=p.id)
        p.categoria_yolo = datos.categoria_yolo
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
    return a_salida(p, bd)


@router.delete("/{producto_id}", status_code=204)
def eliminar(producto_id: int, forzar: bool = False, bd: Session = Depends(obtener_bd)):
    """Elimina un producto. Si tiene ventas registradas o cajas etiquetadas, pide
    `forzar=true`: se borran sus movimientos y sus cajas, y las imágenes que se
    quedan sin cajas vuelven a «pendiente» (si no, el modelo las tomaría como
    imágenes vacías de verdad)."""
    sin_monitoreo()
    p = bd.get(Producto, producto_id)
    if p is None:
        raise HTTPException(404, "Producto no encontrado")
    if bd.scalar(select(func.count()).select_from(Producto)) <= 1:
        raise HTTPException(400, "Debe quedar al menos un producto")
    salida = a_salida(p, bd)
    if (salida.cajas or salida.movimientos) and not forzar:
        raise HTTPException(409, f"«{p.nombre}» tiene {salida.cajas} cajas etiquetadas y {salida.movimientos} "
                                 "movimientos. Confirma para borrarlos junto con el producto.")

    imagenes = set(bd.scalars(select(Anotacion.imagen_id).where(Anotacion.producto_id == p.id)))
    bd.execute(Anotacion.__table__.delete().where(Anotacion.producto_id == p.id))
    bd.execute(Movimiento.__table__.delete().where(Movimiento.producto_id == p.id))
    if imagenes:
        con_cajas = set(bd.scalars(select(Anotacion.imagen_id).where(Anotacion.imagen_id.in_(imagenes))))
        vacias = imagenes - con_cajas
        if vacias:
            bd.execute(update(Imagen).where(Imagen.id.in_(vacias), Imagen.estado == "etiquetada")
                       .values(estado="pendiente"))
    bd.delete(p.inventario)
    bd.delete(p)
    bd.flush()
    renumerar(bd)
    bd.commit()
