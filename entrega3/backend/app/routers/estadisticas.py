"""Estadísticas de ventas y reinicio de los datos de prueba."""

from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import delete, extract, func, select
from sqlalchemy.orm import Session

from ..db import obtener_bd
from ..models import Inventario, Movimiento, Producto
from ..monitor import monitor

router = APIRouter(prefix="/api", tags=["Estadísticas"])


@router.get("/estadisticas")
def estadisticas(dias: int = 30, bd: Session = Depends(obtener_bd)):
    """Ventas de los últimos `dias` días: por producto, por día y por hora del día."""
    dias = max(1, min(dias, 365))
    desde = date.today() - timedelta(days=dias - 1)
    ventas = (Movimiento.tipo == "venta", func.date(Movimiento.fecha_hora) >= desde)
    ingreso = func.sum(Movimiento.cantidad * Movimiento.precio_unitario)

    productos = bd.scalars(select(Producto).order_by(Producto.clase_id)).all()
    por_producto = {pid: (int(u), float(i)) for pid, u, i in bd.execute(
        select(Movimiento.producto_id, func.sum(Movimiento.cantidad), ingreso).where(*ventas)
        .group_by(Movimiento.producto_id)).all()}

    dia = func.date(Movimiento.fecha_hora)
    por_dia = {str(d): (int(u), float(i)) for d, u, i in bd.execute(
        select(dia, func.sum(Movimiento.cantidad), ingreso).where(*ventas).group_by(dia)).all()}

    hora = extract("hour", Movimiento.fecha_hora)
    por_hora = {int(h): (int(u), float(i)) for h, u, i in bd.execute(
        select(hora, func.sum(Movimiento.cantidad), ingreso).where(*ventas).group_by(hora)).all()}

    filas = [{"producto_id": p.id, "clase_id": p.clase_id, "categoria": p.categoria_yolo, "nombre": p.nombre,
              "unidades": por_producto.get(p.id, (0, 0))[0], "ingresos": por_producto.get(p.id, (0, 0.0))[1]}
             for p in productos]
    mas_vendido = max(filas, key=lambda f: f["unidades"]) if any(f["unidades"] for f in filas) else None
    return {
        "dias": dias, "desde": desde.isoformat(),
        "unidades": sum(f["unidades"] for f in filas), "ingresos": sum(f["ingresos"] for f in filas),
        "mas_vendido": mas_vendido,
        "por_producto": filas,
        # Todos los días del rango, también los que no tuvieron ventas (si no, la gráfica engaña).
        "por_dia": [{"fecha": (d := (desde + timedelta(days=i)).isoformat()),
                     "unidades": por_dia.get(d, (0, 0))[0], "ingresos": por_dia.get(d, (0, 0.0))[1]}
                    for i in range(dias)],
        "por_hora": [{"hora": h, "unidades": por_hora.get(h, (0, 0))[0], "ingresos": por_hora.get(h, (0, 0.0))[1]}
                     for h in range(24)],
    }


class Reinicio(BaseModel):
    confirmacion: str = Field(description='debe ser "REINICIAR"')


@router.post("/inventario/reiniciar")
def reiniciar(datos: Reinicio, bd: Session = Depends(obtener_bd)):
    """Borra todos los movimientos y deja el inventario en 0 (para empezar la demostración).

    El catálogo, los precios, el dataset y los modelos no se tocan. Al iniciar el
    monitoreo, la cámara registra lo que hay en la vitrina como conteo inicial.
    """
    if datos.confirmacion != "REINICIAR":
        raise HTTPException(400, 'Escribe REINICIAR para confirmar')
    if monitor.activo:
        raise HTTPException(409, "Detén el monitoreo antes de reiniciar los datos")
    borrados = bd.execute(delete(Movimiento)).rowcount
    bd.execute(Inventario.__table__.update().values(cantidad=0, actualizado_en=func.now()))
    bd.commit()
    return {"movimientos_borrados": borrados}
