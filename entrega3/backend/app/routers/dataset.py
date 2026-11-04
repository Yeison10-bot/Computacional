"""Sesiones de captura, subida de videos y fotos, imágenes y anotaciones."""

import re

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from .. import servicios
from ..config import CRUDOS_FOTOS, CRUDOS_VIDEOS
from ..db import obtener_bd
from ..models import Anotacion, ArchivoMedia, Imagen, Producto, SesionCaptura
from ..schemas import (ArchivoOut, Caja, GuardarAnotaciones, ImagenDetalle, ImagenOut,
                       SesionCrear, SesionOut)

router = APIRouter(prefix="/api", tags=["Dataset"])


def url(ruta: str) -> str:
    return f"/archivos/{ruta}"


def obtener(bd: Session, modelo, id_: int, nombre: str):
    obj = bd.get(modelo, id_)
    if obj is None:
        raise HTTPException(404, f"{nombre} no encontrada")
    return obj


# ---------------------------------------------------------------- Sesiones


def conteos_por_sesion(bd: Session) -> dict:
    conteos = {}
    for sid, videos, fotos in bd.execute(
            select(ArchivoMedia.sesion_id,
                   func.count(case((ArchivoMedia.tipo == "video", 1))),
                   func.count(case((ArchivoMedia.tipo == "foto", 1))))
            .group_by(ArchivoMedia.sesion_id)):
        conteos[sid] = {"videos": videos, "fotos": fotos}
    for sid, total, etiquetadas, pendientes in bd.execute(
            select(Imagen.sesion_id, func.count(),
                   func.count(case((Imagen.estado == "etiquetada", 1))),
                   func.count(case((Imagen.estado == "pendiente", 1))))
            .group_by(Imagen.sesion_id)):
        conteos.setdefault(sid, {}).update(
            imagenes=total, etiquetadas=etiquetadas, pendientes=pendientes)
    return conteos


def sesion_salida(s: SesionCaptura, conteos: dict) -> SesionOut:
    return SesionOut(id=s.id, codigo=s.codigo, descripcion=s.descripcion, iluminacion=s.iluminacion,
                     superficie=s.superficie, prueba_generalizacion=s.prueba_generalizacion,
                     creado_en=s.creado_en, **conteos.get(s.id, {}))


@router.get("/sesiones", response_model=list[SesionOut])
def listar_sesiones(bd: Session = Depends(obtener_bd)):
    conteos = conteos_por_sesion(bd)
    return [sesion_salida(s, conteos)
            for s in bd.scalars(select(SesionCaptura).order_by(SesionCaptura.codigo))]


@router.get("/sesiones/{sesion_id}", response_model=SesionOut)
def ver_sesion(sesion_id: int, bd: Session = Depends(obtener_bd)):
    return sesion_salida(obtener(bd, SesionCaptura, sesion_id, "Sesión"), conteos_por_sesion(bd))


@router.post("/sesiones", response_model=SesionOut, status_code=201)
def crear_sesion(datos: SesionCrear, bd: Session = Depends(obtener_bd)):
    codigo = datos.codigo
    if codigo is None:
        numeros = [int(m.group(1)) for c in bd.scalars(select(SesionCaptura.codigo))
                   if (m := re.fullmatch(r"S(\d+)", c))]
        codigo = f"S{max(numeros, default=0) + 1:02d}"
    elif bd.scalar(select(SesionCaptura).where(SesionCaptura.codigo == codigo)):
        raise HTTPException(409, f"Ya existe la sesión {codigo}")
    s = SesionCaptura(codigo=codigo, **datos.model_dump(exclude={"codigo"}))
    bd.add(s)
    bd.commit()
    return sesion_salida(s, {})


@router.delete("/sesiones/{sesion_id}", status_code=204)
def borrar_sesion(sesion_id: int, bd: Session = Depends(obtener_bd)):
    s = obtener(bd, SesionCaptura, sesion_id, "Sesión")
    for img in s.imagenes:
        servicios.borrar_archivo(img.ruta)
    for a in s.archivos:
        servicios.borrar_archivo(a.ruta)
    bd.delete(s)
    bd.commit()


# ---------------------------------------------------------------- Archivos (videos y fotos)


def archivo_salida(a: ArchivoMedia, n_imagenes: int) -> ArchivoOut:
    return ArchivoOut(id=a.id, sesion_id=a.sesion_id, tipo=a.tipo, nombre_original=a.nombre_original,
                      url=url(a.ruta), tamano_bytes=a.tamano_bytes, estado=a.estado, detalle=a.detalle,
                      imagenes=n_imagenes, creado_en=a.creado_en)


@router.get("/sesiones/{sesion_id}/archivos", response_model=list[ArchivoOut])
def listar_archivos(sesion_id: int, bd: Session = Depends(obtener_bd)):
    obtener(bd, SesionCaptura, sesion_id, "Sesión")
    n = dict(bd.execute(select(Imagen.archivo_id, func.count())
                        .where(Imagen.sesion_id == sesion_id).group_by(Imagen.archivo_id)).all())
    archivos = bd.scalars(select(ArchivoMedia).where(ArchivoMedia.sesion_id == sesion_id)
                          .order_by(ArchivoMedia.id))
    return [archivo_salida(a, n.get(a.id, 0)) for a in archivos]


@router.post("/sesiones/{sesion_id}/archivos", status_code=201)
def subir_archivos(sesion_id: int, tareas: BackgroundTasks,
                   archivos: list[UploadFile] = File(...), bd: Session = Depends(obtener_bd)):
    """Recibe videos y fotos. Las fotos se procesan al instante; los videos en segundo plano."""
    sesion = obtener(bd, SesionCaptura, sesion_id, "Sesión")
    aceptados, rechazados = [], []
    for subido in archivos:
        ext = "." + subido.filename.rsplit(".", 1)[-1].lower() if "." in subido.filename else ""
        if ext in servicios.EXT_HEIC:
            rechazados.append({"nombre": subido.filename, "motivo":
                               "Formato HEIC: en el iPhone activa Ajustes → Cámara → Formatos → "
                               "«Más compatible» o expórtala como JPG"})
            continue
        if ext in servicios.EXT_VIDEO:
            tipo, carpeta = "video", CRUDOS_VIDEOS / sesion.codigo
        elif ext in servicios.EXT_FOTO:
            tipo, carpeta = "foto", CRUDOS_FOTOS / sesion.codigo
        else:
            rechazados.append({"nombre": subido.filename, "motivo": f"Extensión no admitida ({ext or 'sin extensión'})"})
            continue

        destino = servicios.guardar_subida(subido, carpeta)
        archivo = ArchivoMedia(sesion_id=sesion.id, tipo=tipo, nombre_original=subido.filename,
                               ruta=servicios.relativa(destino), tamano_bytes=destino.stat().st_size)
        bd.add(archivo)
        bd.flush()  # asigna el id, que se usa en el nombre de las imágenes

        if tipo == "foto":
            try:
                bd.add(servicios.procesar_foto(archivo, sesion.codigo))
                archivo.estado, archivo.detalle = "procesado", "1 imagen"
            except ValueError as e:
                archivo.estado, archivo.detalle = "error", str(e)
        else:
            tareas.add_task(servicios.extraer_video, archivo.id)
        aceptados.append({"nombre": subido.filename, "tipo": tipo})
    bd.commit()  # las tareas en segundo plano corren después de este commit
    return {"aceptados": aceptados, "rechazados": rechazados}


@router.post("/archivos/{archivo_id}/extraer", status_code=202)
def reextraer(archivo_id: int, tareas: BackgroundTasks, intervalo: float | None = None,
              max_movimiento: float | None = None, min_cambio: float | None = None,
              bd: Session = Depends(obtener_bd)):
    """Vuelve a extraer los fotogramas de un video, opcionalmente con otros umbrales."""
    a = obtener(bd, ArchivoMedia, archivo_id, "Archivo")
    if a.tipo != "video":
        raise HTTPException(400, "Solo los videos se pueden volver a extraer")
    if a.estado == "procesando":
        raise HTTPException(409, "El video ya se está procesando")
    parametros = {k: v for k, v in {"intervalo": intervalo, "max_movimiento": max_movimiento,
                                    "min_cambio": min_cambio}.items() if v is not None}
    a.estado = "procesando"
    bd.commit()
    tareas.add_task(servicios.extraer_video, a.id, parametros)
    return {"estado": "procesando"}


@router.delete("/archivos/{archivo_id}", status_code=204)
def borrar_archivo(archivo_id: int, bd: Session = Depends(obtener_bd)):
    a = obtener(bd, ArchivoMedia, archivo_id, "Archivo")
    for img in a.imagenes:
        servicios.borrar_archivo(img.ruta)
    servicios.borrar_archivo(a.ruta)
    bd.delete(a)
    bd.commit()


# ---------------------------------------------------------------- Imágenes y anotaciones


def imagen_salida(img: Imagen, codigo: str, n_cajas: int) -> ImagenOut:
    return ImagenOut(id=img.id, sesion_id=img.sesion_id, sesion_codigo=codigo, url=url(img.ruta),
                     ancho=img.ancho, alto=img.alto, estado=img.estado, particion=img.particion,
                     cajas=n_cajas)


@router.get("/imagenes")
def listar_imagenes(sesion_id: int | None = None, estado: str | None = None,
                    desde: int = 0, limite: int = 120, bd: Session = Depends(obtener_bd)):
    filtro = []
    if sesion_id is not None:
        filtro.append(Imagen.sesion_id == sesion_id)
    if estado:
        filtro.append(Imagen.estado == estado)
    total = bd.scalar(select(func.count()).select_from(Imagen).where(*filtro))
    n_cajas = (select(func.count()).where(Anotacion.imagen_id == Imagen.id)
               .correlate(Imagen).scalar_subquery())
    filas = bd.execute(select(Imagen, SesionCaptura.codigo, n_cajas).join(Imagen.sesion)
                       .where(*filtro).order_by(Imagen.id).offset(desde).limit(min(limite, 500)))
    return {"total": total, "items": [imagen_salida(i, c, n) for i, c, n in filas]}


@router.get("/imagenes/siguiente-pendiente")
def siguiente_pendiente(sesion_id: int | None = None, bd: Session = Depends(obtener_bd)):
    consulta = select(Imagen.id).where(Imagen.estado == "pendiente").order_by(Imagen.id).limit(1)
    if sesion_id is not None:
        consulta = consulta.where(Imagen.sesion_id == sesion_id)
    return {"id": bd.scalar(consulta)}


def detalle(bd: Session, img: Imagen) -> ImagenDetalle:
    misma_sesion = Imagen.sesion_id == img.sesion_id
    anterior = bd.scalar(select(func.max(Imagen.id)).where(misma_sesion, Imagen.id < img.id))
    siguiente = bd.scalar(select(func.min(Imagen.id)).where(misma_sesion, Imagen.id > img.id))
    cajas = [Caja(clase_id=a.producto.clase_id, x_centro=a.x_centro, y_centro=a.y_centro,
                  ancho=a.ancho, alto=a.alto) for a in img.anotaciones]
    base = imagen_salida(img, img.sesion.codigo, len(cajas))
    return ImagenDetalle(**base.model_dump(), anotaciones=cajas,
                         anterior_id=anterior, siguiente_id=siguiente)


@router.get("/imagenes/{imagen_id}", response_model=ImagenDetalle)
def ver_imagen(imagen_id: int, bd: Session = Depends(obtener_bd)):
    return detalle(bd, obtener(bd, Imagen, imagen_id, "Imagen"))


@router.put("/imagenes/{imagen_id}/anotaciones", response_model=ImagenDetalle)
def guardar_anotaciones(imagen_id: int, datos: GuardarAnotaciones, bd: Session = Depends(obtener_bd)):
    """Reemplaza todas las cajas de la imagen y actualiza su estado.

    Una imagen 'etiquetada' sin cajas es válida: es un ejemplo negativo (vitrina vacía).
    """
    img = obtener(bd, Imagen, imagen_id, "Imagen")
    productos = dict(bd.execute(select(Producto.clase_id, Producto.id)).all())
    desconocidas = {c.clase_id for c in datos.anotaciones} - productos.keys()
    if desconocidas:
        raise HTTPException(422, f"Clases inexistentes: {sorted(desconocidas)}")

    img.anotaciones.clear()
    if datos.estado != "descartada":
        img.anotaciones.extend(
            Anotacion(producto_id=productos[c.clase_id], x_centro=c.x_centro, y_centro=c.y_centro,
                      ancho=c.ancho, alto=c.alto) for c in datos.anotaciones)
    img.estado = datos.estado
    bd.commit()
    bd.refresh(img)
    return detalle(bd, img)


@router.delete("/imagenes/{imagen_id}", status_code=204)
def borrar_imagen(imagen_id: int, bd: Session = Depends(obtener_bd)):
    img = obtener(bd, Imagen, imagen_id, "Imagen")
    servicios.borrar_archivo(img.ruta)
    bd.delete(img)
    bd.commit()
