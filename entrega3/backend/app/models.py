"""Tablas de la base de datos (modelos ORM de SQLAlchemy).

Dos grupos de tablas:
  - Negocio: producto, inventario, movimiento.
  - Dataset: sesion_captura, archivo_media, imagen, anotacion.

El archivo database/schema.sql se genera a partir de estas clases, de modo que
el código y la documentación del esquema nunca se desincronizan.
"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (BigInteger, CheckConstraint, DateTime, Float, ForeignKey, Index,
                        Numeric, String, func)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base

# ---------------------------------------------------------------- Negocio


class Producto(Base):
    __tablename__ = "producto"
    __table_args__ = (CheckConstraint("precio >= 0", name="ck_producto_precio"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    clase_id: Mapped[int] = mapped_column(unique=True)  # índice de la clase en YOLO (0-4)
    categoria_yolo: Mapped[str] = mapped_column(String(30), unique=True)
    nombre: Mapped[str] = mapped_column(String(80))
    precio: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    activo: Mapped[bool] = mapped_column(default=True)

    inventario: Mapped["Inventario"] = relationship(back_populates="producto", uselist=False)


class Inventario(Base):
    __tablename__ = "inventario"
    __table_args__ = (
        CheckConstraint("cantidad >= 0", name="ck_inventario_cantidad"),
        CheckConstraint("stock_minimo >= 0", name="ck_inventario_stock_minimo"),
    )

    producto_id: Mapped[int] = mapped_column(ForeignKey("producto.id"), primary_key=True)
    cantidad: Mapped[int] = mapped_column(default=0)
    stock_minimo: Mapped[int] = mapped_column(default=3)
    actualizado_en: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    producto: Mapped[Producto] = relationship(back_populates="inventario")


class Movimiento(Base):
    __tablename__ = "movimiento"
    __table_args__ = (
        CheckConstraint("tipo IN ('venta', 'reposicion', 'ajuste')", name="ck_movimiento_tipo"),
        CheckConstraint("origen IN ('vision', 'manual')", name="ck_movimiento_origen"),
        CheckConstraint("cantidad <> 0", name="ck_movimiento_cantidad"),
        Index("ix_movimiento_fecha", "fecha_hora"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    producto_id: Mapped[int] = mapped_column(ForeignKey("producto.id"))
    tipo: Mapped[str] = mapped_column(String(15))
    # Venta y reposición: unidades (> 0). Ajuste: diferencia con signo (+ o -).
    cantidad: Mapped[int]
    # Precio vigente en el momento del evento: un cambio de precio posterior no
    # altera los ingresos ya registrados.
    precio_unitario: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    fecha_hora: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    origen: Mapped[str] = mapped_column(String(10))
    confianza: Mapped[float | None] = mapped_column(Float)   # confianza media de YOLO
    evidencia: Mapped[str | None] = mapped_column(String(300))  # ruta de la foto del evento

    producto: Mapped[Producto] = relationship()


# ---------------------------------------------------------------- Dataset


class SesionCaptura(Base):
    """Conjunto de tomas con las mismas condiciones (luz, superficie)."""
    __tablename__ = "sesion_captura"

    id: Mapped[int] = mapped_column(primary_key=True)
    codigo: Mapped[str] = mapped_column(String(10), unique=True)  # S01, S02, ...
    descripcion: Mapped[str] = mapped_column(String(200), default="")
    iluminacion: Mapped[str] = mapped_column(String(30), default="")
    superficie: Mapped[str] = mapped_column(String(50), default="")
    # Sesión con objetos reservados: siempre va completa al conjunto de prueba.
    prueba_generalizacion: Mapped[bool] = mapped_column(default=False)
    creado_en: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    archivos: Mapped[list["ArchivoMedia"]] = relationship(
        back_populates="sesion", cascade="all, delete-orphan", passive_deletes=True)
    imagenes: Mapped[list["Imagen"]] = relationship(
        back_populates="sesion", cascade="all, delete-orphan", passive_deletes=True)


class ArchivoMedia(Base):
    """Video o foto original subido desde la interfaz."""
    __tablename__ = "archivo_media"
    __table_args__ = (
        CheckConstraint("tipo IN ('video', 'foto')", name="ck_archivo_tipo"),
        CheckConstraint("estado IN ('subido', 'procesando', 'procesado', 'error')",
                        name="ck_archivo_estado"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    sesion_id: Mapped[int] = mapped_column(ForeignKey("sesion_captura.id", ondelete="CASCADE"))
    tipo: Mapped[str] = mapped_column(String(10))
    nombre_original: Mapped[str] = mapped_column(String(200))
    ruta: Mapped[str] = mapped_column(String(300), unique=True)  # relativa a la carpeta datos/
    tamano_bytes: Mapped[int] = mapped_column(BigInteger)
    estado: Mapped[str] = mapped_column(String(12), default="subido")
    detalle: Mapped[str | None] = mapped_column(String(300))  # resumen de la extracción o error
    creado_en: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    sesion: Mapped[SesionCaptura] = relationship(back_populates="archivos")
    imagenes: Mapped[list["Imagen"]] = relationship(
        back_populates="archivo", cascade="all, delete-orphan", passive_deletes=True)


class Imagen(Base):
    """Imagen candidata al dataset: un fotograma extraído o una foto procesada."""
    __tablename__ = "imagen"
    __table_args__ = (
        CheckConstraint("estado IN ('pendiente', 'etiquetada', 'descartada')", name="ck_imagen_estado"),
        CheckConstraint("particion IS NULL OR particion IN ('train', 'val', 'test')",
                        name="ck_imagen_particion"),
        Index("ix_imagen_sesion_estado", "sesion_id", "estado"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    sesion_id: Mapped[int] = mapped_column(ForeignKey("sesion_captura.id", ondelete="CASCADE"))
    archivo_id: Mapped[int] = mapped_column(ForeignKey("archivo_media.id", ondelete="CASCADE"))
    ruta: Mapped[str] = mapped_column(String(300), unique=True)  # relativa a la carpeta datos/
    ancho: Mapped[int]
    alto: Mapped[int]
    estado: Mapped[str] = mapped_column(String(12), default="pendiente")
    particion: Mapped[str | None] = mapped_column(String(5))  # asignada al exportar
    creado_en: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    sesion: Mapped[SesionCaptura] = relationship(back_populates="imagenes")
    archivo: Mapped[ArchivoMedia] = relationship(back_populates="imagenes")
    anotaciones: Mapped[list["Anotacion"]] = relationship(
        back_populates="imagen", cascade="all, delete-orphan", passive_deletes=True)


class Anotacion(Base):
    """Caja de un objeto en una imagen, en coordenadas YOLO normalizadas (0 a 1)."""
    __tablename__ = "anotacion"
    __table_args__ = (
        CheckConstraint("x_centro BETWEEN 0 AND 1 AND y_centro BETWEEN 0 AND 1", name="ck_anotacion_centro"),
        CheckConstraint("ancho > 0 AND ancho <= 1 AND alto > 0 AND alto <= 1", name="ck_anotacion_tamano"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    imagen_id: Mapped[int] = mapped_column(ForeignKey("imagen.id", ondelete="CASCADE"), index=True)
    producto_id: Mapped[int] = mapped_column(ForeignKey("producto.id"))
    x_centro: Mapped[float]
    y_centro: Mapped[float]
    ancho: Mapped[float]
    alto: Mapped[float]

    imagen: Mapped[Imagen] = relationship(back_populates="anotaciones")
    producto: Mapped[Producto] = relationship()
