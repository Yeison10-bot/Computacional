"""Formato de los datos que entran y salen de la API (validados con Pydantic)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- Catálogo


class ProductoOut(BaseModel):
    id: int
    clase_id: int
    categoria_yolo: str
    nombre: str
    precio: float
    activo: bool
    cantidad: int
    stock_minimo: int
    cajas: int = 0          # cajas etiquetadas en el dataset
    movimientos: int = 0    # ventas, reposiciones y ajustes registrados


# Nombre de la clase en YOLO: minúsculas, números y guion bajo (p. ej. "taza_roja").
PATRON_CATEGORIA = r"^[a-z][a-z0-9_]{1,29}$"


class ProductoCrear(BaseModel):
    categoria_yolo: str = Field(pattern=PATRON_CATEGORIA)
    nombre: str = Field(min_length=1, max_length=80)
    precio: float = Field(ge=0)
    stock_minimo: int = Field(3, ge=0)


class ProductoActualizar(BaseModel):
    categoria_yolo: str | None = Field(None, pattern=PATRON_CATEGORIA)
    nombre: str | None = Field(None, min_length=1, max_length=80)
    precio: float | None = Field(None, ge=0)
    cantidad: int | None = Field(None, ge=0)
    stock_minimo: int | None = Field(None, ge=0)


# ---------------------------------------------------------------- Inventario por visión


class Deteccion(BaseModel):
    categoria: str
    clase_id: int
    clase_original: str  # nombre en el modelo (con el modelo COCO, p. ej. "bottle")
    confianza: float
    x_centro: float
    y_centro: float
    ancho: float
    alto: float


class FilaConteo(BaseModel):
    producto_id: int
    clase_id: int
    categoria: str
    nombre: str
    detectable: bool        # el modelo actual puede ver esta categoría
    en_sistema: int         # existencias registradas antes de la foto
    detectado: int | None   # None si la categoría no es detectable
    diferencia: int | None  # detectado − en_sistema


class ConteoFoto(BaseModel):
    evidencia: str  # ruta relativa a datos/, se reenvía al sincronizar
    url: str
    ancho: int
    alto: int
    detecciones: list[Deteccion]
    filas: list[FilaConteo]
    modelo: dict
    tiempo_ms: float


class Sincronizar(BaseModel):
    evidencia: str = Field(pattern=r"^evidencias/[\w.-]+\.jpg$")
    # movimientos: bajadas = ventas, subidas = reposiciones (regla del profesor).
    # ajuste: conteo inicial o corrección, no cuenta como venta.
    modo: Literal["movimientos", "ajuste"] = "movimientos"


class MovimientoOut(BaseModel):
    id: int
    producto_id: int
    categoria: str
    nombre: str
    tipo: str
    cantidad: int
    precio_unitario: float
    total: float  # ingreso de la venta (0 en reposiciones y ajustes)
    fecha_hora: datetime
    origen: str
    confianza: float | None
    evidencia_url: str | None


# ---------------------------------------------------------------- Dataset


class SesionCrear(BaseModel):
    codigo: str | None = Field(None, pattern=r"^S\d{2,3}$")  # si no se envía, se asigna el siguiente
    descripcion: str = Field("", max_length=200)
    iluminacion: str = Field("", max_length=30)
    superficie: str = Field("", max_length=50)
    prueba_generalizacion: bool = False


class SesionOut(BaseModel):
    id: int
    codigo: str
    descripcion: str
    iluminacion: str
    superficie: str
    prueba_generalizacion: bool
    creado_en: datetime
    videos: int = 0
    fotos: int = 0
    imagenes: int = 0
    etiquetadas: int = 0
    pendientes: int = 0


class ArchivoOut(BaseModel):
    id: int
    sesion_id: int
    tipo: str
    nombre_original: str
    url: str
    tamano_bytes: int
    estado: str
    detalle: str | None
    imagenes: int
    creado_en: datetime


class Caja(BaseModel):
    clase_id: int = Field(ge=0)
    x_centro: float = Field(ge=0, le=1)
    y_centro: float = Field(ge=0, le=1)
    ancho: float = Field(gt=0, le=1)
    alto: float = Field(gt=0, le=1)


class ImagenOut(BaseModel):
    id: int
    sesion_id: int
    sesion_codigo: str
    url: str
    ancho: int
    alto: int
    estado: str
    particion: str | None
    cajas: int


class ImagenDetalle(ImagenOut):
    anotaciones: list[Caja]
    anterior_id: int | None   # navegación dentro de la misma sesión
    siguiente_id: int | None


class GuardarAnotaciones(BaseModel):
    anotaciones: list[Caja]
    estado: Literal["pendiente", "etiquetada", "descartada"] = "etiquetada"


class ExportarParametros(BaseModel):
    val: float = Field(0.2, ge=0.05, le=0.5)
    test: float = Field(0.1, ge=0, le=0.5)
    semilla: int = 42
