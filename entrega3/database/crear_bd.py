"""Crea la base de datos, sus tablas y el catálogo inicial de los cinco productos.

Ejecutar desde la raíz del proyecto (con el .env configurado):
    .venv\\Scripts\\python.exe database\\crear_bd.py

Es seguro ejecutarlo varias veces: no borra datos ni duplica productos.
También regenera database/schema.sql, el esquema en SQL de PostgreSQL que sirve
como documentación.
"""

import sys
from decimal import Decimal
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from sqlalchemy import create_engine, select, text  # noqa: E402
from sqlalchemy.dialects import postgresql  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.schema import CreateIndex, CreateTable  # noqa: E402

from backend.app.config import url_base_datos  # noqa: E402
from backend.app.db import Base, SesionBD, engine  # noqa: E402
from backend.app.models import Inventario, Producto  # noqa: E402

CATALOGO = [
    # clase_id, categoría YOLO, nombre en la cafetería, precio (COP)
    (0, "botella", "Agua / jugo embotellado", 3500),
    (1, "lata", "Gaseosa en lata", 3000),
    (2, "vaso", "Café para llevar", 4000),
    (3, "paquete", "Snack (papas, galletas)", 2500),
    (4, "fruta", "Fruta", 1500),
]
CANTIDAD_INICIAL = 10
STOCK_MINIMO = 3


def crear_base_si_no_existe():
    url = make_url(url_base_datos())
    if url.get_backend_name() != "postgresql":
        return
    # CREATE DATABASE no puede ir dentro de una transacción: se usa AUTOCOMMIT
    # y una conexión a la base administrativa "postgres".
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as con:
        existe = con.scalar(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": url.database})
        if not existe:
            con.execute(text(f'CREATE DATABASE "{url.database}" ENCODING \'UTF8\''))
            print(f"Base de datos '{url.database}' creada")
        else:
            print(f"Base de datos '{url.database}' ya existía")
    admin.dispose()


def cargar_catalogo():
    with SesionBD() as bd:
        existentes = set(bd.scalars(select(Producto.categoria_yolo)))
        nuevos = 0
        for clase_id, categoria, nombre, precio in CATALOGO:
            if categoria in existentes:
                continue
            p = Producto(clase_id=clase_id, categoria_yolo=categoria, nombre=nombre,
                         precio=Decimal(precio))
            p.inventario = Inventario(cantidad=CANTIDAD_INICIAL, stock_minimo=STOCK_MINIMO)
            bd.add(p)
            nuevos += 1
        bd.commit()
    print(f"Catálogo: {nuevos} producto(s) nuevos, {len(existentes)} ya existían")


def escribir_schema_sql():
    dialecto = postgresql.dialect()
    partes = ["-- Esquema generado automáticamente por database/crear_bd.py a partir de",
              "-- backend/app/models.py. No editar a mano.\n"]
    for tabla in Base.metadata.sorted_tables:
        partes.append(str(CreateTable(tabla).compile(dialect=dialecto)).strip() + ";\n")
        for indice in sorted(tabla.indexes, key=lambda i: i.name):
            partes.append(str(CreateIndex(indice).compile(dialect=dialecto)).strip() + ";\n")
    (RAIZ / "database" / "schema.sql").write_text("\n".join(partes), encoding="utf-8")
    print("database/schema.sql actualizado")


if __name__ == "__main__":
    crear_base_si_no_existe()
    Base.metadata.create_all(engine)
    print(f"Tablas: {', '.join(t.name for t in Base.metadata.sorted_tables)}")
    cargar_catalogo()
    escribir_schema_sql()
