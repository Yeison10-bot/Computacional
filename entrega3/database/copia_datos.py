"""Copia todos los datos de la app de una base PostgreSQL a otra mediante un archivo JSON.

Ejecutar desde la raíz del proyecto (con el .env configurado):
    .venv\\Scripts\\python.exe database\\copia_datos.py exportar <archivo.json>
    .venv\\Scripts\\python.exe database\\copia_datos.py importar <archivo.json>

exportar solo lee: guarda las filas de todas las tablas en el JSON, que no depende de la
versión de PostgreSQL. importar es para una instalación nueva, después de database\\crear_bd.py:
reemplaza el catálogo sembrado por el contenido del archivo, todo en una sola transacción.
Si la base ya tiene datos de uso (sesiones, imágenes, movimientos...), no toca nada.

Códigos de salida: 0 bien, 1 error (no se cambió nada), 2 la base ya tenía datos.
"""

import json
import os
import sys
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from sqlalchemy import func, inspect, select, text  # noqa: E402
from sqlalchemy.exc import DBAPIError  # noqa: E402

from backend.app import models  # noqa: E402,F401  (registra las tablas en Base.metadata)
from backend.app.db import Base, engine  # noqa: E402

FORMATO = 1
TABLAS = Base.metadata.sorted_tables  # en orden de claves foráneas: primero las referenciadas
SEMBRADAS = {"producto", "inventario"}  # las que llena crear_bd.py
ESPERA_BLOQUEO = "15s"


class ErrorCopia(Exception):
    pass


class BaseConDatos(Exception):
    pass


def a_json(valor):
    if isinstance(valor, (datetime, date, time)):
        return valor.isoformat()
    if isinstance(valor, Decimal):
        return str(valor)
    return valor


def desde_json(columna, valor):
    if valor is None:
        return None
    try:
        tipo = columna.type.python_type
    except NotImplementedError:
        return valor
    if tipo in (datetime, date, time):
        return tipo.fromisoformat(valor)
    if tipo is Decimal:
        return Decimal(str(valor))
    if tipo is float:
        return float(valor)
    return valor


def exportar(archivo):
    salida = {"formato": FORMATO, "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
              "tablas": {}}
    # Una sola instantánea de solo lectura para todas las tablas: si la app escribe mientras
    # se exporta, el archivo no queda con, p. ej., anotaciones de una imagen que no se copió.
    with engine.connect().execution_options(isolation_level="REPEATABLE READ",
                                            postgresql_readonly=True) as con:
        for tabla in TABLAS:
            consulta = select(tabla).order_by(*tabla.primary_key.columns)
            salida["tablas"][tabla.name] = [
                {c.name: a_json(fila[c.name]) for c in tabla.columns}
                for fila in con.execute(consulta).mappings()]
    archivo.parent.mkdir(parents=True, exist_ok=True)
    temporal = archivo.with_name(archivo.name + ".tmp")
    try:
        with temporal.open("w", encoding="utf-8") as f:
            json.dump(salida, f, ensure_ascii=False, indent=1)
        os.replace(temporal, archivo)  # un corte a mitad no deja un JSON a medias
    finally:
        temporal.unlink(missing_ok=True)
    for nombre, filas in salida["tablas"].items():
        print(f"  {nombre}: {len(filas)}")
    print(f"Datos exportados a {archivo}")


def leer_archivo(archivo):
    if not archivo.is_file():
        raise ErrorCopia(f"no existe el archivo {archivo}")
    try:
        datos = json.loads(archivo.read_text(encoding="utf-8-sig"))
    except ValueError as e:
        raise ErrorCopia(f"{archivo} no es un JSON válido ({e})") from None
    if not isinstance(datos, dict) or datos.get("formato") != FORMATO:
        raise ErrorCopia(f"{archivo} no es una copia de datos de esta app (formato {FORMATO})")
    tablas = datos.get("tablas")
    esperadas = {t.name for t in TABLAS}
    if not isinstance(tablas, dict) or set(tablas) != esperadas:
        recibidas = set(tablas) if isinstance(tablas, dict) else set()
        raise ErrorCopia(f"las tablas del archivo no coinciden con las de la app "
                         f"(faltan: {sorted(esperadas - recibidas)}, sobran: {sorted(recibidas - esperadas)})")
    filas_por_tabla = {}
    for tabla in TABLAS:
        columnas = {c.name: c for c in tabla.columns}
        filas = tablas[tabla.name]
        if not isinstance(filas, list):
            raise ErrorCopia(f"tabla {tabla.name}: se esperaba una lista de filas")
        convertidas = []
        for i, fila in enumerate(filas, 1):
            if not isinstance(fila, dict) or set(fila) != set(columnas):
                raise ErrorCopia(f"tabla {tabla.name}, fila {i}: sus columnas no coinciden con las de la app")
            try:
                convertidas.append({n: desde_json(columnas[n], v) for n, v in fila.items()})
            except (TypeError, ValueError, ArithmeticError) as e:
                raise ErrorCopia(f"tabla {tabla.name}, fila {i}: valor inválido ({e})") from None
        filas_por_tabla[tabla.name] = convertidas
    return filas_por_tabla


def reajustar_secuencias(con, tabla):
    """Deja la secuencia del id en el máximo copiado (o lista para el 1 si la tabla quedó vacía)."""
    prep = con.dialect.identifier_preparer
    for columna in tabla.primary_key.columns:
        secuencia = con.scalar(text("SELECT pg_get_serial_sequence(:t, :c)"),
                               {"t": prep.format_table(tabla), "c": columna.name})
        if secuencia:
            c = prep.format_column(columna)
            con.execute(text(f"SELECT setval(CAST(:s AS regclass), COALESCE(MAX({c}), 1), "
                             f"MAX({c}) IS NOT NULL) FROM {prep.format_table(tabla)}"), {"s": secuencia})


def importar(archivo):
    filas_por_tabla = leer_archivo(archivo)  # se valida todo antes de tocar la base
    with engine.begin() as con:  # una transacción: cualquier error deshace todo
        base = con.engine.url.database
        faltan = {t.name for t in TABLAS} - set(inspect(con).get_table_names())
        if faltan:
            raise ErrorCopia(f"a la base '{base}' le faltan las tablas {sorted(faltan)}; "
                             f"ejecute primero database\\crear_bd.py")
        # Bloqueo contra escritores hasta el commit: entre la revisión y la carga nadie
        # (la app u otra importación simultánea) puede meter filas que luego se pisarían.
        prep = con.dialect.identifier_preparer
        con.execute(text(f"SET LOCAL lock_timeout = '{ESPERA_BLOQUEO}'"))
        con.execute(text(f"LOCK TABLE {', '.join(prep.format_table(t) for t in TABLAS)} IN EXCLUSIVE MODE"))
        con_datos = {}
        for tabla in TABLAS:
            if tabla.name not in SEMBRADAS:
                n = con.scalar(select(func.count()).select_from(tabla))
                if n:
                    con_datos[tabla.name] = n
        if con_datos:
            detalle = ", ".join(f"{t}: {n}" for t, n in con_datos.items())
            raise BaseConDatos(
                f"La base '{base}' ya tiene datos de uso ({detalle}).\n"
                f"No se importó nada, para no sobrescribirlos. La importación es solo para una base "
                f"nueva recién creada con database\\crear_bd.py.")
        for tabla in reversed(TABLAS):
            con.execute(tabla.delete())
        for tabla in TABLAS:
            if filas_por_tabla[tabla.name]:
                con.execute(tabla.insert(), filas_por_tabla[tabla.name])
            reajustar_secuencias(con, tabla)
    for nombre, filas in filas_por_tabla.items():
        print(f"  {nombre}: {len(filas)}")
    print(f"Datos importados en la base '{base}' desde {archivo}")


def mensaje_bd(e):
    orig = e.orig or e
    msg = str(orig).strip()  # psycopg ya incluye el DETAIL del servidor
    if getattr(orig, "sqlstate", None) == "55P03":
        msg += ("\nOtra conexión (¿la app abierta?) está usando la base. "
                "Ciérrela y vuelva a intentarlo.")
    return msg


def main(argv):
    if len(argv) != 3 or argv[1] not in ("exportar", "importar"):
        print("Uso: python database\\copia_datos.py exportar|importar <archivo.json>")
        return 1
    accion, archivo = argv[1], Path(argv[2])
    try:
        if engine.dialect.name != "postgresql":
            raise ErrorCopia("la copia de datos solo funciona con PostgreSQL")
        if accion == "exportar":
            exportar(archivo)
        else:
            importar(archivo)
    except BaseConDatos as e:
        print(e)
        return 2
    except ErrorCopia as e:
        print(f"Error: {e}")
        return 1
    except DBAPIError as e:
        print(f"Error de la base de datos, no se cambió nada: {mensaje_bd(e)}")
        return 1
    except Exception as e:
        print(f"Error inesperado ({type(e).__name__}), no se cambió nada: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
