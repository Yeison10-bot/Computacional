"""Conexión a la base de datos con SQLAlchemy."""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import url_base_datos

engine = create_engine(url_base_datos(), pool_pre_ping=True)
SesionBD = sessionmaker(bind=engine, expire_on_commit=False)

if engine.dialect.name == "sqlite":
    # SQLite (solo pruebas) no aplica ON DELETE CASCADE salvo que se active.
    @event.listens_for(engine, "connect")
    def _activar_claves_foraneas(conexion, _):
        conexion.execute("PRAGMA foreign_keys=ON")


class Base(DeclarativeBase):
    pass


def obtener_bd():
    """Dependencia de FastAPI: abre una sesión por petición y la cierra al terminar."""
    bd = SesionBD()
    try:
        yield bd
    finally:
        bd.close()
