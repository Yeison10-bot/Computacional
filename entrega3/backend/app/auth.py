"""Inicio de sesión: un usuario administrador y una cookie de sesión firmada.

Configuración (archivo .env):
    ADMIN_USUARIO=admin
    ADMIN_CLAVE=una-clave-larga      ← si está vacía, el login queda DESACTIVADO
                                        (cómodo en el computador de desarrollo)
    SECRET_KEY=...                   ← firma de las cookies; si falta se genera
                                        una y se guarda en datos/.secreto
    COOKIE_SEGURA=true               ← en el servidor con HTTPS (Cloudflare)

La cookie lleva "usuario|vencimiento" firmado con HMAC-SHA256: el servidor no
guarda sesiones, solo verifica la firma. Es HttpOnly (JavaScript no la puede
leer) y SameSite=Lax (otro sitio no puede usarla para enviar formularios).
"""

import base64
import hashlib
import hmac
import os
import secrets
import time

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import DATOS

COOKIE = "sesion_inventario"
DURACION_S = 7 * 24 * 3600          # una semana
INTENTOS_MAXIMOS = 5                # fallos seguidos antes de bloquear
BLOQUEO_S = 60
# Rutas que no requieren sesión (además del frontend, que es público: la página de login vive ahí).
PUBLICAS = ("/api/salud", "/api/auth/")

router = APIRouter(prefix="/api/auth", tags=["Sesión"])
_fallos: dict[str, tuple[int, float]] = {}   # ip → (fallos seguidos, último fallo)


def usuario_admin():
    return os.getenv("ADMIN_USUARIO", "admin")


def login_activo():
    return bool(os.getenv("ADMIN_CLAVE"))


def _secreto() -> bytes:
    if os.getenv("SECRET_KEY"):
        return os.environ["SECRET_KEY"].encode()
    archivo = DATOS / ".secreto"
    if not archivo.exists():
        DATOS.mkdir(parents=True, exist_ok=True)
        archivo.write_text(secrets.token_urlsafe(48), encoding="utf-8")
    return archivo.read_text(encoding="utf-8").strip().encode()


def _firmar(texto: str) -> str:
    return hmac.new(_secreto(), texto.encode(), hashlib.sha256).hexdigest()


def crear_token(usuario: str) -> str:
    cuerpo = f"{usuario}|{int(time.time()) + DURACION_S}"
    return base64.urlsafe_b64encode(cuerpo.encode()).decode() + "." + _firmar(cuerpo)


def leer_token(token: str | None) -> str | None:
    """Devuelve el usuario si el token es auténtico y no venció."""
    try:
        cuerpo_b64, firma = (token or "").split(".", 1)
        cuerpo = base64.urlsafe_b64decode(cuerpo_b64.encode()).decode()
        usuario, vence = cuerpo.rsplit("|", 1)
    except (ValueError, UnicodeDecodeError):
        return None
    if not hmac.compare_digest(firma, _firmar(cuerpo)) or int(vence) < time.time():
        return None
    return usuario


def ip_cliente(request: Request) -> str:
    # Detrás del túnel de Cloudflare la IP real viene en esta cabecera.
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")


def cookie_segura(request: Request) -> bool:
    if os.getenv("COOKIE_SEGURA"):
        return os.getenv("COOKIE_SEGURA", "").lower() == "true"
    return request.headers.get("x-forwarded-proto", request.url.scheme) == "https"


async def exigir_sesion(request: Request, call_next):
    """Middleware: la API y los archivos (fotos, video) solo con sesión iniciada."""
    ruta = request.url.path
    protegida = (ruta.startswith("/api/") or ruta.startswith("/archivos/")) and not ruta.startswith(PUBLICAS)
    if protegida and login_activo() and leer_token(request.cookies.get(COOKIE)) is None:
        return JSONResponse({"detail": "Inicia sesión para continuar"}, status_code=401)
    return await call_next(request)


class Credenciales(BaseModel):
    usuario: str
    clave: str


@router.post("/login")
def login(datos: Credenciales, request: Request, response: Response):
    if not login_activo():
        return {"usuario": usuario_admin(), "login_activo": False}
    ip = ip_cliente(request)
    fallos, ultimo = _fallos.get(ip, (0, 0.0))
    if fallos >= INTENTOS_MAXIMOS and time.time() - ultimo < BLOQUEO_S:
        raise HTTPException(429, f"Demasiados intentos fallidos. Espera {BLOQUEO_S} segundos.")
    # compare_digest compara en tiempo constante (no revela cuántos caracteres acertaron).
    ok_usuario = hmac.compare_digest(datos.usuario.strip().encode(), usuario_admin().encode())
    ok_clave = hmac.compare_digest(datos.clave.encode(), os.environ["ADMIN_CLAVE"].encode())
    if not (ok_usuario and ok_clave):
        _fallos[ip] = (fallos + 1 if time.time() - ultimo < BLOQUEO_S * 10 else 1, time.time())
        time.sleep(0.5)  # frena los intentos automáticos
        raise HTTPException(401, "Usuario o contraseña incorrectos")
    _fallos.pop(ip, None)
    response.set_cookie(COOKIE, crear_token(usuario_admin()), max_age=DURACION_S, httponly=True,
                        samesite="lax", secure=cookie_segura(request), path="/")
    return {"usuario": usuario_admin(), "login_activo": True}


@router.post("/salir")
def salir(response: Response):
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@router.get("/yo")
def yo(request: Request):
    """Quién tiene la sesión iniciada (o 401). La página lo consulta al abrir."""
    from .routers.monitor import camara_servidor_permitida  # evita una importación circular
    if not login_activo():
        return {"usuario": None, "login_activo": False, "camara_servidor": camara_servidor_permitida()}
    usuario = leer_token(request.cookies.get(COOKIE))
    if usuario is None:
        raise HTTPException(401, "Inicia sesión para continuar")
    return {"usuario": usuario, "login_activo": True, "camara_servidor": camara_servidor_permitida()}
