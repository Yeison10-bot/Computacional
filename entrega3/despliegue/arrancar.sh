#!/bin/sh
# Arranque del contenedor: prepara la base de datos (es seguro repetirlo: no borra
# nada ni duplica productos) y luego inicia la API, que también sirve la página web.
set -e
python database/crear_bd.py
# --proxy-headers: detrás del túnel de Cloudflare, respeta https y la IP real del cliente.
exec uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 \
    --proxy-headers --forwarded-allow-ips="*" --timeout-graceful-shutdown 5
