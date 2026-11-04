#!/usr/bin/env bash
# Despliega (o actualiza) la tercera entrega en el servidor y la publica en
# https://entrega3.yeison.stream con el túnel de Cloudflare existente.
#
# Primera vez, en el servidor:
#     git clone https://github.com/Yeison10-bot/Computacional.git ~/Computacional
#     bash ~/Computacional/entrega3/despliegue/desplegar.sh
# Para actualizar después de un push: el mismo segundo comando.
#
# Es seguro ejecutarlo varias veces: cada paso revisa si ya estaba hecho.
set -euo pipefail

DOMINIO="${DOMINIO:-entrega3.yeison.stream}"
TUNEL="${TUNEL:-yeison-casa}"                       # nombre del Named Tunnel
CONTENEDOR_TUNEL="${CONTENEDOR_TUNEL:-tunel-yeison}" # contenedor de cloudflared
CONFIG_TUNEL="${CONFIG_TUNEL:-$HOME/tunel-yeison/config.yml}"
SERVICIO="http://inventario-app:8000"

cd "$(dirname "$0")/.."   # carpeta entrega3/
echo "==> Carpeta: $(pwd)"

echo "==> 1. Código más reciente de GitHub"
git -C .. pull --ff-only

echo "==> 2. Archivo .env"
if [ ! -f .env ]; then
  aleatorio() { head -c 64 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c "$1"; }
  CLAVE="$(aleatorio 20)"
  cat > .env <<EOF
DB_NAME=inventario_cafeteria
DB_USER=postgres
DB_PASSWORD=$(aleatorio 32)
ADMIN_USUARIO=admin
ADMIN_CLAVE=$CLAVE
SECRET_KEY=$(aleatorio 48)
EOF
  chmod 600 .env
  echo "    Se creó .env. Usuario: admin   Contraseña: $CLAVE"
  echo "    (guárdala; para cambiarla edita ADMIN_CLAVE en $(pwd)/.env y vuelve a ejecutar este script)"
else
  echo "    .env ya existe (no se modifica)"
fi

echo "==> 3. Construir e iniciar los contenedores (la primera vez tarda varios minutos)"
docker compose up -d --build

echo -n "==> 4. Esperando a que la app responda"
for _ in $(seq 1 90); do
  if curl -fsS http://127.0.0.1:8090/api/salud >/dev/null 2>&1; then echo " ✓"; break; fi
  echo -n "."; sleep 2
done
curl -fsS http://127.0.0.1:8090/api/salud >/dev/null || { echo; echo "La app no responde: revisa 'docker compose logs app'"; exit 1; }

echo "==> 5. Conectar el túnel ($CONTENEDOR_TUNEL) a la red de la app"
if docker inspect -f '{{json .NetworkSettings.Networks}}' "$CONTENEDOR_TUNEL" | grep -q '"inventario-red"'; then
  echo "    ya estaba conectado"
else
  docker network connect inventario-red "$CONTENEDOR_TUNEL"
  echo "    conectado"
fi

echo "==> 6. Ruta $DOMINIO en $CONFIG_TUNEL"
if grep -q "hostname: *$DOMINIO" "$CONFIG_TUNEL"; then
  echo "    ya existía"
else
  cp "$CONFIG_TUNEL" "$CONFIG_TUNEL.respaldo-$(date +%Y%m%d%H%M%S)"
  # Se inserta antes de la regla final "service: http_status:404" con la misma sangría.
  awk -v dominio="$DOMINIO" -v servicio="$SERVICIO" '
    /service: *http_status:404/ && !hecho {
      match($0, /^ */); s = substr($0, 1, RLENGTH)
      print s "- hostname: " dominio
      print s "  service: " servicio
      hecho = 1
    }
    { print }
    END { if (!hecho) exit 3 }' "$CONFIG_TUNEL" > "$CONFIG_TUNEL.nuevo" \
    || { echo "    No encontré la regla 'http_status:404' en $CONFIG_TUNEL: agrega a mano"; \
         echo "      - hostname: $DOMINIO"; echo "        service: $SERVICIO"; rm -f "$CONFIG_TUNEL.nuevo"; exit 1; }
  mv "$CONFIG_TUNEL.nuevo" "$CONFIG_TUNEL"
  docker restart "$CONTENEDOR_TUNEL" >/dev/null
  echo "    agregada y túnel reiniciado (respaldo guardado junto al archivo)"
fi

echo "==> 7. Registro DNS de $DOMINIO"
CERT=""
for c in "$(dirname "$CONFIG_TUNEL")/cert.pem" "$HOME/.cloudflared/cert.pem"; do [ -f "$c" ] && CERT="$c" && break; done
if [ -n "$CERT" ]; then
  docker run --rm -v "$(dirname "$CERT")":/certificado:ro cloudflare/cloudflared:latest \
    tunnel --origincert /certificado/cert.pem route dns "$TUNEL" "$DOMINIO" \
    || echo "    (si dice que el registro ya existe, no hay problema)"
else
  ID="$(grep -E '^ *tunnel:' "$CONFIG_TUNEL" | awk '{print $2}')"
  echo "    No encontré cert.pem para crearlo automáticamente. Créalo en el panel de Cloudflare:"
  echo "    DNS → Agregar registro → Tipo CNAME · Nombre entrega3 · Destino ${ID:-<id-del-túnel>}.cfargotunnel.com · Proxy activado"
fi

echo
echo "Listo: https://$DOMINIO"
echo "Logs: docker compose logs -f app"
