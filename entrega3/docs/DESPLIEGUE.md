# Despliegue en el servidor (https://entrega3.yeison.stream)

La aplicación corre en el servidor de la casa con Docker y se publica con el túnel
de Cloudflare ya existente (`yeison-casa`, contenedor `tunel-yeison`), igual que
entrega1 y entrega2.

```
iPhone / PC ──https──▶ Cloudflare ──túnel──▶ tunel-yeison ──▶ inventario-app:8000 ──▶ inventario-db
                                                              (API + página web)      (PostgreSQL 17)
```

## Qué cambia respecto al computador de desarrollo

| | Desarrollo (portátil) | Servidor |
|---|---|---|
| Página web | Vite en `http://localhost:5180` | La sirve la misma API (frontend compilado) |
| Login | Desactivado (sin `ADMIN_CLAVE`) | **Obligatorio** |
| Cámara de la vitrina | La del navegador o una conectada al PC | **Solo la del dispositivo que abre la página** (celular o PC). Funciona porque la página va por https |
| YOLO | GPU (RTX 3050) | CPU (más lento al entrenar; la detección en vivo va bien a 5 análisis/s con YOLO11n) |
| Datos | `datos/` del proyecto | Volúmenes de Docker `inventario-datos`, `inventario-modelos`, `inventario-postgres` |

## Primera vez

En el servidor (por SSH):

```bash
git clone https://github.com/Yeison10-bot/Computacional.git ~/Computacional
bash ~/Computacional/entrega3/despliegue/desplegar.sh
```

El script:
1. Actualiza el código desde GitHub.
2. Crea `entrega3/.env` con contraseñas aleatorias **y muestra la contraseña del login** (usuario `admin`). Guárdala.
3. Construye e inicia los contenedores `inventario-db` e `inventario-app` (la primera vez tarda: descarga PyTorch).
4. Conecta el contenedor del túnel a la red `inventario-red`.
5. Agrega `entrega3.yeison.stream → http://inventario-app:8000` al `config.yml` del túnel (guarda un respaldo) y reinicia el túnel.
6. Crea el registro DNS. Si no encuentra el certificado de Cloudflare, indica cómo crear el CNAME a mano en el panel.

## Actualizar después de cambios

En el portátil: copiar los cambios a `D:\Tercer Corte\Computacional\entrega3`, commit y push.
En el servidor: `bash ~/Computacional/entrega3/despliegue/desplegar.sh` (los datos se conservan).

## Tareas comunes

```bash
cd ~/Computacional/entrega3
docker compose logs -f app                  # ver qué pasa
docker compose restart app                  # reiniciar
nano .env && docker compose up -d           # cambiar la contraseña del login
```

**Usar el modelo entrenado en el portátil** (más rápido que entrenar en el servidor):

```bash
# desde el portátil (PowerShell):
scp "D:\Tercer Corte\Vision_Computacional\vision\modelos\best.pt" yeison10@100.122.55.32:/tmp/best.pt
# en el servidor:
docker cp /tmp/best.pt inventario-app:/app/vision/modelos/best.pt
# Copia también best.json (umbrales de confianza por categoría) de la misma manera;
# sin él, o si es de otro best.pt, el detector usa 0.4 para todas las categorías.
# y en la página: Entrenar modelo / Vitrina → el modelo se recarga al iniciar el monitoreo
# (o: curl -X POST con sesión a /api/vision/recargar; lo más simple: docker compose restart app)
```

**Copia de seguridad de la base de datos:**

```bash
docker exec inventario-db pg_dump -U postgres inventario_cafeteria > respaldo_$(date +%F).sql
```

## Límites a tener en cuenta

- **Cloudflare acepta archivos de hasta 100 MB por subida**: los videos de captura deben ser cortos
  (≈ 1 minuto en 1080p «Más compatible»). La página avisa si se pasa.
- **El celular debe quedar con la página abierta y la pantalla encendida** mientras vigila la vitrina
  (la página pide al sistema que no apague la pantalla). Si se bloquea, el servidor detiene el
  monitoreo a los 8 s y hay que volver a iniciarlo.
- Solo una cámara a la vez vigila la vitrina (un monitoreo activo).
