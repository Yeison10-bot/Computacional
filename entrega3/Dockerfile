# Imagen de producción: compila el frontend y lo sirve junto con la API (un solo contenedor).
# PyTorch va en su versión para CPU (el servidor no tiene GPU): la inferencia de YOLO11n
# funciona bien; entrenar en el servidor es posible pero lento (mejor entrenar en el
# portátil con la RTX 3050 y copiar best.pt, ver docs/DESPLIEGUE.md).

# ---------------------------------------------------------------- 1. Frontend (React + Vite)
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------- 2. API (FastAPI + YOLO)
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/app/datos/.ultralytics \
    CAMARA_SERVIDOR=false \
    COOKIE_SEGURA=true
# Bibliotecas de sistema que necesita OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY backend backend
COPY vision vision
COPY database database
COPY despliegue/arrancar.sh despliegue/arrancar.sh
COPY --from=frontend /app/frontend/dist frontend/dist
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/salud', timeout=4)"
CMD ["sh", "despliegue/arrancar.sh"]
