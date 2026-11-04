# Control automático de inventario de cafetería con visión computacional

Proyecto del tercer corte de **Inteligencia Computacional II**.

Una cámara fija observa una vitrina con cinco tipos de productos. Un modelo YOLO
reconoce los productos en cada fotograma; cuando el conteo de un producto baja de
forma estable se registra un **retiro (venta)**, se descuenta del inventario en
PostgreSQL y se refleja en una interfaz web. Los aumentos estables se registran
como **reposición**.

## Estructura

| Carpeta | Contenido |
|---|---|
| `docs/` | Alcance, arquitectura, informe y `evidencias/` (capturas, videos de demostración) |
| `datos/crudos/` | Fotos y videos originales sin procesar |
| `datos/dataset/` | Dataset en formato YOLO (`images/` y `labels/` con `train`, `val`, `test`) |
| `datos/videos_prueba/` | Videos de prueba con los retiros reales anotados |
| `vision/` | Captura con OpenCV, detección con YOLO y lógica de eventos |
| `vision/entrenamiento/` | Scripts de entrenamiento y validación del modelo |
| `vision/modelos/` | Pesos entrenados |
| `backend/` | API con FastAPI |
| `database/` | Esquema SQL y datos iniciales |
| `frontend/` | Interfaz web con React + Vite |
| `evaluacion/` | Scripts de métricas y resultados |

## Requisitos verificados en el equipo

- Python 3.11, Node.js 22, Git
- PostgreSQL 18 (puerto 5432)
- GPU NVIDIA RTX 3050 Laptop (4 GB)

## Instalación (una sola vez)

Todos los comandos se ejecutan en PowerShell, desde `D:\Tercer Corte\Vision_Computacional`.

```powershell
# 1. Entorno de Python (PyTorch con CUDA va primero y por separado)
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe vision\verificar_entorno.py

# 2. Configuración: copiar la plantilla y escribir la contraseña de PostgreSQL
Copy-Item .env.example .env
notepad .env

# 3. Base de datos: crea la BD, las tablas y los 5 productos
.\.venv\Scripts\python.exe database\crear_bd.py

# 4. Dependencias del frontend
cd frontend; npm install; cd ..
```

## Ejecución

```powershell
powershell -ExecutionPolicy Bypass -File .\iniciar.ps1
```

Abre dos ventanas (backend y frontend) y el navegador en http://localhost:5180.
La documentación interactiva de la API queda en http://localhost:8000/docs.

## Interfaz web

| Sección | Para qué sirve |
|---|---|
| Panel | Existencias en vitrina, ventas e ingresos del día, stock bajo y últimos movimientos (se refresca cada 10 s) |
| Vitrina en vivo | Monitoreo continuo: la cámara fija (webcam o iPhone con Iriun) se analiza sin parar y cada retiro o reposición confirmado se registra solo. Muestra el video con las detecciones, el estado de cada producto y los eventos. También permite un conteo puntual subiendo una foto |
| Ventas y estadísticas | Unidades e ingresos por producto, por día y por hora; producto más vendido. Hoy, 7, 30 o 90 días |
| Historial | Todos los movimientos con fecha, origen, confianza del modelo y la foto que los produjo |
| Catálogo | Nombre, precio y stock mínimo de los 5 productos. Las existencias no se editan a mano: las actualiza la cámara. Incluye el botón **Reiniciar datos de prueba** |
| Sesiones de captura | Crear sesiones (S01, S02…) y subir videos y fotos. De los videos se extraen automáticamente los fotogramas quietos y distintos |
| Etiquetar | Dibujar las cajas de cada producto, con atajos de teclado |
| Exportar dataset | Ver el balance de cajas por categoría y generar `datos/dataset` en formato YOLO, dividido por video |
| Entrenar modelo | Entrenar YOLO11n en la GPU con el dataset exportado: avance por época, curvas, métricas por categoría en validación y prueba, matriz de confusión. Al terminar, el modelo nuevo (`vision/modelos/best.pt`) reemplaza al provisional |
| Evaluación de retiros | Subir videos de prueba, marcar en el reproductor cada retiro real y evaluar: aciertos, omitidos, falsos positivos, duplicados, precisión, recall, F1 y latencia, por video y en total |
