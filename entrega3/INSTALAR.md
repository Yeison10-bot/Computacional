# Instalar la aplicación en otro equipo

Inventario de una vitrina con YOLO: catálogo, sesiones de captura con sus etiquetas, entrenamiento del modelo y monitoreo en vivo.

## Requisitos

- Windows 10 u 11, con conexión a internet durante la instalación.
- Espacio libre: unos 10 GB (PyTorch, dependencias de Python, Node y PostgreSQL), más el tamaño del paquete descomprimido.
- Una GPU NVIDIA ayuda al entrenar, pero no es necesaria.

Instale primero estos tres programas (una sola vez):

1. **Python 3.11**: <https://www.python.org/downloads/release/python-3119/>. En el instalador marque **«Add python.exe to PATH»**.
2. **Node.js 22 LTS**: <https://nodejs.org/en/download>.
3. **PostgreSQL 18**: <https://www.postgresql.org/download/windows/>. Anote la **contraseña del usuario `postgres`** que elija durante la instalación; deje el puerto **5432**.

## Pasos

1. Descomprima el .zip (por ejemplo en `C:\entrega3`).
2. Abra PowerShell en esa carpeta (en el Explorador: clic derecho dentro de la carpeta con Mayús pulsada, «Abrir ventana de PowerShell aquí»).
3. Instale (una vez; pedirá la contraseña de PostgreSQL y tarda varios minutos):

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\instalar.ps1
   ```

4. Abra la aplicación:

   ```powershell
   .\iniciar.ps1
   ```

   Se abren dos ventanas (backend y frontend; no las cierre) y el navegador en <http://localhost:5180>.

## Qué va a ver

- **Catálogo**: los 5 productos con sus precios y colores.
- **Sesiones de captura**: S01 a S14, con sus fotos y las etiquetas (cajas) dibujadas.
- **Entrenar modelo**: el último entrenamiento realizado, con sus métricas y gráficas.
- **Vitrina en vivo**: el monitoreo con el modelo entrenado. Sin cámara se puede probar subiendo una foto para un conteo puntual.

## Cómo replicar el entrenamiento

1. Entre a **Exportar dataset** y genere el dataset a partir de las etiquetas.
2. Entre a **Entrenar modelo** y lance un entrenamiento nuevo.

En un equipo sin GPU NVIDIA el entrenamiento funciona, pero es mucho más lento (puede tardar horas); reduzca las épocas si solo quiere comprobar que funciona.

## Archivos originales

En las sesiones, los enlaces de «Archivos originales» no abren: las fotos y videos originales no van en el paquete por su tamaño. Las imágenes y las etiquetas sí están.

## Problemas comunes

- **«La ejecución de scripts está deshabilitada»**: use el comando con `-ExecutionPolicy Bypass` tal como aparece arriba.
- **Contraseña de PostgreSQL incorrecta** (error al crear la base de datos): abra el archivo `.env` con el Bloc de notas, corrija `DB_PASSWORD` y vuelva a ejecutar `instalar.ps1`.
- **No se encuentra Python, Node o npm**: instálelos como se indica arriba y abra una PowerShell nueva (la anterior no ve los cambios del PATH).
- **Puerto ocupado** (8000 o 5180): cierre el programa que lo usa o las ventanas de una ejecución anterior de `iniciar.ps1`.
- **PostgreSQL no responde**: verifique que el servicio `postgresql-x64-18` esté iniciado (Servicios de Windows).
- `instalar.ps1` se puede volver a ejecutar: no pisa el `.env` ni los datos ya cargados.
