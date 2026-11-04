# Pendientes del proyecto — Tercer corte

Última actualización: 28/10/2026

## Desde el 04/11/2026 todo se hace desde la interfaz web (no hace falta tocar código)

## Estado general

| # | Etapa | Estado |
|---|---|---|
| 1 | Alcance | ✅ Hecho (`docs/01_alcance.md`) |
| 2 | Productos y condiciones de captura | 🟡 Categorías y precios definidos; faltan los objetos físicos y el soporte |
| 3 | Arquitectura | ✅ Hecho (`docs/02_arquitectura.md`) |
| 4 | Carpetas y entorno | ✅ Hecho (Python 3.11, PyTorch con CUDA, YOLO, PostgreSQL 18) |
| 5 | Dataset | 🟡 Herramientas listas (subida, extracción, etiquetado, exportación); faltan las fotos y videos |
| 6 | Entrenamiento del modelo | ⬜ Pendiente (hoy se usa el modelo provisional COCO) |
| 7 | Detección de retiros | ✅ Implementado y probado; falta calibrar con videos reales |
| 8 | Base de datos | ✅ Hecho (`inventario_cafeteria` en PostgreSQL 18) |
| 9 | API e interfaz web | 🟡 Casi completo; faltan las gráficas de estadísticas |
| 10 | Integración y pruebas | 🟡 Probado con videos de prueba; falta la prueba con la vitrina real |
| 11 | Evaluación con métricas | ⬜ Pendiente |
| 12 | Documentación y evidencias | ⬜ Pendiente |

---

## 1. Conseguir y preparar los objetos (etapa 2)

- [ ] Reunir **3 a 5 objetos distintos por categoría** y **1 reservado** por
      categoría para la prueba de generalización. Anotarlos en la tabla de
      `docs/03_dataset.md`:
      - `botella` (agua, jugo), `lata` (gaseosa), `vaso` (café, taza),
        `paquete` (papas, galletas), `fruta` (manzana, banano, naranja).
- [ ] Revisar si alguna categoría conviene cambiarla por otra que se tenga en
      casa (que se vea distinta desde arriba).
- [ ] Conseguir o improvisar un **soporte** para dejar el iPhone fijo mirando
      hacia abajo (60–100 cm de altura).
- [ ] Elegir la superficie de la "vitrina" (cartulina o mantel liso, un solo color).

## 2. Configurar el iPhone

- [ ] Ajustes → Cámara → Formatos → **«Más compatible»** (JPG/H.264, no HEIC).
- [ ] Zoom **1×**, Live Photo desactivado, video en 1080p a 30 fps.
- [ ] Instalar **Iriun Webcam** en el iPhone y en el PC y comprobar que aparece
      como cámara en *Vitrina en vivo* (cámara del computador o «Buscar cámaras»).

## 3. Capturar y etiquetar el dataset (etapa 5)

Todo se hace desde la interfaz web (*Sesiones de captura* → *Etiquetar* →
*Exportar dataset*). Guía completa en `docs/03_dataset.md`.

- [ ] **S01** — un objeto a la vez, distintas posiciones y rotaciones (4–5 videos de 20–40 s).
- [ ] **S02** — mezclas de 2 a 8 objetos, separados y juntos.
- [ ] Revisar cuántos fotogramas extrae cada video; si salen muy pocos o
      demasiados, usar «Re-extraer» o ajustar umbrales.
- [ ] Etiquetar ~100 imágenes de S01–S02 y exportar → **primer entrenamiento de prueba**.
- [ ] **S03** — iluminación de noche (lámpara).
- [ ] **S04** — otra superficie.
- [ ] **S05** — escenas con una mano tomando objetos.
- [ ] **S06** — vitrina vacía y objetos distractores (llaves, celular): marcar con **V** (sin objetos).
- [ ] **S07** — solo los objetos reservados (marcar la sesión como *prueba de generalización*).
- [ ] Meta: ~400 imágenes y **≥ 150 cajas por categoría** (la página *Exportar* muestra el avance).

## 4. Entrenar el modelo propio (etapa 6)

- [x] `vision/entrenamiento/entrenar.py` y página **Entrenar modelo** en la interfaz
      (04/11/2026): YOLO11n con *transfer learning*, avance por época, métricas
      por categoría, gráficas y copia automática a `vision/modelos/best.pt`.
- [ ] Primer entrenamiento corto con S01–S02 para validar todo el flujo.
- [ ] Entrenamiento completo con todas las sesiones.
- [ ] Copiar el mejor modelo a `vision/modelos/best.pt`. Al existir, la API lo
      usa automáticamente en lugar del provisional (o con `POST /api/vision/recargar`).
- [ ] Guardar resultados para el informe: curvas de pérdida, **mAP50 y mAP50-95**,
      precisión, recall y **matriz de confusión** por categoría.
- [ ] Evaluar en `test`, incluida la **prueba de generalización** con los objetos reservados (S07).
- [ ] Comprobar que, con el modelo propio, *Vitrina en vivo* cuenta las 5
      categorías (hoy *lata* y *paquete* no son detectables).

## 5. Calibrar el monitoreo con la vitrina real (etapa 7)

- [ ] Grabar videos de prueba en `datos/videos_prueba/` con retiros y
      reposiciones **conocidos** (anotar en un archivo cuántos hubo y cuándo).
- [ ] Probarlos con *Vitrina en vivo* → *Cámara fija del servidor* → *Archivo de video*.
- [ ] Ajustar en la interfaz el **tiempo de confirmación** (sugerido 4–5 s en
      una cafetería), la **quietud** y el **análisis por segundo**.
- [ ] Revisar la regla de **mayoría (60 %)**: equilibrio entre retiros omitidos y
      registros falsos.
- [ ] Probar casos difíciles: dos productos retirados a la vez, cliente que
      toma y devuelve, mano que pasa sin tomar nada, producto tapado, reposición.

## 6. Completar la interfaz (etapa 9)

- [x] Sección **Ventas y estadísticas** con gráficas: ventas por producto,
      ingresos por día y por hora, producto más vendido.
- [x] Botón para **reiniciar los datos de prueba** (en Catálogo) (borrar movimientos y volver
      a un conteo inicial) antes de la sustentación.
- [ ] Opcional: exportar el historial a CSV/Excel.

## 7. Evaluación con métricas (etapa 11)

- [x] Página **Evaluación de retiros** (04/11/2026): se suben los videos de prueba,
      se marcan los retiros reales en el reproductor y se calculan las métricas.
- [ ] Métricas del sistema:
      - Retiros detectados correctamente.
      - **Retiros omitidos** (falsos negativos).
      - **Registros duplicados** y eventos falsos (falsos positivos).
      - Precisión y recall de eventos.
      - **Tiempo de respuesta**: desde que se retira el producto hasta que aparece la venta.
- [ ] Métricas del modelo: mAP, precisión y recall por categoría (de la etapa 6).
- [ ] Rendimiento: inferencia en ms y fps (GPU frente a CPU).
- [ ] Guardar tablas y gráficas en `evaluacion/resultados/`.

## 8. Documentación y entrega (etapa 12)

- [ ] Informe final: problema, objetivos, arquitectura, dataset, entrenamiento,
      lógica de retiros, base de datos, interfaz, resultados, conclusiones.
- [ ] Capturas y videos de evidencia en `docs/evidencias/` (interfaz, video en
      vivo con detecciones, historial con fotos de evidencia).
- [ ] Video corto de demostración: retiro → venta registrada en la web.
- [ ] Completar el `README.md` con la instalación desde cero.
- [ ] Guía para la sustentación: cómo funciona cada parte y por qué se tomó cada decisión.
- [ ] Iniciar un repositorio git y subir el código (sin `.env` ni videos pesados).

---

## Hecho hasta ahora (resumen)

- Entorno: Python 3.11 + PyTorch 2.14 con CUDA en la RTX 3050, Ultralytics, OpenCV 5, PostgreSQL 18.
- Base de datos `inventario_cafeteria`: catálogo con precios, inventario, movimientos, dataset.
- Interfaz web «Inventario con Visión Computacional»: Panel, Vitrina en vivo,
  Historial, Catálogo, Sesiones de captura, Etiquetar y Exportar dataset.
- Extracción automática de fotogramas (descarta los movidos y repetidos).
- Editor de etiquetas propio con atajos de teclado y exportación YOLO dividida por video.
- Monitoreo continuo con dos fuentes (cámara del computador y cámara del servidor):
  moda en ventana, regla de mayoría, detector de movimiento por diferencia de
  fotogramas y confirmación con la escena quieta. Cada retiro se registra como venta.
- Conteo puntual por foto, con la opción de registrarlo como ajuste (conteo inicial).
- Pruebas automáticas con escenarios controlados y con YOLO real.
