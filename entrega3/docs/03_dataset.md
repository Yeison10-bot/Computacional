# Etapa 5 — Preparación del dataset

## 1. Objetivo

Construir un dataset propio de imágenes cenitales, etiquetado en formato YOLO,
con las cinco categorías (`botella`, `lata`, `vaso`, `paquete`, `fruta`), que
permita entrenar el detector y **medir si generaliza** a objetos que no vio
durante el entrenamiento.

## 2. Inventario de objetos

Para cada categoría se reúnen varios objetos distintos. Uno por categoría se
**reserva** y nunca se usa en entrenamiento: sirve para la prueba de
generalización.

| Categoría | Objetos para entrenamiento (≥ 3) | Objeto reservado (prueba) |
|---|---|---|
| `botella` | | |
| `lata` | | |
| `vaso` | | |
| `paquete` | | |
| `fruta` | | |

## 3. Configuración del iPhone antes de capturar

| Ajuste | Dónde | Valor | Por qué |
|---|---|---|---|
| Formato | Ajustes → Cámara → Formatos | **Más compatible** | Guarda JPG y H.264 en lugar de HEIC/HEVC, que dan problemas en Windows |
| Live Photo | En la app Cámara | Desactivado | Evita archivos dobles |
| Zoom | En la app Cámara | **1×** | Mismo campo de visión que en la demostración |
| Video | Ajustes → Cámara → Grabar video | 1080p a 30 fps | Suficiente; 4K solo ocupa más espacio |
| Enfoque / exposición | Mantener pulsado sobre la pantalla | Bloqueo AE/AF | Evita que la imagen cambie de brillo entre tomas |

## 4. Montaje de la escena

- Cámara **cenital** (mirando hacia abajo), a una altura fija de 60–100 cm.
  Mientras no haya soporte: el celular apoyado en el borde de una repisa o entre
  libros, con la lente hacia abajo y sin moverse durante cada sesión.
- Superficie de **color uniforme y mate** (cartulina, mantel liso, mesa).
- Toda la "vitrina" debe quedar dentro del encuadre, con un pequeño margen.

## 5. Plan de captura

El dataset se captura en **sesiones**. Una sesión es un conjunto de tomas con la
misma luz y la misma superficie. Variar las condiciones entre sesiones es lo que
hace que el modelo sea robusto.

| Sesión | Iluminación | Superficie | Contenido |
|---|---|---|---|
| S01 | Luz de día (ventana) | Superficie A | Objetos individuales en distintas posiciones y rotaciones |
| S02 | Luz de día | Superficie A | Mezclas de 2 a 8 objetos, separados y juntos (tocándose) |
| S03 | Bombillo / lámpara (noche) | Superficie A | Mezclas, algunos objetos acostados |
| S04 | Luz de día | Superficie B (otro color) | Mezclas y objetos en los bordes del encuadre |
| S05 | Mixta (lámpara + ventana) | Superficie A | Escenas con **una mano** entrando a tomar un objeto |
| S06 | Cualquiera | Cualquiera | Imágenes **vacías** y con objetos **distractores** (llaves, celular, control) |
| S07 | Luz de día | Superficie A | **Solo objetos reservados** → conjunto de prueba de generalización |

### Variaciones dentro de cada sesión

- Posición: centro, esquinas, bordes.
- Orientación: girar cada objeto entre tomas; botellas de pie y acostadas.
- Cantidad: de 1 a 8 objetos por imagen, mezclando categorías.
- Cercanía: separados, tocándose y alineados como en una vitrina real.

### Cantidad objetivo

| Concepto | Mínimo | Recomendado |
|---|---|---|
| Imágenes en total | 250 | 400 |
| Instancias (cajas) por categoría | 150 | 300 |
| Imágenes vacías o con distractores | 5 % | 10 % |

Como cada imagen contiene varios objetos, 400 imágenes con un promedio de 4
objetos dan unas 1.600 cajas, es decir, más de 300 por categoría.

## 6. Método de captura: video → fotogramas

Tomar 400 fotos una por una es lento. Es más práctico grabar **videos cortos**
(20–40 s) en los que se reacomodan los objetos con calma y subirlos a una sesión
desde la interfaz web. Al subirlos se ejecuta automáticamente
`vision/entrenamiento/extraer_fotogramas.py`, que evalúa un fotograma cada 0,5 s
y descarta:

| Descarte | Técnica | Por qué |
|---|---|---|
| En movimiento | Diferencia absoluta media con el fotograma inmediatamente anterior (> 2,0) | Una mano o un objeto moviéndose produce imágenes movidas o con productos tapados |
| Desenfocado | Varianza del Laplaciano < 5 | Casi no hay bordes: la cámara no enfocó |
| Repetido | Diferencia absoluta media con el último fotograma guardado (< 4,0) | La escena no cambió: sería un duplicado |

**Decisión de diseño.** La primera versión descartaba los fotogramas borrosos con
un umbral fijo sobre la varianza del Laplaciano. Las pruebas mostraron que esa
medida depende de cuántos objetos hay en la escena: una escena nítida con fondo
liso y un solo objeto quedaba por debajo del umbral. Por eso el criterio
principal pasó a ser la **quietud de la escena** (diferencia entre fotogramas
consecutivos), que no depende del contenido; el Laplaciano queda solo como piso
para imágenes desenfocadas.

Recomendación al grabar: reacomodar → retirar la mano → esperar 1–2 s →
reacomodar de nuevo. Así cada escena queda quieta un momento y se captura nítida.

## 7. Organización de los archivos

La interfaz organiza los archivos sola; el código de sesión va en cada nombre:

```
datos/
├── crudos/videos/S01/<id>_IMG_0001.MOV      videos originales
├── crudos/fotos/S01/<id>_IMG_0101.JPG       fotos originales
├── imagenes/S01/S01_a3_f000120.jpg          fotogramas extraídos y fotos procesadas
└── dataset/                                 exportación en formato YOLO
```

Las sesiones, archivos, imágenes y etiquetas se registran en PostgreSQL (tablas
`sesion_captura`, `archivo_media`, `imagen` y `anotacion`).

## 8. División train / val / test

Los fotogramas de un mismo video son muy parecidos entre sí. Si se reparten al
azar, imágenes casi idénticas quedan en entrenamiento y en validación, y el mAP
sale artificialmente alto (**fuga de datos**). Por eso la división se hace por
sesión o por video, no por imagen:

| Conjunto | Contenido | Uso |
|---|---|---|
| `train` (≈ 70 %) | La mayoría de los videos de S01–S06 | Entrenar |
| `val` (≈ 20 %) | Videos completos de S01–S06 apartados | Ajustar y elegir el mejor modelo |
| `test` (≈ 10 % + S07) | Videos apartados + objetos reservados | Evaluación final y generalización |

## 9. Etiquetado

Herramienta: **editor propio de la interfaz web** (sección *Etiquetar*). Los
datos no salen del computador y las etiquetas quedan en la base de datos. Cada
objeto se marca con una caja ajustada a su contorno y su categoría. Atajos:
`1`–`5` categoría, `Enter` guardar y siguiente, `C` copiar las cajas de la
imagen anterior (útil en fotogramas seguidos), `V` imagen sin objetos (ejemplo
negativo), `X` descartar, `Supr` borrar caja.

La sección *Exportar dataset* genera el formato YOLO:

```
# labels/train/S01_IMG_0001_f000120.txt   (una línea por objeto)
# clase  x_centro  y_centro  ancho  alto   (normalizados entre 0 y 1)
2 0.4812 0.3550 0.1203 0.1187
```

Índices de clase: `0 botella`, `1 lata`, `2 vaso`, `3 paquete`, `4 fruta`.

### Criterios de etiquetado

- La caja abarca el objeto completo, sin margen extra.
- Objeto cortado por el borde de la imagen: se etiqueta si se ve más del 50 %.
- Objeto tapado por la mano: se etiqueta solo la parte visible si es reconocible.
- Distractores (llaves, celular): **no** se etiquetan.
