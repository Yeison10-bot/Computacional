# Etapa 1 y 2 — Alcance del proyecto y condiciones de captura

## 1. Objetivo

Desarrollar un sistema que, a partir del video de una cámara fija, reconozca
cinco **categorías** de productos ubicados en una superficie que simula la vitrina
de una cafetería, detecte cuándo se retira o se repone un producto y actualice
automáticamente el inventario, el historial de movimientos y las estadísticas de
ventas en una aplicación web.

## 2. Decisión clave: categorías, no productos individuales

El modelo no se entrena para reconocer *un* objeto concreto (una botella en
particular), sino una **categoría visual** (cualquier botella). Para lograrlo:

- Cada categoría se entrena con **varios objetos distintos** de ese tipo (marcas,
  colores y tamaños diferentes).
- Las cinco categorías se eligen para que sean **visualmente diferentes entre sí**
  vistas desde arriba (forma y tamaño de la silueta).
- Un objeto que no pertenezca a ninguna categoría debe ser ignorado (umbral de
  confianza del detector).

Así, el sistema reconoce productos *parecidos* a los del entrenamiento aunque
nunca los haya visto. Esto se evaluará explícitamente con un conjunto de prueba
que contenga objetos **no usados en el entrenamiento**.

Cada categoría tiene en el catálogo un nombre comercial y un precio de cafetería
(por ejemplo, la categoría *botella* se vende como "Agua / Jugo embotellado"), de
modo que el inventario y los ingresos tienen sentido para el negocio simulado.

### Categorías definidas (provisionales; pueden ajustarse según los objetos disponibles)

| Categoría | Producto en el catálogo | Precio (COP) | Por qué funciona vista desde arriba |
|---|---|---|---|
| `botella` | Agua / jugo embotellado | $3.500 | Tapa circular pequeña + silueta alargada si está acostada |
| `lata` | Gaseosa en lata | $3.000 | Círculo metálico con anillo y abre-fácil |
| `vaso` | Café para llevar | $4.000 | Círculo grande, borde o tapa visibles |
| `paquete` | Snack (papas, galletas) | $2.500 | Rectángulo plano con empaque brillante y colorido |
| `fruta` | Fruta (manzana, banano, naranja) | $1.500 | Forma orgánica, color uniforme, sin bordes rectos |

Los precios son de referencia para una cafetería pequeña en Colombia y pueden
cambiarse desde el catálogo. Cada movimiento guarda el **precio unitario vigente
en el momento del evento**, de modo que un cambio de precio posterior no altera
los ingresos ya registrados.

## 3. Condiciones de captura

| Aspecto | Decisión |
|---|---|
| Cámara | iPhone 11. Fotos y videos grabados con la cámara nativa; en vivo como webcam con Iriun Webcam |
| Ángulo | **Cenital** (desde arriba, perpendicular a la superficie) |
| Distancia | La necesaria para que toda la "vitrina" quede en el encuadre (aprox. 60–100 cm) |
| Superficie | Fondo uniforme y mate (cartulina, mantel o mesa lisa) |
| Iluminación | Luz constante, sin sombras fuertes; se probará también con variaciones |
| Disposición | Productos separados y juntos (también tocándose), en varias posiciones |
| Resolución de trabajo | 1280×720 para capturar, 640 px para el modelo |

Ventajas de la vista cenital:
- Casi no hay oclusiones (un producto rara vez tapa a otro).
- Contar los objetos visibles equivale a contar el inventario de la vitrina.
- La mano del cliente pasa por encima solo unos instantes, lo que se filtra con la
  lógica de estabilidad temporal.

## 4. Qué entra en el alcance

- Una sola cámara fija con vista cenital.
- Cinco categorías de productos.
- Detección de objetos con YOLO entrenado con un dataset propio.
- Detección de **retiros** (cada retiro = una venta, según la indicación del
  profesor) y de **reposiciones**.
- Prevención de registros duplicados mediante una ventana de estabilidad temporal.
- Base de datos PostgreSQL: catálogo, inventario, movimientos.
- API en FastAPI e interfaz web en React con inventario, historial y estadísticas.
- Evaluación del detector (mAP, precisión, recall, matriz de confusión) y del
  sistema (retiros correctos, omitidos, duplicados, falsos positivos, latencia).

## 5. Qué queda fuera del alcance

- Varias cámaras o vitrinas.
- Verificación de pago o identificación de clientes.
- Reconocimiento de la marca o del producto exacto (solo la categoría).
- Productos apilados uno sobre otro (no visibles desde arriba).
- Funcionamiento en condiciones de iluminación extremas (oscuridad, contraluz).

## 6. Supuestos

- La superficie solo contiene productos de la vitrina; otros objetos se retiran
  del encuadre.
- Un retiro se considera válido cuando la disminución del conteo de una categoría
  se mantiene durante un tiempo mínimo (valor a calibrar en la etapa 7).
- La reposición se registra cuando el conteo aumenta de forma estable.

## Pendiente

- [ ] Revisar las cinco categorías según los objetos disponibles en casa
      (mínimo 3 objetos distintos por categoría; idealmente 4–5).
- [x] Cámara: iPhone 11 (Iriun Webcam para video en vivo).
- [ ] Soporte para dejar el celular fijo en vista cenital (necesario para la demostración).
