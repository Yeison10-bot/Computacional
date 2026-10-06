# Guía de sustentación — Proyecto Inteligencia Computacional II (Primer corte)

> Esta guía es para **defender el proyecto ante el profesor**. Todos los números salen de
> ejecutar el código real del repositorio (scikit-learn 1.8.0, pandas 2.2.3, numpy 2.2.4,
> `random_state=42`). Si cambias la semilla o la versión de las librerías, algunos decimales
> pueden variar.
>
> Cómo leerla: las secciones 1 a 11 explican **qué se hizo, cómo y por qué**. La sección 12
> reúne los **puntos débiles** que un profesor exigente puede atacar y cómo responder con
> honestidad. La sección 13 es un **banco de preguntas y respuestas** para repasar. La
> sección 14 es la **hoja de números** para memorizar.

---

## Índice

0. [El proyecto en 1 minuto](#0-el-proyecto-en-1-minuto)
1. [Estructura del código y cómo se ejecuta](#1-estructura-del-código-y-cómo-se-ejecuta)
2. [Las bases de datos](#2-las-bases-de-datos)
3. [Preprocesamiento: qué se hizo y qué no (y por qué)](#3-preprocesamiento-qué-se-hizo-y-qué-no-y-por-qué)
4. [Métricas: fórmulas e implementación (incluye R² ajustado)](#4-métricas-fórmulas-e-implementación)
5. [La matriz de confusión en cada modelo](#5-la-matriz-de-confusión-en-cada-modelo)
6. [Árboles de decisión (clasificación)](#6-árboles-de-decisión-clasificación)
7. [Random Forest (clasificación)](#7-random-forest-clasificación)
8. [Regresión: árboles y Random Forest sobre concreto](#8-regresión-árboles-y-random-forest-sobre-concreto)
9. [KNN](#9-knn)
10. [Clustering (K-means)](#10-clustering-k-means)
11. [La app de Streamlit y las demostraciones extra](#11-la-app-de-streamlit-y-las-demostraciones-extra)
12. [Puntos débiles: lo que te pueden atacar y cómo responder](#12-puntos-débiles-lo-que-te-pueden-atacar-y-cómo-responder)
13. [Banco de preguntas y respuestas](#13-banco-de-preguntas-y-respuestas)
14. [Hoja de números para memorizar](#14-hoja-de-números-para-memorizar)

---

## 0. El proyecto en 1 minuto

Se compararon técnicas de Machine Learning sobre **dos bases de datos reales**:

| Dataset | Problema | Objetivo | Modelos |
|---|---|---|---|
| `Crop_recommendation.csv` (2200 filas) | **Clasificación** (22 cultivos) | `label` | Árbol de decisión, Random Forest, KNN, K-means (sin la etiqueta) |
| `concreto.csv` (721 filas) | **Regresión** | `csMPa` (resistencia a compresión, MPa) | Árbol de regresión, Random Forest de regresión |

El **hilo conductor es el sobreajuste**: en cada técnica se muestra cómo aparece (comparando
métrica en *train* vs. *test*) y cómo se controla (poda, ensamble, elegir K).

Resultados clave:
- Clasificación: el mejor es **Random Forest (49 o 491 árboles): 99.55 % en test** (2 errores de 440).
- Regresión: el mejor es **Random Forest de 491 árboles: R² = 0.939, R² ajustado = 0.935, MAE = 3.12 MPa**.
- KNN (K=1, datos estandarizados): 97.95 %; con variables ponderadas: **98.86 %**.
- K-means con K=22: coincide en un **74.95 %** con los cultivos reales sin haber visto nunca la etiqueta;
  las métricas internas (silueta y Davies-Bouldin) prefieren **K=2**.

---

## 1. Estructura del código y cómo se ejecuta

### 1.1 Archivos

| Archivo | Qué hace | Punto de la rúbrica |
|---|---|---|
| `preprocesamiento.py` | Carga los CSV, divide train/test, estandariza (Z-score), Min-Max, **R² ajustado**, resumen de los datos | Base de todo |
| `arboles_decision.py` | 3 árboles de clasificación + matrices de confusión + pre-poda y post-poda | Punto 4 |
| `random_forest.py` | Bosques de 9, 49 y 491 árboles, tiempos, regla de desempate, evidencia de bootstrap | Punto 5 |
| `regresion.py` | Lo mismo que 4 y 5 pero con regresión (concreto): MAE, RMSE, R², R² ajustado | Punto 6 |
| `knn.py` | Selección de K por validación cruzada, *lazy learning*, ponderaciones | Punto 7 |
| `clustering.py` | K-means, 3 técnicas para elegir K, movimiento de centroides, matriz de confusión clusters vs. clases | Punto 8 |
| `graficos.py` | Funciones de gráficos interactivos (Plotly) compartidas: matriz de confusión, curva de poda, real vs. predicho, convergencia | Visualización |
| `main.py` | Orquestador: corre todos los experimentos en orden y guarda gráficos en `resultados/` | — |
| `app.py` | Interfaz **Streamlit**: cada botón entrena el modelo en vivo, con teoría en cada pestaña | Sustentación |
| `resultados/` | PNG (estáticos) + `_interactivo.html` (Plotly) de cada gráfico | Evidencia |

### 1.2 Cómo se ejecuta

```bash
python main.py          # corre TODOS los experimentos (≈12 s) y llena resultados/
streamlit run app.py    # abre la interfaz interactiva en el navegador
python knn.py           # cada módulo también corre solo (bloque if __name__ == "__main__")
```

Librerías: `pandas`, `numpy`, `scikit-learn`, `scipy` (algoritmo húngaro), `plotly` +
`kaleido` (para exportar PNG desde Plotly), `streamlit`, `matplotlib` (dibujo del árbol en la app).

> **Ojo en Windows:** el archivo `salida.txt` muestra un error viejo (`UnicodeEncodeError`
> por el símbolo ✔). Ya se corrigió (ahora imprime `[OK]`), pero si en la consola sale un
> error parecido, ejecuta con `set PYTHONIOENCODING=utf-8` antes. Los scripts ya solo
> imprimen caracteres compatibles con la consola de Windows.

### 1.3 El flujo común de todos los experimentos

Todos los modelos supervisados siguen el mismo patrón (ver `arboles_decision.py:32`
`entrenar_y_evaluar`):

1. `cargar_crop()` / `cargar_concreto()` → devuelve `X` (variables), `y` (objetivo), `df`.
2. `dividir_datos(...)` → train/test con `random_state=42` (misma partición para todos → comparación justa).
3. (Solo KNN/K-means) `estandarizar(X_train, X_test)`.
4. `inicio = time.time(); modelo.fit(X_train, y_train); tiempo = time.time() - inicio`.
5. `predict` sobre **train y test** → métricas en ambos → se compara la brecha para detectar sobreajuste.
6. Se grafica (matriz de confusión o real vs. predicho) con `graficos.py`.

Convención: **X mayúscula = matriz** (filas = registros, columnas = variables);
**y minúscula = vector** objetivo.

---

## 2. Las bases de datos

### 2.1 `Crop_recommendation.csv` — clasificación

- **2200 registros**, **7 variables predictoras** + `label`, **0 nulos**, **0 filas duplicadas**.
- **Perfectamente balanceado**: 22 cultivos × 100 muestras cada uno.
- Origen: dataset público de Kaggle (*Crop Recommendation Dataset*, India).

| Variable | Significado | Tipo pandas | Tipo estadístico | Rango | Media |
|---|---|---|---|---|---|
| `N` | Nitrógeno en el suelo (proporción) | int64 | Cuantitativa discreta | 0 – 140 | 50.6 |
| `P` | Fósforo | int64 | Cuantitativa discreta | 5 – 145 | 53.4 |
| `K` | Potasio | int64 | Cuantitativa discreta | 5 – 205 | 48.2 |
| `temperature` | Temperatura (°C) | float64 | Cuantitativa continua | 8.8 – 43.7 | 25.6 |
| `humidity` | Humedad relativa (%) | float64 | Cuantitativa continua | 14.3 – 100 | 71.5 |
| `ph` | pH del suelo | float64 | Cuantitativa continua | 3.5 – 9.9 | 6.47 |
| `rainfall` | Lluvia (mm) | float64 | Cuantitativa continua | 20.2 – 298.6 | 103.5 |
| **`label`** | Cultivo (objetivo) | object | **Cualitativa nominal** | 22 clases | — |

Las 22 clases en **orden alfabético** (así las ordena sklearn y así salen en la matriz de
confusión): apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute,
kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya,
pigeonpeas, pomegranate, rice, watermelon.

**Por qué importa el balance:** si una clase tuviera muchas más muestras, el modelo tendería
a predecirla más y el *accuracy* engañaría. Con 100 por clase, el *accuracy* es una métrica
confiable y el promedio *macro* y el *weighted* de F1 dan prácticamente lo mismo.

**Cultivos que se parecen (y por eso se confunden):** las leguminosas
(blackgram, lentil, mothbeans) tienen N, P, K y humedad casi iguales; rice y jute
también (N≈80, P≈47, K=40, humedad≈81 %; solo los separa la lluvia: 236 vs. 175 mm).
Apple y grapes son **muy distintos al resto**: P≈133 y K≈200 (el resto tiene K entre 20 y 50).

### 2.2 `concreto.csv` — regresión

- **721 registros**, **8 variables predictoras** + `csMPa`, **0 nulos**.
- Tenía una columna `Row ID` que es solo un índice → **se elimina** en `cargar_concreto()`
  (`preprocesamiento.py:42`) porque no tiene información predictiva (y podría inducir patrones falsos).
- Origen: es un subconjunto (archivo `train.csv` de Kaggle) del dataset *Concrete Compressive
  Strength* de UCI (Yeh, 1998), que tiene 1030 mezclas en total.
- Hay **16 filas completamente duplicadas** (26 si solo se miran las variables X). **No se
  eliminaron** (ver sección 12).

| Variable | Significado | Unidad | Tipo estadístico | Correlación con csMPa |
|---|---|---|---|---|
| `cement` | Cemento | kg/m³ | Continua | **+0.52** (la más fuerte) |
| `slag` | Escoria de alto horno | kg/m³ | Continua | +0.13 |
| `flyash` | Ceniza volante | kg/m³ | Continua | −0.12 |
| `water` | Agua | kg/m³ | Continua | **−0.30** (más agua → menos resistencia) |
| `superplasticizer` | Aditivo superplastificante | kg/m³ | Continua | +0.40 |
| `coarseaggregate` | Agregado grueso | kg/m³ | Continua | −0.19 |
| `fineaggregate` | Agregado fino (arena) | kg/m³ | Continua | −0.15 |
| `age` | Edad de curado | días | **Discreta** (1, 3, 7, 14, 28, 56, 90… 365) | +0.34 |
| **`csMPa`** | Resistencia a la compresión (objetivo) | MPa | Continua | — |

Estadísticas de `csMPa`: media 36.16, desviación 17.20, mínimo 2.33, máximo 82.6.

**Dato interesante para sustentar:** `age` tiene una correlación lineal de solo 0.34, pero en
el Random Forest es la **variable más importante** (0.351). La razón: la resistencia crece con
el curado de forma **no lineal** (sube rápido los primeros 28 días y luego se aplana). La
correlación de Pearson solo mide relaciones lineales; los árboles capturan las no lineales.

### 2.3 ¿Por qué un dataset es de clasificación y el otro de regresión?

Por el **tipo de la variable objetivo**: `label` es cualitativa nominal (categorías sin orden)
→ clasificación; `csMPa` es cuantitativa continua → regresión. Las variables predictoras no
deciden esto.

---

## 3. Preprocesamiento: qué se hizo y qué no (y por qué)

### 3.1 Lo que SÍ se hizo

**a) Revisión de calidad** (`resumen_distribucion`, `preprocesamiento.py:117`):
`df.isnull().sum().sum()` → 0 nulos en ambos; `value_counts()` → balance de clases;
`describe()` → estadísticas del concreto.

**b) Eliminación de `Row ID`** en el concreto (es un índice, no una característica).

**c) Separación X / y:** `X = df.drop(columns=["label"])`, `y = df["label"]`.

**d) División train/test** (`dividir_datos`, `preprocesamiento.py:57`):

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if estratificar else None)
```

| Parámetro | Qué hace | Por qué |
|---|---|---|
| `test_size=0.2` | 80 % entrenamiento / 20 % prueba (el modelo 3 usa 0.3 → 70/30) | Estándar; deja datos suficientes para ambas cosas |
| `random_state=42` | Semilla fija del barajado | **Reproducibilidad**: siempre sale la misma partición y todos los modelos se comparan sobre el mismo test |
| `stratify=y` | Mantiene la misma proporción de clases en train y test | Con 22 clases, un reparto al azar podría dejar una clase con pocas muestras en test. Resultado: **80 por clase en train y 20 por clase en test** |

En **regresión no se estratifica** porque `y` es continua (no hay clases que proporcionar).

Tamaños resultantes:

| Dataset | Partición | Train | Test |
|---|---|---|---|
| Cultivos | 80/20 | 1760 | 440 |
| Cultivos | 70/30 | 1540 | 660 |
| Concreto | 80/20 | 576 | 145 |
| Concreto | 70/30 | 504 | 217 |

**e) Estandarización Z-score** (`estandarizar`, `preprocesamiento.py:71`) — **solo para KNN y K-means**:

```
z = (x − media) / desviación_estándar      →  cada variable queda con media 0 y desviación 1
```

```python
scaler = StandardScaler()
X_train_esc = scaler.fit_transform(X_train)   # fit: calcula media y desviación SOLO con train
X_test_esc  = scaler.transform(X_test)        # transform: usa la media/desv. de TRAIN
```

**Por qué `fit` solo con train:** si se calcularan la media y la desviación con todos los datos,
información del test "se filtraría" al entrenamiento (**data leakage**) y el resultado en test
sería optimista. El test debe simular datos que el modelo nunca ha visto.

Nota: después de escalar, el resultado es un **array de NumPy** (pierde los nombres de las columnas).

**f) Min-Max** (`escalar_minmax`, `preprocesamiento.py:87`): `x' = (x − min) / (max − min)`
→ todo queda en [0, 1]. **Está implementado, pero no se usa en los experimentos**; solo se
muestra en la app (pestaña Preprocesamiento) como comparación. Se prefirió Z-score porque
Min-Max es muy sensible a valores extremos (un solo valor máximo atípico comprime a todos los demás).

### 3.2 ¿Por qué árboles/Random Forest NO se escalan y KNN/K-means SÍ?

- Un árbol pregunta "¿`rainfall` ≤ 120?". Si multiplico `rainfall` por 10, la pregunta pasa a
  ser "¿`rainfall` ≤ 1200?" y separa **exactamente los mismos datos**. El árbol es
  **invariante a transformaciones monótonas** de cada variable → escalar no cambia nada.
- KNN y K-means calculan **distancias euclidianas**:
  `d = √[(N₁−N₂)² + … + (rainfall₁−rainfall₂)²]`. `rainfall` varía en cientos y `ph` en
  unidades: sin escalar, la distancia la decide casi solo `rainfall`, aunque `ph` sea igual de
  importante. Estandarizar pone todas las variables en la misma escala.

### 3.3 Lo que NO se hizo, y por qué (pregunta muy probable)

| Paso | ¿Se hizo? | Justificación |
|---|---|---|
| Imputación de nulos | No | No hay nulos (0 en ambos datasets) |
| One-Hot Encoding | No | Todas las variables predictoras ya son numéricas. La etiqueta `label` (texto) la maneja sklearn internamente: los clasificadores aceptan `y` como texto y la codifican solos (`classes_`) |
| Balanceo (SMOTE, `class_weight`) | No | El dataset ya está balanceado 1:1 (100 por clase) |
| Tratamiento de atípicos | No | Los árboles y RF son robustos a atípicos en X (solo usan umbrales/orden). En concreto, valores altos de `age` (365 días) son reales, no errores |
| Eliminar duplicados del concreto | No | Ver sección 12 (es un punto débil honesto) |
| Selección de variables | No (se analizó) | Solo hay 7 y 8 variables; la app calcula **Mutual Information** para mostrar que todas aportan (la menor es `ph` en cultivos y `flyash` en concreto, pero ninguna es cero) |
| Transformar `y` en concreto (log) | No | No es necesario para árboles; la distribución de csMPa es aproximadamente simétrica |

### 3.4 Mutual Information (app, pestaña Preprocesamiento)

Mide cuánta incertidumbre sobre el objetivo se elimina al conocer una variable:
`MI(X;Y) = H(Y) − H(Y|X)`. A diferencia de la correlación de Pearson, **detecta relaciones
no lineales**. En sklearn se estima con vecinos más cercanos (`mutual_info_classif` /
`mutual_info_regression`), por eso se fija `random_state=42`.

- Cultivos (MI en nats; máximo posible = ln 22 ≈ 3.09): humidity 1.73, K 1.64, rainfall 1.64,
  P 1.30, temperature 1.02, N 1.00, **ph 0.69 (la menos informativa)**.
- Concreto: age 0.35, cement 0.32, water 0.30, superplasticizer 0.21, coarse 0.21, fine 0.20,
  slag 0.17, **flyash 0.09 (la menos informativa)**.

Esto coincide con la importancia de variables del Random Forest (rainfall, humidity y K
arriba; ph abajo) → **dos métodos independientes dicen lo mismo**.

---

## 4. Métricas: fórmulas e implementación

### 4.1 Clasificación

Para una clase *c*: **TP** = era *c* y se predijo *c*; **FP** = se predijo *c* pero no lo era;
**FN** = era *c* pero se predijo otra cosa.

| Métrica | Fórmula | Pregunta que responde | Código |
|---|---|---|---|
| Accuracy | aciertos / total = traza(CM) / suma(CM) | ¿Qué fracción acertó en total? | `accuracy_score(y_test, pred)` |
| Precision | TP / (TP + FP) | De lo que predijo como *c*, ¿cuánto era *c*? | `classification_report` |
| Recall (sensibilidad) | TP / (TP + FN) | De lo que era *c*, ¿cuánto encontró? | `classification_report` |
| F1 | 2·P·R / (P + R) | Media armónica de P y R | `classification_report` |
| Macro avg | promedio simple entre clases | Todas las clases pesan igual | |
| Weighted avg | promedio ponderado por *support* | Clases grandes pesan más | |

Ejemplo real (árbol modelo 1, clase `blackgram`): de 20 blackgram reales encontró 16 →
**recall 0.80**; todo lo que predijo como blackgram lo era → **precision 1.00**; F1 = 0.889.

### 4.2 Regresión

| Métrica | Fórmula | Interpretación |
|---|---|---|
| MAE | (1/n) Σ \|yᵢ − ŷᵢ\| | Error promedio en MPa. "En promedio me equivoco 3.1 MPa" |
| RMSE | √[(1/n) Σ (yᵢ − ŷᵢ)²] | Igual que MAE pero **castiga más los errores grandes** (los eleva al cuadrado). Siempre RMSE ≥ MAE |
| R² | 1 − SS_res / SS_tot = 1 − Σ(yᵢ−ŷᵢ)² / Σ(yᵢ−ȳ)² | Proporción de la variabilidad explicada. 1 = perfecto; 0 = igual que predecir siempre el promedio; **puede ser negativo** si el modelo es peor que el promedio |
| R² ajustado | 1 − (1 − R²)(n − 1)/(n − p − 1) | R² penalizado por el número de variables |

Implementación (`regresion.py:27`):

```python
mae  = mean_absolute_error(y_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_real, y_pred))   # raíz del MSE
r2   = r2_score(y_real, y_pred)
```

### 4.3 R² ajustado — cómo se usó exactamente

**Fórmula** (`preprocesamiento.py:101`, función `calcular_r2_ajustado`):

```python
def calcular_r2_ajustado(r2, n_muestras, n_features):
    return 1 - ((1 - r2) * (n_muestras - 1) / (n_muestras - n_features - 1))
```

- `n` = número de muestras **del conjunto donde se evalúa** (train o test).
- `p` = número de variables predictoras = `X_train.shape[1]` = **8** en concreto.
- Se llama desde `calcular_metricas` (`regresion.py:41`) **para train y para test**, y en la app
  (`app.py:1485-1486`, `1562`, `1585`).

**¿Por qué existe?** El R² normal **nunca baja** al agregar variables, aunque sean ruido. El
ajustado multiplica el error no explicado por (n−1)/(n−p−1) > 1: más variables → más
penalización. Sirve para comparar modelos con **distinto número de variables**.

**Ejemplo con los números reales** (árbol modelo 1, test, n = 145, p = 8):

```
factor   = (145 − 1) / (145 − 8 − 1) = 144 / 136 = 1.0588
R² adj   = 1 − (1 − 0.8926) × 1.0588 = 1 − 0.1137 = 0.8863   ✔ (coincide con la salida)
```

En train (n = 576) el factor es 575/567 = 1.014 → la penalización es mucho menor
(0.9953 → 0.9953). **Con más muestras, el R² ajustado se acerca al R² normal.**

**Matices que debes saber decir** (el profe puede preguntarlos):
1. El R² ajustado nace en la **regresión lineal**, donde *p* son los coeficientes del modelo.
   En árboles y bosques la "complejidad real" no es 8 (un árbol tiene cientos de hojas), así
   que aquí se usa *p* = número de variables de entrada como **convención**, no como grados de
   libertad reales.
2. Como **todos los modelos usan las mismas 8 variables y el mismo test (n = 145)**, el R²
   ajustado **no cambia el orden** de los modelos: es una transformación monótona del R². Se
   reporta porque la rúbrica lo pide y porque permitiría comparar si se quitaran variables.
3. El modelo 3 usa un test distinto (n = 217), así que su factor es diferente (216/208).

### 4.4 Cómo se detecta el sobreajuste en el código

**Clasificación** (`arboles_decision.py`, función `entrenar_y_evaluar`):
1. Si `acc_train − acc_test > 0.08` → "[!] Posible sobreajuste".
2. Si no, y `acc_test < 0.60` → "[!] Posible **subajuste**" (bajo en train y en test).
3. Si no → "[OK] Buen ajuste".

**Regresión** (`regresion.py`, función `diagnosticar_ajuste`, que también usa la app):
1. Si `R²_train − R²_test > 0.15` **o** `MAE_test / MAE_train > 3` → "[!] Posible sobreajuste".
2. Si no, y `R²_test < 0.60` → "[!] Posible subajuste".
3. Si no → "[OK] Buen ajuste".

**¿Por qué se agregó la razón de MAE?** El R² tiene techo en 1, así que un árbol que memoriza
(MAE en train ≈ 0.10 MPa) puede mostrar una brecha de R² "moderada" (0.10) y aun así
equivocarse en test **39 veces más** que en train. El MAE está en MPa (unidades reales) y
deja ver esa memorización. Los Random Forest quedan en ≈2× (es normal que el error en train
sea algo menor) y el árbol sin restricciones en 39×.

Son **umbrales heurísticos** (elegidos por criterio, no salen de una fórmula); están definidos
como constantes (`UMBRAL_BRECHA_R2`, `UMBRAL_RAZON_MAE`, `UMBRAL_R2_SUBAJUSTE`) para que se
vean y se puedan cambiar.

---

## 5. La matriz de confusión en cada modelo

### 5.1 Qué es

Matriz cuadrada *n × n* (*n* = 22 clases). **Filas = clase real**, **columnas = clase
predicha**. `CM[i, j]` = cuántas muestras de la clase real *i* se predijeron como *j*.

- **Diagonal** = aciertos. **Fuera de la diagonal** = errores, y dice **con qué** se confundió.
- Suma de la fila *i* = total real de esa clase (aquí 20 en test 80/20).
- Suma de la columna *j* = cuántas veces se predijo *j*.
- Recall de *i* = `CM[i,i] / suma fila i`; Precision de *j* = `CM[j,j] / suma columna j`;
  Accuracy = `traza / suma total`.

### 5.2 Cómo se calcula (código)

```python
from sklearn.metrics import confusion_matrix
clases = sorted(y.unique())                         # orden fijo de las 22 clases
cm = confusion_matrix(y_test, pred_test, labels=clases)
```

`labels=clases` es importante: fija el orden de filas/columnas y garantiza que la matriz sea
22×22 aunque alguna clase nunca se prediga (pasa en el modelo 2 del árbol).

**Si te piden hacerla "a mano"** (sin sklearn):

```python
idx = {c: i for i, c in enumerate(clases)}
cm = np.zeros((len(clases), len(clases)), dtype=int)
for real, pred in zip(y_test, pred_test):
    cm[idx[real], idx[pred]] += 1
```

### 5.3 Cómo se implementó en cada modelo

| Modelo | Dónde | Qué se compara | Color |
|---|---|---|---|
| Árbol (3 modelos) | `arboles_decision.py:81` | `y_test` vs `pred_test` (el modelo 3 usa `y_test3`, 660 muestras) | Naranja |
| Random Forest (9/49/491) | `random_forest.py:134` | `y_test` vs `pred_test` | Azul |
| KNN | `knn.py:120` (dentro de `evaluar_knn`) | `y_test` vs `pred_test` sobre datos **estandarizados** | Verde |
| K-means | `clustering.py:166` | `y_real` vs **clusters traducidos a clases** (algoritmo húngaro) | Morado |
| Regresión | `regresion.py:77` | **No hay matriz de confusión** → gráfico *real vs. predicho* | — |

Los cuatro clasificadores llaman a la **misma función de dibujo**
`figura_matriz_confusion` (`graficos.py:49`), así todas se leen igual.

### 5.4 La matriz en K-means (la más difícil de explicar)

K-means no predice "apple", predice "cluster 7". Los números de cluster son arbitrarios, así
que primero hay que **emparejar cada cluster con una clase real**:

1. Se construye una **matriz de coincidencias** (clusters × clases): cuántos puntos de cada
   cultivo cayeron en cada cluster.
2. Se usa el **algoritmo húngaro** (`scipy.optimize.linear_sum_assignment`) para encontrar el
   emparejamiento **uno a uno** cluster↔clase que **maximiza el total de aciertos**. Como esa
   función *minimiza* un costo, se le pasa la matriz con signo negativo (`-matriz_coincidencias`).
3. Si hay más clusters que clases, los sobrantes se mapean a su clase mayoritaria.
4. Con cada punto traducido a clase, se calcula `confusion_matrix` y `accuracy` normal.

¿Por qué húngaro y no simplemente "clase mayoritaria de cada cluster"? Porque con mayoritaria
dos clusters podrían quedar asignados a la misma clase y el resultado se inflaría. El húngaro
garantiza una correspondencia uno a uno.

### 5.5 ¿Y en regresión?

No hay clases, así que no hay matriz de confusión. Su equivalente visual es el gráfico
**real vs. predicho** (`graficos.py:167`): eje X = csMPa real, eje Y = csMPa predicho, línea
diagonal = predicción perfecta. Puntos encima de la línea = sobreestimó; debajo = subestimó.

**Si el profe insiste en una matriz de confusión para regresión**, se puede hacer
**discretizando** la variable en rangos y comparando rangos:

```python
bins = [0, 20, 40, 60, 90]; nombres = ["baja", "media", "alta", "muy alta"]
real_cat = pd.cut(y_test, bins=bins, labels=nombres)
pred_cat = pd.cut(pred_test, bins=bins, labels=nombres)
confusion_matrix(real_cat, pred_cat, labels=nombres)
```

(Esto no está implementado en el proyecto; es la respuesta de "cómo lo harías").

### 5.6 Detalles del diseño del gráfico (`graficos.py:49`)

- Aciertos (diagonal) en la escala de color del modelo; **errores en rojo** con **escala
  logarítmica** (así 1 error se ve, sin que 10 errores saturen el color).
- Celdas en cero quedan grises y vacías (con 22×22 = 484 celdas, casi todas son cero).
- *Hover*: cantidad y % sobre el total de la clase real. Botón para ocultar los números.
- `guardar_figura` (`graficos.py:203`) guarda HTML interactivo + PNG (con `kaleido`).

---

## 6. Árboles de decisión (clasificación)

### 6.1 Cómo funciona (algoritmo CART, el que usa sklearn)

1. En cada nodo prueba **todas las variables y todos los umbrales posibles** y elige la
   división "¿variable ≤ umbral?" que más **reduce la impureza**.
2. Repite recursivamente en cada hijo hasta que los nodos sean puros o se cumpla una
   restricción (pre-poda).
3. Una hoja predice la **clase mayoritaria** de las muestras de entrenamiento que llegaron a ella.

Divisiones **binarias** siempre (CART). Es un algoritmo **voraz (greedy)**: elige la mejor
división local en cada paso, no el mejor árbol global.

**Medidas de impureza:**

| Criterio | Fórmula | Nodo puro | Máximo con 22 clases balanceadas |
|---|---|---|---|
| Gini (por defecto) | 1 − Σ pᵢ² | 0 | 1 − 22·(1/22)² = **0.9545** |
| Entropía | −Σ pᵢ log₂ pᵢ | 0 | log₂ 22 = **4.46 bits** |

**Ganancia de información** = impureza(padre) − promedio ponderado de impureza(hijos). Se
elige la división con mayor ganancia. Gini y entropía casi siempre eligen divisiones
parecidas; Gini es un poco más rápido (no calcula logaritmos).

**Importancia de variables** (`feature_importances_`): suma de la reducción de impureza que
aportó cada variable en todos sus nodos, normalizada para que sume 1. En el árbol 1:
rainfall 0.354, P 0.225, humidity 0.149, K 0.111, N 0.099, temperature 0.054, ph 0.007.

### 6.2 Los 3 modelos (punto 4a: "3 formas de mandarle la base de datos")

| Modelo | Configuración | Partición | Profundidad | Hojas | Acc train | Acc test | Tiempo |
|---|---|---|---|---|---|---|---|
| 1 | Sin restricciones, Gini | 80/20 | 17 | 38 | 100 % | **97.95 %** | ~0.007 s |
| 2 | Pre-poda `max_depth=4`, `min_samples_leaf=5` | 80/20 | 4 | 7 | 31.82 % | **31.82 %** | ~0.005 s |
| 3 | Sin restricciones, **entropía** | **70/30** | 13 | 42 | 100 % | **98.79 %** | ~0.012 s |
| Podado (post-poda) | `ccp_alpha = 0.001213` | 80/20 | 15 | 32 | 99.72 % | **98.18 %** | — |

Qué varía entre ellos: **la partición** (80/20 vs 70/30), **las restricciones** (sin límite vs
pre-poda) y **el criterio** (Gini vs entropía).

**Errores del modelo 1** (9 de 440): blackgram→lentil (2), lentil→mothbeans (2),
blackgram→maize, blackgram→mothbeans, mothbeans→lentil, jute→rice, rice→jute.
→ **Todos entre cultivos con condiciones parecidas** (leguminosas entre sí; arroz y yute).

**Errores del modelo 3** (8 de 660): blackgram→mothbeans (2), papaya→apple (2), rice→jute
(2), blackgram→maize, cotton→maize.

**¿Por qué el modelo 2 da exactamente 31.82 %?** (pregunta muy probable). Con `max_depth=4`
y `min_samples_leaf=5` el árbol terminó con **7 hojas**, así que **solo puede predecir 7 de
los 22 cultivos** (apple, banana, blackgram, chickpea, grapes, kidneybeans, muskmelon). Si
acierta todas las muestras de esas 7 clases: 7 × 20 / 440 = **0.3182**. Es exactamente su
techo. Es un caso claro de **subajuste**: el modelo es demasiado simple para 22 clases (un
árbol de profundidad 4 tiene como máximo 2⁴ = 16 hojas, menos que 22 clases). El script lo
imprime así: `[!] Posible subajuste (accuracy test = 0.3182, bajo en train y test)`. Una
brecha de 0 **no** significa buen ajuste: significa que es igual de malo en ambos.

**¿Por qué el árbol sin restricciones casi no sobreajusta?** Porque el dataset es **limpio y
las clases están bien separadas**: no hay ruido que memorizar. Llega a 100 % en train y aun
así 98 % en test (brecha 2 %). Para ver el sobreajuste de verdad, la app inyecta ruido en las
etiquetas (sección 11).

**¿Se puede comparar el modelo 3 con el 1?** No del todo: el modelo 3 se evalúa en **otro
test** (660 muestras, 70/30). Su 98.79 % no significa que la entropía sea mejor; también
cambiaron los datos de prueba.

### 6.3 Pre-poda vs. post-poda (punto 4c)

| | Pre-poda | Post-poda |
|---|---|---|
| Cuándo | **Antes/durante** el crecimiento | **Después** de crecer el árbol completo |
| Cómo | Restricciones: `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_leaf_nodes` | Recorte de ramas por costo-complejidad (`ccp_alpha`) |
| Ventaja | Rápida | No descarta divisiones que parecían malas pero tenían buenas divisiones debajo |
| Riesgo | Detenerse demasiado pronto (subajuste, como el modelo 2) | Más costosa |

**Post-poda por costo-complejidad (Minimal Cost-Complexity Pruning):**

```
R_α(T) = R(T) + α · |hojas(T)|
```

`R(T)` = error/impureza total del árbol; `|hojas|` = número de hojas; α = penalización por
cada hoja. Con α = 0 se queda el árbol completo; al subir α conviene cortar ramas; con α muy
grande queda solo la raíz.

Implementación (`arboles_decision.py:133`):

```python
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas                           # los alphas donde se poda una rama
alphas_muestra = ccp_alphas[:: max(1, len(ccp_alphas) // 25)]   # máx. ~25 valores
for alpha in alphas_muestra:
    arbol = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    acc_cv = cross_val_score(arbol, X_train, y_train, cv=5).mean()  # ← CRITERIO DE ELECCIÓN
    arbol.fit(X_train, y_train)
    # se guarda también accuracy en train y en test, solo para graficar/reportar
mejor_alpha = alphas_muestra[argmax(acc_cv)]           # elegido SIN mirar el test
```

**El alpha se elige con validación cruzada de 5 pliegues sobre el train**, no con el test. Si
se eligiera con el test, el test se estaría usando para ajustar un hiperparámetro y dejaría de
ser una evaluación independiente (el resultado sería optimista). El test solo se usa al final
para reportar cómo le va al alpha ya elegido.

- En clasificación la ruta tiene **22 alphas** (se prueban todos, porque 22 // 25 = 0 → paso 1).
- Mejor alpha por CV = **0.001213** (CV 98.58 %) → 32 hojas (vs 38), train 99.72 %, test
  **98.18 %** (0.23 puntos más que el árbol completo). Aquí la CV eligió el mismo alpha que
  habría elegido el test, lo que da confianza en que la mejora es real.
- El gráfico `poda_ccp_alpha.png` muestra train, test y la curva de validación cruzada
  (punteada) vs. alpha, con la brecha sombreada y una estrella en el alpha elegido por CV.
- En la app (botón "Ejecutar análisis de post-poda") se hace lo mismo.

---

## 7. Random Forest (clasificación)

### 7.1 Cómo funciona

Es un **ensamble** de árboles (técnica **bagging** = *bootstrap aggregating*) con dos fuentes
de aleatoriedad:

1. **Bootstrap (filas):** cada árbol se entrena con una muestra del mismo tamaño que el train
   (1760) tomada **con reemplazo** → algunas filas se repiten y otras quedan fuera.
2. **Subconjunto aleatorio de variables (columnas):** en cada división el árbol solo puede
   elegir entre `max_features` variables al azar. En clasificación sklearn usa por defecto
   `max_features="sqrt"` → √7 ≈ 2.6 → **2 variables candidatas por división**.

Así los árboles quedan **distintos y poco correlacionados**; cada uno sobreajusta a su manera,
pero al combinarlos los errores se cancelan → **baja la varianza sin subir el sesgo**.

**Cómo decide sklearn:** **no** es votación dura. Promedia las **probabilidades**
(`predict_proba`) de todos los árboles y elige la clase con mayor probabilidad promedio
(*soft voting*). Con árboles completos cada árbol da casi siempre probabilidad 1 a una clase,
así que en la práctica se parece a contar votos.

### 7.2 Por qué 9, 49 y 491 árboles (impares) — punto 5a

Con número **par**, dos clases podrían empatar exactamente en votos. Con número **impar** en
un problema **binario** el empate es imposible; con **22 clases** el impar **reduce pero no
elimina** el riesgo (p. ej. 49 votos: 20 / 20 / 9 sería un empate entre dos clases). Por eso
además se define una **regla de desempate**.

### 7.3 Regla de desempate — punto 5b (`random_forest.py:62`)

`predecir_con_regla_desempate` simula una **votación dura** árbol por árbol para la primera
muestra del test:

```python
votos = [arbol.predict(fila)[0] for arbol in modelo.estimators_]
conteo = Counter(votos)
empatados = [clase for clase, v in conteo.items() if v == max(conteo.values())]
# Si hay empate → gana la clase empatada con mayor probabilidad promedio del bosque (predict_proba)
```

Resultado real con 491 árboles: **479 votos para *orange*** y 12 para *pomegranate* → no
hubo empate.

Detalle de implementación: los árboles internos del bosque (`modelo.estimators_`) se
entrenan con las clases **codificadas como índices** (0.0 … 21.0), así que su `predict`
devuelve un número, no un nombre. El código lo traduce con
`modelo.classes_[int(voto)]` (el índice 16 en orden alfabético es *orange*). Si hay empate,
busca la probabilidad de cada clase empatada en `predict_proba` y gana la mayor; si las
probabilidades también empataran, gana la primera en orden alfabético (determinista).

Se probó forzando un empate real (un bosque de 2 árboles que votan distinto):
`Empate detectado entre: ['grapes', 'pomegranate']` → `gana 'pomegranate'` (mayor
probabilidad promedio).

**Respuesta corta:** "En sklearn el empate prácticamente no ocurre porque se promedian
probabilidades, no votos. Si se usara votación dura y hubiera empate, se desempata con la
probabilidad promedio; si aun así empataran, `argmax` escoge la primera clase en orden (el
índice menor)."

### 7.4 Bootstrap — punto 5d (`random_forest.py:140`)

La probabilidad de que una fila **no** salga en una muestra bootstrap de tamaño *n* es
(1 − 1/n)ⁿ → **e⁻¹ ≈ 0.368**. Es decir, cada árbol ve ≈ **63.2 %** de filas únicas y el 36.8 %
queda fuera (*out-of-bag*, OOB).

Evidencia real del código (n = 1760):
- 640 posiciones repetidas en la muestra del árbol 1 → **36.4 %** (cerca del 36.8 % teórico).
- Coincidencia entre la muestra del árbol 1 y la del árbol 2: 708 / 1760 = **40.2 %**
  (teórico: 0.632 × 0.632 ≈ 0.40). → Cada árbol entrena con un subconjunto distinto.

### 7.5 Resultados y tiempos — punto 5c

| Árboles | Tiempo (s) | Acc train | Acc test | Errores en test |
|---|---|---|---|---|
| 9 | ~0.024 | 100 % | 99.32 % | 3 (rice→jute ×2, blackgram→maize) |
| 49 | ~0.068 | 100 % | **99.55 %** | 2 (blackgram→maize, rice→jute) |
| 491 | ~0.587 | 100 % | **99.55 %** | 2 (los mismos) |

- El **tiempo crece aproximadamente lineal** con el número de árboles (cada árbol es un
  entrenamiento independiente). `n_jobs=-1` usa todos los núcleos del procesador en paralelo.
- El **accuracy se satura**: de 49 a 491 árboles (10× el tiempo) no mejora nada. Conclusión:
  más árboles **no causan sobreajuste**, pero llega un punto en que solo cuestan tiempo.
- Train = 100 % en todos y aun así test 99.5 %: el bosque memoriza el train, pero generaliza
  porque los errores individuales se promedian.
- Los tiempos medidos con `time.time()` varían un poco entre ejecuciones (carga del PC).

**Importancia de variables (491 árboles):** rainfall 0.223, humidity 0.213, K 0.179, P 0.152,
N 0.106, temperature 0.075, ph 0.053. Es el promedio de la reducción de impureza de cada
variable en todos los árboles.

**RF vs. árbol único:** 99.55 % vs 97.95 % (de 9 errores a 2). El RF es más preciso y estable,
pero menos interpretable (no se puede "leer" un bosque de 491 árboles) y más lento.

---

## 8. Regresión: árboles y Random Forest sobre concreto

### 8.1 Qué cambia respecto a clasificación

- **Criterio de división:** en vez de Gini/entropía, se minimiza el **error cuadrático
  (`squared_error`, por defecto)** = la **varianza** dentro de cada hijo. El modelo 3 usa
  `criterion="absolute_error"` (MAE).
- **Predicción de una hoja:** el **promedio** de `csMPa` de sus muestras de entrenamiento
  (con `absolute_error`, sklearn usa la **mediana**).
- **Random Forest:** la predicción es el **promedio** de los árboles (no hay votación).
  Detalle: en sklearn el `RandomForestRegressor` tiene por defecto `max_features=1.0`
  (**todas** las variables en cada división), así que su única fuente de aleatoriedad es el
  bootstrap (es *bagging* de árboles).
- **Métricas:** MAE, RMSE, R², R² ajustado. **Sin matriz de confusión** (gráfico real vs. predicho).
- **Sin estratificar** la partición (y continua).
- Las predicciones de un árbol son **escalonadas**: solo hay tantos valores posibles como hojas.

### 8.2 Árbol de regresión: 3 modelos + post-poda

| Modelo | Config. | Part. | Hojas | R² train | R² test | R² adj test | MAE test | RMSE test |
|---|---|---|---|---|---|---|---|---|
| 1 | Sin restricciones | 80/20 | **543** | 0.9953 | **0.8926** | 0.8863 | 4.005 | 5.445 |
| 2 | `max_depth=4`, `min_samples_leaf=5` | 80/20 | 16 | 0.7602 | 0.7352 | 0.7196 | 7.009 | 8.550 |
| 3 | `criterion="absolute_error"` | 70/30 | 472 | 0.9922 | 0.7968 | 0.7890 | 5.220 | 7.725 |
| Podado (α por CV) | `ccp_alpha = 0.253502` | 80/20 | 73 | 0.9420 | 0.8612 | — | — | — |

Diagnóstico que imprime el script (ver 4.4):

| Modelo | Brecha R² | MAE test / MAE train | Diagnóstico |
|---|---|---|---|
| Árbol 1 | 0.103 | **38.9×** | [!] Sobreajuste |
| Árbol 2 | 0.025 | 1.0× | [OK] |
| Árbol 3 | 0.195 | **54.6×** | [!] Sobreajuste |
| RF 9 / 49 / 491 | 0.054 / 0.050 / 0.044 | 2.0× / 2.1× / 2.0× | [OK] |

Interpretación:
- **Modelo 1 está sobreajustado**: 543 hojas para 576 registros de entrenamiento (casi una
  hoja por dato), MAE en train 0.10 MPa vs **4.0 MPa en test** (39 veces más error). Antes el
  script lo marcaba "[OK]" porque solo miraba la brecha de R² (0.103 < 0.15); ahora también
  mira la razón de MAE y lo detecta. Aun así es el árbol con mejor test: memorizar no
  empeora el test en este dataset, solo deja de mejorarlo.
- **Modelo 2** queda como "[OK]": R² ≈ 0.74 en ambos (brecha pequeña y sobre el umbral de
  subajuste de 0.60). Es simple, pero no tanto como para ser subajuste.
- **Modelo 3** también sobreajusta (brecha 0.195 y 54.6×). Tiene menos datos de entrenamiento
  y otro test, así que no es comparable 1 a 1.
- **Post-poda**: la ruta tiene **506 alphas**; se prueban ≈26 (cada 20). El alpha se elige con
  **validación cruzada 5-fold** (`KFold` barajado) sobre el train → **α = 0.2535** (R² CV
  0.768), que deja un árbol de **73 hojas** (vs 543) con R² test **0.8612**.
  - **Pregunta probable: "¿por qué el árbol podado da peor test que el árbol completo?"**
    Antes, el alpha se elegía mirando el test y salía 0.001112 con R² test 0.8932 (casi el
    árbol completo). Ese número era optimista: el test se usó para escoger. Con validación
    cruzada, que es lo correcto, se elige un árbol 7 veces más pequeño que en test rinde 0.861.
    La diferencia (~0.03) es el "sesgo optimista" que tenía el método anterior; y la CV del
    árbol (≈0.77) muestra que con 576 datos un solo árbol tiene mucha varianza entre particiones.
    Por eso el modelo recomendado para regresión es el **Random Forest** (R² test 0.939).

### 8.3 Random Forest de regresión (9 / 49 / 491)

| Árboles | Tiempo (s) | R² train | R² test | R² adj test | MAE test | RMSE test |
|---|---|---|---|---|---|---|
| 9 | ~0.020 | 0.9759 | 0.9217 | 0.9171 | 3.552 | 4.650 |
| 49 | ~0.048 | 0.9823 | 0.9328 | 0.9289 | 3.244 | 4.306 |
| 491 | ~0.371 | 0.9831 | **0.9389** | **0.9353** | **3.122** | **4.107** |

- Aquí sí mejora un poco con más árboles (en regresión promediar más valores reduce más la
  varianza), pero con rendimientos decrecientes.
- Brecha de R² ≈ 0.044 (vs 0.103 del árbol solo) → **el ensamble reduce el sobreajuste**.
- **Importancia de variables (491 árboles):** age 0.351, cement 0.309, superplasticizer 0.090,
  water 0.084, slag 0.076, fineaggregate 0.042, coarseaggregate 0.030, flyash 0.017.
  Tiene sentido físico: la resistencia depende sobre todo del **tiempo de curado** y de la
  **cantidad de cemento** (y de la relación agua/cemento).

### 8.4 Curva de complejidad y curva de aprendizaje (app)

Usan **validación cruzada 5-fold** (`KFold(5, shuffle=True, random_state=42)`) sobre los 721 datos.

- **Curva de complejidad** (`validation_curve` con `max_depth` de 1 a 20): train sube hasta
  0.996; test sube y luego **se estanca** en ≈0.84 (mejor en profundidad 12 con 0.845). El
  sobreajuste aquí se ve como una **brecha que crece**, no como una caída del test (no hay "U").
- Profundidad "bien ajustada" = la más simple cuyo test está a menos de 1 desviación estándar
  del mejor (**regla de un error estándar**) → **max_depth = 7**.
- Con partición 80/20: subajustado (depth 2) R² test 0.429; bien ajustado (depth 7) 0.874;
  sin restricción 0.893; RF 200 árboles **0.937**.
- Matiz honesto (la app lo muestra): el árbol sin restricción saca un test un poco **mejor**
  que el de depth 7. Se le llama sobreajustado por la **brecha** y por tener una hoja por dato;
  el de depth 7 gana en **simplicidad e interpretabilidad**, no en precisión. Lo que sí mejora
  claramente el test es el Random Forest.
- **Curva de aprendizaje** (`learning_curve`): con pocos datos el árbol memoriza (train ≈ 1)
  y en test le va mal; al agregar datos la brecha se cierra pero no desaparece → con 721
  registros el árbol solo sigue teniendo alta varianza.

---

## 9. KNN

### 9.1 Cómo funciona (punto 7a y 7c)

Para clasificar un punto nuevo:
1. Calcula la **distancia** a **todos** los puntos de entrenamiento (por defecto
   **euclidiana**: `metric="minkowski", p=2`).
2. Toma los **K más cercanos**.
3. **Vota** la clase mayoritaria (con `weights="uniform"`) o pondera por cercanía (`"distance"`).

**"No utiliza un modelo" (*lazy learning*, aprendizaje perezoso / basado en instancias):** el
`fit` solo **guarda los datos** (y construye una estructura de búsqueda); no aprende reglas ni
parámetros. Todo el costo está en **predecir**. Evidencia (`knn.py:100`): `fit` ≈ 0.002–0.003 s
y `predict` de 440 muestras ≈ 0.002–0.008 s. En un árbol es al revés: entrenar cuesta, predecir
es casi instantáneo.

Consecuencias: el "modelo" ocupa tanta memoria como el dataset; predecir es lento con muchos
datos; es muy sensible a la escala (por eso **datos estandarizados**) y a variables irrelevantes.

### 9.2 Cómo se eligió K (punto 7b) — `knn.py:32`

Se prueban K = 1…25 con **validación cruzada de 5 pliegues** (`cross_val_score(cv=5)`)
**solo sobre el train estandarizado** (el test no se toca para elegir K).

Validación cruzada 5-fold: el train se divide en 5 partes; se entrena con 4 y se evalúa con
la restante, 5 veces rotando; se promedian los 5 accuracies. Es más confiable que una sola
partición.

| K | 1 | 2 | 3 | 4 | 5 | 7 | 9 | 11 | 15 | 21 | 25 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Acc CV | **0.9705** | 0.9597 | **0.9699** | 0.9648 | 0.9653 | 0.9619 | 0.9557 | 0.9540 | 0.9449 | 0.9341 | 0.9273 |

**Regla de selección del código:** se toman los K cuyo accuracy está a menos de 0.002 del
mejor (candidatos: K=1 y K=3), se prefieren **impares** (para evitar empates) y entre ellos
el **menor** → **K = 1**.

Lectura de la curva: el accuracy **baja a medida que K crece** (con K grande se promedian
vecinos lejanos de otras clases → **subajuste**). Los **K pares caen** (K=2: 0.9597 vs K=1 y
K=3 ≈ 0.970): con K par hay más empates en la votación → evidencia práctica de por qué se
prefieren impares.

**¿K=1 no es sobreajuste?** En teoría K=1 es el más propenso a sobreajustar (memoriza cada
punto y su ruido). Aquí funciona porque las clases están **muy bien separadas y casi sin
ruido**. K=3 da prácticamente lo mismo (0.9699 vs 0.9705, diferencia de 0.06 puntos) y sería
una elección **más robusta** si hubiera ruido. Decirlo así demuestra criterio.

### 9.3 Resultado final

K=1, uniform, test: **97.95 %** (9 errores de 440): rice→jute (2), blackgram→lentil,
lentil→mothbeans, maize→cotton, mothbeans→blackgram, mothbeans→lentil, mothbeans→mango,
pigeonpeas→kidneybeans.

### 9.4 Ponderación (punto 7d: "dar más peso para evitar empates") — `knn.py:131`

| Estrategia | Qué hace | Acc test (K=1) |
|---|---|---|
| `weights="uniform"` | Todos los K vecinos votan igual | 97.95 % |
| `weights="distance"` | Cada vecino vota con peso 1/distancia (el más cercano pesa más) | 97.95 % |
| **Variables ponderadas** | Multiplica cada variable estandarizada por su importancia relativa | **98.86 %** (5 errores) |

- Con K=1 `uniform` y `distance` dan **lo mismo**, porque solo hay un vecino: no hay nada
  que ponderar. La diferencia aparecería con K ≥ 2.
- **Ponderación por variable:** se entrena un Random Forest de referencia (200 árboles)
  **sobre el train**, se toman sus `feature_importances_` y se normalizan alrededor de 1
  (`pesos = importancias / importancias.mean()`). Pesos: rainfall 1.537, humidity 1.519,
  K 1.266, P 1.059, N 0.723, temperature 0.528, ph 0.366. Luego `X * pesos` en train **y**
  test. Multiplicar una variable por un peso > 1 la "estira" y hace que influya más en la
  distancia.
- Cómo esto evita empates: con `distance`, dos vecinos a distinta distancia nunca pesan igual;
  con variables ponderadas, las variables discriminantes deciden quién está más cerca.
- **Regla de desempate de sklearn** (si de todas formas hay empate de votos): gana la clase
  con **índice menor** en `classes_` (la primera en orden alfabético).

---

## 10. Clustering (K-means)

### 10.1 Datos sin clase vs. con clase (punto 8a y 8d)

Se usa **la misma base de cultivos** pero **sin la columna `label`** al agrupar (aprendizaje
**no supervisado** de verdad). La etiqueta se usa **después**, solo para evaluar qué tan bien
los grupos coinciden con los cultivos reales.

Se estandariza con Z-score (K-means usa distancias). Aquí el `StandardScaler` se ajusta con
**todo X** (`estandarizar(X, X)`), porque no hay train/test: no hay un conjunto de prueba
al que se le pueda filtrar información.

### 10.2 Algoritmo K-means y el centroide (punto 8c)

**Centroide** = el punto promedio (media de cada variable) de los datos de un cluster.

1. **Inicialización:** se eligen K centroides. sklearn usa **k-means++** (el primer centroide
   al azar; los siguientes con mayor probabilidad lejos de los ya elegidos).
2. **Asignación:** cada punto va al centroide más cercano.
3. **Actualización:** cada centroide se recalcula como el promedio de sus puntos.
4. Se repiten 2–3 hasta que los centroides casi no se mueven (**convergencia**).

Minimiza la **inercia** (WCSS) = Σ ‖x − μ_cluster‖². `n_init=10` → corre el algoritmo 10 veces
con distintas inicializaciones y se queda con la de **menor inercia** (K-means puede caer en
óptimos locales).

**Evidencia del movimiento de los centroides** (`clustering.py:133`): se implementa K-means
**a mano** (con NumPy, inicialización aleatoria simple) y se mide cuánto se mueven los
centroides en cada iteración (K=22):

| Iteración | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Desplazamiento promedio | 0.697 | 0.317 | 0.186 | 0.100 | 0.086 |

Disminuye → converge. (El K-means de sklearn con K=22 convergió en 10 iteraciones.)

### 10.3 Tres técnicas para elegir K (punto 8b) — `clustering.py:36`

Se prueba K = 2…30.

| Técnica | Qué mide | Cómo se elige | Resultado |
|---|---|---|---|
| **Codo (inercia)** | Suma de distancias² de cada punto a su centroide | Donde la curva deja de bajar bruscamente (visual) | No hay un codo nítido |
| **Silueta** | s = (b − a) / max(a, b); *a* = distancia media a su propio cluster, *b* = al cluster vecino más cercano. Rango −1 a 1 | **Máximo** | **K = 2** (0.417) |
| **Davies-Bouldin** | Promedio, para cada cluster, del peor cociente (dispersión_i + dispersión_j) / distancia entre centroides | **Mínimo** | **K = 2** (0.797) |

| K | 2 | 5 | 10 | 20 | 22 | 30 |
|---|---|---|---|---|---|---|
| Inercia | 11738 | 6856 | 4110 | 2390 | 2200 | 1713 |
| Silueta | **0.417** | 0.293 | 0.349 | 0.357 | 0.348 | 0.346 |
| Davies-Bouldin | **0.797** | 1.280 | 1.057 | 1.092 | 1.106 | 1.064 |

La inercia **siempre baja** al subir K (con K = n sería 0), por eso no sirve sola: se busca el
codo. Dato: con datos estandarizados la inercia con K=1 es n·p = 2200 × 7 = 15400; con K=22
queda en 2200 → los 22 grupos explican ≈ 86 % de la variabilidad.

**¿Por qué las métricas eligen K=2 si hay 22 cultivos?** Con K=2, K-means separa
**apple + grapes** (cluster 1, exactamente 200 puntos) de **los otros 20 cultivos**. Apple y
grapes tienen P ≈ 133 y K ≈ 200, muy lejos de todos los demás. Esa separación es enorme y
limpia → silueta alta. Los otros 20 cultivos se solapan entre sí, y dividirlos en grupos
más pequeños da clusters menos compactos y menos separados. **Las métricas internas miden
geometría, no "verdad agronómica".**

### 10.4 Comparación con la realidad (punto 8b y 8e)

- **K = 22** (el real): coincidencia con los cultivos = **74.95 %** sin haber visto nunca las etiquetas.
- **K = 2** (el de silueta/DB): coincidencia = **9.09 %** = 2/22, porque con 2 clusters solo
  se pueden "acertar" 2 clases (200 de 2200 puntos).

Este 74.95 % **no es accuracy de un clasificador**: es qué tan bien los grupos naturales
coinciden con las clases después de emparejarlos con el algoritmo húngaro (sección 5.4).

### 10.5 PCA (app)

Para ver 7 dimensiones en un plano se usa **PCA** (2 componentes). PC1 explica 27.6 % y PC2
18.5 % de la varianza (**46.1 %** en total) → es solo una proyección para visualizar; **el
K-means se hizo con las 7 variables**. PC1 está dominado por **P (0.644) y K (0.623)**, que es
justo lo que separa a apple y grapes.

---

## 11. La app de Streamlit y las demostraciones extra

`streamlit run app.py` → 7 pestañas. Cada botón **entrena en vivo** (nada precalculado) y
cada sección tiene un desplegable "📚 Teoría".

| Pestaña | Qué tiene |
|---|---|
| 📖 Introducción | Supervisado/no supervisado, sobreajuste, tipos de variable, descripción de datasets, histograma de csMPa, métricas, **ejemplo real de matriz de confusión** con blackgram/lentil/mothbeans |
| ⚙️ Preprocesamiento | Antes/después de Z-score y Min-Max (estadísticas y filas), **Mutual Information** |
| 🌳 Árboles | Árbol configurable (`max_depth`, criterio, % test), matriz de confusión, precision/recall/F1 por clase, **dibujo del árbol completo** (SVG con zoom), comparación de los 3 modelos, post-poda, **demostración con ruido** |
| 🌲 Random Forest | Bosque de 9/49/491, matriz, importancia de variables, ver cualquier árbol individual del bosque, comparación de tamaños, bootstrap |
| 📈 Regresión | Árbol o RF, MAE/RMSE/R²/R² ajustado, real vs. predicho, comparación de 3 árboles + 3 bosques, curvas de complejidad y aprendizaje |
| 🔎 KNN | K y ponderación configurables, curva K vs. accuracy (CV), matriz, comparación de ponderaciones |
| 🧩 Clustering | K configurable, silueta, DB, coincidencia con clases reales (húngaro), matriz, **PCA** con centroides, codo/silueta/DB de 2 a 30 |

**El dibujo del árbol** (`app.py:338-579`) no usa `plot_tree` de sklearn (se solapa con 22
clases). Usa un layout propio: a cada **hoja** se le asigna su propia columna y cada nodo
interno se ubica en el **promedio** de sus hijos → es imposible que dos cajas se solapen. Se
exporta como **SVG** (vectorial, no se pixela con zoom) y PDF.

### 11.1 Demostración de sobreajuste con ruido (pestaña Árboles)

Como el dataset es tan limpio que casi no se ve sobreajuste, se **corrompe el 25 % de las
etiquetas de entrenamiento** (440 de 1760 se cambian a un cultivo incorrecto al azar). El
test se deja limpio.

| Modelo | Acc train (con ruido) | Acc test (limpio) | Hojas |
|---|---|---|---|
| 🔴 Sobreajustado (sin restricción) | 100 % | 66.6 % | 619 |
| 🟢 Bien ajustado (`ccp_alpha` elegido por **validación cruzada** 3-fold sobre el train) | 74.0 % | **94.5 %** | 29 |
| 🟡 Subajustado (`max_depth=1`) | 9.1 % | 9.1 % | 2 |
| 🌲 Random Forest (200 árboles) | 100 % | **99.1 %** | — |

Puntos para explicar:
- El sobreajustado **memorizó el ruido** (619 hojas, 100 % en train) → cae a 66.6 % en test.
- **¿Por qué el bien ajustado tiene más accuracy en test que en train?** Porque el train tiene
  25 % de etiquetas falsas: cuando el modelo predice el cultivo **verdadero** de una muestra
  corrupta, eso cuenta como "error" contra la etiqueta falsa. El máximo razonable en train es
  ≈ 75 %, y sacó 74 %. **Es la señal de que aprendió el patrón real e ignoró el ruido.**
- El subajustado: 2 hojas → solo puede predecir 2 cultivos → 2/22 = 9.1 %.
- El RF memoriza el train (100 %) pero generaliza (99.1 %): cada árbol ve una muestra bootstrap
  y variables distintas, así que el ruido no se repite igual en todos y se cancela al promediar.
- Aquí el alpha se elige **sin mirar el test** (validación cruzada), que es la forma correcta.

---

## 12. Puntos débiles: lo que te pueden atacar y cómo responder

Un profesor exigente busca ver que **entiendes las limitaciones**. Reconocerlas con
argumentos suma más que negarlas.

| # | Lo que puede señalar | Cómo responder |
|---|---|---|
| 1 | **¿Cómo diagnostican el ajuste del modelo 2 (31.8 %)?** *(corregido)* | "Como **subajuste**. Una brecha de 0 no es buen ajuste si el accuracy es bajo en ambos; el script revisa la brecha **y** que `acc_test ≥ 0.60`. Con 7 hojas solo puede predecir 7 de 22 clases (techo 7×20/440 = 31.8 %)." |
| 2 | **¿Por qué el árbol de regresión 1 es sobreajuste si la brecha de R² es solo 0.10?** *(corregido)* | "Porque el R² tiene techo en 1 y esconde la memorización. El diagnóstico también mira MAE test/train: 4.0 vs 0.10 MPa = 39×. 543 hojas para 576 datos. Los RF quedan en ≈2×." |
| 3 | **¿Cómo eligieron el `ccp_alpha`?** *(corregido)* | "Con validación cruzada 5-fold sobre el train; el test solo reporta el resultado. En clasificación da el mismo alpha (0.001213). En regresión da un árbol más podado (73 hojas) con R² test 0.861; antes, eligiéndolo con el test, salía 0.893, pero ese número era optimista." |
| 4 | **R² ajustado con p = 8 en árboles** | Ver 4.3: es la convención con *p* = variables de entrada; en modelos no lineales no son grados de libertad reales, y como todos usan las mismas 8 variables no cambia el ranking. |
| 5 | **¿Por qué R² ajustado sobre el test y no sobre train?** | "Se calculan ambos. En la tabla se reporta el de test porque es el que mide generalización; *n* es el tamaño del conjunto evaluado." |
| 6 | **K = 1 en KNN** | "Ganó en CV (0.9705), pero K=3 está a 0.06 puntos y sería más robusto. El dataset casi no tiene ruido, por eso K=1 no se castiga." |
| 7 | **Duplicados en concreto (16 filas)** | "No se eliminaron. Si un duplicado cae en train y su copia en test, el test es un poco optimista (especialmente para el árbol sin restricción, que memoriza). Mejora: `df.drop_duplicates()` antes de dividir." |
| 8 | **Una sola partición train/test** | "Los resultados dependen de la semilla 42. Para estimaciones más robustas se puede usar validación cruzada (se hizo en KNN, en la curva de complejidad y en la demo con ruido). Con 440 muestras de test, 1 error = 0.23 puntos, así que diferencias de 1–2 errores no son significativas." |
| 9 | **El modelo 3 no es comparable** | "Usa otro test (70/30). Su 98.79 % no prueba que la entropía sea mejor; cambiaron dos cosas a la vez (partición y criterio)." |
| 10 | **¿Probaron la regla de desempate?** *(corregido)* | "Sí. Los árboles internos del RF devuelven índices de clase, así que se traducen con `modelo.classes_[int(voto)]`. Se forzó un empate (2 árboles votando grapes y pomegranate) y la regla eligió pomegranate por mayor probabilidad promedio. En el bosque de 491 no hay empate: 479 votos para orange." |
| 11 | **El scaler de clustering se ajusta con todos los datos** | "No hay test en clustering; no existe fuga de información posible. Es lo correcto." |
| 12 | **Tiempos medidos con `time.time()`** | "Son tiempos de reloj, varían por la carga del equipo; sirven para comparar órdenes de magnitud. Para más precisión, `time.perf_counter()` y promediar varias corridas." |
| 13 | **Ruta `data/` vs carpeta `Data/`** | "Windows no distingue mayúsculas; en Linux/Mac habría que poner `Data/`." |
| 14 | **El informe menciona seaborn para las matrices** | "La versión final migró todas las gráficas a Plotly interactivo (`graficos.py`); seaborn ya no se usa para las matrices." |
| 15 | **¿No hay *data leakage* en la ponderación de KNN?** | "No: el Random Forest que da los pesos se entrena solo con `X_train_esc`; el test solo se transforma con esos pesos." |
| 16 | **Las métricas de clustering no encuentran 22** | "Silueta y DB miden compacidad/separación geométrica. Apple y grapes forman un grupo aislado y los demás cultivos se solapan; eso favorece K=2. Que no coincidan con 22 es un hallazgo, no un error." |

---

## 13. Banco de preguntas y respuestas

### Generales

**¿Qué es sobreajuste y cómo lo detectaron?**
El modelo memoriza detalles y ruido del entrenamiento y no generaliza. Se detecta comparando
la métrica en train vs. test: si train es mucho mejor, hay sobreajuste. Subajuste: ambos malos.

**¿Por qué separar en train y test?**
Para medir el desempeño con datos que el modelo nunca vio, que es lo que pasará en la realidad.
Medir en train premia memorizar.

**¿Qué es `random_state=42`?**
La semilla del generador aleatorio. Fija el barajado del split, el bootstrap del RF, el desempate
entre divisiones iguales en los árboles y la inicialización de K-means → resultados reproducibles.
42 es un valor convencional, no tiene nada especial.

**¿Qué es validación cruzada?**
Dividir el train en *k* partes, entrenar con *k−1* y validar con la restante, rotando *k* veces y
promediando. Da una estimación más estable y permite elegir hiperparámetros sin tocar el test.

**¿Diferencia entre parámetro e hiperparámetro?**
Los parámetros los aprende el modelo (umbrales de un árbol, centroides). Los hiperparámetros los
elige uno antes de entrenar (`max_depth`, `n_estimators`, K, `ccp_alpha`).

**¿Supervisado vs no supervisado?**
Supervisado: los datos traen la respuesta (`label`, `csMPa`) → árboles, RF, KNN. No
supervisado: sin respuesta, se buscan grupos → K-means.

### Datos y preprocesamiento

**¿Cuántos registros y variables tiene cada dataset?** Cultivos: 2200 × 7 + label (22 clases
de 100). Concreto: 721 × 8 + csMPa.

**¿Por qué quitaron `Row ID`?** Es un identificador; no describe la mezcla. Si se dejara, el
modelo podría aprender patrones falsos del orden de las filas.

**¿Por qué estratificar?** Para que train y test tengan la misma proporción de cada cultivo
(80/20 por clase). No se hace en regresión porque *y* es continua.

**¿Qué pasa si aplicas `fit_transform` al test?** El test se escalaría con su propia media y
desviación: train y test quedarían en escalas distintas y además habría fuga de información.
Siempre `fit` con train, `transform` con test.

**¿Por qué Z-score y no Min-Max?** Z-score es menos sensible a valores extremos y es el estándar
para métodos de distancia. Min-Max se muestra en la app como comparación.

**¿Cómo trataron la variable `label` si es texto?** sklearn acepta etiquetas de texto en los
clasificadores y las codifica internamente (`modelo.classes_` en orden alfabético). En la app,
para Mutual Information se usa `LabelEncoder`.

**¿Tipos de variable?** N, P, K: cuantitativas discretas (enteros). temperature, humidity, ph,
rainfall y todas las del concreto excepto age: cuantitativas continuas. age: discreta (días).
label: cualitativa nominal.

### Árboles

**¿Qué es Gini?** 1 − Σpᵢ². Probabilidad de clasificar mal un elemento al azar si se etiqueta
al azar según la distribución del nodo. 0 = puro.

**¿Gini vs entropía?** Ambas miden impureza; la entropía viene de teoría de la información
(−Σ p log₂ p). Dan árboles muy parecidos; Gini es algo más rápido.

**¿Qué hace `min_samples_leaf=5`?** Obliga a que cada hoja tenga al menos 5 muestras; impide
hojas que representan un solo dato (que suelen ser ruido).

**¿Qué es `ccp_alpha`?** Penalización por hoja en la post-poda: R_α(T) = R(T) + α·|hojas|.
Más alpha = árbol más pequeño.

**¿Qué significan los valores de cada caja del árbol dibujado?** Variable y umbral de la
división (con su índice X[i]), clase mayoritaria (o valor promedio en regresión), impureza
(gini/mse) y n (muestras de entrenamiento que llegan ahí). "Sí" = cumple ≤ umbral.

**¿Cuál es la primera división?** La que más reduce la impureza; las variables de las primeras
divisiones suelen ser las más importantes (aquí rainfall y P son las de mayor importancia).

### Random Forest

**¿Bagging vs boosting?** Bagging (RF): árboles independientes en paralelo sobre muestras
bootstrap, se promedian → reduce varianza. Boosting (AdaBoost, Gradient Boosting): árboles en
secuencia, cada uno corrige los errores del anterior → reduce sesgo. No se usó boosting.

**¿Qué es out-of-bag?** El ≈36.8 % de filas que un árbol no vio. Sirve como validación gratis
(`oob_score=True`); no se activó en el proyecto.

**¿Más árboles causan sobreajuste?** No. Más árboles estabilizan el promedio; solo cuesta más
tiempo. Lo que sobreajusta son árboles individuales muy profundos, y el ensamble lo compensa.

**¿Por qué el RF tiene 100 % en train y aun así es bueno?** Cada árbol memoriza su muestra,
pero el promedio de árboles distintos generaliza (99.55 % en test).

**¿Qué es `n_jobs=-1`?** Usar todos los núcleos del procesador para entrenar árboles en paralelo.

### Regresión

**¿Cómo predice un árbol de regresión?** La hoja devuelve el promedio de csMPa de sus muestras
de entrenamiento (mediana si el criterio es `absolute_error`).

**¿MAE vs RMSE?** Ambos en MPa. RMSE eleva al cuadrado los errores, así que pesa más los grandes.
Si RMSE ≫ MAE, hay algunos errores grandes.

**¿Qué significa R² = 0.939?** El modelo explica el 93.9 % de la variabilidad de la resistencia
del concreto en datos no vistos.

**¿Puede R² ser negativo?** Sí, en test, si el modelo predice peor que usar siempre el promedio.

**¿Cuál es la diferencia entre R² y R² ajustado?** El ajustado penaliza el número de variables:
1 − (1 − R²)(n − 1)/(n − p − 1). Siempre ≤ R². Sirve para comparar modelos con distinto
número de variables.

### KNN

**¿Por qué KNN no "entrena"?** Solo almacena los datos; toda la computación se hace al predecir.

**¿Qué pasa con K muy grande?** Subajuste: con K = n siempre predice la clase más frecuente.

**¿Qué distancia se usa?** Euclidiana (Minkowski con p=2), la de sklearn por defecto.

**¿Por qué `uniform` y `distance` dan igual?** Porque K=1: con un solo vecino no hay votos que ponderar.

### Clustering

**¿Qué es un centroide?** El promedio de los puntos de un cluster, variable por variable.

**¿Por qué `n_init=10`?** K-means depende de la inicialización y puede quedar en un óptimo local;
se ejecuta 10 veces y se queda con la menor inercia.

**¿Qué es k-means++?** Una inicialización que elige centroides iniciales alejados entre sí; converge
más rápido y mejor que elegirlos al azar.

**¿Qué es la silueta?** (b − a)/max(a, b) por punto, promediado. Cerca de 1: bien asignado; cerca
de 0: en la frontera; negativo: probablemente mal asignado.

**¿Qué es el algoritmo húngaro?** Un algoritmo de asignación óptima (problema de asignación) que
encuentra el emparejamiento uno a uno de costo mínimo; aquí, el emparejamiento cluster↔cultivo
con más coincidencias.

---

## 14. Hoja de números para memorizar

**Datos**
- Cultivos: 2200 filas, 7 variables, 22 clases × 100, 0 nulos. Split 80/20 → 1760 / 440 (80 y 20 por clase).
- Concreto: 721 filas, 8 variables, 0 nulos, 16 duplicados. Split 80/20 → 576 / 145.

**Clasificación (accuracy en test)**

| Modelo | Test |
|---|---|
| Árbol sin restricción (Gini, 80/20) | 97.95 % (9 errores) |
| Árbol pre-poda depth 4 | 31.82 % (7 hojas → 7 clases) |
| Árbol entropía 70/30 | 98.79 % |
| Árbol post-poda (α = 0.001213, elegido por CV) | 98.18 % |
| RF 9 / 49 / 491 | 99.32 / **99.55** / **99.55** % |
| KNN K=1 (uniform = distance) | 97.95 % |
| KNN variables ponderadas | 98.86 % |
| K-means K=22 vs clases reales | 74.95 % |
| K-means K=2 vs clases reales | 9.09 % |

**Regresión (test)**

| Modelo | R² | R² adj | MAE | RMSE |
|---|---|---|---|---|
| Árbol sin restricción | 0.8926 | 0.8863 | 4.005 | 5.445 |
| Árbol depth 4 | 0.7352 | 0.7196 | 7.009 | 8.550 |
| Árbol MAE 70/30 | 0.7968 | 0.7890 | 5.220 | 7.725 |
| Árbol post-poda (α = 0.2535 por CV, 73 hojas) | 0.8612 | — | — | — |
| RF 9 | 0.9217 | 0.9171 | 3.552 | 4.650 |
| RF 49 | 0.9328 | 0.9289 | 3.244 | 4.306 |
| **RF 491** | **0.9389** | **0.9353** | **3.122** | **4.107** |

**Otros**
- KNN: mejor K por CV = 1 (0.9705); K=3 = 0.9699; K=2 = 0.9597.
- Clustering: silueta y DB → K=2 (separa apple + grapes); silueta 0.417, DB 0.797.
- Bootstrap: 36.4 % repetidos (teórico 36.8 %); coincidencia entre dos árboles 40.2 %.
- Importancia RF clasificación: rainfall > humidity > K > P > N > temperature > ph.
- Importancia RF regresión: age > cement > superplasticizer > water > …
- Gini máximo con 22 clases: 0.9545. Entropía máxima: 4.46 bits.
- Demo con ruido 25 %: sobreajustado 100 / 66.6; bien ajustado 74.0 / 94.5; subajustado 9.1 / 9.1; RF 100 / 99.1.
- Regresión, curva de complejidad: mejor test en depth 12 (R² CV 0.845); bien ajustado depth 7.
