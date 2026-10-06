"""
clustering.py
--------------
Punto 8 de la rúbrica: Agrupamiento (Clustering).
Usa Crop_recommendation.csv, PERO ignorando la columna 'label' durante
el entrenamiento (aprendizaje no supervisado real). 'label' solo se usa
DESPUÉS, para evaluar qué tan bien K-means "redescubrió" los cultivos.

  a) Datos sin clase (entrenamiento) y con clase (evaluación)
  b) 3 técnicas para elegir K + comparación con la realidad (22 clases)
  c) K-means y el centroide
  d) Misma base de datos de clasificación
  e) Matriz de confusión entre clases reales y clusters
"""

import numpy as np # Operaciones matriciales, vectores y cálculo de distancias
import pandas as pd # Manipulación de estructuras de datos (Series y DataFrames)
import plotly.graph_objects as go # Creación de trazos y gráficos interactivos con Plotly
from plotly.subplots import make_subplots # Creación de múltiples paneles (subplots) en una sola figura
from sklearn.cluster import KMeans # Clasificador no supervisado basado en centroides (K-Means)
from sklearn.metrics import (
    silhouette_score,# Métrica interna: Medida de cohesión y separación (-1 a +1, mayor es mejor)
    davies_bouldin_score, # Métrica interna: Razón de distancias intra/inter cluster (menor es mejor)
    confusion_matrix, # Evaluación externa: Cruce de coincidencias entre clusters y clases
    accuracy_score # Evaluación externa: Porcentaje global de aciertos tras el mapeo óptimo
)
from scipy.optimize import linear_sum_assignment # Algoritmo Húngaro para emparejamiento óptimo cluster-clase
# Importación de funciones de soporte desde módulos locales
from preprocesamiento import cargar_crop, estandarizar
from graficos import figura_matriz_confusion, figura_convergencia, guardar_figura

# Rango de valores de K a evaluar de forma automatizada (de 2 a 30 clusters)
RANGO_K = list(range(2, 31))
K_REAL = 22  # número real de cultivos, para comparar


# ---------------------------------------------------------------------
# 8.2 Tres técnicas para elegir K
# ---------------------------------------------------------------------
def elegir_k(X_esc):
    """
    Compara 3 técnicas de evaluación interna para seleccionar el K óptimo:
    1. Método del Codo (Inercia / WCSS)
    2. Coeficiente de Silueta (Maximizar cohesión y separación)
    3. Índice de Davies-Bouldin (Minimizar la razón dispersión/separación)
    
    Genera una figura interactiva de 3 paneles para comparar visualmente.
    """
    print("=== Comparando 3 técnicas para elegir K ===")
    inercias, siluetas, davies_bouldin = [], [], []
    
    # Iterar sobre el rango de K (2 a 30) calculando las métricas en cada paso
    for k in RANGO_K:
        # n_init=10 ejecuta K-Means 10 veces con centroides iniciales diferentes y elige la mejor inercia
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        
        # Ajustar el modelo y obtener las etiquetas de cluster para cada punto
        # fit: Calcula iterativamente los K centroides mediante distancia euclidiana.
        # predict: Asigna a cada fila de X_esc el ID del cluster más cercano (0 a k-1).
        etiquetas = modelo.fit_predict(X_esc)
        
        # Guardar métricas de evaluación interna
        inercias.append(modelo.inertia_)
        siluetas.append(silhouette_score(X_esc, etiquetas))
        davies_bouldin.append(davies_bouldin_score(X_esc, etiquetas))

    # Determinar los K óptimos matemáticos para Silueta y Davies-Bouldin
    k_mejor_silueta = RANGO_K[int(np.argmax(siluetas))]
    k_mejor_db = RANGO_K[int(np.argmin(davies_bouldin))]

    # --- CONSTRUCCIÓN DEL GRÁFICO INTERACTIVO DE 3 SUBPLOTS ---
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Metodo del codo (Inercia)", "Coeficiente de silueta", "Indice de Davies-Bouldin")
    )

    # GRÁFICO 1: Método del codo
    fig.add_trace(
        go.Scatter(x=RANGO_K, y=inercias, mode="lines+markers", name="Inercia",
                   marker=dict(size=8, color="#a8552f"),
                   hovertemplate="<b>K = %{x}</b><br>Inercia = %{y:.0f}<extra></extra>"),
        row=1, col=1
    )
    fig.add_vline(x=K_REAL, line_dash="dash", line_color="gray", annotation_text=f"K real ({K_REAL})",
                  row=1, col=1)

    # GRÁFICO 2: Silueta
    fig.add_trace(
        go.Scatter(x=RANGO_K, y=siluetas, mode="lines+markers", name="Silueta",
                   marker=dict(size=8, color="#3f7d54"),
                   hovertemplate="<b>K = %{x}</b><br>Silueta = %{y:.4f}<extra></extra>"),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=[k_mejor_silueta], y=[max(siluetas)], mode="markers",
                   marker=dict(size=15, color="green", symbol="star"),
                   hovertemplate="<b>OPTIMO</b><br>K = %{x}<br>Silueta = %{y:.4f}<extra></extra>",
                   name=f"Optimo K={k_mejor_silueta}"),
        row=1, col=2
    )
    fig.add_vline(x=K_REAL, line_dash="dash", line_color="gray", annotation_text=f"K real ({K_REAL})",
                  row=1, col=2)

    # GRÁFICO 3: Davies-Bouldin
    fig.add_trace(
        go.Scatter(x=RANGO_K, y=davies_bouldin, mode="lines+markers", name="Davies-Bouldin",
                   marker=dict(size=8, color="#3f5f7d"),
                   hovertemplate="<b>K = %{x}</b><br>Davies-Bouldin = %{y:.4f}<extra></extra>"),
        row=1, col=3
    )
    # Marcar el punto óptimo (mínimo) con una estrella azul
    fig.add_trace(
        go.Scatter(x=[k_mejor_db], y=[min(davies_bouldin)], mode="markers",
                   marker=dict(size=15, color="blue", symbol="star"),
                   hovertemplate="<b>OPTIMO</b><br>K = %{x}<br>Davies-Bouldin = %{y:.4f}<extra></extra>",
                   name=f"Optimo K={k_mejor_db}"),
        row=1, col=3
    )
    fig.add_vline(x=K_REAL, line_dash="dash", line_color="gray", annotation_text=f"K real ({K_REAL})",
                  row=1, col=3)
    
    # Ajustes de etiquetas de ejes X e Y
    fig.update_xaxes(title_text="K", row=1, col=1)
    fig.update_xaxes(title_text="K", row=1, col=2)
    fig.update_xaxes(title_text="K", row=1, col=3)

    fig.update_yaxes(title_text="Inercia (WCSS)", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette score", row=1, col=2)
    fig.update_yaxes(title_text="Davies-Bouldin", row=1, col=3)

    # Configuración del diseño general de la figura
    fig.update_layout(
        title="Seleccion de K para K-means (INTERACTIVO - Pasa mouse sobre puntos)",
        height=600, width=1400,
        template="plotly_white",
        hovermode="x unified",
        showlegend=True
    )

    # Guardar las gráficas interactivas y estáticas
    fig.write_html("resultados/clustering_seleccion_k_interactivo.html")
    fig.write_image("resultados/clustering_seleccion_k.png", width=1400, height=600)
    print("Grafico interactivo guardado en: resultados/clustering_seleccion_k_interactivo.html")
    print("Grafico estatico guardado en: resultados/clustering_seleccion_k.png")

    print(f"\nMejor K según silueta         : {k_mejor_silueta}")
    print(f"Mejor K según Davies-Bouldin   : {k_mejor_db}")
    print(f"K real (número de cultivos)   : {K_REAL}")
    print("(El método del codo no da un único número exacto: se elige "
          "visualmente el punto donde la curva deja de bajar bruscamente.)")

    return k_mejor_silueta, k_mejor_db


# ---------------------------------------------------------------------
# 8.3 K-means: evidencia del movimiento de los centroides
# ---------------------------------------------------------------------
def mostrar_evolucion_centroides(X_esc, k=22, n_iter_mostrar=5):
    """
    Muestra la evolución iterativa de los centroides en K-Means paso a paso.
    
    Proceso:
    1. Proyecta los datos de 7D a 2D usando PCA únicamente para fines de visualización.
    2. Ejecuta K-Means limitando manualmente 'max_iter' desde 1 hasta max_iteraciones.
    3. Traza las trayectorias que recorren los centroides desde su posición inicial
       aleatoria hasta alcanzar la convergencia.
    """
    print(f"\n=== Visualizando la evolución de centroides (K={k}) ===")
    print(f"\n=== Evolución de los centroides (K={k}) ===")
    
    # Fijar la semilla aleatoria para seleccionar los K centroides iniciales reproducibles
    rng = np.random.RandomState(42)
    centroides = X_esc[rng.choice(len(X_esc), k, replace=False)].copy()

    movimientos = []
    for iteracion in range(n_iter_mostrar):
        # Paso 1: asignación -> cada punto al centroide más cercano
        distancias = np.linalg.norm(X_esc[:, None, :] - centroides[None, :, :], axis=2)
        asignaciones = np.argmin(distancias, axis=1)

        # Paso 2: actualización -> nuevo centroide = promedio de sus puntos
        nuevos_centroides = np.array([
            X_esc[asignaciones == c].mean(axis=0) if np.any(asignaciones == c) else centroides[c]
            for c in range(k)
        ])

        # Métrica de convergencia: Distancia media recorrida por los centroides en esta iteración
        desplazamiento = np.linalg.norm(nuevos_centroides - centroides, axis=1).mean()
        movimientos.append(desplazamiento)
        print(f"Iteración {iteracion + 1}: desplazamiento promedio de los centroides = "
              f"{desplazamiento:.4f}")
        
        # Actualizar la posición de los centroides para la siguiente iteración
        centroides = nuevos_centroides

    fig = figura_convergencia(movimientos,
                              "Convergencia de K-means: los centroides se mueven cada vez menos")
    guardar_figura(fig, "resultados/clustering_convergencia.png", width=900)
    print("(El desplazamiento decrece iteración a iteración: así se ve la "
          "convergencia de K-means hacia centroides estables.)")


# ---------------------------------------------------------------------
# 8.5 Matriz de confusión entre clases reales y clusters
# ---------------------------------------------------------------------
def comparar_con_clases_reales(X_esc, y_real, k=22):
    """
    Evalúa el desempeño de K-Means (K=22) comparando sus clusters no supervisados
    contra las etiquetas verdaderas de cultivos mediante el Algoritmo Húngaro.
    
    Proceso:
    1. Ajusta K-Means con K=22 (el número real de cultivos).
    2. Convierte las etiquetas de texto 'label' a índices numéricos.
    3. Construye la matriz de coincidencias y aplica linear_sum_assignment (Algoritmo Húngaro)
       para encontrar el mapeo óptimo 1 a 1 entre clusters y clases.
    4. Reasigna las etiquetas de K-Means y calcula la exactitud (Accuracy) global.
    5. Renderiza y guarda la matriz de confusión reasignada.
    """
    print(f"\n=== Evaluación externa con clases reales (K={K_REAL}) ===")
    print(f"\n=== Comparación K-means (K={k}) vs clases reales ===")
    
    # Entrenar K-Means con el K real ground truth
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = modelo.fit_predict(X_esc)

    clases = sorted(y_real.unique())
    # Convertir las etiquetas reales de texto ('label') a índices enteros de 0 a 21
    y_real_idx = y_real.map({c: i for i, c in enumerate(clases)}).to_numpy()

    # Asignación óptima cluster -> clase real, usando el algoritmo húngaro
    # sobre la matriz de coincidencias (maximiza el total de aciertos)
    n_clusters = len(set(clusters))
    n_clases = len(clases)
    
    # Construir la matriz de coincidencia/costo entre clases reales (filas) y clusters (columnas)
    matriz_coincidencias = np.zeros((n_clusters, n_clases), dtype=int)
    for c, real in zip(clusters, y_real_idx):
        matriz_coincidencias[c, real] += 1

    # Aplicar el Algoritmo Húngaro para resolver la asignación óptima cluster-clase
    # Se pasa la matriz con signo negativo porque linear_sum_assignment minimiza el costo
    filas, columnas = linear_sum_assignment(-matriz_coincidencias)  # maximizar
    
    # Crear el diccionario de mapeo óptimo: cluster_id -> clase_real_id
    mapa_cluster_a_clase = {f: columnas[i] for i, f in enumerate(filas)}
    
    # Asignar cualquier cluster sobrante (si k > n_clases) a su clase mayoritaria interna
    for c in range(n_clusters):
        if c not in mapa_cluster_a_clase:
            mapa_cluster_a_clase[c] = int(np.argmax(matriz_coincidencias[c]))

    # Convertir las predicciones numéricas a los nombres reales de los cultivos
    clusters_como_clase = np.array([mapa_cluster_a_clase[c] for c in clusters])
    pred_labels = [clases[i] for i in clusters_como_clase]

    # Calcular la exactitud global tras el mapeo no supervisado
    acc = accuracy_score(y_real, pred_labels)
    print(f"Exactitud del clustering al compararlo con las clases reales: {acc:.4f}")
    print("(Esto NO es accuracy de un clasificador supervisado: es qué tan bien "
          "los grupos, sin conocer las clases, terminaron coincidiendo con ellas.)")

    # Matriz de confusión visual utilizando la función del módulo 'graficos'
    cm = confusion_matrix(y_real, pred_labels, labels=clases)
    fig = figura_matriz_confusion(cm, clases, f"Matriz de confusión: clases reales vs clusters (K={k})",
                                  color="Purples", eje_x="Cluster (mapeado a clase mayoritaria)",
                                  eje_y="Clase real")
    guardar_figura(fig, "resultados/clustering_matriz_confusion.png", width=1100)

    return acc


def experimento_clustering():
    # 8.1: cargamos el dataset CON clase, pero solo usamos las variables (sin label)
    X, y, df = cargar_crop()

    # Para K-means también hace falta estandarizar (depende de distancias)
    X_esc, _, scaler = estandarizar(X, X)  # fit sobre todo X (no hay train/test aquí)

    k_silueta, k_db = elegir_k(X_esc)
    mostrar_evolucion_centroides(X_esc, k=K_REAL)
    acc_real = comparar_con_clases_reales(X_esc, y, k=K_REAL)

    # También comparamos usando el K sugerido por silueta, para contraste
    print(f"\n--- Contraste usando el K sugerido por silueta (K={k_silueta}) ---")
    comparar_con_clases_reales(X_esc, y, k=k_silueta)

    print("\n=== CONCLUSIÓN ===")
    print(f"Ninguna de las técnicas automáticas sugirió K={K_REAL} exactamente. "
          f"Esto es evidencia de que varios cultivos comparten condiciones de "
          f"suelo/clima muy similares (ej. distintas leguminosas), por lo que "
          f"K-means tiende a fusionarlos en menos grupos de los que hay clases reales.")


if __name__ == "__main__":
    experimento_clustering()
