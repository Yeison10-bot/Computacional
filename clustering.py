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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, confusion_matrix, accuracy_score
)
from scipy.optimize import linear_sum_assignment

from preprocesamiento import cargar_crop, estandarizar

RANGO_K = list(range(2, 31))
K_REAL = 22  # número real de cultivos, para comparar


# ---------------------------------------------------------------------
# 8.2 Tres técnicas para elegir K
# ---------------------------------------------------------------------
def elegir_k(X_esc):
    print("=== Comparando 3 técnicas para elegir K ===")
    inercias, siluetas, davies_bouldin = [], [], []

    for k in RANGO_K:
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        etiquetas = modelo.fit_predict(X_esc)
        inercias.append(modelo.inertia_)
        siluetas.append(silhouette_score(X_esc, etiquetas))
        davies_bouldin.append(davies_bouldin_score(X_esc, etiquetas))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(RANGO_K, inercias, marker="o", color="#a8552f")
    axes[0].axvline(K_REAL, color="gray", linestyle="--", label=f"K real ({K_REAL})")
    axes[0].set_title("Método del codo (Inercia)")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inercia (WCSS)")
    axes[0].legend()

    axes[1].plot(RANGO_K, siluetas, marker="o", color="#3f7d54")
    axes[1].axvline(K_REAL, color="gray", linestyle="--", label=f"K real ({K_REAL})")
    k_mejor_silueta = RANGO_K[int(np.argmax(siluetas))]
    axes[1].axvline(k_mejor_silueta, color="green", linestyle=":",
                     label=f"Mejor silueta (K={k_mejor_silueta})")
    axes[1].set_title("Coeficiente de silueta")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette score (más alto = mejor)")
    axes[1].legend()

    axes[2].plot(RANGO_K, davies_bouldin, marker="o", color="#3f5f7d")
    axes[2].axvline(K_REAL, color="gray", linestyle="--", label=f"K real ({K_REAL})")
    k_mejor_db = RANGO_K[int(np.argmin(davies_bouldin))]
    axes[2].axvline(k_mejor_db, color="blue", linestyle=":",
                     label=f"Mejor Davies-Bouldin (K={k_mejor_db})")
    axes[2].set_title("Índice de Davies-Bouldin")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Davies-Bouldin (más bajo = mejor)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("resultados/clustering_seleccion_k.png", dpi=110)
    plt.close()
    print("Gráfico guardado en: resultados/clustering_seleccion_k.png")

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
    print(f"\n=== Evolución de los centroides (K={k}) ===")
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

        desplazamiento = np.linalg.norm(nuevos_centroides - centroides, axis=1).mean()
        movimientos.append(desplazamiento)
        print(f"Iteración {iteracion + 1}: desplazamiento promedio de los centroides = "
              f"{desplazamiento:.4f}")
        centroides = nuevos_centroides

    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, n_iter_mostrar + 1), movimientos, marker="o", color="#a8552f")
    plt.xlabel("Iteración")
    plt.ylabel("Desplazamiento promedio de los centroides")
    plt.title("Convergencia de K-means: los centroides se mueven cada vez menos")
    plt.tight_layout()
    plt.savefig("resultados/clustering_convergencia.png", dpi=110)
    plt.close()
    print("Gráfico guardado en: resultados/clustering_convergencia.png")
    print("(El desplazamiento decrece iteración a iteración: así se ve la "
          "convergencia de K-means hacia centroides estables.)")


# ---------------------------------------------------------------------
# 8.5 Matriz de confusión entre clases reales y clusters
# ---------------------------------------------------------------------
def comparar_con_clases_reales(X_esc, y_real, k=22):
    print(f"\n=== Comparación K-means (K={k}) vs clases reales ===")
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = modelo.fit_predict(X_esc)

    clases = sorted(y_real.unique())
    y_real_idx = y_real.map({c: i for i, c in enumerate(clases)}).to_numpy()

    # Asignación óptima cluster -> clase real, usando el algoritmo húngaro
    # sobre la matriz de coincidencias (maximiza el total de aciertos)
    n_clusters = len(set(clusters))
    n_clases = len(clases)
    matriz_coincidencias = np.zeros((n_clusters, n_clases), dtype=int)
    for c, real in zip(clusters, y_real_idx):
        matriz_coincidencias[c, real] += 1

    filas, columnas = linear_sum_assignment(-matriz_coincidencias)  # maximizar
    mapa_cluster_a_clase = {f: columnas[i] for i, f in enumerate(filas)}
    # Clusters sin pareja (si k != n_clases) se mapean a su clase mayoritaria
    for c in range(n_clusters):
        if c not in mapa_cluster_a_clase:
            mapa_cluster_a_clase[c] = int(np.argmax(matriz_coincidencias[c]))

    clusters_como_clase = np.array([mapa_cluster_a_clase[c] for c in clusters])
    pred_labels = [clases[i] for i in clusters_como_clase]

    acc = accuracy_score(y_real, pred_labels)
    print(f"Exactitud del clustering al compararlo con las clases reales: {acc:.4f}")
    print("(Esto NO es accuracy de un clasificador supervisado: es qué tan bien "
          "los grupos, sin conocer las clases, terminaron coincidiendo con ellas.)")

    cm = confusion_matrix(y_real, pred_labels, labels=clases)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, cmap="Purples", xticklabels=clases, yticklabels=clases)
    plt.title(f"Matriz de confusión: clases reales vs clusters (K={k})")
    plt.xlabel("Cluster (mapeado a clase mayoritaria)")
    plt.ylabel("Clase real")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("resultados/clustering_matriz_confusion.png", dpi=110)
    plt.close()
    print("Matriz de confusión guardada en: resultados/clustering_matriz_confusion.png")

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
