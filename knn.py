"""
knn.py
-------
Punto 7 de la rúbrica: Aprendizaje basado en instancias (KNN).
Usa Crop_recommendation.csv (clasificación).

  a) KNN: K vecinos más cercanos
  b) Cómo seleccionar el K
  c) No utiliza un modelo (lazy learning)
  d) Dar más peso a ciertas variables para evitar empates

IMPORTANTE: KNN depende de distancias -> se usan los datos ESTANDARIZADOS
(a diferencia de árboles / Random Forest).
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

from preprocesamiento import cargar_crop, dividir_datos, estandarizar

VALORES_K = list(range(1, 26))  # 1 a 25


# ---------------------------------------------------------------------
# 7.2 Selección del valor óptimo de K
# ---------------------------------------------------------------------
def seleccionar_mejor_k(X_train_esc, y_train):
    """
    Prueba varios K con validación cruzada (5-fold) sobre el set de
    entrenamiento y grafica K vs accuracy para elegir el mejor.
    """
    print("=== Selección de K mediante validación cruzada (5-fold) ===")
    accuracies = []
    for k in VALORES_K:
        modelo = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(modelo, X_train_esc, y_train, cv=5, scoring="accuracy")
        accuracies.append(scores.mean())
        print(f"K={k:2d} -> accuracy promedio (CV) = {scores.mean():.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(VALORES_K, accuracies, marker="o", color="#a8552f")
    plt.xlabel("K (número de vecinos)")
    plt.ylabel("Accuracy promedio (validación cruzada)")
    plt.title("Selección de K en KNN")
    plt.xticks(VALORES_K)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("resultados/knn_seleccion_k.png", dpi=110)
    plt.close()
    print("\nGráfico guardado en: resultados/knn_seleccion_k.png")

    # Preferir K impar entre los mejores candidatos, para minimizar empates
    mejor_acc = max(accuracies)
    candidatos = [k for k, a in zip(VALORES_K, accuracies) if a >= mejor_acc - 0.002]
    candidatos_impares = [k for k in candidatos if k % 2 == 1]
    mejor_k = min(candidatos_impares) if candidatos_impares else min(candidatos)

    print(f"\nMejor K encontrado (preferido impar): {mejor_k} "
          f"(accuracy CV = {accuracies[VALORES_K.index(mejor_k)]:.4f})")
    return mejor_k, accuracies


# ---------------------------------------------------------------------
# 7.1 / 7.3: Entrenar y evaluar KNN, mostrando que "no hay modelo"
# ---------------------------------------------------------------------
def evaluar_knn(k, X_train_esc, X_test_esc, y_train, y_test, clases, weights="uniform"):
    modelo = KNeighborsClassifier(n_neighbors=k, weights=weights)

    # "Entrenar" un KNN es solo guardar los datos: es prácticamente instantáneo
    inicio = time.time()
    modelo.fit(X_train_esc, y_train)
    tiempo_fit = time.time() - inicio

    # Predecir SÍ tiene costo real: recorre todo el set guardado
    inicio = time.time()
    pred_test = modelo.predict(X_test_esc)
    tiempo_predict = time.time() - inicio

    acc = accuracy_score(y_test, pred_test)

    print(f"\n--- KNN (K={k}, weights='{weights}') ---")
    print(f"Tiempo de 'entrenamiento' (solo guardar datos): {tiempo_fit:.5f} s")
    print(f"Tiempo de predicción sobre {len(X_test_esc)} muestras: {tiempo_predict:.5f} s")
    print(f"Accuracy en test: {acc:.4f}")

    cm = confusion_matrix(y_test, pred_test, labels=clases)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, cmap="Greens", xticklabels=clases, yticklabels=clases)
    plt.title(f"Matriz de confusión - KNN (K={k}, weights='{weights}')")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    ruta = f"resultados/cm_knn_k{k}_{weights}.png"
    plt.savefig(ruta, dpi=110)
    plt.close()
    print(f"Matriz de confusión guardada en: {ruta}")

    return modelo, acc


# ---------------------------------------------------------------------
# 7.4 Ponderación de variables (pesos por variable / por distancia)
# ---------------------------------------------------------------------
def comparar_ponderacion(mejor_k, X_train_esc, X_test_esc, y_train, y_test, clases,
                          nombres_columnas):
    print("\n=== Comparación de estrategias de ponderación ===")
    _, acc_uniforme = evaluar_knn(mejor_k, X_train_esc, X_test_esc, y_train, y_test,
                                   clases, weights="uniform")
    _, acc_distancia = evaluar_knn(mejor_k, X_train_esc, X_test_esc, y_train, y_test,
                                    clases, weights="distance")

    # Ponderación manual: usamos la importancia de variables de un Random
    # Forest como "proxy" de qué tan discriminante es cada una, y las usamos
    # como pesos multiplicativos sobre las variables ya estandarizadas.
    from sklearn.ensemble import RandomForestClassifier
    rf_ref = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_ref.fit(X_train_esc, y_train)
    importancias = rf_ref.feature_importances_
    pesos = importancias / importancias.mean()  # normalizado alrededor de 1.0

    print("\nPesos por variable (según importancia relativa):")
    for col, peso in zip(nombres_columnas, pesos):
        print(f"  {col:12s}: {peso:.3f}")

    X_train_ponderado = X_train_esc * pesos
    X_test_ponderado = X_test_esc * pesos

    modelo_pond = KNeighborsClassifier(n_neighbors=mejor_k)
    modelo_pond.fit(X_train_ponderado, y_train)
    pred_pond = modelo_pond.predict(X_test_ponderado)
    acc_ponderado = accuracy_score(y_test, pred_pond)
    print(f"\nAccuracy con variables ponderadas manualmente: {acc_ponderado:.4f}")

    print("\n=== RESUMEN ESTRATEGIAS DE PONDERACIÓN (K={}) ===".format(mejor_k))
    print(f"{'Estrategia':30s} {'Accuracy':>10s}")
    print(f"{'Uniforme (todos igual)':30s} {acc_uniforme:>10.4f}")
    print(f"{'Por distancia':30s} {acc_distancia:>10.4f}")
    print(f"{'Variables ponderadas':30s} {acc_ponderado:>10.4f}")


def experimento_knn():
    X, y, df = cargar_crop()
    X_train, X_test, y_train, y_test = dividir_datos(X, y, test_size=0.2, estratificar=True)

    # KNN necesita estandarización (a diferencia de árboles / RF)
    X_train_esc, X_test_esc, scaler = estandarizar(X_train, X_test)

    clases = sorted(y.unique())

    mejor_k, accuracies = seleccionar_mejor_k(X_train_esc, y_train)

    print(f"\n=== Evaluación final con K óptimo = {mejor_k} ===")
    evaluar_knn(mejor_k, X_train_esc, X_test_esc, y_train, y_test, clases)

    comparar_ponderacion(mejor_k, X_train_esc, X_test_esc, y_train, y_test,
                          clases, X.columns.tolist())

    return mejor_k


if __name__ == "__main__":
    experimento_knn()
