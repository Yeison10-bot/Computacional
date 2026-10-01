"""
random_forest.py
------------------
Punto 5 de la rúbrica: Random Forest (clasificación, dataset cultivos).

  a) 3 modelos con distinta cantidad de árboles: 9, 49, 491 (impares -> evitar empates)
  b) Regla de desempate si llegara a haber empate en la votación
  c) Temporizador y distribución de tiempos de entrenamiento
  d) Explicación práctica de bootstrap (subconjuntos aleatorios de datos)

No se aplica escalado (igual que en árboles): Random Forest es invariante
a la escala de las variables.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

from preprocesamiento import cargar_crop, dividir_datos

N_ARBOLES = [9, 49, 491]  # impares, para minimizar empates en la votación


def entrenar_bosque(n_arboles, X_train, X_test, y_train, y_test):
    """Entrena un RandomForest con n_arboles y mide tiempo/exactitud."""
    modelo = RandomForestClassifier(
        n_estimators=n_arboles,
        random_state=42,
        n_jobs=-1,       # usa todos los núcleos disponibles
        bootstrap=True,  # cada árbol se entrena con una muestra aleatoria CON reemplazo
    )
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo = time.time() - inicio

    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)
    acc_train = accuracy_score(y_train, pred_train)
    acc_test = accuracy_score(y_test, pred_test)

    print(f"\n--- Random Forest con {n_arboles} árboles ---")
    print(f"Tiempo de entrenamiento : {tiempo:.4f} s")
    print(f"Accuracy en TRAIN       : {acc_train:.4f}")
    print(f"Accuracy en TEST        : {acc_test:.4f}")

    return {
        "n_arboles": n_arboles,
        "modelo": modelo,
        "tiempo": tiempo,
        "acc_train": acc_train,
        "acc_test": acc_test,
        "pred_test": pred_test,
    }


def predecir_con_regla_desempate(modelo, X_muestra):
    """
    Muestra cómo resolver un empate en la votación de los árboles.

    Random Forest en scikit-learn en realidad promedia las PROBABILIDADES
    de cada árbol (no cuenta votos discretos "duros"), lo cual ya evita casi
    todos los empates. Aun así, aquí se simula explícitamente una votación
    dura árbol por árbol para mostrar qué pasaría y cómo se resuelve un empate.
    """
    # Voto de cada árbol individual para la primera muestra de X_muestra
    fila = X_muestra.iloc[[0]].to_numpy()
    votos = [arbol.predict(fila)[0] for arbol in modelo.estimators_]
    conteo = Counter(votos)
    max_votos = max(conteo.values())
    empatados = [clase for clase, v in conteo.items() if v == max_votos]

    print(f"\nVotos de los primeros árboles: {conteo.most_common(5)}")
    if len(empatados) > 1:
        print(f"⚠ Empate detectado entre: {empatados}")
        # Regla de desempate: usar la probabilidad promedio del bosque completo
        # (predict_proba) en vez del conteo de votos duros
        proba = modelo.predict_proba(fila)[0]
        clases = modelo.classes_
        proba_empatados = {c: proba[list(clases).index(c)] for c in empatados}
        ganador = max(proba_empatados, key=proba_empatados.get)
        print(f"Regla de desempate (mayor probabilidad promedio): gana '{ganador}'")
    else:
        print(f"No hay empate. Clase ganadora por votación: '{conteo.most_common(1)[0][0]}'")


def graficar_tiempos_y_precision(resultados):
    """Grafica cantidad de árboles vs tiempo y vs accuracy."""
    n_arboles = [r["n_arboles"] for r in resultados]
    tiempos = [r["tiempo"] for r in resultados]
    acc_test = [r["acc_test"] for r in resultados]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    color1 = "tab:orange"
    ax1.set_xlabel("Número de árboles")
    ax1.set_ylabel("Tiempo de entrenamiento (s)", color=color1)
    ax1.plot(n_arboles, tiempos, marker="o", color=color1, label="Tiempo")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.set_ylabel("Accuracy en test", color=color2)
    ax2.plot(n_arboles, acc_test, marker="s", color=color2, label="Accuracy")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Random Forest: árboles vs tiempo vs accuracy")
    fig.tight_layout()
    plt.savefig("resultados/rf_tiempo_vs_precision.png", dpi=110)
    plt.close()
    print("\nGráfico guardado en: resultados/rf_tiempo_vs_precision.png")


def graficar_matriz_confusion(y_test, pred_test, nombre, clases, ruta_salida):
    cm = confusion_matrix(y_test, pred_test, labels=clases)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=clases, yticklabels=clases)
    plt.title(f"Matriz de confusión - {nombre}")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=110)
    plt.close()
    print(f"Matriz de confusión guardada en: {ruta_salida}")


def explicar_bootstrap(X_train):
    """
    Muestra de forma concreta que cada árbol del bosque ve una muestra
    distinta de los datos (bootstrap: muestreo aleatorio CON reemplazo).
    """
    print("\n=== Evidencia de bootstrap (subconjuntos aleatorios) ===")
    n = len(X_train)
    rng = np.random.RandomState(0)
    muestra_arbol_1 = rng.choice(X_train.index, size=n, replace=True)
    muestra_arbol_2 = rng.choice(X_train.index, size=n, replace=True)

    repetidos_1 = n - len(set(muestra_arbol_1))
    interseccion = len(set(muestra_arbol_1) & set(muestra_arbol_2))

    print(f"Tamaño del set de entrenamiento: {n}")
    print(f"Índices repetidos dentro de la muestra del árbol 1: {repetidos_1} "
          f"({repetidos_1/n*100:.1f}% de duplicados, típico del bootstrap)")
    print(f"Coincidencia entre la muestra del árbol 1 y la del árbol 2: "
          f"{interseccion}/{n} ({interseccion/n*100:.1f}%) -> confirma que cada árbol "
          f"entrena con un subconjunto distinto")


def experimento_random_forest():
    X, y, df = cargar_crop()
    clases = sorted(y.unique())
    X_train, X_test, y_train, y_test = dividir_datos(X, y, test_size=0.2, estratificar=True)

    resultados = []
    for n in N_ARBOLES:
        r = entrenar_bosque(n, X_train, X_test, y_train, y_test)
        resultados.append(r)

    # Matrices de confusión de los 3 bosques
    for r in resultados:
        graficar_matriz_confusion(
            y_test, r["pred_test"], f"Random Forest ({r['n_arboles']} árboles)",
            clases, f"resultados/cm_rf_{r['n_arboles']}arboles.png"
        )

    graficar_tiempos_y_precision(resultados)

    # Demostrar la regla de desempate con el bosque más grande
    modelo_grande = resultados[-1]["modelo"]
    predecir_con_regla_desempate(modelo_grande, X_test)

    # Evidencia de bootstrap
    explicar_bootstrap(X_train)

    # Resumen final
    print("\n=== RESUMEN COMPARATIVO RANDOM FOREST ===")
    print(f"{'N° árboles':>10s} {'Tiempo(s)':>10s} {'Acc Train':>10s} {'Acc Test':>10s}")
    for r in resultados:
        print(f"{r['n_arboles']:>10d} {r['tiempo']:>10.4f} {r['acc_train']:>10.4f} {r['acc_test']:>10.4f}")

    return resultados


if __name__ == "__main__":
    experimento_random_forest()
