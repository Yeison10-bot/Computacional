"""
arboles_decision.py
--------------------
Punto 4 de la rúbrica: Árbol de decisión (clasificación, dataset cultivos).

  a) 3 modelos distintos -> 3 formas de mandarle la db para entrenamiento
  b) Matriz de confusión de cada modelo
  c) Podas (pre-poda y post-poda) para obtener modelos diferentes

No se aplica escalado: los árboles son invariantes a la escala de las
variables (dividen comparando valor vs umbral, variable por variable).
"""
# Medición de tiempos de ejecución (para evaluar rendimiento computacional)
import time
# Manejo de estructuras de datos y operaciones numéricas
import numpy as np
# DecisionTreeClassifier: Algoritmo de clasificación basado en árboles de decisión (CART).
# plot_tree: Herramienta para renderizar la estructura visual del árbol generado
from sklearn.tree import DecisionTreeClassifier, plot_tree
# Métricas de evaluación del modelo:
# - accuracy_score: Porcentaje global de predicciones correctas.
# - confusion_matrix: Matriz NxN de aciertos y desvíos entre clases.
# - classification_report: Reporte detallado de precisión, recall y F1-score.
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# cross_val_score: validación cruzada, para elegir el ccp_alpha sin mirar el test
from sklearn.model_selection import cross_val_score
# Importación de funciones propias desde el módulo local preprocesamiento.py:
# - cargar_crop: Carga y separa la base de datos de cultivos en X e y.
# - dividir_datos: Genera las particiones de Entrenamiento (Train) y Prueba (Test).
from preprocesamiento import cargar_crop, dividir_datos
from graficos import figura_matriz_confusion, figura_curva_poda, guardar_figura


def entrenar_y_evaluar(modelo, X_train, X_test, y_train, y_test, nombre):
    """Entrena un modelo, mide tiempo y calcula métricas train/test."""
    # --------------------------------------------------------------------------
    # 1. ENTRENAMIENTO Y MEDICIÓN DE TIEMPO
    # --------------------------------------------------------------------------
    inicio = time.time()
    modelo.fit(X_train, y_train) # El árbol construye las reglas de decisión dividiendo los datos
    tiempo_entrenamiento = time.time() - inicio
    # --------------------------------------------------------------------------
    # 2. GENERACIÓN DE PREDICCIONES
    # --------------------------------------------------------------------------
    pred_train = modelo.predict(X_train) # Predicciones sobre datos memorizados/conocidos
    pred_test = modelo.predict(X_test) # Predicciones sobre datos nuevos/no vistos

    # --------------------------------------------------------------------------
    # 3. CÁLCULO DE MÉTRICAS (EXACTITUD / ACCURACY)
    # accuracy_score = (Predicciones Correctas) / (Total de Muestras)
    # --------------------------------------------------------------------------
    acc_train = accuracy_score(y_train, pred_train) # Mide el rendimiento en entrenamiento
    acc_test = accuracy_score(y_test, pred_test) # Mide la capacidad de generalización
    # Imprimir reporte básico en consola
    print(f"\n--- {nombre} ---")
    print(f"Tiempo de entrenamiento : {tiempo_entrenamiento:.4f} s")
    print(f"Accuracy en TRAIN       : {acc_train:.4f}")
    print(f"Accuracy en TEST        : {acc_test:.4f}")
    # --------------------------------------------------------------------------
    # 4. AUDITORÍA DE SOBREAJUSTE (OVERFITTING) Y SUBAJUSTE (UNDERFITTING)
    # --------------------------------------------------------------------------
    # Si la exactitud en Train supera a la de Test por más de 8 puntos porcentuales (0.08),
    # indica que el árbol memorizó ruido de entrenamiento y pierde capacidad de generalizar.
    # Una brecha pequeña NO basta para decir "buen ajuste": si el accuracy es bajo en
    # ambos conjuntos (< 0.60), el modelo es demasiado simple para las 22 clases.
    diferencia = acc_train - acc_test
    if diferencia > 0.08:
        print(f"[!] Posible sobreajuste (diferencia train-test = {diferencia:.4f})")
    elif acc_test < 0.60:
        print(f"[!] Posible subajuste (accuracy test = {acc_test:.4f}, bajo en train y test; "
              f"diferencia train-test = {diferencia:.4f})")
    else:
        print(f"[OK] Buen ajuste (diferencia train-test = {diferencia:.4f})")
    # --------------------------------------------------------------------------
    # 5. RETORNO DE RESULTADOS
    # Devuelve un diccionario con las métricas, el modelo ajustado y las predicciones.
    # --------------------------------------------------------------------------
    return {
        "nombre": nombre,
        "modelo": modelo,
        "tiempo": tiempo_entrenamiento,
        "acc_train": acc_train,
        "acc_test": acc_test,
        "pred_test": pred_test,
    }


def graficar_matriz_confusion(y_test, pred_test, nombre, clases, ruta_salida):
    """Genera y guarda la matriz de confusión (PNG + HTML interactivo)."""
    # Calcula la matriz cuadrada NxN (donde N es el número de clases).
    # labels=clases asegura el orden de las etiquetas en ejes X e Y.
    cm = confusion_matrix(y_test, pred_test, labels=clases)
    # Aciertos (diagonal) en naranja, errores en rojo y celdas en cero vacías.
    fig = figura_matriz_confusion(cm, clases, f"Matriz de confusión - {nombre}", color="Oranges")
    guardar_figura(fig, ruta_salida, width=1100)


def experimento_arboles():
    X, y, df = cargar_crop()
    clases = sorted(y.unique())

    # Partición base (80/20, estratificada para mantener balance de clases)
    X_train, X_test, y_train, y_test = dividir_datos(X, y, test_size=0.2, estratificar=True)

    resultados = []

    # ---------- Modelo 1: árbol sin restricciones (referencia / overfit) ----------
    m1 = DecisionTreeClassifier(random_state=42)
    r1 = entrenar_y_evaluar(m1, X_train, X_test, y_train, y_test,
                             "Modelo 1: Sin restricciones (80/20)")
    resultados.append(r1)

    # ---------- Modelo 2: pre-poda (profundidad limitada) ----------
    m2 = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
    r2 = entrenar_y_evaluar(m2, X_train, X_test, y_train, y_test,
                             "Modelo 2: Pre-poda (max_depth=4, min_samples_leaf=5)")
    resultados.append(r2)

    # ---------- Modelo 3: distinta partición de datos (70/30) ----------
    X_train3, X_test3, y_train3, y_test3 = dividir_datos(X, y, test_size=0.3, estratificar=True)
    m3 = DecisionTreeClassifier(criterion="entropy", random_state=42)
    r3 = entrenar_y_evaluar(m3, X_train3, X_test3, y_train3, y_test3,
                             "Modelo 3: Partición 70/30 + criterio entropía")
    resultados.append(r3)

    # Matrices de confusión de los 3
    graficar_matriz_confusion(y_test, r1["pred_test"], r1["nombre"], clases,
                               "resultados/cm_arbol_modelo1.png")
    graficar_matriz_confusion(y_test, r2["pred_test"], r2["nombre"], clases,
                               "resultados/cm_arbol_modelo2.png")
    graficar_matriz_confusion(y_test3, r3["pred_test"], r3["nombre"], clases,
                               "resultados/cm_arbol_modelo3.png")

    # ---------- Post-poda: cost complexity pruning (ccp_alpha) ----------
    print("\n=== Post-poda (cost-complexity pruning) sobre el Modelo 1 ===")
    # 1. Se crea un árbol completo (sin restricciones) para calcular la ruta de poda
    arbol_completo = DecisionTreeClassifier(random_state=42)
    # cost_complexity_pruning_path: Devuelve los valores de alpha (niveles de tijera)
    # donde una rama deja de aportar suficiente precisión y debe recortarse.
    path = arbol_completo.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    # 2. Se selecciona una muestra de máximo 25 valores de alpha para agilizar la prueba
    alphas_muestra = ccp_alphas[:: max(1, len(ccp_alphas) // 25)]
    acc_train_list, acc_test_list, acc_cv_list = [], [], []
    # 3. Bucle de prueba: Se entrena un nuevo árbol por cada nivel de alpha en la muestra
    for alpha in alphas_muestra:
        arbol = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
        # Validación cruzada (5 pliegues estratificados) SOLO con el set de entrenamiento:
        # es el criterio para elegir alpha, así el test no participa en la elección.
        acc_cv_list.append(cross_val_score(arbol, X_train, y_train, cv=5, scoring="accuracy").mean())
        arbol.fit(X_train, y_train)
        # Registra el desempeño en train y test para cada nivel de poda (solo para graficar)
        acc_train_list.append(accuracy_score(y_train, arbol.predict(X_train)))
        acc_test_list.append(accuracy_score(y_test, arbol.predict(X_test)))

    # 4. Selección del mejor ccp_alpha: el que maximiza el accuracy de VALIDACIÓN CRUZADA.
    # Elegirlo con el test sería usar el test para ajustar un hiperparámetro (resultado
    # optimista); el test se reserva para la evaluación final del alpha ya elegido.
    mejor_idx = int(np.argmax(acc_cv_list))
    mejor_alpha = alphas_muestra[mejor_idx]
    fig = figura_curva_poda(alphas_muestra, acc_train_list, acc_test_list, "Accuracy",
                            "Efecto de la post-poda en accuracy (train vs test) - Clasificación",
                            metrica_cv=acc_cv_list)
    guardar_figura(fig, "resultados/poda_ccp_alpha.png")

    print(f"\nMejor ccp_alpha (elegido por validación cruzada 5-fold): {mejor_alpha:.6f}")
    print(f"Accuracy CV    con ese alpha: {acc_cv_list[mejor_idx]:.4f}")
    print(f"Accuracy train con ese alpha: {acc_train_list[mejor_idx]:.4f}")
    print(f"Accuracy test  con ese alpha: {acc_test_list[mejor_idx]:.4f}")

    # Tabla resumen final
    print("\n=== RESUMEN COMPARATIVO ===")
    print(f"{'Modelo':45s} {'Tiempo(s)':>10s} {'Acc Train':>10s} {'Acc Test':>10s}")
    for r in resultados:
        print(f"{r['nombre']:45s} {r['tiempo']:>10.4f} {r['acc_train']:>10.4f} {r['acc_test']:>10.4f}")
    print(f"{'Árbol podado (ccp_alpha por CV)':45s} {'-':>10s} "
          f"{acc_train_list[mejor_idx]:>10.4f} {acc_test_list[mejor_idx]:>10.4f}")

    return resultados


if __name__ == "__main__":
    experimento_arboles()
