"""
regresion.py
-------------
Punto 6 de la rúbrica: "Hacer lo mismo, pero para regresión"
Usa concreto.csv (train.csv) -> variable objetivo continua: csMPa

Replica la misma estructura de arboles_decision.py y random_forest.py,
pero con métricas de regresión (MAE, RMSE, R2) en vez de accuracy /
matriz de confusión, y gráficas de "real vs predicho" como equivalente
visual de la matriz de confusión.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocesamiento import cargar_concreto, dividir_datos

N_ARBOLES = [9, 49, 491]


def calcular_metricas(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    return mae, rmse, r2


def entrenar_y_evaluar(modelo, X_train, X_test, y_train, y_test, nombre):
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo = time.time() - inicio

    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    mae_tr, rmse_tr, r2_tr = calcular_metricas(y_train, pred_train)
    mae_te, rmse_te, r2_te = calcular_metricas(y_test, pred_test)

    print(f"\n--- {nombre} ---")
    print(f"Tiempo de entrenamiento : {tiempo:.4f} s")
    print(f"TRAIN -> MAE: {mae_tr:.3f} | RMSE: {rmse_tr:.3f} | R2: {r2_tr:.4f}")
    print(f"TEST  -> MAE: {mae_te:.3f} | RMSE: {rmse_te:.3f} | R2: {r2_te:.4f}")

    diff_r2 = r2_tr - r2_te
    if diff_r2 > 0.15:
        print(f"⚠ Posible sobreajuste (diferencia R2 train-test = {diff_r2:.4f})")
    else:
        print(f"✔ Buen ajuste (diferencia R2 train-test = {diff_r2:.4f})")

    return {
        "nombre": nombre, "modelo": modelo, "tiempo": tiempo,
        "mae_test": mae_te, "rmse_test": rmse_te, "r2_test": r2_te,
        "r2_train": r2_tr, "pred_test": pred_test,
    }


def graficar_real_vs_predicho(y_test, pred_test, nombre, ruta_salida):
    """Equivalente visual de la matriz de confusión, pero para regresión."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred_test, alpha=0.5, color="#a8552f")
    lim_min = min(y_test.min(), pred_test.min())
    lim_max = max(y_test.max(), pred_test.max())
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "k--", label="Predicción perfecta")
    plt.xlabel("csMPa real")
    plt.ylabel("csMPa predicho")
    plt.title(f"Real vs predicho - {nombre}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=110)
    plt.close()
    print(f"Gráfico real vs predicho guardado en: {ruta_salida}")


# ---------------------------------------------------------------------
# 6.1 Árbol de decisión de regresión: 3 variantes + podas
# ---------------------------------------------------------------------
def experimento_arbol_regresion():
    print("\n############ ÁRBOL DE DECISIÓN - REGRESIÓN ############")
    X, y, df = cargar_concreto()
    X_train, X_test, y_train, y_test = dividir_datos(X, y, test_size=0.2)

    resultados = []

    m1 = DecisionTreeRegressor(random_state=42)
    r1 = entrenar_y_evaluar(m1, X_train, X_test, y_train, y_test,
                             "Modelo 1: Sin restricciones (80/20)")
    resultados.append(r1)

    m2 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42)
    r2 = entrenar_y_evaluar(m2, X_train, X_test, y_train, y_test,
                             "Modelo 2: Pre-poda (max_depth=4, min_samples_leaf=5)")
    resultados.append(r2)

    X_train3, X_test3, y_train3, y_test3 = dividir_datos(X, y, test_size=0.3)
    m3 = DecisionTreeRegressor(criterion="absolute_error", random_state=42)
    r3 = entrenar_y_evaluar(m3, X_train3, X_test3, y_train3, y_test3,
                             "Modelo 3: Partición 70/30 + criterio MAE")
    resultados.append(r3)

    graficar_real_vs_predicho(y_test, r1["pred_test"], r1["nombre"],
                               "resultados/reg_arbol_modelo1.png")
    graficar_real_vs_predicho(y_test, r2["pred_test"], r2["nombre"],
                               "resultados/reg_arbol_modelo2.png")
    graficar_real_vs_predicho(y_test3, r3["pred_test"], r3["nombre"],
                               "resultados/reg_arbol_modelo3.png")

    # Post-poda (cost-complexity pruning) sobre el Modelo 1
    print("\n=== Post-poda (cost-complexity pruning) sobre el Modelo 1 ===")
    arbol_completo = DecisionTreeRegressor(random_state=42)
    path = arbol_completo.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    alphas_muestra = ccp_alphas[:: max(1, len(ccp_alphas) // 25)]

    r2_train_list, r2_test_list = [], []
    for alpha in alphas_muestra:
        arbol = DecisionTreeRegressor(random_state=42, ccp_alpha=alpha)
        arbol.fit(X_train, y_train)
        r2_train_list.append(r2_score(y_train, arbol.predict(X_train)))
        r2_test_list.append(r2_score(y_test, arbol.predict(X_test)))

    plt.figure(figsize=(8, 5))
    plt.plot(alphas_muestra, r2_train_list, marker="o", label="Train")
    plt.plot(alphas_muestra, r2_test_list, marker="o", label="Test")
    plt.xlabel("ccp_alpha (nivel de poda)")
    plt.ylabel("R²")
    plt.title("Efecto de la post-poda en R² (train vs test) - Regresión")
    plt.legend()
    plt.tight_layout()
    plt.savefig("resultados/reg_poda_ccp_alpha.png", dpi=110)
    plt.close()
    print("Gráfico de post-poda guardado en: resultados/reg_poda_ccp_alpha.png")

    mejor_idx = int(np.argmax(r2_test_list))
    print(f"\nMejor ccp_alpha encontrado: {alphas_muestra[mejor_idx]:.6f}")
    print(f"R2 train con ese alpha: {r2_train_list[mejor_idx]:.4f}")
    print(f"R2 test  con ese alpha: {r2_test_list[mejor_idx]:.4f}")

    print("\n=== RESUMEN ÁRBOL DE REGRESIÓN ===")
    print(f"{'Modelo':50s} {'Tiempo(s)':>10s} {'MAE':>8s} {'RMSE':>8s} {'R2 test':>9s}")
    for r in resultados:
        print(f"{r['nombre']:50s} {r['tiempo']:>10.4f} {r['mae_test']:>8.3f} "
              f"{r['rmse_test']:>8.3f} {r['r2_test']:>9.4f}")

    return resultados


# ---------------------------------------------------------------------
# 6.2 Random Forest de regresión: 9 / 49 / 491 árboles
# ---------------------------------------------------------------------
def experimento_rf_regresion():
    print("\n############ RANDOM FOREST - REGRESIÓN ############")
    X, y, df = cargar_concreto()
    X_train, X_test, y_train, y_test = dividir_datos(X, y, test_size=0.2)

    resultados = []
    for n in N_ARBOLES:
        modelo = RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
        r = entrenar_y_evaluar(modelo, X_train, X_test, y_train, y_test,
                                f"Random Forest Regresión ({n} árboles)")
        r["n_arboles"] = n
        resultados.append(r)

    for r in resultados:
        graficar_real_vs_predicho(
            y_test, r["pred_test"], r["nombre"],
            f"resultados/reg_rf_{r['n_arboles']}arboles.png"
        )

    # Gráfico árboles vs tiempo vs R2 (equivalente al de clasificación)
    n_arboles = [r["n_arboles"] for r in resultados]
    tiempos = [r["tiempo"] for r in resultados]
    r2_test = [r["r2_test"] for r in resultados]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.set_xlabel("Número de árboles")
    ax1.set_ylabel("Tiempo de entrenamiento (s)", color="tab:orange")
    ax1.plot(n_arboles, tiempos, marker="o", color="tab:orange")
    ax1.tick_params(axis="y", labelcolor="tab:orange")
    ax1.set_xscale("log")
    ax2 = ax1.twinx()
    ax2.set_ylabel("R² en test", color="tab:blue")
    ax2.plot(n_arboles, r2_test, marker="s", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    plt.title("Random Forest Regresión: árboles vs tiempo vs R²")
    fig.tight_layout()
    plt.savefig("resultados/reg_rf_tiempo_vs_r2.png", dpi=110)
    plt.close()
    print("\nGráfico guardado en: resultados/reg_rf_tiempo_vs_r2.png")

    print("\n=== RESUMEN RANDOM FOREST REGRESIÓN ===")
    print(f"{'N° árboles':>10s} {'Tiempo(s)':>10s} {'MAE':>8s} {'RMSE':>8s} {'R2 test':>9s}")
    for r in resultados:
        print(f"{r['n_arboles']:>10d} {r['tiempo']:>10.4f} {r['mae_test']:>8.3f} "
              f"{r['rmse_test']:>8.3f} {r['r2_test']:>9.4f}")

    return resultados


if __name__ == "__main__":
    experimento_arbol_regresion()
    experimento_rf_regresion()
