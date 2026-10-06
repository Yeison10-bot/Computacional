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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

from preprocesamiento import cargar_concreto, dividir_datos, calcular_r2_ajustado
from graficos import figura_curva_poda, figura_real_vs_predicho, guardar_figura

N_ARBOLES = [9, 49, 491]


def calcular_metricas(y_real, y_pred, n_features=None):
    """
    Calcula MAE, RMSE, R² y R² Ajustado.

    n_features: número de variables del modelo (para calcular R² ajustado)
    """
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)

    # R² Ajustado: penaliza agregar variables innecesarias
    r2_ajustado = None
    if n_features is not None:
        n_muestras = len(y_real)
        r2_ajustado = calcular_r2_ajustado(r2, n_muestras, n_features)

    return mae, rmse, r2, r2_ajustado


# Umbrales del diagnóstico de ajuste (heurísticos, ver diagnosticar_ajuste)
UMBRAL_BRECHA_R2 = 0.15      # R² train - R² test
UMBRAL_RAZON_MAE = 3.0       # MAE test / MAE train
UMBRAL_R2_SUBAJUSTE = 0.60   # R² test por debajo de esto = modelo demasiado simple


def diagnosticar_ajuste(r2_train, r2_test, mae_train, mae_test):
    """
    Clasifica el ajuste de un modelo de regresión en 'sobreajuste', 'subajuste' o 'bueno'.

    La brecha de R² sola no basta: R² está acotado en 1, así que un modelo que memoriza
    (MAE train ≈ 0) puede tener una brecha de R² "moderada" y aun así equivocarse en test
    decenas de veces más que en train. Por eso también se mira la RAZÓN entre el MAE de
    test y el de train (en MPa, las unidades reales del problema).
    """
    brecha_r2 = r2_train - r2_test
    razon_mae = mae_test / mae_train if mae_train > 0 else float("inf")
    detalle = (f"diferencia R2 train-test = {brecha_r2:.4f}, "
               f"MAE test/train = {razon_mae:.1f}x")
    if brecha_r2 > UMBRAL_BRECHA_R2 or razon_mae > UMBRAL_RAZON_MAE:
        return "sobreajuste", detalle
    if r2_test < UMBRAL_R2_SUBAJUSTE:
        return "subajuste", detalle
    return "bueno", detalle


def entrenar_y_evaluar(modelo, X_train, X_test, y_train, y_test, nombre):
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo = time.time() - inicio

    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    n_features = X_train.shape[1]  # Número de variables
    mae_tr, rmse_tr, r2_tr, r2_adj_tr = calcular_metricas(y_train, pred_train, n_features)
    mae_te, rmse_te, r2_te, r2_adj_te = calcular_metricas(y_test, pred_test, n_features)

    print(f"\n--- {nombre} ---")
    print(f"Tiempo de entrenamiento : {tiempo:.4f} s")
    print(f"TRAIN -> MAE: {mae_tr:.3f} | RMSE: {rmse_tr:.3f} | R2: {r2_tr:.4f} | R2 Adj: {r2_adj_tr:.4f}")
    print(f"TEST  -> MAE: {mae_te:.3f} | RMSE: {rmse_te:.3f} | R2: {r2_te:.4f} | R2 Adj: {r2_adj_te:.4f}")

    ajuste, detalle = diagnosticar_ajuste(r2_tr, r2_te, mae_tr, mae_te)
    if ajuste == "sobreajuste":
        print(f"[!] Posible sobreajuste ({detalle})")
    elif ajuste == "subajuste":
        print(f"[!] Posible subajuste ({detalle})")
    else:
        print(f"[OK] Buen ajuste ({detalle})")

    return {
        "nombre": nombre, "modelo": modelo, "tiempo": tiempo,
        "mae_test": mae_te, "rmse_test": rmse_te, "r2_test": r2_te,
        "r2_adj_test": r2_adj_te, "r2_train": r2_tr, "r2_adj_train": r2_adj_tr,
        "pred_test": pred_test,
    }


def graficar_real_vs_predicho(y_test, pred_test, nombre, ruta_salida):
    """Equivalente visual de la matriz de confusión, pero para regresión."""
    fig = figura_real_vs_predicho(y_test, pred_test, f"Real vs predicho - {nombre}")
    guardar_figura(fig, ruta_salida, width=700)


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

    # El alpha se elige con validación cruzada (5 pliegues barajados) SOLO sobre el train;
    # el test se usa únicamente para reportar el resultado del alpha ya elegido.
    cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_train_list, r2_test_list, r2_cv_list = [], [], []
    for alpha in alphas_muestra:
        arbol = DecisionTreeRegressor(random_state=42, ccp_alpha=alpha)
        r2_cv_list.append(cross_val_score(arbol, X_train, y_train, cv=cv5, scoring="r2").mean())
        arbol.fit(X_train, y_train)
        r2_train_list.append(r2_score(y_train, arbol.predict(X_train)))
        r2_test_list.append(r2_score(y_test, arbol.predict(X_test)))

    # Gráfico interactivo de post-poda
    mejor_idx = int(np.argmax(r2_cv_list))
    fig = figura_curva_poda(alphas_muestra, r2_train_list, r2_test_list, "R²",
                            "Efecto de la post-poda en R² (train vs test) - Regresión",
                            metrica_cv=r2_cv_list)
    guardar_figura(fig, "resultados/reg_poda_ccp_alpha.png")
    print(f"\nMejor ccp_alpha (elegido por validación cruzada 5-fold): {alphas_muestra[mejor_idx]:.6f}")
    print(f"R2 CV    con ese alpha: {r2_cv_list[mejor_idx]:.4f}")
    print(f"R2 train con ese alpha: {r2_train_list[mejor_idx]:.4f}")
    print(f"R2 test  con ese alpha: {r2_test_list[mejor_idx]:.4f}")

    print("\n=== RESUMEN ÁRBOL DE REGRESIÓN ===")
    print(f"{'Modelo':50s} {'Tiempo(s)':>10s} {'MAE':>8s} {'RMSE':>8s} {'R2 test':>9s} {'R2 Adj':>9s}")
    for r in resultados:
        print(f"{r['nombre']:50s} {r['tiempo']:>10.4f} {r['mae_test']:>8.3f} "
              f"{r['rmse_test']:>8.3f} {r['r2_test']:>9.4f} {r['r2_adj_test']:>9.4f}")

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

    # Gráfico interactivo con Plotly
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Línea de tiempo
    fig.add_trace(
        go.Scatter(x=n_arboles, y=tiempos, mode="lines+markers",
                   name="Tiempo (s)", marker=dict(size=10, color="#a8552f"), line=dict(color="#a8552f"),
                   hovertemplate="<b>N arboles = %{x}</b><br>Tiempo = %{y:.4f}s<extra></extra>"),
        secondary_y=False
    )

    # Línea de R²
    fig.add_trace(
        go.Scatter(x=n_arboles, y=r2_test, mode="lines+markers",
                   name="R2 test", marker=dict(size=10, color="#3f7d54", symbol="square"), line=dict(color="#3f7d54"),
                   hovertemplate="<b>N arboles = %{x}</b><br>R2 test = %{y:.4f}<extra></extra>"),
        secondary_y=True
    )

    fig.update_xaxes(title_text="Numero de arboles", type="log")
    fig.update_yaxes(title_text="Tiempo de entrenamiento (s)", secondary_y=False)
    fig.update_yaxes(title_text="R2 en test", secondary_y=True)

    fig.update_layout(
        title="Random Forest Regresion: arboles vs tiempo vs R2 (INTERACTIVO)",
        height=600, width=1000,
        template="plotly_white",
        hovermode="x unified"
    )

    fig.write_html("resultados/reg_rf_tiempo_vs_r2_interactivo.html")
    fig.write_image("resultados/reg_rf_tiempo_vs_r2.png", width=1000, height=600)
    print("\nGrafico interactivo guardado en: resultados/reg_rf_tiempo_vs_r2_interactivo.html")
    print("Grafico estatico guardado en: resultados/reg_rf_tiempo_vs_r2.png")

    print("\n=== RESUMEN RANDOM FOREST REGRESIÓN ===")
    print(f"{'N° árboles':>10s} {'Tiempo(s)':>10s} {'MAE':>8s} {'RMSE':>8s} {'R2 test':>9s} {'R2 Adj':>9s}")
    for r in resultados:
        print(f"{r['n_arboles']:>10d} {r['tiempo']:>10.4f} {r['mae_test']:>8.3f} "
              f"{r['rmse_test']:>8.3f} {r['r2_test']:>9.4f} {r['r2_adj_test']:>9.4f}")

    return resultados


if __name__ == "__main__":
    experimento_arbol_regresion()
    experimento_rf_regresion()
