"""
redes_neuronales.py  (ENTREGA 2)
---------------------------------
Perceptrón multicapa (MLP) con scikit-learn:
  · Clasificación de cultivos -> MLPClassifier (salida softmax, 22 neuronas)
  · Regresión de csMPa        -> MLPRegressor  (salida lineal, 1 neurona; y estandarizada)

3 redes por tarea, de menor a mayor complejidad, para ver si más capas/neuronas/pesos mejoran:
    Red 1 (básica)     : (8,)            1 capa oculta
    Red 2 (intermedia) : (64, 32)        2 capas ocultas
    Red 3 (compleja)   : (256, 128, 64)  3 capas ocultas
Todo lo demás igual en las tres (ReLU, Adam, lotes de 200, misma semilla y mismas épocas),
para que la ÚNICA diferencia sea la complejidad.

Se entrena época a época (partial_fit) para registrar la pérdida en train y test: las curvas son
solo diagnóstico de sobreajuste, no se usan para elegir nada.
"""

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import log_loss, mean_squared_error

from preprocesamiento import preparar_clasificacion, preparar_regresion
from metricas import metricas_clasif, metricas_regresion

warnings.filterwarnings("ignore")

ARQUITECTURAS = {"Red 1 (básica)": (8,), "Red 2 (intermedia)": (64, 32), "Red 3 (compleja)": (256, 128, 64)}
EPOCAS_CLASIF = 300
EPOCAS_REGR = 400


def contar_pesos(n_entrada, capas, n_salida):
    """Cuenta pesos (conexiones) y sesgos de la red: Σ (entradas × salidas) + salidas por capa."""
    dims = [n_entrada] + list(capas) + [n_salida]
    pesos = sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))
    sesgos = sum(dims[1:])
    return {"pesos": pesos, "sesgos": sesgos, "total": pesos + sesgos}


# ------------------------------------------------------------------ CLASIFICACIÓN
def entrenar_red_clasificacion(d, capas, epocas=EPOCAS_CLASIF, semilla=42):
    """Entrena una MLP de clasificación época a época y devuelve modelo, curvas y métricas."""
    modelo = MLPClassifier(hidden_layer_sizes=capas, random_state=semilla, batch_size=200)
    clases = np.array(d["clases"])
    Xtr, Xte = d["X_train_esc"], d["X_test_esc"]
    ytr, yte = d["y_train"].values, d["y_test"].values
    curvas = {"loss_train": [], "loss_test": [], "acc_train": [], "acc_test": []}
    tiempo = 0.0                      # solo se cuenta el entrenamiento, no el cálculo de las curvas
    for e in range(epocas):
        t0 = time.time()
        modelo.partial_fit(Xtr, ytr, classes=clases)
        tiempo += time.time() - t0
        p_tr, p_te = modelo.predict_proba(Xtr), modelo.predict_proba(Xte)
        curvas["loss_train"].append(log_loss(ytr, p_tr, labels=clases))
        curvas["loss_test"].append(log_loss(yte, p_te, labels=clases))
        curvas["acc_train"].append(float(np.mean(clases[p_tr.argmax(1)] == ytr)))
        curvas["acc_test"].append(float(np.mean(clases[p_te.argmax(1)] == yte)))
    pred_tr, pred_te = modelo.predict(Xtr), modelo.predict(Xte)
    return {"modelo": modelo, "capas": capas, "curvas": curvas, "tiempo": tiempo, "epocas": epocas,
            "pesos": contar_pesos(Xtr.shape[1], capas, len(clases)),
            "train": metricas_clasif(ytr, pred_tr, d["clases"]),
            "test": metricas_clasif(yte, pred_te, d["clases"]), "pred_test": pred_te}


# ------------------------------------------------------------------ REGRESIÓN
def entrenar_red_regresion(d, capas, epocas=EPOCAS_REGR, semilla=42):
    """Entrena una MLP de regresión época a época (y estandarizada); métricas en MPa reales."""
    modelo = MLPRegressor(hidden_layer_sizes=capas, random_state=semilla, batch_size=200)
    Xtr, Xte = d["X_train_esc"], d["X_test_esc"]
    ytr_n = d["y_train_norm"]
    yte_n = (d["y_test"].values - d["y_media"]) / d["y_desv"]
    curvas = {"loss_train": [], "loss_test": [], "r2_train": [], "r2_test": []}
    tiempo = 0.0
    for e in range(epocas):
        t0 = time.time()
        modelo.partial_fit(Xtr, ytr_n)
        tiempo += time.time() - t0
        p_tr, p_te = modelo.predict(Xtr), modelo.predict(Xte)
        curvas["loss_train"].append(mean_squared_error(ytr_n, p_tr))
        curvas["loss_test"].append(mean_squared_error(yte_n, p_te))
        curvas["r2_train"].append(1 - curvas["loss_train"][-1] / np.var(ytr_n))
        curvas["r2_test"].append(1 - curvas["loss_test"][-1] / np.var(yte_n))
    a_mpa = lambda v: v * d["y_desv"] + d["y_media"]
    p_tr, p_te = a_mpa(modelo.predict(Xtr)), a_mpa(modelo.predict(Xte))
    n_var = Xtr.shape[1]
    return {"modelo": modelo, "capas": capas, "curvas": curvas, "tiempo": tiempo, "epocas": epocas,
            "pesos": contar_pesos(n_var, capas, 1),
            "train": metricas_regresion(d["y_train"], p_tr, n_var),
            "test": metricas_regresion(d["y_test"], p_te, n_var), "pred_test": p_te}


def entrenar_red(d, tarea, nombre, capas=None, epocas=None, semilla=42):
    """Entrena UNA variante (por nombre de ARQUITECTURAS) de la tarea 'clasificacion' o 'regresion'."""
    f = entrenar_red_clasificacion if tarea == "clasificacion" else entrenar_red_regresion
    ep = epocas or (EPOCAS_CLASIF if tarea == "clasificacion" else EPOCAS_REGR)
    r = f(d, capas or ARQUITECTURAS[nombre], epocas=ep, semilla=semilla)
    r["nombre"] = nombre
    return r


def comparar_redes(d, tarea, arquitecturas=ARQUITECTURAS, epocas=None, semilla=42):
    """Entrena las 3 variantes de una tarea ('clasificacion' o 'regresion') y las deja en una lista."""
    return [entrenar_red(d, tarea, nombre, capas, epocas, semilla) for nombre, capas in arquitecturas.items()]


def tabla_comparativa(resultados, tarea):
    """DataFrame resumen: capas, pesos, tiempo, métricas train/test y brecha (sobreajuste)."""
    filas = []
    for r in resultados:
        base = {"Red": r["nombre"], "Capas ocultas": str(r["capas"]), "Pesos": r["pesos"]["pesos"],
                "Parámetros (pesos+sesgos)": r["pesos"]["total"], "Tiempo (s)": round(r["tiempo"], 2)}
        if tarea == "clasificacion":
            base.update({"Acc. train": round(r["train"]["accuracy"], 4), "Acc. test": round(r["test"]["accuracy"], 4),
                         "Brecha": round(r["train"]["accuracy"] - r["test"]["accuracy"], 4),
                         "F1 macro (test)": round(r["test"]["f1_macro"], 4),
                         "Loss test final": round(r["curvas"]["loss_test"][-1], 4),
                         "Mejor loss test": round(min(r["curvas"]["loss_test"]), 4)})
        else:
            base.update({"R² train": round(r["train"]["r2"], 4), "R² test": round(r["test"]["r2"], 4),
                         "Brecha": round(r["train"]["r2"] - r["test"]["r2"], 4),
                         "MAE (MPa)": round(r["test"]["mae"], 3), "RMSE (MPa)": round(r["test"]["rmse"], 3),
                         "R² ajustado": round(r["test"]["r2_ajustado"], 4)})
        filas.append(base)
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ FORWARD A MANO
def forward_manual(modelo, x, tarea):
    """
    Recorre la red capa por capa con numpy (a = ReLU(W·a + b)) para UNA muestra y comprueba que da lo
    mismo que predict_proba/predict de scikit-learn. Devuelve las activaciones de cada capa.
    """
    a = np.asarray(x, dtype=float)
    activaciones = [a]
    n_capas = len(modelo.coefs_)
    for i, (W, b) in enumerate(zip(modelo.coefs_, modelo.intercepts_)):
        z = a @ W + b
        if i < n_capas - 1:
            a = np.maximum(z, 0)                      # ReLU
        elif tarea == "clasificacion":
            e = np.exp(z - z.max())
            a = e / e.sum()                           # softmax: probabilidades que suman 1
        else:
            a = z                                     # salida lineal (regresión)
        activaciones.append(a)
    ref = modelo.predict_proba([x])[0] if tarea == "clasificacion" else modelo.predict([x])
    return activaciones, bool(np.allclose(activaciones[-1], ref, atol=1e-6))


if __name__ == "__main__":
    dc = preparar_clasificacion()
    res = comparar_redes(dc, "clasificacion")
    print("=== CLASIFICACIÓN ===")
    print(tabla_comparativa(res, "clasificacion").to_string(index=False))
    dr = preparar_regresion()
    res_r = comparar_redes(dr, "regresion")
    print("\n=== REGRESIÓN ===")
    print(tabla_comparativa(res_r, "regresion").to_string(index=False))
    x = dc["X_test_esc"][0]
    acts, ok = forward_manual(res[1]["modelo"], x, "clasificacion")
    print("\nForward manual == scikit-learn (clasificación):", ok)
    acts, ok = forward_manual(res_r[1]["modelo"], dr["X_test_esc"][0], "regresion")
    print("Forward manual == scikit-learn (regresión):", ok)
