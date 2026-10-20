"""
entrega1_resultados.py
-----------------------
La entrega 1 imprimió sus resultados pero no los guardó. Este script re-entrena TODOS sus modelos
con EXACTAMENTE la misma configuración (hiperparámetros, partición y random_state=42 de
"Primer Corte/") y guarda las métricas en resultados/resultados_entrega1.json.

  Clasificación: Árbol 1/2/3, árbol podado (ccp_alpha por CV), Random Forest 9/49/491,
                 KNN (K por CV; uniforme, por distancia y con variables ponderadas),
                 K-Means (K=22 y K sugerido por silueta; no supervisado)
  Regresión    : Árbol 1/2/3, árbol podado (ccp_alpha por CV), Random Forest 9/49/491
"""

import json
import os
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score, silhouette_score)

from preprocesamiento import cargar_crop, cargar_concreto, dividir_datos, estandarizar

RUTA_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", "resultados_entrega1.json")


def _registro_clasif(nombre, familia, pred_tr, pred_te, ytr, yte, clases, test_size, tiempo):
    p_m, r_m, f_m, _ = precision_recall_fscore_support(yte, pred_te, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(yte, pred_te, average="weighted", zero_division=0)
    return {"nombre": nombre, "familia": familia, "tarea": "clasificacion", "entrega": 1,
            "test_size": test_size, "tiempo_s": round(tiempo, 4),
            "acc_train": None if pred_tr is None else float(accuracy_score(ytr, pred_tr)),
            "acc_test": float(accuracy_score(yte, pred_te)),
            "precision_macro": float(p_m), "recall_macro": float(r_m), "f1_macro": float(f_m),
            "precision_ponderada": float(p_w), "recall_ponderado": float(r_w), "f1_ponderado": float(f_w),
            "matriz_confusion": confusion_matrix(yte, pred_te, labels=clases).tolist(), "clases": list(clases)}


def metricas_clasificacion(nombre, familia, modelo, Xtr, Xte, ytr, yte, clases, test_size):
    t0 = time.time()
    modelo.fit(Xtr, ytr)
    tiempo = time.time() - t0
    return _registro_clasif(nombre, familia, modelo.predict(Xtr), modelo.predict(Xte), ytr, yte, clases, test_size, tiempo)


def metricas_regresion(nombre, familia, modelo, Xtr, Xte, ytr, yte, test_size):
    t0 = time.time()
    modelo.fit(Xtr, ytr)
    tiempo = time.time() - t0
    pred_te = modelo.predict(Xte)
    mse = mean_squared_error(yte, pred_te)
    return {"nombre": nombre, "familia": familia, "tarea": "regresion", "entrega": 1,
            "test_size": test_size, "tiempo_s": round(tiempo, 4),
            "r2_train": float(r2_score(ytr, modelo.predict(Xtr))), "r2_test": float(r2_score(yte, pred_te)),
            "mae_test": float(mean_absolute_error(yte, pred_te)), "mse_test": float(mse), "rmse_test": float(np.sqrt(mse)),
            "y_real": [float(v) for v in np.asarray(yte)], "y_pred": [float(v) for v in pred_te]}


# ------------------------------------------------------------------ MISMA LÓGICA DE SELECCIÓN QUE LA ENTREGA 1
def alpha_poda_cv(modelo_base, Xtr, ytr, cv, scoring):
    """arboles_decision.py / regresion.py: ~25 alphas de la ruta de poda, se elige el de mejor CV."""
    alphas = modelo_base.cost_complexity_pruning_path(Xtr, ytr).ccp_alphas
    alphas = alphas[:: max(1, len(alphas) // 25)]
    cvs = [cross_val_score(type(modelo_base)(random_state=42, ccp_alpha=a), Xtr, ytr, cv=cv, scoring=scoring).mean()
           for a in alphas]
    return float(alphas[int(np.argmax(cvs))])


def k_knn_cv(Xtr_e, ytr, tolerancia=0.002):
    """knn.py: K de 1 a 25 por CV; entre los que están a ≤0.002 del mejor, el impar más pequeño."""
    ks = list(range(1, 26))
    accs = [cross_val_score(KNeighborsClassifier(n_neighbors=k), Xtr_e, ytr, cv=5, scoring="accuracy").mean() for k in ks]
    cand = [k for k, a in zip(ks, accs) if a >= max(accs) - tolerancia]
    impares = [k for k in cand if k % 2 == 1]
    return min(impares) if impares else min(cand)


def kmeans_como_clasificador(X_esc, y, k, clases):
    """clustering.py: K-Means sin etiquetas y luego cada cluster -> cultivo con el algoritmo húngaro."""
    t0 = time.time()
    clusters = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_esc)
    tiempo = time.time() - t0
    y_idx = y.map({c: i for i, c in enumerate(clases)}).to_numpy()
    coinc = np.zeros((k, len(clases)), dtype=int)
    np.add.at(coinc, (clusters, y_idx), 1)
    filas, cols = linear_sum_assignment(-coinc)
    mapa = {f: cols[i] for i, f in enumerate(filas)}
    for c in range(k):
        mapa.setdefault(c, int(np.argmax(coinc[c])))
    pred = np.array([clases[mapa[c]] for c in clusters])
    return pred, tiempo


def generar():
    resultados = {"clasificacion": [], "regresion": []}

    # ---------------- CLASIFICACIÓN (cultivos) ----------------
    X, y, _ = cargar_crop()
    clases = sorted(y.unique())
    Xtr, Xte, ytr, yte = dividir_datos(X, y, test_size=0.2, estratificar=True)
    Xtr70, Xte70, ytr70, yte70 = dividir_datos(X, y, test_size=0.3, estratificar=True)
    Xtr_e, Xte_e, _ = estandarizar(Xtr, Xte)

    c = resultados["clasificacion"]
    c.append(metricas_clasificacion("Árbol 1 (sin restricción)", "Árbol", DecisionTreeClassifier(random_state=42), Xtr, Xte, ytr, yte, clases, 0.2))
    c.append(metricas_clasificacion("Árbol 2 (max_depth=4)", "Árbol", DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42), Xtr, Xte, ytr, yte, clases, 0.2))
    c.append(metricas_clasificacion("Árbol 3 (70/30, entropía)", "Árbol", DecisionTreeClassifier(criterion="entropy", random_state=42), Xtr70, Xte70, ytr70, yte70, clases, 0.3))
    a = alpha_poda_cv(DecisionTreeClassifier(random_state=42), Xtr, ytr, cv=5, scoring="accuracy")
    c.append(metricas_clasificacion(f"Árbol podado (ccp_alpha={a:.4f})", "Árbol", DecisionTreeClassifier(random_state=42, ccp_alpha=a), Xtr, Xte, ytr, yte, clases, 0.2))
    for n in (9, 49, 491):
        c.append(metricas_clasificacion(f"Random Forest ({n} árboles)", "Random Forest", RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1), Xtr, Xte, ytr, yte, clases, 0.2))

    k = k_knn_cv(Xtr_e, ytr)
    c.append(metricas_clasificacion(f"KNN (K={k}, uniforme)", "KNN", KNeighborsClassifier(n_neighbors=k), Xtr_e, Xte_e, ytr, yte, clases, 0.2))
    c.append(metricas_clasificacion(f"KNN (K={k}, por distancia)", "KNN", KNeighborsClassifier(n_neighbors=k, weights="distance"), Xtr_e, Xte_e, ytr, yte, clases, 0.2))
    imp = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr_e, ytr).feature_importances_
    pesos = imp / imp.mean()
    c.append(metricas_clasificacion(f"KNN (K={k}, variables ponderadas)", "KNN", KNeighborsClassifier(n_neighbors=k), Xtr_e * pesos, Xte_e * pesos, ytr, yte, clases, 0.2))

    # K-Means: no supervisado, evaluado sobre los 2200 registros (así se hizo en la entrega 1)
    X_esc, _, _ = estandarizar(X, X)
    siluetas = [silhouette_score(X_esc, KMeans(n_clusters=kk, random_state=42, n_init=10).fit_predict(X_esc)) for kk in range(2, 31)]
    k_sil = int(np.argmax(siluetas)) + 2
    for kk, txt in ((22, "K=22, nº real de cultivos"), (k_sil, f"K={k_sil}, sugerido por silueta")):
        pred, t = kmeans_como_clasificador(X_esc, y, kk, clases)
        reg = _registro_clasif(f"K-Means ({txt})", "K-Means", None, pred, None, y, clases, None, t)
        reg["no_supervisado"] = True
        c.append(reg)

    # ---------------- REGRESIÓN (concreto) ----------------
    Xk, yk, _ = cargar_concreto()
    Xtr, Xte, ytr, yte = dividir_datos(Xk, yk, test_size=0.2)
    Xtr70, Xte70, ytr70, yte70 = dividir_datos(Xk, yk, test_size=0.3)

    r = resultados["regresion"]
    r.append(metricas_regresion("Árbol 1 (sin restricción)", "Árbol", DecisionTreeRegressor(random_state=42), Xtr, Xte, ytr, yte, 0.2))
    r.append(metricas_regresion("Árbol 2 (max_depth=4)", "Árbol", DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42), Xtr, Xte, ytr, yte, 0.2))
    r.append(metricas_regresion("Árbol 3 (70/30, MAE)", "Árbol", DecisionTreeRegressor(criterion="absolute_error", random_state=42), Xtr70, Xte70, ytr70, yte70, 0.3))
    a = alpha_poda_cv(DecisionTreeRegressor(random_state=42), Xtr, ytr, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring="r2")
    r.append(metricas_regresion(f"Árbol podado (ccp_alpha={a:.4f})", "Árbol", DecisionTreeRegressor(random_state=42, ccp_alpha=a), Xtr, Xte, ytr, yte, 0.2))
    for n in (9, 49, 491):
        r.append(metricas_regresion(f"Random Forest ({n} árboles)", "Random Forest", RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1), Xtr, Xte, ytr, yte, 0.2))

    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=1)
    print(f"Resultados de la entrega 1 guardados en {RUTA_SALIDA}")
    return resultados


if __name__ == "__main__":
    res = generar()
    print("\nCLASIFICACIÓN  (acc_train / acc_test / F1 macro)")
    for m in res["clasificacion"]:
        tr = "  —   " if m["acc_train"] is None else f"{m['acc_train']:.4f}"
        print(f"  {m['nombre']:42s} {tr} / {m['acc_test']:.4f} / {m['f1_macro']:.4f}  ({m['tiempo_s']}s)")
    print("\nREGRESIÓN  (R² train / R² test / MAE / RMSE)")
    for m in res["regresion"]:
        print(f"  {m['nombre']:42s} {m['r2_train']:.4f} / {m['r2_test']:.4f} / {m['mae_test']:.3f} / {m['rmse_test']:.3f}")
