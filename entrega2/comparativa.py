"""
comparativa.py  (ENTREGA 2)
----------------------------
Compara los modelos de la entrega 2 (Naive Bayes, SVM, redes neuronales) contra
TODOS los de la entrega 1 (árboles, árbol podado, Random Forest, KNN y K-Means).

Comparación justa: misma partición (80/20, random_state=42; estratificada en clasificación), así que
el set de prueba es idéntico. Excepciones (marcadas en las tablas):
  *  "Árbol 3" de la entrega 1 usó 70/30.
  †  K-Means es no supervisado: se evaluó sobre los 2200 registros (no tiene train/test).
"""

import json
import os
import pandas as pd

from preprocesamiento import preparar_clasificacion, preparar_regresion
from metricas import r2_ajustado
from naive_bayes import experimento_naive_bayes
from svm import buscar_svc, entrenar_svc, buscar_svr, entrenar_svr, nombre_svm
from redes_neuronales import comparar_redes

CARPETA_RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
RUTA_E1 = os.path.join(CARPETA_RESULTADOS, "resultados_entrega1.json")
RUTA_E2 = os.path.join(CARPETA_RESULTADOS, "resultados_entrega2.json")


# ------------------------------------------------------------------ REGISTROS (mismo esquema que entrega 1)
def registro_clasif(nombre, familia, res, d):
    t, te = res["train"], res["test"]
    return {"nombre": nombre, "familia": familia, "tarea": "clasificacion", "entrega": 2, "test_size": 0.2,
            "tiempo_s": round(float(res["tiempo"]), 4),
            "acc_train": t["accuracy"], "acc_test": te["accuracy"],
            "precision_macro": te["precision_macro"], "recall_macro": te["recall_macro"], "f1_macro": te["f1_macro"],
            "precision_ponderada": te["precision_ponderada"], "recall_ponderado": te["recall_ponderado"],
            "f1_ponderado": te["f1_ponderado"], "matriz_confusion": te["matriz_confusion"].tolist(),
            "clases": [str(c) for c in d["clases"]]}


def registro_regr(nombre, familia, res, d):
    t, te = res["train"], res["test"]
    return {"nombre": nombre, "familia": familia, "tarea": "regresion", "entrega": 2, "test_size": 0.2,
            "tiempo_s": round(float(res["tiempo"]), 4),
            "r2_train": t["r2"], "r2_test": te["r2"], "r2_ajustado_test": te["r2_ajustado"],
            "mae_test": te["mae"], "mse_test": te["mse"], "rmse_test": te["rmse"],
            "y_real": [float(v) for v in d["y_test"].values], "y_pred": [float(v) for v in res["pred_test"]]}


def ejecutar_entrega2(nb=None, svc=None, svr=None, redes_clasif=None, redes_regr=None):
    """
    Entrena los modelos de la entrega 2 (o reutiliza los que se pasen ya entrenados) y devuelve
    {'clasificacion': [...], 'regresion': [...]}. SVM con hiperparámetros elegidos por CV.
    svc / svr = (resultado_entrenamiento, mejores_parametros).
    """
    dc, dr = preparar_clasificacion(), preparar_regresion()
    nb = nb or experimento_naive_bayes()
    if svc is None:
        p = buscar_svc(dc)["mejores"]
        svc = (entrenar_svc(dc, **p), p)
    if svr is None:
        p = buscar_svr(dr)["mejores"]
        svr = (entrenar_svr(dr, **p), p)

    clasif = [registro_clasif("Naive Bayes gaussiano", "Naive Bayes", nb, dc),
              registro_clasif(nombre_svm("SVC", svc[1]), "SVM", svc[0], dc)]
    for r in (redes_clasif or comparar_redes(dc, "clasificacion")):
        clasif.append(registro_clasif(r["nombre"].replace("Red", "Red neuronal"), "Red neuronal", r, dc))

    regr = [registro_regr(nombre_svm("SVR", svr[1]), "SVM", svr[0], dr)]
    for r in (redes_regr or comparar_redes(dr, "regresion")):
        regr.append(registro_regr(r["nombre"].replace("Red", "Red neuronal"), "Red neuronal", r, dr))
    return {"clasificacion": clasif, "regresion": regr}


def guardar_entrega2(resultados, ruta=RUTA_E2):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=1)


def cargar_json(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ TABLAS
def etiqueta(m):
    """Nombre del modelo con su marca: * partición 70/30, † no supervisado."""
    if m.get("no_supervisado"):
        return m["nombre"] + " †"
    return m["nombre"] + (" *" if m["test_size"] != 0.2 else "")


def etiqueta_entrega(m):
    return f"{etiqueta(m)} (E{m['entrega']})"


def tabla_clasificacion(e1, e2):
    filas = []
    for m in list(e1["clasificacion"]) + list(e2["clasificacion"]):
        tr = m["acc_train"]
        filas.append({"Entrega": m["entrega"], "Modelo": etiqueta(m), "Familia": m["familia"],
                      "Acc. train": float("nan") if tr is None else round(tr, 4), "Accuracy": round(m["acc_test"], 4),
                      "Brecha": float("nan") if tr is None else round(tr - m["acc_test"], 4),
                      "Precision": round(m["precision_macro"], 4), "Recall": round(m["recall_macro"], 4),
                      "F1": round(m["f1_macro"], 4), "Tiempo (s)": m["tiempo_s"]})
    return pd.DataFrame(filas)


def tabla_regresion(e1, e2):
    filas = []
    for m in list(e1["regresion"]) + list(e2["regresion"]):
        r2a = m.get("r2_ajustado_test", r2_ajustado(m["r2_test"], len(m["y_real"]), 8))
        filas.append({"Entrega": m["entrega"], "Modelo": etiqueta(m), "Familia": m["familia"],
                      "R² train": round(m["r2_train"], 4), "R²": round(m["r2_test"], 4),
                      "R² ajustado": round(r2a, 4), "Brecha": round(m["r2_train"] - m["r2_test"], 4),
                      "MAE": round(m["mae_test"], 3), "MSE": round(m["mse_test"], 3),
                      "RMSE": round(m["rmse_test"], 3), "Tiempo (s)": m["tiempo_s"]})
    return pd.DataFrame(filas)


def mejor_por_familia(modelos, clave):
    """Mejor modelo de cada (familia, entrega) según `clave`."""
    mejores = {}
    for m in modelos:
        k = (m["familia"], m["entrega"])
        if k not in mejores or m[clave] > mejores[k][clave]:
            mejores[k] = m
    return list(mejores.values())


def supervisados(modelos):
    return [m for m in modelos if not m.get("no_supervisado")]


if __name__ == "__main__":
    e1 = cargar_json(RUTA_E1)
    e2 = cargar_json(RUTA_E2) if os.path.exists(RUTA_E2) else ejecutar_entrega2()
    print(tabla_clasificacion(e1, e2).to_string(index=False))
    print()
    print(tabla_regresion(e1, e2).to_string(index=False))
