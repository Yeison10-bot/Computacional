"""
svm.py  (ENTREGA 2)
--------------------
Máquinas de Soporte Vectorial:
  · SVC -> clasificación de cultivos (22 clases, uno-contra-uno)
  · SVR -> regresión de la resistencia del concreto (y estandarizada; métricas en MPa)

Hiperparámetros: kernel, C, gamma (rbf/sigmoid) y epsilon (SVR).
Se busca TODO en UNA sola rejilla con validación cruzada de 5 pliegues sobre train, y el modelo
final se entrena exactamente con la mejor combinación encontrada. El test se usa una sola vez.
Los heatmaps son "cortes" de esa misma búsqueda (cada celda = mejor valor CV con el resto libre).
Requiere estandarizar: el kernel mide distancias.
"""

import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

from preprocesamiento import preparar_clasificacion, preparar_regresion
from metricas import metricas_clasif, metricas_regresion

VALORES_C = [0.1, 1, 10, 100]
GAMMAS = ["scale", 0.01, 0.1, 1]
EPSILONS = [0.01, 0.05, 0.1, 0.2, 0.5]
KERNELS_SVC = ["linear", "poly", "rbf", "sigmoid"]
KERNELS_SVR = ["linear", "poly", "rbf"]

REJILLA_SVC = [{"kernel": ["linear"], "C": VALORES_C},
               {"kernel": ["poly"], "C": VALORES_C},                       # grado 3, gamma="scale"
               {"kernel": ["rbf", "sigmoid"], "C": VALORES_C, "gamma": GAMMAS}]
REJILLA_SVR = [{"kernel": ["linear", "poly"], "C": VALORES_C, "epsilon": EPSILONS},
               {"kernel": ["rbf"], "C": VALORES_C, "gamma": GAMMAS, "epsilon": EPSILONS}]


def _resultados(gs):
    """cv_results_ como DataFrame con gamma/epsilon legibles (los kernels que no los usan quedan en 'scale'/'—')."""
    res = pd.DataFrame(gs.cv_results_)
    for p, defecto in (("param_gamma", "scale"), ("param_epsilon", 0.1)):
        if p in res:
            res[p] = res[p].astype(object).where(res[p].notna(), defecto).astype(str)
    return res


def _corte(res, filas, columnas, filtro=None, valor="mean_test_score"):
    """Tabla filas × columnas con el MEJOR valor de la búsqueda en cada celda (opcionalmente filtrada)."""
    r = res if filtro is None else res.query(filtro)
    t = r.pivot_table(index=filas, columns=columnas, values=valor, aggfunc="max").astype(float)
    t.columns = [f"{c:g}" for c in t.columns]
    t.columns.name, t.index.name = columnas.replace("param_", ""), filas.replace("param_", "")
    return t


def _marcar(tabla, fila, columna):
    """Guarda en la tabla la celda del modelo elegido por GridSearchCV, para resaltarla en el heatmap."""
    tabla.attrs["mejor"] = (str(fila), f"{columna:g}")
    return tabla


def _limpiar(params):
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}


# ------------------------------------------------------------------ CLASIFICACIÓN (SVC)
def buscar_svc(d):
    """Búsqueda conjunta kernel × C × gamma. Devuelve mejores parámetros y cortes para graficar."""
    gs = GridSearchCV(SVC(random_state=42), REJILLA_SVC, cv=5, scoring="accuracy", n_jobs=-1, return_train_score=True)
    gs.fit(d["X_train_esc"], d["y_train"])
    res = _resultados(gs)
    mejores = _limpiar(gs.best_params_)
    k_gamma = mejores["kernel"] if mejores["kernel"] in ("rbf", "sigmoid") else "rbf"
    k, C = mejores["kernel"], mejores["C"]
    tabla_gamma = _corte(res, "param_gamma", "param_C", f"param_kernel == '{k_gamma}'")
    if k == k_gamma:
        _marcar(tabla_gamma, mejores["gamma"], C)
    return {"mejores": mejores, "mejor_cv": float(gs.best_score_), "n_combinaciones": len(res),
            "tabla_cv": _marcar(_corte(res, "param_kernel", "param_C"), k, C),
            "tabla_train": _marcar(_corte(res, "param_kernel", "param_C", valor="mean_train_score"), k, C),
            "tabla_gamma": tabla_gamma,
            "kernel_gamma": k_gamma}


def entrenar_svc(d, kernel="rbf", C=10, gamma="scale"):
    modelo = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
    t0 = time.time()
    modelo.fit(d["X_train_esc"], d["y_train"])
    tiempo = time.time() - t0
    pred_te = modelo.predict(d["X_test_esc"])
    return {"modelo": modelo, "tiempo": tiempo, "n_vectores_soporte": int(modelo.n_support_.sum()),
            "train": metricas_clasif(d["y_train"], modelo.predict(d["X_train_esc"]), d["clases"]),
            "test": metricas_clasif(d["y_test"], pred_te, d["clases"]), "pred_test": pred_te}


def regiones_decision_pca(d, kernel="rbf", C=1, gamma="scale", resolucion=250):
    """
    Solo para VISUALIZAR: proyecta los 7 atributos a 2 componentes principales (PCA ajustado con train),
    entrena un SVC con los mismos hiperparámetros sobre esas 2 dimensiones y evalúa una rejilla de puntos.
    Al perder 5 dimensiones el accuracy 2D es menor que el del modelo real: la gráfica ilustra la idea.
    """
    pca = PCA(n_components=2, random_state=42).fit(d["X_train_esc"])
    Xtr2, Xte2 = pca.transform(d["X_train_esc"]), pca.transform(d["X_test_esc"])
    modelo = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42).fit(Xtr2, d["y_train"])
    lo, hi = np.vstack([Xtr2, Xte2]).min(0) - 0.5, np.vstack([Xtr2, Xte2]).max(0) + 0.5
    xs, ys = np.linspace(lo[0], hi[0], resolucion), np.linspace(lo[1], hi[1], resolucion)
    xx, yy = np.meshgrid(xs, ys)
    pred = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    clases = np.array(d["clases"])
    return {"xs": xs, "ys": ys, "Z": np.searchsorted(clases, pred).reshape(xx.shape), "clases": clases,
            "X_test_2d": Xte2, "y_test": d["y_test"].values, "varianza": pca.explained_variance_ratio_,
            "acc_train_2d": float(np.mean(modelo.predict(Xtr2) == d["y_train"].values)),
            "acc_test_2d": float(np.mean(modelo.predict(Xte2) == d["y_test"].values))}


# ------------------------------------------------------------------ REGRESIÓN (SVR)
def buscar_svr(d):
    """Búsqueda conjunta kernel × C × gamma × epsilon (R² CV, y estandarizada)."""
    gs = GridSearchCV(SVR(), REJILLA_SVR, cv=5, scoring="r2", n_jobs=-1, return_train_score=True)
    gs.fit(d["X_train_esc"], d["y_train_norm"])
    res = _resultados(gs)
    mejores = _limpiar(gs.best_params_)
    k, C = mejores["kernel"], mejores["C"]
    tabla_gamma = _corte(res, "param_gamma", "param_C", "param_kernel == 'rbf'")
    if k == "rbf":
        _marcar(tabla_gamma, mejores["gamma"], C)
    return {"mejores": mejores, "mejor_cv": float(gs.best_score_), "n_combinaciones": len(res),
            "tabla_cv": _marcar(_corte(res, "param_kernel", "param_C"), k, C),
            "tabla_train": _marcar(_corte(res, "param_kernel", "param_C", valor="mean_train_score"), k, C),
            "tabla_epsilon": _marcar(_corte(res, "param_epsilon", "param_C", f"param_kernel == '{k}'"), mejores["epsilon"], C),
            "tabla_gamma": tabla_gamma}


def entrenar_svr(d, kernel="rbf", C=10, epsilon=0.1, gamma="scale"):
    modelo = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
    t0 = time.time()
    modelo.fit(d["X_train_esc"], d["y_train_norm"])
    tiempo = time.time() - t0
    a_mpa = lambda v: v * d["y_desv"] + d["y_media"]
    p_tr, p_te = a_mpa(modelo.predict(d["X_train_esc"])), a_mpa(modelo.predict(d["X_test_esc"]))
    n_var = d["X_train_esc"].shape[1]
    return {"modelo": modelo, "tiempo": tiempo, "n_vectores_soporte": int(len(modelo.support_)),
            "train": metricas_regresion(d["y_train"], p_tr, n_var),
            "test": metricas_regresion(d["y_test"], p_te, n_var), "pred_test": p_te}


def nombre_svm(prefijo, p):
    """Etiqueta corta con los hiperparámetros: 'SVC (rbf, C=10, gamma=0.1)'."""
    partes = [p["kernel"], f"C={p['C']}"]
    if "gamma" in p and p["kernel"] in ("rbf", "sigmoid"):
        partes.append(f"gamma={p['gamma']}")
    if "epsilon" in p:
        partes.append(f"eps={p['epsilon']}")
    return f"{prefijo} ({', '.join(partes)})"


if __name__ == "__main__":
    dc = preparar_clasificacion()
    b = buscar_svc(dc)
    print(f"SVC — {b['n_combinaciones']} combinaciones | mejores: {b['mejores']} (CV {b['mejor_cv']:.4f})")
    print(b["tabla_cv"].round(4))
    r = entrenar_svc(dc, **b["mejores"])
    print(f"{nombre_svm('SVC', b['mejores'])} -> train {r['train']['accuracy']:.4f} | test {r['test']['accuracy']:.4f} "
          f"| vect. soporte {r['n_vectores_soporte']}")
    dr = preparar_regresion()
    b = buscar_svr(dr)
    print(f"\nSVR — {b['n_combinaciones']} combinaciones | mejores: {b['mejores']} (R² CV {b['mejor_cv']:.4f})")
    print(b["tabla_epsilon"].round(4))
    r = entrenar_svr(dr, **b["mejores"])
    print(f"{nombre_svm('SVR', b['mejores'])} -> R² train {r['train']['r2']:.4f} | R² test {r['test']['r2']:.4f} "
          f"| MAE {r['test']['mae']:.3f} | vect. soporte {r['n_vectores_soporte']}")
