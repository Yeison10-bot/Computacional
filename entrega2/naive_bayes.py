"""
naive_bayes.py  (ENTREGA 2)
----------------------------
Naive Bayes GAUSSIANO sobre Crop_recommendation (22 cultivos), IMPLEMENTADO A MANO y verificado
contra GaussianNB de scikit-learn.

    P(C | x) = P(C) · Π_i P(x_i | C) / P(x)

  · P(C)        probabilidad de clase (a priori): proporción del cultivo en train
  · P(x_i | C)  probabilidad condicional del atributo: normal con μ y σ² del atributo en ese cultivo
  · P(x)        evidencia = Σ_C P(C) · Π_i P(x_i | C)  (solo normaliza)
  · "Ingenuo":  atributos independientes dado el cultivo -> se multiplican
Se suman logaritmos (evita underflow). No necesita escalado.
"""

import time
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from preprocesamiento import preparar_clasificacion
from metricas import metricas_clasif


class NaiveBayesGaussiano:
    """Naive Bayes gaussiano implementado desde cero (mismo suavizado que scikit-learn)."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.clases_ = np.array(sorted(np.unique(y)))
        n_clases, n_vars = len(self.clases_), X.shape[1]
        self.a_priori_ = np.zeros(n_clases)                 # P(C)
        self.media_ = np.zeros((n_clases, n_vars))          # μ de cada variable en cada clase
        self.varianza_ = np.zeros((n_clases, n_vars))       # σ² de cada variable en cada clase
        for k, c in enumerate(self.clases_):
            Xc = X[y == c]
            self.a_priori_[k] = len(Xc) / len(X)
            self.media_[k] = Xc.mean(axis=0)
            self.varianza_[k] = Xc.var(axis=0)
        # Suavizado: evita varianza 0 (división por cero). Es el mismo que usa scikit-learn.
        self.varianza_ += self.var_smoothing * X.var(axis=0).max()
        return self

    def verosimilitud(self, x):
        """P(x_i | C) para cada clase (filas) y cada variable (columnas)."""
        x = np.asarray(x, dtype=float)
        return (1.0 / np.sqrt(2 * np.pi * self.varianza_)) * \
               np.exp(-((x - self.media_) ** 2) / (2 * self.varianza_))

    def log_posteriori_no_normalizado(self, X):
        """log P(C) + Σ log P(x_i|C). Se usan logaritmos para evitar underflow con muchos atributos."""
        X = np.asarray(X, dtype=float)
        salida = np.zeros((len(X), len(self.clases_)))
        for k in range(len(self.clases_)):
            log_lik = -0.5 * np.log(2 * np.pi * self.varianza_[k]) \
                      - ((X - self.media_[k]) ** 2) / (2 * self.varianza_[k])
            salida[:, k] = np.log(self.a_priori_[k]) + log_lik.sum(axis=1)
        return salida

    def predict_proba(self, X):
        """Probabilidad a posteriori P(C|x) normalizada (cada fila suma 1)."""
        lp = self.log_posteriori_no_normalizado(X)
        lp -= lp.max(axis=1, keepdims=True)                 # estabilidad numérica
        p = np.exp(lp)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.clases_[np.argmax(self.log_posteriori_no_normalizado(X), axis=1)]


def explicar_muestra(modelo, x, columnas, n_top=3):
    """
    Desglosa paso a paso la clasificación de UNA muestra:
    a priori, probabilidades condicionales, producto, evidencia y a posteriori.
    """
    x = np.asarray(x, dtype=float)
    lik = modelo.verosimilitud(x)                            # (clases, variables)
    conjunta = modelo.a_priori_ * lik.prod(axis=1)           # P(C) · Π P(x_i|C)
    evidencia = conjunta.sum()                               # P(x)
    posteriori = conjunta / evidencia                        # P(C|x)
    top = np.argsort(posteriori)[::-1][:n_top]
    nombres = [str(modelo.clases_[i]) for i in top]
    return {
        "x": dict(zip(columnas, x)), "top": nombres,
        "a_priori": {n: float(modelo.a_priori_[i]) for n, i in zip(nombres, top)},
        "condicionales": pd.DataFrame(lik[top], index=nombres, columns=columnas),
        "producto": {n: float(lik[i].prod()) for n, i in zip(nombres, top)},
        "conjunta": {n: float(conjunta[i]) for n, i in zip(nombres, top)},
        "evidencia": float(evidencia),
        "posteriori": {n: float(posteriori[i]) for n, i in zip(nombres, top)},
    }


def muestra_mas_incierta(modelo, X):
    """Índice de la muestra donde el modelo está MENOS seguro (mayor posteriori más baja):
    la mejor para explicar, porque ahí compiten varias clases."""
    return int(np.argmin(modelo.predict_proba(X).max(axis=1)))


def tabla_parametros(modelo, cultivo, columnas):
    """Media y varianza que el modelo aprendió para un cultivo (lo único que 'guarda' Naive Bayes)."""
    k = list(modelo.clases_).index(cultivo)
    return pd.DataFrame({"media (μ)": modelo.media_[k], "varianza (σ²)": modelo.varianza_[k]},
                        index=columnas).round(3)


def correlacion_intra_clase(X, y):
    """
    Correlación absoluta promedio entre variables DENTRO de cada cultivo.
    Sirve para discutir el supuesto 'ingenuo': si es baja, la independencia es razonable.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    mats = [X[y == c].corr().abs().values for c in sorted(y.unique())]
    prom = np.mean(mats, axis=0)
    return pd.DataFrame(prom, index=X.columns, columns=X.columns)



def experimento_naive_bayes():
    """Entrena Naive Bayes (manual + scikit-learn), verifica que coinciden y calcula métricas."""
    d = preparar_clasificacion()
    Xtr, Xte, ytr, yte = d["X_train"].values, d["X_test"].values, d["y_train"].values, d["y_test"].values

    t0 = time.perf_counter()
    manual = NaiveBayesGaussiano().fit(Xtr, ytr)
    tiempo = time.perf_counter() - t0
    sk = GaussianNB().fit(Xtr, ytr)

    pred_manual, pred_sk = manual.predict(Xte), sk.predict(Xte)
    coincide = bool(np.all(pred_manual == pred_sk))
    prob_coincide = bool(np.allclose(manual.predict_proba(Xte), sk.predict_proba(Xte), atol=1e-8))

    m_te = metricas_clasif(yte, pred_manual, d["clases"])
    m_tr = metricas_clasif(ytr, manual.predict(Xtr), d["clases"])
    return {"modelo": manual, "sklearn": sk, "datos": d, "tiempo": tiempo,
            "pred_test": pred_manual, "coincide_predicciones": coincide,
            "coincide_probabilidades": prob_coincide, "train": m_tr, "test": m_te}


if __name__ == "__main__":
    r = experimento_naive_bayes()
    d, nb = r["datos"], r["modelo"]
    print(f"Naive Bayes — acc train {r['train']['accuracy']:.4f} | acc test {r['test']['accuracy']:.4f} "
          f"| F1 macro {r['test']['f1_macro']:.4f} | entrenamiento {r['tiempo']:.4f}s")
    print("¿Mi implementación manual == scikit-learn?  predicciones:", r["coincide_predicciones"],
          "| probabilidades:", r["coincide_probabilidades"])
    i = muestra_mas_incierta(nb, d["X_test"].values)
    x = d["X_test"].values[i]
    e = explicar_muestra(nb, x, d["columnas"])
    print(f"\nMuestra de prueba {i} (la más incierta; real:", d["y_test"].values[i], ")")
    print("  a priori      :", {k: round(v, 4) for k, v in e["a_priori"].items()})
    print("  evidencia P(x):", f"{e['evidencia']:.3e}")
    print("  a posteriori  :", {k: round(v, 4) for k, v in e["posteriori"].items()})
    print(e["condicionales"].round(4))
    cor = correlacion_intra_clase(d["X_train"], d["y_train"])
    print("\nCorrelación absoluta intra-clase promedio (fuera de la diagonal):",
          round(float((cor.values.sum() - np.trace(cor.values)) / (cor.size - len(cor))), 3))
