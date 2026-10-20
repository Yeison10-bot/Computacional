"""
metricas.py  (ENTREGA 2)
-------------------------
Métricas compartidas por todos los módulos de la entrega 2 (las gráficas están en graficos.py).

Clasificación : accuracy, precision, recall y F1 (macro y ponderado), matriz de confusión.
Regresión     : MAE, MSE, RMSE, R² y R² ajustado (nuevo; usa el nº de variables).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix,
                             classification_report, mean_absolute_error, mean_squared_error, r2_score)


# ------------------------------------------------------------------ CLASIFICACIÓN
def metricas_clasif(y_real, y_pred, clases):
    """Devuelve un diccionario con todas las métricas de clasificación sobre un conjunto."""
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_real, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_real, y_pred, average="weighted", zero_division=0)
    return {"accuracy": float(accuracy_score(y_real, y_pred)),
            "precision_macro": float(p_m), "recall_macro": float(r_m), "f1_macro": float(f_m),
            "precision_ponderada": float(p_w), "recall_ponderado": float(r_w), "f1_ponderado": float(f_w),
            "matriz_confusion": confusion_matrix(y_real, y_pred, labels=clases)}


def reporte_por_clase(y_real, y_pred):
    """Tabla precision/recall/F1/soporte por cada cultivo."""
    rep = classification_report(y_real, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(rep).T.round(3)
    return df.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")



# ------------------------------------------------------------------ REGRESIÓN
def r2_ajustado(r2, n_muestras, n_variables):
    """
    R² ajustado = 1 - (1 - R²) * (n - 1) / (n - p - 1)
    Penaliza añadir variables que no mejoran el modelo (p = nº de variables, n = nº de muestras).
    """
    return 1 - (1 - r2) * (n_muestras - 1) / (n_muestras - n_variables - 1)


def metricas_regresion(y_real, y_pred, n_variables):
    """MAE, MSE, RMSE, R² y R² ajustado (todo en MPa reales)."""
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    mse = mean_squared_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    return {"mae": float(mean_absolute_error(y_real, y_pred)), "mse": float(mse),
            "rmse": float(np.sqrt(mse)), "r2": float(r2),
            "r2_ajustado": float(r2_ajustado(r2, len(y_real), n_variables))}

