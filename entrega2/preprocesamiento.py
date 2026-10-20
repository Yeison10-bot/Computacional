"""
preprocesamiento.py  (ENTREGA 2)
---------------------------------
Mismas bases de datos y MISMA partición que la primera entrega, para que la
comparación entre entregas sea justa (el set de prueba es exactamente el mismo):

  - Crop_recommendation.csv -> Clasificación: 80/20, estratificado, random_state=42
  - concreto.csv            -> Regresión:     80/20, sin estratificar, random_state=42

Reglas de preprocesamiento por modelo (nuevo en esta entrega):
  - Naive Bayes     -> NO necesita escalado (cada variable se modela por separado
                       con su propia media y varianza; escalar no cambia el resultado).
  - SVM (SVC/SVR)   -> SÍ: usa distancias/productos punto entre puntos (kernel).
  - Redes neuronales-> SÍ: la escala afecta al gradiente y a la convergencia.
  - Regresión       -> además se estandariza la VARIABLE OBJETIVO (csMPa) para SVR y
                       redes; las métricas se calculan siempre en MPa reales.
El escalador se ajusta (fit) SOLO con train, para evitar fuga de información.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Rutas relativas a este archivo (no a la carpeta desde donde se ejecuta), para que funcione al desplegar
CARPETA_DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RUTA_CROP = os.path.join(CARPETA_DATOS, "Crop_recommendation.csv")
RUTA_CONCRETO = os.path.join(CARPETA_DATOS, "concreto.csv")


def cargar_crop():
    """Dataset de clasificación de cultivos (X, y, df)."""
    df = pd.read_csv(RUTA_CROP)
    return df.drop(columns=["label"]), df["label"], df


def cargar_concreto():
    """Dataset de regresión de resistencia del concreto (X, y, df)."""
    df = pd.read_csv(RUTA_CONCRETO)
    if "Row ID" in df.columns:          # solo es un índice, no aporta información
        df = df.drop(columns=["Row ID"])
    return df.drop(columns=["csMPa"]), df["csMPa"], df


def dividir_datos(X, y, test_size=0.2, random_state=42, estratificar=False):
    """Partición train/test idéntica a la de la primera entrega."""
    stratify = y if estratificar else None
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=stratify)


def estandarizar(X_train, X_test):
    """Z-score ajustado SOLO con train. Devuelve arreglos numpy y el escalador."""
    escalador = StandardScaler()
    return escalador.fit_transform(X_train), escalador.transform(X_test), escalador


def estandarizar_objetivo(y_train):
    """
    Estandariza la variable objetivo de regresión (media 0, desv. 1) usando solo train.
    Devuelve (y_train_norm, media, desviacion). Para volver a MPa: y = y_norm*desv + media.
    """
    media, desv = float(np.mean(y_train)), float(np.std(y_train))
    return (np.asarray(y_train) - media) / desv, media, desv


def preparar_clasificacion():
    """
    Deja listo TODO lo que necesitan los modelos de clasificación.
    Devuelve un diccionario con datos crudos, datos escalados y nombres de clases.
    """
    X, y, _ = cargar_crop()
    X_tr, X_te, y_tr, y_te = dividir_datos(X, y, test_size=0.2, estratificar=True)
    X_tr_esc, X_te_esc, esc = estandarizar(X_tr, X_te)
    return {"X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te,
            "X_train_esc": X_tr_esc, "X_test_esc": X_te_esc, "escalador": esc,
            "clases": sorted(y.unique()), "columnas": list(X.columns)}


def preparar_regresion():
    """Igual que la anterior pero para concreto (regresión)."""
    X, y, _ = cargar_concreto()
    X_tr, X_te, y_tr, y_te = dividir_datos(X, y, test_size=0.2)
    X_tr_esc, X_te_esc, esc = estandarizar(X_tr, X_te)
    y_tr_norm, y_media, y_desv = estandarizar_objetivo(y_tr)
    return {"X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te,
            "X_train_esc": X_tr_esc, "X_test_esc": X_te_esc, "escalador": esc,
            "y_train_norm": y_tr_norm, "y_media": y_media, "y_desv": y_desv,
            "columnas": list(X.columns)}
