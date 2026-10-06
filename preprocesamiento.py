"""
preprocesamiento.py
--------------------
Carga y preprocesamiento de las dos bases de datos del proyecto:
  - Crop_recommendation.csv  -> Clasificación (label = cultivo)
  - concreto.csv (train.csv) -> Regresión (csMPa = resistencia)

Reglas aplicadas (ver guía teórica):
  - Árboles / Random Forest -> NO requieren escalado.
  - KNN / K-means           -> SÍ requieren estandarización (Z-score).
  - El escalador se ajusta (fit) SOLO con el set de entrenamiento,
    para evitar fuga de información hacia el set de prueba.
"""
#-------------------Importacion de la librerias 
#pandas: lee el CSV y lo guarda en una tabla (DataFrame)
import pandas as pd
#
#train_test_split: divide los datos en entrenamiento y prueba.
from sklearn.model_selection import train_test_split
#StandardScaler / MinMaxScaler: son los escaladores.
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import os

# Rutas relativas a este archivo (no a la carpeta desde donde se ejecuta) y con la
# "D" mayúscula de la carpeta Data/: en Linux (servidor de despliegue) importa.
CARPETA_DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
#Base de datos de cultivos (clasificación)
RUTA_CROP = os.path.join(CARPETA_DATOS, "Crop_recommendation.csv")
#Base de datos de concreto (regresión)
RUTA_CONCRETO = os.path.join(CARPETA_DATOS, "concreto.csv")


def cargar_crop():
    """Carga el dataset de clasificación de cultivos."""
    #.read_csv(...) es una función de pandas que abre un archivo CSV (valores separados por comas) y lo convierte en un DataFrame, que es una tabla con filas y columnas, parecida a una hoja de Excel.
    df = pd.read_csv(RUTA_CROP)
    X = df.drop(columns=["label"])
    #drop no cambia df. Devuelve una tabla nueva sin esa columna, y esa tabla nueva se guarda en X
    y = df["label"]
    #df["label"] selecciona solo la columna label de df
    return X, y, df


def cargar_concreto():
    """Carga el dataset de regresión de resistencia del concreto."""
    df = pd.read_csv(RUTA_CONCRETO)
    # 'Row ID' es solo un índice, no aporta información predictiva
    if "Row ID" in df.columns:
        df = df.drop(columns=["Row ID"])
    X = df.drop(columns=["csMPa"])
    y = df["csMPa"]
    return X, y, df

# train_test_split corta los datos:
    # - X e y: Datos de entrada y etiquetas a dividir.
    # - test_size=0.2: Reserva el 20% para prueba y el 80% para entrenamiento.
    # - random_state=42: Semilla fija para que el orden aleatorio sea siempre reproducible.
    # - stratify=stratify: Mantiene la proporción idéntica de clases en clasificación.
def dividir_datos(X, y, test_size=0.2, random_state=42, estratificar=False):
    """
    Divide en train/test.
    estratificar=True se usa en clasificación para mantener la misma
    proporción de clases en train y test (importante con 22 clases).
    """
    # Condición ternaria: Si estratificar es True, usa las etiquetas (y) para equilibrar el reparto. 
    # Si es False, asigna None (reparto aleatorio libre).
    stratify = y if estratificar else None
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )


def estandarizar(X_train, X_test):
    """Z-score: media 0, desviación estándar 1. Para KNN y K-means."""
    # crea un objeto StandardScaler vacío.
    scaler = StandardScaler()
    # fit_transform (Solo en Train): 
        # - fit: Calcula la media y desviación estándar de cada columna.
        # - transform: Aplica la fórmula (valor - media) / desviación.
    X_train_esc = scaler.fit_transform(X_train)
    # transform (Solo en Test):
        # Usa la media y desviación del entrenamiento para escalar el test, 
        # evitando la fuga de información (data leakage).
    X_test_esc = scaler.transform(X_test)
    return X_train_esc, X_test_esc, scaler
    # Nota: El resultado pasa de ser un DataFrame a un array de NumPy 
    # (pierde los nombres de las columnas y quedan solo los números).

def escalar_minmax(X_train, X_test):
    """Escalado a rango [0, 1]."""
    # Crea un objeto MinMaxScaler vacío. 
    scaler = MinMaxScaler()
    # fit_transform (Solo en Train): 
    # - fit: Calcula el mínimo y máximo de cada columna.
    # - transform: Aplica la fórmula (valor - min) / (max - min). Todo queda entre 0 y 1.
    X_train_esc = scaler.fit_transform(X_train)
    # transform (Solo en Test):
    # Usa el min y max del entrenamiento para escalar el test sin reajustar (evita data leakage).
    X_test_esc = scaler.transform(X_test)
    return X_train_esc, X_test_esc, scaler


def calcular_r2_ajustado(r2, n_muestras, n_features):
    """
    Calcula R² Ajustado usando la fórmula:
    R² Ajustado = 1 - [(1 - R²) × (n - 1) / (n - p - 1)]

    Parámetros:
      r2: Coeficiente de determinación (R²) tradicional
      n_muestras: número de observaciones
      n_features: número de variables predictivas

    Retorna:
      r2_ajustado: R² penalizado por número de variables
    """
    return 1 - ((1 - r2) * (n_muestras - 1) / (n_muestras - n_features - 1))


def resumen_distribucion(df, columna_clase=None):
    """Imprime un resumen rápido de la distribución del dataset."""
    # Imprime el número total de filas (registros) del DataFrame.
    print(f"Registros: {len(df)}")
    # Auditoría de limpieza: Cuenta cuántos valores nulos o vacíos hay en todo el DataFrame.
    print(f"Nulos por columna:\n{df.isnull().sum().sum()} nulos en total")
    # Si se especifica una columna de clase (Clasificación):
    # Muestra cuántos registros existen para cada categoría (conteo de frecuencias).
    if columna_clase:
        print(f"\nDistribución de clases ({columna_clase}):")
        print(df[columna_clase].value_counts())
    # Si no se especifica columna (Regresión):
    # Muestra un resumen estadístico completo (media, min, max, desviación, etc.) de las variables.
    else:
        print(f"\nEstadísticas descriptivas:\n{df.describe()}")


if __name__ == "__main__":
    # Bloque de ejecución principal (funciona como un banco de pruebas autónomo).
    # Solo se ejecuta si corres este archivo directamente, no si lo importas desde otro script.
    print("=== Dataset de cultivos (clasificación) ===")
    X, y, df_crop = cargar_crop()
    resumen_distribucion(df_crop, columna_clase="label")# Muestra conteo por clases.

    print("\n=== Dataset de concreto (regresión) ===")# Muestra estadísticas descriptivas numéricas.
    X2, y2, df_conc = cargar_concreto()
    resumen_distribucion(df_conc)
