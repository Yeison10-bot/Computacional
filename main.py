"""
main.py
--------
Orquestador del proyecto. Corre TODOS los experimentos en orden y deja
la evidencia (gráficos, matrices de confusión) en la carpeta resultados/.

Uso:
    python3 main.py
"""

import os
import time

from preprocesamiento import cargar_crop, cargar_concreto, resumen_distribucion
from arboles_decision import experimento_arboles
from random_forest import experimento_random_forest
from regresion import experimento_arbol_regresion, experimento_rf_regresion
from knn import experimento_knn
from clustering import experimento_clustering


def separador(titulo):
    print("\n" + "=" * 70)
    print(titulo.center(70))
    print("=" * 70)


def main():
    os.makedirs("resultados", exist_ok=True)
    inicio_total = time.time()

    separador("0. RESUMEN DE LAS BASES DE DATOS")
    X, y, df_crop = cargar_crop()
    resumen_distribucion(df_crop, columna_clase="label")
    X2, y2, df_conc = cargar_concreto()
    resumen_distribucion(df_conc)

    separador("1. ÁRBOLES DE DECISIÓN (Clasificación - cultivos)")
    experimento_arboles()

    separador("2. RANDOM FOREST (Clasificación - cultivos)")
    experimento_random_forest()

    separador("3. ÁRBOLES Y RANDOM FOREST (Regresión - concreto)")
    experimento_arbol_regresion()
    experimento_rf_regresion()

    separador("4. KNN (Clasificación - cultivos)")
    experimento_knn()

    separador("5. CLUSTERING / K-MEANS (cultivos)")
    experimento_clustering()

    tiempo_total = time.time() - inicio_total
    separador("PROYECTO COMPLETO")
    print(f"Tiempo total de ejecución: {tiempo_total:.2f} s")
    print("Todos los gráficos y matrices de confusión quedaron en: resultados/")


if __name__ == "__main__":
    main()
