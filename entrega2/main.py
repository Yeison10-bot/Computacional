"""
main.py  (ENTREGA 2)
---------------------
Corre toda la segunda entrega y deja la evidencia en resultados/ (cada gráfica como HTML
interactivo + PNG, igual que en la entrega 1):
  0. Resultados de la entrega 1 (JSON)      -> entrega1_resultados.py
  1. Naive Bayes gaussiano                  -> naive_bayes.py
  2. SVM: SVC y SVR                         -> svm.py
  3. Redes neuronales (3 por tarea)         -> redes_neuronales.py
  4. Comparativa con la entrega 1           -> comparativa.py

Uso:  python main.py
"""

import os
import sys
import time

import pandas as pd

from preprocesamiento import preparar_clasificacion, preparar_regresion
import graficos as g
import entrega1_resultados
import naive_bayes as nb
import svm
import redes_neuronales as rn
import comparativa as cmp

OUT = "resultados/"
sys.stdout.reconfigure(encoding="utf-8")


def guardar(fig, nombre, width=1000):
    g.guardar_figura(fig, OUT + nombre, width=width)


def titulo(t):
    print("\n" + "=" * 72 + f"\n{t.center(72)}\n" + "=" * 72)


def main():
    os.makedirs(OUT, exist_ok=True)
    t_ini = time.time()
    dc, dr = preparar_clasificacion(), preparar_regresion()

    titulo("0. RESULTADOS DE LA ENTREGA 1")
    e1 = entrega1_resultados.generar()

    # ------------------------------------------------------------ NAIVE BAYES
    titulo("1. NAIVE BAYES GAUSSIANO")
    r_nb = nb.experimento_naive_bayes()
    print(f"Accuracy train {r_nb['train']['accuracy']:.4f} | test {r_nb['test']['accuracy']:.4f} | "
          f"F1 {r_nb['test']['f1_macro']:.4f} | manual == sklearn: {r_nb['coincide_predicciones']}")
    guardar(g.figura_matriz_confusion(r_nb["test"]["matriz_confusion"], dc["clases"], "Naive Bayes gaussiano — matriz de confusión"),
            "cm_naive_bayes.png", 1100)
    i = nb.muestra_mas_incierta(r_nb["modelo"], dc["X_test"].values)
    e = nb.explicar_muestra(r_nb["modelo"], dc["X_test"].values[i], dc["columnas"])
    guardar(g.figura_gaussianas(r_nb["modelo"], dc["X_test"].values[i], dc["columnas"], "humidity", e["top"]), "nb_gaussianas_humidity.png")
    guardar(g.figura_correlacion(nb.correlacion_intra_clase(dc["X_train"], dc["y_train"])), "nb_correlacion_intra_clase.png", 700)

    # ------------------------------------------------------------ SVM
    titulo("2. SVM (SVC y SVR)")
    b = svm.buscar_svc(dc)
    r_svc = svm.entrenar_svc(dc, **b["mejores"])
    print(f"SVC — {b['n_combinaciones']} combinaciones; mejores {b['mejores']} (CV {b['mejor_cv']:.4f}) -> "
          f"test {r_svc['test']['accuracy']:.4f} | vectores de soporte {r_svc['n_vectores_soporte']}")
    guardar(g.figura_heatmap(b["tabla_cv"], "SVC: accuracy CV (kernel × C)", "C", "kernel"), "svc_rejilla_kernel_C.png", 800)
    guardar(g.figura_heatmap(b["tabla_gamma"], f"SVC {b['kernel_gamma']}: accuracy CV (gamma × C)", "C", "gamma"), "svc_rejilla_gamma_C.png", 800)
    guardar(g.figura_matriz_confusion(r_svc["test"]["matriz_confusion"], dc["clases"], "SVC — matriz de confusión", color="Greens"),
            "cm_svc.png", 1100)
    reg = svm.regiones_decision_pca(dc, **b["mejores"])
    print(f"Regiones de decisión (PCA 2D): varianza {reg['varianza'].sum():.1%} | accuracy test 2D {reg['acc_test_2d']:.4f}")
    guardar(g.figura_regiones_decision(reg, "SVC — regiones de decisión sobre PC1 y PC2"), "svc_regiones_pca.png", 1100)
    bs = svm.buscar_svr(dr)
    r_svr = svm.entrenar_svr(dr, **bs["mejores"])
    print(f"SVR — {bs['n_combinaciones']} combinaciones; mejores {bs['mejores']} (R² CV {bs['mejor_cv']:.4f}) -> "
          f"R² test {r_svr['test']['r2']:.4f} | MAE {r_svr['test']['mae']:.3f}")
    guardar(g.figura_heatmap(bs["tabla_cv"], "SVR: R² CV (kernel × C)", "C", "kernel"), "svr_rejilla_kernel_C.png", 800)
    guardar(g.figura_heatmap(bs["tabla_epsilon"], f"SVR {bs['mejores']['kernel']}: R² CV (epsilon × C)", "C", "epsilon"), "svr_rejilla_epsilon_C.png", 800)
    guardar(g.figura_real_vs_predicho(dr["y_test"], r_svr["pred_test"], "SVR — real vs predicho"), "svr_real_vs_predicho.png", 700)
    guardar(g.figura_residuos(dr["y_test"], r_svr["pred_test"], "SVR — residuos"), "svr_residuos.png", 800)

    # ------------------------------------------------------------ REDES NEURONALES
    titulo("3. REDES NEURONALES (3 por tarea)")
    redes_c = rn.comparar_redes(dc, "clasificacion")
    print(rn.tabla_comparativa(redes_c, "clasificacion").to_string(index=False))
    for k, red in enumerate(redes_c, 1):
        guardar(g.figura_arquitectura(7, red["capas"], 22, f"{red['nombre']} — clasificación"), f"arq_clasif_red{k}.png", 900)
        guardar(g.figura_curvas_red(red["curvas"], "clasificacion", f"{red['nombre']} — clasificación"), f"curvas_clasif_red{k}.png")
        guardar(g.figura_matriz_confusion(red["test"]["matriz_confusion"], dc["clases"], f"{red['nombre']} — matriz de confusión",
                                          color="Purples"), f"cm_red{k}.png", 1100)
    guardar(g.figura_complejidad(redes_c, "clasificacion", "Clasificación: complejidad vs accuracy"), "redes_complejidad_clasif.png", 800)
    redes_r = rn.comparar_redes(dr, "regresion")
    print("\n" + rn.tabla_comparativa(redes_r, "regresion").to_string(index=False))
    for k, red in enumerate(redes_r, 1):
        guardar(g.figura_arquitectura(8, red["capas"], 1, f"{red['nombre']} — regresión"), f"arq_regr_red{k}.png", 900)
        guardar(g.figura_curvas_red(red["curvas"], "regresion", f"{red['nombre']} — regresión"), f"curvas_regr_red{k}.png")
        guardar(g.figura_real_vs_predicho(dr["y_test"], red["pred_test"], f"{red['nombre']} — real vs predicho"),
                f"real_vs_pred_red{k}.png", 700)
        guardar(g.figura_residuos(dr["y_test"], red["pred_test"], f"{red['nombre']} — residuos"), f"residuos_red{k}.png", 800)
    guardar(g.figura_complejidad(redes_r, "regresion", "Regresión: complejidad vs R²"), "redes_complejidad_regr.png", 800)

    # ------------------------------------------------------------ COMPARATIVA
    titulo("4. COMPARATIVA CON LA ENTREGA 1")
    e2 = cmp.ejecutar_entrega2(nb=r_nb, svc=(r_svc, b["mejores"]), svr=(r_svr, bs["mejores"]),
                               redes_clasif=redes_c, redes_regr=redes_r)
    cmp.guardar_entrega2(e2)
    tc, tr = cmp.tabla_clasificacion(e1, e2), cmp.tabla_regresion(e1, e2)
    tc.to_csv(OUT + "tabla_comparativa_clasificacion.csv", index=False, encoding="utf-8-sig")
    tr.to_csv(OUT + "tabla_comparativa_regresion.csv", index=False, encoding="utf-8-sig")
    print(tc.to_string(index=False)); print(); print(tr.to_string(index=False))

    todos_c, todos_r = e1["clasificacion"] + e2["clasificacion"], e1["regresion"] + e2["regresion"]
    sup_c = cmp.supervisados(todos_c)
    et = cmp.etiqueta
    guardar(g.figura_barras_comparativa(todos_c, "acc_test", "Accuracy en test — todos los modelos", "Accuracy", et), "comp_clasif_accuracy.png")
    guardar(g.figura_prf(sorted(cmp.mejor_por_familia(sup_c, "acc_test"), key=lambda m: (m["entrega"], -m["acc_test"]))), "comp_clasif_prf.png")
    guardar(g.figura_brecha(sup_c, "acc_train", "acc_test", "Accuracy train vs test (brecha = sobreajuste)", et), "comp_clasif_brecha.png")
    guardar(g.figura_tiempo_vs_metrica(sup_c, "acc_test", "Accuracy", et, "Clasificación: tiempo vs accuracy"), "comp_clasif_tiempo.png")
    guardar(g.figura_barras_comparativa(todos_r, "r2_test", "R² en test — todos los modelos", "R²", et), "comp_regr_r2.png")
    guardar(g.figura_errores_regresion(todos_r, et), "comp_regr_errores.png")
    guardar(g.figura_brecha(todos_r, "r2_train", "r2_test", "R² train vs test (brecha = sobreajuste)", et), "comp_regr_brecha.png")
    guardar(g.figura_tiempo_vs_metrica(todos_r, "r2_test", "R²", et, "Regresión: tiempo vs R²"), "comp_regr_tiempo.png")

    titulo("FIN")
    print(f"Tiempo total: {time.time() - t_ini:.1f} s — evidencia en {OUT}")


if __name__ == "__main__":
    main()
