"""
app.py  (ENTREGA 2)
--------------------
Interfaz de la segunda entrega. Igual que en la primera: los modelos se entrenan EN VIVO con cada
botón y todas las gráficas son interactivas (Plotly). Los resultados de la entrega 1 se leen del JSON.

    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

from preprocesamiento import preparar_clasificacion, preparar_regresion
from metricas import reporte_por_clase
import graficos as g
import naive_bayes as nbm
import svm as svmm
import redes_neuronales as rn
import comparativa as cmp
import entrega1_resultados

st.set_page_config(page_title="Inteligencia Computacional 2 — Entrega 2", layout="wide", page_icon="🧠")


# ---------------------------------------------------------------------------------------------
# Caché
# ---------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def datos_clasif():
    return preparar_clasificacion()


@st.cache_resource(show_spinner=False)
def datos_regr():
    return preparar_regresion()


@st.cache_resource(show_spinner=False)
def naive_bayes_entrenado():
    return nbm.experimento_naive_bayes()


@st.cache_resource(show_spinner=False)
def svc_optimo():
    d = datos_clasif()
    b = svmm.buscar_svc(d)
    return b, svmm.entrenar_svc(d, **b["mejores"])


@st.cache_resource(show_spinner=False)
def regiones_svc(kernel, C, gamma):
    return svmm.regiones_decision_pca(datos_clasif(), kernel=kernel, C=C, gamma=gamma)


@st.cache_resource(show_spinner=False)
def svr_optimo():
    d = datos_regr()
    b = svmm.buscar_svr(d)
    return b, svmm.entrenar_svr(d, **b["mejores"])


@st.cache_resource(show_spinner=False)
def red_entrenada(tarea, nombre):
    return rn.entrenar_red(datos_clasif() if tarea == "clasificacion" else datos_regr(), tarea, nombre)


def redes_entrenadas(tarea):
    return [red_entrenada(tarea, n) for n in rn.ARQUITECTURAS]


@st.cache_data(show_spinner=False)
def resultados_entrega1():
    if not os.path.exists(cmp.RUTA_E1):
        os.makedirs(os.path.dirname(cmp.RUTA_E1), exist_ok=True)
        entrega1_resultados.generar()
    return cmp.cargar_json(cmp.RUTA_E1)


# ---------------------------------------------------------------------------------------------
# Utilidades de presentación
# ---------------------------------------------------------------------------------------------
RESALTE = "background-color: #2e7d32; color: white; font-weight: bold"   # legible en tema claro y oscuro


def grafica(fig):
    st.plotly_chart(fig, width="stretch")


def palabras_clave(items):
    """Expander con términos clave del tema: término -> definición de una línea."""
    with st.expander("🔑 Palabras clave"):
        st.markdown("\n".join(f"- **{k}** — {v}" for k, v in items.items()))


def fila_clasif(m_tr, m_te, tiempo=None):
    c = st.columns(6 if tiempo is not None else 5)
    c[0].metric("Accuracy train", f"{m_tr['accuracy']:.2%}")
    c[1].metric("Accuracy test", f"{m_te['accuracy']:.2%}")
    c[2].metric("Precision", f"{m_te['precision_macro']:.4f}")
    c[3].metric("Recall", f"{m_te['recall_macro']:.4f}")
    c[4].metric("F1", f"{m_te['f1_macro']:.4f}")
    if tiempo is not None:
        c[5].metric("Tiempo", f"{tiempo:.3f} s")
    brecha = m_tr["accuracy"] - m_te["accuracy"]
    if brecha > 0.03:
        st.warning(f"Sobreajuste: train supera a test por {brecha:.2%}.")
    elif m_te["accuracy"] < 0.9:
        st.warning("Subajuste: accuracy de test baja.")
    else:
        st.success(f"Buen ajuste — brecha train−test = {brecha:.2%}.")


def fila_regr(m_tr, m_te, tiempo=None):
    c = st.columns(6 if tiempo is not None else 5)
    c[0].metric("R² train", f"{m_tr['r2']:.4f}")
    c[1].metric("R² test", f"{m_te['r2']:.4f}")
    c[2].metric("R² ajustado", f"{m_te['r2_ajustado']:.4f}")
    c[3].metric("MAE", f"{m_te['mae']:.2f} MPa")
    c[4].metric("RMSE", f"{m_te['rmse']:.2f} MPa")
    if tiempo is not None:
        c[5].metric("Tiempo", f"{tiempo:.3f} s")
    brecha = m_tr["r2"] - m_te["r2"]
    if brecha > 0.08:
        st.warning(f"Sobreajuste: R² train supera a test por {brecha:.3f}.")
    elif m_te["r2"] < 0.8:
        st.warning("Subajuste: R² de test bajo.")
    else:
        st.success(f"Ajuste razonable — brecha R² train−test = {brecha:.3f}.")


def boton(texto, clave):
    """Botón que deja el estado 'activado' entre recargas de Streamlit."""
    if st.button(texto, type="primary", key=f"btn_{clave}"):
        st.session_state[clave] = True
    return st.session_state.get(clave, False)


# ---------------------------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Entrega 2")
    st.markdown("""
    1. **Inicio**
    2. **Naive Bayes**
    3. **SVM**
    4. **Redes neuronales**
    5. **Comparativa** con la entrega 1
    """)
    st.caption("Todo se entrena en vivo. Misma partición que la entrega 1 (80/20, random_state=42).")

st.title("Inteligencia Computacional 2 — Segunda entrega")
st.caption("Razonamiento probabilístico · SVM · Redes neuronales — cultivos (clasificación) y concreto (regresión)")

tabs = st.tabs(["📖 Inicio", "🎲 Naive Bayes", "🧲 SVM", "🧠 Redes neuronales",
                "⚖️ Comparativa"])

# =============================================================================================
# INICIO
# =============================================================================================
with tabs[0]:
    st.header("Qué se hace")
    st.markdown("""
    | Base de datos | Tarea | Entrega 2 | Entrega 1 (comparación) |
    |---|---|---|---|
    | `Crop_recommendation.csv` · 2200 registros · 7 variables · 22 cultivos | Clasificación | Naive Bayes · SVC · 3 redes neuronales | Árboles · Árbol podado · Random Forest · KNN · K-Means |
    | `concreto.csv` · 721 registros · 8 variables · `csMPa` | Regresión | SVR · 3 redes neuronales | Árboles · Árbol podado · Random Forest |
    """)
    st.markdown("""
    | Modelo | Escalado de X | Hiperparámetros elegidos con |
    |---|---|---|
    | Naive Bayes gaussiano | No | — (no tiene) |
    | SVC / SVR | Sí (Z-score) | kernel, C, gamma, epsilon por CV 5 pliegues |
    | Redes neuronales | Sí (y también `csMPa` en regresión) | Arquitectura fija: básica → intermedia → compleja |
    """)
    st.info("**Comparación justa:** misma partición 80/20 con `random_state=42` → el set de prueba es idéntico al de la entrega 1 "
            "(440 cultivos, 145 concretos). El escalador se ajusta solo con train. El test se usa una única vez.")
    palabras_clave({
        "Partición estratificada": "train y test conservan la proporción de cada cultivo.",
        "Fuga de información (data leakage)": "usar datos de test para ajustar algo (escalador, hiperparámetros); se evita ajustando todo solo con train.",
        "Validación cruzada (k-fold)": "train se parte en k pliegues; se entrena en k−1 y se valida en el restante, k veces; se promedia.",
        "Sobreajuste / subajuste": "train mucho mejor que test / ambos malos.",
        "Estandarización (Z-score)": "(x − μ)/σ con μ y σ de train; necesaria en modelos basados en distancias o gradientes.",
    })

# =============================================================================================
# NAIVE BAYES
# =============================================================================================
with tabs[1]:
    st.header("Naive Bayes gaussiano")
    st.latex(r"P(C\mid x)=\frac{P(C)\,\prod_{i=1}^{7}P(x_i\mid C)}{P(x)}\qquad P(x_i\mid C)=\frac{1}{\sqrt{2\pi\sigma^2_{C,i}}}\,e^{-\frac{(x_i-\mu_{C,i})^2}{2\sigma^2_{C,i}}}")
    palabras_clave({
        "Probabilidad de clase / a priori P(C)": "proporción del cultivo en train (100/2200 → 1/22 = 4.55 %, dataset balanceado).",
        "Probabilidad condicional del atributo P(xᵢ|C)": "qué tan probable es ese valor del atributo si el cultivo es C; aquí una normal con μ y σ² del cultivo.",
        "Probabilidad a posteriori P(C|x)": "probabilidad del cultivo después de ver la muestra; se predice la mayor (regla MAP).",
        "Evidencia P(x)": "Σ P(C)·Π P(xᵢ|C); solo normaliza, no cambia cuál clase gana.",
        "Supuesto ingenuo": "atributos independientes dado el cultivo → la conjunta es un producto.",
        "Log-probabilidades": "se suman logs en vez de multiplicar 7 densidades pequeñas (evita underflow).",
        "var_smoothing": "pequeña cantidad sumada a las varianzas para no dividir por cero.",
        "Modelo generativo": "modela cómo se generan los datos de cada clase (P(x|C)), no solo la frontera.",
    })
    if boton("🎲 Entrenar Naive Bayes", "nb"):
        r = naive_bayes_entrenado()
        d, nb = r["datos"], r["modelo"]
        fila_clasif(r["train"], r["test"], r["tiempo"])
        st.caption("Datos sin escalar: cada variable se modela con su propia media y varianza.")
        c1, c2 = st.columns([3, 2])
        with c1:
            grafica(g.figura_matriz_confusion(r["test"]["matriz_confusion"], d["clases"], "Matriz de confusión (test)"))
        with c2:
            st.dataframe(reporte_por_clase(d["y_test"], r["pred_test"]), height=560)

        st.subheader("Paso a paso de una muestra")
        modo = st.radio("Muestra", ["La más incierta del test", "Elegir índice"], horizontal=True, key="nb_modo")
        idx = nbm.muestra_mas_incierta(nb, d["X_test"].values) if modo.startswith("La") else \
            int(st.number_input("Índice (0–439)", 0, len(d["X_test"]) - 1, 0, key="nb_idx"))
        x = d["X_test"].values[idx]
        e = nbm.explicar_muestra(nb, x, d["columnas"], n_top=3)
        st.markdown(f"Muestra **#{idx}** — real: **{d['y_test'].values[idx]}** · predicho: **{nb.predict([x])[0]}**")
        st.dataframe(pd.DataFrame([e["x"]]).round(2), hide_index=True)
        st.markdown("**1. P(C)** · **2. P(xᵢ|C)** para los 3 cultivos más probables")
        st.dataframe(e["condicionales"].assign(**{"P(C)": pd.Series(e["a_priori"])}).round(5))
        tabla = pd.DataFrame({"Π P(xᵢ|C)": e["producto"], "P(C)·Π P(xᵢ|C)": e["conjunta"], "P(C|x)": e["posteriori"]})
        st.markdown(f"**3. Conjunta, evidencia P(x) = {e['evidencia']:.3e} y posteriori**")
        st.dataframe(tabla.style.format("{:.3e}", subset=tabla.columns[:2]).format("{:.4f}", subset=["P(C|x)"]))
        c1, c2 = st.columns(2)
        with c1:
            grafica(g.figura_barras(list(e["posteriori"]), list(e["posteriori"].values()), "Probabilidad a posteriori", "P(C|x)"))
        with c2:
            var = st.selectbox("Variable", d["columnas"], index=d["columnas"].index("humidity"), key="nb_var")
            grafica(g.figura_gaussianas(nb, x, d["columnas"], var, e["top"]))

        st.subheader("¿Se cumple el supuesto ingenuo?")
        cor = nbm.correlacion_intra_clase(d["X_train"], d["y_train"])
        prom = (cor.values.sum() - np.trace(cor.values)) / (cor.size - len(cor))
        c1, c2 = st.columns([3, 2])
        with c1:
            grafica(g.figura_correlacion(cor))
        with c2:
            st.metric("Correlación intra-cultivo promedio", f"{prom:.3f}")
            st.markdown("Dentro de cada cultivo las variables casi no están correlacionadas → la independencia condicional "
                        "es razonable, y por eso un modelo tan simple alcanza ≈99.5 %.")
            cult = st.selectbox("μ y σ² aprendidas para", list(nb.clases_), key="nb_cult")
            st.dataframe(nbm.tabla_parametros(nb, cult, d["columnas"]))

# =============================================================================================
# SVM
# =============================================================================================
with tabs[2]:
    st.header("Máquinas de soporte vectorial — SVC y SVR")
    palabras_clave({
        "Hiperplano de margen máximo": "la frontera que deja la mayor distancia posible a los puntos más cercanos de cada clase.",
        "Vectores de soporte": "los puntos sobre o dentro del margen; son los únicos que definen el modelo.",
        "Margen suave y C": "C penaliza los errores: C alto → margen estrecho, riesgo de sobreajuste; C bajo → margen ancho, más simple.",
        "Kernel trick": "calcular productos punto en un espacio de más dimensiones sin construirlo; permite fronteras no lineales.",
        "Kernels": "linear, poly (polinomio), rbf = exp(−γ‖x−x′‖²), sigmoid (tipo tanh).",
        "gamma (γ)": "alcance de cada punto en rbf: alto → frontera muy local (sobreajuste); bajo → suave.",
        "Uno-contra-uno": "con 22 clases se entrenan 22·21/2 = 231 SVM binarios y se vota.",
        "SVR y tubo ε": "en regresión, los errores menores que ε no se penalizan; ε grande → modelo más simple.",
        "Búsqueda en rejilla + CV": "se prueban todas las combinaciones con validación cruzada; el modelo final usa la mejor.",
        "PCA (componentes principales)": "comprime las variables en pocas direcciones que conservan la mayor varianza; aquí solo para dibujar en 2D.",
        "Residuo": "real − predicho; un buen modelo de regresión tiene residuos centrados en 0 y sin colas largas.",
    })
    tarea_svm = st.radio("Tarea", ["Clasificación (SVC)", "Regresión (SVR)"], horizontal=True, key="tarea_svm")
    if tarea_svm.startswith("Clasificación"):
        d = datos_clasif()
        if boton("🔍 Buscar hiperparámetros y entrenar el mejor SVC", "svc"):
            with st.spinner("Probando kernel × C × gamma con CV de 5 pliegues..."):
                b, r = svc_optimo()
            st.success(f"{b['n_combinaciones']} combinaciones probadas. Mejor por CV: **{svmm.nombre_svm('SVC', b['mejores'])}** "
                       f"(accuracy CV {b['mejor_cv']:.4f}). El modelo final se entrena con exactamente esos valores.")
            c1, c2, c3 = st.columns(3)
            with c1:
                grafica(g.figura_heatmap(b["tabla_cv"], "Accuracy CV (kernel × C)", "C", "kernel"))
            with c2:
                grafica(g.figura_heatmap(b["tabla_train"], "Accuracy TRAIN (kernel × C)", "C", "kernel"))
            with c3:
                grafica(g.figura_heatmap(b["tabla_gamma"], f"{b['kernel_gamma']}: accuracy CV (gamma × C)", "C", "gamma"))
            st.caption("Cada celda es el mejor valor con los demás hiperparámetros libres. Train ≫ CV indica sobreajuste.")
            fila_clasif(r["train"], r["test"], r["tiempo"])
            st.metric("Vectores de soporte", f"{r['n_vectores_soporte']} de {len(d['X_train'])}")
            grafica(g.figura_matriz_confusion(r["test"]["matriz_confusion"], d["clases"], "SVC — matriz de confusión (test)", color="Greens"))
            with st.expander("🗺️ Regiones de decisión (PCA a 2 dimensiones)"):
                p = b["mejores"]
                with st.spinner("Calculando regiones..."):
                    reg = regiones_svc(p["kernel"], p["C"], p.get("gamma", "scale"))
                grafica(g.figura_regiones_decision(reg, f"Regiones de decisión — {svmm.nombre_svm('SVC', p)} sobre PC1 y PC2"))
                c1, c2, c3 = st.columns(3)
                c1.metric("Varianza explicada (PC1+PC2)", f"{reg['varianza'].sum():.1%}")
                c2.metric("Accuracy test en 2D", f"{reg['acc_test_2d']:.2%}")
                c3.metric("Accuracy test con las 7 variables", f"{r['test']['accuracy']:.2%}")
                st.caption("Cada color de fondo es la región que el SVM asigna a un cultivo; los puntos son el set de prueba. "
                           "PCA comprime las 7 variables en 2 para poder dibujarlas, pero conserva menos de la mitad de la información: "
                           "por eso en 2D se confunden cultivos que con las 7 variables se separan casi perfecto.")
        with st.expander("🧪 Probar otros hiperparámetros a mano"):
            c1, c2, c3 = st.columns(3)
            kernel = c1.selectbox("kernel", svmm.KERNELS_SVC, index=2, key="svc_k")
            C = c2.select_slider("C", [0.01, 0.1, 1, 10, 100, 1000], value=1, key="svc_c")
            gamma = c3.selectbox("gamma", svmm.GAMMAS + [10], key="svc_g")
            if st.button("Entrenar", key="svc_manual"):
                r = svmm.entrenar_svc(d, kernel=kernel, C=C, gamma=gamma)
                fila_clasif(r["train"], r["test"], r["tiempo"])
                st.caption(f"Vectores de soporte: {r['n_vectores_soporte']}")
    else:
        d = datos_regr()
        if boton("🔍 Buscar hiperparámetros y entrenar el mejor SVR", "svr"):
            with st.spinner("Probando kernel × C × gamma × epsilon con CV de 5 pliegues..."):
                b, r = svr_optimo()
            st.success(f"{b['n_combinaciones']} combinaciones probadas. Mejor por CV: **{svmm.nombre_svm('SVR', b['mejores'])}** "
                       f"(R² CV {b['mejor_cv']:.4f}).")
            c1, c2, c3 = st.columns(3)
            with c1:
                grafica(g.figura_heatmap(b["tabla_cv"], "R² CV (kernel × C)", "C", "kernel"))
            with c2:
                grafica(g.figura_heatmap(b["tabla_epsilon"], f"{b['mejores']['kernel']}: R² CV (epsilon × C)", "C", "epsilon"))
            with c3:
                grafica(g.figura_heatmap(b["tabla_gamma"], "rbf: R² CV (gamma × C)", "C", "gamma"))
            st.caption(f"ε está en unidades de csMPa estandarizada: ε = {b['mejores']['epsilon']} ≈ {b['mejores']['epsilon'] * d['y_desv']:.1f} MPa. "
                       "El kernel lineal se queda corto: la resistencia depende de la mezcla de forma no lineal.")
            fila_regr(r["train"], r["test"], r["tiempo"])
            st.metric("Vectores de soporte", f"{r['n_vectores_soporte']} de {len(d['X_train'])}")
            c1, c2 = st.columns(2)
            with c1:
                grafica(g.figura_real_vs_predicho(d["y_test"], r["pred_test"], "SVR — real vs predicho (test)"))
            with c2:
                grafica(g.figura_residuos(d["y_test"], r["pred_test"]))
        with st.expander("🧪 Probar otros hiperparámetros a mano"):
            c1, c2, c3, c4 = st.columns(4)
            kernel = c1.selectbox("kernel", svmm.KERNELS_SVR, index=2, key="svr_k")
            C = c2.select_slider("C", [0.01, 0.1, 1, 10, 100, 1000], value=100, key="svr_c")
            gamma = c3.selectbox("gamma", svmm.GAMMAS + [10], key="svr_g")
            eps = c4.select_slider("epsilon", [0.01, 0.05, 0.1, 0.2, 0.5, 1.0], value=0.2, key="svr_e")
            if st.button("Entrenar", key="svr_manual"):
                r = svmm.entrenar_svr(d, kernel=kernel, C=C, epsilon=eps, gamma=gamma)
                fila_regr(r["train"], r["test"], r["tiempo"])

# =============================================================================================
# REDES NEURONALES
# =============================================================================================
with tabs[3]:
    st.header("Redes neuronales — básica, intermedia y compleja")
    st.markdown("Las tres redes son iguales en todo (ReLU, Adam, lotes de 200, misma semilla y épocas) "
                "**excepto en la complejidad**: más capas, más neuronas y más pesos.")
    palabras_clave({
        "Perceptrón multicapa (MLP)": "capas de neuronas totalmente conectadas: entrada → ocultas → salida.",
        "Neurona": "a = f(Σ wᵢxᵢ + b): suma ponderada de las entradas más un sesgo, pasada por una activación.",
        "Pesos y sesgos": "los parámetros que se aprenden; una capa de n→m neuronas tiene n·m pesos y m sesgos.",
        "ReLU": "f(z) = max(0, z) en las capas ocultas; introduce no linealidad.",
        "Softmax": "salida de clasificación: 22 probabilidades que suman 1.",
        "Función de pérdida": "log-loss (entropía cruzada) en clasificación; MSE en regresión.",
        "Forward / backpropagation": "forward calcula la salida; backprop usa la regla de la cadena para obtener el gradiente de cada peso.",
        "Adam": "optimizador: descenso de gradiente con tasa de aprendizaje adaptativa por peso.",
        "Época / lote (batch)": "época = una pasada por todo train; lote = 200 muestras por actualización.",
        "Capacidad": "cuánto puede memorizar la red; más parámetros = más capacidad = más riesgo de sobreajuste.",
    })
    tarea_nn = st.radio("Tarea", ["Clasificación (cultivos)", "Regresión (concreto)"], horizontal=True, key="tarea_nn")
    tarea = "clasificacion" if tarea_nn.startswith("Clasificación") else "regresion"
    d = datos_clasif() if tarea == "clasificacion" else datos_regr()
    n_ent, n_sal = (7, 22) if tarea == "clasificacion" else (8, 1)
    st.dataframe(pd.DataFrame([{"Red": n, "Capas ocultas": str(c), "Neuronas ocultas": sum(c),
                                "Pesos": rn.contar_pesos(n_ent, c, n_sal)["pesos"],
                                "Parámetros": rn.contar_pesos(n_ent, c, n_sal)["total"]}
                               for n, c in rn.ARQUITECTURAS.items()]), hide_index=True)
    if boton("🧠 Entrenar las 3 redes", f"nn_{tarea}"):
        redes = []
        for nombre in rn.ARQUITECTURAS:
            with st.spinner(f"Entrenando {nombre} época por época..."):
                redes.append(red_entrenada(tarea, nombre))
        st.dataframe(rn.tabla_comparativa(redes, tarea), hide_index=True)
        c1, c2 = st.columns([3, 2])
        with c1:
            grafica(g.figura_complejidad(redes, tarea))
        with c2:
            clave = "accuracy" if tarea == "clasificacion" else "r2"
            mejor = max(redes, key=lambda r: r["test"][clave])
            st.markdown("**Lectura**")
            for r in redes:
                gap = r["train"][clave] - r["test"][clave]
                st.markdown(f"- **{r['nombre']}**: test {r['test'][clave]:.4f}, brecha {gap:.4f}, {r['tiempo']:.1f} s")
            st.markdown(f"La mejor en test es **{mejor['nombre']}**. Si la red compleja no supera a la intermedia, "
                        "la complejidad extra solo aumenta la brecha y el tiempo: más capacidad de la que el problema necesita.")
        for r in redes:
            with st.expander(f"{r['nombre']} — {r['capas']} — {r['pesos']['total']:,} parámetros"):
                c1, c2 = st.columns([2, 3])
                with c1:
                    grafica(g.figura_arquitectura(n_ent, r["capas"], n_sal, r["nombre"]))
                with c2:
                    grafica(g.figura_curvas_red(r["curvas"], tarea, "Train vs test por época"))
                if tarea == "clasificacion":
                    fila_clasif(r["train"], r["test"], r["tiempo"])
                    grafica(g.figura_matriz_confusion(r["test"]["matriz_confusion"], d["clases"], "Matriz de confusión (test)", color="Purples"))
                else:
                    fila_regr(r["train"], r["test"], r["tiempo"])
                    c1, c2 = st.columns(2)
                    with c1:
                        grafica(g.figura_real_vs_predicho(d["y_test"], r["pred_test"], "Real vs predicho (test)"))
                    with c2:
                        grafica(g.figura_residuos(d["y_test"], r["pred_test"]))
        st.caption("Curvas: si la pérdida de test deja de bajar (o sube) mientras la de train sigue bajando, hay sobreajuste. "
                   "Son solo diagnóstico: no se usan para elegir nada.")

        st.subheader("Forward a mano")
        c1, c2 = st.columns(2)
        nombre_red = c1.selectbox("Red", list(rn.ARQUITECTURAS), key=f"fw_red_{tarea}")
        red = next(r for r in redes if r["nombre"] == nombre_red)
        idx = int(c2.number_input(f"Muestra del test (0–{len(d['X_test']) - 1})", 0, len(d["X_test"]) - 1, 0, key=f"fw_idx_{tarea}"))
        acts, ok = rn.forward_manual(red["modelo"], d["X_test_esc"][idx], tarea)
        for i, a in enumerate(acts[1:], 1):
            capa = f"Oculta {i} (ReLU)" if i < len(acts) - 1 else ("Salida (softmax)" if tarea == "clasificacion" else "Salida (lineal)")
            st.markdown(f"- **{capa}** · {len(a)} neuronas · primeras: `{np.round(a[:5], 3).tolist()}`")
        if tarea == "clasificacion":
            top = np.argsort(acts[-1])[::-1][:3]
            st.markdown("Salida: " + " · ".join(f"**{red['modelo'].classes_[j]}** {acts[-1][j]:.3f}" for j in top)
                        + f" — real: **{d['y_test'].values[idx]}**")
        else:
            st.markdown(f"Salida {acts[-1][0]:.3f} (estandarizada) → **{acts[-1][0] * d['y_desv'] + d['y_media']:.2f} MPa** "
                        f"— real: **{d['y_test'].values[idx]:.2f} MPa**")
        st.caption(f"a = ReLU(W·a + b) capa por capa con numpy; coincide con scikit-learn: **{ok}**")

# =============================================================================================
# COMPARATIVA
# =============================================================================================
with tabs[4]:
    st.header("Comparativa con la entrega 1")
    st.markdown("Azul = entrega 1 · naranja = entrega 2. **\\*** partición 70/30 · **†** K-Means (no supervisado, evaluado sobre los 2200 registros).")
    usar_guardado = st.checkbox("Usar resultados guardados de la entrega 2 (rápido)", value=os.path.exists(cmp.RUTA_E2),
                                disabled=not os.path.exists(cmp.RUTA_E2))
    if boton("⚖️ Comparar", "cmp"):
        e1 = resultados_entrega1()
        if usar_guardado:
            e2 = cmp.cargar_json(cmp.RUTA_E2)
        else:
            with st.spinner("Entrenando todos los modelos de la entrega 2..."):
                e2 = cmp.ejecutar_entrega2(nb=naive_bayes_entrenado(),
                                           svc=(svc_optimo()[1], svc_optimo()[0]["mejores"]),
                                           svr=(svr_optimo()[1], svr_optimo()[0]["mejores"]),
                                           redes_clasif=redes_entrenadas("clasificacion"), redes_regr=redes_entrenadas("regresion"))
        et = cmp.etiqueta
        sub = st.tabs(["Clasificación (cultivos)", "Regresión (concreto)"])
        with sub[0]:
            todos = e1["clasificacion"] + e2["clasificacion"]
            sup = cmp.supervisados(todos)
            st.dataframe(cmp.tabla_clasificacion(e1, e2).style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1"], props=RESALTE)
                         .format(precision=4, na_rep="—"), hide_index=True, height=35 * (len(todos) + 1) + 3)
            n_test = len(datos_clasif()["y_test"])
            b1 = max(e1["clasificacion"], key=lambda m: m["acc_test"])
            b2 = max(e2["clasificacion"], key=lambda m: m["acc_test"])
            st.success(f"Mejor E1: **{b1['nombre']}** {b1['acc_test']:.2%} · Mejor E2: **{b2['nombre']}** {b2['acc_test']:.2%}. "
                       f"Un acierto = {1 / n_test:.2%} del test: diferencias de 1–2 muestras no son significativas.")
            grafica(g.figura_barras_comparativa(todos, "acc_test", "Accuracy en test", "Accuracy", et))
            c1, c2 = st.columns(2)
            with c1:
                grafica(g.figura_prf(sorted(cmp.mejor_por_familia(sup, "acc_test"), key=lambda m: (m["entrega"], -m["acc_test"]))))
            with c2:
                grafica(g.figura_tiempo_vs_metrica(sup, "acc_test", "Accuracy", et, "Tiempo vs accuracy"))
            grafica(g.figura_brecha(sup, "acc_train", "acc_test", "Train vs test (brecha = sobreajuste)", et))
            nombres = [cmp.etiqueta_entrega(m) for m in todos]
            sel = st.multiselect("Matrices de confusión", nombres, max_selections=4,
                                 default=[n for n in nombres if "491" in n or "gaussiano" in n][:2], key="cmp_cm")
            cols = st.columns(2)
            for k, n in enumerate(sel):
                m = todos[nombres.index(n)]
                with cols[k % 2]:
                    grafica(g.figura_matriz_confusion(m["matriz_confusion"], m["clases"], f"{n} — acc {m['acc_test']:.4f}",
                                                      color="Oranges" if m["entrega"] == 2 else "Blues"))
        with sub[1]:
            todos = e1["regresion"] + e2["regresion"]
            st.dataframe(cmp.tabla_regresion(e1, e2).style.highlight_max(subset=["R²", "R² ajustado"], props=RESALTE)
                         .highlight_min(subset=["MAE", "MSE", "RMSE"], props=RESALTE).format(precision=4),
                         hide_index=True, height=35 * (len(todos) + 1) + 3)
            b1 = max(e1["regresion"], key=lambda m: m["r2_test"])
            b2 = max(e2["regresion"], key=lambda m: m["r2_test"])
            st.success(f"Mejor E1: **{b1['nombre']}** R² {b1['r2_test']:.4f} (MAE {b1['mae_test']:.2f} MPa) · "
                       f"Mejor E2: **{b2['nombre']}** R² {b2['r2_test']:.4f} (MAE {b2['mae_test']:.2f} MPa).")
            grafica(g.figura_barras_comparativa(todos, "r2_test", "R² en test", "R²", et))
            c1, c2 = st.columns(2)
            with c1:
                grafica(g.figura_errores_regresion(todos, et))
            with c2:
                grafica(g.figura_tiempo_vs_metrica(todos, "r2_test", "R²", et, "Tiempo vs R²"))
            grafica(g.figura_brecha(todos, "r2_train", "r2_test", "Train vs test (brecha = sobreajuste)", et))
            nombres = [cmp.etiqueta_entrega(m) for m in todos]
            sel = st.multiselect("Real vs predicho", nombres, max_selections=4,
                                 default=[n for n in nombres if "491" in n or "SVR" in n][:2], key="cmp_rp")
            cols = st.columns(2)
            for k, n in enumerate(sel):
                m = todos[nombres.index(n)]
                with cols[k % 2]:
                    grafica(g.figura_real_vs_predicho(m["y_real"], m["y_pred"], f"{n} — R² {m['r2_test']:.3f}",
                                                      color=g.COLOR_E2 if m["entrega"] == 2 else g.COLOR_E1))
