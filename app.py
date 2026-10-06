"""
app.py
-------
Interfaz interactiva y AUTOSUFICIENTE del proyecto.

Todo pasa aquí: no hace falta correr nada antes. Cada modelo se entrena
EN VIVO cuando presionas el botón (nada viene precalculado), y cada
sección trae su propia explicación teórica en un expander "📚 Teoría"
para poder sustentar el proceso sin depender de un documento aparte.

Ejecutar con:
    streamlit run app.py
"""

import time
import io
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, _tree
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score, silhouette_score, davies_bouldin_score,
    classification_report,
)
from sklearn.model_selection import (
    cross_val_score, train_test_split, validation_curve, learning_curve, KFold
)
from scipy.optimize import linear_sum_assignment

from preprocesamiento import cargar_crop, cargar_concreto, dividir_datos, estandarizar, calcular_r2_ajustado
from graficos import figura_matriz_confusion, figura_curva_poda
from regresion import diagnosticar_ajuste
from knn import VALORES_K, elegir_k_optimo
from clustering import RANGO_K

st.set_page_config(page_title="Inteligencia Computacional 2", layout="wide", page_icon="🧠")


# =====================================================================
# SELECCIÓN AUTOMÁTICA DE K (se calcula una vez y queda en caché)
# =====================================================================
PONDERACION_VARIABLES = "variables ponderadas"


@st.cache_data(show_spinner=False)
def pesos_variables_rf():
    """Pesos por variable = importancia según un Random Forest entrenado solo con
    train, normalizada para que el peso promedio sea 1."""
    X, y, _ = cargar_crop()
    X_train, X_test, y_train, _ = dividir_datos(X, y, test_size=0.2, estratificar=True)
    X_train_esc, _, _ = estandarizar(X_train, X_test)
    rf_ref = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_ref.fit(X_train_esc, y_train)
    return rf_ref.feature_importances_ / rf_ref.feature_importances_.mean()


@st.cache_data(show_spinner=False)
def curva_k_knn(ponderacion="uniform"):
    """Accuracy por validación cruzada (5-fold, solo train) para cada K y el K
    óptimo. Cuando todos los vecinos votan igual ('uniform' y variables ponderadas)
    se usa la regla de knn.py (impar más pequeño entre los mejores, para evitar
    empates); con 'distance' se toma el máximo directamente."""
    X, y, _ = cargar_crop()
    X_train, X_test, y_train, _ = dividir_datos(X, y, test_size=0.2, estratificar=True)
    X_train_esc, _, _ = estandarizar(X_train, X_test)
    weights = "distance" if ponderacion == "distance" else "uniform"
    if ponderacion == PONDERACION_VARIABLES:
        X_train_esc = X_train_esc * pesos_variables_rf()
    accs = [cross_val_score(KNeighborsClassifier(n_neighbors=kk, weights=weights),
                            X_train_esc, y_train, cv=5).mean() for kk in VALORES_K]
    return accs, elegir_k_optimo(VALORES_K, accs, preferir_impar=(weights == "uniform"))


@st.cache_data(show_spinner=False)
def curvas_k_kmeans():
    """Inercia, silueta y Davies-Bouldin para K=2..30, y el mejor K de cada métrica."""
    X, _, _ = cargar_crop()
    X_esc, _, _ = estandarizar(X, X)
    inercias, siluetas, dbs = [], [], []
    for kk in RANGO_K:
        m = KMeans(n_clusters=kk, random_state=42, n_init=10)
        etq = m.fit_predict(X_esc)
        inercias.append(m.inertia_)
        siluetas.append(silhouette_score(X_esc, etq))
        dbs.append(davies_bouldin_score(X_esc, etq))
    k_sil = RANGO_K[int(np.argmax(siluetas))]
    k_db = RANGO_K[int(np.argmin(dbs))]
    return inercias, siluetas, dbs, k_sil, k_db


# =====================================================================
# FUNCIONES HELPER PARA GRÁFICOS INTERACTIVOS (Plotly)
# =====================================================================
def grafico_confusion_matrix_interactivo(cm, labels, titulo="Matriz de Confusión", color="Blues", **kwargs):
    """Matriz de confusión interactiva: celdas en cero vacías, aciertos y errores en
    escalas de color separadas y números discretos (se pueden ocultar con un botón)."""
    return figura_matriz_confusion(cm, labels, titulo, color=color, **kwargs)


def grafico_barras_interactivo(valores, nombres, titulo="", ylabel="", color="#3f7d54"):
    """Crea un gráfico de barras interactivo (pasa mouse para ver valores)"""
    fig = go.Figure(data=go.Bar(
        x=nombres,
        y=valores,
        marker_color=color,
        text=[f"{v:.4f}" for v in valores],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Valor: %{y:.4f}<extra></extra>"
    ))
    fig.update_layout(
        title=titulo,
        yaxis_title=ylabel,
        template="plotly_white",
        height=500,
        hovermode="x"
    )
    return fig


def grafico_scatter_interactivo(x, y, titulo="", xlabel="", ylabel="", labels=None):
    """Crea un scatter plot interactivo (real vs predicho)"""
    lim_min = min(min(x), min(y))
    lim_max = max(max(x), max(y))

    fig = go.Figure()

    # Línea perfecta
    fig.add_trace(go.Scatter(
        x=[lim_min, lim_max], y=[lim_min, lim_max],
        mode="lines",
        name="Predicción perfecta",
        line=dict(dash="dash", color="black"),
        hoverinfo="skip"
    ))

    # Puntos
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        name="Predicciones",
        marker=dict(size=6, color="#a8552f", opacity=0.6),
        hovertemplate="Real: %{y:.2f}<br>Predicho: %{x:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=titulo,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_white",
        height=600,
        width=700,
        hovermode="closest"
    )
    return fig


def grafico_validation_curve_interactivo(x_vals, tr_mean, tr_std, te_mean, te_std, titulo="", xlabel=""):
    """Crea una curva de validación interactiva con bandas de desviación"""
    fig = go.Figure()

    # Banda train
    fig.add_trace(go.Scatter(
        x=list(x_vals) + list(reversed(x_vals)),
        y=list(tr_mean + tr_std) + list(reversed(tr_mean - tr_std)),
        fill="toself",
        fillcolor="rgba(168, 85, 47, 0.2)",
        line=dict(color="rgba(168, 85, 47, 0)"),
        hoverinfo="skip",
        name="Train ± 1 desv"
    ))

    # Línea train
    fig.add_trace(go.Scatter(
        x=x_vals, y=tr_mean,
        mode="lines+markers",
        name="Train",
        line=dict(color="#a8552f", width=2),
        marker=dict(size=6),
        hovertemplate="Depth: %{x}<br>R² Train: %{y:.4f}<extra></extra>"
    ))

    # Banda test
    fig.add_trace(go.Scatter(
        x=list(x_vals) + list(reversed(x_vals)),
        y=list(te_mean + te_std) + list(reversed(te_mean - te_std)),
        fill="toself",
        fillcolor="rgba(63, 125, 84, 0.2)",
        line=dict(color="rgba(63, 125, 84, 0)"),
        hoverinfo="skip",
        name="Test ± 1 desv"
    ))

    # Línea test
    fig.add_trace(go.Scatter(
        x=x_vals, y=te_mean,
        mode="lines+markers",
        name="Test (CV 5-fold)",
        line=dict(color="#3f7d54", width=2),
        marker=dict(size=6),
        hovertemplate="Depth: %{x}<br>R² Test: %{y:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=titulo,
        xaxis_title=xlabel,
        yaxis_title="R²",
        template="plotly_white",
        height=600,
        width=1000,
        hovermode="x unified"
    )
    return fig


def grafico_learning_curve_interactivo(tamanos, tr_mean, tr_std, te_mean, te_std, titulo=""):
    """Crea una curva de aprendizaje interactiva con bandas de desviación"""
    fig = go.Figure()

    # Banda train
    fig.add_trace(go.Scatter(
        x=list(tamanos) + list(reversed(tamanos)),
        y=list(tr_mean + tr_std) + list(reversed(tr_mean - tr_std)),
        fill="toself",
        fillcolor="rgba(168, 85, 47, 0.2)",
        line=dict(color="rgba(168, 85, 47, 0)"),
        hoverinfo="skip",
        name="Train ± 1 desv"
    ))

    # Línea train
    fig.add_trace(go.Scatter(
        x=tamanos, y=tr_mean,
        mode="lines+markers",
        name="Train",
        line=dict(color="#a8552f", width=2),
        marker=dict(size=6),
        hovertemplate="Datos: %{x:.0f}<br>R² Train: %{y:.4f}<extra></extra>"
    ))

    # Banda test
    fig.add_trace(go.Scatter(
        x=list(tamanos) + list(reversed(tamanos)),
        y=list(te_mean + te_std) + list(reversed(te_mean - te_std)),
        fill="toself",
        fillcolor="rgba(63, 125, 84, 0.2)",
        line=dict(color="rgba(63, 125, 84, 0)"),
        hoverinfo="skip",
        name="Test ± 1 desv"
    ))

    # Línea test
    fig.add_trace(go.Scatter(
        x=tamanos, y=te_mean,
        mode="lines+markers",
        name="Test (CV 5-fold)",
        line=dict(color="#3f7d54", width=2),
        marker=dict(size=6),
        hovertemplate="Datos: %{x:.0f}<br>R² Test: %{y:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=titulo,
        xaxis_title="Cantidad de datos de entrenamiento",
        yaxis_title="R²",
        template="plotly_white",
        height=600,
        width=1000,
        hovermode="x unified"
    )
    return fig


def grafico_pca_clusters_interactivo(X_2d, clusters, cent_2d, var_exp, k_cluster):
    """Crea scatter plot interactivo de clusters en PCA"""
    # Paleta de 20 colores
    colores_tab20 = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbbd22", "#9edae5"
    ]

    fig = go.Figure()

    # Scatter de puntos (con color por cluster % 20)
    colores_puntos = [colores_tab20[c % 20] for c in clusters]
    fig.add_trace(go.Scatter(
        x=X_2d[:, 0], y=X_2d[:, 1],
        mode="markers",
        marker=dict(size=6, color=colores_puntos, opacity=0.7),
        text=[f"Cluster {c}" for c in clusters],
        hovertemplate="PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>%{text}<extra></extra>",
        name="Puntos de datos"
    ))

    # Scatter de centroides
    fig.add_trace(go.Scatter(
        x=cent_2d[:, 0], y=cent_2d[:, 1],
        mode="markers",
        marker=dict(size=15, color="black", symbol="x", line=dict(color="white", width=2)),
        text=[f"Centroide {i}" for i in range(len(cent_2d))],
        hovertemplate="PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>%{text}<extra></extra>",
        name="Centroides"
    ))

    fig.update_layout(
        title=f"Clusters de K-means (K={k_cluster}) - Vista PCA (INTERACTIVO)",
        xaxis_title=f"PC1 ({var_exp[0]:.1%} de la varianza)",
        yaxis_title=f"PC2 ({var_exp[1]:.1%} de la varianza)",
        template="plotly_white",
        height=600,
        width=900,
        hovermode="closest"
    )
    return fig


def grafico_pca_cultivos_interactivo(X_2d, y_idx, cultivos, var_exp):
    """Crea scatter plot interactivo de cultivos reales en PCA"""
    # Paleta de 20 colores
    colores_tab20 = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbbd22", "#9edae5"
    ]

    fig = go.Figure()

    # Scatter de puntos (con color por cultivo real % 20)
    colores_puntos = [colores_tab20[y % 20] for y in y_idx]
    fig.add_trace(go.Scatter(
        x=X_2d[:, 0], y=X_2d[:, 1],
        mode="markers",
        marker=dict(size=6, color=colores_puntos, opacity=0.7),
        text=[f"Cultivo: {cultivos[y]}" for y in y_idx],
        hovertemplate="PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>%{text}<extra></extra>",
        name="Cultivos"
    ))

    fig.update_layout(
        title="Cultivos reales - Vista PCA (INTERACTIVO)",
        xaxis_title=f"PC1 ({var_exp[0]:.1%} de la varianza)",
        yaxis_title=f"PC2 ({var_exp[1]:.1%} de la varianza)",
        template="plotly_white",
        height=600,
        width=900,
        hovermode="closest"
    )
    return fig


# =====================================================================
# Funciones auxiliares reutilizadas en varias pestañas
# =====================================================================
def mostrar_reporte_clasificacion(y_test, pred_test, titulo="Reporte de métricas por clase"):
    """Muestra precisión, recall y F1-score por clase (no solo accuracy global)."""
    reporte = classification_report(y_test, pred_test, output_dict=True, zero_division=0)
    df_reporte = pd.DataFrame(reporte).transpose().round(3)
    # Separar las filas de clases de las de resumen (accuracy/macro/weighted)
    filas_clases = df_reporte.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
    filas_resumen = df_reporte.loc[df_reporte.index.isin(["macro avg", "weighted avg"])]

    st.markdown(f"**{titulo}**")
    st.dataframe(filas_clases[["precision", "recall", "f1-score", "support"]],
                 use_container_width=True)
    c1, c2 = st.columns(2)
    if "macro avg" in df_reporte.index:
        c1.metric("F1 promedio macro", f"{df_reporte.loc['macro avg', 'f1-score']:.3f}",
                   help="Promedio simple del F1 de todas las clases, sin importar cuántas muestras tenga cada una")
    if "weighted avg" in df_reporte.index:
        c2.metric("F1 promedio ponderado", f"{df_reporte.loc['weighted avg', 'f1-score']:.3f}",
                   help="Promedio del F1 ponderado por la cantidad de muestras de cada clase")
    st.caption("**Precisión**: de lo que el modelo predijo como esa clase, cuánto acertó. "
               "**Recall**: de lo que realmente era esa clase, cuánto detectó. "
               "**F1**: combina ambas en un solo número.")


def _posiciones_arbol(tree_):
    """
    Calcula la posición (x, y) de cada nodo SIN que se puedan solapar:
    a cada hoja se le asigna su propia columna (de izquierda a derecha) y
    cada nodo interno se ubica exactamente en el promedio de sus hijos.
    Es el mismo principio que usan los graficadores de árboles genealógicos:
    como cada hoja tiene su propio "carril" horizontal reservado, dos nodos
    nunca pueden terminar en el mismo lugar.
    """
    posiciones = {}
    contador = {"x": 0}

    def recorrer(node_id, profundidad):
        izq = tree_.children_left[node_id]
        der = tree_.children_right[node_id]
        if izq == _tree.TREE_LEAF:  # nodo hoja
            x = contador["x"]
            contador["x"] += 1
            posiciones[node_id] = (x, -profundidad)
            return x
        x_izq = recorrer(izq, profundidad + 1)
        x_der = recorrer(der, profundidad + 1)
        x = (x_izq + x_der) / 2
        posiciones[node_id] = (x, -profundidad)
        return x

    recorrer(0, 0)
    return posiciones


def _etiqueta_nodo(tree_, node_id, nombres_columnas, nombres_clases, es_regresion):
    """Texto CORTO por nodo (a diferencia de plot_tree, que imprime el array
    completo de conteos por clase y por eso las cajas crecen y se solapan)."""
    es_hoja = tree_.children_left[node_id] == _tree.TREE_LEAF
    n = tree_.n_node_samples[node_id]
    impureza = tree_.impurity[node_id]
    lineas = []
    if not es_hoja:
        f = tree_.feature[node_id]
        t = tree_.threshold[node_id]
        lineas.append(f"{nombres_columnas[f]}")
        lineas.append(f"≤ {t:.2f}")
    if es_regresion:
        valor = tree_.value[node_id][0][0]
        lineas.append(f"valor = {valor:.2f}")
        lineas.append(f"mse = {impureza:.2f}")
    else:
        conteos = tree_.value[node_id][0]
        idx_mayoria = int(conteos.argmax())
        clase = nombres_clases[idx_mayoria] if nombres_clases else str(idx_mayoria)
        lineas.append(f"clase: {clase}")
        lineas.append(f"gini = {impureza:.2f}")
    lineas.append(f"n = {n}")
    return "\n".join(lineas), es_hoja


def _dibujar_arbol_en_figura(modelo, nombres_columnas, nombres_clases=None):
    """
    Dibuja el árbol COMPLETO con layout propio (garantiza cero solapamientos)
    y devuelve la figura de matplotlib lista para exportar a SVG/PDF.
    """
    tree_ = modelo.tree_
    es_regresion = nombres_clases is None
    posiciones = _posiciones_arbol(tree_)
    n_hojas = modelo.get_n_leaves()
    profundidad = modelo.get_depth()

    # Espacio horizontal/vertical reservado por nodo, en pulgadas. Al ser
    # vectorial (SVG) el tamaño real de la figura no afecta la nitidez,
    # solo cuánto hay que desplazarse/zoomear para leerlo.
    dx, dy = 1.55, 2.0
    box_w, box_h = dx * 0.86, dy * 0.62
    fontsize = 7.5 if n_hojas <= 80 else (6.5 if n_hojas <= 250 else 5.5)

    ancho = min(max(10, n_hojas * dx), 320)
    alto = min(max(6, (profundidad + 1) * dy), 90)

    fig, ax = plt.subplots(figsize=(ancho, alto))

    # Colores: clasificación -> un color por clase, mezclado con blanco según
    # la pureza del nodo (igual que hace sklearn); regresión -> escala de
    # calor según el valor predicho.
    if not es_regresion:
        n_clases = len(nombres_clases) if nombres_clases else int(tree_.value.shape[2])
        cmap = plt.get_cmap("tab20", max(n_clases, 1))
    else:
        valores_todos = tree_.value[:, 0, 0]
        vmin, vmax = float(valores_todos.min()), float(valores_todos.max())
        cmap_reg = plt.get_cmap("YlOrBr")

    for node_id, (x, y) in posiciones.items():
        etiqueta, es_hoja = _etiqueta_nodo(tree_, node_id, nombres_columnas,
                                            nombres_clases, es_regresion)
        px, py = x * dx, y * dy

        if es_regresion:
            valor = tree_.value[node_id][0][0]
            norm = (valor - vmin) / (vmax - vmin + 1e-9)
            color = cmap_reg(0.15 + 0.7 * norm)
        else:
            conteos = tree_.value[node_id][0]
            total = conteos.sum()
            idx_mayoria = int(conteos.argmax())
            pureza = conteos[idx_mayoria] / total if total > 0 else 0
            base = cmap(idx_mayoria)
            color = tuple(base[k] * pureza + 1.0 * (1 - pureza) for k in range(3)) + (1.0,)

        borde = "#a8552f" if es_hoja else "#555555"
        rect = plt.Rectangle((px - box_w / 2, py - box_h / 2), box_w, box_h,
                              facecolor=color, edgecolor=borde, linewidth=0.9,
                              zorder=2, joinstyle="round")
        ax.add_patch(rect)
        ax.text(px, py, etiqueta, ha="center", va="center", fontsize=fontsize, zorder=3)

    def dibujar_conexiones(node_id):
        izq = tree_.children_left[node_id]
        der = tree_.children_right[node_id]
        if izq == _tree.TREE_LEAF:
            return
        x0, y0 = posiciones[node_id]
        for hijo, texto_borde in [(izq, "Sí"), (der, "No")]:
            x1, y1 = posiciones[hijo]
            px0, py0 = x0 * dx, y0 * dy - box_h / 2
            px1, py1 = x1 * dx, y1 * dy + box_h / 2
            ax.plot([px0, px1], [py0, py1], color="#999999", linewidth=0.9, zorder=1)
            ax.text((px0 + px1) / 2, (py0 + py1) / 2 + 0.05 * dy, texto_borde,
                     fontsize=max(fontsize - 1.5, 5), color="#a8552f", zorder=3,
                     ha="center")
        dibujar_conexiones(izq)
        dibujar_conexiones(der)

    dibujar_conexiones(0)

    ax.set_xlim(-dx * 0.6, (n_hojas - 1) * dx + dx * 0.6)
    ax.set_ylim(-(profundidad + 0.6) * dy, dy * 0.6)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    return fig, n_hojas, profundidad


def mostrar_arbol(modelo, nombres_columnas, nombres_clases=None, titulo="Árbol entrenado",
                   key=""):
    """
    Dibuja el árbol de decisión COMPLETO (sin límite de profundidad) como SVG
    (gráfico VECTORIAL, no una imagen de píxeles), embebido en un visor con
    zoom/pan por mouse o pellizco. Como es vectorial, se puede acercar todo
    lo que se quiera sin que se vea borroso ni pixelado.

    El layout de los nodos es propio (NO usa plot_tree de sklearn), con
    posiciones calculadas matemáticamente para que sea IMPOSIBLE que dos
    cajas se solapen, sin importar cuántas clases o cuántos nodos tenga.

    Cada variable se etiqueta con su posición/índice de columna y su nombre,
    ej. 'X[4] humidity', para que quede clara la ubicación de la variable
    dentro del dataset además de su nombre.
    """
    nombres_con_indice = [f"X[{i}] {col}" for i, col in enumerate(nombres_columnas)]

    with st.spinner("Calculando el layout del árbol (sin solapamientos)..."):
        fig, n_hojas, profundidad = _dibujar_arbol_en_figura(
            modelo, nombres_con_indice, nombres_clases
        )
        st.markdown(f"**{titulo}** — árbol completo: profundidad {profundidad}, {n_hojas} hojas")

        buffer_svg = io.StringIO()
        fig.savefig(buffer_svg, format="svg", bbox_inches="tight")
        svg_code = buffer_svg.getvalue()

        buffer_pdf = io.BytesIO()
        fig.savefig(buffer_pdf, format="pdf", bbox_inches="tight")
        pdf_bytes = buffer_pdf.getvalue()
        plt.close(fig)

    # Visor con zoom (rueda del mouse / pellizco) y arrastre (paneo), usando
    # el propio SVG vectorial: nunca se pixela porque no es una imagen rasterizada.
    visor_html = f"""
    <div id="contenedor-{key}" style="width:100%; height:600px; overflow:hidden;
         border:1px solid #ccc; border-radius:8px; cursor:grab; background:white;">
        <div id="lienzo-{key}" style="transform-origin: 0 0; width:fit-content;">
            {svg_code}
        </div>
    </div>
    <p style="font-size:0.8em; color:#888;">
        🖱️ Rueda del mouse (o pellizco en móvil) = zoom · Clic y arrastrar = mover el árbol
    </p>
    <script>
        (function() {{
            const contenedor = document.getElementById("contenedor-{key}");
            const lienzo = document.getElementById("lienzo-{key}");
            let escala = 1, offsetX = 0, offsetY = 0;
            let arrastrando = false, iniX = 0, iniY = 0;

            function aplicar() {{
                lienzo.style.transform = `translate(${{offsetX}}px, ${{offsetY}}px) scale(${{escala}})`;
            }}

            contenedor.addEventListener("wheel", function(e) {{
                e.preventDefault();
                const delta = e.deltaY < 0 ? 1.12 : 0.89;
                escala = Math.min(Math.max(escala * delta, 0.2), 15);
                aplicar();
            }}, {{ passive: false }});

            contenedor.addEventListener("mousedown", function(e) {{
                arrastrando = true;
                iniX = e.clientX - offsetX; iniY = e.clientY - offsetY;
                contenedor.style.cursor = "grabbing";
            }});
            window.addEventListener("mouseup", function() {{
                arrastrando = false;
                contenedor.style.cursor = "grab";
            }});
            window.addEventListener("mousemove", function(e) {{
                if (!arrastrando) return;
                offsetX = e.clientX - iniX; offsetY = e.clientY - iniY;
                aplicar();
            }});
        }})();
    </script>
    """
    components.html(visor_html, height=630, scrolling=False)

    if n_hojas > 100:
        st.info(f"Este árbol tiene {n_hojas} hojas. Usa la rueda del mouse dentro del "
                "recuadro para acercar — al ser vectorial (SVG), no se pixela sin importar "
                "cuánto te acerques, y el layout garantiza que ninguna caja tape a otra.")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("⬇️ Descargar árbol como SVG (vectorial, zoom perfecto)",
                            data=svg_code, file_name=f"{titulo.replace(' ', '_')}.svg",
                            mime="image/svg+xml", key=f"dl_svg_{key}")
    with col_dl2:
        st.download_button("⬇️ Descargar árbol como PDF (para imprimir/anexar al informe)",
                            data=pdf_bytes, file_name=f"{titulo.replace(' ', '_')}.pdf",
                            mime="application/pdf", key=f"dl_pdf_{key}")

    st.caption("Cada caja muestra: la variable usada (con su posición de columna 'X[i]' y "
               "su nombre), el umbral de división ('Sí' = cumple la condición, 'No' = no la "
               "cumple), la impureza (gini/mse), cuántas muestras caen ahí y la clase/valor "
               "predicho en ese nodo. El espacio de cada nodo está reservado matemáticamente, "
               "así que nunca se monta uno encima de otro.")


# ---------------------------------------------------------------------
# Barra lateral: mapa del proyecto
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Mapa del proyecto")
    st.markdown("""
    1. **Introducción** — ML, sobreajuste, los datos
    2. **Preprocesamiento** — estandarización / escalado
    3. **Árboles de decisión**
    4. **Random Forest**
    5. **Regresión** (árboles + RF)
    6. **KNN**
    7. **Clustering (K-means)**
    """)
    st.markdown("---")
    st.caption("Cada botón entrena el modelo EN VIVO con los datos reales. "
               "Nada está precalculado.")

st.title("Proyecto de Inteligencia Computacional 2")
st.caption("Cultivos (clasificación) · Concreto (regresión) — Árboles, Random Forest, KNN y Clustering")

tabs = st.tabs([
    "📖 Introducción", "⚙️ Preprocesamiento", "🌳 Árboles", "🌲 Random Forest",
    "📈 Regresión", "🔎 KNN", "🧩 Clustering",
])

# =====================================================================
# TAB 0: INTRODUCCIÓN — teoría general + descripción de los datos
# =====================================================================
with tabs[0]:
    st.header("¿Qué es el Machine Learning y por qué importa?")
    st.markdown("""
    El **aprendizaje automático** permite que un programa aprenda patrones a partir de datos
    en vez de que un humano programe reglas explícitas para cada caso. Se divide en tres
    grandes familias:

    - **Supervisado**: el algoritmo recibe ejemplos con su respuesta correcta.
      Aquí caen la **clasificación** (predecir una categoría) y la **regresión** (predecir un número).
    - **No supervisado**: el algoritmo recibe datos sin etiquetas y debe encontrar
      estructura por sí mismo (**clustering**).
    - **Por refuerzo**: aprendizaje por ensayo y error con recompensas (no se usa en este proyecto).

    En este proyecto usamos **dos bases de datos reales** que cubren ambos escenarios supervisados:
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌱 Crop Recommendation")
        st.markdown("**Problema: Clasificación** (22 cultivos posibles)")
    with col2:
        st.subheader("🧱 Concreto")
        st.markdown("**Problema: Regresión** (resistencia en MPa, valor continuo)")

    st.markdown("---")
    st.header("El sobreajuste (overfitting) — el hilo conductor de todo el proyecto")
    with st.expander("📚 Teoría: sobreajuste y subajuste", expanded=True):
        st.markdown("""
        - **Sobreajuste (overfitting):** el modelo memoriza el ruido y los detalles particulares
          del set de entrenamiento, en vez de aprender el patrón general. Síntoma: accuracy muy alta
          en *train*, pero baja en *test*.
        - **Subajuste (underfitting):** el modelo es demasiado simple para capturar los patrones
          reales; rinde mal tanto en *train* como en *test*.
        - **¿Cómo se detecta?** Siempre comparando una métrica en datos de entrenamiento vs. datos
          de prueba (que el modelo nunca vio).

        | Modelo | Causa típica de sobreajuste | Cómo se controla |
        |---|---|---|
        | Árbol de decisión | Crece hasta separar cada dato individual | Poda (pre/post) |
        | Random Forest | Árboles muy profundos sin variedad | Bootstrap + variables aleatorias |
        | KNN | K muy bajo (K=1 memoriza cada punto) | Elegir K óptimo por validación |
        | K-means | K igual al número de puntos | Elegir K con codo/silueta/Davies-Bouldin |

        A lo largo de esta interfaz, cada sección muestra la comparación *train vs. test*
        para que puedas señalar en vivo si hay sobreajuste, subajuste o buen ajuste.
        """)

    st.markdown("---")
    st.header("Tipos de variable (repaso estadístico)")
    with st.expander("📚 Teoría: cualitativas vs. cuantitativas, discretas vs. continuas", expanded=True):
        st.markdown("""
        Antes de describir cada base de datos, conviene repasar cómo se clasifican
        estadísticamente las variables:

        - **Cualitativas (categóricas):** describen una cualidad o categoría, no un número.
          - *Nominales*: no tienen un orden natural (ej. el cultivo: "arroz", "café", "maíz"...).
          - *Ordinales*: sí tienen un orden (ej. "bajo / medio / alto"). No aparecen en
            nuestros datasets, pero es bueno mencionarlas para completar la clasificación.
        - **Cuantitativas (numéricas):** representan una cantidad medible.
          - *Discretas*: solo toman valores enteros, contables uno por uno
            (ej. cantidad de días, número de nutrientes en unidades enteras).
          - *Continuas*: pueden tomar cualquier valor dentro de un rango, con decimales
            (ej. temperatura, humedad, resistencia del concreto).

        El **tipo de dato en pandas** (`int64`, `float64`, `object`) es una pista, pero no
        define por sí solo la categoría estadística: por ejemplo, una variable guardada
        como entero (`int64`) puede ser cuantitativa discreta (un conteo) — como pasa con
        `age` en el dataset de concreto, medida en días completos.
        """)

    st.header("Descripción de las bases de datos")

    subtab1, subtab2 = st.tabs(["🌱 Crop_recommendation.csv", "🧱 concreto.csv"])

    with subtab1:
        X_crop, y_crop, df_crop = cargar_crop()
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros", len(df_crop))
        c2.metric("Variables", X_crop.shape[1])
        c3.metric("Clases", y_crop.nunique())

        st.markdown("""
        | Variable | Descripción | Rango aprox. |
        |---|---|---|
        | N, P, K | Nitrógeno, Fósforo, Potasio en el suelo | 5 – 205 |
        | temperature | Temperatura ambiente (°C) | 8 – 44 |
        | humidity | Humedad relativa (%) | 14 – 100 |
        | ph | pH del suelo | 3.5 – 9.9 |
        | rainfall | Precipitación (mm) | 20 – 298 |
        | **label** | Cultivo recomendado (objetivo) | 22 clases |
        """)

        st.markdown("**Tipo de dato y clasificación estadística de cada variable:**")
        tipo_estadistico_crop = {
            "N": "Cuantitativa discreta", "P": "Cuantitativa discreta", "K": "Cuantitativa discreta",
            "temperature": "Cuantitativa continua", "humidity": "Cuantitativa continua",
            "ph": "Cuantitativa continua", "rainfall": "Cuantitativa continua",
            "label": "Cualitativa nominal",
        }
        tipos_crop = pd.DataFrame({
            "Variable": df_crop.columns,
            "Tipo de dato (pandas)": df_crop.dtypes.astype(str).values,
            "Clasificación estadística": [tipo_estadistico_crop[c] for c in df_crop.columns],
        })
        st.dataframe(tipos_crop, hide_index=True, use_container_width=True)
        st.caption("N, P y K se miden en unidades enteras de nutriente (discretas); temperatura, "
                   "humedad, pH y lluvia pueden tomar cualquier valor decimal (continuas); "
                   "`label` es la variable objetivo, categórica sin orden (nominal) — por eso "
                   "este dataset es de **clasificación**.")

        st.dataframe(df_crop.head(6))

        st.subheader("Distribución de clases (balance del dataset)")
        st.bar_chart(y_crop.value_counts())
        st.info("El dataset está **perfectamente balanceado**: exactamente 100 muestras por "
                "cada una de las 22 clases. Esto evita que el modelo se sesgue hacia una clase "
                "mayoritaria, así que no hace falta aplicar técnicas de balanceo.")

    with subtab2:
        X_conc, y_conc, df_conc = cargar_concreto()
        c1, c2 = st.columns(2)
        c1.metric("Registros", len(df_conc))
        c2.metric("Variables", X_conc.shape[1])

        st.markdown("""
        | Variable | Descripción | Unidad |
        |---|---|---|
        | cement, slag, flyash | Componentes cementantes de la mezcla | kg/m³ |
        | water | Agua | kg/m³ |
        | superplasticizer | Aditivo plastificante | kg/m³ |
        | coarseaggregate, fineaggregate | Agregado grueso / fino | kg/m³ |
        | age | Edad de curado | días |
        | **csMPa** | Resistencia a la compresión (objetivo) | MPa |
        """)

        st.markdown("**Tipo de dato y clasificación estadística de cada variable:**")
        tipo_estadistico_conc = {
            "cement": "Cuantitativa continua", "slag": "Cuantitativa continua",
            "flyash": "Cuantitativa continua", "water": "Cuantitativa continua",
            "superplasticizer": "Cuantitativa continua",
            "coarseaggregate": "Cuantitativa continua", "fineaggregate": "Cuantitativa continua",
            "age": "Cuantitativa discreta", "csMPa": "Cuantitativa continua",
        }
        tipos_conc = pd.DataFrame({
            "Variable": df_conc.columns,
            "Tipo de dato (pandas)": df_conc.dtypes.astype(str).values,
            "Clasificación estadística": [tipo_estadistico_conc[c] for c in df_conc.columns],
        })
        st.dataframe(tipos_conc, hide_index=True, use_container_width=True)
        st.caption("Todos los componentes de la mezcla (cemento, agua, aditivos, agregados) y "
                   "la resistencia `csMPa` son cuantitativos continuos (cualquier valor decimal); "
                   "`age` es la única discreta, porque se cuenta en días completos (1, 3, 7, 28...). "
                   "No hay variables cualitativas: por eso este dataset es de **regresión**, no de clasificación.")

        st.dataframe(df_conc.head(6))

        st.subheader("Distribución de la variable objetivo (csMPa)")
        fig = go.Figure(go.Histogram(
            x=y_conc, nbinsx=25, marker=dict(color="#a8552f", line=dict(color="white", width=1)),
            hovertemplate="csMPa: %{x}<br>Frecuencia: %{y}<extra></extra>"))
        fig.update_layout(template="plotly_white", height=380, bargap=0.02,
                          xaxis_title="csMPa", yaxis_title="Frecuencia",
                          margin=dict(t=30, b=40))
        st.plotly_chart(fig, use_container_width=True)
        st.info("Al no tener clases discretas, este dataset es el que usamos para toda "
                "la parte de **regresión**.")

    st.markdown("---")
    st.header("¿Cómo sabremos si un modelo fue efectivo? (métricas)")
    with st.expander("📚 Teoría: métricas de evaluación", expanded=True):
        colm1, colm2 = st.columns(2)
        with colm1:
            st.markdown("""
            **Clasificación (cultivos):**

            - **Accuracy** — ¿Cuántas predicciones fueron correctas (sobre el total)?
            - **Precision** — Cuando el modelo predijo una clase, ¿cuántas veces acertó?
            - **Recall** — De todos los casos que REALMENTE eran de una clase, ¿cuántos encontró?
            - **F1-score** — Combina precision y recall en un solo número.
            - **Matriz de confusión** — Permite ver exactamente qué clases está confundiendo el modelo.
            """)
        with colm2:
            st.markdown("""
            **Regresión (concreto):**
            - *MAE*: error absoluto promedio (mismas unidades que csMPa).
            - *RMSE*: como el MAE, pero penaliza más los errores grandes.
            - *R²*: qué proporción de la variación logra explicar el modelo.
            """)

    st.subheader("Ejemplo de matriz de confusión, interpretado paso a paso")
    st.caption("En vez de un ejemplo inventado, esto entrena un mini-clasificador real usando "
               "3 cultivos de tu dataset que SÍ se confunden entre sí (leguminosas con "
               "condiciones de suelo parecidas), para que la interpretación tenga sentido.")

    if st.button("🎯 Generar ejemplo real con Blackgram / Lentil / Mothbeans"):
        X, y, df = cargar_crop()
        cultivos_ejemplo = ["blackgram", "lentil", "mothbeans"]
        nombres_es = {"blackgram": "Blackgram", "lentil": "Lentil", "mothbeans": "Mothbeans"}

        mascara = y.isin(cultivos_ejemplo)
        X_sub, y_sub = X[mascara], y[mascara]
        Xtr, Xte, ytr, yte = dividir_datos(X_sub, y_sub, test_size=0.3, estratificar=True)

        modelo_ej = DecisionTreeClassifier(random_state=42, max_depth=5)
        modelo_ej.fit(Xtr, ytr)
        pred_ej = modelo_ej.predict(Xte)

        cm_ej = confusion_matrix(yte, pred_ej, labels=cultivos_ejemplo)
        cm_df = pd.DataFrame(cm_ej,
                              index=[f"Real: {nombres_es[c]}" for c in cultivos_ejemplo],
                              columns=[f"Predijo: {nombres_es[c]}" for c in cultivos_ejemplo])

        # Gráfico interactivo con Plotly
        labels_es = [nombres_es[c] for c in cultivos_ejemplo]
        fig_cm = grafico_confusion_matrix_interactivo(cm_ej, labels_es,
                                                       titulo="Matriz de Confusión - Ejemplo (INTERACTIVA)")
        st.plotly_chart(fig_cm, use_container_width=True)
        st.dataframe(cm_df, use_container_width=True)

        st.markdown("**¿Cómo se interpreta?**")
        for i, cultivo_real in enumerate(cultivos_ejemplo):
            total_fila = cm_ej[i].sum()
            texto = f"**Real = {nombres_es[cultivo_real]}** ({total_fila} casos de prueba): "
            partes = []
            for j, cultivo_pred in enumerate(cultivos_ejemplo):
                cantidad = cm_ej[i, j]
                if cantidad > 0:
                    partes.append(f"{cantidad} → predijo {nombres_es[cultivo_pred]}"
                                  + (" ✔" if i == j else " ✗"))
            st.markdown(texto + " · ".join(partes))

        acc_ej = accuracy_score(yte, pred_ej)
        st.success(f"Con estos 3 cultivos, el árbol acertó {acc_ej:.1%} de las veces en total "
                   "(diagonal de la matriz / total de casos).")

        mostrar_reporte_clasificacion(yte, pred_ej, "Precision, recall y F1 de este ejemplo")

# =====================================================================
# TAB 1: PREPROCESAMIENTO
# =====================================================================
with tabs[1]:
    st.header("Preprocesamiento: estandarización y escalado")
    with st.expander("📚 Teoría: ¿por qué preprocesar?", expanded=True):
        st.markdown("""
        | Técnica | Fórmula | Cuándo se usa |
        |---|---|---|
        | **Estandarización (Z-score)** | `z = (x - media) / desviación` | Modelos basados en distancias: **KNN, K-means** |
        | **Escalado Min-Max** | `x' = (x - min) / (max - min)` | Cuando se quiere conservar la forma exacta de la distribución |

        **Importante:** los **árboles de decisión y Random Forest NO necesitan escalado**,
        porque comparan un valor contra un umbral variable por variable — la escala no afecta
        esa comparación. En cambio, **KNN y K-means sí son sensibles a la escala** porque
        calculan distancias entre puntos: si `rainfall` llega a 298 y `ph` no pasa de 10, la
        distancia quedaría dominada por `rainfall` aunque `ph` sea igual de importante.
        """)

    st.subheader("Compruébalo en vivo con el dataset de cultivos")
    st.caption("Aquí se aplican los cambios REALES sobre los datos, para que veas exactamente "
               "qué transforma cada técnica, fila por fila y variable por variable.")

    if st.button("Aplicar y comparar preprocesamiento"):
        X, y, df = cargar_crop()
        scaler_z = StandardScaler()
        X_std = pd.DataFrame(scaler_z.fit_transform(X), columns=X.columns)

        from sklearn.preprocessing import MinMaxScaler
        scaler_mm = MinMaxScaler()
        X_mm = pd.DataFrame(scaler_mm.fit_transform(X), columns=X.columns)

        st.markdown("### 1) ¿Qué le pasa a cada VARIABLE? (estadísticas antes/después)")
        resumen = pd.concat({
            "Original": X.describe().loc[["min", "max", "mean", "std"]],
            "Estandarizado (Z-score)": X_std.describe().loc[["min", "max", "mean", "std"]],
            "Min-Max [0,1]": X_mm.describe().loc[["min", "max", "mean", "std"]],
        }, axis=0).round(3)
        st.dataframe(resumen, use_container_width=True)
        st.markdown("""
        **Qué cambió exactamente:**
        - **Estandarización:** cada variable pasó a tener media ≈ **0** y desviación ≈ **1**,
          sin importar su escala original (`rainfall` iba de 20 a 298; ahora sus valores
          quedan casi todos entre -2 y +3).
        - **Min-Max:** cada variable quedó comprimida exactamente entre **0 y 1**, conservando
          la forma de la distribución original (el valor mínimo real ahora es 0, el máximo es 1).
        - **Ningún método cambia el orden de los datos ni las relaciones entre variables** —
          solo cambia la escala en la que se miden.
        """)

        st.markdown("### 2) ¿Qué le pasa a cada REGISTRO? (mismas 5 filas, 3 versiones)")
        idx_muestra = X.index[:5]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Original**")
            st.dataframe(X.loc[idx_muestra].round(2), use_container_width=True)
        with col2:
            st.markdown("**Estandarizado**")
            st.dataframe(X_std.loc[idx_muestra].round(2), use_container_width=True)
        with col3:
            st.markdown("**Min-Max**")
            st.dataframe(X_mm.loc[idx_muestra].round(2), use_container_width=True)
        st.caption("Compara, por ejemplo, la columna `rainfall`: el mismo registro tiene un "
                   "valor grande en la versión original, cercano a 0/1 en Min-Max, y puede "
                   "ser positivo o negativo en la estandarizada según si está por encima o "
                   "por debajo del promedio.")

    # -----------------------------------------------------------------
    # Análisis de relevancia de variables (Mutual Information)
    # Idea tomada del proyecto de referencia, con umbrales normalizados.
    # -----------------------------------------------------------------
    st.markdown("---")
    st.subheader("Análisis de relevancia de variables (Mutual Information)")
    with st.expander("📚 Teoría: qué mide la Mutual Information (MI) y para qué sirve", expanded=True):
        st.markdown("""
        Antes de entrenar, conviene saber **cuánta información aporta cada variable** para
        predecir el objetivo. La **Mutual Information** mide cuánto se reduce la
        incertidumbre (entropía) sobre el objetivo al conocer una variable:

        `MI = Entropía(objetivo) − Entropía(objetivo | variable)`

        - **MI = 0** → la variable no aporta nada.
        - A diferencia de la correlación de Pearson (solo relaciones **lineales**), la MI
          detecta cualquier tipo de relación, lineal o no.
        - Se calcula **antes** de modelar y **no depende de ningún algoritmo**, así que sirve
          para contrastar después con la importancia de variables del Random Forest.

        **Cómo interpretarla aquí:** en clasificación, la MI máxima posible es la entropía
        del objetivo = `ln(nº de clases)`. Con 22 cultivos eso es ≈ 3.09, por eso se muestra
        también como **% de esa incertidumbre que la variable elimina** (un umbral fijo como
        0.40 no sirve para comparar problemas con distinto número de clases). En regresión no
        hay tope fijo, así que se compara de forma relativa entre variables.
        """)

    ds_mi = st.radio("Dataset a analizar", ["Cultivos (clasificación)", "Concreto (regresión)"],
                     horizontal=True, key="mi_dataset")
    if st.button("📊 Calcular Mutual Information", key="btn_mi"):
        with st.spinner("Calculando MI (usa vecinos más cercanos, puede tardar unos segundos)..."):
            if ds_mi.startswith("Cultivos"):
                X_mi, y_mi, _ = cargar_crop()
                y_enc_mi = LabelEncoder().fit_transform(y_mi.astype(str))
                mi_vals = mutual_info_classif(X_mi, y_enc_mi, random_state=42)
                h_max = float(np.log(len(np.unique(y_enc_mi))))
            else:
                X_mi, y_mi, _ = cargar_concreto()
                mi_vals = mutual_info_regression(X_mi, y_mi, random_state=42)
                h_max = None
        mi_serie = pd.Series(mi_vals, index=X_mi.columns).sort_values(ascending=False)

        tabla_mi = pd.DataFrame({"Variable": mi_serie.index, "MI (nats)": mi_serie.values.round(4)})
        if h_max is not None:
            tabla_mi["% de incertidumbre eliminada"] = (mi_serie.values / h_max * 100).round(1)
        else:
            tabla_mi["% respecto a la mejor variable"] = (mi_serie.values / mi_serie.max() * 100).round(1)
        st.dataframe(tabla_mi, hide_index=True, use_container_width=True)

        # Gráfico interactivo con Plotly (pasa mouse para ver valores)
        import plotly.graph_objects as go

        fig_mi = go.Figure()

        # Crear hover text con valores exactos
        hover_text = [f"<b>{var}</b><br>MI: {val:.4f}<br>% relativo: {(val/mi_serie.max()*100):.1f}%"
                      for var, val in zip(mi_serie.index[::-1], mi_serie.values[::-1])]

        fig_mi.add_trace(go.Bar(
            y=mi_serie.index[::-1],
            x=mi_serie.values[::-1],
            orientation='h',
            marker=dict(color="#3f7d54"),
            hovertext=hover_text,
            hoverinfo="text",
            text=[f"{v:.4f}" for v in mi_serie.values[::-1]],
            textposition="outside"
        ))

        if h_max is not None:
            fig_mi.add_vline(x=h_max, line_dash="dash", line_color="red",
                            annotation_text=f"Max posible = {h_max:.2f}",
                            annotation_position="top right")
            x_max = h_max * 1.2
        else:
            x_max = mi_serie.max() * 1.3

        fig_mi.update_layout(
            title=f"Relevancia de cada variable — {ds_mi} (INTERACTIVO: pasa mouse)",
            xaxis_title="Mutual Information (nats)",
            yaxis_title="Variable",
            template="plotly_white",
            height=500,
            hovermode="y",
            xaxis=dict(range=[0, x_max])
        )

        st.plotly_chart(fig_mi, use_container_width=True)
        st.caption("Barras más largas = la variable reduce más la incertidumbre sobre el "
                   "objetivo. Compara este orden con el gráfico de 'Importancia de variables' "
                   "del Random Forest: si coinciden, hay evidencia de que esas variables "
                   "realmente son las más informativas, con dos métodos independientes.")

    with st.expander("📚 Otros pasos de preparación que aparecen en otros datasets (no aplican a estos dos)"):
        st.markdown("""
        - **Fuga de datos (data leakage):** una variable que ya contiene, directa o
          indirectamente, la respuesta (ej. predecir 'aprobó' usando 'nota final'). Hace que
          el modelo parezca perfecto pero no sirva en la realidad; se **excluye** antes de
          entrenar. Ninguna variable de cultivos ni de concreto lo hace.
        - **Codificación One-Hot:** convierte variables categóricas (texto) en columnas 0/1
          para que el modelo las use (`pd.get_dummies`). Aquí no hace falta: todas las
          variables predictoras ya son numéricas (la etiqueta `label` se codifica aparte).
        - **Desbalance de clases:** si una clase tiene muchas más muestras que otra, se puede
          usar `class_weight='balanced'` o *undersampling* (recortar la clase grande, solo en
          entrenamiento). Nuestro dataset de cultivos ya está balanceado (100 por clase, ratio
          1:1), por eso no se aplica.
        - **Orden correcto con el escalado:** primero dividir en train/test y **después**
          ajustar el escalador solo con train. Ajustarlo con todos los datos antes de dividir
          filtra información del test hacia el entrenamiento.
        """)

# =====================================================================
# TAB 2: ÁRBOLES DE DECISIÓN
# =====================================================================
with tabs[2]:
    st.header("Árbol de decisión — Clasificación de cultivos")
    with st.expander("📚 Teoría: cómo decide un árbol y qué es la poda", expanded=True):
        st.markdown("""
        Un árbol de decisión aprende preguntas tipo *"¿la variable X es mayor que un umbral?"*
        que van dividiendo los datos en grupos cada vez más puros. Para elegir dónde dividir,
        usa medidas de impureza: **Gini** o **Entropía / ganancia de información**.

        **Podas para controlar el sobreajuste:**
        - **Pre-poda**: limitar el crecimiento *antes* de entrenar (`max_depth`, `min_samples_leaf`).
        - **Post-poda** (`ccp_alpha`): dejar crecer el árbol completo y luego recortar las ramas
          que menos aportan.
        """)

    c1, c2, c3 = st.columns(3)
    with c1:
        sin_limite = st.checkbox("Sin restricción (árbol completo)", value=False)
    with c2:
        max_depth = st.slider("max_depth", 1, 20, 4, disabled=sin_limite)
    with c3:
        criterio = st.selectbox("Criterio de división", ["gini", "entropy"])
    test_size = st.slider("Proporción de test", 0.1, 0.4, 0.2, key="arbol_test")

    if st.button("🌳 Entrenar árbol de decisión", type="primary"):
        X, y, _ = cargar_crop()
        X_train, X_test, y_train, y_test = dividir_datos(X, y, test_size=test_size, estratificar=True)

        with st.spinner("Entrenando árbol con los datos reales..."):
            modelo = DecisionTreeClassifier(
                max_depth=None if sin_limite else max_depth,
                criterion=criterio, random_state=42,
            )
            inicio = time.time()
            modelo.fit(X_train, y_train)
            tiempo = time.time() - inicio
            pred_train = modelo.predict(X_train)
            pred_test = modelo.predict(X_test)

        acc_train = accuracy_score(y_train, pred_train)
        acc_test = accuracy_score(y_test, pred_test)

        st.markdown(f"**Partición de datos:** {len(X_train)} registros de entrenamiento "
                    f"({(1-test_size):.0%}) · {len(X_test)} registros de prueba ({test_size:.0%})")

        c1, c2, c3 = st.columns(3)
        c1.metric("Tiempo entrenamiento", f"{tiempo:.4f} s")
        c2.metric("Accuracy train", f"{acc_train:.2%}")
        c3.metric("Accuracy test", f"{acc_test:.2%}")

        diff = acc_train - acc_test
        if diff > 0.08:
            st.warning(f"⚠ Posible **sobreajuste**: diferencia train-test de {diff:.2%}.")
        elif acc_test < 0.6:
            st.warning("⚠ Posible **subajuste**: el árbol es demasiado simple para 22 clases.")
        else:
            st.success(f"✔ Buen ajuste (diferencia train-test de {diff:.2%}).")

        clases = sorted(y.unique())
        cm = confusion_matrix(y_test, pred_test, labels=clases)

        # Gráfico interactivo con Plotly
        fig_cm = grafico_confusion_matrix_interactivo(cm, [str(c) for c in clases],
                                                       titulo="Matriz de Confusión - Árbol de Decisión (INTERACTIVA)",
                                                       color="Oranges")
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("La matriz de confusión muestra en qué cultivos se equivoca el árbol "
                   "(normalmente entre leguminosas con requerimientos de suelo parecidos).")

        mostrar_reporte_clasificacion(y_test, pred_test, "Precisión, recall y F1 por cultivo")

        mostrar_arbol(modelo, X.columns.tolist(), [str(c) for c in modelo.classes_],
                      titulo="Árbol de decisión completo (cultivos)", key="arbol_clas")

    st.markdown("---")
    st.subheader("Comparar los 3 modelos de la rúbrica (punto 4a)")
    with st.expander("📚 ¿Qué compara este botón?"):
        st.markdown("""
        Entrena las **3 formas distintas de mandarle la base de datos** al árbol,
        tal como pide la rúbrica, y las compara en una sola tabla:
        - **Modelo 1:** partición 80/20, sin restricciones (árbol completo).
        - **Modelo 2:** partición 80/20, pre-poda (`max_depth=4`, `min_samples_leaf=5`).
        - **Modelo 3:** partición 70/30, criterio de entropía.
        """)

    if st.button("📊 Comparar los 3 modelos", key="comparar_arboles"):
        X, y, _ = cargar_crop()

        configs = [
            ("Modelo 1: Sin restricciones (80/20)", 0.2,
             DecisionTreeClassifier(random_state=42)),
            ("Modelo 2: Pre-poda max_depth=4 (80/20)", 0.2,
             DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)),
            ("Modelo 3: Partición 70/30 + entropía", 0.3,
             DecisionTreeClassifier(criterion="entropy", random_state=42)),
        ]

        filas = []
        with st.spinner("Entrenando los 3 modelos..."):
            for nombre, ts, modelo in configs:
                Xtr, Xte, ytr, yte = dividir_datos(X, y, test_size=ts, estratificar=True)
                inicio = time.time()
                modelo.fit(Xtr, ytr)
                tiempo_m = time.time() - inicio
                acc_tr = accuracy_score(ytr, modelo.predict(Xtr))
                acc_te = accuracy_score(yte, modelo.predict(Xte))
                filas.append({
                    "Modelo": nombre,
                    "% Train": f"{(1-ts):.0%}",
                    "% Test": f"{ts:.0%}",
                    "Registros train": len(Xtr),
                    "Registros test": len(Xte),
                    "Tiempo (s)": round(tiempo_m, 4),
                    "Accuracy train": f"{acc_tr:.2%}",
                    "Accuracy test": f"{acc_te:.2%}",
                })

        st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
        st.caption("Compara directamente cómo cambia el desempeño y el tiempo según cómo "
                   "se le entregan los datos al árbol: partición, restricciones y criterio.")

    st.markdown("---")
    st.subheader("Post-poda: efecto del `ccp_alpha`")
    with st.expander("📚 ¿Qué hace este botón?"):
        st.markdown("Entrena un árbol completo, calcula distintos niveles de poda "
                    "(`ccp_alpha`) y grafica cómo cambia el accuracy en train y test "
                    "a medida que se poda más el árbol.")

    if st.button("✂️ Ejecutar análisis de post-poda"):
        X, y, _ = cargar_crop()
        X_train, X_test, y_train, y_test = dividir_datos(X, y, test_size=0.2, estratificar=True)

        with st.spinner("Calculando ruta de poda y validación cruzada para cada alpha..."):
            arbol_completo = DecisionTreeClassifier(random_state=42)
            path = arbol_completo.cost_complexity_pruning_path(X_train, y_train)
            alphas = path.ccp_alphas[:: max(1, len(path.ccp_alphas) // 25)]
            acc_tr, acc_te, acc_cv = [], [], []
            for a in alphas:
                m = DecisionTreeClassifier(random_state=42, ccp_alpha=a)
                # El alpha se elige con validación cruzada sobre el TRAIN (no con el test)
                acc_cv.append(cross_val_score(m, X_train, y_train, cv=5).mean())
                m.fit(X_train, y_train)
                acc_tr.append(accuracy_score(y_train, m.predict(X_train)))
                acc_te.append(accuracy_score(y_test, m.predict(X_test)))

        fig = figura_curva_poda(alphas, acc_tr, acc_te, "Accuracy",
                                "Efecto de la post-poda en accuracy (train vs test)",
                                metrica_cv=acc_cv)
        st.plotly_chart(fig, use_container_width=True)

        mejor = int(np.argmax(acc_cv))
        st.success(f"Mejor `ccp_alpha` (elegido por validación cruzada 5-fold) = {alphas[mejor]:.6f} → "
                   f"Accuracy CV = {acc_cv[mejor]:.2%}, train = {acc_tr[mejor]:.2%}, "
                   f"test = {acc_te[mejor]:.2%}")
        st.caption("El test NO se usa para elegir el alpha: elegirlo con el test sería ajustar un "
                   "hiperparámetro con los datos de evaluación y daría un resultado optimista. "
                   "El test solo reporta cómo le va al alpha ya elegido.")

    st.markdown("---")
    st.subheader("🔬 Ver un sobreajuste REAL: sobreajustado vs. bien ajustado vs. subajustado")
    with st.expander("📚 ¿Por qué hace falta esto?", expanded=True):
        st.markdown("""
        Cuando entrenas el árbol sin restricciones sobre `Crop_recommendation.csv` normal,
        casi no se ve sobreajuste (train ≈ test) — **no porque el árbol sin restricciones
        no pueda sobreajustar, sino porque el dataset es limpio y las clases están muy bien
        separadas**: no hay ruido que el árbol pueda "memorizar de más".

        Para ver un sobreajuste de verdad, aquí se **inyecta ruido artificial**: a un
        porcentaje de las etiquetas de ENTRENAMIENTO se les asigna al azar un cultivo
        incorrecto (el set de prueba se deja limpio, tal como sería en la realidad). Con
        ese ruido:
        - Un árbol **sin restricciones** va a esforzarse por clasificar también los casos
          ruidosos → memoriza el ruido → accuracy en train muy alta, pero cae en test.
        - Un árbol **podado/con profundidad limitada razonable** va a ignorar ese ruido
          como si fueran casos aislados → generaliza mejor.
        - Un árbol **demasiado simple** (profundidad 1-2) ni siquiera aprende el patrón
          real → rinde mal en ambos.

        También se entrena un **Random Forest** con los mismos datos ruidosos, para mostrar
        por qué el ensamble resiste mejor el ruido que un solo árbol (cada árbol del bosque
        ve una muestra distinta, así que el ruido no se repite igual en todos).
        """)

    c1, c2 = st.columns(2)
    with c1:
        pct_ruido = st.slider("% de etiquetas de entrenamiento con ruido (al azar)",
                               0, 50, 25, step=5)
    with c2:
        semilla_ruido = st.number_input("Semilla aleatoria (cámbiala para otro ruido)",
                                         0, 999, 42)

    if st.button("🧪 Generar comparación con ruido", type="primary"):
        X, y, _ = cargar_crop()
        Xtr, Xte, ytr, yte = dividir_datos(X, y, test_size=0.2, estratificar=True)

        # Inyectar ruido SOLO en las etiquetas de entrenamiento
        rng = np.random.RandomState(semilla_ruido)
        ytr_ruidoso = ytr.copy().reset_index(drop=True)
        Xtr_reset = Xtr.reset_index(drop=True)
        clases_todas = sorted(y.unique())
        n_ruido = int(len(ytr_ruidoso) * pct_ruido / 100)
        idx_ruido = rng.choice(len(ytr_ruidoso), size=n_ruido, replace=False)
        for idx in idx_ruido:
            otras = [c for c in clases_todas if c != ytr_ruidoso[idx]]
            ytr_ruidoso[idx] = rng.choice(otras)

        st.caption(f"Se corrompieron {n_ruido} de {len(ytr_ruidoso)} etiquetas de "
                   f"entrenamiento ({pct_ruido}%). El set de prueba ({len(Xte)} registros) "
                   f"sigue limpio.")

        configs = [
            ("🔴 Sobreajustado (sin restricción)", DecisionTreeClassifier(random_state=42)),
            ("🟢 Bien ajustado (podado con CV)", None),  # se calcula abajo
            ("🟡 Subajustado (max_depth=1)", DecisionTreeClassifier(max_depth=1, random_state=42)),
        ]

        # Para el "bien ajustado": elegir ccp_alpha por validación cruzada SOBRE
        # los datos de entrenamiento (ruidosos), sin mirar el set de prueba —
        # así es una elección legítima, no trampa.
        arbol_ref = DecisionTreeClassifier(random_state=42)
        path = arbol_ref.cost_complexity_pruning_path(Xtr_reset, ytr_ruidoso)
        alphas_cand = path.ccp_alphas[:: max(1, len(path.ccp_alphas) // 15)]
        mejores_cv = []
        with st.spinner("Eligiendo la mejor poda por validación cruzada..."):
            for a in alphas_cand:
                m = DecisionTreeClassifier(random_state=42, ccp_alpha=a)
                score = cross_val_score(m, Xtr_reset, ytr_ruidoso, cv=3).mean()
                mejores_cv.append(score)
        mejor_alpha = alphas_cand[int(np.argmax(mejores_cv))]
        configs[1] = ("🟢 Bien ajustado (podado con CV)",
                       DecisionTreeClassifier(random_state=42, ccp_alpha=mejor_alpha))

        filas = []
        modelos_entrenados = []
        with st.spinner("Entrenando los 3 árboles sobre datos con ruido..."):
            for nombre, modelo in configs:
                modelo.fit(Xtr_reset, ytr_ruidoso)
                acc_tr = accuracy_score(ytr_ruidoso, modelo.predict(Xtr_reset))
                acc_te = accuracy_score(yte, modelo.predict(Xte))
                filas.append({
                    "Modelo": nombre, "Accuracy train (con ruido)": f"{acc_tr:.2%}",
                    "Accuracy test (limpio)": f"{acc_te:.2%}",
                    "Diferencia (gap)": f"{acc_tr - acc_te:+.2%}",
                    "Hojas": modelo.get_n_leaves(), "Profundidad": modelo.get_depth(),
                })
                modelos_entrenados.append((nombre, modelo))

        # Random Forest sobre los MISMOS datos ruidosos, para comparar
        with st.spinner("Entrenando Random Forest sobre los mismos datos ruidosos..."):
            rf_ruido = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            rf_ruido.fit(Xtr_reset, ytr_ruidoso)
            acc_tr_rf = accuracy_score(ytr_ruidoso, rf_ruido.predict(Xtr_reset))
            acc_te_rf = accuracy_score(yte, rf_ruido.predict(Xte))
            filas.append({
                "Modelo": "🌲 Random Forest (200 árboles, sin restricción)",
                "Accuracy train (con ruido)": f"{acc_tr_rf:.2%}",
                "Accuracy test (limpio)": f"{acc_te_rf:.2%}",
                "Diferencia (gap)": f"{acc_tr_rf - acc_te_rf:+.2%}",
                "Hojas": "—", "Profundidad": "—",
            })

        st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)

        nombres_barras = [f["Modelo"] for f in filas]
        nombres_cortos = [n.split(" ", 1)[1] if " " in n else n for n in nombres_barras]
        acc_train_vals = [float(f["Accuracy train (con ruido)"].strip('%'))/100 for f in filas]
        acc_test_vals = [float(f["Accuracy test (limpio)"].strip('%'))/100 for f in filas]
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(x=nombres_cortos, y=acc_train_vals, name="Train (con ruido)",
                                 marker_color="#a8552f",
                                 hovertemplate="%{x}<br>Train: %{y:.2%}<extra></extra>"))
        fig_cmp.add_trace(go.Bar(x=nombres_cortos, y=acc_test_vals, name="Test (limpio)",
                                 marker_color="#3f7d54",
                                 hovertemplate="%{x}<br>Test: %{y:.2%}<extra></extra>"))
        fig_cmp.update_layout(barmode="group", template="plotly_white", height=460,
                              title=f"Efecto del {pct_ruido}% de ruido en cada estrategia",
                              yaxis=dict(title="Accuracy", tickformat=".0%"),
                              xaxis=dict(tickangle=-20), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.markdown("""
        **Cómo leerlo:** entre más grande la barra café (train) comparada con la verde
        (test), más está memorizando ruido en vez de generalizar — ese es el sobreajuste.
        El Random Forest, aunque cada uno de sus 200 árboles individuales crece sin
        restricción (como el "sobreajustado"), termina con un gap mucho menor: promediar
        muchos árboles entrenados con submuestras distintas diluye el efecto del ruido.
        """)

        st.markdown("### Los 3 árboles (completos, sin solapamientos)")
        for nombre, modelo in modelos_entrenados:
            mostrar_arbol(modelo, X.columns.tolist(), [str(c) for c in modelo.classes_],
                          titulo=f"{nombre} — con {pct_ruido}% de ruido",
                          key=f"ruido_{nombre[:4]}_{pct_ruido}")

# =====================================================================
# TAB 3: RANDOM FOREST
# =====================================================================
with tabs[3]:
    st.header("Random Forest — Clasificación de cultivos")
    with st.expander("📚 Teoría: bagging, empates y bootstrap", expanded=True):
        st.markdown("""
        Un Random Forest entrena **muchos árboles** de forma ligeramente distinta y decide
        por **votación mayoritaria**.

        - **¿Por qué números impares de árboles (9, 49, 491)?** Con un número par, dos clases
          podrían empatar exactamente en votos. Con un problema de más de 2 clases (como
          nuestros 22 cultivos), el número impar reduce pero no elimina del todo el riesgo de
          empate entre dos clases minoritarias.
        - **Regla de desempate:** en ese caso, se puede usar la probabilidad promedio
          (`predict_proba`) del bosque en vez del conteo de votos duros.
        - **Bootstrap:** cada árbol se entrena con una muestra aleatoria **con reemplazo** del
          set de entrenamiento, y en cada división solo se consideran algunas variables al azar.
          Por eso cada árbol termina siendo distinto, aunque vengan de la misma base de datos.
        """)

    n_arboles = st.select_slider("Número de árboles (impares, para minimizar empates)",
                                  options=[9, 49, 491], value=49)

    if st.button("🌲 Entrenar Random Forest", type="primary"):
        X, y, _ = cargar_crop()
        X_train, X_test, y_train, y_test = dividir_datos(X, y, test_size=0.2, estratificar=True)

        with st.spinner(f"Entrenando {n_arboles} árboles..."):
            modelo = RandomForestClassifier(n_estimators=n_arboles, random_state=42, n_jobs=-1)
            inicio = time.time()
            modelo.fit(X_train, y_train)
            tiempo = time.time() - inicio
            pred_test = modelo.predict(X_test)

        acc_test = accuracy_score(y_test, pred_test)
        st.markdown(f"**Partición de datos:** {len(X_train)} registros de entrenamiento (80%) · "
                    f"{len(X_test)} registros de prueba (20%)")
        c1, c2 = st.columns(2)
        c1.metric("Tiempo entrenamiento", f"{tiempo:.4f} s")
        c2.metric("Accuracy test", f"{acc_test:.2%}")

        clases = sorted(y.unique())
        cm = confusion_matrix(y_test, pred_test, labels=clases)

        # Gráfico interactivo con Plotly
        fig_cm = grafico_confusion_matrix_interactivo(cm, [str(c) for c in clases],
                                                       titulo="Matriz de Confusión - Random Forest (INTERACTIVA)")
        st.plotly_chart(fig_cm, use_container_width=True)

        mostrar_reporte_clasificacion(y_test, pred_test, "Precisión, recall y F1 por cultivo")

        st.markdown("**Importancia de variables (promediada entre todos los árboles del bosque) - INTERACTIVO**")
        importancias = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig_imp = grafico_barras_interactivo(importancias.values, importancias.index,
                                             titulo="Importancia de Variables - Random Forest",
                                             ylabel="Importancia", color="#3f5f7d")
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Cada árbol vota con variables ligeramente distintas; esta importancia "
                   "es el promedio de cuánto contribuyó cada variable a reducir la impureza "
                   "en TODOS los árboles del bosque.")

        st.markdown("**Un árbol individual dentro del bosque (completo)**")
        idx_arbol = st.number_input(f"Ver el árbol N° (0 a {n_arboles - 1})", 0, n_arboles - 1, 0)
        mostrar_arbol(modelo.estimators_[idx_arbol], X.columns.tolist(),
                      [str(c) for c in modelo.classes_],
                      titulo=f"Árbol #{idx_arbol} del bosque (de {n_arboles} en total)",
                      key=f"rf_clas_{idx_arbol}")
        st.caption("Cada árbol del bosque se ve distinto porque cada uno entrenó con una "
                   "muestra bootstrap distinta y con variables aleatorias en cada división "
                   "— compáralo con el árbol de la pestaña anterior o con otro número aquí.")

    st.markdown("---")
    st.subheader("Comparar los 3 bosques (9, 49 y 491 árboles) — punto 5a/5c")
    if st.button("📊 Comparar los 3 tamaños de bosque", key="comparar_rf"):
        X, y, _ = cargar_crop()
        X_train, X_test, y_train, y_test = dividir_datos(X, y, test_size=0.2, estratificar=True)

        filas = []
        with st.spinner("Entrenando 9, 49 y 491 árboles..."):
            for n in [9, 49, 491]:
                m = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
                inicio = time.time()
                m.fit(X_train, y_train)
                t = time.time() - inicio
                acc_tr = accuracy_score(y_train, m.predict(X_train))
                acc_te = accuracy_score(y_test, m.predict(X_test))
                filas.append({
                    "N° árboles": n,
                    "% Train": "80%", "% Test": "20%",
                    "Registros train": len(X_train), "Registros test": len(X_test),
                    "Tiempo (s)": round(t, 4),
                    "Accuracy train": f"{acc_tr:.2%}",
                    "Accuracy test": f"{acc_te:.2%}",
                })
        st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
        st.caption("El tiempo crece de forma prácticamente lineal con el número de árboles, "
                   "pero fíjate cuánto (o qué tan poco) mejora el accuracy de 49 a 491.")

    st.markdown("---")
    st.subheader("Evidencia de bootstrap (subconjuntos aleatorios)")
    if st.button("🔀 Mostrar evidencia de bootstrap"):
        X, y, _ = cargar_crop()
        X_train, _, _, _ = dividir_datos(X, y, test_size=0.2, estratificar=True)
        n = len(X_train)
        rng = np.random.RandomState(0)
        m1 = rng.choice(X_train.index, size=n, replace=True)
        m2 = rng.choice(X_train.index, size=n, replace=True)
        repetidos = n - len(set(m1))
        interseccion = len(set(m1) & set(m2))

        c1, c2, c3 = st.columns(3)
        c1.metric("Tamaño set entrenamiento", n)
        c2.metric("Duplicados en muestra árbol 1", f"{repetidos} ({repetidos/n:.1%})")
        c3.metric("Coincidencia árbol 1 vs árbol 2", f"{interseccion}/{n} ({interseccion/n:.1%})")
        st.caption("Cada árbol del bosque recibe una muestra distinta (con reemplazo) del "
                   "set de entrenamiento: por eso los árboles terminan siendo diferentes "
                   "entre sí aunque provengan de la misma base de datos.")

# =====================================================================
# TAB 4: REGRESIÓN
# =====================================================================
with tabs[4]:
    st.header("Regresión — Resistencia del concreto (csMPa)")
    with st.expander("📚 Teoría: lo mismo, pero con números continuos", expanded=True):
        st.markdown("""
        Un árbol/bosque de **regresión** funciona igual que en clasificación, pero:
        - Cada división minimiza la **varianza** dentro del grupo resultante (no la impureza de clase).
        - La predicción de una hoja es el **promedio** de `csMPa` de las muestras que caen en ella.
        - En Random Forest de regresión, la predicción final es el **promedio** de todos los
          árboles (no una votación, porque no hay clases discretas).
        - En vez de matriz de confusión, usamos una gráfica de **real vs. predicho**: los
          puntos deberían alinearse sobre la diagonal.
        """)

    modelo_tipo = st.radio("Tipo de modelo", ["Árbol de decisión", "Random Forest"], horizontal=True)

    if modelo_tipo == "Árbol de decisión":
        c1, c2 = st.columns(2)
        with c1:
            sin_limite_r = st.checkbox("Sin restricción de profundidad", value=False, key="reg_sinlim")
        with c2:
            max_depth_r = st.slider("max_depth", 1, 20, 6, key="reg_depth", disabled=sin_limite_r)
    else:
        n_arboles_r = st.select_slider("Número de árboles", options=[9, 49, 491], value=49, key="reg_narb")

    if st.button("📈 Entrenar modelo de regresión", type="primary"):
        X, y, _ = cargar_concreto()
        X_train, X_test, y_train, y_test = dividir_datos(X, y, test_size=0.2)

        with st.spinner("Entrenando..."):
            if modelo_tipo == "Árbol de decisión":
                modelo = DecisionTreeRegressor(
                    max_depth=None if sin_limite_r else max_depth_r, random_state=42
                )
            else:
                modelo = RandomForestRegressor(n_estimators=n_arboles_r, random_state=42, n_jobs=-1)

            inicio = time.time()
            modelo.fit(X_train, y_train)
            tiempo = time.time() - inicio
            pred_train = modelo.predict(X_train)
            pred_test = modelo.predict(X_test)

        mae = mean_absolute_error(y_test, pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        r2 = r2_score(y_test, pred_test)
        r2_train = r2_score(y_train, pred_train)

        n_features = X_test.shape[1]
        r2_adj = calcular_r2_ajustado(r2, len(y_test), n_features)
        r2_adj_train = calcular_r2_ajustado(r2_train, len(y_train), n_features)

        st.markdown(f"**Partición de datos:** {len(X_train)} registros de entrenamiento (80%) · "
                    f"{len(X_test)} registros de prueba (20%)")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tiempo", f"{tiempo:.4f} s")
        c2.metric("MAE", f"{mae:.2f} MPa")
        c3.metric("RMSE", f"{rmse:.2f} MPa")
        c4.metric("R² test", f"{r2:.4f}")
        c5.metric("R² Adj test", f"{r2_adj:.4f}")

        mae_train = mean_absolute_error(y_train, pred_train)
        ajuste, _ = diagnosticar_ajuste(r2_train, r2, mae_train, mae)
        resumen_ajuste = (f"R² train={r2_train:.3f} vs R² test={r2:.3f}, "
                          f"R² Adj train={r2_adj_train:.3f} vs R² Adj test={r2_adj:.3f}, "
                          f"MAE train={mae_train:.2f} vs MAE test={mae:.2f} MPa")
        if ajuste == "sobreajuste":
            st.warning(f"⚠ Posible **sobreajuste** ({resumen_ajuste}). Se marca si la brecha de R² "
                       "pasa de 0.15 o si el MAE de test es más del triple del de train.")
        elif ajuste == "subajuste":
            st.warning(f"⚠ Posible **subajuste** ({resumen_ajuste}): R² test por debajo de 0.60.")
        else:
            st.success(f"✔ Buen ajuste ({resumen_ajuste}).")

        # Gráfico interactivo con Plotly
        fig = grafico_scatter_interactivo(pred_test, y_test,
                                          titulo="Real vs Predicho - Regresión (INTERACTIVO)",
                                          xlabel="csMPa predicho",
                                          ylabel="csMPa real")
        st.plotly_chart(fig, use_container_width=True)

        if modelo_tipo == "Árbol de decisión":
            mostrar_arbol(modelo, X.columns.tolist(), titulo="Árbol de regresión completo (concreto)",
                          key="arbol_reg")
            st.caption("En cada hoja se ve el `value` = predicción de csMPa para las "
                       "muestras que caen ahí (es un promedio, no una clase).")
        else:
            st.markdown("**Importancia de variables (Random Forest) - INTERACTIVO**")
            importancias = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
            fig_imp = grafico_barras_interactivo(importancias.values, importancias.index,
                                                 titulo="Importancia de Variables - Random Forest Regresión",
                                                 ylabel="Importancia", color="#3f5f7d")
            st.plotly_chart(fig_imp, use_container_width=True)

            st.markdown("**Un árbol individual dentro del bosque (completo)**")
            idx_arbol_r = st.number_input(f"Ver el árbol N° (0 a {n_arboles_r - 1})",
                                           0, n_arboles_r - 1, 0, key="idx_arbol_reg")
            mostrar_arbol(modelo.estimators_[idx_arbol_r], X.columns.tolist(),
                          titulo=f"Árbol #{idx_arbol_r} del bosque de regresión",
                          key=f"rf_reg_{idx_arbol_r}")

    st.markdown("---")
    st.subheader("Comparar los 3 modelos de la rúbrica (punto 6, igual que en clasificación)")
    with st.expander("📚 ¿Qué compara este botón?"):
        st.markdown("""
        Igual que en la sección de Árboles, aquí se entrenan las 3 variantes sobre el
        dataset de **concreto** y se comparan lado a lado:
        - **Árbol 1:** 80/20, sin restricciones.
        - **Árbol 2:** 80/20, pre-poda (`max_depth=4`).
        - **Árbol 3:** 70/30, criterio de error absoluto (MAE).
        - Y los **3 Random Forest** (9, 49, 491 árboles) con partición 80/20.
        """)

    if st.button("📊 Comparar los 3 árboles + los 3 bosques de regresión"):
        X, y, _ = cargar_concreto()

        configs_arbol = [
            ("Árbol 1: Sin restricciones (80/20)", 0.2, DecisionTreeRegressor(random_state=42)),
            ("Árbol 2: Pre-poda max_depth=4 (80/20)", 0.2,
             DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42)),
            ("Árbol 3: Partición 70/30 + criterio MAE", 0.3,
             DecisionTreeRegressor(criterion="absolute_error", random_state=42)),
        ]

        filas = []
        with st.spinner("Entrenando los 3 árboles de regresión..."):
            for nombre, ts, modelo in configs_arbol:
                Xtr, Xte, ytr, yte = dividir_datos(X, y, test_size=ts)
                inicio = time.time()
                modelo.fit(Xtr, ytr)
                t = time.time() - inicio
                pred_te = modelo.predict(Xte)
                r2_te = r2_score(yte, pred_te)
                r2_adj_te = calcular_r2_ajustado(r2_te, len(yte), Xte.shape[1])
                filas.append({
                    "Modelo": nombre,
                    "% Train": f"{(1-ts):.0%}", "% Test": f"{ts:.0%}",
                    "Registros test": len(Xte),
                    "Tiempo (s)": round(t, 4),
                    "MAE": round(mean_absolute_error(yte, pred_te), 2),
                    "RMSE": round(np.sqrt(mean_squared_error(yte, pred_te)), 2),
                    "R² test": round(r2_te, 4),
                    "R² Adj test": round(r2_adj_te, 4),
                })
        st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)

        filas_rf = []
        with st.spinner("Entrenando los 3 Random Forest de regresión (9, 49, 491)..."):
            Xtr, Xte, ytr, yte = dividir_datos(X, y, test_size=0.2)
            for n in [9, 49, 491]:
                m = RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
                inicio = time.time()
                m.fit(Xtr, ytr)
                t = time.time() - inicio
                pred_te = m.predict(Xte)
                r2_te = r2_score(yte, pred_te)
                r2_adj_te = calcular_r2_ajustado(r2_te, len(yte), Xte.shape[1])
                filas_rf.append({
                    "N° árboles": n,
                    "% Train": "80%", "% Test": "20%",
                    "Registros test": len(Xte),
                    "Tiempo (s)": round(t, 4),
                    "MAE": round(mean_absolute_error(yte, pred_te), 2),
                    "RMSE": round(np.sqrt(mean_squared_error(yte, pred_te)), 2),
                    "R² test": round(r2_te, 4),
                    "R² Adj test": round(r2_adj_te, 4),
                })
        st.dataframe(pd.DataFrame(filas_rf), hide_index=True, use_container_width=True)

    # -----------------------------------------------------------------
    # Sobreajuste REAL en regresión (sin inyectar ruido: el concreto ya trae el suyo)
    # -----------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔬 Sobreajuste real en regresión: curva de complejidad y curva de aprendizaje")
    with st.expander("📚 ¿Cómo se ve el sobreajuste en regresión? (léelo antes de graficar)", expanded=True):
        st.markdown("""
        En regresión **no hay clases ni "balance"**; lo que hace que un árbol sobreajuste es
        tener **pocos datos y mediciones con variabilidad** (721 mezclas de concreto, cada
        una con su ruido de laboratorio). Por eso aquí **no hace falta inyectar ruido**.

        **Gráfica 1 — Curva de complejidad:** eje X = `max_depth` (qué tan complejo es el
        árbol); eje Y = R². Dos líneas: **train** y **test** (con validación cruzada de 5
        pliegues, que es más confiable que una sola partición; la banda es ± 1 desviación).
        - Al principio ambas suben juntas: el árbol **aprende el patrón real**.
        - Luego train sigue subiendo hacia 1.0 pero test **se estanca**: el árbol solo está
          **memorizando** cada mezcla. **Esa brecha creciente es el sobreajuste.**
        - Ojo: en este dataset el test **no baja** después (no forma una "U"), simplemente
          deja de mejorar. Lo que delata el sobreajuste es la brecha, no una caída.

        **Gráfica 2 — Curva de aprendizaje:** eje X = cuántos datos de entrenamiento se usan;
        árbol sin restricciones. Un modelo que sobreajusta tiene train ≈ 1.0 siempre, y test
        **mucho más bajo** que va subiendo al darle más datos: la brecha se cierra con más datos.
        """)

    if st.button("🧪 Graficar sobreajuste en regresión", type="primary", key="btn_sobre_reg"):
        Xc, yc, _ = cargar_concreto()
        cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
        profundidades = list(range(1, 21))

        with st.spinner("Calculando curva de complejidad (20 profundidades x 5 pliegues)..."):
            tr_sc, te_sc = validation_curve(
                DecisionTreeRegressor(random_state=42), Xc, yc,
                param_name="max_depth", param_range=profundidades, cv=cv5, scoring="r2")
        tr_m, tr_s = tr_sc.mean(axis=1), tr_sc.std(axis=1)
        te_m, te_s = te_sc.mean(axis=1), te_sc.std(axis=1)

        # Profundidad "bien ajustada": la más simple cuyo test queda dentro de 1 desviación
        # del mejor (regla de "un error estándar": no se elige un árbol más complejo si no
        # aporta una mejora real). Se decide con validación cruzada, sin mirar el test final.
        i_mejor = int(np.argmax(te_m))
        umbral = te_m[i_mejor] - te_s[i_mejor]
        prof_ok = next(d for d, m in zip(profundidades, te_m) if m >= umbral)

        fig_vc = grafico_validation_curve_interactivo(
            profundidades, tr_m, tr_s, te_m, te_s,
            titulo="Curva de complejidad — árbol de regresión (concreto)",
            xlabel="max_depth (complejidad del árbol)"
        )
        st.plotly_chart(fig_vc, use_container_width=True)
        st.caption(f"Con profundidad {prof_ok} el test ya está dentro de 1 desviación del mejor "
                   f"resultado posible (profundidad {profundidades[i_mejor]}); ir más allá agranda "
                   f"la brecha (train llega a {tr_m[-1]:.3f}, test se queda en {te_m[-1]:.3f}) "
                   "sin mejorar el test de forma que se distinga del ruido. Las zonas coloreadas "
                   "son orientativas, no cortes exactos.")

        with st.spinner("Calculando curva de aprendizaje..."):
            tamanos, tr_lc, te_lc = learning_curve(
                DecisionTreeRegressor(random_state=42), Xc, yc,
                train_sizes=np.linspace(0.1, 1.0, 8), cv=cv5, scoring="r2")
        fig_lc = grafico_learning_curve_interactivo(
            tamanos, tr_lc.mean(axis=1), tr_lc.std(axis=1),
            te_lc.mean(axis=1), te_lc.std(axis=1),
            titulo="Curva de aprendizaje — árbol SIN restricciones (concreto)"
        )
        st.plotly_chart(fig_lc, use_container_width=True)
        st.caption(f"Con solo {int(tamanos[0])} datos el árbol memoriza todo (train "
                   f"{tr_lc.mean(axis=1)[0]:.2f}) y en test saca {te_lc.mean(axis=1)[0]:.2f}. "
                   "La brecha se va cerrando al añadir datos, pero no desaparece: con estos "
                   "721 registros el árbol sin restricciones sigue con alta varianza.")

        # ---- Tres modelos + Random Forest, sobre una partición 80/20 ----
        st.markdown("### Los tres ajustes lado a lado (partición 80/20)")
        Xtr_r, Xte_r, ytr_r, yte_r = dividir_datos(Xc, yc, test_size=0.2)
        configs_reg = [
            ("🟡 Subajustado (max_depth=2)", DecisionTreeRegressor(max_depth=2, random_state=42)),
            (f"🟢 Bien ajustado (max_depth={prof_ok})", DecisionTreeRegressor(max_depth=prof_ok, random_state=42)),
            ("🔴 Sobreajustado (sin restricción)", DecisionTreeRegressor(random_state=42)),
            ("🌲 Random Forest (200 árboles)", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
        filas_sr, entrenados_sr = [], []
        with st.spinner("Entrenando los cuatro modelos..."):
            for nombre, mod in configs_reg:
                mod.fit(Xtr_r, ytr_r)
                p_tr, p_te = mod.predict(Xtr_r), mod.predict(Xte_r)
                filas_sr.append({
                    "Modelo": nombre,
                    "R² train": round(r2_score(ytr_r, p_tr), 3),
                    "R² test": round(r2_score(yte_r, p_te), 3),
                    "Brecha (train − test)": round(r2_score(ytr_r, p_tr) - r2_score(yte_r, p_te), 3),
                    "MAE train (MPa)": round(mean_absolute_error(ytr_r, p_tr), 2),
                    "MAE test (MPa)": round(mean_absolute_error(yte_r, p_te), 2),
                    "Hojas": mod.get_n_leaves() if hasattr(mod, "get_n_leaves") else "—",
                })
                entrenados_sr.append((nombre, mod, p_tr, p_te))
        st.dataframe(pd.DataFrame(filas_sr), hide_index=True, use_container_width=True)

        # Nota honesta calculada con los números reales de esta ejecución
        f_bien, f_sobre = filas_sr[1], filas_sr[2]
        if f_sobre["R² test"] >= f_bien["R² test"]:
            st.warning(
                f"⚠️ **Matiz importante:** en esta partición el árbol sin restricciones "
                f"(R² test {f_sobre['R² test']}, MAE {f_sobre['MAE test (MPa)']}) saca un resultado "
                f"de test **igual o algo mejor** que el 'bien ajustado' (R² test {f_bien['R² test']}, "
                f"MAE {f_bien['MAE test (MPa)']}). Es decir: aquí **memorizar no empeora el test**, "
                f"solo lo estanca. Se le llama sobreajustado por la **brecha** train−test "
                f"({f_sobre['Brecha (train − test)']} vs {f_bien['Brecha (train − test)']}) y porque usa "
                f"{f_sobre['Hojas']} hojas para ~{len(Xtr_r)} registros (casi una hoja por dato). "
                f"El 'bien ajustado' no gana en precisión sino en **simplicidad**: {f_bien['Hojas']} hojas, "
                f"más estable e interpretable, con una diferencia de test que cae dentro del "
                f"ruido entre particiones. Lo que sí mejora claramente el test es el **Random Forest**.")
        else:
            st.info("El árbol 'bien ajustado' generaliza mejor que el sobreajustado en esta partición.")

        lim_r = [float(yc.min()), float(yc.max())]
        filas_rp = [("TRAIN", ytr_r, 2, "#a8552f"), ("TEST", yte_r, 3, "#3f7d54")]
        titulos_rp = [f"{e[0].split(' ', 1)[1]}<br>{etiqueta} (R²={r2_score(real, e[idx]):.3f})"
                      for etiqueta, real, idx, _ in filas_rp for e in entrenados_sr]
        fig_rp = make_subplots(rows=2, cols=len(entrenados_sr), shared_xaxes=True, shared_yaxes=True,
                               horizontal_spacing=0.03, vertical_spacing=0.12,
                               subplot_titles=titulos_rp)
        for j, entrenado in enumerate(entrenados_sr):
            for i, (etiqueta, real, idx, color) in enumerate(filas_rp):
                fig_rp.add_trace(go.Scatter(
                    x=np.asarray(real), y=entrenado[idx], mode="markers", showlegend=False,
                    marker=dict(size=5, color=color, opacity=0.5),
                    hovertemplate=f"{etiqueta}<br>Real: %{{x:.2f}}<br>Predicho: %{{y:.2f}}<extra></extra>"),
                    row=i + 1, col=j + 1)
                fig_rp.add_trace(go.Scatter(x=lim_r, y=lim_r, mode="lines", showlegend=False,
                                            line=dict(color="black", dash="dash", width=1),
                                            hoverinfo="skip"),
                                 row=i + 1, col=j + 1)
        fig_rp.update_xaxes(title_text="csMPa real", row=2)
        fig_rp.update_yaxes(title_text="csMPa predicho", col=1)
        fig_rp.update_annotations(font_size=11)
        fig_rp.update_layout(template="plotly_white", height=720, margin=dict(t=80))
        st.plotly_chart(fig_rp, use_container_width=True)
        st.caption("Fila de arriba: datos con los que se entrenó. Fila de abajo: datos que el "
                   "modelo nunca vio. En el árbol sobreajustado los puntos de TRAIN caen casi "
                   "exactamente sobre la diagonal (memorizó), pero en TEST se dispersan: esa "
                   "diferencia entre ambas filas ES el sobreajuste. En el subajustado, ambas "
                   "filas se ven igual de mal (los puntos forman 'escalones').")

        if st.checkbox("Dibujar también los tres árboles (el sobreajustado tiene ~540 hojas y tarda unos segundos)",
                       value=True, key="chk_arboles_sobre_reg"):
            for nombre, mod, _, _ in entrenados_sr[:3]:
                mostrar_arbol(mod, Xc.columns.tolist(), titulo=nombre,
                              key=f"sobre_reg_{nombre[2:6]}")

# =====================================================================
# TAB 5: KNN
# =====================================================================
with tabs[5]:
    st.header("KNN — Aprendizaje basado en instancias")
    with st.expander("📚 Teoría: KNN no entrena un modelo", expanded=True):
        st.markdown("""
        KNN es distinto a todo lo anterior: durante el "entrenamiento" solo **guarda los datos**
        en memoria (*lazy learning*, aprendizaje perezoso). Al predecir, calcula la distancia
        entre el punto nuevo y todos los puntos guardados, toma los **K vecinos más cercanos**
        y vota la clase mayoritaria entre ellos.

        - **Selección de K:** se prueba un rango de valores y se elige el que mejor generaliza
          en validación cruzada (ni muy bajo = sobreajuste, ni muy alto = subajuste).
        - **Ponderación:** en vez de que todos los vecinos voten igual (`uniform`), se puede
          dar más peso a los más cercanos (`distance`), o ponderar variables según qué tan
          discriminantes son, para reducir empates.
        - **Requiere datos estandarizados**, porque todo se basa en distancias.
        """)

    with st.spinner("Cargando..."):
        _, k_optimo = curva_k_knn("uniform")

    k = st.slider("K (número de vecinos)", 1, 25, k_optimo, key="knn_k")
    ponderacion = st.selectbox("Ponderación", ["uniform", "distance", PONDERACION_VARIABLES])
    weights = "distance" if ponderacion == "distance" else "uniform"

    if st.button("🔎 Entrenar y evaluar KNN", type="primary"):
        X, y, _ = cargar_crop()
        X_train, X_test, y_train, y_test = dividir_datos(X, y, test_size=0.2, estratificar=True)
        X_train_esc, X_test_esc, _ = estandarizar(X_train, X_test)
        if ponderacion == PONDERACION_VARIABLES:
            # Los pesos (Random Forest) se calculan antes de medir el tiempo de KNN
            pesos = pesos_variables_rf()
            X_train_esc, X_test_esc = X_train_esc * pesos, X_test_esc * pesos

        with st.spinner("Prediciendo..."):
            inicio_fit = time.time()
            modelo = KNeighborsClassifier(n_neighbors=k, weights=weights)
            modelo.fit(X_train_esc, y_train)
            tiempo_fit = time.time() - inicio_fit

            inicio_pred = time.time()
            pred_test = modelo.predict(X_test_esc)
            tiempo_pred = time.time() - inicio_pred

        acc = accuracy_score(y_test, pred_test)
        st.markdown(f"**Partición de datos:** {len(X_train)} registros de entrenamiento (80%) · "
                    f"{len(X_test)} registros de prueba (20%) — estandarizados con Z-score")
        c1, c2, c3 = st.columns(3)
        c1.metric("Tiempo 'entrenamiento' (solo guardar datos)", f"{tiempo_fit:.5f} s")
        c2.metric("Tiempo de predicción", f"{tiempo_pred:.5f} s")
        c3.metric("Accuracy test", f"{acc:.2%}")
        st.caption("Nota cómo 'entrenar' es casi instantáneo (solo copia los datos), "
                   "pero predecir sí toma tiempo real: tiene que comparar contra "
                   "todo el set guardado. Eso es justo lo que significa que KNN "
                   "'no utiliza un modelo'.")

        with st.spinner("Calculando curva K vs accuracy (validación cruzada)..."):
            accs, k_opt_w = curva_k_knn(ponderacion)
        fig = go.Figure(go.Scatter(
            x=VALORES_K, y=accs, mode="lines+markers", name="Accuracy (CV 5-fold)",
            line=dict(color="#a8552f", width=2.5), marker=dict(size=7),
            hovertemplate="K = %{x}<br>Accuracy = %{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[k_opt_w], y=[accs[VALORES_K.index(k_opt_w)]], mode="markers+text",
            name=f"K óptimo = {k_opt_w}", marker=dict(size=18, color="green", symbol="star"),
            text=[f"K óptimo = {k_opt_w}"], textposition="middle right",
            textfont=dict(color="green"),
            hovertemplate="<b>ÓPTIMO</b><br>K = %{x}<br>Accuracy = %{y:.4f}<extra></extra>"))
        if k != k_opt_w:
            fig.add_vline(x=k, line_dash="dash", line_color="gray",
                          annotation_text=f"K usado = {k}", annotation_position="top")
        fig.update_layout(template="plotly_white", height=420, xaxis_title="K",
                          yaxis_title="Accuracy (validación cruzada)", margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

        clases = sorted(y.unique())
        cm = confusion_matrix(y_test, pred_test, labels=clases)

        # Gráfico interactivo con Plotly
        fig_cm = grafico_confusion_matrix_interactivo(cm, [str(c) for c in clases],
                                                       titulo="Matriz de Confusión - KNN (INTERACTIVA)",
                                                       color="Greens")
        st.plotly_chart(fig_cm, use_container_width=True)

        mostrar_reporte_clasificacion(y_test, pred_test, "Precisión, recall y F1 por cultivo")

    st.markdown("---")
    st.subheader("Comparar ponderaciones (punto 7d) con el K elegido arriba")
    with st.expander("📚 ¿Para qué sirve comparar uniform, distance y variables ponderadas?", expanded=True):
        st.markdown("""
        El punto 7d pide **dar más peso a ciertas cosas para evitar empates**. Un empate pasa
        cuando, por ejemplo con K=4, dos vecinos dicen *arroz* y dos dicen *maíz*: el voto no
        decide. Hay dos lugares donde se puede poner peso, y por eso se comparan 3 estrategias:

        | Estrategia | ¿Qué pesa? | Idea |
        |---|---|---|
        | `weights='uniform'` | Nada (línea base) | Cada uno de los K vecinos vale **1 voto**, esté cerca o lejos. |
        | `weights='distance'` | **Los vecinos** | Cada vecino vota con peso **1 / distancia**: el que está más cerca pesa más. Un empate 2 vs 2 casi nunca queda empatado, porque las distancias casi nunca son iguales. |
        | Variables ponderadas | **Las variables (columnas)** | Antes de medir distancias, cada variable se multiplica por su importancia (sacada de un Random Forest). Así, una diferencia en una variable muy discriminante (p. ej. `rainfall` o `humidity`, peso ≈ 1.5) aleja más a los puntos que una diferencia en una variable poco útil (p. ej. `ph`, peso ≈ 0.4). |

        **Ejemplo:** un punto nuevo tiene 2 vecinos *arroz* a distancia 0.2 y 0.3, y 2 vecinos
        *maíz* a distancia 1.5 y 1.8.
        - `uniform`: 2 votos vs 2 votos → **empate**.
        - `distance`: arroz = 1/0.2 + 1/0.3 = 8.3 · maíz = 1/1.5 + 1/1.8 = 1.2 → gana **arroz**.

        **Cómo leer el resultado:** si las 3 estrategias dan casi el mismo accuracy, los datos
        ya están bien separados y la ponderación no aporta mucho. Con **K=1** solo hay un
        vecino, así que nunca hay empate y `uniform` y `distance` dan **exactamente lo mismo**;
        para ver la diferencia, prueba con un K par (2, 4, 6) en el slider de arriba.
        Con K=2, por ejemplo: uniform ≈ 96.8% → distance ≈ 98.0% → variables ponderadas ≈ 98.4%.
        """)
    if st.button("📊 Comparar uniform vs distance vs variables ponderadas"):
        X, y, _ = cargar_crop()
        X_train, X_test, y_train, y_test = dividir_datos(X, y, test_size=0.2, estratificar=True)
        X_train_esc, X_test_esc, _ = estandarizar(X_train, X_test)

        filas = []
        for w in ["uniform", "distance"]:
            m = KNeighborsClassifier(n_neighbors=k, weights=w)
            m.fit(X_train_esc, y_train)
            acc_w = accuracy_score(y_test, m.predict(X_test_esc))
            filas.append({"Estrategia": f"weights='{w}'", "Accuracy test": f"{acc_w:.2%}"})

        # Ponderación manual por importancia de variables (Random Forest como proxy)
        pesos = pesos_variables_rf()
        m_pond = KNeighborsClassifier(n_neighbors=k)
        m_pond.fit(X_train_esc * pesos, y_train)
        acc_pond = accuracy_score(y_test, m_pond.predict(X_test_esc * pesos))
        filas.append({"Estrategia": "Variables ponderadas por importancia", "Accuracy test": f"{acc_pond:.2%}"})

        st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)

        tabla_pesos = pd.DataFrame({"Variable": X.columns, "Peso relativo": pesos.round(3)})
        st.dataframe(tabla_pesos.sort_values("Peso relativo", ascending=False),
                     hide_index=True, use_container_width=True)
        st.caption("Los pesos salen de la importancia de cada variable según un Random Forest "
                   "de referencia: las variables más discriminantes pesan más al calcular distancia.")

# =====================================================================
# TAB 6: CLUSTERING
# =====================================================================
with tabs[6]:
    st.header("Clustering — K-means sobre cultivos")
    with st.expander("📚 Teoría: aprendizaje no supervisado y el centroide", expanded=True):
        st.markdown("""
        Aquí el algoritmo recibe **solo las variables numéricas, sin la columna `label`**, y
        debe encontrar grupos parecidos por sí solo. La columna `label` se usa DESPUÉS,
        únicamente para evaluar qué tan bien esos grupos coinciden con los cultivos reales.

        **K-means, paso a paso:**
        1. Se eligen K centroides iniciales.
        2. Cada punto se asigna al centroide más cercano.
        3. Cada centroide se recalcula como el **promedio** de los puntos que le fueron asignados.
        4. Se repite 2-3 hasta que los centroides casi no se mueven (convergencia).

        **3 técnicas para elegir K:**
        - **Codo (elbow):** se busca el punto donde la inercia deja de bajar bruscamente.
        - **Silueta:** mide qué tan parecido es un punto a su cluster vs. al más cercano (más alto = mejor).
        - **Davies-Bouldin:** compara dispersión interna vs. separación entre clusters (más bajo = mejor).
        """)

    k_cluster = st.slider("Número de clusters (K)", 2, 30, 22, key="cluster_k")

    if st.button("🧩 Ejecutar K-means", type="primary"):
        X, y, _ = cargar_crop()
        X_esc, _, _ = estandarizar(X, X)

        with st.spinner("Agrupando..."):
            modelo = KMeans(n_clusters=k_cluster, random_state=42, n_init=10)
            clusters = modelo.fit_predict(X_esc)
            sil = silhouette_score(X_esc, clusters)
            db = davies_bouldin_score(X_esc, clusters)

        c1, c2 = st.columns(2)
        c1.metric("Silhouette score", f"{sil:.4f}", help="Más alto es mejor")
        c2.metric("Davies-Bouldin", f"{db:.4f}", help="Más bajo es mejor")

        clases = sorted(y.unique())
        y_idx = y.map({c: i for i, c in enumerate(clases)}).to_numpy()
        matriz = np.zeros((k_cluster, len(clases)), dtype=int)
        for c, real in zip(clusters, y_idx):
            matriz[c, real] += 1
        filas, columnas = linear_sum_assignment(-matriz)
        mapa = {f: columnas[i] for i, f in enumerate(filas)}
        for c in range(k_cluster):
            if c not in mapa:
                mapa[c] = int(np.argmax(matriz[c]))
        pred_labels = [clases[mapa[c]] for c in clusters]
        acc = accuracy_score(y, pred_labels)
        st.metric("Coincidencia con las clases reales", f"{acc:.2%}")
        st.caption("Esto NO es el accuracy de un clasificador supervisado: mide qué tan bien "
                   "los grupos, sin conocer las clases, terminaron coincidiendo con la realidad "
                   "(después de emparejar cada cluster con su clase mayoritaria).")

        cm = confusion_matrix(y, pred_labels, labels=clases)

        # Gráfico interactivo con Plotly
        fig_cm = grafico_confusion_matrix_interactivo(cm, [str(c) for c in clases],
                                                       titulo="Matriz de Confusión - K-Means vs Clases Reales (INTERACTIVA)",
                                                       color="Purples",
                                                       eje_x="Cluster (mapeado a clase mayoritaria)")
        st.plotly_chart(fig_cm, use_container_width=True)

        # --- Vista 2D de los clusters con PCA (idea tomada del proyecto de referencia) ---
        st.markdown("**Vista 2D de los clusters (PCA) con sus centroides**")
        with st.expander("📚 ¿Qué es PCA y cómo leer esta gráfica?"):
            st.markdown("""
            Las 7 variables no se pueden dibujar en una hoja (7 dimensiones). **PCA**
            (Análisis de Componentes Principales) las resume en 2 ejes nuevos (PC1 y PC2) que
            conservan la mayor variación posible de los datos. Es solo una *proyección para
            poder verlos*: el K-means se hizo con las 7 variables, no con estos 2 ejes.

            - **Izquierda:** cada punto coloreado según el **cluster** que encontró K-means.
              Las **X negras** son los centroides (el "promedio" de cada grupo).
            - **Derecha:** los mismos puntos coloreados según el **cultivo real**.
            - Si ambos mapas se parecen, el agrupamiento redescubrió los cultivos; las zonas
              donde los colores se mezclan son cultivos con condiciones parecidas.
            - Con 22 grupos, los colores se repiten (la paleta tiene 20): fíjate en las
              *zonas*, no en cada color individual.
            """)
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_esc)
        cent_2d = pca.transform(modelo.cluster_centers_)
        var_exp = pca.explained_variance_ratio_

        # Dos gráficos interactivos lado a lado
        col1, col2 = st.columns(2)
        with col1:
            fig_pca_clusters = grafico_pca_clusters_interactivo(X_2d, clusters, cent_2d, var_exp, k_cluster)
            st.plotly_chart(fig_pca_clusters, use_container_width=True)
        with col2:
            fig_pca_cultivos = grafico_pca_cultivos_interactivo(X_2d, y_idx, clases, var_exp)
            st.plotly_chart(fig_pca_cultivos, use_container_width=True)
        st.caption(f"Estos 2 ejes conservan solo el {var_exp.sum():.1%} de la información "
                   "de las 7 variables, así que es normal que algunos grupos se vean "
                   "encimados aunque en 7 dimensiones sí estén separados.")

        # Cultivos que quedan claramente aislados del resto en PC1 (calculado, no escrito a mano)
        pc1_por_cultivo = pd.Series(X_2d[:, 0]).groupby(np.array(y)).mean()
        aislados = pc1_por_cultivo[pc1_por_cultivo > 2.5].index.tolist()
        if aislados:
            pesos_pc1 = pd.Series(pca.components_[0], index=X.columns)
            top_vars = pesos_pc1.abs().sort_values(ascending=False).index[:2].tolist()
            st.info(f"🔎 Se ve un bloque separado del resto formado por: **{', '.join(aislados)}**. "
                    f"El eje PC1 está dominado por **{top_vars[0]}** y **{top_vars[1]}**. "
                    "Esto explica por qué la silueta y Davies-Bouldin favorecen K=2: hay un "
                    "grupo claramente alejado, mientras el resto de cultivos se solapa entre sí.")

        st.info("💡 Prueba K=2 (el que suelen preferir silueta/Davies-Bouldin) vs K=22 "
                "(el número real de cultivos) y compara la coincidencia con la realidad — "
                "es una de las conclusiones más interesantes del proyecto.")

    st.markdown("---")
    st.subheader("Comparar las 3 técnicas para elegir K")
    if st.button("📐 Calcular codo, silueta y Davies-Bouldin (rango K=2 a 30)"):
        rango_k = RANGO_K
        inercias, siluetas, dbs, k_sil, k_db = curvas_k_kmeans()
        fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07,
                            subplot_titles=("Codo (Inercia)", "Silueta", "Davies-Bouldin"))
        for col, (valores, color, nombre) in enumerate([
                (inercias, "#a8552f", "Inercia"), (siluetas, "#3f7d54", "Silueta"),
                (dbs, "#3f5f7d", "Davies-Bouldin")], start=1):
            fig.add_trace(go.Scatter(x=rango_k, y=valores, mode="lines+markers", name=nombre,
                                     line=dict(color=color, width=2), marker=dict(size=6),
                                     hovertemplate=f"K = %{{x}}<br>{nombre} = %{{y:.4f}}<extra></extra>"),
                          row=1, col=col)
            fig.add_vline(x=22, line_dash="dash", line_color="gray", row=1, col=col,
                          annotation_text="K real (22)", annotation_font_size=10)
        for col, k_opt, y_opt, color in [(2, k_sil, max(siluetas), "green"),
                                         (3, k_db, min(dbs), "blue")]:
            fig.add_trace(go.Scatter(
                x=[k_opt], y=[y_opt], mode="markers+text",
                marker=dict(size=16, color=color, symbol="star"),
                text=[f"K óptimo = {k_opt}"], textposition="middle right",
                textfont=dict(color=color, size=11),
                hovertemplate="<b>ÓPTIMO</b><br>K = %{x}<br>Valor = %{y:.4f}<extra></extra>"),
                row=1, col=col)
        fig.update_xaxes(title_text="K")
        fig.update_layout(template="plotly_white", height=430, showlegend=False, margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)
        st.warning(f"Silueta sugiere K={k_sil}, Davies-Bouldin sugiere K={k_db} — "
                  f"ninguno coincide exactamente con el K real (22). Esto sugiere que "
                  f"varios cultivos comparten condiciones de suelo/clima muy similares.")
