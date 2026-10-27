"""
graficos.py  (ENTREGA 2)
-------------------------
Gráficas interactivas (Plotly), con el mismo estilo de la entrega 1.
Las usan main.py (guarda HTML interactivo + PNG) y app.py (st.plotly_chart).
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_TRAIN = "#a8552f"
COLOR_TEST = "#3f7d54"
COLOR_CV = "#3f5f7d"
COLOR_OPTIMO = "#2a7f2a"
COLOR_E1, COLOR_E2 = "#3f5f7d", "#a8552f"
PLANTILLA = "plotly_white"

ESCALAS_ACIERTOS = {
    "Blues":   ["#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    "Greens":  ["#c7e9c0", "#74c476", "#238b45", "#00441b"],
    "Oranges": ["#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"],
    "Purples": ["#dadaeb", "#9e9ac8", "#6a51a3", "#3f007d"],
}
ESCALA_ERRORES = ["#fcbba1", "#fb6a4a", "#de2d26", "#a50f15"]


def _escala(colores):
    return [[float(p), c] for p, c in zip(np.linspace(0, 1, len(colores)), colores)]


def _ticks_log(maximo):
    valores = [v for v in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] if v <= maximo] or [1]
    if maximo / valores[-1] >= 1.5:
        valores.append(int(maximo))
    return [float(np.log10(v)) for v in valores], [str(v) for v in valores]


# ------------------------------------------------------------------ COMUNES (igual que entrega 1)
def figura_matriz_confusion(cm, labels, titulo="Matriz de confusión", color="Blues",
                            eje_x="Predicción", eje_y="Real", mostrar_numeros=True):
    """Aciertos (diagonal) en `color`, errores en rojo con escala log; celdas en cero vacías."""
    cm = np.asarray(cm)
    n = len(labels)
    labels = [str(l) for l in labels]
    es_diag = np.eye(n, dtype=bool)
    tot = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, tot, out=np.zeros(cm.shape, dtype=float), where=tot > 0)
    info = np.dstack([cm, pct * 100])
    hover = ("Real: <b>%{y}</b><br>Predicho: <b>%{x}</b><br>"
             "Cantidad: <b>%{customdata[0]:.0f}</b> (%{customdata[1]:.1f}% de la clase real)<extra></extra>")
    z_ok = np.where(es_diag & (cm > 0), cm, np.nan).astype(float)
    z_err = np.where(~es_diag & (cm > 0), cm, np.nan).astype(float)
    max_ok = float(np.nanmax(z_ok)) if np.any(~np.isnan(z_ok)) else 1.0
    max_err = float(np.nanmax(z_err)) if np.any(~np.isnan(z_err)) else 1.0
    log_max = max(np.log10(max_err), 0.3)
    tv, tt = _ticks_log(max_err)

    comunes = dict(x=labels, y=labels, xgap=1.5, ygap=1.5, customdata=info, hovertemplate=hover)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=np.zeros((n, n)), colorscale=[[0, "#f1f3f5"], [1, "#f1f3f5"]], showscale=False, **comunes))
    fig.add_trace(go.Heatmap(z=z_ok, colorscale=_escala(ESCALAS_ACIERTOS.get(color, ESCALAS_ACIERTOS["Blues"])),
                             zmin=0, zmax=max_ok, name="Aciertos",
                             colorbar=dict(title=dict(text="Aciertos", side="right"), x=1.0, len=0.45, y=0.78,
                                           thickness=14, outlinewidth=0), **comunes))
    fig.add_trace(go.Heatmap(z=np.log10(z_err), colorscale=_escala(ESCALA_ERRORES), zmin=0, zmax=log_max, name="Errores",
                             colorbar=dict(title=dict(text="Errores", side="right"), x=1.0, len=0.45, y=0.22,
                                           thickness=14, outlinewidth=0, tickvals=tv, ticktext=tt), **comunes))
    anot = []
    for i in range(n):
        for j in range(n):
            v = int(cm[i, j])
            if v == 0:
                continue
            oscuro = (v / max_ok > 0.45) if es_diag[i, j] else (np.log10(v) / log_max > 0.65)
            anot.append(dict(x=labels[j], y=labels[i], text=str(v), showarrow=False,
                             font=dict(size=9 if n > 10 else 12,
                                       color="rgba(255,255,255,0.7)" if oscuro else "rgba(40,40,40,0.55)")))
    fig.update_layout(
        title=dict(text=titulo, x=0.01), template=PLANTILLA, plot_bgcolor="white",
        height=max(420, min(820, 30 * n + 220)), margin=dict(l=10, r=10, t=90, b=10),
        annotations=anot if mostrar_numeros else [],
        updatemenus=[dict(type="buttons", direction="left", x=1, xanchor="right", y=1.01, yanchor="bottom",
                          showactive=True, pad=dict(r=4, t=0), font=dict(size=11),
                          buttons=[dict(label="Con números", method="relayout", args=[{"annotations": anot}]),
                                   dict(label="Solo colores", method="relayout", args=[{"annotations": []}])],
                          active=0 if mostrar_numeros else 1)])
    fig.update_xaxes(title_text=eje_x, tickangle=-60 if n > 6 else 0, showgrid=False, tickfont=dict(size=10))
    fig.update_yaxes(title_text=eje_y, autorange="reversed", showgrid=False, tickfont=dict(size=10))
    return fig


def figura_real_vs_predicho(y_real, y_pred, titulo="", unidad="csMPa", color=COLOR_TRAIN):
    """Dispersión real vs predicho con la diagonal de predicción perfecta."""
    y_real, y_pred = np.asarray(y_real, dtype=float), np.asarray(y_pred, dtype=float)
    lim = [float(min(y_real.min(), y_pred.min())), float(max(y_real.max(), y_pred.max()))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Predicción perfecta",
                             line=dict(dash="dash", color="#555"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=y_real, y=y_pred, mode="markers", name="Predicciones",
                             marker=dict(size=7, color=color, opacity=0.55), customdata=y_pred - y_real,
                             hovertemplate=(f"Real: %{{x:.2f}} {unidad}<br>Predicho: %{{y:.2f}} {unidad}"
                                            "<br>Error: %{customdata:+.2f}<extra></extra>")))
    fig.update_layout(title=titulo, template=PLANTILLA, height=520, xaxis_title=f"{unidad} real",
                      yaxis_title=f"{unidad} predicho", legend=dict(orientation="h", y=-0.15))
    fig.update_yaxes(scaleanchor="x")
    return fig


def figura_heatmap(tabla, titulo, eje_x, eje_y, formato=".3f", mayor_es_mejor=True):
    """Heatmap de una rejilla de hiperparámetros (DataFrame) con el valor de cada celda; marca el modelo elegido."""
    z = tabla.values.astype(float)
    xs, ys = [str(c) for c in tabla.columns], [str(i) for i in tabla.index]
    fig = go.Figure(go.Heatmap(z=z, x=xs, y=ys, colorscale="YlGn" if mayor_es_mejor else "YlGn_r",
                               text=[[format(v, formato) if not np.isnan(v) else "" for v in fila] for fila in z],
                               texttemplate="%{text}", hovertemplate=f"{eje_y}=%{{y}}<br>{eje_x}=%{{x}}<br>valor=%{{z:{formato}}}<extra></extra>",
                               colorbar=dict(thickness=12, outlinewidth=0)))
    if tabla.attrs.get("mejor") and tabla.attrs["mejor"][0] in ys and tabla.attrs["mejor"][1] in xs:
        # Celda del modelo elegido por GridSearchCV (en un empate no siempre es el máximo que se ve redondeado)
        i, j = ys.index(tabla.attrs["mejor"][0]), xs.index(tabla.attrs["mejor"][1])
        fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5, line=dict(color="#b00020", width=3))
    elif np.any(~np.isnan(z)):
        i, j = np.unravel_index(np.nanargmax(z) if mayor_es_mejor else np.nanargmin(z), z.shape)
        fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5, line=dict(color="#b00020", width=3))
    fig.update_layout(title=titulo, template=PLANTILLA, height=120 + 55 * len(ys), margin=dict(t=60, b=40),
                      xaxis=dict(title=eje_x, type="category"), yaxis=dict(title=eje_y, type="category"))
    return fig


def figura_barras(nombres, valores, titulo="", eje_y="", color=COLOR_TRAIN, formato=".4f"):
    fig = go.Figure(go.Bar(x=[str(n) for n in nombres], y=valores, marker_color=color,
                           text=[format(v, formato) for v in valores], textposition="outside"))
    fig.update_layout(title=titulo, template=PLANTILLA, height=380, yaxis_title=eje_y,
                      yaxis_range=[0, max(valores) * 1.18 if len(valores) else 1])
    return fig


def guardar_figura(fig, ruta_png, width=1000, height=None):
    """Guarda HTML interactivo (…_interactivo.html) y PNG estático (si kaleido está disponible)."""
    ruta_html = ruta_png.rsplit(".", 1)[0] + "_interactivo.html"
    fig.write_html(ruta_html, include_plotlyjs="cdn")
    try:
        go.Figure(fig).update_layout(updatemenus=[]).write_image(
            ruta_png, width=width, height=height or fig.layout.height or 600, scale=1.5)
    except Exception as e:                                     # sin Chrome/kaleido: queda solo el HTML
        print(f"  (PNG no generado para {ruta_png}: {type(e).__name__})")


# ------------------------------------------------------------------ NAIVE BAYES
def figura_gaussianas(modelo, x, columnas, variable, cultivos):
    """Normales P(variable | cultivo) aprendidas, con el valor de la muestra marcado."""
    j = columnas.index(variable)
    idx = [list(modelo.clases_).index(c) for c in cultivos]
    mus, sds = modelo.media_[idx, j], np.sqrt(modelo.varianza_[idx, j])
    xs = np.linspace((mus - 3.5 * sds).min(), (mus + 3.5 * sds).max(), 400)
    fig = go.Figure()
    for c, mu, sd in zip(cultivos, mus, sds):
        fig.add_trace(go.Scatter(x=xs, y=np.exp(-((xs - mu) ** 2) / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)),
                                 mode="lines", name=str(c), line=dict(width=2.5)))
    fig.add_vline(x=float(x[j]), line_dash="dash", line_color="black",
                  annotation_text=f"muestra = {x[j]:.1f}", annotation_position="top")
    fig.update_layout(title=f"P({variable} | cultivo)", template=PLANTILLA, height=380,
                      xaxis_title=variable, yaxis_title="densidad", hovermode="x unified")
    return fig


def figura_correlacion(df, titulo="Correlación absoluta promedio dentro de cada cultivo"):
    fig = go.Figure(go.Heatmap(z=df.values, x=list(df.columns), y=list(df.index), zmin=0, zmax=1, colorscale="Reds",
                               text=np.round(df.values, 2), texttemplate="%{text}", colorbar=dict(thickness=12)))
    fig.update_layout(title=titulo, template=PLANTILLA, height=450, yaxis_autorange="reversed")
    return fig


# ------------------------------------------------------------------ REDES NEURONALES
def figura_arquitectura(n_entrada, capas, n_salida, titulo="", max_nodos=10):
    """Esquema de la red: neuronas como puntos y conexiones como líneas (máx. `max_nodos` por capa)."""
    tamanos = [n_entrada] + list(capas) + [n_salida]
    nombres = ["Entrada"] + [f"Oculta {i + 1}" for i in range(len(capas))] + ["Salida"]
    colores = [COLOR_CV] + [COLOR_TRAIN] * len(capas) + [COLOR_TEST]
    pos = [np.linspace(-(min(n, max_nodos) - 1) / 2, (min(n, max_nodos) - 1) / 2, min(n, max_nodos)) for n in tamanos]
    ex, ey = [], []
    for c in range(len(tamanos) - 1):
        for y0 in pos[c]:
            for y1 in pos[c + 1]:
                ex += [c, c + 1, None]
                ey += [y0, y1, None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#cccccc", width=0.6), hoverinfo="skip", showlegend=False))
    for c, ys in enumerate(pos):
        fig.add_trace(go.Scatter(x=[c] * len(ys), y=ys, mode="markers", marker=dict(size=14, color=colores[c],
                                 line=dict(color="white", width=1)), hovertext=f"{nombres[c]}: {tamanos[c]} neuronas",
                                 hoverinfo="text", showlegend=False))
    fig.update_layout(title=titulo, template=PLANTILLA, height=380, margin=dict(t=60, b=50),
                      xaxis=dict(tickvals=list(range(len(tamanos))),
                                 ticktext=[f"{n}<br>({t})" for n, t in zip(nombres, tamanos)], showgrid=False, zeroline=False),
                      yaxis=dict(visible=False))
    return fig


def figura_curvas_red(curvas, tarea, titulo=""):
    """Pérdida y métrica por época (train vs test); la línea punteada marca la menor pérdida en test."""
    e = np.arange(1, len(curvas["loss_train"]) + 1)
    clave = "acc" if tarea == "clasificacion" else "r2"
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pérdida por época", "Accuracy por época" if clave == "acc" else "R² por época"))
    for col, (k_tr, k_te) in enumerate([("loss_train", "loss_test"), (f"{clave}_train", f"{clave}_test")], 1):
        fig.add_trace(go.Scatter(x=e, y=curvas[k_tr], name="Train", line=dict(color=COLOR_TRAIN), showlegend=col == 1), 1, col)
        fig.add_trace(go.Scatter(x=e, y=curvas[k_te], name="Test", line=dict(color=COLOR_TEST), showlegend=col == 1), 1, col)
    mejor = int(np.argmin(curvas["loss_test"])) + 1
    fig.add_vline(x=mejor, line_dash="dot", line_color="gray", row=1, col=1,
                  annotation_text=f"mín. test (época {mejor})", annotation_position="top right")
    fig.update_yaxes(type="log", title_text="log-loss" if clave == "acc" else "MSE (y estandarizada)", row=1, col=1)
    fig.update_yaxes(range=[0.6 if clave == "acc" else 0.3, 1.01], row=1, col=2)
    fig.update_xaxes(title_text="Época")
    fig.update_layout(title=titulo, template=PLANTILLA, height=400, hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    return fig


def figura_complejidad(redes, tarea, titulo="¿Más complejidad = mejor resultado?"):
    """Parámetros (escala log) vs métrica en train y test para las 3 redes."""
    clave = "accuracy" if tarea == "clasificacion" else "r2"
    x = [r["pesos"]["total"] for r in redes]
    fig = go.Figure()
    for conj, color, simb in [("train", COLOR_TRAIN, "circle"), ("test", COLOR_TEST, "square")]:
        fig.add_trace(go.Scatter(x=x, y=[r[conj][clave] for r in redes], mode="lines+markers+text", name=conj.capitalize(),
                                 text=[r["nombre"].split(" (")[0] for r in redes] if conj == "test" else None,
                                 textposition="bottom center", line=dict(color=color, width=2.5), marker=dict(size=11, symbol=simb)))
    fig.update_layout(title=titulo, template=PLANTILLA, height=400, xaxis=dict(type="log", title="Parámetros (pesos + sesgos)"),
                      yaxis_title="Accuracy" if clave == "accuracy" else "R²", hovermode="x unified")
    return fig


# ------------------------------------------------------------------ COMPARATIVA ENTRE ENTREGAS
def _color(m):
    return COLOR_E2 if m["entrega"] == 2 else COLOR_E1


def _eje_min(vals):
    """Límite izquierdo del eje: los modelos muy por debajo del resto quedan 'fuera de escala' (se ven con el mouse)."""
    return max(min(vals) - 0.03, float(np.median(vals)) - 0.15)


def _leyenda_entregas(fig):
    for nombre, c in [("Entrega 1", COLOR_E1), ("Entrega 2", COLOR_E2)]:
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=c, name=nombre, showlegend=True))


def figura_barras_comparativa(modelos, clave, titulo, eje, etiqueta):
    """Barras horizontales ordenadas (azul = E1, naranja = E2). Si un modelo es atípico muy bajo, el eje se recorta."""
    orden = sorted(modelos, key=lambda m: m[clave])
    vals = [m[clave] for m in orden]
    xmin = _eje_min(vals)
    textos = [f"{v:.4f}" + ("  ← fuera de escala" if v < xmin else "") for v in vals]
    fig = go.Figure(go.Bar(y=[etiqueta(m) for m in orden], x=[max(v, xmin) for v in vals], orientation="h",
                           marker_color=[_color(m) for m in orden], text=textos, textposition="outside", cliponaxis=False,
                           customdata=vals, hovertemplate="%{y}<br>" + eje + " = %{customdata:.4f}<extra></extra>", showlegend=False))
    _leyenda_entregas(fig)
    fig.update_layout(title=titulo, template=PLANTILLA, height=28 * len(orden) + 160, xaxis=dict(title=eje, range=[xmin, max(vals) + 0.02]),
                      legend=dict(orientation="h", y=-0.12), margin=dict(l=10))
    return fig


def figura_brecha(modelos, clave_tr, clave_te, titulo, etiqueta):
    """Train vs test por modelo: la distancia entre las barras es el sobreajuste."""
    orden = sorted([m for m in modelos if m.get(clave_tr) is not None], key=lambda m: m[clave_te])
    nombres = [etiqueta(m) for m in orden]
    xmin = _eje_min([m[clave_te] for m in orden])
    fig = go.Figure()
    for conj, clave, color in (("Train", clave_tr, COLOR_TRAIN), ("Test", clave_te, COLOR_TEST)):
        vals = [m[clave] for m in orden]
        fig.add_trace(go.Bar(y=nombres, x=[max(v, xmin) for v in vals], orientation="h", name=conj, marker_color=color,
                             text=[f"{v:.3f} ← fuera de escala" if v < xmin else "" for v in vals], textposition="outside",
                             cliponaxis=False,
                             customdata=np.c_[vals, [m[clave_tr] - m[clave_te] for m in orden]],
                             hovertemplate="%{y}<br>" + conj.lower() + " = %{customdata[0]:.4f}<br>brecha = %{customdata[1]:.4f}<extra></extra>"))
    fig.update_layout(title=titulo, template=PLANTILLA, barmode="group", height=34 * len(orden) + 160,
                      xaxis=dict(range=[xmin, 1.01]), legend=dict(orientation="h", y=-0.12), margin=dict(l=10))
    return fig


def figura_prf(mejores, titulo="Precision, Recall y F1 (macro) — mejor modelo de cada familia"):
    x = [f"{m['familia']} (E{m['entrega']})" for m in mejores]
    fig = go.Figure()
    for clave, nombre, c in [("precision_macro", "Precision", COLOR_CV), ("recall_macro", "Recall", COLOR_TEST), ("f1_macro", "F1", COLOR_TRAIN)]:
        fig.add_trace(go.Bar(x=x, y=[m[clave] for m in mejores], name=nombre, marker_color=c,
                             hovertemplate="%{x}<br>" + nombre + " = %{y:.4f}<extra></extra>"))
    fig.update_layout(title=titulo, template=PLANTILLA, barmode="group", height=430,
                      yaxis_range=[min(m["f1_macro"] for m in mejores) - 0.02, 1.003], legend=dict(orientation="h", y=-0.25))
    return fig


def figura_errores_regresion(modelos, etiqueta):
    orden = sorted(modelos, key=lambda m: -m["mae_test"])
    nombres = [etiqueta(m) for m in orden]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=nombres, x=[m["rmse_test"] for m in orden], orientation="h", name="RMSE", marker_color=COLOR_TRAIN))
    fig.add_trace(go.Bar(y=nombres, x=[m["mae_test"] for m in orden], orientation="h", name="MAE", marker_color=COLOR_CV))
    fig.update_layout(title="MAE y RMSE por modelo (MPa, menor es mejor)", template=PLANTILLA, barmode="group",
                      height=34 * len(orden) + 160, legend=dict(orientation="h", y=-0.12), margin=dict(l=10))
    return fig


def figura_tiempo_vs_metrica(modelos, clave, eje, etiqueta, titulo="Costo vs desempeño"):
    """Tiempo de entrenamiento (log) vs métrica: muestra qué modelos dan más por menos."""
    fig = go.Figure()
    for ent, c in [(1, COLOR_E1), (2, COLOR_E2)]:
        ms = [m for m in modelos if m["entrega"] == ent and m.get("tiempo_s")]
        fig.add_trace(go.Scatter(x=[max(m["tiempo_s"], 1e-4) for m in ms], y=[m[clave] for m in ms], mode="markers",
                                 name=f"Entrega {ent}", marker=dict(size=12, color=c), text=[etiqueta(m) for m in ms],
                                 hovertemplate="%{text}<br>tiempo = %{x:.4f} s<br>" + eje + " = %{y:.4f}<extra></extra>"))
    fig.update_layout(title=titulo, template=PLANTILLA, height=430, xaxis=dict(type="log", title="Tiempo de entrenamiento (s, escala log)"),
                      yaxis_title=eje, legend=dict(orientation="h", y=-0.2))
    return fig


def figura_residuos(y_real, y_pred, titulo="Distribución de residuos (real − predicho)", color=COLOR_TRAIN):
    """Histograma de los errores: centrado en 0 y simétrico = sin sesgo; colas largas = errores grandes."""
    res = np.asarray(y_real, dtype=float) - np.asarray(y_pred, dtype=float)
    fig = go.Figure(go.Histogram(x=res, nbinsx=30, marker_color=color, opacity=0.85,
                                 hovertemplate="residuo %{x}<br>%{y} muestras<extra></extra>"))
    fig.add_vline(x=0, line_dash="dash", line_color="#555")
    fig.add_vline(x=float(res.mean()), line_dash="dot", line_color=COLOR_TEST,
                  annotation_text=f"media {res.mean():+.2f}", annotation_position="top right")
    fig.update_layout(title=titulo, template=PLANTILLA, height=420, xaxis_title="residuo (MPa)",
                      yaxis_title="nº de muestras", bargap=0.05)
    return fig


def figura_regiones_decision(r, titulo=""):
    """Regiones de decisión de un SVC entrenado sobre las 2 primeras componentes principales (PCA)."""
    import plotly.express as px
    clases = list(r["clases"])
    paleta = (px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24)[:len(clases)]
    escala = []
    for i, c in enumerate(paleta):                       # escala discreta: un color por índice de clase
        escala += [[i / len(clases), c], [(i + 1) / len(clases), c]]
    fig = go.Figure(go.Heatmap(x=r["xs"], y=r["ys"], z=r["Z"], zmin=-0.5, zmax=len(clases) - 0.5, colorscale=escala,
                               opacity=0.35, showscale=False, hoverinfo="skip"))
    idx = np.searchsorted(clases, r["y_test"])
    for i, c in enumerate(clases):
        m = idx == i
        fig.add_trace(go.Scatter(x=r["X_test_2d"][m, 0], y=r["X_test_2d"][m, 1], mode="markers", name=str(c),
                                 marker=dict(size=7, color=paleta[i], line=dict(color="white", width=0.6)),
                                 hovertemplate=f"real: {c}<extra></extra>"))
    v = r["varianza"]
    fig.update_layout(title=titulo, template=PLANTILLA, height=620, legend=dict(font=dict(size=10)),
                      xaxis_title=f"PC1 ({v[0]:.1%} de la varianza)", yaxis_title=f"PC2 ({v[1]:.1%} de la varianza)")
    return fig
