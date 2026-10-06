"""
graficos.py
Funciones de graficación interactiva (Plotly) compartidas por los scripts
del proyecto y por la app de Streamlit.

- figura_matriz_confusion: matriz de confusión legible para muchas clases
  (celdas en cero vacías, aciertos y errores en escalas de color separadas,
  escala logarítmica para que los errores pequeños sí se noten, números
  discretos que se pueden ocultar con un botón).
- figura_curva_poda, figura_real_vs_predicho, figura_convergencia: versiones
  interactivas de los gráficos que antes eran imágenes de matplotlib.
- guardar_figura: guarda cada figura como HTML interactivo + PNG estático.
"""

import numpy as np
import plotly.graph_objects as go

COLOR_TRAIN = "#a8552f"
COLOR_TEST = "#3f7d54"
COLOR_OPTIMO = "#2a7f2a"
COLOR_CV = "#3f5f7d"

# Escala de color de los ACIERTOS (diagonal) según el modelo.
# Empiezan en un tono medio-claro (no blanco) para que ninguna celda con
# aciertos se confunda con una celda vacía.
ESCALAS_ACIERTOS = {
    "Blues":   ["#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    "Greens":  ["#c7e9c0", "#74c476", "#238b45", "#00441b"],
    "Oranges": ["#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"],
    "Purples": ["#dadaeb", "#9e9ac8", "#6a51a3", "#3f007d"],
}
# Los ERRORES (fuera de la diagonal) siempre en rojo, para que resalten.
ESCALA_ERRORES = ["#fcbba1", "#fb6a4a", "#de2d26", "#a50f15"]


def _escala(colores):
    pasos = np.linspace(0, 1, len(colores))
    return [[float(p), c] for p, c in zip(pasos, colores)]


def _ticks_log(maximo):
    """Marcas legibles (1, 2, 5, 10, 20...) para una barra de color logarítmica."""
    candidatos = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valores = [v for v in candidatos if v <= maximo] or [1]
    if maximo / valores[-1] >= 1.5:
        valores.append(int(maximo))
    return [float(np.log10(v)) for v in valores], [str(v) for v in valores]


def figura_matriz_confusion(cm, labels, titulo="Matriz de confusión", color="Blues",
                            eje_x="Predicción", eje_y="Real", mostrar_numeros=True):
    """
    Matriz de confusión interactiva.
    - Diagonal (aciertos) en la escala `color`; fuera de la diagonal (errores) en rojo.
    - Las celdas en cero no se colorean ni se rotulan (quedan como cuadros grises claros).
    - Los errores usan escala logarítmica: 1 error ya se ve, sin que 10 errores saturen.
    - Al pasar el mouse se ve la cantidad y el % sobre el total de esa clase real.
    """
    cm = np.asarray(cm)
    n = len(labels)
    labels = [str(l) for l in labels]
    es_diag = np.eye(n, dtype=bool)
    totales_fila = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, totales_fila, out=np.zeros(cm.shape, dtype=float), where=totales_fila > 0)
    info = np.dstack([cm, pct * 100])
    hover = ("Real: <b>%{y}</b><br>Predicho: <b>%{x}</b><br>"
             "Cantidad: <b>%{customdata[0]:.0f}</b> (%{customdata[1]:.1f}% de la clase real)"
             "<extra></extra>")

    z_aciertos = np.where(es_diag & (cm > 0), cm, np.nan).astype(float)
    z_errores = np.where(~es_diag & (cm > 0), cm, np.nan).astype(float)
    max_acierto = float(np.nanmax(z_aciertos)) if np.any(~np.isnan(z_aciertos)) else 1.0
    max_error = float(np.nanmax(z_errores)) if np.any(~np.isnan(z_errores)) else 1.0
    log_max_error = max(np.log10(max_error), 0.3)
    tick_vals, tick_text = _ticks_log(max_error)

    comunes = dict(x=labels, y=labels, xgap=1.5, ygap=1.5, customdata=info, hovertemplate=hover)
    fig = go.Figure()
    # Fondo: todas las celdas en gris muy claro (así las celdas en cero se ven como cuadrícula vacía)
    fig.add_trace(go.Heatmap(z=np.zeros((n, n)), colorscale=[[0, "#f1f3f5"], [1, "#f1f3f5"]],
                             showscale=False, **comunes))
    fig.add_trace(go.Heatmap(
        z=z_aciertos, colorscale=_escala(ESCALAS_ACIERTOS.get(color, ESCALAS_ACIERTOS["Blues"])),
        zmin=0, zmax=max_acierto, name="Aciertos",
        colorbar=dict(title=dict(text="Aciertos", side="right"), x=1.0, len=0.45, y=0.78,
                      thickness=14, outlinewidth=0),
        **comunes))
    fig.add_trace(go.Heatmap(
        z=np.log10(z_errores), colorscale=_escala(ESCALA_ERRORES),
        zmin=0, zmax=log_max_error, name="Errores",
        colorbar=dict(title=dict(text="Errores", side="right"), x=1.0, len=0.45, y=0.22,
                      thickness=14, outlinewidth=0, tickvals=tick_vals, ticktext=tick_text),
        **comunes))

    # Números discretos: pequeños, semitransparentes y solo en celdas distintas de cero.
    anotaciones = []
    for i in range(n):
        for j in range(n):
            v = int(cm[i, j])
            if v == 0:
                continue
            if es_diag[i, j]:
                oscuro = v / max_acierto > 0.45
            else:
                oscuro = np.log10(v) / log_max_error > 0.65
            anotaciones.append(dict(
                x=labels[j], y=labels[i], text=str(v), showarrow=False,
                font=dict(size=9 if n > 10 else 12,
                          color="rgba(255,255,255,0.7)" if oscuro else "rgba(40,40,40,0.55)")))

    fig.update_layout(
        title=dict(text=titulo, x=0.01),
        template="plotly_white",
        plot_bgcolor="white",
        height=max(420, min(820, 30 * n + 220)),
        margin=dict(l=10, r=10, t=90, b=10),
        annotations=anotaciones if mostrar_numeros else [],
        updatemenus=[dict(
            type="buttons", direction="left", x=1, xanchor="right", y=1.01, yanchor="bottom",
            showactive=True, pad=dict(r=4, t=0), font=dict(size=11),
            buttons=[
                dict(label="Con números", method="relayout", args=[{"annotations": anotaciones}]),
                dict(label="Solo colores", method="relayout", args=[{"annotations": []}]),
            ],
            active=0 if mostrar_numeros else 1,
        )],
    )
    fig.update_xaxes(title_text=eje_x, tickangle=-60 if n > 6 else 0, showgrid=False,
                     tickfont=dict(size=10))
    fig.update_yaxes(title_text=eje_y, autorange="reversed", showgrid=False, tickfont=dict(size=10))
    return fig


def figura_curva_poda(alphas, metrica_train, metrica_test, nombre_metrica="Accuracy", titulo="",
                      metrica_cv=None):
    """
    Curva ccp_alpha vs métrica (train y test), con la brecha sombreada y el óptimo marcado.

    Si se pasa `metrica_cv` (promedio de validación cruzada sobre el TRAIN), el óptimo se
    elige con esa curva y no con el test: así el test queda como evaluación final honesta.
    """
    alphas = list(map(float, alphas))
    metrica_train = list(map(float, metrica_train))
    metrica_test = list(map(float, metrica_test))
    if metrica_cv is not None:
        metrica_cv = list(map(float, metrica_cv))
    curva_eleccion = metrica_cv if metrica_cv is not None else metrica_test
    mejor = int(np.argmax(curva_eleccion))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=alphas, y=metrica_train, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=alphas, y=metrica_test, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(214,39,40,0.12)",
                             name="Brecha (overfitting)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=alphas, y=metrica_train, mode="lines+markers", name="Train",
                             line=dict(color=COLOR_TRAIN, width=2.5), marker=dict(size=7),
                             hovertemplate=f"{nombre_metrica} train: %{{y:.4f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=alphas, y=metrica_test, mode="lines+markers", name="Test",
                             line=dict(color=COLOR_TEST, width=2.5), marker=dict(size=7, symbol="square"),
                             hovertemplate=f"{nombre_metrica} test: %{{y:.4f}}<extra></extra>"))
    if metrica_cv is not None:
        fig.add_trace(go.Scatter(x=alphas, y=metrica_cv, mode="lines+markers",
                                 name="Validación cruzada (train)",
                                 line=dict(color=COLOR_CV, width=2.5, dash="dash"),
                                 marker=dict(size=7, symbol="diamond"),
                                 hovertemplate=f"{nombre_metrica} CV: %{{y:.4f}}<extra></extra>"))
    texto_cv = f"CV: {metrica_cv[mejor]:.4f}<br>" if metrica_cv is not None else ""
    fig.add_trace(go.Scatter(
        x=[alphas[mejor]], y=[curva_eleccion[mejor]], mode="markers",
        name=(f"Elegido por CV (alpha={alphas[mejor]:.6f})" if metrica_cv is not None
              else f"Óptimo (alpha={alphas[mejor]:.6f})"),
        marker=dict(size=20, color=COLOR_OPTIMO, symbol="star", line=dict(color="white", width=1)),
        hovertemplate=(f"<b>ÓPTIMO</b><br>alpha = {alphas[mejor]:.6f}<br>{texto_cv}"
                       f"Test: {metrica_test[mejor]:.4f}<br>Train: {metrica_train[mejor]:.4f}"
                       "<extra></extra>")))
    fig.update_layout(
        title=titulo, template="plotly_white", height=500, hovermode="x unified",
        xaxis_title="ccp_alpha (nivel de poda)", yaxis_title=nombre_metrica,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def figura_real_vs_predicho(y_real, y_pred, titulo="", unidad="csMPa"):
    """Dispersión real vs predicho con la diagonal de predicción perfecta."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lim = [float(min(y_real.min(), y_pred.min())), float(max(y_real.max(), y_pred.max()))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Predicción perfecta",
                             line=dict(dash="dash", color="#555"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=y_real, y=y_pred, mode="markers", name="Predicciones",
        marker=dict(size=7, color=COLOR_TRAIN, opacity=0.55),
        customdata=y_pred - y_real,
        hovertemplate=(f"Real: %{{x:.2f}} {unidad}<br>Predicho: %{{y:.2f}} {unidad}"
                       "<br>Error: %{customdata:+.2f}<extra></extra>")))
    fig.update_layout(title=titulo, template="plotly_white", height=560,
                      xaxis_title=f"{unidad} real", yaxis_title=f"{unidad} predicho",
                      legend=dict(orientation="h", y=-0.15))
    fig.update_yaxes(scaleanchor="x")
    return fig


def figura_convergencia(movimientos, titulo=""):
    """Desplazamiento promedio de los centroides por iteración de K-means."""
    iteraciones = list(range(1, len(movimientos) + 1))
    fig = go.Figure(go.Scatter(
        x=iteraciones, y=list(map(float, movimientos)), mode="lines+markers+text",
        line=dict(color=COLOR_TRAIN, width=3), marker=dict(size=11),
        text=[f"{m:.3f}" for m in movimientos], textposition="top center",
        textfont=dict(size=11, color="rgba(60,60,60,0.8)"),
        hovertemplate="Iteración %{x}<br>Desplazamiento: %{y:.4f}<extra></extra>"))
    fig.update_layout(title=titulo, template="plotly_white", height=480,
                      xaxis=dict(title="Iteración", tickmode="linear", dtick=1),
                      yaxis_title="Desplazamiento promedio de los centroides")
    return fig


def guardar_figura(fig, ruta_png, width=1000, height=None):
    """Guarda la figura como HTML interactivo (misma ruta, extensión .html) y como PNG."""
    ruta_html = ruta_png.rsplit(".", 1)[0] + "_interactivo.html"
    fig.write_html(ruta_html)
    # En la imagen estática los botones no sirven, así que se quitan
    fig_png = go.Figure(fig).update_layout(updatemenus=[])
    fig_png.write_image(ruta_png, width=width, height=height or fig.layout.height or 600, scale=1.5)
    print(f"Gráfico guardado en: {ruta_png}  (interactivo: {ruta_html})")
