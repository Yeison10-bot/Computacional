# Segunda entrega — Inteligencia Computacional 2

Usa las MISMAS bases de datos y la MISMA partición que la primera entrega
(`data/Crop_recommendation.csv` → clasificación, `data/concreto.csv` → regresión; 80/20, `random_state=42`).

## Archivos
| Archivo | Contenido |
|---|---|
| `preprocesamiento.py` | Carga, partición 80/20 (estratificada en clasificación) y estandarización ajustada solo con train |
| `metricas.py` | Accuracy, precision, recall, F1, matriz de confusión · MAE, MSE, RMSE, R², R² ajustado |
| `naive_bayes.py` | Naive Bayes gaussiano **hecho a mano** (a priori, condicionales, evidencia, a posteriori), verificado contra scikit-learn |
| `svm.py` | SVC y SVR: una búsqueda conjunta kernel × C × gamma × epsilon (CV 5 pliegues); el modelo final usa la mejor combinación |
| `redes_neuronales.py` | 3 redes por tarea: básica (8,) · intermedia (64, 32) · compleja (256, 128, 64); curvas por época y forward a mano |
| `entrega1_resultados.py` | Re-entrena todos los modelos de la entrega 1 con su configuración original y los guarda en JSON |
| `comparativa.py` | Tablas entrega 1 vs entrega 2 |
| `graficos.py` | Gráficas interactivas (Plotly), mismo estilo que la entrega 1 |
| `main.py` | Corre todo y guarda la evidencia en `resultados/` (HTML interactivo + PNG) |
| `app.py` | Interfaz Streamlit con palabras clave por tema |

## Cómo correrlo
    pip install -r requirements.txt
    python main.py          # ~3-4 min; genera tablas y gráficas en resultados/
    streamlit run app.py    # interfaz interactiva
