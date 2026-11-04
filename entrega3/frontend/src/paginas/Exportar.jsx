import { useEffect, useState } from "react";
import { COLORES_CLASE, api } from "../api.js";

// Resumen del dataset y exportación a la carpeta datos/dataset en formato YOLO.
export default function Exportar() {
  const [resumen, setResumen] = useState(null);
  const [param, setParam] = useState({ val: 0.2, test: 0.1, semilla: 42 });
  const [resultado, setResultado] = useState(null);
  const [exportando, setExportando] = useState(false);
  const [error, setError] = useState("");

  const cargar = () => api.resumenDataset().then(setResumen).catch((e) => setError(e.message));
  useEffect(() => {
    cargar();
  }, []);

  async function exportar() {
    setExportando(true);
    setError("");
    try {
      setResultado(await api.exportarDataset(param));
      cargar();
    } catch (e) {
      setError(e.message);
      setResultado(null);
    } finally {
      setExportando(false);
    }
  }

  if (!resumen) return error ? <div className="alerta error">{error}</div> : <p>Cargando…</p>;

  const maxCajas = Math.max(1, ...resumen.clases.map((c) => c.cajas));
  const totalCajas = resumen.clases.reduce((s, c) => s + c.cajas, 0);
  const META = 150; // mínimo recomendado de cajas por categoría (docs/03_dataset.md)

  return (
    <>
      <div className="encabezado">
        <div>
          <h1>Exportar dataset</h1>
          <p className="subtitulo">Genera <code>datos/dataset</code> en formato YOLO, listo para entrenar.</p>
        </div>
      </div>
      {error && <div className="alerta error">{error}</div>}

      <div className="rejilla" style={{ marginBottom: 18 }}>
        <div className="panel"><div className="subtitulo">Imágenes etiquetadas</div><h1>{resumen.imagenes.etiquetada}</h1></div>
        <div className="panel"><div className="subtitulo">Pendientes</div><h1>{resumen.imagenes.pendiente}</h1></div>
        <div className="panel"><div className="subtitulo">Cajas en total</div><h1>{totalCajas}</h1></div>
      </div>

      <div className="panel tabla-scroll">
        <h2>Cajas por categoría</h2>
        <p className="subtitulo">
          Un dataset equilibrado evita que el modelo favorezca a la categoría con más ejemplos. Meta mínima: {META} cajas por categoría.
        </p>
        <table>
          <thead>
            <tr><th>Categoría</th><th>Producto</th><th className="num">Cajas</th><th style={{ width: "40%" }}></th><th>Meta</th></tr>
          </thead>
          <tbody>
            {resumen.clases.map((c) => (
              <tr key={c.clase_id}>
                <td><span className="punto" style={{ background: COLORES_CLASE[c.clase_id] }} /><code>{c.categoria}</code></td>
                <td>{c.nombre}</td>
                <td className="num">{c.cajas}</td>
                <td>
                  <div className="barra"><div style={{ width: `${(100 * c.cajas) / maxCajas}%`, background: COLORES_CLASE[c.clase_id] }} /></div>
                </td>
                <td>{c.cajas >= META ? <span className="etiqueta ok">cumple</span> : <span className="etiqueta aviso">faltan {META - c.cajas}</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <h2>División y exportación</h2>
        <p className="subtitulo">
          La división se hace por video (no por imagen): los fotogramas de un mismo video son casi iguales y
          repartirlos entre entrenamiento y validación inflaría las métricas. Las sesiones de prueba de
          generalización van completas a <code>test</code>.
        </p>
        <div className="fila" style={{ alignItems: "flex-end", marginTop: 10 }}>
          <label className="campo">
            Validación: {Math.round(param.val * 100)} %
            <input type="range" min="0.05" max="0.4" step="0.05" value={param.val}
              onChange={(e) => setParam({ ...param, val: Number(e.target.value) })} />
          </label>
          <label className="campo">
            Prueba: {Math.round(param.test * 100)} %
            <input type="range" min="0" max="0.3" step="0.05" value={param.test}
              onChange={(e) => setParam({ ...param, test: Number(e.target.value) })} />
          </label>
          <label className="campo">
            Semilla
            <input type="number" value={param.semilla} style={{ width: 90 }}
              onChange={(e) => setParam({ ...param, semilla: Number(e.target.value) })} />
          </label>
          <span className="subtitulo">Entrenamiento: {Math.round((1 - param.val - param.test) * 100)} %</span>
          <span className="espaciador" />
          <button className="primario" disabled={exportando || resumen.imagenes.etiquetada < 3} onClick={exportar}>
            {exportando ? "Exportando…" : "Exportar dataset"}
          </button>
        </div>
      </div>

      {resultado && (
        <div className="panel">
          <h2>Resultado</h2>
          {resultado.advertencias.map((a) => <div key={a} className="alerta aviso">{a}</div>)}
          <table>
            <thead>
              <tr><th>Partición</th><th className="num">Imágenes</th>{resumen.clases.map((c) => <th key={c.clase_id} className="num">{c.categoria}</th>)}</tr>
            </thead>
            <tbody>
              {Object.entries(resultado.particiones).map(([p, r]) => (
                <tr key={p}>
                  <td><code>{p}</code></td>
                  <td className="num">{r.imagenes}</td>
                  {resumen.clases.map((c) => <td key={c.clase_id} className="num">{r.cajas[c.categoria] ?? 0}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ marginTop: 12 }}>Dataset escrito en <code>{resultado.ruta}</code> con su archivo <code>data.yaml</code>.</p>
        </div>
      )}
    </>
  );
}
