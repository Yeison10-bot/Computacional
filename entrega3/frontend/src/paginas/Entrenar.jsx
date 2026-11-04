import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, fechaHora } from "../api.js";

const FASES = {
  preparando: ["Preparando", "aviso"], entrenando: ["Entrenando", "aviso"], evaluando: ["Evaluando", "aviso"],
  terminado: ["Terminado", "ok"], error: ["Error", "error"], detenido: ["Detenido", ""], interrumpido: ["Interrumpido", ""],
};
const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)} %`);
const duracion = (s) => (s == null ? "—" : s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`
  : s >= 60 ? `${Math.round(s / 60)} min` : `${s} s`);

// Gráfica de líneas sencilla en SVG: una serie por clave del historial.
function Curvas({ historial, series, titulo }) {
  const A = 520, H = 180, M = { i: 40, d: 10, s: 10, b: 24 };
  const valores = historial.flatMap((h) => series.map((s) => h[s.clave])).filter((v) => v != null);
  if (historial.length < 2 || !valores.length) return null;
  const max = Math.max(...valores) * 1.05 || 1;
  const ult = historial.at(-1).epoca;
  const x = (e) => M.i + ((e - 1) / Math.max(1, ult - 1)) * (A - M.i - M.d);
  const y = (v) => M.s + (1 - v / max) * (H - M.s - M.b);
  return (
    <div>
      <div className="fila" style={{ justifyContent: "space-between" }}>
        <b>{titulo}</b>
        <span className="fila" style={{ gap: 12, fontSize: "0.8rem" }}>
          {series.map((s) => (
            <span key={s.clave}><span className="punto" style={{ background: s.color }} />{s.nombre}</span>
          ))}
        </span>
      </div>
      <svg viewBox={`0 0 ${A} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label={titulo}>
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line x1={M.i} x2={A - M.d} y1={y(max * f)} y2={y(max * f)} stroke="var(--borde)" />
            <text x={M.i - 6} y={y(max * f) + 4} textAnchor="end" fontSize="10" fill="var(--texto-suave)">{(max * f).toFixed(2)}</text>
          </g>
        ))}
        <text x={M.i} y={H - 6} fontSize="10" fill="var(--texto-suave)">época 1</text>
        <text x={A - M.d} y={H - 6} textAnchor="end" fontSize="10" fill="var(--texto-suave)">época {ult}</text>
        {series.map((s) => (
          <polyline key={s.clave} fill="none" stroke={s.color} strokeWidth="2"
            points={historial.filter((h) => h[s.clave] != null).map((h) => `${x(h.epoca)},${y(h[s.clave])}`).join(" ")} />
        ))}
      </svg>
    </div>
  );
}

function TablaMetricas({ titulo, m }) {
  return (
    <div className="tabla-scroll">
      <h3>{titulo}</h3>
      <table>
        <thead><tr><th>Categoría</th><th className="num">Precisión</th><th className="num">Recall</th><th className="num">mAP50</th><th className="num">mAP50-95</th></tr></thead>
        <tbody>
          {m.por_clase.map((c) => (
            <tr key={c.categoria}><td><code>{c.categoria}</code></td><td className="num">{pct(c.precision)}</td>
              <td className="num">{pct(c.recall)}</td><td className="num">{pct(c.mAP50)}</td><td className="num">{pct(c.mAP50_95)}</td></tr>
          ))}
          <tr><td><b>Todas</b></td><td className="num"><b>{pct(m.precision)}</b></td><td className="num"><b>{pct(m.recall)}</b></td>
            <td className="num"><b>{pct(m.mAP50)}</b></td><td className="num"><b>{pct(m.mAP50_95)}</b></td></tr>
        </tbody>
      </table>
      {m.inferencia_ms != null && <p className="subtitulo">Inferencia: {m.inferencia_ms} ms por imagen.</p>}
    </div>
  );
}

// Entrena YOLO con el dataset exportado y muestra el avance época por época.
export default function Entrenar() {
  const [req, setReq] = useState(null);
  const [lista, setLista] = useState([]);
  const [elegido, setElegido] = useState(null);
  const [detalle, setDetalle] = useState(null);
  const [param, setParam] = useState({ epocas: 50, imgsz: 640, lote: 8, paciencia: 15, base: "coco" });
  const [error, setError] = useState("");
  const [enviando, setEnviando] = useState(false);

  const cargar = () => {
    api.requisitosEntrenamiento().then(setReq).catch((e) => setError(e.message));
    api.entrenamientos().then((l) => {
      setLista(l);
      setElegido((actual) => actual ?? l[0]?.id ?? null);
    }).catch((e) => setError(e.message));
  };
  useEffect(cargar, []);

  const activo = detalle?.activo;
  useEffect(() => {
    if (!elegido) return;
    const traer = () => api.entrenamiento(elegido).then(setDetalle).catch((e) => setError(e.message));
    traer();
    // Mientras entrena se consulta el avance cada 3 s.
    if (!activo) return;
    const t = setInterval(traer, 3000);
    return () => clearInterval(t);
  }, [elegido, activo]);
  useEffect(() => { if (detalle && !detalle.activo) cargar(); }, [detalle?.fase]);

  async function iniciar() {
    setEnviando(true); setError("");
    try {
      const { id } = await api.iniciarEntrenamiento(param);
      setDetalle(null); setElegido(id); cargar();
      setTimeout(() => api.entrenamiento(id).then(setDetalle), 800);
    } catch (e) { setError(e.message); } finally { setEnviando(false); }
  }
  async function detener() {
    if (!confirm("¿Detener el entrenamiento? Se pierde el avance y el modelo en uso no cambia.")) return;
    try { setDetalle(await api.detenerEntrenamiento()); cargar(); } catch (e) { setError(e.message); }
  }

  if (!req) return error ? <div className="alerta error">{error}</div> : <p>Cargando…</p>;
  const sinDataset = !req.dataset_exportado || !req.imagenes.train || !req.imagenes.val;
  const bloqueado = sinDataset || req.monitor_activo || req.entrenando || enviando;
  const campo = (clave, rotulo, attrs) => (
    <label className="campo">{rotulo}
      <input type="number" value={param[clave]} style={{ width: 90 }} {...attrs}
        onChange={(e) => setParam({ ...param, [clave]: Number(e.target.value) })} />
    </label>
  );
  const [nombreFase, claseFase] = FASES[detalle?.fase] ?? [detalle?.fase, ""];

  return (
    <>
      <div className="encabezado">
        <div>
          <h1>Entrenar modelo</h1>
          <p className="subtitulo">YOLO11n con transfer learning sobre el dataset exportado. Al terminar, el modelo nuevo
            reemplaza al provisional en la vitrina y en el conteo por foto.</p>
        </div>
      </div>
      {error && <div className="alerta error">{error}</div>}

      <div className="panel">
        <h2>Nuevo entrenamiento</h2>
        {!req.dataset_exportado ? (
          <div className="alerta aviso"><span>Primero etiqueta imágenes y genera el dataset en <Link to="/exportar">Exportar dataset</Link>.</span></div>
        ) : (
          <p className="subtitulo">
            Dataset exportado: {req.imagenes.train} imágenes de entrenamiento, {req.imagenes.val} de validación
            y {req.imagenes.test} de prueba. Si etiquetaste más desde entonces, vuelve a <Link to="/exportar">exportar</Link>.
          </p>
        )}
        {req.dataset_exportado && sinDataset && <div className="alerta aviso">Entrenamiento y validación no pueden estar vacíos.</div>}
        {req.monitor_activo && <div className="alerta aviso">El monitoreo de la vitrina está activo: detenlo antes de entrenar (comparten la GPU).</div>}
        <div className="fila" style={{ alignItems: "flex-end", marginTop: 10 }}>
          {campo("epocas", "Épocas", { min: 1, max: 300 })}
          <label className="campo">Tamaño de imagen
            <select value={param.imgsz} onChange={(e) => setParam({ ...param, imgsz: Number(e.target.value) })}>
              {[416, 512, 640, 800].map((v) => <option key={v} value={v}>{v} px</option>)}
            </select>
          </label>
          {campo("lote", "Lote", { min: 1, max: 64 })}
          {campo("paciencia", "Paciencia", { min: 0, max: 100, title: "Épocas sin mejorar antes de parar (0 = nunca)" })}
          <label className="campo">Partir de
            <select value={param.base} onChange={(e) => setParam({ ...param, base: e.target.value })}>
              <option value="coco">yolo11n.pt (COCO)</option>
              <option value="propio" disabled={!req.modelo_propio}>modelo propio actual</option>
            </select>
          </label>
          <span className="espaciador" />
          <button className="primario" disabled={bloqueado} onClick={iniciar}>
            {req.entrenando ? "Entrenando…" : "Iniciar entrenamiento"}
          </button>
        </div>
        <p className="subtitulo" style={{ marginTop: 10 }}>
          Lote 8 con 640 px cabe en los 4 GB de la RTX 3050; si sale «sin memoria», baja el lote a 4. La paciencia detiene
          el entrenamiento cuando el mAP de validación deja de mejorar.
        </p>
      </div>

      {detalle && (
        <div className="panel">
          <div className="panel-titulo">
            <h2>Entrenamiento {detalle.id}</h2>
            <div className="fila">
              <span className={`etiqueta ${claseFase}`}>{nombreFase}</span>
              {detalle.activo && <button onClick={detener}>Detener</button>}
            </div>
          </div>
          <p className="subtitulo">
            {detalle.mensaje} · partió de <code>{detalle.parametros?.modelo_base}</code>, {detalle.parametros?.imgsz} px, lote {detalle.parametros?.lote}
            {detalle.iniciado_en && <> · iniciado {fechaHora.format(new Date(detalle.iniciado_en))}</>}
          </p>
          {detalle.error && <div className="alerta error"><span>{detalle.error} (detalles en <code>datos/entrenamientos/{detalle.id}/registro.txt</code>)</span></div>}
          {detalle.activo && (
            <div style={{ margin: "12px 0" }}>
              <div className="barra"><div style={{ width: `${(100 * detalle.epoca) / detalle.epocas}%` }} /></div>
              <p className="subtitulo">Época {detalle.epoca} de {detalle.epocas}
                {detalle.restante_s != null && <> · faltan unos {duracion(detalle.restante_s)}</>}</p>
            </div>
          )}

          {detalle.historial?.length > 0 && (
            <>
              <div className="rejilla" style={{ marginBottom: 14 }}>
                {["mAP50", "mAP50_95", "precision", "recall"].map((k) => (
                  <div key={k} className="indicador"><div className="rotulo">{k.replace("_", "-")} (validación)</div>
                    <div className="valor">{pct(detalle.historial.at(-1)[k])}</div></div>
                ))}
              </div>
              <div className="rejilla">
                <Curvas titulo="Pérdidas" historial={detalle.historial} series={[
                  { clave: "box_loss", nombre: "caja (train)", color: "var(--clase-0)" },
                  { clave: "cls_loss", nombre: "clase (train)", color: "var(--clase-1)" },
                  { clave: "val_box_loss", nombre: "caja (val)", color: "var(--clase-2)" },
                ]} />
                <Curvas titulo="Métricas de validación" historial={detalle.historial} series={[
                  { clave: "mAP50", nombre: "mAP50", color: "var(--clase-3)" },
                  { clave: "mAP50_95", nombre: "mAP50-95", color: "var(--clase-4)" },
                ]} />
              </div>
            </>
          )}

          {detalle.resultado && (
            <>
              <div className="alerta ok" style={{ marginTop: 14 }}><span>
                Modelo guardado en <code>{detalle.resultado.modelo}</code> ({detalle.resultado.epocas_completadas} épocas en {duracion(detalle.resultado.duracion_s)}, {detalle.resultado.dispositivo}).
                La vitrina lo usará al volver a iniciar el monitoreo.</span>
              </div>
              <TablaMetricas titulo="Validación" m={detalle.resultado.val} />
              {detalle.resultado.test
                ? <TablaMetricas titulo="Prueba (imágenes nunca vistas)" m={detalle.resultado.test} />
                : <p className="subtitulo">No hay imágenes en la partición de prueba: no se evaluó en test.</p>}
            </>
          )}

          {detalle.graficas?.length > 0 && (
            <>
              <h3>Gráficas para el informe</h3>
              <div className="rejilla">
                {detalle.graficas.map((g) => (
                  <a key={g.url} href={g.url} target="_blank" rel="noreferrer">
                    <img src={g.url} alt={g.nombre} style={{ width: "100%", borderRadius: 8, border: "1px solid var(--borde)" }} />
                    <div className="subtitulo">{g.nombre}</div>
                  </a>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {lista.length > 1 && (
        <div className="panel tabla-scroll">
          <h2>Entrenamientos anteriores</h2>
          <table>
            <thead><tr><th>Fecha</th><th>Estado</th><th className="num">Épocas</th><th className="num">mAP50 (val)</th><th className="num">mAP50 (test)</th><th></th></tr></thead>
            <tbody>
              {lista.map((e) => (
                <tr key={e.id}>
                  <td>{e.iniciado_en ? fechaHora.format(new Date(e.iniciado_en)) : e.id}</td>
                  <td><span className={`etiqueta ${(FASES[e.fase] ?? [])[1] ?? ""}`}>{(FASES[e.fase] ?? [e.fase])[0]}</span></td>
                  <td className="num">{e.epoca} / {e.epocas}</td>
                  <td className="num">{pct(e.resultado?.val?.mAP50)}</td>
                  <td className="num">{pct(e.resultado?.test?.mAP50)}</td>
                  <td>{e.id !== elegido && <button onClick={() => { setDetalle(null); setElegido(e.id); }}>Ver</button>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
