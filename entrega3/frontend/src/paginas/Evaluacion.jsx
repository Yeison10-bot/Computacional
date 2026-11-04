import { useEffect, useRef, useState } from "react";
import { COLORES_CLASE, api, subirVideoPrueba } from "../api.js";
import { useAvisos } from "../componentes/Avisos.jsx";
import { IconoSubir } from "../componentes/Iconos.jsx";

const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)} %`);
const seg = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;
const RESULTADOS = {
  acierto: ["Acierto", "ok"], omitido: ["Omitido", "error"], falso: ["Falso positivo", "aviso"], duplicado: ["Duplicado", "aviso"],
};

function Indicador({ rotulo, valor, nota }) {
  return (
    <div className="indicador">
      <div className="rotulo">{rotulo}</div>
      <div className="valor">{valor}</div>
      {nota && <div className="subtitulo">{nota}</div>}
    </div>
  );
}

// Videos de prueba: el usuario marca los retiros reales y el sistema se evalúa contra ellos.
export default function Evaluacion() {
  const avisar = useAvisos();
  const [resumen, setResumen] = useState(null);
  const [videos, setVideos] = useState([]);
  const [productos, setProductos] = useState([]);
  const [elegido, setElegido] = useState(null);
  const [video, setVideo] = useState(null);
  const [eventos, setEventos] = useState([]);
  const [tolerancia, setTolerancia] = useState(15);
  const [cambios, setCambios] = useState(false);
  const [nuevo, setNuevo] = useState({ categoria: "", tipo: "venta", cantidad: 1 });
  const [subida, setSubida] = useState(null);
  const reproductor = useRef(null);

  const cargarLista = () => {
    api.videosPrueba().then(setVideos).catch((e) => avisar(e.message, "error"));
    api.resumenEvaluacion().then(setResumen).catch(() => {});
  };
  useEffect(() => {
    cargarLista();
    api.productos().then((p) => {
      setProductos(p);
      setNuevo((n) => ({ ...n, categoria: p[0]?.categoria_yolo ?? "" }));
    });
  }, []);

  const evaluando = video?.evaluando;
  useEffect(() => {
    if (!elegido) return;
    const traer = (inicial) => api.videoPrueba(elegido).then((v) => {
      setVideo(v);
      if (inicial) { setEventos(v.eventos_reales); setTolerancia(v.tolerancia_s); setCambios(false); }
      if (!v.evaluando && !inicial) cargarLista();
    });
    traer(true);
    if (!evaluando) return;
    const t = setInterval(() => traer(false), 1500);
    return () => clearInterval(t);
  }, [elegido, evaluando]);

  async function subir(e) {
    const archivo = e.target.files[0];
    e.target.value = "";
    if (!archivo) return;
    const nombre = prompt("Nombre del video de prueba (p. ej. «3 retiros con luz de día»):", archivo.name.replace(/\.[^.]+$/, ""));
    if (nombre === null) return;
    setSubida(0);
    try {
      const v = await subirVideoPrueba(archivo, nombre, setSubida);
      avisar(`Video «${v.nombre}» subido (${v.duracion_s} s)`);
      cargarLista();
      setElegido(v.id);
    } catch (err) { avisar(err.message, "error"); } finally { setSubida(null); }
  }

  function marcar() {
    const t = Math.round((reproductor.current?.currentTime ?? 0) * 10) / 10;
    setEventos((l) => [...l, { ...nuevo, t, nota: "" }].sort((a, b) => a.t - b.t));
    setCambios(true);
  }
  const editar = (i, campo, valor) => {
    setEventos((l) => l.map((e, j) => (j === i ? { ...e, [campo]: valor } : e)));
    setCambios(true);
  };
  const quitar = (i) => { setEventos((l) => l.filter((_, j) => j !== i)); setCambios(true); };
  const ir = (t) => { if (reproductor.current) { reproductor.current.currentTime = t; reproductor.current.pause(); } };

  async function guardar() {
    try {
      const v = await api.guardarEventosReales(elegido, eventos.map((e) => ({ ...e, t: Number(e.t), cantidad: Number(e.cantidad) })), Number(tolerancia));
      setVideo(v); setEventos(v.eventos_reales); setCambios(false); cargarLista();
      avisar("Eventos reales guardados");
      return true;
    } catch (e) { avisar(e.message, "error"); return false; }
  }
  async function evaluar() {
    if (cambios && !(await guardar())) return;
    try { setVideo(await api.evaluarVideo(elegido)); } catch (e) { avisar(e.message, "error"); }
  }
  async function borrar() {
    if (!confirm(`¿Borrar el video «${video.nombre}» y su evaluación?`)) return;
    try { await api.borrarVideoPrueba(elegido); setElegido(null); setVideo(null); cargarLista(); } catch (e) { avisar(e.message, "error"); }
  }

  const color = (cat) => COLORES_CLASE[productos.find((p) => p.categoria_yolo === cat)?.clase_id] ?? "var(--texto-tenue)";
  const r = video?.resultado;

  return (
    <>
      <div className="encabezado">
        <div>
          <h1>Evaluación de retiros</h1>
          <p className="subtitulo">Sube videos de prueba, marca cuándo ocurre cada retiro real y compara con lo que registra el sistema.</p>
        </div>
        <label className="boton primario">
          <IconoSubir /> {subida !== null ? `Subiendo… ${Math.round(subida * 100)} %` : "Subir video de prueba"}
          <input type="file" accept="video/*" hidden disabled={subida !== null} onChange={subir} />
        </label>
      </div>

      {resumen && resumen.videos > 0 && (
        <div className="panel">
          <h2>Resultado global ({resumen.videos} {resumen.videos === 1 ? "video" : "videos"})</h2>
          <div className="indicadores" style={{ marginBottom: 0 }}>
            <Indicador rotulo="Retiros detectados" valor={resumen.aciertos} nota={`de ${resumen.aciertos + resumen.omitidos} reales`} />
            <Indicador rotulo="Omitidos" valor={resumen.omitidos} />
            <Indicador rotulo="Falsos / duplicados" valor={`${resumen.falsos_positivos - resumen.duplicados} / ${resumen.duplicados}`} />
            <Indicador rotulo="Precisión · Recall" valor={`${pct(resumen.precision)} · ${pct(resumen.recall)}`} nota={`F1 ${pct(resumen.f1)}`} />
            <Indicador rotulo="Latencia media" valor={resumen.latencia_media_s == null ? "—" : `${resumen.latencia_media_s} s`} nota="del retiro real al registro" />
          </div>
        </div>
      )}

      <div className="panel tabla-scroll">
        <h2>Videos de prueba</h2>
        {videos.length === 0 ? (
          <div className="vacio">
            Graba videos con la cámara fija en los que hagas retiros y reposiciones conocidos (también casos difíciles:
            tomar y devolver, mano que pasa sin tomar nada, dos productos a la vez) y súbelos aquí.
          </div>
        ) : (
          <table>
            <thead><tr><th>Video</th><th className="num">Duración</th><th className="num">Eventos reales</th><th className="num">Precisión</th><th className="num">Recall</th><th className="num">Latencia</th><th></th></tr></thead>
            <tbody>
              {videos.map((v) => (
                <tr key={v.id}>
                  <td><b>{v.nombre}</b><div className="subtitulo" style={{ fontSize: "0.78rem" }}>{v.resolucion} · {v.fps} fps</div></td>
                  <td className="num">{v.duracion_s} s</td>
                  <td className="num">{v.eventos_reales.length}</td>
                  <td className="num">{v.evaluando ? "…" : pct(v.resultado?.precision)}</td>
                  <td className="num">{v.evaluando ? "…" : pct(v.resultado?.recall)}</td>
                  <td className="num">{v.resultado?.latencia_media_s == null ? "—" : `${v.resultado.latencia_media_s} s`}</td>
                  <td>{v.id !== elegido && <button className="pequeno" onClick={() => setElegido(v.id)}>Abrir</button>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {video && (
        <div className="panel">
          <div className="panel-titulo">
            <h2>{video.nombre}</h2>
            <div className="fila">
              <button className="peligro pequeno" onClick={borrar} disabled={evaluando}>Borrar</button>
            </div>
          </div>
          <div className="dos-columnas">
            <div>
              <video ref={reproductor} src={video.url} controls style={{ width: "100%", borderRadius: 8, background: "#000" }} />
              <p className="subtitulo">Reproduce el video y, en el instante en que la mano <b>se lleva</b> (o deja) un producto,
                pausa y pulsa «Marcar aquí». Si el video no se ve, grábalo con el iPhone en formato «Más compatible».</p>
            </div>
            <div>
              <h3>Eventos reales</h3>
              <div className="fila" style={{ alignItems: "flex-end" }}>
                <label className="campo">Producto
                  <select value={nuevo.categoria} onChange={(e) => setNuevo({ ...nuevo, categoria: e.target.value })}>
                    {productos.map((p) => <option key={p.id} value={p.categoria_yolo}>{p.categoria_yolo}</option>)}
                  </select>
                </label>
                <label className="campo">Tipo
                  <select value={nuevo.tipo} onChange={(e) => setNuevo({ ...nuevo, tipo: e.target.value })}>
                    <option value="venta">Retiro (venta)</option>
                    <option value="reposicion">Reposición</option>
                  </select>
                </label>
                <label className="campo">Cant.
                  <input type="number" min="1" max="20" value={nuevo.cantidad} style={{ width: 60 }}
                    onChange={(e) => setNuevo({ ...nuevo, cantidad: Number(e.target.value) })} />
                </label>
                <button className="primario" onClick={marcar}>Marcar aquí</button>
              </div>
              <table style={{ marginTop: 10 }}>
                <thead><tr><th>Segundo</th><th>Producto</th><th>Tipo</th><th className="num">Cant.</th><th></th></tr></thead>
                <tbody>
                  {eventos.length === 0 && <tr><td colSpan="5" className="subtitulo">Aún no hay eventos marcados.</td></tr>}
                  {eventos.map((e, i) => (
                    <tr key={i}>
                      <td><input type="number" step="0.1" min="0" value={e.t} style={{ width: 70 }} onChange={(x) => editar(i, "t", x.target.value)} />
                        <button className="pequeno" title="Ir a ese momento" onClick={() => ir(Number(e.t))}>▶</button></td>
                      <td><span className="punto" style={{ background: color(e.categoria) }} />{e.categoria}</td>
                      <td>{e.tipo === "venta" ? "venta" : "reposición"}</td>
                      <td className="num">{e.cantidad}</td>
                      <td><button className="peligro pequeno" onClick={() => quitar(i)}>Quitar</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="fila" style={{ marginTop: 12, alignItems: "flex-end" }}>
                <label className="campo" title="Cuánto tiempo después del evento real se acepta el registro del sistema">
                  Tolerancia (s)
                  <input type="number" min="2" max="60" value={tolerancia} style={{ width: 70 }}
                    onChange={(e) => { setTolerancia(e.target.value); setCambios(true); }} />
                </label>
                <span className="espaciador" />
                <button disabled={!cambios} onClick={guardar}>Guardar</button>
                <button className="primario" disabled={evaluando} onClick={evaluar}>
                  {evaluando ? `Evaluando… ${Math.round((video.progreso ?? 0) * 100)} %` : r ? "Volver a evaluar" : "Evaluar"}
                </button>
              </div>
              {evaluando && <div className="barra" style={{ marginTop: 8 }}><div style={{ width: `${(video.progreso ?? 0) * 100}%` }} /></div>}
            </div>
          </div>

          {r?.error && <div className="alerta error" style={{ marginTop: 14 }}>{r.error}</div>}
          {r && !r.error && !evaluando && (
            <>
              <h3 style={{ marginTop: 18 }}>Resultado</h3>
              <div className="indicadores">
                <Indicador rotulo="Aciertos" valor={`${r.aciertos} / ${r.aciertos + r.omitidos}`} />
                <Indicador rotulo="Omitidos" valor={r.omitidos} />
                <Indicador rotulo="Falsos · duplicados" valor={`${r.falsos} · ${r.duplicados}`} />
                <Indicador rotulo="Precisión · Recall" valor={`${pct(r.precision)} · ${pct(r.recall)}`} />
                <Indicador rotulo="Latencia media" valor={r.latencia_media_s == null ? "—" : `${r.latencia_media_s} s`} nota={r.latencia_max_s != null ? `máxima ${r.latencia_max_s} s` : null} />
              </div>
              <p className="subtitulo">
                Modelo <code>{r.modelo?.modelo}</code> ({r.modelo?.tipo}) · confirmación {r.parametros?.confirmacion_s} s,
                quietud {r.parametros?.quietud_s} s, {r.parametros?.fps_analisis} análisis/s (se ajustan en Vitrina en vivo)
                · inferencia {r.inferencia_ms} ms · el sistema quedó listo en el segundo {r.inicio_s ?? "—"}.
              </p>
              {r.excluidos?.length > 0 && (
                <div className="alerta aviso"><span>{r.excluidos.length} evento(s) real(es) ocurrieron antes de que el sistema terminara de iniciar
                  (segundo {r.inicio_s}) y no se cuentan. Deja la vitrina quieta unos segundos al empezar a grabar.</span></div>
              )}
              <table>
                <thead><tr><th>Resultado</th><th>Producto</th><th>Tipo</th><th className="num">Real</th><th className="num">Registrado</th><th className="num">Latencia</th><th>Nota</th></tr></thead>
                <tbody>
                  {r.filas.map((f, i) => {
                    const e = f.real ?? f.detectado;
                    const [nombre, clase] = RESULTADOS[f.resultado];
                    return (
                      <tr key={i}>
                        <td><span className={`etiqueta ${clase}`}>{nombre}</span></td>
                        <td><span className="punto" style={{ background: color(e.categoria) }} />{e.categoria}</td>
                        <td>{e.tipo === "venta" ? "venta" : "reposición"}</td>
                        <td className="num">{f.real ? <button className="pequeno" onClick={() => ir(f.real.t)}>{seg(f.real.t)}</button> : "—"}</td>
                        <td className="num">{f.detectado ? <button className="pequeno" onClick={() => ir(f.detectado.t)}>{seg(f.detectado.t)}</button> : "—"}</td>
                        <td className="num">{f.latencia_s != null ? `${f.latencia_s} s` : "—"}</td>
                        <td className="subtitulo">{f.motivo ?? ""}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </>
          )}
        </div>
      )}
    </>
  );
}
