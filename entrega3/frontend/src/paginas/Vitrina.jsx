import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { COLORES_CLASE, FASES, TIPOS_MOVIMIENTO, api, hora, pesos } from "../api.js";
import { useAvisos } from "../componentes/Avisos.jsx";
import { useCamaraNavegador } from "../componentes/CamaraNavegador.jsx";
import {
  IconoAlerta, IconoCamara, IconoFlecha, IconoInfo, IconoPlay, IconoStop, IconoSubir, IconoVision,
} from "../componentes/Iconos.jsx";

// Vitrina en vivo: la cámara queda fija y el backend analiza su imagen sin parar
// (backend/app/monitor.py). Aquí se ve el video con las detecciones, el estado de
// cada producto y los eventos que se registran solos. La pestaña "Conteo por foto"
// queda para un conteo puntual sin cámara (p. ej. con una foto tomada aparte).

// ================================================================ En vivo

// Video de la cámara del computador a su velocidad normal, con los recuadros de
// la última detección del servidor dibujados encima (coordenadas normalizadas).
function VideoNavegador({ flujo, detecciones, fase, productos }) {
  const video = useRef(null);
  const [tamano, setTamano] = useState(null);

  useEffect(() => {
    if (video.current) video.current.srcObject = flujo;
  }, [flujo]);

  const clase = (cat) => productos.find((p) => p.categoria_yolo === cat)?.clase_id;
  return (
    <div className={`en-vivo ${fase === "movimiento" ? "con-movimiento" : ""}`}
      style={tamano ? { aspectRatio: `${tamano.w} / ${tamano.h}` } : undefined}>
      <video ref={video} autoPlay muted playsInline
        onLoadedMetadata={(e) => setTamano({ w: e.target.videoWidth, h: e.target.videoHeight })} />
      {tamano && (
        <svg className="capa" viewBox={`0 0 ${tamano.w} ${tamano.h}`}>
          {detecciones.map((d, i) => {
            const x = (d.x_centro - d.ancho / 2) * tamano.w;
            const y = (d.y_centro - d.alto / 2) * tamano.h;
            const color = COLORES_CLASE[clase(d.categoria)] ?? "#fff";
            return (
              <g key={i}>
                <rect x={x} y={y} width={d.ancho * tamano.w} height={d.alto * tamano.h} fill="none"
                  strokeWidth="3" vectorEffect="non-scaling-stroke" style={{ stroke: color }} />
                <text x={x + 4} y={Math.max(y - 8, tamano.w / 45)} fontSize={tamano.w / 45}>
                  {d.categoria} {Math.round(d.confianza * 100)}%
                </text>
              </g>
            );
          })}
        </svg>
      )}
      {fase === "cargando" && <div className="aviso-video">Cargando el modelo YOLO…</div>}
      <span className="marca-vivo"><span className="estado-conexion vivo" /> EN VIVO · {detecciones.length} detectado(s)</span>
    </div>
  );
}

// Velocidad del análisis y tiempos de confirmación, ajustables en caliente.
function AjustesMonitor({ parametros, alGuardar }) {
  const avisar = useAvisos();
  const [valores, setValores] = useState(parametros);
  useEffect(() => setValores(parametros), [parametros]);
  if (!valores) return null;

  const cambiado = ["fps_analisis", "confirmacion_s", "quietud_s"].some((k) => valores[k] !== parametros[k]);
  async function guardar() {
    try {
      alGuardar(await api.ajustarMonitor({
        fps_analisis: valores.fps_analisis, confirmacion_s: valores.confirmacion_s, quietud_s: valores.quietud_s,
      }));
      avisar("Ajustes guardados; se aplican al instante.");
    } catch (e) {
      avisar(e.message, "error");
    }
  }
  const control = (clave, texto, min, max, paso, unidad, ayuda) => (
    <label className="campo" key={clave}>
      <span className="fila" style={{ justifyContent: "space-between" }}>
        <span>{texto}</span><b style={{ color: "var(--texto)" }}>{valores[clave]} {unidad}</b>
      </span>
      <input type="range" min={min} max={max} step={paso} value={valores[clave]}
        onChange={(e) => setValores({ ...valores, [clave]: Number(e.target.value) })} />
      <small>{ayuda}</small>
    </label>
  );

  return (
    <section className="panel">
      <h2>Ajustes del monitoreo</h2>
      <div style={{ display: "grid", gap: 14 }}>
        {control("confirmacion_s", "Tiempo para confirmar un retiro", 1, 15, 0.5, "s",
          "Cuánto debe mantenerse el cambio, con la escena quieta, antes de registrar la venta.")}
        {control("quietud_s", "Quietud requerida", 0.5, 5, 0.5, "s",
          "Tiempo sin movimiento (sin manos) para empezar a contar.")}
        {control("fps_analisis", "Análisis por segundo", 1, 10, 1, "fps",
          "Más análisis no registra ventas más rápido; solo detecta antes el movimiento. Menos alivia la GPU.")}
      </div>
      <button className="primario" style={{ marginTop: 14, width: "100%" }} disabled={!cambiado} onClick={guardar}>Guardar ajustes</button>
    </section>
  );
}

function EnVivo({ productos }) {
  const avisar = useAvisos();
  const camara = useCamaraNavegador();
  const [estado, setEstado] = useState(null);
  const [tipoFuente, setTipoFuente] = useState("navegador"); // "navegador" | "servidor"
  const [camaraNavegador, setCamaraNavegador] = useState("");
  const [camaras, setCamaras] = useState(null);
  const [fuente, setFuente] = useState("0");
  const [archivo, setArchivo] = useState("");
  const [ocupado, setOcupado] = useState(false);

  useEffect(() => {
    const leer = () => api.monitor().then(setEstado).catch(() => {});
    leer();
    const t = setInterval(leer, 700);
    return () => clearInterval(t);
  }, []);

  async function buscarCamaras() {
    setOcupado(true);
    try {
      const lista = await api.camaras();
      setCamaras(lista);
      if (lista.length && !lista.some((c) => String(c.indice) === fuente)) setFuente(String(lista[0].indice));
      if (!lista.length) avisar("No se encontró ninguna cámara conectada.", "error");
    } catch (e) {
      avisar(e.message, "error");
    } finally {
      setOcupado(false);
    }
  }

  async function iniciar() {
    setOcupado(true);
    try {
      if (tipoFuente === "navegador") {
        // Primero la cámara (pide permiso); solo si abre se inicia el análisis.
        if (!(await camara.abrir(camaraNavegador))) return;
        try {
          setEstado(await api.iniciarMonitor("navegador"));
        } catch (e) {
          camara.cerrar();
          throw e;
        }
        camara.empezarEnvio();
      } else {
        setEstado(await api.iniciarMonitor(fuente === "archivo" ? archivo : fuente));
      }
      avisar("Monitoreo iniciado. Deja la escena quieta unos segundos para el conteo inicial.");
    } catch (e) {
      avisar(e.message, "error");
    } finally {
      setOcupado(false);
    }
  }

  // Tras recargar la página el servidor sigue esperando fotogramas del navegador.
  async function reconectar() {
    if (await camara.abrir(camaraNavegador)) camara.empezarEnvio();
  }

  async function detener() {
    setOcupado(true);
    try {
      setEstado(await api.detenerMonitor());
    } finally {
      camara.cerrar();
      setOcupado(false);
    }
  }

  if (!estado) return <p>Cargando…</p>;
  const activo = estado.activo;
  const fase = FASES[estado.fase] ?? FASES.detenido;
  const nombre = (cat) => productos.find((p) => p.categoria_yolo === cat);

  return (
    <div className="dos-columnas">
      <section className="panel">
        <div className="panel-titulo">
          <h2>Cámara</h2>
          <span className={`pildora ${estado.fase}`} title={fase.descripcion}>{fase.texto}</span>
        </div>

        {activo && estado.fuente === "navegador" && camara.flujo ? (
          <VideoNavegador flujo={camara.flujo} detecciones={camara.analisis.detecciones}
            fase={estado.fase} productos={productos} />
        ) : (
          <div className={`en-vivo ${activo && estado.fase === "movimiento" ? "con-movimiento" : ""}`}>
            {activo ? (
              <>
                <div className="apagado">{estado.fase === "cargando" ? "Cargando el modelo YOLO…" : "Conectando con la cámara…"}</div>
                {/* Video MJPEG del backend con los recuadros; la clave fuerza a reconectar en cada inicio */}
                {estado.fuente !== "navegador" && (
                  <img className="capa" key={estado.iniciado_en} src={`/api/monitor/video?inicio=${encodeURIComponent(estado.iniciado_en)}`} alt="Video en vivo de la vitrina" />
                )}
                <span className="marca-vivo"><span className="estado-conexion vivo" /> EN VIVO</span>
              </>
            ) : (
              <div className="apagado">
                <IconoVision tamano="44" />
                <p>La cámara no se está analizando.</p>
                <p style={{ fontSize: "0.85rem" }}>Ubica la cámara fija mirando la vitrina desde arriba e inicia el monitoreo.</p>
              </div>
            )}
          </div>
        )}

        {activo ? (
          <>
            {estado.fuente === "navegador" && !camara.abierta && (
              <div className="alerta aviso" style={{ marginTop: 12 }}>
                <IconoAlerta />
                <span style={{ flex: 1 }}>El servidor espera imágenes de la cámara de este computador, pero no se están enviando (¿recargaste la página?).</span>
                <button className="pequeno" onClick={reconectar}>Reconectar cámara</button>
              </div>
            )}
            <div className="metricas-vivo">
              <span>Fuente <b>{estado.fuente === "navegador" ? "cámara de este computador"
                : /^\d+$/.test(estado.fuente) ? `cámara ${estado.fuente} del servidor` : "video"}</b></span>
              {estado.fuente === "navegador" && camara.abierta && <span>Enviados <b>{camara.enviados}</b></span>}
              <span>Analizados <b>{estado.fotogramas}</b></span>
              {estado.resolucion && <span>Resolución <b>{estado.resolucion}</b></span>}
              <span>Análisis <b>{estado.fps} fps</b></span>
              <span>Inferencia <b>{estado.inferencia_ms} ms</b></span>
              <span>Movimiento <b>{(estado.fraccion_movimiento * 100).toFixed(1)} %</b> de la imagen</span>
              {estado.modelo && <span>Modelo <b>{estado.modelo.modelo}</b></span>}
            </div>
            <button className="peligro" style={{ marginTop: 14 }} disabled={ocupado} onClick={detener}>
              <IconoStop /> Detener monitoreo
            </button>
          </>
        ) : (
          <div className="fuente-video">
            {estado.error && <div className="alerta error"><IconoAlerta /><span>{estado.error}</span></div>}
            {camara.error && <div className="alerta error"><IconoAlerta /><span>{camara.error}</span></div>}

            <div className="opciones-modo" style={{ margin: 0 }}>
              <label className="opcion-modo">
                <input type="radio" name="fuente" checked={tipoFuente === "navegador"} onChange={() => setTipoFuente("navegador")} />
                <span>
                  Cámara de este computador
                  <small>Se abre desde el navegador (la webcam o el iPhone con Iriun). Mantén esta pestaña abierta: puedes navegar por las demás secciones.</small>
                </span>
              </label>
              <label className="opcion-modo">
                <input type="radio" name="fuente" checked={tipoFuente === "servidor"} onChange={() => setTipoFuente("servidor")} />
                <span>
                  Cámara fija conectada al servidor
                  <small>El backend la abre con OpenCV y la vigila aunque cierres el navegador. Ideal para dejarla instalada.</small>
                </span>
              </label>
            </div>

            {tipoFuente === "navegador" ? (
              camara.dispositivos.length > 1 && (
                <label className="campo">
                  Cámara
                  <select value={camaraNavegador} onChange={(e) => setCamaraNavegador(e.target.value)}>
                    <option value="">Predeterminada del navegador</option>
                    {camara.dispositivos.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `Cámara ${i + 1}`}</option>)}
                  </select>
                </label>
              )
            ) : (
              <>
                <div className="fila" style={{ alignItems: "flex-end" }}>
                  <label className="campo" style={{ flex: 1, minWidth: 200 }}>
                    Fuente de video
                    <select value={fuente} onChange={(e) => setFuente(e.target.value)}>
                      {(camaras ?? [{ indice: 0 }]).map((c) => (
                        <option key={c.indice} value={String(c.indice)}>
                          Cámara {c.indice}{c.resolucion ? ` (${c.resolucion})` : ""}{c.indice === 0 ? " · normalmente la del portátil" : ""}
                        </option>
                      ))}
                      <option value="archivo">Archivo de video (pruebas)</option>
                    </select>
                  </label>
                  <button disabled={ocupado} onClick={buscarCamaras}>Buscar cámaras</button>
                </div>
                {fuente === "archivo" && (
                  <label className="campo">
                    Ruta del video en este computador
                    <input placeholder="D:\Tercer Corte\Vision_Computacional\datos\videos_prueba\prueba1.mp4"
                      value={archivo} onChange={(e) => setArchivo(e.target.value)} />
                  </label>
                )}
              </>
            )}

            <button className="primario grande" disabled={ocupado || (tipoFuente === "servidor" && fuente === "archivo" && !archivo)} onClick={iniciar}>
              <IconoPlay /> Iniciar monitoreo
            </button>
            {tipoFuente === "servidor" && (
              <p className="subtitulo" style={{ fontSize: "0.82rem" }}>
                Con el iPhone y Iriun Webcam, la cámara del iPhone aparece como «Cámara 1» al buscar.
                Para que arranque solo con el servidor, pon <code>MONITOR_AUTOINICIO=true</code> en el <code>.env</code>.
              </p>
            )}
          </div>
        )}
      </section>

      <div>
        <section className="panel">
          <h2>Estado por producto</h2>
          <table className="comparacion">
            <thead>
              <tr><th>Producto</th><th className="num">Inventario</th><th className="num">Viendo</th><th>Estado</th></tr>
            </thead>
            <tbody>
              {productos.map((p) => {
                const detectable = estado.categorias.includes(p.categoria_yolo);
                const pend = estado.pendientes?.[p.categoria_yolo];
                const viendo = estado.moda?.[p.categoria_yolo];
                return (
                  <tr key={p.id} style={{ opacity: !activo || detectable ? 1 : 0.5 }}>
                    <td><span className="categoria"><span className="punto" style={{ background: COLORES_CLASE[p.clase_id] }} />{p.nombre}</span></td>
                    <td className="num"><b>{estado.estable?.[p.categoria_yolo] ?? p.cantidad}</b></td>
                    <td className="num">{activo && detectable ? viendo ?? "—" : "—"}</td>
                    <td>
                      {!activo ? <small className="subtitulo">—</small>
                        : !detectable ? <small className="subtitulo">no detectable</small>
                        : pend ? (
                          <span className={`diferencia ${pend.valor < (estado.estable?.[p.categoria_yolo] ?? 0) ? "baja" : "sube"}`}>
                            <IconoFlecha direccion={pend.valor < (estado.estable?.[p.categoria_yolo] ?? 0) ? "abajo" : "arriba"} tamano="0.9em" />
                            <span className="contador-confirmacion">
                              {pend.dudosa ? `detección dudosa (${Math.round(pend.certeza * 100)} %)`
                                : pend.restante_s == null ? "esperando quietud" : `confirmando ${pend.restante_s.toFixed(1)} s`}
                            </span>
                          </span>
                        ) : <span className="diferencia igual">estable</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {activo && estado.categorias.length < productos.length && (
            <p className="subtitulo" style={{ fontSize: "0.82rem", marginTop: 10 }}>
              El modelo provisional no reconoce {productos.filter((p) => !estado.categorias.includes(p.categoria_yolo)).map((p) => p.categoria_yolo).join(" ni ")}; esas existencias no se modifican.
            </p>
          )}
        </section>

        <section className="panel">
          <div className="panel-titulo">
            <h2>Eventos registrados</h2>
            <Link to="/historial" className="migas">Historial</Link>
          </div>
          {estado.eventos.length === 0 ? (
            <p className="subtitulo">Cuando alguien retire o reponga un producto, aparecerá aquí al instante.</p>
          ) : (
            <div className="lista-movs">
              {estado.eventos.map((ev, i) => ev.tipo === "info" ? (
                <div key={i} className="mov" style={{ gridTemplateColumns: "34px 1fr" }}>
                  <div className="icono"><IconoInfo /></div>
                  <div><div className="que">{ev.texto}</div><div className="cuando">{hora.format(new Date(ev.fecha_hora))}</div></div>
                </div>
              ) : (
                <div key={i} className="mov">
                  <div className="icono" style={{ color: COLORES_CLASE[nombre(ev.categoria)?.clase_id] }}><IconoCamara /></div>
                  <div>
                    <div className="que">{Math.abs(ev.cantidad)} × {ev.nombre}</div>
                    <div className="cuando">
                      <span className={`etiqueta ${TIPOS_MOVIMIENTO[ev.tipo].clase}`}>{TIPOS_MOVIMIENTO[ev.tipo].texto}</span>{" "}
                      {hora.format(new Date(ev.fecha_hora))}
                    </div>
                  </div>
                  <div className="monto">{ev.tipo === "venta" ? pesos.format(ev.total) : `${ev.cantidad > 0 ? "+" : ""}${ev.cantidad} u.`}</div>
                </div>
              ))}
            </div>
          )}
        </section>

        <AjustesMonitor parametros={estado.parametros} alGuardar={(p) => setEstado((e) => ({ ...e, parametros: p }))} />

        <section className="panel">
          <h2>Cómo decide el sistema</h2>
          <div className="fases">
            {["estable", "movimiento", "confirmando"].map((f) => (
              <div key={f}><span className={`pildora ${f}`}>{FASES[f].texto}</span><span>{FASES[f].descripcion}</span></div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

// ================================================================ Conteo por foto

function Detecciones({ resultado }) {
  const { ancho: W, alto: H } = resultado;
  return (
    <div className="detecciones">
      <img src={resultado.url} alt="Foto analizada de la vitrina" />
      <svg viewBox={`0 0 ${W} ${H}`}>
        {resultado.detecciones.map((d, i) => {
          const x = (d.x_centro - d.ancho / 2) * W;
          const y = (d.y_centro - d.alto / 2) * H;
          return (
            <g key={i} style={{ stroke: COLORES_CLASE[d.clase_id] }}>
              <rect x={x} y={y} width={d.ancho * W} height={d.alto * H} fill="none" strokeWidth="3" vectorEffect="non-scaling-stroke" />
              <text x={x + 4} y={Math.max(y - 6, W / 60)} fontSize={W / 60} style={{ stroke: "rgb(0 0 0 / 75%)" }}>
                {d.categoria} {Math.round(d.confianza * 100)}%
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function ConteoPorFoto({ monitorActivo }) {
  const avisar = useAvisos();
  const selector = useRef(null);
  const [arrastrando, setArrastrando] = useState(false);
  const [resultado, setResultado] = useState(null);
  const [modo, setModo] = useState("ajuste");
  const [analizando, setAnalizando] = useState(false);
  const [aplicando, setAplicando] = useState(false);
  const [aplicados, setAplicados] = useState(null);

  async function analizar(archivo) {
    setAnalizando(true);
    setAplicados(null);
    try {
      setResultado(await api.contarFoto(archivo));
    } catch (e) {
      avisar(e.message, "error");
    } finally {
      setAnalizando(false);
    }
  }

  async function aplicar() {
    setAplicando(true);
    try {
      const movs = await api.sincronizar(resultado.evidencia, modo);
      setAplicados(movs);
      avisar(movs.length ? `Inventario actualizado: ${movs.length} cambio(s).` : "Inventario verificado: coincide con la foto.");
    } catch (e) {
      avisar(e.message, "error");
    } finally {
      setAplicando(false);
    }
  }

  const filas = resultado?.filas ?? [];
  const cambios = filas.filter((f) => f.diferencia);
  const noDetectables = filas.filter((f) => !f.detectable);

  return (
    <div className="dos-columnas">
      <section className="panel">
        {resultado && !analizando ? (
          <>
            <div className="panel-titulo">
              <h2>{resultado.detecciones.length} producto(s) detectados</h2>
              <button onClick={() => { setResultado(null); setAplicados(null); }}>Otra foto</button>
            </div>
            <Detecciones resultado={resultado} />
            <p className="subtitulo" style={{ marginTop: 10, fontSize: "0.85rem" }}>
              Modelo <code>{resultado.modelo.modelo}</code> ({resultado.modelo.tipo}) en {resultado.modelo.dispositivo} · inferencia de {resultado.tiempo_ms} ms
            </p>
          </>
        ) : (
          <div className={`zona-subida ${arrastrando ? "activa" : ""}`} style={{ padding: 48 }}
            onClick={() => !analizando && selector.current.click()}
            onDragOver={(e) => { e.preventDefault(); setArrastrando(true); }}
            onDragLeave={() => setArrastrando(false)}
            onDrop={(e) => { e.preventDefault(); setArrastrando(false); e.dataTransfer.files[0] && analizar(e.dataTransfer.files[0]); }}>
            <IconoSubir tamano="32" />
            <p><b>{analizando ? "Analizando…" : "Arrastra una foto de la vitrina o haz clic para elegirla"}</b></p>
            <p>Útil para un conteo puntual o para probar el modelo sin dejar la cámara fija.</p>
            <input ref={selector} type="file" accept="image/*" hidden
              onChange={(e) => { e.target.files[0] && analizar(e.target.files[0]); e.target.value = ""; }} />
          </div>
        )}
      </section>

      <section className="panel">
        <h2>Conteo frente al inventario</h2>
        {monitorActivo && (
          <div className="alerta aviso"><IconoAlerta /><span>El monitoreo en vivo está activo: lo normal es dejar que él actualice el inventario.</span></div>
        )}
        {!resultado ? (
          <p className="subtitulo">Sube una foto para ver qué cambió frente al inventario.</p>
        ) : (
          <>
            {noDetectables.length > 0 && (
              <div className="alerta aviso"><IconoAlerta /><span>El modelo provisional no reconoce {noDetectables.map((f) => f.categoria).join(" ni ")}; esas existencias no se modifican.</span></div>
            )}
            <table className="comparacion">
              <thead><tr><th>Producto</th><th className="num">Registrado</th><th className="num">En la foto</th><th className="num">Cambio</th></tr></thead>
              <tbody>
                {filas.map((f) => (
                  <tr key={f.producto_id} style={{ opacity: f.detectable ? 1 : 0.5 }}>
                    <td><span className="categoria"><span className="punto" style={{ background: COLORES_CLASE[f.clase_id] }} />{f.nombre}</span></td>
                    <td className="num">{f.en_sistema}</td>
                    <td className="num">{f.detectable ? <b>{f.detectado}</b> : <small>no detectable</small>}</td>
                    <td className="num">
                      {f.diferencia == null ? "—" : f.diferencia === 0 ? <span className="diferencia igual">sin cambio</span> : (
                        <span className={`diferencia ${f.diferencia < 0 ? "baja" : "sube"}`}>
                          <IconoFlecha direccion={f.diferencia < 0 ? "abajo" : "arriba"} tamano="0.9em" />{f.diferencia > 0 ? "+" : ""}{f.diferencia}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {aplicados ? (
              <div className="alerta ok" style={{ marginTop: 14 }}>Inventario actualizado ({aplicados.length} cambio(s)).</div>
            ) : (
              <>
                <div className="opciones-modo">
                  <label className="opcion-modo">
                    <input type="radio" name="modo" checked={modo === "ajuste"} onChange={() => setModo("ajuste")} />
                    <span>Conteo inicial o corrección<small>Iguala el inventario sin registrar ventas.</small></span>
                  </label>
                  <label className="opcion-modo">
                    <input type="radio" name="modo" checked={modo === "movimientos"} onChange={() => setModo("movimientos")} />
                    <span>Registrar ventas y reposiciones<small>Lo que falta se vende; lo que sobra es reposición.</small></span>
                  </label>
                </div>
                <button className="primario grande" style={{ width: "100%" }} disabled={aplicando} onClick={aplicar}>
                  {aplicando ? "Guardando…" : cambios.length ? `Actualizar inventario (${cambios.length} cambio${cambios.length > 1 ? "s" : ""})` : "Confirmar conteo (sin cambios)"}
                </button>
              </>
            )}
          </>
        )}
      </section>
    </div>
  );
}

// ================================================================ Página

export default function Vitrina() {
  const [pestana, setPestana] = useState("vivo");
  const [productos, setProductos] = useState([]);
  const [monitorActivo, setMonitorActivo] = useState(false);

  useEffect(() => {
    const leer = () => {
      api.productos().then(setProductos).catch(() => {});
      api.monitor().then((e) => setMonitorActivo(e.activo)).catch(() => {});
    };
    leer();
    const t = setInterval(leer, 3000); // el inventario cambia solo mientras la cámara vigila
    return () => clearInterval(t);
  }, []);

  return (
    <>
      <div className="encabezado">
        <div>
          <h1>Vitrina en vivo</h1>
          <p className="subtitulo">La cámara fija vigila la vitrina y el inventario se actualiza solo cuando se retira o se repone un producto.</p>
        </div>
      </div>
      <div className="pestanas">
        <button className={pestana === "vivo" ? "activa" : ""} onClick={() => setPestana("vivo")}><IconoVision /> En vivo</button>
        <button className={pestana === "foto" ? "activa" : ""} onClick={() => setPestana("foto")}><IconoSubir /> Conteo por foto</button>
      </div>
      {pestana === "vivo" ? <EnVivo productos={productos} /> : <ConteoPorFoto monitorActivo={monitorActivo} />}
    </>
  );
}
