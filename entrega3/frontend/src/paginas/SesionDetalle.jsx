import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, subirArchivo, tamano } from "../api.js";
import { IconoSubir } from "../componentes/Iconos.jsx";

const ESTADOS = { subido: "aviso", procesando: "aviso", procesado: "ok", error: "error" };
const FILTROS = [
  ["", "Todas"],
  ["pendiente", "Pendientes"],
  ["etiquetada", "Etiquetadas"],
  ["descartada", "Descartadas"],
];
const POR_PAGINA = 120;

export default function SesionDetalle() {
  const { id } = useParams();
  const navegar = useNavigate();
  const [sesion, setSesion] = useState(null);
  const [archivos, setArchivos] = useState([]);
  const [imagenes, setImagenes] = useState({ total: 0, items: [] });
  const [filtro, setFiltro] = useState("");
  const [cola, setCola] = useState([]); // subidas en curso: {nombre, progreso, estado, mensaje}
  const [arrastrando, setArrastrando] = useState(false);
  const [error, setError] = useState("");
  const selector = useRef(null);
  const camaraFoto = useRef(null);
  const camaraVideo = useRef(null);

  const recargar = useCallback(async () => {
    try {
      const [s, a, i] = await Promise.all([
        api.sesion(id),
        api.archivos(id),
        api.imagenes({ sesion_id: id, estado: filtro, limite: POR_PAGINA }),
      ]);
      setSesion(s);
      setArchivos(a);
      setImagenes(i);
    } catch (e) {
      setError(e.message);
    }
  }, [id, filtro]);

  useEffect(() => {
    recargar();
  }, [recargar]);

  // Mientras algún video se esté procesando, se consulta su estado cada 2 s.
  const procesando = archivos.some((a) => a.estado === "procesando" || a.estado === "subido");
  useEffect(() => {
    if (!procesando) return;
    const t = setInterval(recargar, 2000);
    return () => clearInterval(t);
  }, [procesando, recargar]);

  async function subir(lista) {
    const nuevos = [...lista].map((f) => ({ clave: crypto.randomUUID(), archivo: f, nombre: f.name, progreso: 0, estado: "en cola" }));
    setCola((c) => [...c, ...nuevos]);
    // Se suben de uno en uno para mostrar el progreso de cada archivo.
    for (const item of nuevos) {
      const actualizar = (cambios) =>
        setCola((c) => c.map((x) => (x.clave === item.clave ? { ...x, ...cambios } : x)));
      actualizar({ estado: "subiendo" });
      try {
        const r = await subirArchivo(id, item.archivo, (p) => actualizar({ progreso: p }));
        const rechazo = r.rechazados[0];
        actualizar(rechazo ? { estado: "rechazado", mensaje: rechazo.motivo } : { estado: "listo", progreso: 1 });
      } catch (e) {
        actualizar({ estado: "error", mensaje: e.message });
      }
      recargar();
    }
  }

  async function cargarMas() {
    const mas = await api.imagenes({ sesion_id: id, estado: filtro, desde: imagenes.items.length, limite: POR_PAGINA });
    setImagenes((i) => ({ total: mas.total, items: [...i.items, ...mas.items] }));
  }

  async function accion(promesa) {
    try {
      await promesa;
      setError("");
      recargar();
    } catch (e) {
      setError(e.message);
    }
  }

  async function borrarSesion() {
    if (!confirm(`¿Borrar la sesión ${sesion.codigo} con todos sus videos, fotos, imágenes y etiquetas?`)) return;
    await api.borrarSesion(id);
    navegar("/sesiones");
  }

  if (!sesion) return error ? <div className="alerta error">{error}</div> : <p>Cargando…</p>;

  return (
    <>
      <div className="encabezado">
        <div>
          <Link to="/sesiones" className="migas">← Sesiones</Link>
          <h1>
            Sesión {sesion.codigo}{" "}
            {sesion.prueba_generalizacion && <span className="etiqueta aviso">prueba de generalización</span>}
          </h1>
          <p className="subtitulo">
            {[sesion.descripcion, sesion.iluminacion, sesion.superficie].filter(Boolean).join(" · ") || "Sin descripción"}
          </p>
        </div>
        <div className="fila">
          <Link className="boton primario" to={`/etiquetar?sesion=${id}`}>
            Etiquetar pendientes ({sesion.pendientes})
          </Link>
          <button className="peligro" onClick={borrarSesion}>Borrar sesión</button>
        </div>
      </div>
      {error && <div className="alerta error">{error}</div>}

      <div className="panel">
        <h2>Subir videos y fotos</h2>
        {/* En el celular: abre directamente la cámara trasera. Cada foto se sube al tomarla. */}
        <div className="botones-camara">
          <button className="primario grande" onClick={() => camaraFoto.current.click()}>📷 Tomar foto</button>
          <button className="primario grande" onClick={() => camaraVideo.current.click()}>🎥 Grabar video</button>
          <button className="grande" onClick={() => selector.current.click()}>🖼️ Elegir de la galería</button>
          <input ref={camaraFoto} type="file" accept="image/*" capture="environment" hidden
            onChange={(e) => { subir(e.target.files); e.target.value = ""; }} />
          <input ref={camaraVideo} type="file" accept="video/*" capture="environment" hidden
            onChange={(e) => { subir(e.target.files); e.target.value = ""; }} />
          {cola.length > 0 && (
            <p className="subtitulo">
              {cola.filter((c) => c.estado === "listo").length} de {cola.length} subidos en esta visita.
              Sigue tomando fotos: cada una se sube sola.
            </p>
          )}
        </div>
        <div
          className={`zona-subida ${arrastrando ? "activa" : ""}`}
          onClick={() => selector.current.click()}
          onDragOver={(e) => { e.preventDefault(); setArrastrando(true); }}
          onDragLeave={() => setArrastrando(false)}
          onDrop={(e) => { e.preventDefault(); setArrastrando(false); subir(e.dataTransfer.files); }}
        >
          <IconoSubir tamano="30" />
          <div><b>Arrastra aquí los archivos o haz clic para elegirlos</b></div>
          <p style={{ marginTop: 6 }}>
            Videos (.mov, .mp4) → se extraen los fotogramas quietos y distintos automáticamente.
            Fotos (.jpg, .png) → se agregan directamente. Las fotos HEIC del iPhone no se admiten:
            usa Ajustes → Cámara → Formatos → «Más compatible».
          </p>
          <input ref={selector} type="file" multiple hidden accept="video/*,image/*,.mov"
            onChange={(e) => { subir(e.target.files); e.target.value = ""; }} />
        </div>
        {cola.length > 0 && (
          <div className="cola">
            {cola.map((c) => (
              <div className="item" key={c.clave}>
                <span title={c.mensaje}>{c.nombre}{c.mensaje && <small className="subtitulo"> — {c.mensaje}</small>}</span>
                <div className="barra"><div style={{ width: `${c.progreso * 100}%` }} /></div>
                <span className={`etiqueta ${{ listo: "ok", rechazado: "error", error: "error" }[c.estado] ?? "aviso"}`}>{c.estado}</span>
              </div>
            ))}
            <div><button className="pequeno" onClick={() => setCola([])}>Limpiar lista</button></div>
          </div>
        )}
      </div>

      {archivos.length > 0 && (
        <div className="panel tabla-scroll">
          <h2>Archivos originales</h2>
          <table>
            <thead>
              <tr><th>Archivo</th><th>Tipo</th><th className="num">Tamaño</th><th>Estado</th><th>Resultado</th><th className="num">Imágenes</th><th></th></tr>
            </thead>
            <tbody>
              {archivos.map((a) => (
                <tr key={a.id}>
                  <td><a href={a.url} target="_blank" rel="noreferrer">{a.nombre_original}</a></td>
                  <td>{a.tipo}</td>
                  <td className="num">{tamano(a.tamano_bytes)}</td>
                  <td><span className={`etiqueta ${ESTADOS[a.estado]}`}>{a.estado}</span></td>
                  <td><small>{a.detalle}</small></td>
                  <td className="num">{a.imagenes}</td>
                  <td className="fila" style={{ flexWrap: "nowrap" }}>
                    {a.tipo === "video" && (
                      <button className="pequeno" disabled={a.estado === "procesando"}
                        title="Vuelve a extraer los fotogramas (borra las etiquetas de este video)"
                        onClick={() => confirm("Se reemplazarán las imágenes de este video y sus etiquetas. ¿Continuar?") && accion(api.reextraer(a.id))}>
                        Re-extraer
                      </button>
                    )}
                    <button className="pequeno peligro"
                      onClick={() => confirm(`¿Borrar ${a.nombre_original} y sus imágenes?`) && accion(api.borrarArchivo(a.id))}>
                      Borrar
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="panel">
        <div className="fila" style={{ marginBottom: 12 }}>
          <h2 style={{ margin: 0 }}>Imágenes del dataset</h2>
          <span className="subtitulo">{imagenes.total} {filtro ? `${filtro}s` : "en total"}</span>
        </div>
        <div className="pestanas">
          {FILTROS.map(([valor, texto]) => (
            <button key={valor} className={filtro === valor ? "activa" : ""} onClick={() => setFiltro(valor)}>{texto}</button>
          ))}
        </div>
        {imagenes.items.length === 0 ? (
          <div className="vacio">No hay imágenes {filtro && `${filtro}s`} en esta sesión.</div>
        ) : (
          <>
            <div className="galeria">
              {imagenes.items.map((img) => (
                <Link key={img.id} to={`/etiquetar/${img.id}`} className="miniatura" title={`Imagen ${img.id}`}>
                  <img src={img.url} loading="lazy" alt="" />
                  <span className={`etiqueta ${{ etiquetada: "ok", descartada: "error" }[img.estado] ?? ""}`}>{img.estado}</span>
                  {img.estado === "etiquetada" && <span className="cajas">{img.cajas} cajas</span>}
                </Link>
              ))}
            </div>
            {imagenes.items.length < imagenes.total && (
              <div style={{ textAlign: "center", marginTop: 12 }}>
                <button onClick={cargarMas}>Cargar más ({imagenes.total - imagenes.items.length} restantes)</button>
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}
