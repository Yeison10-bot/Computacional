import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { COLORES_CLASE, api } from "../api.js";

// Editor de cajas delimitadoras.
// Internamente cada caja se guarda con sus esquinas (x1, y1, x2, y2) normalizadas
// entre 0 y 1, porque así es más fácil moverla y redimensionarla. Al guardar se
// convierte al formato YOLO: centro (x, y), ancho y alto, también normalizados.

const TAMANO_MINIMO = 0.006; // cajas más pequeñas se consideran un clic accidental
const recortar = (v) => Math.min(1, Math.max(0, v));

const aEsquinas = (c) => ({
  clase_id: c.clase_id,
  x1: c.x_centro - c.ancho / 2, y1: c.y_centro - c.alto / 2,
  x2: c.x_centro + c.ancho / 2, y2: c.y_centro + c.alto / 2,
});

const aYolo = ({ clase_id, x1, y1, x2, y2 }) => {
  const [a, b, c, d] = [recortar(x1), recortar(y1), recortar(x2), recortar(y2)];
  return { clase_id, x_centro: (a + c) / 2, y_centro: (b + d) / 2, ancho: c - a, alto: d - b };
};

const ordenar = (b) => ({ ...b, x1: Math.min(b.x1, b.x2), x2: Math.max(b.x1, b.x2), y1: Math.min(b.y1, b.y2), y2: Math.max(b.y1, b.y2) });

export default function Etiquetar() {
  const { id } = useParams();
  const [parametros] = useSearchParams();
  const navegar = useNavigate();
  const [productos, setProductos] = useState([]);
  const [detalle, setDetalle] = useState(null);
  const [cajas, setCajas] = useState([]);
  const [seleccion, setSeleccion] = useState(null);
  const [clase, setClase] = useState(0);
  const [modificado, setModificado] = useState(false);
  const [mensaje, setMensaje] = useState(null); // {tipo, texto}
  const svg = useRef(null);
  const arrastre = useRef(null);

  useEffect(() => {
    api.productos().then(setProductos);
  }, []);

  // Sin id en la ruta: se abre la primera imagen pendiente (de la sesión indicada, si la hay).
  useEffect(() => {
    if (id) return;
    api.siguientePendiente(parametros.get("sesion")).then(({ id: sig }) => {
      if (sig) navegar(`/etiquetar/${sig}`, { replace: true });
      else setMensaje({ tipo: "ok", texto: "No hay imágenes pendientes de etiquetar." });
    });
  }, [id, parametros, navegar]);

  useEffect(() => {
    if (!id) return;
    // Se limpia la imagen anterior de inmediato: así ninguna tecla presionada
    // mientras carga la nueva puede modificar la imagen equivocada.
    setDetalle(null);
    setCajas([]);
    api.imagen(id)
      .then((d) => {
        setDetalle(d);
        setCajas(d.anotaciones.map(aEsquinas));
        setSeleccion(null);
        setModificado(false);
      })
      .catch((e) => setMensaje({ tipo: "error", texto: e.message }));
  }, [id]);

  const nombreClase = (c) => productos.find((p) => p.clase_id === c)?.categoria_yolo ?? `clase ${c}`;

  // ---------------------------------------------------------------- Guardar y navegar

  // sinCajas: guarda la imagen como ejemplo negativo (vitrina vacía) aunque haya cajas dibujadas.
  async function guardar(estado, { avanzar = true, sinCajas = false } = {}) {
    try {
      const anotaciones = estado === "descartada" || sinCajas ? [] : cajas.map(aYolo);
      const d = await api.guardarAnotaciones(detalle.id, anotaciones, estado);
      setModificado(false);
      if (!avanzar) return d;
      const { id: sig } = await api.siguientePendiente(detalle.sesion_id);
      if (sig) {
        setMensaje(null);
        navegar(`/etiquetar/${sig}`);
      } else {
        setDetalle(d);
        setMensaje({ tipo: "ok", texto: `Guardado. No quedan imágenes pendientes en la sesión ${d.sesion_codigo}.` });
      }
    } catch (e) {
      setMensaje({ tipo: "error", texto: e.message });
    }
  }

  async function ir(destino) {
    if (!destino) return;
    // Si hay cambios sin guardar, se guardan antes de cambiar de imagen.
    if (modificado && !(await guardar("etiquetada", { avanzar: false }))) return;
    setMensaje(null);
    navegar(`/etiquetar/${destino}`);
  }

  async function copiarAnterior() {
    if (!detalle?.anterior_id) return;
    const anterior = await api.imagen(detalle.anterior_id);
    setCajas(anterior.anotaciones.map(aEsquinas));
    setSeleccion(null);
    setModificado(true);
  }

  function cambiarCajas(fn) {
    setCajas(fn);
    setModificado(true);
  }

  // ---------------------------------------------------------------- Teclado

  useEffect(() => {
    function alPresionar(e) {
      if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
      if (!detalle || String(detalle.id) !== id) return; // la imagen de la URL aún no ha cargado
      const tecla = e.key.toLowerCase();
      if (/^[1-9]$/.test(tecla) && Number(tecla) <= productos.length) {
        const c = Number(tecla) - 1;
        setClase(c);
        if (seleccion !== null) cambiarCajas((cs) => cs.map((b, i) => (i === seleccion ? { ...b, clase_id: c } : b)));
      } else if ((tecla === "delete" || tecla === "backspace") && seleccion !== null) {
        cambiarCajas((cs) => cs.filter((_, i) => i !== seleccion));
        setSeleccion(null);
      } else if (tecla === "escape") {
        setSeleccion(null);
      } else if (tecla === "enter" || tecla === "s") {
        guardar("etiquetada");
      } else if (tecla === "v") {
        if (cajas.length === 0 || confirm("¿Marcar como imagen sin objetos? Se borrarán las cajas.")) {
          guardar("etiquetada", { sinCajas: true });
        }
      } else if (tecla === "x") {
        guardar("descartada");
      } else if (tecla === "c") {
        copiarAnterior();
      } else if (tecla === "arrowleft") {
        ir(detalle.anterior_id);
      } else if (tecla === "arrowright") {
        ir(detalle.siguiente_id);
      } else {
        return;
      }
      e.preventDefault();
    }
    window.addEventListener("keydown", alPresionar);
    return () => window.removeEventListener("keydown", alPresionar);
  });

  // ---------------------------------------------------------------- Ratón (dibujar, mover, redimensionar)

  function punto(e) {
    const r = svg.current.getBoundingClientRect();
    return { x: recortar((e.clientX - r.left) / r.width), y: recortar((e.clientY - r.top) / r.height) };
  }

  function empezar(e, modo, indice, esquina) {
    if (e.button !== 0) return;
    e.stopPropagation();
    svg.current.setPointerCapture(e.pointerId);
    const p = punto(e);
    if (modo === "dibujar") {
      indice = cajas.length;
      setCajas((cs) => [...cs, { clase_id: clase, x1: p.x, y1: p.y, x2: p.x, y2: p.y }]);
    }
    setSeleccion(indice);
    arrastre.current = { modo, indice, esquina, inicio: p, original: cajas[indice] };
  }

  function mover(e) {
    const a = arrastre.current;
    if (!a) return;
    const p = punto(e);
    setCajas((cs) => cs.map((b, i) => {
      if (i !== a.indice) return b;
      if (a.modo === "dibujar") return { ...b, x1: a.inicio.x, y1: a.inicio.y, x2: p.x, y2: p.y };
      if (a.modo === "mover") {
        const o = a.original;
        // El desplazamiento se limita para que la caja no salga de la imagen.
        const dx = Math.min(1 - o.x2, Math.max(-o.x1, p.x - a.inicio.x));
        const dy = Math.min(1 - o.y2, Math.max(-o.y1, p.y - a.inicio.y));
        return { ...o, x1: o.x1 + dx, x2: o.x2 + dx, y1: o.y1 + dy, y2: o.y2 + dy };
      }
      // Redimensionar: la esquina opuesta queda fija.
      const o = a.original;
      const fija = { x: a.esquina.includes("o") ? o.x2 : o.x1, y: a.esquina.includes("n") ? o.y2 : o.y1 };
      return { ...o, x1: fija.x, y1: fija.y, x2: p.x, y2: p.y };
    }));
  }

  function terminar() {
    const a = arrastre.current;
    if (!a || !cajas[a.indice]) return;
    arrastre.current = null;
    const b = ordenar(cajas[a.indice]);
    if (b.x2 - b.x1 < TAMANO_MINIMO || b.y2 - b.y1 < TAMANO_MINIMO) {
      // Un clic sin arrastrar: se descarta la caja nueva o se restaura la original.
      if (a.modo === "dibujar") {
        setCajas((cs) => cs.filter((_, i) => i !== a.indice));
        setSeleccion(null);
      } else {
        setCajas((cs) => cs.map((x, i) => (i === a.indice ? a.original : x)));
      }
      return;
    }
    setCajas((cs) => cs.map((x, i) => (i === a.indice ? b : x)));
    // Seleccionar una caja sin moverla no cuenta como cambio.
    if (a.modo === "dibujar" || JSON.stringify(b) !== JSON.stringify(a.original)) setModificado(true);
  }

  // ---------------------------------------------------------------- Vista

  if (!id || !detalle) {
    return (
      <>
        <h1>Etiquetar</h1>
        {mensaje ? <div className={`alerta ${mensaje.tipo}`}>{mensaje.texto}</div> : <p>Cargando…</p>}
        <Link to="/sesiones" className="boton">Ir a las sesiones</Link>
      </>
    );
  }

  const { ancho: W, alto: H } = detalle;
  const proporcion = W / H;
  const conteo = cajas.reduce((m, b) => ({ ...m, [b.clase_id]: (m[b.clase_id] ?? 0) + 1 }), {});

  return (
    <>
      <div className="encabezado">
        <div>
          <Link to={`/sesiones/${detalle.sesion_id}`} className="migas">← Sesión {detalle.sesion_codigo}</Link>
          <h1>
            Imagen {detalle.id}{" "}
            <span className={`etiqueta ${{ etiquetada: "ok", descartada: "error" }[detalle.estado] ?? ""}`}>{detalle.estado}</span>{" "}
            {modificado && <span className="etiqueta aviso">sin guardar</span>}
          </h1>
        </div>
        <div className="fila">
          <button disabled={!detalle.anterior_id} onClick={() => ir(detalle.anterior_id)}>← Anterior</button>
          <button disabled={!detalle.siguiente_id} onClick={() => ir(detalle.siguiente_id)}>Siguiente →</button>
        </div>
      </div>
      {mensaje && <div className={`alerta ${mensaje.tipo}`}>{mensaje.texto}</div>}

      <div className="editor">
        {/* El contenedor toma la proporción exacta de la imagen para que las cajas
            (en coordenadas de la imagen) coincidan con lo que se ve. */}
        <div className="lienzo" style={{ aspectRatio: proporcion, width: `min(100%, calc((100vh - 170px) * ${proporcion}))` }}>
          <img src={detalle.url} alt="" draggable={false} />
          <svg ref={svg} viewBox={`0 0 ${W} ${H}`} onPointerDown={(e) => empezar(e, "dibujar")}
            onPointerMove={mover} onPointerUp={terminar} onPointerCancel={terminar}>
            {cajas.map((b0, i) => {
              const b = ordenar(b0);
              const color = COLORES_CLASE[b.clase_id] ?? "#fff";
              const sel = i === seleccion;
              return (
                <g key={i}>
                  <rect x={b.x1 * W} y={b.y1 * H} width={(b.x2 - b.x1) * W} height={(b.y2 - b.y1) * H}
                    fillOpacity={sel ? 0.25 : 0.1} strokeWidth={sel ? 3 : 2}
                    vectorEffect="non-scaling-stroke" style={{ fill: color, stroke: color, cursor: "move" }}
                    onPointerDown={(e) => empezar(e, "mover", i)} />
                  <text className="nombre-caja" x={b.x1 * W + 4} y={b.y1 * H - 6} fontSize={W / 75}>
                    {nombreClase(b.clase_id)}
                  </text>
                  {sel && ["no", "ne", "so", "se"].map((esq) => (
                    <rect key={esq} width={W / 90} height={W / 90} fill="#fff" strokeWidth={2}
                      vectorEffect="non-scaling-stroke"
                      x={(esq.includes("o") ? b.x1 : b.x2) * W - W / 180}
                      y={(esq.includes("n") ? b.y1 : b.y2) * H - W / 180}
                      style={{ stroke: color, cursor: esq === "no" || esq === "se" ? "nwse-resize" : "nesw-resize" }}
                      onPointerDown={(e) => empezar(e, "redimensionar", i, esq)} />
                  ))}
                </g>
              );
            })}
          </svg>
        </div>

        <div>
          <div className="panel">
            <h2>Categoría</h2>
            <div className="clases">
              {productos.map((p) => (
                <button key={p.clase_id} className={clase === p.clase_id ? "activa" : ""}
                  style={{ color: COLORES_CLASE[p.clase_id] }}
                  onClick={() => {
                    setClase(p.clase_id);
                    if (seleccion !== null) cambiarCajas((cs) => cs.map((b, i) => (i === seleccion ? { ...b, clase_id: p.clase_id } : b)));
                  }}>
                  <kbd>{p.clase_id + 1}</kbd>
                  <span style={{ color: "var(--texto)" }}>{p.categoria_yolo}</span>
                  <span className="contador">{conteo[p.clase_id] ?? 0}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="panel" style={{ display: "grid", gap: 8 }}>
            <button className="primario" onClick={() => guardar("etiquetada")}>
              Guardar y siguiente <kbd>Enter</kbd>
            </button>
            <div className="fila">
              <button style={{ flex: 1 }} disabled={!detalle.anterior_id} onClick={copiarAnterior}
                title="Útil en fotogramas seguidos: luego solo ajustas lo que cambió">
                Copiar <kbd>C</kbd>
              </button>
              <button style={{ flex: 1 }} onClick={() => guardar("descartada")} title="Imagen inservible (movida, tapada)">
                Descartar <kbd>X</kbd>
              </button>
            </div>
            <button disabled={cajas.length > 0} onClick={() => guardar("etiquetada", { sinCajas: true })}
              title="Vitrina vacía o solo distractores: sirve como ejemplo negativo">
              Sin objetos (ejemplo negativo) <kbd>V</kbd>
            </button>
          </div>

          <div className="panel atajos">
            <b style={{ color: "var(--texto)" }}>Cómo etiquetar</b>
            <span>Arrastra sobre la imagen para dibujar una caja con la categoría activa.</span>
            <span>Clic en una caja para seleccionarla; arrástrala para moverla o usa sus esquinas.</span>
            <span><kbd>1</kbd>–<kbd>5</kbd> cambia la categoría (también la de la caja seleccionada).</span>
            <span><kbd>Supr</kbd> borra la caja seleccionada · <kbd>Esc</kbd> la deselecciona.</span>
            <span><kbd>←</kbd> <kbd>→</kbd> imagen anterior / siguiente (guarda los cambios).</span>
          </div>
        </div>
      </div>
    </>
  );
}
