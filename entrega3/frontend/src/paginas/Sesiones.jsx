import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api.js";

const ILUMINACIONES = ["Luz de día", "Lámpara / noche", "Mixta", "Otra"];
const FORMULARIO_VACIO = { descripcion: "", iluminacion: ILUMINACIONES[0], superficie: "", prueba_generalizacion: false };

// Una sesión agrupa los videos y fotos tomados con las mismas condiciones.
export default function Sesiones() {
  const [sesiones, setSesiones] = useState([]);
  const [formulario, setFormulario] = useState(FORMULARIO_VACIO);
  const [error, setError] = useState("");
  const navegar = useNavigate();

  useEffect(() => {
    api.sesiones().then(setSesiones).catch((e) => setError(e.message));
  }, []);

  async function crear(e) {
    e.preventDefault();
    try {
      const s = await api.crearSesion(formulario);
      navegar(`/sesiones/${s.id}`);
    } catch (err) {
      setError(err.message);
    }
  }

  const campo = (nombre) => ({
    value: formulario[nombre],
    onChange: (e) => setFormulario({ ...formulario, [nombre]: e.target.value }),
  });

  return (
    <>
      <div className="encabezado">
        <div>
          <h1>Sesiones de captura</h1>
          <p className="subtitulo">
            Cada sesión reúne tomas con la misma luz y superficie. Variar las condiciones entre sesiones hace más robusto al modelo.
          </p>
        </div>
      </div>
      {error && <div className="alerta error">{error}</div>}

      <form className="panel" onSubmit={crear}>
        <h2>Nueva sesión</h2>
        <div className="fila" style={{ alignItems: "flex-end" }}>
          <label className="campo" style={{ flex: 2, minWidth: 220 }}>
            Descripción
            <input placeholder="Ej.: mezclas de 2 a 8 objetos, separados y juntos" {...campo("descripcion")} />
          </label>
          <label className="campo">
            Iluminación
            <select {...campo("iluminacion")}>
              {ILUMINACIONES.map((i) => <option key={i}>{i}</option>)}
            </select>
          </label>
          <label className="campo">
            Superficie
            <input placeholder="Ej.: cartulina blanca" {...campo("superficie")} />
          </label>
          <label className="check" title="Objetos reservados que el modelo nunca ve al entrenar">
            <input type="checkbox" checked={formulario.prueba_generalizacion}
              onChange={(e) => setFormulario({ ...formulario, prueba_generalizacion: e.target.checked })} />
            Prueba de generalización
          </label>
          <button className="primario">Crear sesión</button>
        </div>
      </form>

      {sesiones.length === 0 ? (
        <div className="panel vacio">Todavía no hay sesiones. Crea la primera (S01) para empezar a subir videos y fotos.</div>
      ) : (
        <div className="rejilla">
          {sesiones.map((s) => {
            // Las descartadas no cuentan: el avance es sobre las imágenes útiles.
            const utiles = s.etiquetadas + s.pendientes;
            return (
            <Link key={s.id} to={`/sesiones/${s.id}`} className="panel tarjeta-sesion">
              <div className="fila">
                <span className="codigo">{s.codigo}</span>
                {s.prueba_generalizacion && <span className="etiqueta aviso">generalización</span>}
              </div>
              <p>{s.descripcion || <span className="subtitulo">Sin descripción</span>}</p>
              <small className="subtitulo">{[s.iluminacion, s.superficie].filter(Boolean).join(" · ")}</small>
              <div className="metricas">
                <div><b>{s.videos}</b><small>videos</small></div>
                <div><b>{s.fotos}</b><small>fotos</small></div>
                <div><b>{s.imagenes}</b><small>imágenes</small></div>
              </div>
              <div style={{ marginTop: 10 }}>
                <div className="fila" style={{ justifyContent: "space-between", fontSize: "0.85rem" }}>
                  <span>Etiquetadas</span>
                  <span>{s.etiquetadas} / {utiles}</span>
                </div>
                <div className="barra">
                  <div style={{ width: `${utiles ? (100 * s.etiquetadas) / utiles : 0}%` }} />
                </div>
              </div>
            </Link>
            );
          })}
        </div>
      )}
    </>
  );
}
