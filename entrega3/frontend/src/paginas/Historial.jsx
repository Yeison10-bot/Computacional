import { useEffect, useState } from "react";
import { COLORES_CLASE, TIPOS_MOVIMIENTO, api, fechaHora, pesos } from "../api.js";
import { IconoHistorial } from "../componentes/Iconos.jsx";

const FILTROS = [["", "Todos"], ["venta", "Ventas"], ["reposicion", "Reposiciones"], ["ajuste", "Ajustes"]];
const POR_PAGINA = 50;

// Historial de movimientos del inventario, con la foto que originó cada uno.
export default function Historial() {
  const [tipo, setTipo] = useState("");
  const [datos, setDatos] = useState({ total: 0, items: [] });
  const [productos, setProductos] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    api.productos().then(setProductos);
  }, []);

  useEffect(() => {
    api.movimientos({ tipo, limite: POR_PAGINA }).then(setDatos).catch((e) => setError(e.message));
  }, [tipo]);

  async function cargarMas() {
    const mas = await api.movimientos({ tipo, desde: datos.items.length, limite: POR_PAGINA });
    setDatos((d) => ({ total: mas.total, items: [...d.items, ...mas.items] }));
  }

  const clase = (id) => productos.find((p) => p.id === id)?.clase_id;
  const ingresos = datos.items.reduce((s, m) => s + m.total, 0);

  return (
    <>
      <div className="encabezado">
        <div>
          <h1>Historial</h1>
          <p className="subtitulo">Cada cambio del inventario queda registrado con su fecha, origen y la foto que lo produjo.</p>
        </div>
      </div>
      {error && <div className="alerta error">{error}</div>}

      <div className="panel">
        <div className="fila" style={{ marginBottom: 6 }}>
          <div className="pestanas" style={{ margin: 0 }}>
            {FILTROS.map(([v, t]) => (
              <button key={v} className={tipo === v ? "activa" : ""} onClick={() => setTipo(v)}>{t}</button>
            ))}
          </div>
          <span className="espaciador" />
          <span className="subtitulo">
            {datos.total} movimiento(s){tipo === "venta" && datos.items.length > 0 && ` · ${pesos.format(ingresos)} en los mostrados`}
          </span>
        </div>

        {datos.items.length === 0 ? (
          <div className="vacio"><IconoHistorial tamano="32" /><p>No hay movimientos{tipo && " de este tipo"}.</p></div>
        ) : (
          <div className="tabla-scroll">
            <table>
              <thead>
                <tr>
                  <th>Fecha y hora</th><th>Producto</th><th>Tipo</th><th className="num">Cantidad</th>
                  <th className="num">Precio</th><th className="num">Total</th><th>Origen</th><th className="num">Confianza</th><th>Foto</th>
                </tr>
              </thead>
              <tbody>
                {datos.items.map((m) => (
                  <tr key={m.id}>
                    <td style={{ whiteSpace: "nowrap" }}>{fechaHora.format(new Date(m.fecha_hora))}</td>
                    <td><span className="categoria"><span className="punto" style={{ background: COLORES_CLASE[clase(m.producto_id)] }} />{m.nombre}</span></td>
                    <td><span className={`etiqueta ${TIPOS_MOVIMIENTO[m.tipo].clase}`}>{TIPOS_MOVIMIENTO[m.tipo].texto}</span></td>
                    <td className="num">{m.tipo === "ajuste" && m.cantidad > 0 ? "+" : ""}{m.cantidad}</td>
                    <td className="num">{pesos.format(m.precio_unitario)}</td>
                    <td className="num"><b>{m.tipo === "venta" ? pesos.format(m.total) : "—"}</b></td>
                    <td>{m.origen === "vision" ? "Cámara" : "Manual"}</td>
                    <td className="num">{m.confianza != null ? `${Math.round(m.confianza * 100)} %` : "—"}</td>
                    <td>
                      {m.evidencia_url
                        ? <a href={m.evidencia_url} target="_blank" rel="noreferrer"><img className="evidencia-mini" src={m.evidencia_url} alt="Foto del evento" loading="lazy" /></a>
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {datos.items.length < datos.total && (
          <div style={{ textAlign: "center", marginTop: 12 }}>
            <button onClick={cargarMas}>Cargar más ({datos.total - datos.items.length} restantes)</button>
          </div>
        )}
      </div>
    </>
  );
}
