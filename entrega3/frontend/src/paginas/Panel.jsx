import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { COLORES_CLASE, FASES, TIPOS_MOVIMIENTO, api, haceCuanto, hora, pesos } from "../api.js";
import { IconoAlerta, IconoCamara, IconoHistorial, IconoProducto } from "../componentes/Iconos.jsx";

// Página de inicio: lo que hay en la vitrina y cómo van las ventas de hoy.
// Se refresca sola cada 5 s, porque el inventario lo actualiza la cámara.
export default function Panel() {
  const [datos, setDatos] = useState(null);
  const [monitor, setMonitor] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const cargar = () => {
      api.panel().then((d) => { setDatos(d); setError(""); }).catch((e) => setError(e.message));
      api.monitor().then(setMonitor).catch(() => {});
    };
    cargar();
    const t = setInterval(cargar, 5000);
    return () => clearInterval(t);
  }, []);

  if (!datos) return error ? <div className="alerta error">{error}</div> : <p>Cargando…</p>;

  const bajos = datos.productos.filter((p) => p.cantidad <= p.stock_minimo);

  return (
    <>
      <div className="encabezado">
        <div>
          <h1>Panel de la vitrina</h1>
          <p className="subtitulo">
            Inventario actualizado {haceCuanto(datos.ultima_actualizacion)} · {datos.unidades} unidades en vitrina
          </p>
        </div>
        <div className="fila">
          {monitor && (
            <span className={`pildora ${monitor.activo ? monitor.fase : ""}`}>
              {monitor.activo ? `Cámara vigilando · ${FASES[monitor.fase]?.texto.toLowerCase()}` : "Cámara detenida"}
            </span>
          )}
          <Link to="/vitrina" className="boton primario grande"><IconoCamara /> {monitor?.activo ? "Ver vitrina en vivo" : "Iniciar la cámara"}</Link>
        </div>
      </div>

      <div className="indicadores">
        <div className="indicador">
          <div className="rotulo">Ingresos de hoy</div>
          <div className="valor">{pesos.format(datos.ventas_hoy.ingresos)}</div>
          <div className="detalle">Total histórico: {pesos.format(datos.ingresos_total)}</div>
        </div>
        <div className="indicador">
          <div className="rotulo">Unidades vendidas hoy</div>
          <div className="valor">{datos.ventas_hoy.unidades}</div>
          <div className="detalle">Cada retiro detectado cuenta como venta</div>
        </div>
        <div className="indicador">
          <div className="rotulo">Valor del inventario</div>
          <div className="valor">{pesos.format(datos.valor_inventario)}</div>
          <div className="detalle">{datos.unidades} unidades a precio de venta</div>
        </div>
        <div className="indicador">
          <div className="rotulo">{bajos.length > 0 && <IconoAlerta style={{ color: "var(--peligro)" }} />} Stock bajo</div>
          <div className="valor">{bajos.length}</div>
          <div className="detalle">{bajos.length ? bajos.map((p) => p.nombre).join(", ") : "Todos los productos por encima del mínimo"}</div>
        </div>
      </div>

      <div className="dos-columnas">
        <section>
          <div className="panel-titulo"><h2>En la vitrina</h2></div>
          <div className="productos">
            {datos.productos.map((p) => {
              const bajo = p.cantidad <= p.stock_minimo;
              // La barra se escala respecto a 3 veces el mínimo (o la cantidad, si es mayor).
              const tope = Math.max(p.stock_minimo * 3, p.cantidad, 1);
              return (
                <div key={p.id} className={`producto ${bajo ? "bajo" : ""}`}>
                  <div className="cabeza">
                    <div className="icono" style={{ color: COLORES_CLASE[p.clase_id] }}>
                      <IconoProducto categoria={p.categoria} tamano="22" />
                    </div>
                    <div>
                      <div className="nombre">{p.nombre}</div>
                      <div className="cat">{p.categoria} · {pesos.format(p.precio)}</div>
                    </div>
                  </div>
                  <div className="cantidad">{p.cantidad}<small>unidades</small></div>
                  <div className="barra" title={`Mínimo: ${p.stock_minimo}`}>
                    <div style={{ width: `${(100 * p.cantidad) / tope}%`, background: bajo ? "var(--peligro)" : COLORES_CLASE[p.clase_id] }} />
                  </div>
                  <div className="pie">
                    {bajo ? <span className="etiqueta error"><IconoAlerta tamano="0.9em" /> Stock bajo</span> : <span>Mínimo {p.stock_minimo}</span>}
                    <span>{p.vendidas_hoy} {p.vendidas_hoy === 1 ? "vendida" : "vendidas"} hoy</span>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="panel">
          <div className="panel-titulo">
            <h2>Últimos movimientos</h2>
            <Link to="/historial" className="migas">Ver todo</Link>
          </div>
          {datos.ultimos_movimientos.length === 0 ? (
            <div className="vacio">
              <IconoHistorial tamano="32" />
              <p>Aún no hay movimientos. Inicia la cámara en «Vitrina en vivo»: el primer conteo se registra solo.</p>
            </div>
          ) : (
            <div className="lista-movs">
              {datos.ultimos_movimientos.map((m) => {
                const clase = datos.productos.find((p) => p.id === m.producto_id)?.clase_id;
                return (
                  <div key={m.id} className="mov">
                    <div className="icono" style={{ color: COLORES_CLASE[clase] }}><IconoProducto categoria={m.categoria} /></div>
                    <div>
                      <div className="que">{Math.abs(m.cantidad)} × {m.nombre}</div>
                      <div className="cuando">
                        <span className={`etiqueta ${TIPOS_MOVIMIENTO[m.tipo].clase}`}>{TIPOS_MOVIMIENTO[m.tipo].texto}</span>{" "}
                        {hora.format(new Date(m.fecha_hora))} · {m.origen === "vision" ? "cámara" : "manual"}
                      </div>
                    </div>
                    <div className="monto">
                      {m.tipo === "venta" ? pesos.format(m.total) : `${m.cantidad > 0 ? "+" : ""}${m.cantidad} u.`}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </div>
    </>
  );
}
