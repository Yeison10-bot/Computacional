import { useEffect, useRef, useState } from "react";
import { COLORES_CLASE, api, pesos } from "../api.js";
import { useAvisos } from "../componentes/Avisos.jsx";

const fechaCorta = new Intl.DateTimeFormat("es-CO", { day: "numeric", month: "short" });
const diaDe = (iso) => fechaCorta.format(new Date(`${iso}T12:00:00`));

// Barras verticales en SVG con tooltip al pasar el mouse. Un solo color (una sola serie).
// Mide el ancho real del contenedor: así el texto y el alto no crecen con la pantalla.
function useAncho() {
  const ref = useRef(null);
  const [ancho, setAncho] = useState(600);
  useEffect(() => {
    const o = new ResizeObserver(([e]) => setAncho(Math.max(240, Math.round(e.contentRect.width))));
    o.observe(ref.current);
    return () => o.disconnect();
  }, []);
  return [ref, ancho];
}

function Barras({ datos, etiqueta, valor, detalle, color = "var(--acento)", cadaEtiqueta = 1, titulo }) {
  const [activo, setActivo] = useState(null);
  const [ref, A] = useAncho();
  const H = 220, M = { i: 34, d: 6, s: 10, b: 22 };
  const max = Math.max(1, ...datos.map(valor));
  const paso = (A - M.i - M.d) / datos.length;
  const ancho = Math.max(2, paso - 2); // 2 px de separación entre barras
  const y = (v) => M.s + (1 - v / max) * (H - M.s - M.b);
  // Etiquetas del eje x: como mínimo ~64 px entre una y otra para que no se encimen.
  const salto = Math.max(cadaEtiqueta, Math.ceil(datos.length / Math.max(1, Math.floor((A - M.i) / 64))));
  const marcas = [0, Math.ceil(max / 2), max].filter((v, i, a) => a.indexOf(v) === i);
  return (
    <div ref={ref} style={{ position: "relative" }} onMouseLeave={() => setActivo(null)}>
      <svg width={A} height={H} viewBox={`0 0 ${A} ${H}`} style={{ display: "block" }} role="img" aria-label={titulo}>
        {marcas.map((v) => (
          <g key={v}>
            <line x1={M.i} x2={A - M.d} y1={y(v)} y2={y(v)} stroke="var(--borde)" />
            <text x={M.i - 6} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--texto-suave)">{v}</text>
          </g>
        ))}
        {datos.map((d, i) => {
          const v = valor(d);
          const x = M.i + i * paso + 1;
          const alto = Math.max(v > 0 ? 2 : 0, y(0) - y(v));
          return (
            <g key={i} onMouseEnter={() => setActivo(i)}>
              {/* zona activa más grande que la barra */}
              <rect x={M.i + i * paso} y={M.s} width={paso} height={H - M.s - M.b} fill="transparent" />
              {alto > 0 && (
                <g fill={color} opacity={activo === null || activo === i ? 1 : 0.45}>
                  {/* extremo superior redondeado (4 px) y base plana sobre el eje */}
                  <rect x={x} y={y(0) - alto} width={ancho} height={alto} rx={Math.min(4, ancho / 2, alto / 2)} />
                  <rect x={x} y={y(0) - Math.min(alto, 4)} width={ancho} height={Math.min(alto, 4)} />
                </g>
              )}
              {i % salto === 0 && (
                <text x={x + ancho / 2} y={H - 6} textAnchor="middle" fontSize="11" fill="var(--texto-suave)">{etiqueta(d)}</text>
              )}
            </g>
          );
        })}
        <line x1={M.i} x2={A - M.d} y1={y(0)} y2={y(0)} stroke="var(--texto-tenue)" />
      </svg>
      {activo !== null && (
        <div className="tooltip-grafica" style={{ left: Math.min(A - 90, Math.max(90, M.i + (activo + 0.5) * paso)) }}>
          <b>{etiqueta(datos[activo], true)}</b>
          <div>{detalle(datos[activo])}</div>
        </div>
      )}
    </div>
  );
}

// Barras horizontales por producto: el color identifica la categoría y cada barra lleva su nombre y valor.
function BarrasProducto({ filas, valor, formato }) {
  const max = Math.max(1, ...filas.map(valor));
  return (
    <div style={{ display: "grid", gap: 10 }}>
      {filas.map((f) => (
        <div key={f.categoria} title={`${f.nombre}: ${f.unidades} unidades · ${pesos.format(f.ingresos)}`}>
          <div className="fila" style={{ justifyContent: "space-between", fontSize: "0.88rem" }}>
            <span><span className="punto" style={{ background: COLORES_CLASE[f.clase_id] }} />{f.nombre}</span>
            <b className="num">{formato(valor(f))}</b>
          </div>
          <div className="barra" style={{ height: 12 }}>
            <div style={{ width: `${(100 * valor(f)) / max}%`, background: COLORES_CLASE[f.clase_id] }} />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function Estadisticas() {
  const avisar = useAvisos();
  const [dias, setDias] = useState(7);
  const [datos, setDatos] = useState(null);
  const [medida, setMedida] = useState("unidades");

  useEffect(() => {
    api.estadisticas(dias).then(setDatos).catch((e) => avisar(e.message, "error"));
  }, [dias, avisar]);

  if (!datos) return <p>Cargando…</p>;
  const v = (d) => d[medida];
  const fmt = (n) => (medida === "ingresos" ? pesos.format(n) : `${n} u.`);
  const detalle = (d) => `${d.unidades} unidades · ${pesos.format(d.ingresos)}`;
  const horaPico = datos.por_hora.reduce((a, b) => (b.unidades > a.unidades ? b : a));

  return (
    <>
      <div className="encabezado">
        <div>
          <h1>Ventas y estadísticas</h1>
          <p className="subtitulo">Cada retiro confirmado por la cámara cuenta como una venta, al precio vigente en ese momento.</p>
        </div>
        <div className="fila">
          <div className="pestanas">
            {[[1, "Hoy"], [7, "7 días"], [30, "30 días"], [90, "90 días"]].map(([n, t]) => (
              <button key={n} className={dias === n ? "activa" : ""} onClick={() => setDias(n)}>{t}</button>
            ))}
          </div>
          <div className="pestanas">
            {[["unidades", "Unidades"], ["ingresos", "Ingresos"]].map(([k, t]) => (
              <button key={k} className={medida === k ? "activa" : ""} onClick={() => setMedida(k)}>{t}</button>
            ))}
          </div>
        </div>
      </div>

      <div className="indicadores">
        <div className="indicador"><div className="rotulo">Unidades vendidas</div><div className="valor">{datos.unidades}</div></div>
        <div className="indicador"><div className="rotulo">Ingresos</div><div className="valor">{pesos.format(datos.ingresos)}</div></div>
        <div className="indicador"><div className="rotulo">Producto más vendido</div>
          <div className="valor" style={{ fontSize: "1.2rem" }}>{datos.mas_vendido ? datos.mas_vendido.nombre : "—"}</div>
          {datos.mas_vendido && <div className="subtitulo">{datos.mas_vendido.unidades} unidades</div>}</div>
        <div className="indicador"><div className="rotulo">Hora con más ventas</div>
          <div className="valor">{horaPico.unidades ? `${horaPico.hora}:00` : "—"}</div></div>
      </div>

      {datos.unidades === 0 && (
        <div className="alerta info"><span>Todavía no hay ventas en este periodo. Aparecerán cuando la cámara confirme retiros en <b>Vitrina en vivo</b>.</span></div>
      )}

      <div className="dos-columnas">
        <div className="panel">
          <h2>{medida === "ingresos" ? "Ingresos" : "Ventas"} por {dias === 1 ? "hora" : "día"}</h2>
          {dias === 1 ? (
            <Barras titulo="Ventas por hora" datos={datos.por_hora} valor={v} detalle={detalle} cadaEtiqueta={3}
              etiqueta={(d, largo) => (largo ? `${d.hora}:00 – ${d.hora}:59` : `${d.hora}h`)} />
          ) : (
            <Barras titulo="Ventas por día" datos={datos.por_dia} valor={v} detalle={detalle}
              cadaEtiqueta={Math.ceil(datos.por_dia.length / 8)} etiqueta={(d) => diaDe(d.fecha)} />
          )}
        </div>
        <div className="panel">
          <h2>Por producto</h2>
          <BarrasProducto filas={datos.por_producto} valor={v} formato={fmt} />
        </div>
      </div>

      {dias > 1 && (
        <div className="panel">
          <h2>Ventas por hora del día</h2>
          <p className="subtitulo">Suma de los últimos {dias} días: muestra en qué horas se vende más.</p>
          <Barras titulo="Ventas por hora del día" datos={datos.por_hora} valor={v} detalle={detalle} cadaEtiqueta={3}
            etiqueta={(d, largo) => (largo ? `${d.hora}:00 – ${d.hora}:59` : `${d.hora}h`)} />
        </div>
      )}

      <div className="panel tabla-scroll">
        <h2>Tabla</h2>
        <table>
          <thead><tr><th>Producto</th><th className="num">Unidades</th><th className="num">Ingresos</th><th className="num">% de los ingresos</th></tr></thead>
          <tbody>
            {datos.por_producto.map((f) => (
              <tr key={f.categoria}>
                <td><span className="punto" style={{ background: COLORES_CLASE[f.clase_id] }} />{f.nombre}</td>
                <td className="num">{f.unidades}</td>
                <td className="num">{pesos.format(f.ingresos)}</td>
                <td className="num">{datos.ingresos ? `${((100 * f.ingresos) / datos.ingresos).toFixed(1)} %` : "—"}</td>
              </tr>
            ))}
            <tr><td><b>Total</b></td><td className="num"><b>{datos.unidades}</b></td><td className="num"><b>{pesos.format(datos.ingresos)}</b></td><td></td></tr>
          </tbody>
        </table>
      </div>
    </>
  );
}
