import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { COLORES_CLASE, api, pesos } from "../api.js";
import { useAvisos } from "../componentes/Avisos.jsx";
import { IconoCamara, IconoInfo, IconoProducto } from "../componentes/Iconos.jsx";

// Catálogo: nombre comercial, precio y stock mínimo de cada categoría.
// Las existencias NO se editan aquí: las actualiza la cámara desde la Vitrina.
export default function Catalogo() {
  const avisar = useAvisos();
  const [productos, setProductos] = useState([]);
  const [borradores, setBorradores] = useState({}); // cambios sin guardar, por id

  useEffect(() => {
    api.productos().then(setProductos).catch((e) => avisar(e.message, "error"));
  }, [avisar]);

  const editar = (id, campo, valor) =>
    setBorradores((b) => ({ ...b, [id]: { ...b[id], [campo]: valor } }));

  async function reiniciar() {
    const texto = prompt("Esto borra TODO el historial de ventas y deja las existencias en 0.\nEscribe REINICIAR para confirmar:");
    if (texto === null) return;
    if (texto.trim() !== "REINICIAR") return avisar("No se reinició: hay que escribir REINICIAR", "error");
    try {
      const r = await api.reiniciarDatos();
      setProductos(await api.productos());
      avisar(`Datos reiniciados: ${r.movimientos_borrados} movimientos borrados`);
    } catch (e) {
      avisar(e.message, "error");
    }
  }

  async function guardar(p) {
    const datos = {};
    for (const [campo, valor] of Object.entries(borradores[p.id])) {
      datos[campo] = campo === "nombre" ? valor : Number(valor);
    }
    try {
      const actualizado = await api.actualizarProducto(p.id, datos);
      setProductos((lista) => lista.map((x) => (x.id === p.id ? actualizado : x)));
      setBorradores(({ [p.id]: _, ...resto }) => resto);
      avisar(`«${actualizado.nombre}» guardado`);
    } catch (e) {
      avisar(e.message, "error");
    }
  }

  return (
    <>
      <div className="encabezado">
        <div>
          <h1>Catálogo</h1>
          <p className="subtitulo">Cada categoría que detecta el modelo corresponde a un producto de la cafetería.</p>
        </div>
      </div>

      <div className="alerta info">
        <IconoInfo />
        <span>
          Las existencias se actualizan solas con lo que ve la cámara. Para contarlas, ve a{" "}
          <Link to="/vitrina"><IconoCamara tamano="0.95em" /> Vitrina</Link>. Un cambio de precio no altera las ventas
          ya registradas: cada venta guarda el precio que tenía en ese momento.
        </span>
      </div>

      <div className="panel tabla-scroll">
        <table>
          <thead>
            <tr>
              <th>Producto</th>
              <th>Nombre en la cafetería</th>
              <th className="num">Precio (COP)</th>
              <th className="num">Stock mínimo</th>
              <th className="num">Existencias</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {productos.map((p) => {
              const b = borradores[p.id] ?? {};
              const valor = (campo) => b[campo] ?? p[campo];
              return (
                <tr key={p.id}>
                  <td>
                    <span className="categoria" style={{ color: COLORES_CLASE[p.clase_id] }}>
                      <IconoProducto categoria={p.categoria_yolo} />
                      <code style={{ color: "var(--texto)" }}>{p.categoria_yolo}</code>
                    </span>
                    <div className="subtitulo" style={{ fontSize: "0.78rem", marginTop: 2 }}>clase {p.clase_id} del modelo</div>
                  </td>
                  <td>
                    <input value={valor("nombre")} onChange={(e) => editar(p.id, "nombre", e.target.value)} style={{ width: "100%", minWidth: 180 }} />
                  </td>
                  <td className="num">
                    <input type="number" min="0" step="100" value={valor("precio")}
                      onChange={(e) => editar(p.id, "precio", e.target.value)} style={{ width: 110, textAlign: "right" }} />
                    <div className="subtitulo" style={{ fontSize: "0.78rem" }}>{pesos.format(valor("precio") || 0)}</div>
                  </td>
                  <td className="num">
                    <input type="number" min="0" value={valor("stock_minimo")}
                      onChange={(e) => editar(p.id, "stock_minimo", e.target.value)} style={{ width: 80, textAlign: "right" }} />
                  </td>
                  <td className="num">
                    <b style={{ fontSize: "1.15rem" }}>{p.cantidad}</b>
                    {p.cantidad <= p.stock_minimo && <div><span className="etiqueta error">stock bajo</span></div>}
                  </td>
                  <td>
                    <button className="primario pequeno" disabled={!borradores[p.id]} onClick={() => guardar(p)}>Guardar</button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <h2>Reiniciar datos de prueba</h2>
        <p className="subtitulo">
          Borra todas las ventas, reposiciones y ajustes, y deja las existencias en 0. Úsalo antes de la sustentación.
          No toca el catálogo, los precios, el dataset ni el modelo. Al iniciar la Vitrina en vivo, la cámara registra lo
          que hay en la vitrina como conteo inicial.
        </p>
        <button className="peligro" onClick={reiniciar}>Reiniciar datos de prueba</button>
      </div>
    </>
  );
}
