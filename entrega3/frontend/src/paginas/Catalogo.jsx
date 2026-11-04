import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { COLORES_CLASE, api, pesos } from "../api.js";
import { useAvisos } from "../componentes/Avisos.jsx";
import { IconoCamara, IconoInfo, IconoProducto } from "../componentes/Iconos.jsx";

const MAXIMO = 8;
const PATRON = /^[a-z][a-z0-9_]{1,29}$/;
// "Taza Roja" → "taza_roja": el nombre de la clase en YOLO, sugerido a partir del nombre comercial.
const aClase = (texto) => texto.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "")
  .replace(/[^a-z0-9]+/g, "_").replace(/^[^a-z]+|_+$/g, "").slice(0, 30);
const VACIO = { nombre: "", categoria_yolo: "", precio: "", stock_minimo: 3 };

// Catálogo: cada producto es una clase del modelo. Se pueden agregar, modificar y eliminar.
// Las existencias NO se editan aquí: las actualiza la cámara desde la Vitrina.
export default function Catalogo() {
  const avisar = useAvisos();
  const [productos, setProductos] = useState([]);
  const [borradores, setBorradores] = useState({}); // cambios sin guardar, por id
  const [nuevo, setNuevo] = useState(VACIO);
  const [claseTocada, setClaseTocada] = useState(false);

  const cargar = () => api.productos().then(setProductos).catch((e) => avisar(e.message, "error"));
  useEffect(() => { cargar(); }, []);

  const editar = (id, campo, valor) =>
    setBorradores((b) => ({ ...b, [id]: { ...b[id], [campo]: valor } }));

  async function guardar(p) {
    const datos = {};
    for (const [campo, valor] of Object.entries(borradores[p.id])) {
      datos[campo] = ["nombre", "categoria_yolo"].includes(campo) ? valor.trim() : Number(valor);
    }
    if (datos.categoria_yolo !== undefined && datos.categoria_yolo !== p.categoria_yolo) {
      if (!PATRON.test(datos.categoria_yolo)) return avisar("La clase solo admite minúsculas, números y _ (p. ej. taza_roja)", "error");
      if (p.cajas > 0 && !confirm(`Cambiar la clase de «${p.nombre}» obliga a exportar el dataset y reentrenar el modelo. ¿Continuar?`)) return;
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

  async function agregar(e) {
    e.preventDefault();
    const datos = { ...nuevo, nombre: nuevo.nombre.trim(), categoria_yolo: nuevo.categoria_yolo.trim(),
      precio: Number(nuevo.precio), stock_minimo: Number(nuevo.stock_minimo) };
    if (!PATRON.test(datos.categoria_yolo)) return avisar("La clase solo admite minúsculas, números y _ (p. ej. taza_roja)", "error");
    try {
      const p = await api.crearProducto(datos);
      avisar(`«${p.nombre}» agregado como clase ${p.clase_id} (${p.categoria_yolo})`);
      setNuevo(VACIO);
      setClaseTocada(false);
      cargar();
    } catch (err) {
      avisar(err.message, "error");
    }
  }

  async function eliminar(p) {
    const usos = p.cajas || p.movimientos
      ? `\n\nTiene ${p.cajas} cajas etiquetadas y ${p.movimientos} movimientos (ventas, reposiciones, ajustes) que también se borrarán. ` +
        "Las imágenes que queden sin cajas volverán a «pendiente»."
      : "";
    if (!confirm(`¿Eliminar «${p.nombre}» (clase ${p.categoria_yolo})?${usos}\n\nDespués hay que exportar el dataset y reentrenar.`)) return;
    try {
      await api.borrarProducto(p.id, Boolean(usos));
      avisar(`«${p.nombre}» eliminado`);
      cargar();
    } catch (e) {
      avisar(e.message, "error");
    }
  }

  async function reiniciar() {
    const texto = prompt("Esto borra TODO el historial de ventas y deja las existencias en 0.\nEscribe REINICIAR para confirmar:");
    if (texto === null) return;
    if (texto.trim() !== "REINICIAR") return avisar("No se reinició: hay que escribir REINICIAR", "error");
    try {
      const r = await api.reiniciarDatos();
      cargar();
      avisar(`Datos reiniciados: ${r.movimientos_borrados} movimientos borrados`);
    } catch (e) {
      avisar(e.message, "error");
    }
  }

  return (
    <>
      <div className="encabezado">
        <div>
          <h1>Catálogo</h1>
          <p className="subtitulo">Cada producto es una clase que el modelo aprende a reconocer. Usa objetos concretos y fáciles de distinguir entre sí.</p>
        </div>
      </div>

      <div className="alerta info">
        <IconoInfo />
        <span>
          Las existencias se actualizan solas con lo que ve la cámara (<Link to="/vitrina"><IconoCamara tamano="0.95em" /> Vitrina</Link>).
          Un cambio de precio no altera las ventas ya registradas. <b>Agregar, renombrar la clase o eliminar productos</b> obliga
          a volver a exportar el dataset y a reentrenar: el modelo solo reconoce las clases con las que se entrenó.
        </span>
      </div>

      <div className="panel tabla-scroll">
        <table>
          <thead>
            <tr>
              <th>Clase del modelo</th>
              <th>Nombre en la cafetería</th>
              <th className="num">Precio (COP)</th>
              <th className="num">Mínimo</th>
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
                      <input value={valor("categoria_yolo")} onChange={(e) => editar(p.id, "categoria_yolo", e.target.value.toLowerCase())}
                        style={{ width: 130, fontFamily: "ui-monospace, monospace" }} aria-label="Clase del modelo" />
                    </span>
                    <div className="subtitulo" style={{ fontSize: "0.78rem", marginTop: 2 }}>clase {p.clase_id} · tecla {p.clase_id + 1} · {p.cajas} cajas</div>
                  </td>
                  <td>
                    <input value={valor("nombre")} onChange={(e) => editar(p.id, "nombre", e.target.value)} style={{ width: "100%", minWidth: 150 }} />
                  </td>
                  <td className="num">
                    <input type="number" min="0" step="100" value={valor("precio")}
                      onChange={(e) => editar(p.id, "precio", e.target.value)} style={{ width: 95, textAlign: "right" }} />
                    <div className="subtitulo" style={{ fontSize: "0.78rem" }}>{pesos.format(valor("precio") || 0)}</div>
                  </td>
                  <td className="num">
                    <input type="number" min="0" value={valor("stock_minimo")}
                      onChange={(e) => editar(p.id, "stock_minimo", e.target.value)} style={{ width: 65, textAlign: "right" }} />
                  </td>
                  <td className="num">
                    <b style={{ fontSize: "1.15rem" }}>{p.cantidad}</b>
                    {p.cantidad <= p.stock_minimo && <div><span className="etiqueta error">stock bajo</span></div>}
                  </td>
                  <td>
                    <div className="fila" style={{ flexWrap: "nowrap" }}>
                      <button className="primario pequeno" disabled={!borradores[p.id]} onClick={() => guardar(p)}>Guardar</button>
                      <button className="peligro pequeno" disabled={productos.length <= 1} onClick={() => eliminar(p)}>Eliminar</button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <h2>Agregar producto</h2>
        {productos.length >= MAXIMO ? (
          <p className="subtitulo">Ya hay {MAXIMO} productos, el máximo. Elimina uno para agregar otro.</p>
        ) : (
          <form className="fila" style={{ alignItems: "flex-end" }} onSubmit={agregar}>
            <label className="campo" style={{ flex: 2, minWidth: 200 }}>Nombre en la cafetería
              <input value={nuevo.nombre} required maxLength={80} placeholder="Ej.: Taza roja de cerámica"
                onChange={(e) => setNuevo({ ...nuevo, nombre: e.target.value, ...(!claseTocada && { categoria_yolo: aClase(e.target.value) }) })} />
            </label>
            <label className="campo" style={{ flex: 1, minWidth: 150 }}>Clase del modelo
              <input value={nuevo.categoria_yolo} required placeholder="taza_roja" style={{ fontFamily: "ui-monospace, monospace" }}
                onChange={(e) => { setClaseTocada(true); setNuevo({ ...nuevo, categoria_yolo: e.target.value.toLowerCase() }); }} />
            </label>
            <label className="campo">Precio (COP)
              <input type="number" min="0" step="100" required value={nuevo.precio} style={{ width: 110 }}
                onChange={(e) => setNuevo({ ...nuevo, precio: e.target.value })} />
            </label>
            <label className="campo">Stock mínimo
              <input type="number" min="0" value={nuevo.stock_minimo} style={{ width: 80 }}
                onChange={(e) => setNuevo({ ...nuevo, stock_minimo: e.target.value })} />
            </label>
            <button className="primario" type="submit">Agregar</button>
          </form>
        )}
        <p className="subtitulo" style={{ marginTop: 10 }}>
          La clase es el nombre interno que aprende el modelo: minúsculas, números y guion bajo. Puedes tener hasta {MAXIMO} productos;
          con objetos muy parecidos entre sí (dos botellas casi iguales) el modelo necesitará más fotos para distinguirlos.
        </p>
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
