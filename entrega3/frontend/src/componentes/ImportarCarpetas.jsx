import { useRef, useState } from "react";
import { api, subirArchivos } from "../api.js";
import { MAXIMO_POR_DEFECTO, agrupar, archivosDeFila, filaPorDefecto, lotes } from "../importacion.js";

// Importa varias carpetas de fotos: una sesión por carpeta, con tope de fotos por carpeta.
export default function ImportarCarpetas({ alTerminar, alCerrar }) {
  const [archivos, setArchivos] = useState([]);
  const [incluirVideos, setIncluirVideos] = useState(false);
  const [maximo, setMaximo] = useState(MAXIMO_POR_DEFECTO);
  const [filas, setFilas] = useState([]);
  const [fase, setFase] = useState("elegir"); // elegir | importando | listo
  const [avance, setAvance] = useState({}); // índice de fila -> {estado, subidos, total, codigo, rechazados, error}
  const cancelar = useRef(false);

  async function armar(lista, videos, tope) {
    let descripciones = [];
    try {
      descripciones = (await api.sesiones()).map((s) => s.descripcion);
    } catch { /* se arma sin detectar las carpetas ya importadas */ }
    setFilas(agrupar(lista, { incluirVideos: videos }).map((g) => filaPorDefecto(g, tope, descripciones)));
  }

  function elegir(e) {
    const lista = [...e.target.files];
    setArchivos(lista);
    armar(lista, incluirVideos, maximo);
  }

  function cambiarVideos(v) {
    setIncluirVideos(v);
    armar(archivos, v, maximo);
  }

  function aplicarMaximo(valor) {
    const n = Math.max(1, Number(valor) || 1);
    setMaximo(n);
    setFilas((fs) => fs.map((f) => ({ ...f, aSubir: Math.min(f.fotos.length, n) })));
  }

  const editar = (i, cambios) => setFilas((fs) => fs.map((f, j) => (j === i ? { ...f, ...cambios } : f)));
  const marcadas = filas.flatMap((f, i) => (f.incluir && f.fotos.length + f.videos.length > 0 ? [i] : []));

  async function importar() {
    cancelar.current = false;
    setFase("importando");
    const inicial = {};
    for (const i of marcadas) inicial[i] = { estado: "en espera", subidos: 0, total: archivosDeFila(filas[i]).length, rechazados: [] };
    setAvance(inicial);
    const poner = (i, c) => setAvance((a) => ({ ...a, [i]: { ...a[i], ...c } }));

    for (const i of marcadas) {
      if (cancelar.current) {
        poner(i, { estado: "cancelada" });
        continue;
      }
      const fila = filas[i];
      poner(i, { estado: "subiendo" });
      try {
        const sesion = await api.crearSesion({
          descripcion: fila.descripcion,
          iluminacion: fila.iluminacion,
          superficie: fila.superficie,
          prueba_generalizacion: false,
        });
        poner(i, { codigo: sesion.codigo });
        const lista = lotes(archivosDeFila(fila));
        const rechazados = [];
        let subidos = 0;
        let enviados = 0;
        // Secuencial y en orden: el backend reparte train/val/test por bloques de id dentro de la sesión.
        for (const lote of lista) {
          if (cancelar.current) break;
          const r = await subirArchivos(sesion.id, lote, () => {});
          rechazados.push(...(r?.rechazados ?? []));
          enviados += 1;
          subidos += lote.length;
          poner(i, { subidos, rechazados: [...rechazados] });
        }
        poner(i, { estado: enviados < lista.length ? "cancelada" : "lista" });
      } catch (err) {
        poner(i, { estado: "error", error: err.message });
      }
    }
    setFase("listo");
    alTerminar?.();
  }

  const totalArchivos = Object.values(avance).reduce((s, a) => s + a.total, 0);
  const totalSubidos = Object.values(avance).reduce((s, a) => s + a.subidos, 0);

  return (
    <div className="panel">
      <div className="fila" style={{ justifyContent: "space-between" }}>
        <h2>Importar carpetas</h2>
        <button type="button" onClick={alCerrar} disabled={fase === "importando"}>Cerrar</button>
      </div>

      {fase === "elegir" && (
        <>
          <div className="fila" style={{ alignItems: "flex-end" }}>
            <label className="campo">
              Carpeta raíz (se recorren sus subcarpetas)
              <input type="file" webkitdirectory="" multiple onChange={elegir} />
            </label>
            <label className="check" title="Los videos no cuentan para el tope por carpeta">
              <input type="checkbox" checked={incluirVideos} onChange={(e) => cambiarVideos(e.target.checked)} />
              Incluir videos
            </label>
            <label className="campo">
              Máximo de fotos por carpeta
              <input type="number" min="1" value={maximo} style={{ width: 110 }} onChange={(e) => aplicarMaximo(e.target.value)} />
            </label>
          </div>

          {filas.length > 0 && (
            <div className="tabla-scroll" style={{ marginTop: 12 }}>
              <table>
                <thead>
                  <tr>
                    <th></th><th>Carpeta</th><th className="num">Fotos encontradas</th><th>Fotos a subir</th>
                    <th>Descripción</th><th>Iluminación</th><th>Superficie</th>
                  </tr>
                </thead>
                <tbody>
                  {filas.map((f, i) => (
                    <tr key={f.ruta}>
                      <td><input type="checkbox" checked={f.incluir} onChange={(e) => editar(i, { incluir: e.target.checked })} /></td>
                      <td>
                        <code>{f.ruta || "(raíz)"}</code>
                        {f.yaImportada && <> <span className="etiqueta aviso">ya importada</span></>}
                        {f.videos.length > 0 && <small className="subtitulo"> + {f.videos.length} videos</small>}
                      </td>
                      <td className="num">{f.fotos.length}</td>
                      <td>
                        <div className="fila" style={{ flexWrap: "nowrap" }}>
                          <input type="number" min="0" max={f.fotos.length} value={f.aSubir} style={{ width: 80 }}
                            onChange={(e) => editar(i, { aSubir: Math.max(0, Math.min(f.fotos.length, Number(e.target.value) || 0)) })} />
                          <button type="button" onClick={() => editar(i, { aSubir: f.fotos.length })}>todas</button>
                        </div>
                      </td>
                      <td><input value={f.descripcion} maxLength={200} onChange={(e) => editar(i, { descripcion: e.target.value })} /></td>
                      <td><input value={f.iluminacion} maxLength={30} style={{ width: 120 }} onChange={(e) => editar(i, { iluminacion: e.target.value })} /></td>
                      <td><input value={f.superficie} maxLength={50} style={{ width: 120 }} onChange={(e) => editar(i, { superficie: e.target.value })} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {archivos.length > 0 && filas.length === 0 && <p className="subtitulo">No hay fotos (.jpg, .jpeg, .png) en esa carpeta.</p>}
          <div className="fila" style={{ marginTop: 12 }}>
            <button type="button" className="primario" disabled={marcadas.length === 0} onClick={importar}>
              Importar {marcadas.length} {marcadas.length === 1 ? "carpeta" : "carpetas"}
            </button>
          </div>
        </>
      )}

      {fase !== "elegir" && (
        <>
          <p>Total: {totalSubidos} / {totalArchivos} archivos</p>
          <div className="barra"><div style={{ width: `${totalArchivos ? (100 * totalSubidos) / totalArchivos : 0}%` }} /></div>
          {fase === "importando" && (
            <button type="button" style={{ marginTop: 10 }} onClick={() => { cancelar.current = true; }}>
              Cancelar (termina el lote en curso)
            </button>
          )}
          <table style={{ marginTop: 12 }}>
            <thead><tr><th>Carpeta</th><th>Sesión</th><th>Estado</th><th className="num">Subidos</th></tr></thead>
            <tbody>
              {marcadas.map((i) => {
                const a = avance[i] ?? {};
                return (
                  <tr key={filas[i].ruta}>
                    <td><code>{filas[i].ruta}</code></td>
                    <td>{a.codigo ?? "—"}</td>
                    <td>
                      {a.estado === "error" ? <span className="etiqueta error">error: {a.error}</span> : a.estado}
                      {a.rechazados?.length > 0 && (
                        <details>
                          <summary>{a.rechazados.length} rechazados</summary>
                          <ul>{a.rechazados.map((r, k) => <li key={k}>{r.nombre}: {r.motivo}</li>)}</ul>
                        </details>
                      )}
                    </td>
                    <td className="num">{a.subidos ?? 0} / {a.total ?? 0}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {fase === "listo" && (
            <div className="alerta ok" style={{ marginTop: 12 }}>
              Códigos creados: {marcadas.map((i) => avance[i]?.codigo).filter(Boolean).join(", ") || "ninguno"}.
            </div>
          )}
        </>
      )}
    </div>
  );
}
