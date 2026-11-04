// Funciones para hablar con la API de FastAPI. Todas las rutas son relativas:
// en desarrollo Vite las reenvía al backend (ver vite.config.js).

async function pedir(metodo, ruta, cuerpo) {
  const opciones = { method: metodo, headers: {} };
  if (cuerpo !== undefined) {
    opciones.headers["Content-Type"] = "application/json";
    opciones.body = JSON.stringify(cuerpo);
  }
  const r = await fetch(ruta, opciones);
  if (!r.ok) {
    // Sesión vencida: la aplicación vuelve a la pantalla de ingreso.
    if (r.status === 401 && !ruta.startsWith("/api/auth/")) window.dispatchEvent(new Event("sesion-vencida"));
    const error = new Error(await mensajeDeError(r));
    error.estado = r.status;
    throw error;
  }
  return r.status === 204 ? null : r.json();
}

async function mensajeDeError(respuesta) {
  try {
    const { detail } = await respuesta.json();
    if (typeof detail === "string") return detail;
    // Errores de validación de FastAPI: lista de {loc, msg}
    if (Array.isArray(detail)) return detail.map((d) => `${d.loc.at(-1)}: ${d.msg}`).join("; ");
  } catch {
    /* la respuesta no era JSON */
  }
  return `Error ${respuesta.status} en el servidor`;
}

const consulta = (params) => {
  const limpio = Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== ""));
  const texto = new URLSearchParams(limpio).toString();
  return texto ? `?${texto}` : "";
};

export const api = {
  yo: () => pedir("GET", "/api/auth/yo"),
  login: (usuario, clave) => pedir("POST", "/api/auth/login", { usuario, clave }),
  salir: () => pedir("POST", "/api/auth/salir"),

  productos: () => pedir("GET", "/api/productos"),
  actualizarProducto: (id, datos) => pedir("PATCH", `/api/productos/${id}`, datos),
  crearProducto: (datos) => pedir("POST", "/api/productos", datos),
  borrarProducto: (id, forzar) => pedir("DELETE", `/api/productos/${id}${forzar ? "?forzar=true" : ""}`),

  sesiones: () => pedir("GET", "/api/sesiones"),
  sesion: (id) => pedir("GET", `/api/sesiones/${id}`),
  crearSesion: (datos) => pedir("POST", "/api/sesiones", datos),
  borrarSesion: (id) => pedir("DELETE", `/api/sesiones/${id}`),

  archivos: (sesionId) => pedir("GET", `/api/sesiones/${sesionId}/archivos`),
  reextraer: (id, params) => pedir("POST", `/api/archivos/${id}/extraer${consulta(params)}`),
  borrarArchivo: (id) => pedir("DELETE", `/api/archivos/${id}`),

  imagenes: (params) => pedir("GET", `/api/imagenes${consulta(params)}`),
  imagen: (id) => pedir("GET", `/api/imagenes/${id}`),
  siguientePendiente: (sesionId) => pedir("GET", `/api/imagenes/siguiente-pendiente${consulta({ sesion_id: sesionId })}`),
  guardarAnotaciones: (id, anotaciones, estado) =>
    pedir("PUT", `/api/imagenes/${id}/anotaciones`, { anotaciones, estado }),
  borrarImagen: (id) => pedir("DELETE", `/api/imagenes/${id}`),

  resumenDataset: () => pedir("GET", "/api/dataset/resumen"),
  exportarDataset: (params) => pedir("POST", "/api/dataset/exportar", params),

  entrenamientos: () => pedir("GET", "/api/entrenamiento"),
  entrenamiento: (id) => pedir("GET", `/api/entrenamiento/${id}`),
  requisitosEntrenamiento: () => pedir("GET", "/api/entrenamiento/requisitos"),
  iniciarEntrenamiento: (params) => pedir("POST", "/api/entrenamiento/iniciar", params),
  detenerEntrenamiento: () => pedir("POST", "/api/entrenamiento/detener"),

  estadisticas: (dias) => pedir("GET", `/api/estadisticas${consulta({ dias })}`),
  reiniciarDatos: () => pedir("POST", "/api/inventario/reiniciar", { confirmacion: "REINICIAR" }),

  videosPrueba: () => pedir("GET", "/api/evaluacion/videos"),
  videoPrueba: (id) => pedir("GET", `/api/evaluacion/videos/${id}`),
  guardarEventosReales: (id, eventos, tolerancia_s) => pedir("PUT", `/api/evaluacion/videos/${id}/eventos`, { eventos, tolerancia_s }),
  evaluarVideo: (id) => pedir("POST", `/api/evaluacion/videos/${id}/evaluar`),
  borrarVideoPrueba: (id) => pedir("DELETE", `/api/evaluacion/videos/${id}`),
  resumenEvaluacion: () => pedir("GET", "/api/evaluacion/resumen"),

  salud: () => pedir("GET", "/api/salud"),
  panel: () => pedir("GET", "/api/panel"),
  modelo: () => pedir("GET", "/api/vision/modelo"),
  movimientos: (params) => pedir("GET", `/api/movimientos${consulta(params)}`),
  sincronizar: (evidencia, modo) => pedir("POST", "/api/inventario/sincronizar", { evidencia, modo }),
  monitor: () => pedir("GET", "/api/monitor/estado"),
  iniciarMonitor: (fuente) => pedir("POST", "/api/monitor/iniciar", { fuente }),
  detenerMonitor: () => pedir("POST", "/api/monitor/detener"),
  camaras: () => pedir("GET", "/api/monitor/camaras"),
  ajustarMonitor: (cambios) => pedir("PATCH", "/api/monitor/parametros", cambios),
  contarFoto: async (archivo) => {
    const datos = new FormData();
    datos.append("foto", archivo);
    const r = await fetch("/api/inventario/conteo", { method: "POST", body: datos });
    if (!r.ok) throw new Error(await mensajeDeError(r));
    return r.json();
  },
};

// Sube un archivo con XMLHttpRequest (fetch no informa el progreso de subida).
export function subirArchivo(sesionId, archivo, alProgresar) {
  return subirFormulario(`/api/sesiones/${sesionId}/archivos`, { archivos: archivo }, alProgresar);
}

// Sube un video de prueba para la evaluación de retiros.
export const subirVideoPrueba = (archivo, nombre, alProgresar) =>
  subirFormulario("/api/evaluacion/videos", { archivo, nombre }, alProgresar);

function subirFormulario(ruta, campos, alProgresar) {
  return new Promise((resolver, rechazar) => {
    const datos = new FormData();
    for (const [k, v] of Object.entries(campos)) datos.append(k, v);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", ruta);
    xhr.upload.onprogress = (e) => e.lengthComputable && alProgresar(e.loaded / e.total);
    xhr.onload = () => {
      let cuerpo = null;
      try {
        cuerpo = JSON.parse(xhr.responseText);
      } catch {
        /* sin cuerpo JSON */
      }
      if (xhr.status >= 200 && xhr.status < 300) return resolver(cuerpo);
      if (xhr.status === 401) window.dispatchEvent(new Event("sesion-vencida"));
      if (xhr.status === 413) {
        return rechazar(new Error("El archivo pesa más de 100 MB (límite de Cloudflare): graba videos más cortos (≈ 1 minuto)."));
      }
      rechazar(new Error(cuerpo?.detail ?? `Error ${xhr.status}`));
    };
    xhr.onerror = () => rechazar(new Error("No se pudo conectar con el servidor"));
    xhr.send(datos);
  });
}

export const pesos = new Intl.NumberFormat("es-CO", { style: "currency", currency: "COP", maximumFractionDigits: 0 });

export function tamano(bytes) {
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

// Un color por clase, el mismo en toda la interfaz (índice = clase_id). Son
// variables CSS (ver estilos.css) con un tono para modo claro y otro para oscuro,
// tomados de una paleta validada para daltonismo. Como en modo claro tres de
// ellos tienen poco contraste, el color nunca va solo: siempre con el nombre.
// Hasta 8 productos: un color fijo por clase (paleta validada para daltonismo en estilos.css).
export const COLORES_CLASE = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => `var(--clase-${i})`);

export const fechaHora = new Intl.DateTimeFormat("es-CO", { dateStyle: "medium", timeStyle: "short" });
export const hora = new Intl.DateTimeFormat("es-CO", { timeStyle: "short" });

export function haceCuanto(fecha) {
  if (!fecha) return "nunca";
  const s = (Date.now() - new Date(fecha).getTime()) / 1000;
  if (s < 60) return "hace un momento";
  if (s < 3600) return `hace ${Math.floor(s / 60)} min`;
  if (s < 86400) return `hace ${Math.floor(s / 3600)} h`;
  return fechaHora.format(new Date(fecha));
}

// Fases del monitoreo continuo (backend/app/monitor.py).
export const FASES = {
  detenido: { texto: "Detenido", descripcion: "La cámara no se está analizando." },
  cargando: { texto: "Cargando modelo", descripcion: "Cargando YOLO en la GPU; tarda unos segundos solo la primera vez." },
  iniciando: { texto: "Iniciando", descripcion: "Llenando la ventana de fotogramas y esperando que la escena esté quieta para el conteo inicial." },
  estable: { texto: "Estable", descripcion: "Lo que se ve coincide con el inventario." },
  movimiento: { texto: "Movimiento", descripcion: "Hay movimiento (una mano): no se confirma ningún cambio hasta que la escena se quede quieta." },
  confirmando: { texto: "Confirmando", descripcion: "Un conteo cambió con la escena quieta; si se mantiene el tiempo de confirmación se registra la venta o reposición." },
  finalizado: { texto: "Finalizado", descripcion: "Terminó el video de prueba." },
  error: { texto: "Error", descripcion: "El monitoreo se detuvo por un error." },
};

export const TIPOS_MOVIMIENTO = {
  venta: { texto: "Venta", clase: "venta" },
  reposicion: { texto: "Reposición", clase: "reposicion" },
  ajuste: { texto: "Ajuste", clase: "ajuste" },
};
