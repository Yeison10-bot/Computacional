// Lógica pura de «Importar carpetas»: agrupar, ordenar en orden temporal,
// submuestrear y valores por defecto. Sin dependencias de React ni del DOM.

export const MAXIMO_POR_DEFECTO = 70;
export const TAMANO_LOTE = 10;
const EXT_FOTO = [".jpg", ".jpeg", ".png"];
const EXT_VIDEO = [".mp4", ".mov"];

const extension = (nombre) => {
  const i = nombre.lastIndexOf(".");
  return i < 0 ? "" : nombre.slice(i).toLowerCase();
};
const dos = (n) => String(n).padStart(2, "0");

export const esFoto = (nombre) => EXT_FOTO.includes(extension(nombre));
export const esVideo = (nombre) => EXT_VIDEO.includes(extension(nombre));

// Clave de orden temporal de un nombre, o null si no se reconoce el patrón.
//   WhatsApp Image 2026-11-13 at 9.14.25 AM (2).jpeg  (reloj de 12 h; «(n)» desempata)
//   PHOTO-2026-11-14-17-19-100.jpg                    (aaaa-mm-dd-hh-mm-secuencia)
export function claveTemporal(nombre) {
  let m = /(\d{4})-(\d{2})-(\d{2}) at (\d{1,2})\.(\d{2})\.(\d{2})\s*([AP]M)(?:\s*\((\d+)\))?/i.exec(nombre);
  if (m) {
    let h = Number(m[4]) % 12;
    if (m[7].toUpperCase() === "PM") h += 12;
    return { fecha: `${m[1]}-${m[2]}-${m[3]}T${dos(h)}:${m[5]}:${m[6]}`, n: m[8] ? Number(m[8]) : 0 };
  }
  m = /PHOTO-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d+)/i.exec(nombre);
  if (m) return { fecha: `${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:00`, n: Number(m[6]) };
  return null;
}

const natural = (a, b) => a.localeCompare(b, undefined, { numeric: true });

// Orden total: primero los nombres con fecha reconocida (por fecha y desempate), luego el resto en orden natural.
export function compararNombres(a, b) {
  const ka = claveTemporal(a);
  const kb = claveTemporal(b);
  if (ka && kb) return ka.fecha < kb.fecha ? -1 : ka.fecha > kb.fecha ? 1 : ka.n - kb.n || natural(a, b);
  if (ka) return -1;
  if (kb) return 1;
  return natural(a, b);
}

const rutaCarpeta = (f) => {
  const partes = (f.webkitRelativePath || f.name).split("/");
  partes.pop();
  return partes.join("/");
};

// Agrupa por carpeta padre completa. Devuelve [{ruta, fotos, videos}] con cada lista en orden temporal.
export function agrupar(archivos, { incluirVideos = false } = {}) {
  const grupos = new Map();
  for (const f of archivos) {
    const foto = esFoto(f.name);
    if (!foto && !(incluirVideos && esVideo(f.name))) continue;
    const ruta = rutaCarpeta(f);
    if (!grupos.has(ruta)) grupos.set(ruta, { ruta, fotos: [], videos: [] });
    grupos.get(ruta)[foto ? "fotos" : "videos"].push(f);
  }
  const porNombre = (a, b) => compararNombres(a.name, b.name);
  return [...grupos.values()]
    .map((g) => ({ ...g, fotos: g.fotos.sort(porNombre), videos: g.videos.sort(porNombre) }))
    .sort((a, b) => natural(a.ruta, b.ruta));
}

// n elementos equiespaciados sobre la lista (primero y último incluidos): conserva variedad y orden.
export function submuestrear(lista, n) {
  const m = lista.length;
  n = Math.max(0, Math.min(Math.floor(n), m));
  if (n >= m) return [...lista];
  if (n === 0) return [];
  if (n === 1) return [lista[0]];
  return Array.from({ length: n }, (_, i) => lista[Math.round((i * (m - 1)) / (n - 1))]);
}

export function superficiePorDefecto(ruta) {
  const r = ruta.toUpperCase();
  return r.includes("NEGRO") ? "negra" : r.includes("BLANCO") ? "blanca" : "";
}

// Fila de la tabla previa con sus valores por defecto.
export function filaPorDefecto(grupo, maximo, descripcionesExistentes = []) {
  const descripcion = grupo.ruta.slice(-200);
  const yaImportada = descripcionesExistentes.includes(descripcion);
  return {
    ...grupo,
    incluir: !yaImportada,
    yaImportada,
    aSubir: Math.min(grupo.fotos.length, maximo),
    descripcion,
    iluminacion: "",
    superficie: superficiePorDefecto(grupo.ruta).slice(0, 50),
  };
}

// Archivos a subir de una fila, en el orden de envío: fotos submuestreadas y después los videos.
export const archivosDeFila = (fila) => [...submuestrear(fila.fotos, fila.aSubir), ...fila.videos];

export function lotes(lista, tamano = TAMANO_LOTE) {
  const salida = [];
  for (let i = 0; i < lista.length; i += tamano) salida.push(lista.slice(i, i + tamano));
  return salida;
}
