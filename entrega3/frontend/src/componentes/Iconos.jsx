// Íconos en SVG de trazo, dibujados a mano para no depender de una librería.
// Heredan el color del texto (currentColor) y miden 1em por defecto.

function Icono({ children, tamano = "1.15em", ...resto }) {
  return (
    <svg width={tamano} height={tamano} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...resto}>
      {children}
    </svg>
  );
}

export const IconoPanel = (p) => (
  <Icono {...p}><rect x="3" y="3" width="7" height="9" rx="1.5" /><rect x="14" y="3" width="7" height="5" rx="1.5" /><rect x="14" y="12" width="7" height="9" rx="1.5" /><rect x="3" y="16" width="7" height="5" rx="1.5" /></Icono>
);
export const IconoCamara = (p) => (
  <Icono {...p}><path d="M4 7h3l2-3h6l2 3h3a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1z" /><circle cx="12" cy="13" r="4" /></Icono>
);
export const IconoHistorial = (p) => (
  <Icono {...p}><path d="M3 12a9 9 0 1 0 3-6.7" /><path d="M3 4v5h5" /><path d="M12 7v5l3 2" /></Icono>
);
export const IconoCatalogo = (p) => (
  <Icono {...p}><path d="M21 8 12 3 3 8v8l9 5 9-5z" /><path d="m3 8 9 5 9-5" /><path d="M12 13v8" /></Icono>
);
export const IconoCarpeta = (p) => (
  <Icono {...p}><path d="M3 6a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z" /></Icono>
);
export const IconoEtiqueta = (p) => (
  <Icono {...p}><rect x="3" y="3" width="18" height="18" rx="2" strokeDasharray="3 3" /><rect x="7" y="8" width="8" height="8" rx="1" /></Icono>
);
export const IconoExportar = (p) => (
  <Icono {...p}><path d="M12 3v12" /><path d="m7 10 5 5 5-5" /><path d="M4 19h16" /></Icono>
);
export const IconoEntrenar = (p) => (
  <Icono {...p}><rect x="5" y="5" width="14" height="14" rx="2" /><rect x="9" y="9" width="6" height="6" /><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" /></Icono>
);
export const IconoGrafica = (p) => (
  <Icono {...p}><path d="M4 20h16" /><path d="M7 16v-5" /><path d="M12 16V7" /><path d="M17 16v-8" /></Icono>
);
export const IconoMeta = (p) => (
  <Icono {...p}><circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="4" /><circle cx="12" cy="12" r="0.8" /></Icono>
);
export const IconoSubir = (p) => (
  <Icono {...p}><path d="M12 16V4" /><path d="m7 9 5-5 5 5" /><path d="M4 20h16" /></Icono>
);
export const IconoAlerta = (p) => (
  <Icono {...p}><path d="M12 3 2 20h20z" /><path d="M12 10v4" /><path d="M12 17h.01" /></Icono>
);
export const IconoCheck = (p) => (
  <Icono {...p}><circle cx="12" cy="12" r="9" /><path d="m8 12 3 3 5-6" /></Icono>
);
export const IconoInfo = (p) => (
  <Icono {...p}><circle cx="12" cy="12" r="9" /><path d="M12 11v5" /><path d="M12 8h.01" /></Icono>
);
export const IconoVision = (p) => (
  <Icono {...p}><path d="M3 8V5a2 2 0 0 1 2-2h3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M8 21H5a2 2 0 0 1-2-2v-3" /><path d="M6.5 12s2-4 5.5-4 5.5 4 5.5 4-2 4-5.5 4-5.5-4-5.5-4z" /><circle cx="12" cy="12" r="1.6" /></Icono>
);
export const IconoPlay = (p) => (
  <Icono {...p}><path d="M7 4.5v15l12-7.5z" /></Icono>
);
export const IconoStop = (p) => (
  <Icono {...p}><rect x="6" y="6" width="12" height="12" rx="1.5" /></Icono>
);
export const IconoFlecha =({ direccion = "abajo", ...p }) => (
  <Icono {...p} style={{ transform: direccion === "arriba" ? "rotate(180deg)" : undefined }}>
    <path d="M12 5v14" /><path d="m6 13 6 6 6-6" />
  </Icono>
);

// Un ícono por categoría de producto (vista aproximada desde un costado).
const PRODUCTOS = {
  botella: <><path d="M10 2h4v3l1.5 2.5V20a2 2 0 0 1-2 2h-3a2 2 0 0 1-2-2V7.5L10 5z" /><path d="M8.5 12h7" /></>,
  lata: <><ellipse cx="12" cy="5" rx="5" ry="2" /><path d="M7 5v14c0 1.1 2.2 2 5 2s5-.9 5-2V5" /><path d="M7 9c0 1.1 2.2 2 5 2s5-.9 5-2" /></>,
  vaso: <><path d="M5 6h14l-1.5 15h-11z" /><path d="M4 3h16v3H4z" /><path d="M6.5 11h11" /></>,
  paquete: <><path d="M5 3h14l-1 3 1 3-1 3 1 3-1 3 1 3H5l1-3-1-3 1-3-1-3 1-3z" /><path d="M9 10h6" /></>,
  fruta: <><path d="M12 7c-3-2-8-1-8 5 0 5 3 9 5 9 1.2 0 1.8-.6 3-.6s1.8.6 3 .6c2 0 5-4 5-9 0-6-5-7-8-5z" /><path d="M12 7c0-2 1-4 3-5" /></>,
};

export function IconoProducto({ categoria, ...p }) {
  return <Icono {...p}>{PRODUCTOS[categoria] ?? <rect x="4" y="4" width="16" height="16" rx="3" />}</Icono>;
}
