import { useEffect, useState } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import { api } from "./api.js";
import { ProveedorAvisos } from "./componentes/Avisos.jsx";
import { ProveedorCamara } from "./componentes/CamaraNavegador.jsx";
import { ProveedorSesion, useSesion } from "./componentes/Sesion.jsx";
import {
  IconoCamara, IconoCarpeta, IconoCatalogo, IconoEntrenar, IconoEtiqueta, IconoExportar, IconoGrafica, IconoMeta, IconoHistorial, IconoPanel, IconoVision,
} from "./componentes/Iconos.jsx";
import Catalogo from "./paginas/Catalogo.jsx";
import Entrenar from "./paginas/Entrenar.jsx";
import Estadisticas from "./paginas/Estadisticas.jsx";
import Evaluacion from "./paginas/Evaluacion.jsx";
import Etiquetar from "./paginas/Etiquetar.jsx";
import Exportar from "./paginas/Exportar.jsx";
import Historial from "./paginas/Historial.jsx";
import Panel from "./paginas/Panel.jsx";
import SesionDetalle from "./paginas/SesionDetalle.jsx";
import Sesiones from "./paginas/Sesiones.jsx";
import Vitrina from "./paginas/Vitrina.jsx";

// Comprueba cada 5 s que la API responde y si la cámara está vigilando.
function useConexion() {
  const [estado, setEstado] = useState({ api: "pendiente", camara: null });
  useEffect(() => {
    const comprobar = () =>
      api.monitor()
        .then((m) => setEstado({ api: "ok", camara: m }))
        .catch(() => setEstado({ api: "error", camara: null }));
    comprobar();
    const t = setInterval(comprobar, 5000);
    return () => clearInterval(t);
  }, []);
  return estado;
}

export default function App() {
  return (
    <ProveedorAvisos>
      <ProveedorSesion>
        <Aplicacion />
      </ProveedorSesion>
    </ProveedorAvisos>
  );
}

function Aplicacion() {
  const { api: conexion, camara } = useConexion();
  const sesion = useSesion();
  // En el celular el menú se abre con el botón ☰ y se cierra al elegir una página.
  const [menuAbierto, setMenuAbierto] = useState(false);
  const ubicacion = useLocation();
  useEffect(() => setMenuAbierto(false), [ubicacion.pathname]);
  return (
      <ProveedorCamara>
      <div className="app">
        <nav className={`menu ${menuAbierto ? "abierto" : ""}`}>
          <div className="marca">
            <div className="logo"><IconoVision tamano="24" /></div>
            <div>
              <b>Inventario con <br />Visión Computacional</b>
              <small>Cafetería · Inteligencia Computacional II</small>
            </div>
            <button className="menu-boton" aria-label="Abrir el menú" aria-expanded={menuAbierto}
              onClick={() => setMenuAbierto((a) => !a)}>{menuAbierto ? "✕" : "☰"}</button>
          </div>
          <div className="grupo">Operación</div>
          <NavLink to="/" end><IconoPanel /> Panel</NavLink>
          <NavLink to="/vitrina"><IconoCamara /> Vitrina en vivo</NavLink>
          <NavLink to="/historial"><IconoHistorial /> Historial</NavLink>
          <NavLink to="/estadisticas"><IconoGrafica /> Ventas y estadísticas</NavLink>
          <NavLink to="/catalogo"><IconoCatalogo /> Catálogo</NavLink>
          <div className="grupo">Dataset</div>
          <NavLink to="/sesiones"><IconoCarpeta /> Sesiones de captura</NavLink>
          <NavLink to="/etiquetar"><IconoEtiqueta /> Etiquetar</NavLink>
          <NavLink to="/exportar"><IconoExportar /> Exportar dataset</NavLink>
          <NavLink to="/entrenar"><IconoEntrenar /> Entrenar modelo</NavLink>
          <div className="grupo">Evaluación</div>
          <NavLink to="/evaluacion"><IconoMeta /> Evaluación de retiros</NavLink>
          <div className="pie">
            <div>
              <span className={`estado-conexion ${camara?.activo ? "vivo" : ""}`} />
              {camara?.activo ? "Cámara vigilando la vitrina" : "Cámara detenida"}
            </div>
            <div>
              <span className={`estado-conexion ${conexion}`} />
              {{ ok: "Servidor conectado", error: "Sin conexión con el servidor", pendiente: "Conectando…" }[conexion]}
            </div>
            {sesion.login_activo && (
              <div className="sesion-usuario">
                <span>👤 {sesion.usuario}</span>
                <button className="pequeno" onClick={sesion.salir}>Salir</button>
              </div>
            )}
          </div>
        </nav>
        <main className="contenido">
          {conexion === "error" && (
            <div className="alerta error">
              No hay conexión con la API. Verifica que el backend esté corriendo (iniciar.ps1) y que PostgreSQL esté activo.
            </div>
          )}
          <Routes>
            <Route path="/" element={<Panel />} />
            <Route path="/vitrina" element={<Vitrina />} />
            <Route path="/historial" element={<Historial />} />
            <Route path="/catalogo" element={<Catalogo />} />
            <Route path="/sesiones" element={<Sesiones />} />
            <Route path="/sesiones/:id" element={<SesionDetalle />} />
            <Route path="/etiquetar" element={<Etiquetar />} />
            <Route path="/etiquetar/:id" element={<Etiquetar />} />
            <Route path="/exportar" element={<Exportar />} />
            <Route path="/entrenar" element={<Entrenar />} />
            <Route path="/estadisticas" element={<Estadisticas />} />
            <Route path="/evaluacion" element={<Evaluacion />} />
            <Route path="*" element={<p>Página no encontrada.</p>} />
          </Routes>
        </main>
      </div>
      </ProveedorCamara>
  );
}
