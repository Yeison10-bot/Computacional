import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { api } from "../api.js";
import Login from "../paginas/Login.jsx";

// Sesión del usuario: si el servidor exige login y no hay sesión, muestra la
// pantalla de ingreso en lugar de la aplicación. También informa si el servidor
// permite la cámara conectada a él (en el servidor desplegado, no).
const Contexto = createContext({ usuario: null, login_activo: false, camara_servidor: true });

export function ProveedorSesion({ children }) {
  const [sesion, setSesion] = useState(undefined); // undefined = consultando, null = sin sesión
  const [error, setError] = useState("");

  const consultar = useCallback(() =>
    api.yo().then((s) => { setSesion(s); setError(""); }).catch((e) => {
      if (e.estado === 401) setSesion(null);
      else setError("No hay conexión con el servidor. Reintentando…");
    }), []);

  useEffect(() => {
    consultar();
    // Si cualquier petición recibe 401 (la sesión venció), se vuelve al login.
    const vencida = () => setSesion(null);
    window.addEventListener("sesion-vencida", vencida);
    return () => window.removeEventListener("sesion-vencida", vencida);
  }, [consultar]);

  useEffect(() => {
    if (sesion !== undefined) return;
    const t = setInterval(consultar, 3000);
    return () => clearInterval(t);
  }, [sesion, consultar]);

  if (sesion === undefined) return <div className="pantalla-login"><p>{error || "Cargando…"}</p></div>;
  if (sesion === null) return <Login alEntrar={setSesion} />;
  const salir = async () => { await api.salir().catch(() => {}); setSesion(null); };
  return <Contexto.Provider value={{ ...sesion, salir }}>{children}</Contexto.Provider>;
}

export const useSesion = () => useContext(Contexto);
