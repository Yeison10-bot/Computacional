import { useState } from "react";
import { api } from "../api.js";
import { IconoVision } from "../componentes/Iconos.jsx";

export default function Login({ alEntrar }) {
  const [usuario, setUsuario] = useState("");
  const [clave, setClave] = useState("");
  const [error, setError] = useState("");
  const [enviando, setEnviando] = useState(false);

  async function entrar(e) {
    e.preventDefault();
    setEnviando(true);
    setError("");
    try {
      await api.login(usuario, clave);
      alEntrar(await api.yo());
    } catch (err) {
      setError(err.message);
      setClave("");
    } finally {
      setEnviando(false);
    }
  }

  return (
    <div className="pantalla-login">
      <form className="panel tarjeta-login" onSubmit={entrar}>
        <div className="marca" style={{ padding: "0 0 18px" }}>
          <div className="logo"><IconoVision tamano="24" /></div>
          <div>
            <b style={{ color: "var(--texto)" }}>Inventario con <br />Visión Computacional</b>
            <small style={{ color: "var(--texto-suave)" }}>Cafetería · Inteligencia Computacional II</small>
          </div>
        </div>
        <h2>Iniciar sesión</h2>
        {error && <div className="alerta error"><span>{error}</span></div>}
        <label className="campo">Usuario
          <input value={usuario} onChange={(e) => setUsuario(e.target.value)} autoComplete="username"
            autoCapitalize="none" autoCorrect="off" required autoFocus />
        </label>
        <label className="campo">Contraseña
          <input type="password" value={clave} onChange={(e) => setClave(e.target.value)} autoComplete="current-password" required />
        </label>
        <button className="primario grande" type="submit" disabled={enviando} style={{ width: "100%", marginTop: 6 }}>
          {enviando ? "Entrando…" : "Entrar"}
        </button>
      </form>
    </div>
  );
}
