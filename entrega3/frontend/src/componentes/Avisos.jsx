import { createContext, useCallback, useContext, useState } from "react";
import { IconoAlerta, IconoCheck } from "./Iconos.jsx";

// Notificaciones flotantes ("toasts") que desaparecen solas.
// Uso: const avisar = useAvisos(); avisar("Guardado"); avisar("Falló", "error");

const Contexto = createContext(() => {});

export function ProveedorAvisos({ children }) {
  const [avisos, setAvisos] = useState([]);

  const avisar = useCallback((texto, tipo = "ok") => {
    const id = crypto.randomUUID();
    setAvisos((a) => [...a, { id, texto, tipo }]);
    setTimeout(() => setAvisos((a) => a.filter((x) => x.id !== id)), tipo === "error" ? 7000 : 4000);
  }, []);

  return (
    <Contexto.Provider value={avisar}>
      {children}
      <div className="avisos" role="status" aria-live="polite">
        {avisos.map((a) => (
          <div key={a.id} className={`aviso ${a.tipo}`}>
            {a.tipo === "error" ? <IconoAlerta /> : <IconoCheck />}
            <span>{a.texto}</span>
          </div>
        ))}
      </div>
    </Contexto.Provider>
  );
}

export const useAvisos = () => useContext(Contexto);
