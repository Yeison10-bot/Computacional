import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";

// Cámara del computador abierta desde el navegador para el monitoreo continuo.
// Envía unos 5 fotogramas por segundo al backend (POST /api/monitor/fotograma),
// que los analiza con el mismo algoritmo que una cámara conectada al servidor.
// Vive a nivel de toda la aplicación: cambiar de página no corta el envío.

const FPS_ENVIO = 5;
const ANCHO_MAXIMO = 1280;
const Contexto = createContext(null);
const esperar = (ms) => new Promise((r) => setTimeout(r, ms));

export function ProveedorCamara({ children }) {
  const [abierta, setAbierta] = useState(false);
  const [enviados, setEnviados] = useState(0);
  const [error, setError] = useState("");
  const [dispositivos, setDispositivos] = useState([]);
  // Respuesta del servidor a cada envío: detecciones del último fotograma analizado,
  // que la página dibuja sobre el video en vivo.
  const [analisis, setAnalisis] = useState({ detecciones: [], fase: null, analizados: 0 });
  const [flujoVisible, setFlujoVisible] = useState(null); // MediaStream, para mostrarlo en la página
  const flujo = useRef(null);
  const video = useRef(null);
  const enviando = useRef(false);

  const listar = useCallback(async () => {
    const todos = (await navigator.mediaDevices?.enumerateDevices()) ?? [];
    setDispositivos(todos.filter((d) => d.kind === "videoinput"));
  }, []);

  useEffect(() => {
    listar();
  }, [listar]);

  const cerrar = useCallback(() => {
    enviando.current = false;
    flujo.current?.getTracks().forEach((t) => t.stop());
    flujo.current = null;
    video.current?.remove();
    video.current = null;
    setFlujoVisible(null);
    setAbierta(false);
    setAnalisis({ detecciones: [], fase: null, analizados: 0 });
  }, []);

  // Pide permiso y abre la cámara. Devuelve false (con el error en `error`) si no se pudo.
  const abrir = useCallback(async (dispositivoId) => {
    setError("");
    if (!navigator.mediaDevices) {
      setError("Este navegador no permite usar la cámara (se necesita http://localhost o https).");
      return false;
    }
    try {
      cerrar();
      flujo.current = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, ...(dispositivoId && { deviceId: { exact: dispositivoId } }) },
      });
      // Video de captura: va dentro del documento (invisible), porque algunos
      // navegadores no actualizan los fotogramas de un <video> suelto y la captura sale negra.
      const v = document.createElement("video");
      v.muted = true;
      v.playsInline = true;
      v.style.cssText = "position:fixed;left:0;top:0;width:2px;height:2px;opacity:0;pointer-events:none";
      document.body.appendChild(v);
      v.srcObject = flujo.current;
      await v.play();
      video.current = v;
      setFlujoVisible(flujo.current);
      setAbierta(true);
      listar(); // con el permiso concedido ya aparecen los nombres de las cámaras
      return true;
    } catch (e) {
      setError(e.name === "NotAllowedError" ? "Permiso de cámara denegado en el navegador."
        : e.name === "NotReadableError" ? "La cámara está siendo usada por otro programa (¿el monitoreo del servidor?)."
        : `No se pudo abrir la cámara: ${e.message}`);
      return false;
    }
  }, [cerrar, listar]);

  // Bucle de envío: captura el fotograma actual, lo comprime a JPEG y lo envía.
  // Se espera la respuesta antes de enviar el siguiente, así nunca se acumulan.
  const empezarEnvio = useCallback(async () => {
    if (enviando.current) return;
    enviando.current = true;
    setEnviados(0);
    const lienzo = document.createElement("canvas");
    while (enviando.current) {
      const inicio = performance.now();
      const v = video.current;
      if (v?.videoWidth) {
        const escala = Math.min(1, ANCHO_MAXIMO / v.videoWidth);
        lienzo.width = Math.round(v.videoWidth * escala);
        lienzo.height = Math.round(v.videoHeight * escala);
        lienzo.getContext("2d").drawImage(v, 0, 0, lienzo.width, lienzo.height);
        const jpeg = await new Promise((r) => lienzo.toBlob(r, "image/jpeg", 0.85));
        try {
          const r = await fetch("/api/monitor/fotograma", { method: "POST", headers: { "Content-Type": "image/jpeg" }, body: jpeg });
          if (r.status === 409) {
            cerrar(); // el monitoreo se detuvo en el servidor: se libera la cámara
            break;
          }
          if (r.ok) {
            setEnviados((n) => n + 1);
            setAnalisis(await r.json());
          }
        } catch {
          /* fallo de red momentáneo: se reintenta con el siguiente fotograma */
        }
      }
      await esperar(Math.max(0, 1000 / FPS_ENVIO - (performance.now() - inicio)));
    }
  }, [cerrar]);

  useEffect(() => cerrar, [cerrar]); // al cerrar la aplicación se apaga la cámara

  return (
    <Contexto.Provider value={{ abierta, enviados, error, dispositivos, analisis, flujo: flujoVisible, abrir, cerrar, empezarEnvio }}>
      {children}
    </Contexto.Provider>
  );
}

export const useCamaraNavegador = () => useContext(Contexto);
