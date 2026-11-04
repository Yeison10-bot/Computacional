import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// En desarrollo, las peticiones a /api y /archivos se reenvían a FastAPI (puerto 8000),
// así el navegador ve un solo origen y no hay problemas de CORS.
// API_URL permite apuntar a otro backend (p. ej. uno de pruebas) sin tocar este archivo.
const API = process.env.API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    // 5180: los puertos 5173/5174 los usa otro proyecto (OGRD) en este equipo.
    port: 5180,
    strictPort: true,
    // host: true → también escucha en la red local, para abrir la página desde el iPhone.
    host: true,
    proxy: {
      "/api": API,
      "/archivos": API,
    },
  },
});
