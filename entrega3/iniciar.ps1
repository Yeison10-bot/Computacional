# Inicia el backend (FastAPI, puerto 8000) y el frontend (Vite, puerto 5180),
# cada uno en su propia ventana de PowerShell, y abre la aplicación en el navegador.
#
# Uso, desde la raíz del proyecto:
#     powershell -ExecutionPolicy Bypass -File .\iniciar.ps1

$raiz = $PSScriptRoot

if (-not (Test-Path "$raiz\.env")) {
    Write-Host "Falta el archivo .env: copia .env.example como .env y escribe la contraseña de PostgreSQL." -ForegroundColor Red
    exit 1
}

Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "cd '$raiz'; .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --reload-dir backend --reload-dir vision --timeout-graceful-shutdown 3"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$raiz\frontend'; npm run dev"

Start-Sleep -Seconds 4
Start-Process "http://localhost:5180"
Write-Host "Backend:  http://localhost:8000/docs"
Write-Host "Frontend: http://localhost:5180"
