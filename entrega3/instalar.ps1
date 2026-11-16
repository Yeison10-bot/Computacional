# Instala la aplicación en este equipo (se corre una sola vez, pero se puede repetir):
# crea el .env, el entorno de Python con PyTorch, la base de datos (con la copia de datos
# incluida en el paquete) y las dependencias del frontend.
#
# Uso, desde la carpeta descomprimida:
#     powershell -ExecutionPolicy Bypass -File .\instalar.ps1

$ErrorActionPreference = 'Stop'
$raiz = $PSScriptRoot
Set-Location $raiz

function Fallo($mensaje) {
    Write-Host $mensaje -ForegroundColor Red
    exit 1
}

# 1. Requisitos.
Write-Host "1/5 Verificando requisitos..."
$faltan = @()
$hayPy = $false
try { $null = & py -3.11 --version 2>$null; $hayPy = ($LASTEXITCODE -eq 0) } catch { }
if (-not $hayPy) { $faltan += "Python 3.11 (con la casilla 'Add python.exe to PATH')" }

if (Get-Command node -ErrorAction SilentlyContinue) {
    $v = (node --version).TrimStart('v')
    if ([int]($v.Split('.')[0]) -lt 20) { $faltan += "Node.js 20 o superior (tiene $v; instale la versión 22 LTS)" }
} else {
    $faltan += "Node.js 22 LTS"
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { $faltan += "npm (viene con Node.js)" }

if ($faltan.Count -gt 0) {
    Write-Host "Falta instalar:" -ForegroundColor Red
    $faltan | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Fallo "Siga INSTALAR.md (enlaces de descarga) y vuelva a ejecutar este script."
}

# 2. Archivo .env.
Write-Host "2/5 Configuración (.env)..."
if (Test-Path "$raiz\.env") {
    Write-Host "  .env ya existe, no se modifica."
} else {
    if (-not (Test-Path "$raiz\.env.example")) { Fallo "Falta .env.example." }
    $segura = Read-Host "Contraseña del usuario postgres de PostgreSQL" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($segura)
    try { $clave = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
    if (-not $clave) { Fallo "La contraseña no puede estar vacía." }
    $lineas = Get-Content "$raiz\.env.example" -Encoding UTF8 | ForEach-Object {
        if ($_ -match '^DB_PASSWORD=') { "DB_PASSWORD=$clave" } else { $_ }
    }
    [IO.File]::WriteAllLines("$raiz\.env", [string[]]$lineas, (New-Object Text.UTF8Encoding($false)))
    $clave = $null
    Write-Host "  .env creado."
}

# 3. Entorno de Python.
Write-Host "3/5 Entorno de Python (puede tardar varios minutos)..."
$py = "$raiz\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    py -3.11 -m venv "$raiz\.venv"
    if ($LASTEXITCODE -ne 0) { Fallo "No se pudo crear el entorno .venv." }
}

$bloqueo = "$raiz\requirements-lock.txt"
if (-not (Test-Path $bloqueo)) { Fallo "Falta requirements-lock.txt." }
$todas = Get-Content $bloqueo -Encoding UTF8 | ForEach-Object { $_.TrimStart([char]0xFEFF).Trim() } | Where-Object { $_ }
$torch = @($todas | Where-Object { $_ -match '^(torch|torchvision)==' } | ForEach-Object { $_ -replace '\+.*$', '' })
$resto = @($todas | Where-Object { $_ -notmatch '^(torch|torchvision)==' })
if ($torch.Count -ne 2) { Fallo "No se encontraron torch y torchvision en requirements-lock.txt." }

$gpu = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try { $null = & nvidia-smi 2>$null; if ($LASTEXITCODE -eq 0) { $gpu = $true } } catch { }
}
if ($gpu) {
    $indice = 'https://download.pytorch.org/whl/cu126'
    Write-Host "  GPU NVIDIA detectada: PyTorch con CUDA."
} else {
    $indice = 'https://download.pytorch.org/whl/cpu'
    Write-Host "  Sin GPU NVIDIA: PyTorch para CPU."
}

& $py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Fallo "No se pudo actualizar pip." }
& $py -m pip install @torch --index-url $indice
if ($LASTEXITCODE -ne 0) { Fallo "Falló la instalación de PyTorch. Revise la conexión a internet." }

$temporal = Join-Path $env:TEMP 'requirements-sin-torch.txt'
[IO.File]::WriteAllLines($temporal, [string[]]$resto, (New-Object Text.UTF8Encoding($false)))
& $py -m pip install -r $temporal
$codigo = $LASTEXITCODE
Remove-Item $temporal -ErrorAction SilentlyContinue
if ($codigo -ne 0) { Fallo "Falló la instalación de las dependencias de Python." }

# 4. Base de datos.
Write-Host "4/5 Base de datos..."
& $py "$raiz\database\crear_bd.py"
if ($LASTEXITCODE -ne 0) {
    Fallo "No se pudo crear la base de datos. Revise que PostgreSQL esté en marcha y la contraseña en .env."
}
if (Test-Path "$raiz\database\copia_datos.json") {
    & $py "$raiz\database\copia_datos.py" importar "$raiz\database\copia_datos.json"
    if ($LASTEXITCODE -eq 2) {
        Write-Host "  La copia ya estaba cargada (la base ya tenía datos, no se tocó)."
    } elseif ($LASTEXITCODE -ne 0) {
        Fallo "Falló la importación de la copia de datos (código $LASTEXITCODE)."
    } else {
        Write-Host "  Copia de datos importada."
    }
}

# 5. Frontend.
Write-Host "5/5 Dependencias del frontend..."
Push-Location "$raiz\frontend"
if (Test-Path "package-lock.json") { npm ci } else { npm install }
$codigo = $LASTEXITCODE
Pop-Location
if ($codigo -ne 0) { Fallo "Falló la instalación de las dependencias del frontend." }

Write-Host ""
Write-Host "Instalación terminada." -ForegroundColor Green
Write-Host "Para abrir la aplicación:  powershell -ExecutionPolicy Bypass -File .\iniciar.ps1"
Write-Host "Se abre en http://localhost:5180"
