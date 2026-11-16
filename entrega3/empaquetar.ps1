# Genera el paquete .zip para el profesor (en el Escritorio): el código sale de git (HEAD),
# más el modelo entrenado, las imágenes, las evidencias, el último entrenamiento y una
# copia de los datos de la base de datos.
#
# Uso, desde la carpeta entrega3 (con todo confirmado en git):
#     powershell -ExecutionPolicy Bypass -File .\empaquetar.ps1

$ErrorActionPreference = 'Stop'
$raiz = $PSScriptRoot
# tar de Windows (bsdtar); el tar de Git no entiende rutas con 'C:'
$tar = "$env:SystemRoot\System32\tar.exe"
if (-not (Test-Path $tar)) { $tar = 'tar' }

function Fallo($mensaje) {
    Write-Host $mensaje -ForegroundColor Red
    exit 1
}

function Copiar-Carpeta($origen, $destino) {
    New-Item -ItemType Directory -Force -Path $destino | Out-Null
    Copy-Item -Path (Join-Path $origen '*') -Destination $destino -Recurse -Force
}

Set-Location $raiz
$repo = (git rev-parse --show-toplevel)
if ($LASTEXITCODE -ne 0) { Fallo "No se pudo ubicar el repositorio git." }
$repo = $repo.Trim() -replace '/', '\'

# 1. El código sale de git: no debe haber cambios sin confirmar.
$pendientes = git status --porcelain --untracked-files=no -- .
if ($LASTEXITCODE -ne 0) { Fallo "git status falló." }
if ($pendientes) {
    Write-Host "Hay archivos de entrega3 modificados sin commit. El paquete se arma desde git, haz commit primero:" -ForegroundColor Red
    $pendientes | ForEach-Object { Write-Host "  $_" }
    exit 1
}

# 2. Carpeta de trabajo temporal limpia.
$tmp = Join-Path $env:TEMP 'empaquetar_entrega3'
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$dest = Join-Path $tmp 'entrega3'
New-Item -ItemType Directory -Force -Path $dest | Out-Null

# 3. Código versionado.
Write-Host "Extrayendo el código desde git..."
Push-Location $repo
git archive --format=tar -o "$tmp\codigo.tar" HEAD:entrega3
$codigo = $LASTEXITCODE
Pop-Location
if ($codigo -ne 0) { Fallo "git archive falló." }
& $tar -xf "$tmp\codigo.tar" -C $dest
if ($LASTEXITCODE -ne 0) { Fallo "No se pudo extraer codigo.tar." }
Remove-Item "$tmp\codigo.tar"

# 4. Modelos.
Write-Host "Copiando modelos..."
if (-not (Test-Path "$raiz\vision\modelos\best.pt")) {
    Fallo "Falta vision\modelos\best.pt (el modelo entrenado). No se puede empaquetar."
}
New-Item -ItemType Directory -Force -Path "$dest\vision\modelos" | Out-Null
foreach ($f in 'best.pt', 'best.json', 'yolo11s.pt') {
    if (Test-Path "$raiz\vision\modelos\$f") {
        Copy-Item "$raiz\vision\modelos\$f" "$dest\vision\modelos\$f" -Force
    } else {
        Write-Host "  aviso: no existe vision\modelos\$f, se omite." -ForegroundColor Yellow
    }
}

# 5. Datos: imágenes, evidencias y SOLO el entrenamiento más reciente.
Write-Host "Copiando imágenes, evidencias y el último entrenamiento..."
if (Test-Path "$raiz\datos\imagenes") { Copiar-Carpeta "$raiz\datos\imagenes" "$dest\datos\imagenes" }
if (Test-Path "$raiz\datos\evidencias") { Copiar-Carpeta "$raiz\datos\evidencias" "$dest\datos\evidencias" }
$ultimo = $null
if (Test-Path "$raiz\datos\entrenamientos") {
    $ultimo = Get-ChildItem "$raiz\datos\entrenamientos" -Directory | Sort-Object Name | Select-Object -Last 1
}
if ($ultimo) {
    Copiar-Carpeta $ultimo.FullName "$dest\datos\entrenamientos\$($ultimo.Name)"
} else {
    Write-Host "  aviso: no hay entrenamientos en datos\entrenamientos." -ForegroundColor Yellow
}

# 6. Copia de los datos de la base de datos.
Write-Host "Exportando la base de datos..."
$py = "$raiz\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Fallo "No existe .venv\Scripts\python.exe." }
& $py "$raiz\database\copia_datos.py" exportar "$dest\database\copia_datos.json"
if ($LASTEXITCODE -ne 0) { Fallo "La exportación de la base de datos falló (código $LASTEXITCODE)." }

# 7. Nunca se incluyen secretos ni carpetas pesadas (por si algo se coló).
foreach ($r in '.env', '.venv', 'frontend\node_modules', 'frontend\dist', 'datos\dataset') {
    if (Test-Path "$dest\$r") { Remove-Item -Recurse -Force "$dest\$r" }
}
if (Test-Path "$dest\datos\crudos") {
    Get-ChildItem "$dest\datos\crudos" -Recurse -File | Where-Object { $_.Name -ne '.gitkeep' } | Remove-Item -Force
}

# 8. Zip en el Escritorio.
$escritorio = [Environment]::GetFolderPath('Desktop')
$zip = Join-Path $escritorio ("entrega3_profesor_{0}.zip" -f (Get-Date -Format 'yyyyMMdd'))
if (Test-Path $zip) { Remove-Item -Force $zip }
Write-Host "Comprimiendo..."
& $tar -a -c -f $zip -C $tmp entrega3
if ($LASTEXITCODE -ne 0) {
    Write-Host "tar falló, se usa Compress-Archive..." -ForegroundColor Yellow
    if (Test-Path $zip) { Remove-Item -Force $zip }
    Compress-Archive -Path $dest -DestinationPath $zip
}
if (-not (Test-Path $zip)) { Fallo "No se pudo crear el zip." }

$mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
$nImg = @(Get-ChildItem "$dest\datos\imagenes" -Recurse -File -ErrorAction SilentlyContinue).Count
Write-Host ""
Write-Host "Paquete listo: $zip ($mb MB)" -ForegroundColor Green
Write-Host "Contiene: código de git (HEAD), modelo best.pt, $nImg imágenes, evidencias,"
Write-Host "el entrenamiento $(if ($ultimo) { $ultimo.Name } else { '(ninguno)' }) y la copia de la base de datos."

Remove-Item -Recurse -Force $tmp
