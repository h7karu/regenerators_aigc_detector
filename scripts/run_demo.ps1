$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$pythonExecutable = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw "Missing project environment. Run scripts\setup_windows.ps1 first."
}

$env:PYTHONNOUSERSITE = "1"
Set-Location -LiteralPath $projectRoot
& $pythonExecutable -u demo.py @args
