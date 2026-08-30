param(
    [switch]$TrainModel
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$lockFile = Join-Path $projectRoot "requirements.lock"
$env:UV_CACHE_DIR = Join-Path $projectRoot ".uv-cache"

Set-Location -LiteralPath $projectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install it from https://docs.astral.sh/uv/ and rerun this script."
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    uv venv .venv --python 3.13
}

$requirements = if (Test-Path -LiteralPath $lockFile) {
    $lockFile
} else {
    Join-Path $projectRoot "requirements.txt"
}

uv pip sync --python $venvPython $requirements
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed with exit code $LASTEXITCODE."
}

$checkpoint = Join-Path $projectRoot "models\notebook_dual_branch.joblib"
if ($TrainModel -and -not (Test-Path -LiteralPath $checkpoint)) {
    & (Join-Path $projectRoot "scripts\train_demo_model.ps1")
}

Write-Host "UI environment is ready."
Write-Host "Launch with: .\run_ui.cmd"
