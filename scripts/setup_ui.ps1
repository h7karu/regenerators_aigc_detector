param(
    [switch]$TrainModel
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$venvPython = Join-Path $projectRoot "venv\Scripts\python.exe"
$lockFile = Join-Path $projectRoot "requirements.lock"

Set-Location -LiteralPath $projectRoot

# uv is faster and installs the exact pinned set, but it is optional -- the rest
# of the repo sets up with stock venv + pip, so fall back to that when uv is absent.
$useUv = [bool](Get-Command uv -ErrorAction SilentlyContinue)

if ($useUv) {
    $env:UV_CACHE_DIR = Join-Path $projectRoot ".uv-cache"
    if (-not (Test-Path -LiteralPath $venvPython)) {
        uv venv venv --python 3.13
        if ($LASTEXITCODE -ne 0) { throw "Failed to create venv with uv (exit $LASTEXITCODE)." }
    }
    $requirements = if (Test-Path -LiteralPath $lockFile) { $lockFile } else { Join-Path $projectRoot "requirements.txt" }
    uv pip sync --python $venvPython $requirements
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed with exit code $LASTEXITCODE." }
} else {
    Write-Host "uv not found - using python -m venv and pip instead."
    if (-not (Test-Path -LiteralPath $venvPython)) {
        python -m venv venv
        if ($LASTEXITCODE -ne 0) { throw "Failed to create venv (exit $LASTEXITCODE). Is Python on PATH?" }
    }
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip (exit $LASTEXITCODE)." }
    & $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed with exit code $LASTEXITCODE." }
}

$checkpoint = Join-Path $projectRoot "models\notebook_dual_branch.joblib"
if ($TrainModel -and -not (Test-Path -LiteralPath $checkpoint)) {
    & (Join-Path $projectRoot "scripts\train_demo_model.ps1")
}

Write-Host "UI environment is ready."
Write-Host "Launch with: .\run_ui.cmd"
