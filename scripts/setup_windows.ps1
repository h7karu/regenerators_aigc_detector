param(
    [switch]$SkipVerification
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$environmentRoot = Join-Path $projectRoot ".venv"
$pythonExecutable = Join-Path $environmentRoot "Scripts\python.exe"
$requirementsFile = Join-Path $projectRoot "requirements.txt"

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    throw "uv is required. Install it with: winget install --id=astral-sh.uv -e"
}

Set-Location -LiteralPath $projectRoot

& uv python install 3.12.14
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install Python 3.12.14 through uv."
}

if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    & uv venv $environmentRoot --python 3.12.14 --seed
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create $environmentRoot."
    }
}

$pythonVersion = & $pythonExecutable -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
if ($pythonVersion -ne "3.12.14") {
    throw "$environmentRoot uses Python $pythonVersion, expected 3.12.14. Remove that disposable environment and rerun this script."
}

& uv pip install --python $pythonExecutable `
    --index-url https://download.pytorch.org/whl/cpu `
    torch==2.13.0+cpu torchvision==0.28.0+cpu
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch installation failed."
}

& uv pip install --python $pythonExecutable -r $requirementsFile
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

& $pythonExecutable -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "Installed packages have incompatible dependencies."
}

if (-not $SkipVerification) {
    & $pythonExecutable scripts\verify_environment.py
    if ($LASTEXITCODE -ne 0) {
        throw "Environment verification failed."
    }
}

Write-Host "Setup complete. Activate with: .\.venv\Scripts\Activate.ps1"
