$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot "venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found at venv\. Run .\setup_ui.cmd, or create it manually with 'python -m venv venv' and 'pip install -r requirements.txt'."
}

Set-Location -LiteralPath $projectRoot
& $python "app.py"

