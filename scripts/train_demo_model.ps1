param(
    [int]$MaxPerClass = 150,
    [int]$AugmentCopies = 2,
    [int]$Seed = 42,
    [string]$Output = "models\notebook_dual_branch.joblib"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = Join-Path $projectRoot "venv\Scripts\python.exe"
$dataDir = Join-Path $projectRoot "data\cifake\train"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found. Run scripts\setup_ui.ps1 first."
}
if (-not (Test-Path -LiteralPath $dataDir)) {
    throw "CIFAKE training data not found at $dataDir. Download it before training."
}

Set-Location -LiteralPath $projectRoot
& $python -m aigc_detector.train `
    --data-dir $dataDir `
    --max-per-class $MaxPerClass `
    --augment-copies $AugmentCopies `
    --seed $Seed `
    --output $Output

if ($LASTEXITCODE -ne 0) {
    throw "Demo model training failed with exit code $LASTEXITCODE."
}

