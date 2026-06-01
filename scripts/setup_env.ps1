# scripts/setup_env.ps1
$ErrorActionPreference = 'Stop'

function Run-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string]$Message
    )
    Write-Host $Message
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Command"
    }
}

# Move to project root (script is under scripts/)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Join-Path $scriptDir '..'
Set-Location $projectRoot

$venvPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$venvActivate = Join-Path (Get-Location) '.venv\Scripts\Activate.ps1'

Write-Host "Project root: $(Get-Location)"

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host 'No .venv detected. Creating virtual environment...'
    python -m venv .venv
}

if (-not (Test-Path -LiteralPath $venvActivate)) {
    throw "Cannot find activation script: $venvActivate"
}

Write-Host 'Activating virtual environment...'
. $venvActivate

Write-Host 'Installing dependencies from requirements.txt...'
Run-Step -Command "python -m pip install --upgrade pip" -Message "Upgrading pip..."
Run-Step -Command "python -m pip install -r requirements.txt" -Message "Installing requirements..."

Run-Step -Command "python database/create_database.py" -Message "Initializing database (MySQL or SQLite fallback)..."

Run-Step -Command "python database/import_data.py --input data/sample_arxiv_20000.jsonl --limit 20000 --batch 500 --to auto" -Message "Importing dataset (sample_arxiv_20000.jsonl) into database (auto: prefer MySQL, fallback SQLite)..."

Write-Host 'Done.'
Write-Host 'Start app with: streamlit run app_streamlit.py --server.port 8502'
