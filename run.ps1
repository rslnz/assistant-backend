$VENV_PATH = ".\venv"

if (-not (Test-Path $VENV_PATH)) {
    Write-Host "Virtual environment not found. Creating a new one..."
    python -m venv $VENV_PATH
}

$ActivateScript = Join-Path $VENV_PATH "Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    & $ActivateScript
} else {
    Write-Host "Error: Failed to find activation script"
    exit 1
}

python -m pip install --upgrade pip
pip install -r requirements.txt --upgrade

$env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)"

python src/main.py

deactivate