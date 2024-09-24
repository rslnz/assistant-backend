$ErrorActionPreference = "Stop"

$VENV_PATH = ".\venv"
$REQ_FILE = "requirements.txt"

if (-not (Test-Path $VENV_PATH)) {
    Write-Host "Virtual environment not found. Creating a new one..."
    python -m venv $VENV_PATH
}

$ActivateScript = Join-Path $VENV_PATH "Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    try {
        & $ActivateScript
    } catch {
        Write-Host "Error: Failed to activate virtual environment"
        Write-Host "Please check if the virtual environment is correctly set up"
        return
    }
} else {
    Write-Host "Error: Activation script not found"
    Write-Host "Please check if the virtual environment is correctly set up"
    return
}

$RequirementsInstalled = Join-Path $VENV_PATH ".requirements_installed"
if (-not (Test-Path $RequirementsInstalled) -or (Get-Item $REQ_FILE).LastWriteTime -gt (Get-Item $RequirementsInstalled).LastWriteTime) {
    Write-Host "Installing/Updating packages..."
    pip install -r $REQ_FILE
    New-Item -ItemType File -Path $RequirementsInstalled -Force | Out-Null
} else {
    Write-Host "Requirements are up to date."
}

$env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)"

python src/main.py
$EXIT_CODE = $LASTEXITCODE

if ($EXIT_CODE -ne 0) {
    Write-Host "Python script exited with code $EXIT_CODE"
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

deactivate