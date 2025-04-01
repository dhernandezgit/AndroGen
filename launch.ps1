# Exit on any error
$ErrorActionPreference = "Stop"

# Step 1: Create virtual environment if it doesn't exist
if (-Not (Test-Path -Path ".venv")) {
    Write-Host "[1/4] Creating Python 3.10 virtual environment..."
    python -m venv .venv
} else {
    Write-Host "[1/4] Virtual environment already exists, skipping."
}

# Activate the virtual environment
. .\.venv\Scripts\Activate.ps1

Write-Host "[2/4] Installing Python dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 3: Download resources if folder doesn't exist or is empty
if (-Not (Test-Path -Path "resources") -or !(Get-ChildItem "resources" -Recurse)) {
    Write-Host "[3/4] Downloading resources folder from OneDrive..."
    New-Item -ItemType Directory -Force -Path "resources"
    
    # Download the file
    Invoke-WebRequest -Uri "https://urjc-my.sharepoint.com/:u:/g/personal/daniel_hernandezf_urjc_es/EVlXB2ilZDdEuZu1BzqVJ4sBeFIICT4IPsYs7VSbZ_ssdQ?e=WvBE59&download=1" -OutFile "$env:TEMP\resources.zip"
    
    # Extract the zip file
    Expand-Archive -Path "$env:TEMP\resources.zip" -DestinationPath "resources" -Force
    Remove-Item -Path "$env:TEMP\resources.zip" -Force

    # Move contents from the nested 'resources' folder
    Move-Item -Path "resources\resources\*" -Destination "resources"
    Remove-Item -Path "resources\resources" -Force
} else {
    Write-Host "[3/4] Resources folder already exists, skipping download."
}

# Step 4: Launch the application
Write-Host "[4/4] Launching the app..."
python src\frontend\App.py
