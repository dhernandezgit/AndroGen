#!/bin/bash
set -e  # Exit on any error

# Step 1: Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating Python 3.10 virtual environment..."
    python3 -m venv .venv
else
    echo "[1/4] Virtual environment already exists, skipping."
fi
source .venv/bin/activate

echo "[2/4] Installing Python dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -d "resources" ] || [ -z "$(ls -A resources)" ]; then
    echo "[3/4] Downloading resources folder from OneDrive..."
    
    mkdir -p resources

    # Replace with your direct download ZIP link
    ONEDRIVE_URL="https://urjc-my.sharepoint.com/:u:/g/personal/daniel_hernandezf_urjc_es/EVlXB2ilZDdEuZu1BzqVJ4sBeFIICT4IPsYs7VSbZ_ssdQ?e=WvBE59&download=1"

    wget -O /tmp/resources.zip "$ONEDRIVE_URL"
    unzip -q /tmp/resources.zip -d resources
    mv resources/resources/* resources/ && rmdir resources/resources
    rm /tmp/resources.zip
else
    echo "[3/4] Resources folder already exists, skipping download."
fi

# Step 4: Launch the application
echo "[4/4] Launching the app..."
python src/frontend/App.py