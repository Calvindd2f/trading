# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install base requirements
pip install -r requirements.txt

# Check for NVIDIA GPU
$gpu = Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like "*NVIDIA*"}
if ($gpu) {
    Write-Host "NVIDIA GPU detected. Installing GPU dependencies..."
    pip install -r requirements-gpu.txt
    pip install ".[gpu]"
} else {
    Write-Host "No NVIDIA GPU detected. Skipping GPU dependencies."
}

# Install anomaly detection dependencies
Write-Host "Installing anomaly detection dependencies..."
pip install ".[anomaly]"

# Install development dependencies (optional)
if ($args[0] -eq "--dev") {
    Write-Host "Installing development dependencies..."
    pip install ".[dev]"
}

# Install the package in development mode
pip install -e .

# Create necessary directories
New-Item -ItemType Directory -Force -Path "results\training"
New-Item -ItemType Directory -Force -Path "results\models"
New-Item -ItemType Directory -Force -Path "results\anomaly"
New-Item -ItemType Directory -Force -Path "logs"

# Check for TA-Lib
try {
    python -c "import talib" 2>$null
} catch {
    Write-Host "TA-Lib not found. Please install it manually:"
    Write-Host "Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib"
    Write-Host "Then run: pip install TA_Lib‑0.4.24‑cp38‑cp38‑win_amd64.whl"
}

Write-Host "Setup complete! To activate the virtual environment, run:"
Write-Host ".\venv\Scripts\activate" 