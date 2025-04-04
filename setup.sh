#!/bin/bash

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install base requirements
pip install -r requirements.txt

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing GPU dependencies..."
    pip install -r requirements-gpu.txt
    pip install ".[gpu]"
else
    echo "No NVIDIA GPU detected. Skipping GPU dependencies."
fi

# Install anomaly detection dependencies
echo "Installing anomaly detection dependencies..."
pip install ".[anomaly]"

# Install development dependencies (optional)
if [ "$1" == "--dev" ]; then
    echo "Installing development dependencies..."
    pip install ".[dev]"
fi

# Install the package in development mode
pip install -e .

# Create necessary directories
mkdir -p results/training
mkdir -p results/models
mkdir -p results/anomaly
mkdir -p logs

# Check for TA-Lib
if ! python -c "import talib" &> /dev/null; then
    echo "TA-Lib not found. Please install it manually:"
    echo "For Ubuntu/Debian: sudo apt-get install ta-lib"
    echo "For macOS: brew install ta-lib"
    echo "For Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib"
fi

echo "Setup complete! To activate the virtual environment, run:"
echo "source venv/bin/activate" 