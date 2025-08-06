#!/bin/bash

# Quantum Portfolio Optimization Environment Setup
# This script creates a conda environment with all necessary dependencies

echo "=== Quantum Portfolio Optimization Environment Setup ==="
echo "Setting up conda environment for quantum portfolio optimization..."

# Environment name
ENV_NAME="quantum_portfolio"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Remove existing environment if it exists
echo "Checking for existing environment..."
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "Removing existing environment: ${ENV_NAME}"
    conda env remove -n ${ENV_NAME} -y
fi

# Create new conda environment with Python 3.10 (stable for quantum computing)
echo "Creating new conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=3.10 -y

# Activate the environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Install conda packages first (faster and more stable)
echo "Installing base packages via conda..."
conda install -c conda-forge numpy pandas scipy matplotlib seaborn plotly jupyter jupyterlab psutil openpyxl xlrd -y

# Install PennyLane and quantum computing packages via pip
echo "Installing quantum computing packages..."
pip install pennylane pennylane-lightning

# Install PyTorch (let conda handle CUDA detection)
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio -c pytorch -y

# Install additional scientific packages
echo "Installing additional scientific packages..."
pip install scikit-learn sympy networkx

# Install testing framework
pip install pytest

# Optional: Install GPU support if NVIDIA GPU is detected
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Would you like to install GPU support? (y/n)"
    read -r gpu_support
    if [[ $gpu_support =~ ^[Yy]$ ]]; then
        echo "Installing GPU support packages..."
        # Detect CUDA version
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
        echo "Detected CUDA version: $CUDA_VERSION"
        
        if [[ $CUDA_VERSION == "11."* ]]; then
            pip install cupy-cuda11x
        elif [[ $CUDA_VERSION == "12."* ]]; then
            pip install cupy-cuda12x
        else
            echo "Unsupported CUDA version. Skipping GPU packages."
        fi
        
        # Try to install GPU-enabled PennyLane plugins
        pip install pennylane-lightning[gpu] || echo "GPU lightning plugin failed to install"
    fi
else
    echo "No NVIDIA GPU detected. Skipping GPU packages."
fi

# Verify installation
echo "Verifying installation..."
python -c "
import pennylane as qml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
print('✓ All core packages imported successfully')

# Test PennyLane devices
devices = ['default.qubit', 'lightning.qubit']
for device_name in devices:
    try:
        dev = qml.device(device_name, wires=2)
        print(f'✓ {device_name} device available')
    except Exception as e:
        print(f'✗ {device_name} device failed: {e}')

# Check if GPU devices are available
try:
    dev_gpu = qml.device('lightning.gpu', wires=2)
    print('✓ GPU acceleration available')
except:
    print('ℹ GPU acceleration not available (CPU-only)')

print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
print(f'PyTorch version: {torch.__version__}')
print(f'PennyLane version: {qml.__version__}')
"

echo ""
echo "=== Environment Setup Complete ==="
echo "Environment name: ${ENV_NAME}"
echo ""
echo "To activate this environment in the future, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To deactivate the environment, run:"
echo "  conda deactivate"
echo ""
echo "To remove this environment, run:"
echo "  conda env remove -n ${ENV_NAME}"
echo ""
echo "Your quantum portfolio optimization environment is ready!"
