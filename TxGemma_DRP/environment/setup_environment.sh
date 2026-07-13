#!/bin/bash
#
# Environment Setup Script for TxGemma Drug Response Prediction
#
# This script creates a conda environment with all necessary dependencies
#

set -e  # Exit on error

echo "====================================="
echo "TxGemma DRP Environment Setup"
echo "====================================="

# Configuration
ENV_NAME="txgemma_drp"
PYTHON_VERSION="3.9"
CUDA_VERSION="11.8"  # Adjust based on your system

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create environment
echo ""
echo "Step 1: Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
echo ""
echo "Step 2: Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Install CUDA toolkit
echo ""
echo "Step 3: Installing CUDA toolkit ${CUDA_VERSION}..."
conda install -c "nvidia/label/cuda-${CUDA_VERSION}" cuda-toolkit -y

# Verify CUDA installation
echo ""
echo "Verifying CUDA installation..."
nvcc --version || echo "Warning: nvcc not found in PATH"

# Install PyTorch
echo ""
echo "Step 4: Installing PyTorch for CUDA ${CUDA_VERSION}..."
if [[ "$CUDA_VERSION" == "11.8" ]]; then
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_VERSION" == "12.1" ]]; then
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch with default CUDA support..."
    pip3 install torch torchvision torchaudio
fi

# Install requirements
echo ""
echo "Step 5: Installing Python packages from requirements.txt..."
pip install -r ../requirements.txt

# Verify ai4clinic installation
echo ""
echo "Verifying ai4clinic installation..."
python -c "import ai4clinic; print(f'ai4clinic installed successfully')" || echo "Warning: ai4clinic not found"

# Verify installation
echo ""
echo "====================================="
echo "Verifying Installation"
echo "====================================="

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" || echo "CUDA not available"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import peft; print(f'PEFT version: {peft.__version__}')"
python -c "import trl; print(f'TRL version: {trl.__version__}')"
python -c "import ai4clinic; print(f'ai4clinic: Ready for drug response analysis')"

echo ""
echo "====================================="
echo "Setup Complete!"
echo "====================================="
echo ""
echo "To activate this environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To deactivate, run:"
echo "  conda deactivate"
echo ""
