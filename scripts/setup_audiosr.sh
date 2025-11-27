#!/bin/bash
# Setup AudioSR environment (separate venv due to numpy version conflicts)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
AUDIOSR_DIR="$PROJECT_DIR/audiosr_env"

echo "Setting up AudioSR environment..."

# Create venv with Python 3.10
cd "$AUDIOSR_DIR"
uv venv --python 3.10

# Detect GPU type and install appropriate PyTorch
if command -v rocminfo &> /dev/null; then
    echo "Detected AMD ROCm, installing PyTorch for ROCm..."
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
elif command -v nvidia-smi &> /dev/null; then
    echo "Detected NVIDIA GPU, installing PyTorch for CUDA..."
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No GPU detected, installing CPU-only PyTorch..."
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install AudioSR and dependencies
echo "Installing AudioSR..."
uv pip install audiosr matplotlib setuptools

echo "AudioSR setup complete!"
echo "You can now run: uv run audio-enhancer input.opus -o output.flac"
