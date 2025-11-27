# syntax=docker/dockerfile:1
# Audio Enhancer - Multi-GPU Dockerfile (ROCm & CUDA)
# Build with: docker compose build --build-arg GPU_TYPE=rocm|cuda|cpu

ARG GPU_TYPE=cpu

# ============================================================================
# Base stage with common dependencies
# ============================================================================
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY audiosr_env/run_audiosr.py ./audiosr_env/

# ============================================================================
# ROCm stage (AMD GPUs)
# ============================================================================
FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0 AS rocm-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY audiosr_env/run_audiosr.py ./audiosr_env/

# Create venv and install dependencies with ROCm PyTorch
RUN uv venv && \
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2 && \
    uv pip install -e .

# Setup AudioSR venv
RUN cd audiosr_env && \
    uv venv --python 3.10 && \
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2 && \
    uv pip install audiosr matplotlib setuptools

# ROCm environment variables
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# ============================================================================
# CUDA stage (NVIDIA GPUs)
# ============================================================================
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04 AS cuda-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY audiosr_env/run_audiosr.py ./audiosr_env/

# Create venv and install dependencies with CUDA PyTorch
RUN uv venv && \
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    uv pip install -e .

# Setup AudioSR venv
RUN cd audiosr_env && \
    uv venv --python 3.10 && \
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    uv pip install audiosr matplotlib setuptools

# ============================================================================
# CPU-only stage
# ============================================================================
FROM base AS cpu-base

# Create venv and install dependencies with CPU PyTorch
RUN uv venv && \
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install -e .

# Setup AudioSR venv
RUN cd audiosr_env && \
    uv venv --python 3.10 && \
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install audiosr matplotlib setuptools

# ============================================================================
# Final stage selection based on GPU_TYPE
# ============================================================================
FROM ${GPU_TYPE}-base AS final

WORKDIR /app

# Create input/output directories
RUN mkdir -p /input /output

# Default entrypoint
ENTRYPOINT ["uv", "run", "audio-enhancer"]
CMD ["--help"]
