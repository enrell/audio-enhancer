"""GPU detection and configuration for ROCm/CUDA/CPU."""

import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class GPUBackend(Enum):
    """Available GPU backends."""

    ROCM = "rocm"
    CUDA = "cuda"
    CPU = "cpu"


@dataclass
class GPUInfo:
    """Information about detected GPU."""

    backend: GPUBackend
    device_name: str
    device_index: int
    vram_gb: float
    is_available: bool


def detect_rocm() -> Optional[GPUInfo]:
    """Detect AMD ROCm GPU."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if "GPU" in line or "gfx" in line.lower():
                    # Get VRAM info
                    vram_result = subprocess.run(
                        ["rocm-smi", "--showmeminfo", "vram"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    vram_gb = 8.0
                    if vram_result.returncode == 0:
                        for vram_line in vram_result.stdout.split("\n"):
                            if "Total" in vram_line:
                                try:
                                    parts = vram_line.split()
                                    for i, p in enumerate(parts):
                                        if p.isdigit():
                                            val = int(p)
                                            if val > 1_000_000_000:
                                                vram_gb = val / 1_000_000_000
                                            elif val > 1_000_000:
                                                vram_gb = val / 1_000
                                            break
                                except (ValueError, IndexError):
                                    pass

                    return GPUInfo(
                        backend=GPUBackend.ROCM,
                        device_name=line.strip(),
                        device_index=0,
                        vram_gb=vram_gb,
                        is_available=True,
                    )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def detect_cuda() -> Optional[GPUInfo]:
    """Detect NVIDIA CUDA GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            return GPUInfo(
                backend=GPUBackend.CUDA,
                device_name=device_name,
                device_index=0,
                vram_gb=vram_bytes / (1024**3),
                is_available=True,
            )
    except Exception:
        pass
    return None


def detect_gpu() -> GPUInfo:
    """Detect best available GPU, falling back to CPU."""
    # Try ROCm first (AMD)
    rocm_gpu = detect_rocm()
    if rocm_gpu:
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")  # RX 6000
        os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
        return rocm_gpu

    cuda_gpu = detect_cuda()
    if cuda_gpu:
        return cuda_gpu

    # Fallback to CPU
    return GPUInfo(
        backend=GPUBackend.CPU,
        device_name="CPU",
        device_index=-1,
        vram_gb=0.0,
        is_available=True,
    )


def get_torch_device(gpu_info: GPUInfo) -> str:
    """Get PyTorch device string from GPU info."""
    if gpu_info.backend == GPUBackend.CUDA:
        return f"cuda:{gpu_info.device_index}"
    elif gpu_info.backend == GPUBackend.ROCM:
        # ROCm uses the same 'cuda' device in PyTorch
        return f"cuda:{gpu_info.device_index}"
    return "cpu"


def configure_environment(gpu_info: GPUInfo) -> None:
    """Configure environment variables for optimal GPU usage."""
    if gpu_info.backend == GPUBackend.ROCM:
        # AMD ROCm specific settings
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_info.device_index)
    elif gpu_info.backend == GPUBackend.CUDA:
        # NVIDIA CUDA specific settings
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_info.device_index)
