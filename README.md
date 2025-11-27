# Audio Enhancer

Recover audio quality lost from lossy compression. Transform low-bitrate audio (YouTube 127kbps opus/webm) back to high-quality FLAC/opus with restored high frequencies and improved depth.

## What It Does

```
Original FLAC (studio) → 127kbps opus (YouTube) → Audio Enhancer → High-quality FLAC
```

The pipeline uses **AudioSR** (neural super-resolution) to recover frequencies lost during compression, then applies professional mastering for clean output.

## Features

- **Super Resolution** - Neural network (AudioSR) recovers lost high frequencies from compression
- **Denoising** - Optional noise reduction for noisy recordings (noisereduce + GPU acceleration)
- **Harmonic Enhancement** - Optional harmonic exciter and stereo widening
- **Final Mastering** - Soft limiter, optional loudness normalization, dithering
- **AMD ROCm Support** - Full GPU acceleration on AMD RX 6000/7000 series

## Requirements

- Python 3.10-3.12
- FFmpeg (for audio extraction)
- AMD GPU with ROCm 6.2+ or NVIDIA GPU with CUDA
- Or Docker (for containerized usage)

## Installation

### Native Installation

```bash
# Clone repository with submodules
git clone --recurse-submodules https://github.com/enrell/audio-enhancer.git
cd audio-enhancer

# Install with uv (recommended)
uv sync

# Set up AudioSR environment (separate venv due to numpy conflicts)
./scripts/setup_audiosr.sh
```

### Docker Installation

```bash
git clone --recurse-submodules https://github.com/enrell/audio-enhancer.git
cd audio-enhancer

# Build for your GPU (auto-detected)
./scripts/docker-run.sh --help

# Or build manually for specific GPU
docker compose --profile rocm build   # AMD GPU
docker compose --profile cuda build   # NVIDIA GPU
docker compose --profile cpu build    # CPU only
```

## Usage

### CLI

```bash
# Basic usage - super resolution + mastering (default)
uv run audio-enhancer input.opus -o output.flac

# With optional stages
uv run audio-enhancer input.mp3 -o output.flac --denoise --harmonic

# Enable loudness normalization
uv run audio-enhancer input.opus -o output.flac --normalize

# Output formats: flac, wav, ogg, opus
uv run audio-enhancer input.webm -o output.opus --format opus
```

### GUI

```bash
uv run audio-enhancer --gui
```

### Docker

```bash
# Auto-detect GPU and process file
./scripts/docker-run.sh input.opus output.flac

# With extra options
./scripts/docker-run.sh input.mp3 output.flac --denoise --normalize

# Or use docker compose directly (place files in input/ folder)
mkdir -p input output
cp myfile.opus input/
docker compose --profile rocm run --rm audio-enhancer-rocm /input/myfile.opus -o /output/myfile.flac
```

### Options

| Flag | Description |
|------|-------------|
| `--denoise` | Enable denoising (for noisy recordings) |
| `--harmonic` | Enable harmonic enhancement |
| `--normalize` | Enable loudness normalization (-14 LUFS) |
| `--no-super-res` | Skip super-resolution stage |
| `--format FORMAT` | Output format: flac, wav, ogg, opus (default: flac) |
| `--sample-rate RATE` | Output sample rate (default: 48000) |
| `--gui` | Launch GUI mode |
| `--info` | Show GPU and system info |

## Pipeline Stages

### 1. Extraction
Extracts audio from any format (mp3, opus, webm, ogg, etc.) using FFmpeg, resamples to target sample rate.

### 2. Super Resolution (AudioSR)
Neural network that reconstructs high frequencies lost during lossy compression. Runs in a separate Python environment due to numpy version conflicts.

Falls back to DSP-based bandwidth extension if AudioSR fails or runs out of VRAM.

### 3. Denoising (Optional)
GPU-accelerated noise reduction using noisereduce. Processes audio in chunks to manage VRAM usage.

### 4. Harmonic Enhancement (Optional)
- Harmonic exciter (soft saturation for warmth)
- Transient shaping
- High-shelf EQ for "air"
- Stereo widening

### 5. Final Mastering
- Soft-knee limiter (prevents clipping)
- Optional loudness normalization to -14 LUFS
- TPDF dithering for bit-depth reduction

## GPU Support

### AMD ROCm

Tested on RX 6600 (8GB VRAM). Uses chunked processing to fit within VRAM limits.

```bash
# Check GPU detection
uv run audio-enhancer --info
```

### NVIDIA CUDA

Should work with CUDA-enabled GPUs. Install PyTorch with CUDA instead of ROCm:

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Project Structure

```
audio-enhancer/
├── src/audio_enhancer/
│   ├── core.py              # Pipeline orchestrator
│   ├── gpu.py               # GPU detection (ROCm/CUDA)
│   ├── gui.py               # Tkinter GUI
│   ├── main.py              # CLI entry point
│   └── pipeline/
│       ├── base.py          # Base classes
│       ├── extraction.py    # Audio extraction (FFmpeg)
│       ├── denoise.py       # Noise reduction
│       ├── super_resolution.py  # AudioSR / DSP fallback
│       ├── harmonic.py      # Harmonic enhancement
│       └── mastering.py     # Final mastering + export
├── audiosr_env/             # AudioSR runner script
│   └── run_audiosr.py       # AudioSR subprocess wrapper
├── scripts/
│   ├── setup_audiosr.sh     # AudioSR venv setup script
│   └── docker-run.sh        # Docker auto-detect runner
├── Dockerfile               # Multi-GPU Docker build
├── docker-compose.yml       # Docker Compose profiles
└── pyproject.toml
```

## Troubleshooting

### AudioSR runs out of VRAM
AudioSR requires ~7GB VRAM. If it fails, the pipeline automatically falls back to DSP-based bandwidth extension.

### "No module named 'pkg_resources'"
```bash
cd audiosr_env && uv pip install setuptools
```

### ROCm not detected
Ensure ROCm is installed and `HSA_OVERRIDE_GFX_VERSION` is set for your GPU:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For RX 6600
```

## License

MIT
