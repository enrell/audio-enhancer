#!/bin/bash
# Auto-detect GPU and run audio-enhancer in Docker
# Usage: ./scripts/docker-run.sh input.opus output.flac [extra args...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_FILE="$1"
OUTPUT_FILE="$2"
shift 2 2>/dev/null || true
EXTRA_ARGS="$@"

if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <input_file> <output_file> [extra args...]"
    echo ""
    echo "Examples:"
    echo "  $0 music.opus enhanced.flac"
    echo "  $0 music.opus enhanced.flac --super-resolution --denoise"
    exit 1
fi

INPUT_PATH="$(realpath "$INPUT_FILE")"
INPUT_DIR="$(dirname "$INPUT_PATH")"
INPUT_NAME="$(basename "$INPUT_PATH")"

OUTPUT_PATH="$(realpath -m "$OUTPUT_FILE")"
OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
OUTPUT_NAME="$(basename "$OUTPUT_PATH")"

mkdir -p "$OUTPUT_DIR"

detect_gpu() {
    if command -v rocminfo &> /dev/null; then
        if rocminfo 2>/dev/null | grep -q "gfx"; then
            echo "rocm"
            return
        fi
    fi
    
    if [ -e /dev/kfd ]; then
        echo "rocm"
        return
    fi
    
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo "cuda"
            return
        fi
    fi
    
    echo "cpu"
}

GPU_TYPE="${GPU_TYPE:-$(detect_gpu)}"
echo "Detected GPU: $GPU_TYPE"

case "$GPU_TYPE" in
    rocm) SERVICE="audio-enhancer-rocm" ;;
    cuda) SERVICE="audio-enhancer-cuda" ;;
    *)    SERVICE="audio-enhancer-cpu" ;;
esac

echo "Building Docker image for $GPU_TYPE..."
cd "$PROJECT_DIR"
docker compose --profile "$GPU_TYPE" build

echo "Processing: $INPUT_NAME -> $OUTPUT_NAME"
docker compose --profile "$GPU_TYPE" run --rm \
    -v "$INPUT_DIR:/input:ro" \
    -v "$OUTPUT_DIR:/output" \
    "$SERVICE" \
    "/input/$INPUT_NAME" \
    -o "/output/$OUTPUT_NAME" \
    $EXTRA_ARGS

echo "Done! Output: $OUTPUT_PATH"
