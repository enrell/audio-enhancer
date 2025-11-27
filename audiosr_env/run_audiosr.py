#!/usr/bin/env python3
"""AudioSR runner script - runs in isolated venv with numpy 1.x."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run AudioSR super-resolution")
    parser.add_argument("input", type=Path, help="Input audio file")
    parser.add_argument("output", type=Path, help="Output audio file")
    parser.add_argument("--ddim-steps", type=int, default=50, help="DDIM steps")
    parser.add_argument(
        "--guidance-scale", type=float, default=3.5, help="Guidance scale"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    import torch
    import audiosr
    import soundfile as sf
    import numpy as np

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", file=sys.stderr)

    # Load model
    print("Loading AudioSR model...", file=sys.stderr)
    model = audiosr.build_model(model_name="basic", device=device)

    # Process audio
    print(f"Processing: {args.input}", file=sys.stderr)
    waveform = audiosr.super_resolution(
        model,
        str(args.input),
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        ddim_steps=args.ddim_steps,
    )

    # Save output (audiosr returns 48kHz by default)
    # waveform shape: (batch, channels, samples)
    output_audio = waveform[0].cpu().numpy()

    # Transpose if needed (channels, samples) -> (samples, channels)
    if output_audio.ndim == 2 and output_audio.shape[0] <= 2:
        output_audio = output_audio.T

    sf.write(str(args.output), output_audio, 48000)
    print(f"Saved: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
