"""CLI entry point for Audio Enhancer."""

import argparse
import sys
from pathlib import Path

from .core import AudioReconstructor, create_default_config
from .gpu import detect_gpu


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Audio Enhancer - Recover quality from compressed audio"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Input audio file (launches GUI if not provided)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path",
    )
    parser.add_argument(
        "--format",
        choices=["flac", "wav", "ogg", "opus"],
        default="flac",
        help="Output format (default: flac)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Output sample rate (default: 48000)",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Enable denoising (for noisy recordings)",
    )
    parser.add_argument(
        "--no-super-res",
        action="store_true",
        help="Skip super-resolution stage",
    )
    parser.add_argument(
        "--harmonic",
        action="store_true",
        help="Enable harmonic enhancement",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable loudness normalization to -14 LUFS",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI mode",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show GPU and system info",
    )

    args = parser.parse_args()

    if args.info:
        gpu = detect_gpu()
        print(f"GPU: {gpu.device_name}")
        print(f"Backend: {gpu.backend.value}")
        print(f"VRAM: {gpu.vram_gb:.1f} GB")
        return

    if args.gui or args.input is None:
        from .gui import main as gui_main

        gui_main()
        return

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    config = create_default_config()
    config.output_format = args.format
    config.output_sample_rate = args.sample_rate
    config.denoise.enabled = args.denoise
    config.super_resolution.enabled = not args.no_super_res
    config.harmonic_enhancement.enabled = args.harmonic
    config.final_mastering.params["normalize_loudness"] = args.normalize

    if args.output:
        output_path = args.output
    else:
        output_path = args.input.parent / f"{args.input.stem}_restored.{args.format}"

    gpu = detect_gpu()
    print(f"Using: {gpu.device_name} ({gpu.backend.value})")
    print(f"Processing: {args.input}")

    reconstructor = AudioReconstructor(config)

    def progress(p: float, stage: str, msg: str) -> None:
        print(f"[{p*100:5.1f}%] {stage}: {msg}")

    reconstructor.process(args.input, output_path, progress)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
