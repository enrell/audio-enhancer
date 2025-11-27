"""Stage 6: Final mastering - loudness normalization, limiting, and export."""

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .base import AudioData, PipelineStage, StageConfig, StageType


class FinalMasteringStage(PipelineStage):
    """Final mastering: limiting, loudness normalization, dithering."""

    stage_type = StageType.FINAL_MASTERING

    def __init__(self, config: StageConfig, device: str = "cpu"):
        super().__init__(config, device)
        self.normalize_loudness = config.params.get("normalize_loudness", False)
        self.target_lufs = config.params.get("target_lufs", -14.0)
        self.true_peak_dbtp = config.params.get("true_peak_dbtp", -1.0)
        self.apply_dither = config.params.get("apply_dither", True)
        self.output_bit_depth = config.params.get("output_bit_depth", 24)

    def load_model(self) -> None:
        """No model needed for mastering."""
        pass

    def process(
        self,
        audio: AudioData,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> AudioData:
        """Apply final mastering chain."""
        if progress_callback:
            progress_callback(0.0, "Applying final mastering...")

        samples = audio.samples.copy()

        if progress_callback:
            progress_callback(0.25, "Applying limiter...")
        samples = self._brickwall_limiter(samples, audio.sample_rate)

        # Normalize loudness (optional)
        if self.normalize_loudness:
            if progress_callback:
                progress_callback(0.5, "Normalizing loudness...")
            samples = self._normalize_loudness(samples, audio.sample_rate)

        if self.apply_dither and self.output_bit_depth < 32:
            if progress_callback:
                progress_callback(0.75, "Applying dither...")
            samples = self._apply_dither(samples)

        if progress_callback:
            progress_callback(1.0, "Mastering complete")

        return AudioData(
            samples=samples,
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            source_path=audio.source_path,
            metadata={
                **audio.metadata,
                "mastered": True,
                "target_lufs": self.target_lufs,
                "bit_depth": self.output_bit_depth,
            },
        )

    def _brickwall_limiter(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply soft-knee limiter at true peak ceiling."""
        # Convert dBTP to linear
        ceiling = 10 ** (self.true_peak_dbtp / 20)
        threshold = ceiling * 0.85  # Start limiting below ceiling

        # Simple soft limiter using tanh
        # Scale to threshold, apply tanh, scale back
        scaled = samples / threshold
        limited = np.tanh(scaled) * threshold
        limited = np.clip(limited, -ceiling, ceiling)

        return limited

    def _normalize_loudness(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Normalize to target LUFS."""
        import pyloudnorm as pyln

        meter = pyln.Meter(sample_rate)

        # Handle mono/stereo
        if samples.ndim == 1:
            measure_samples = samples.reshape(-1, 1)
        else:
            measure_samples = samples

        loudness = meter.integrated_loudness(measure_samples)

        if np.isinf(loudness):
            return samples

        # Calculate gain needed
        gain_db = self.target_lufs - loudness
        gain_linear = 10 ** (gain_db / 20)

        normalized = samples * gain_linear

        # Ensure we don't exceed true peak
        ceiling = 10 ** (self.true_peak_dbtp / 20)
        peak = np.max(np.abs(normalized))
        if peak > ceiling:
            normalized = normalized * (ceiling / peak)

        return normalized

    def _apply_dither(self, samples: np.ndarray) -> np.ndarray:
        """Apply TPDF (triangular probability density function) dither."""
        # Calculate dither amplitude based on target bit depth
        # Dither amplitude should be 1 LSB at target bit depth
        dither_amplitude = 1.0 / (2 ** (self.output_bit_depth - 1))

        # Generate TPDF dither (sum of two uniform random values)
        rng = np.random.default_rng(seed=42)
        dither = (
            rng.uniform(-dither_amplitude, dither_amplitude, samples.shape)
            + rng.uniform(-dither_amplitude, dither_amplitude, samples.shape)
        ) / 2

        return samples + dither

    def export(
        self,
        audio: AudioData,
        output_path: Path,
        format: str = "flac",
    ) -> Path:
        """Export audio to file."""
        import soundfile as sf

        if format == "opus":
            return self._export_opus(audio, output_path)

        format_upper = format.upper()
        if format == "ogg":
            subtype = "VORBIS"
        elif format == "wav":
            subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}.get(
                self.output_bit_depth, "PCM_24"
            )
        elif format == "flac":
            subtype = {16: "PCM_16", 24: "PCM_24"}.get(self.output_bit_depth, "PCM_24")
        else:
            subtype = "PCM_24"

        sf.write(
            str(output_path),
            audio.samples,
            audio.sample_rate,
            format=format_upper,
            subtype=subtype,
        )

        return output_path

    def _export_opus(
        self,
        audio: AudioData,
        output_path: Path,
    ) -> Path:
        """Export to Opus format using FFmpeg."""
        import subprocess
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            sf.write(str(tmp_path), audio.samples, audio.sample_rate, subtype="FLOAT")

            # Encode to Opus using FFmpeg with high quality settings
            cmd = [
                "ffmpeg",
                "-i",
                str(tmp_path),
                "-c:a",
                "libopus",
                "-b:a",
                "360k",  # High bitrate for quality
                "-vbr",
                "on",
                "-compression_level",
                "10",
                "-y",
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg opus encoding failed: {result.stderr}")

        finally:
            tmp_path.unlink(missing_ok=True)

        return output_path
