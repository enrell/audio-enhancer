"""Stage 1: Audio extraction and format conversion using FFmpeg."""

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from .base import AudioData, PipelineStage, StageConfig, StageType


class ExtractionStage(PipelineStage):
    """Extract and convert audio to working format."""

    stage_type = StageType.EXTRACTION

    def __init__(self, config: StageConfig, device: str = "cpu"):
        super().__init__(config, device)
        self.target_sample_rate = config.params.get("sample_rate", 48000)
        self.normalize = config.params.get("normalize", True)

    def load_model(self) -> None:
        """No model needed for extraction."""
        pass

    def process(
        self,
        audio: AudioData,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> AudioData:
        """Extract audio from input file."""
        if progress_callback:
            progress_callback(0.0, "Extracting audio...")

        if audio.source_path is None:
            raise ValueError("Source path required for extraction stage")

        source_path = audio.source_path

        suffix = source_path.suffix.lower()
        needs_ffmpeg = suffix in {".webm", ".mp4", ".mkv", ".avi", ".ogg", ".opus"}

        if needs_ffmpeg:
            samples, sample_rate = self._extract_with_ffmpeg(source_path)
        else:
            samples, sample_rate = sf.read(str(source_path), dtype="float32")  # type: ignore[misc]

        if progress_callback:
            progress_callback(0.5, "Resampling...")

        if sample_rate != self.target_sample_rate:
            samples = self._resample(samples, sample_rate, self.target_sample_rate)
            sample_rate = self.target_sample_rate

        peak = np.max(np.abs(samples))
        if peak > 0.95:
            samples = samples * (0.95 / peak)

        if progress_callback:
            progress_callback(1.0, "Extraction complete")

        channels = 2 if samples.ndim == 2 and samples.shape[1] == 2 else 1

        return AudioData(
            samples=samples,
            sample_rate=sample_rate,
            channels=channels,
            source_path=source_path,
            metadata={"original_format": suffix},
        )

    def _extract_with_ffmpeg(self, source_path: Path) -> tuple[np.ndarray, int]:
        """Extract audio using FFmpeg."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                "ffmpeg",
                "-i",
                str(source_path),
                "-acodec",
                "pcm_f32le",
                "-ar",
                str(self.target_sample_rate),
                "-y",
                tmp_path,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300  # 5 min timeout
            )
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")

            samples, sample_rate = sf.read(tmp_path, dtype="float32")  # type: ignore[misc]
            return samples, sample_rate
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _resample(
        self, samples: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resample audio using scipy."""
        from scipy import signal

        if orig_sr == target_sr:
            return samples

        # Calculate resampling ratio
        num_samples = int(len(samples) * target_sr / orig_sr)
        resampled: np.ndarray = signal.resample(samples, num_samples, axis=0)  # type: ignore[assignment]
        return resampled

    def _normalize_loudness(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Normalize loudness to -14 LUFS (streaming standard)."""
        import pyloudnorm as pyln

        meter = pyln.Meter(sample_rate)

        # Handle mono/stereo
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
            was_mono = True
        else:
            was_mono = False

        loudness = meter.integrated_loudness(samples)

        # Normalize to -14 LUFS
        if not np.isinf(loudness):
            samples = pyln.normalize.loudness(samples, loudness, -14.0)

        if was_mono:
            samples = samples.flatten()

        return samples
