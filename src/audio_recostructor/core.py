"""Main pipeline orchestrator."""

from pathlib import Path
from typing import Callable, Optional

from .gpu import GPUInfo, configure_environment, detect_gpu, get_torch_device
from .pipeline import (
    AudioData,
    DenoiseStage,
    ExtractionStage,
    FinalMasteringStage,
    HarmonicEnhancementStage,
    PipelineConfig,
    StageConfig,
    SuperResolutionStage,
)


class AudioReconstructor:
    """Main audio reconstruction pipeline orchestrator."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        gpu_info: Optional[GPUInfo] = None,
    ):
        self.config = config or PipelineConfig()

        self.gpu_info = gpu_info or detect_gpu()
        configure_environment(self.gpu_info)
        self.device = get_torch_device(self.gpu_info)

        self._stages = self._create_stages()

    def _create_stages(self) -> list:
        """Create pipeline stages based on configuration."""
        stages = []

        if self.config.extraction.enabled:
            stages.append(ExtractionStage(self.config.extraction, self.device))

        if self.config.denoise.enabled:
            stages.append(DenoiseStage(self.config.denoise, self.device))

        if self.config.super_resolution.enabled:
            stages.append(
                SuperResolutionStage(self.config.super_resolution, self.device)
            )

        if self.config.harmonic_enhancement.enabled:
            stages.append(
                HarmonicEnhancementStage(self.config.harmonic_enhancement, self.device)
            )

        if self.config.final_mastering.enabled:
            stages.append(FinalMasteringStage(self.config.final_mastering, self.device))

        return stages

    def process(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[float, str, str], None]] = None,
    ) -> AudioData:
        """Process audio file through the pipeline.

        Args:
            input_path: Path to input audio file
            output_path: Optional path for output file
            progress_callback: Callback with (progress: 0-1, stage_name, message)

        Returns:
            Processed AudioData
        """
        import numpy as np

        audio = AudioData(
            np.array([]),
            0,  # sample_rate
            0,  # channels
            Path(input_path),
        )

        total_stages = len(self._stages)

        for i, stage in enumerate(self._stages):
            stage_name = stage.stage_type.name.replace("_", " ").title()

            def stage_progress(p: float, msg: str) -> None:
                if progress_callback:
                    overall = (i + p) / max(total_stages, 1)
                    progress_callback(overall, stage_name, msg)

            if not stage.is_model_loaded:
                stage_progress(0.0, "Loading model...")
                stage.load_model()

            # Process
            audio = stage.process(audio, stage_progress)

        if output_path:
            self._export_audio(audio, output_path)

        return audio

    def _export_audio(self, audio: AudioData, output_path: Path) -> None:
        """Export audio to file."""
        import soundfile as sf
        import subprocess
        import tempfile

        format_str = self.config.output_format
        bit_depth = self.config.output_bit_depth

        # Opus requires FFmpeg encoding
        if format_str == "opus":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                sf.write(
                    str(tmp_path), audio.samples, audio.sample_rate, subtype="FLOAT"
                )
                cmd = [
                    "ffmpeg",
                    "-i",
                    str(tmp_path),
                    "-c:a",
                    "libopus",
                    "-b:a",
                    "256k",
                    "-vbr",
                    "on",
                    "-compression_level",
                    "10",
                    "-y",
                    str(output_path),
                ]
                subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300, check=True
                )
            finally:
                tmp_path.unlink(missing_ok=True)
            return

        # Determine format and subtype
        format_upper = format_str.upper()
        if format_str == "ogg":
            subtype = "VORBIS"
        elif format_str == "wav":
            subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}.get(bit_depth, "PCM_24")
        elif format_str == "flac":
            subtype = {16: "PCM_16", 24: "PCM_24"}.get(bit_depth, "PCM_24")
        else:
            subtype = "PCM_24"

        sf.write(
            str(output_path),
            audio.samples,
            audio.sample_rate,
            format=format_upper,
            subtype=subtype,
        )

    def unload_models(self) -> None:
        """Unload all models to free memory."""
        for stage in self._stages:
            stage.unload_model()


def create_default_config() -> PipelineConfig:
    """Create default pipeline configuration.

    Chunk sizes are tuned for 8GB VRAM GPUs like RX 6600.
    Reduce chunk_seconds if you have less VRAM.
    """
    return PipelineConfig(
        extraction=StageConfig(
            enabled=True,
            params={"sample_rate": 48000, "normalize": True},
        ),
        denoise=StageConfig(
            enabled=False,  # Disabled by default - use for noisy recordings only
            params={
                "use_resemble_enhance": False,
                "chunk_seconds": 20.0,
            },
        ),
        super_resolution=StageConfig(
            enabled=True,
            params={
                "target_sample_rate": 48000,
                "harmonic_extension": True,
                "extension_gain_db": -6.0,
            },
        ),
        harmonic_enhancement=StageConfig(
            enabled=False,  # Disabled by default - AudioSR already enhances harmonics
            params={
                "harmonic_boost_db": 3.0,
                "stereo_width": 1.2,
                "high_shelf_gain_db": 2.0,
            },
        ),
        final_mastering=StageConfig(
            enabled=True,
            params={
                "normalize_loudness": False,
                "target_lufs": -14.0,
                "true_peak_dbtp": -1.0,
                "apply_dither": True,
                "output_bit_depth": 24,
            },
        ),
        output_format="flac",
        output_sample_rate=48000,
    )
