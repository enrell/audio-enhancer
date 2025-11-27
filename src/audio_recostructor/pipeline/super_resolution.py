"""Stage 3: Audio super-resolution using AudioSR or DSP fallback."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from .base import AudioData, PipelineStage, StageConfig, StageType


# Path to AudioSR venv
AUDIOSR_VENV = Path(__file__).parent.parent.parent.parent / "audiosr_env" / ".venv"
AUDIOSR_PYTHON = AUDIOSR_VENV / "bin" / "python"
AUDIOSR_SCRIPT = (
    Path(__file__).parent.parent.parent.parent / "audiosr_env" / "run_audiosr.py"
)


class SuperResolutionStage(PipelineStage):
    """Upsample audio bandwidth using AudioSR or DSP fallback.

    AudioSR runs in a separate venv due to numpy version conflicts.
    Falls back to DSP-based bandwidth extension if AudioSR is unavailable.
    """

    stage_type = StageType.SUPER_RESOLUTION

    def __init__(self, config: StageConfig, device: str = "cpu"):
        super().__init__(config, device)
        self.target_sample_rate = config.params.get("target_sample_rate", 48000)
        self.use_audiosr = config.params.get("use_audiosr", True)
        self.ddim_steps = config.params.get("ddim_steps", 50)
        self.guidance_scale = config.params.get("guidance_scale", 3.5)
        # DSP fallback settings
        self.harmonic_extension = config.params.get("harmonic_extension", True)
        self.extension_gain_db = config.params.get("extension_gain_db", -6.0)

    def load_model(self) -> None:
        """Check if AudioSR is available."""
        if self.use_audiosr and AUDIOSR_PYTHON.exists() and AUDIOSR_SCRIPT.exists():
            self._model = "audiosr"
        else:
            self._model = "dsp"

    def process(
        self,
        audio: AudioData,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> AudioData:
        """Apply super-resolution to audio."""
        if self._model == "audiosr":
            return self._process_audiosr(audio, progress_callback)
        else:
            return self._process_dsp(audio, progress_callback)

    def _process_audiosr(
        self,
        audio: AudioData,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> AudioData:
        """Process using AudioSR in separate venv."""
        if progress_callback:
            progress_callback(0.0, "Running AudioSR super-resolution...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            tmp_input = Path(tmp_in.name)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            tmp_output = Path(tmp_out.name)

        try:
            sf.write(str(tmp_input), audio.samples, audio.sample_rate)

            if progress_callback:
                progress_callback(0.1, "Loading AudioSR model...")

            cmd = [
                str(AUDIOSR_PYTHON),
                str(AUDIOSR_SCRIPT),
                str(tmp_input),
                str(tmp_output),
                "--ddim-steps",
                str(self.ddim_steps),
                "--guidance-scale",
                str(self.guidance_scale),
            ]

            env = os.environ.copy()
            # Ensure ROCm is used
            env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

            if progress_callback:
                progress_callback(0.2, "Processing with AudioSR...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )

            if result.returncode != 0:
                raise RuntimeError(f"AudioSR failed: {result.stderr}")

            if progress_callback:
                progress_callback(0.9, "Loading result...")

            samples, sample_rate = sf.read(str(tmp_output), dtype="float32")  # type: ignore[misc]

            if progress_callback:
                progress_callback(1.0, "AudioSR complete")

            return AudioData(
                samples=samples,
                sample_rate=sample_rate,
                channels=2 if samples.ndim == 2 else 1,
                source_path=audio.source_path,
                metadata={
                    **audio.metadata,
                    "super_resolved": True,
                    "method": "audiosr",
                    "ddim_steps": self.ddim_steps,
                },
            )

        except Exception as e:
            if progress_callback:
                progress_callback(0.0, f"AudioSR failed ({e}), using DSP fallback...")
            return self._process_dsp(audio, progress_callback)

        finally:
            tmp_input.unlink(missing_ok=True)
            tmp_output.unlink(missing_ok=True)

    def _process_dsp(
        self,
        audio: AudioData,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> AudioData:
        """Process using DSP-based bandwidth extension."""
        if progress_callback:
            progress_callback(0.0, "Applying DSP bandwidth extension...")

        samples = audio.samples.copy()
        current_sr = audio.sample_rate

        if current_sr < self.target_sample_rate:
            if progress_callback:
                progress_callback(0.3, "Upsampling...")
            samples = self._resample(samples, current_sr, self.target_sample_rate)
            current_sr = self.target_sample_rate

        if self.harmonic_extension:
            if progress_callback:
                progress_callback(0.6, "Extending harmonics...")
            samples = self._harmonic_bandwidth_extension(samples, current_sr)

        if progress_callback:
            progress_callback(1.0, "Bandwidth extension complete")

        return AudioData(
            samples=samples,
            sample_rate=current_sr,
            channels=audio.channels,
            source_path=audio.source_path,
            metadata={
                **audio.metadata,
                "super_resolved": True,
                "method": "dsp",
                "original_sample_rate": audio.sample_rate,
            },
        )

    def _resample(
        self, samples: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        from scipy import signal

        if orig_sr == target_sr:
            return samples

        num_samples = int(len(samples) * target_sr / orig_sr)
        resampled: np.ndarray = signal.resample(samples, num_samples, axis=0)  # type: ignore[assignment]
        return resampled

    def _harmonic_bandwidth_extension(
        self, samples: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Extend bandwidth by generating harmonics from existing content."""
        from scipy import signal

        nyq = sample_rate / 2

        # Extract frequencies for harmonic generation (2-8 kHz)
        low_cut = 2000 / nyq
        high_cut = min(8000 / nyq, 0.95)

        b, a = signal.butter(4, [low_cut, high_cut], btype="band")  # type: ignore

        if samples.ndim == 1:
            mid_highs = signal.filtfilt(b, a, samples)
        else:
            mid_highs = np.apply_along_axis(
                lambda x: signal.filtfilt(b, a, x), 0, samples
            )

        # Generate harmonics through soft clipping
        drive = 3.0
        harmonics = np.tanh(mid_highs * drive)

        # Highpass to get only new high frequencies
        hp_cut = min(10000 / nyq, 0.95)
        b_hp, a_hp = signal.butter(4, hp_cut, btype="high")  # type: ignore

        if harmonics.ndim == 1:
            new_highs = signal.filtfilt(b_hp, a_hp, harmonics)
        else:
            new_highs = np.apply_along_axis(
                lambda x: signal.filtfilt(b_hp, a_hp, x), 0, harmonics
            )

        # Mix with gain control
        gain = 10 ** (self.extension_gain_db / 20)
        result = samples + new_highs * gain

        # Soft limit
        max_val = np.max(np.abs(result))
        if max_val > 0.99:
            result = np.tanh(result * 0.99 / max_val)

        return result
