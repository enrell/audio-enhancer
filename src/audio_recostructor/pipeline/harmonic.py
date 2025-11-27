"""Stage 4: Harmonic enhancement and spatial processing."""

from typing import Callable, Optional

import numpy as np

from .base import AudioData, PipelineStage, StageConfig, StageType


class HarmonicEnhancementStage(PipelineStage):
    """Enhance harmonics and spatial characteristics."""

    stage_type = StageType.HARMONIC_ENHANCEMENT

    def __init__(self, config: StageConfig, device: str = "cpu"):
        super().__init__(config, device)
        self.harmonic_boost_db = config.params.get("harmonic_boost_db", 3.0)
        self.transient_sharpness = config.params.get("transient_sharpness", 0.5)
        self.stereo_width = config.params.get("stereo_width", 1.2)
        self.high_shelf_gain_db = config.params.get("high_shelf_gain_db", 2.0)
        self.high_shelf_freq = config.params.get("high_shelf_freq", 8000)

    def load_model(self) -> None:
        """No ML model needed - using DSP techniques."""
        pass

    def process(
        self,
        audio: AudioData,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> AudioData:
        """Apply harmonic enhancement."""
        if progress_callback:
            progress_callback(0.0, "Enhancing harmonics...")

        samples = audio.samples.copy()

        # Apply harmonic exciter
        if progress_callback:
            progress_callback(0.2, "Applying harmonic exciter...")
        samples = self._harmonic_exciter(samples, audio.sample_rate)

        # Apply transient shaping
        if progress_callback:
            progress_callback(0.4, "Shaping transients...")
        samples = self._transient_shaper(samples, audio.sample_rate)

        # Apply high shelf EQ for "air"
        if progress_callback:
            progress_callback(0.6, "Adding air frequencies...")
        samples = self._high_shelf_eq(samples, audio.sample_rate)

        # Apply stereo widening if stereo
        if audio.channels == 2 and samples.ndim == 2:
            if progress_callback:
                progress_callback(0.8, "Widening stereo field...")
            samples = self._stereo_widener(samples)

        if progress_callback:
            progress_callback(1.0, "Enhancement complete")

        return AudioData(
            samples=samples,
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            source_path=audio.source_path,
            metadata={**audio.metadata, "enhanced": True},
        )

    def _harmonic_exciter(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Add harmonic excitement through soft saturation."""
        from scipy import signal

        # High-pass filter to get high frequencies
        nyq = sample_rate / 2
        high_cutoff = min(2000 / nyq, 0.95)  # Ensure cutoff is valid
        sos = signal.butter(2, high_cutoff, btype="high", output="sos")

        if samples.ndim == 1:
            highs = signal.sosfiltfilt(sos, samples)
        else:
            highs = np.apply_along_axis(
                lambda x: signal.sosfiltfilt(sos, x), 0, samples
            )

        # Soft clip to generate harmonics
        drive = 2.0
        excited = np.tanh(highs * drive) / drive

        # Mix back with gain
        gain = 10 ** (self.harmonic_boost_db / 20)
        result = samples + excited * gain * 0.3

        # Soft limit instead of hard clipping
        result = np.tanh(result * 0.95) / 0.95

        return result

    def _transient_shaper(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Shape transients using envelope following."""
        from scipy import signal

        window_ms = 10
        window_samples = int(sample_rate * window_ms / 1000)

        if samples.ndim == 1:
            envelope = self._get_envelope(samples, window_samples)
        else:
            envelope = np.apply_along_axis(
                lambda x: self._get_envelope(x, window_samples), 0, samples
            )

        # Calculate transient gain (emphasize attack)
        attack_ms = 5
        release_ms = 50
        attack_samples = int(sample_rate * attack_ms / 1000)
        release_samples = int(sample_rate * release_ms / 1000)

        # Simple transient detection via envelope derivative
        envelope_diff = np.diff(envelope, axis=0, prepend=envelope[0:1])
        transients = np.maximum(0, envelope_diff)

        # Smooth the transient envelope
        cutoff = min(1000 / (sample_rate / 2), 0.95)
        sos = signal.butter(1, cutoff, btype="low", output="sos")
        if transients.ndim == 1:
            transient_env = signal.sosfiltfilt(sos, transients)
        else:
            transient_env = np.apply_along_axis(
                lambda x: signal.sosfiltfilt(sos, x), 0, transients
            )

        # Apply transient boost
        boost = 1.0 + transient_env * self.transient_sharpness * 2
        return samples * boost

    def _get_envelope(self, x: np.ndarray, window: int) -> np.ndarray:
        """Calculate signal envelope."""
        from scipy.ndimage import maximum_filter1d

        return maximum_filter1d(np.abs(x), size=window)

    def _high_shelf_eq(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply high shelf EQ for air frequencies."""
        from scipy import signal

        # Design high shelf filter
        nyq = sample_rate / 2
        freq = min(self.high_shelf_freq, nyq * 0.9)

        # Biquad high shelf coefficients
        gain = 10 ** (self.high_shelf_gain_db / 40)
        w0 = 2 * np.pi * freq / sample_rate
        A = gain
        alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / 0.7 - 1) + 2)

        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        b = [b0 / a0, b1 / a0, b2 / a0]
        a = [1, a1 / a0, a2 / a0]

        if samples.ndim == 1:
            return signal.filtfilt(b, a, samples)
        else:
            return np.apply_along_axis(lambda x: signal.filtfilt(b, a, x), 0, samples)

    def _stereo_widener(self, samples: np.ndarray) -> np.ndarray:
        """Widen stereo field using M/S processing."""
        if samples.shape[1] != 2:
            return samples

        left = samples[:, 0]
        right = samples[:, 1]

        # Convert to M/S
        mid = (left + right) / 2
        side = (left - right) / 2

        # Widen by boosting side
        side = side * self.stereo_width

        # Convert back to L/R
        new_left = mid + side
        new_right = mid - side

        result = np.column_stack([new_left, new_right])

        # Prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > 0.99:
            result = result / max_val * 0.99

        return result
