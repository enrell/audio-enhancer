"""Stage 2: Denoising and artifact removal."""

import gc
from typing import Callable, Optional

import numpy as np

from .base import AudioData, PipelineStage, StageConfig, StageType


def _clear_gpu_cache() -> None:
    """Clear GPU memory cache."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()


class DenoiseStage(PipelineStage):
    """Remove noise and compression artifacts."""

    stage_type = StageType.DENOISE

    def __init__(self, config: StageConfig, device: str = "cpu"):
        super().__init__(config, device)
        # resemble-enhance is optional and has torch version conflicts
        self.use_resemble = False
        self.stationary_reduction = config.params.get("stationary_reduction", 0.75)
        self.prop_decrease = config.params.get("prop_decrease", 0.8)
        # Chunk size in seconds for memory efficiency (8GB VRAM)
        self.chunk_seconds = config.params.get("chunk_seconds", 30.0)

    def load_model(self) -> None:
        """Load denoising models (noisereduce is always available)."""
        # noisereduce doesn't require pre-loading
        self._model = "noisereduce"

    def process(
        self,
        audio: AudioData,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> AudioData:
        """Apply denoising to audio."""
        if progress_callback:
            progress_callback(0.0, "Denoising audio...")

        samples = audio.samples.copy()
        samples = self._denoise_chunked(samples, audio.sample_rate, progress_callback)

        _clear_gpu_cache()

        if progress_callback:
            progress_callback(1.0, "Denoising complete")

        return AudioData(
            samples=samples,
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            source_path=audio.source_path,
            metadata={**audio.metadata, "denoised": True},
        )

    def _denoise_chunked(
        self,
        samples: np.ndarray,
        sample_rate: int,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> np.ndarray:
        """Denoise audio in chunks to avoid VRAM overflow."""
        chunk_samples = int(self.chunk_seconds * sample_rate)
        total_samples = len(samples)

        if total_samples <= chunk_samples:
            return self._denoise_noisereduce(samples, sample_rate)

        overlap = int(sample_rate * 0.5)
        result = np.zeros_like(samples)
        weight = np.zeros(total_samples if samples.ndim == 1 else (total_samples,))

        pos = 0
        chunk_idx = 0
        total_chunks = max(
            (total_samples + chunk_samples - 1) // (chunk_samples - overlap), 1
        )

        while pos < total_samples:
            end = min(pos + chunk_samples, total_samples)
            chunk = samples[pos:end]

            if progress_callback:
                progress = 0.1 + 0.8 * (chunk_idx / total_chunks)
                progress_callback(
                    progress, f"Denoising chunk {chunk_idx + 1}/{total_chunks}"
                )

            denoised_chunk = self._denoise_noisereduce(chunk, sample_rate)

            # Create fade window for crossfade
            chunk_len = len(denoised_chunk)
            fade = np.ones(chunk_len)
            if pos > 0:
                fade[:overlap] = np.linspace(0, 1, overlap)
            if end < total_samples:
                fade[-overlap:] = np.linspace(1, 0, overlap)

            # Apply fade based on audio shape
            if samples.ndim == 2:
                fade = fade.reshape(-1, 1)

            result[pos:end] += denoised_chunk * fade
            if samples.ndim == 1:
                weight[pos:end] += fade
            else:
                weight[pos:end] += fade.flatten()

            # Clear GPU cache between chunks
            _clear_gpu_cache()

            pos += chunk_samples - overlap
            chunk_idx += 1

        # Normalize by weight
        if samples.ndim == 1:
            result = result / np.maximum(weight, 1e-8)
        else:
            result = result / np.maximum(weight.reshape(-1, 1), 1e-8)

        return result

    def _denoise_noisereduce(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Denoise using noisereduce library."""
        import noisereduce as nr

        # noisereduce expects shape (channels, samples) for multi-channel
        # but our data is (samples, channels), so we need to transpose
        is_stereo = samples.ndim == 2 and samples.shape[1] == 2

        if is_stereo:
            # Process each channel separately to avoid memory issues
            left = nr.reduce_noise(
                y=samples[:, 0],
                sr=sample_rate,
                stationary=True,
                prop_decrease=self.prop_decrease,
                n_std_thresh_stationary=self.stationary_reduction,
                use_torch=True,
                device=self.device,
            )
            _clear_gpu_cache()

            right = nr.reduce_noise(
                y=samples[:, 1],
                sr=sample_rate,
                stationary=True,
                prop_decrease=self.prop_decrease,
                n_std_thresh_stationary=self.stationary_reduction,
                use_torch=True,
                device=self.device,
            )
            _clear_gpu_cache()

            return np.column_stack([left, right])
        else:
            # Mono audio
            result = nr.reduce_noise(
                y=samples,
                sr=sample_rate,
                stationary=True,
                prop_decrease=self.prop_decrease,
                n_std_thresh_stationary=self.stationary_reduction,
                use_torch=True,
                device=self.device,
            )
            return result
