"""Audio processing pipeline stages."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

import numpy as np


class StageType(Enum):
    """Pipeline stage types."""

    EXTRACTION = auto()
    DENOISE = auto()
    SUPER_RESOLUTION = auto()
    HARMONIC_ENHANCEMENT = auto()
    FINAL_MASTERING = auto()


@dataclass
class AudioData:
    """Container for audio data between pipeline stages."""

    samples: np.ndarray
    sample_rate: int
    channels: int
    source_path: Optional[Path] = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Get audio duration in seconds."""
        if self.sample_rate == 0:
            return 0.0
        return len(self.samples) / self.sample_rate

    @property
    def is_stereo(self) -> bool:
        """Check if audio is stereo."""
        return self.channels == 2


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""

    enabled: bool = True
    params: dict = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    extraction: StageConfig = field(default_factory=StageConfig)
    denoise: StageConfig = field(default_factory=StageConfig)
    super_resolution: StageConfig = field(default_factory=StageConfig)
    harmonic_enhancement: StageConfig = field(default_factory=StageConfig)
    final_mastering: StageConfig = field(default_factory=StageConfig)

    # Global settings
    output_format: str = "flac"
    output_sample_rate: int = 48000
    output_bit_depth: int = 24
    chunk_duration_seconds: float = 30.0  # Process in chunks for memory efficiency


class PipelineStage(ABC):
    """Base class for pipeline stages."""

    stage_type: StageType

    def __init__(self, config: StageConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self._model = None

    @abstractmethod
    def process(
        self,
        audio: AudioData,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> AudioData:
        """Process audio through this stage."""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Load any required ML models."""
        pass

    def unload_model(self) -> None:
        """Unload models to free memory."""
        self._model = None

    @property
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
