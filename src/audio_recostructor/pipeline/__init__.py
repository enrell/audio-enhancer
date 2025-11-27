"""Pipeline module - orchestrates all processing stages."""

from .base import AudioData, PipelineConfig, StageConfig, StageType
from .denoise import DenoiseStage
from .extraction import ExtractionStage
from .harmonic import HarmonicEnhancementStage
from .mastering import FinalMasteringStage
from .super_resolution import SuperResolutionStage

__all__ = [
    "AudioData",
    "PipelineConfig",
    "StageConfig",
    "StageType",
    "ExtractionStage",
    "DenoiseStage",
    "SuperResolutionStage",
    "HarmonicEnhancementStage",
    "FinalMasteringStage",
]
