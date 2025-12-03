# Fix for PyTorch 2.6+ compatibility with pyannote models
# PyTorch 2.6+ defaults to weights_only=True which breaks loading pyannote models
# that use omegaconf for configuration storage
import warnings
import torch

# Suppress noisy deprecation warnings from torchaudio/torchcodec migration
warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*In 2.9, this function's implementation.*")

# Monkeypatch torch.load to always use weights_only=False for pyannote compatibility
# This is needed because lightning explicitly passes weights_only=True
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False for backward compatibility with pyannote models
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from .blocks import (
    SpeakerDiarization,
    Pipeline,
    SpeakerDiarizationConfig,
    PipelineConfig,
    VoiceActivityDetection,
    VoiceActivityDetectionConfig,
)
from .models import load_from_pipeline
