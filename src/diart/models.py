from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Optional, Text, Union, Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from requests import HTTPError

try:
    from pyannote.audio import Model, Pipeline
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    from pyannote.audio.utils.powerset import Powerset

    IS_PYANNOTE_AVAILABLE = True
except ImportError:
    IS_PYANNOTE_AVAILABLE = False

try:
    import onnxruntime as ort

    IS_ONNX_AVAILABLE = True
except ImportError:
    IS_ONNX_AVAILABLE = False


class PowersetAdapter(nn.Module):
    def __init__(self, segmentation_model: nn.Module):
        super().__init__()
        self.model = segmentation_model
        specs = self.model.specifications
        max_speakers_per_frame = specs.powerset_max_classes
        max_speakers_per_chunk = len(specs.classes)
        self.powerset = Powerset(max_speakers_per_chunk, max_speakers_per_frame)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.powerset.to_multilabel(self.model(waveform))


class PyannoteLoader:
    def __init__(self, model_info, hf_token: Union[Text, bool, None] = True):
        super().__init__()
        self.model_info = model_info
        self.hf_token = hf_token

    def __call__(self) -> Callable:
        try:
            # Try new API first (pyannote.audio >= 3.1), then fall back to old API
            try:
                model = Model.from_pretrained(self.model_info, token=self.hf_token)
            except TypeError:
                model = Model.from_pretrained(self.model_info, use_auth_token=self.hf_token)
            specs = getattr(model, "specifications", None)
            if specs is not None and specs.powerset:
                model = PowersetAdapter(model)
            return model
        except HTTPError:
            pass
        except ModuleNotFoundError:
            pass
        # Try new API first for PretrainedSpeakerEmbedding
        try:
            return PretrainedSpeakerEmbedding(self.model_info, token=self.hf_token)
        except TypeError:
            return PretrainedSpeakerEmbedding(self.model_info, use_auth_token=self.hf_token)


class PipelineLoader:
    """Loader that extracts models from a pyannote Pipeline (e.g., speaker-diarization-community-1)."""
    
    _cached_pipeline = None
    _cached_pipeline_name = None
    _cached_segmentation = None
    _cached_embedding = None
    
    def __init__(
        self, 
        pipeline_name: str, 
        model_type: str,  # "segmentation" or "embedding"
        hf_token: Union[Text, bool, None] = True
    ):
        super().__init__()
        self.pipeline_name = pipeline_name
        self.model_type = model_type
        self.hf_token = hf_token

    @classmethod
    def _load_pipeline(cls, pipeline_name: str, hf_token: Union[Text, bool, None]) -> Pipeline:
        """Load and cache the pipeline to avoid loading it twice."""
        if cls._cached_pipeline is None or cls._cached_pipeline_name != pipeline_name:
            try:
                cls._cached_pipeline = Pipeline.from_pretrained(pipeline_name, token=hf_token)
            except TypeError:
                cls._cached_pipeline = Pipeline.from_pretrained(pipeline_name, use_auth_token=hf_token)
            cls._cached_pipeline_name = pipeline_name
            cls._cached_segmentation = None
            cls._cached_embedding = None
        return cls._cached_pipeline

    @classmethod
    def _extract_segmentation(cls, pipeline) -> nn.Module:
        """Extract segmentation model from pipeline."""
        if cls._cached_segmentation is not None:
            return cls._cached_segmentation
            
        model = None
        # Try various attribute names used by different pyannote pipeline versions
        for attr in ["_segmentation", "segmentation_model", "segmentation"]:
            candidate = getattr(pipeline, attr, None)
            if candidate is not None:
                # Check if it's a model or a sub-pipeline
                if hasattr(candidate, "model"):
                    model = candidate.model
                elif hasattr(candidate, "forward") or hasattr(candidate, "__call__"):
                    model = candidate
                break
        
        if model is None:
            # Debug: print available attributes
            attrs = [a for a in dir(pipeline) if not a.startswith("__")]
            raise ValueError(
                f"Could not find segmentation model. "
                f"Available pipeline attributes: {attrs}"
            )
        
        # Wrap with PowersetAdapter if needed
        specs = getattr(model, "specifications", None)
        if specs is not None and getattr(specs, "powerset", False):
            model = PowersetAdapter(model)
        
        cls._cached_segmentation = model
        return model

    @classmethod
    def _extract_embedding(cls, pipeline) -> Callable:
        """Extract embedding model from pipeline."""
        if cls._cached_embedding is not None:
            return cls._cached_embedding
            
        model = None
        # Try various attribute names
        for attr in ["_embedding", "embedding", "embedding_model"]:
            candidate = getattr(pipeline, attr, None)
            if candidate is not None:
                model = candidate
                break
        
        if model is None:
            attrs = [a for a in dir(pipeline) if not a.startswith("__")]
            raise ValueError(
                f"Could not find embedding model. "
                f"Available pipeline attributes: {attrs}"
            )
        
        cls._cached_embedding = model
        return model

    def __call__(self) -> Callable:
        pipeline = self._load_pipeline(self.pipeline_name, self.hf_token)
        
        if self.model_type == "segmentation":
            return self._extract_segmentation(pipeline)
        elif self.model_type == "embedding":
            return self._extract_embedding(pipeline)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")


def load_from_pipeline(
    pipeline_name: str = "pyannote/speaker-diarization-community-1",
    hf_token: Union[Text, bool, None] = True
) -> Tuple["SegmentationModel", "EmbeddingModel"]:
    """
    Load segmentation and embedding models from a pyannote pipeline.
    
    This is useful for using newer unified pipelines like speaker-diarization-community-1
    which bundle both models together.
    
    Parameters
    ----------
    pipeline_name : str
        The HuggingFace model ID for the pipeline.
        Default: "pyannote/speaker-diarization-community-1"
    hf_token : str | bool | None
        The HuggingFace access token. If True, uses huggingface-cli login token.
        
    Returns
    -------
    segmentation_model : SegmentationModel
    embedding_model : EmbeddingModel
    
    Example
    -------
    >>> from diart.models import load_from_pipeline
    >>> segmentation, embedding = load_from_pipeline("pyannote/speaker-diarization-community-1")
    """
    assert IS_PYANNOTE_AVAILABLE, "No pyannote.audio installation found"
    
    seg_loader = PipelineLoader(pipeline_name, "segmentation", hf_token)
    emb_loader = PipelineLoader(pipeline_name, "embedding", hf_token)
    
    return SegmentationModel(seg_loader), EmbeddingModel(emb_loader)


class ONNXLoader:
    def __init__(self, path: str | Path, input_names: List[str], output_name: str):
        super().__init__()
        self.path = Path(path)
        self.input_names = input_names
        self.output_name = output_name

    def __call__(self) -> ONNXModel:
        return ONNXModel(self.path, self.input_names, self.output_name)


class ONNXModel:
    def __init__(self, path: Path, input_names: List[str], output_name: str):
        super().__init__()
        self.path = path
        self.input_names = input_names
        self.output_name = output_name
        self.device = torch.device("cpu")
        self.session = None
        self.recreate_session()

    @property
    def execution_provider(self) -> str:
        device = "CUDA" if self.device.type == "cuda" else "CPU"
        return f"{device}ExecutionProvider"

    def recreate_session(self):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.path,
            sess_options=options,
            providers=[self.execution_provider],
        )

    def to(self, device: torch.device) -> ONNXModel:
        if device.type != self.device.type:
            self.device = device
            self.recreate_session()
        return self

    def __call__(self, *args) -> torch.Tensor:
        inputs = {
            name: arg.cpu().numpy().astype(np.float32)
            for name, arg in zip(self.input_names, args)
        }
        output = self.session.run([self.output_name], inputs)[0]
        return torch.from_numpy(output).float().to(args[0].device)


class LazyModel(ABC):
    def __init__(self, loader: Callable[[], Callable]):
        super().__init__()
        self.get_model = loader
        self.model: Optional[Callable] = None

    def is_in_memory(self) -> bool:
        """Return whether the model has been loaded into memory"""
        return self.model is not None

    def load(self):
        if not self.is_in_memory():
            self.model = self.get_model()

    def to(self, device: torch.device) -> LazyModel:
        self.load()
        self.model = self.model.to(device)
        return self

    def __call__(self, *args, **kwargs):
        self.load()
        return self.model(*args, **kwargs)

    def eval(self) -> LazyModel:
        self.load()
        if isinstance(self.model, nn.Module):
            self.model.eval()
        return self


class SegmentationModel(LazyModel):
    """
    Minimal interface for a segmentation model.
    """

    @staticmethod
    def from_pyannote(
        model, use_hf_token: Union[Text, bool, None] = True
    ) -> "SegmentationModel":
        """
        Returns a `SegmentationModel` wrapping a pyannote model.

        Parameters
        ----------
        model: pyannote.PipelineModel
            The pyannote.audio model to fetch.
        use_hf_token: str | bool, optional
            The Huggingface access token to use when downloading the model.
            If True, use huggingface-cli login token.
            Defaults to None.

        Returns
        -------
        wrapper: SegmentationModel
        """
        assert IS_PYANNOTE_AVAILABLE, "No pyannote.audio installation found"
        return SegmentationModel(PyannoteLoader(model, use_hf_token))

    @staticmethod
    def from_pipeline(
        pipeline_name: str = "pyannote/speaker-diarization-community-1",
        use_hf_token: Union[Text, bool, None] = True,
    ) -> "SegmentationModel":
        """
        Extract a SegmentationModel from a pyannote pipeline.

        Parameters
        ----------
        pipeline_name : str
            The HuggingFace model ID for the pipeline.
            Default: "pyannote/speaker-diarization-community-1"
        use_hf_token : str | bool | None
            The HuggingFace access token.

        Returns
        -------
        wrapper: SegmentationModel
        """
        assert IS_PYANNOTE_AVAILABLE, "No pyannote.audio installation found"
        return SegmentationModel(PipelineLoader(pipeline_name, "segmentation", use_hf_token))

    @staticmethod
    def from_onnx(
        model_path: Union[str, Path],
        input_name: str = "waveform",
        output_name: str = "segmentation",
    ) -> "SegmentationModel":
        assert IS_ONNX_AVAILABLE, "No ONNX installation found"
        return SegmentationModel(ONNXLoader(model_path, [input_name], output_name))

    @staticmethod
    def from_pretrained(
        model, use_hf_token: Union[Text, bool, None] = True
    ) -> "SegmentationModel":
        if isinstance(model, str) or isinstance(model, Path):
            if Path(model).name.endswith(".onnx"):
                return SegmentationModel.from_onnx(model)
        return SegmentationModel.from_pyannote(model, use_hf_token)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Call the forward pass of the segmentation model.
        Parameters
        ----------
        waveform: torch.Tensor, shape (batch, channels, samples)
        Returns
        -------
        speaker_segmentation: torch.Tensor, shape (batch, frames, speakers)
        """
        return super().__call__(waveform)


class EmbeddingModel(LazyModel):
    """Minimal interface for an embedding model."""

    @staticmethod
    def from_pyannote(
        model, use_hf_token: Union[Text, bool, None] = True
    ) -> "EmbeddingModel":
        """
        Returns an `EmbeddingModel` wrapping a pyannote model.

        Parameters
        ----------
        model: pyannote.PipelineModel
            The pyannote.audio model to fetch.
        use_hf_token: str | bool, optional
            The Huggingface access token to use when downloading the model.
            If True, use huggingface-cli login token.
            Defaults to None.

        Returns
        -------
        wrapper: EmbeddingModel
        """
        assert IS_PYANNOTE_AVAILABLE, "No pyannote.audio installation found"
        loader = PyannoteLoader(model, use_hf_token)
        return EmbeddingModel(loader)

    @staticmethod
    def from_pipeline(
        pipeline_name: str = "pyannote/speaker-diarization-community-1",
        use_hf_token: Union[Text, bool, None] = True,
    ) -> "EmbeddingModel":
        """
        Extract an EmbeddingModel from a pyannote pipeline.

        Parameters
        ----------
        pipeline_name : str
            The HuggingFace model ID for the pipeline.
            Default: "pyannote/speaker-diarization-community-1"
        use_hf_token : str | bool | None
            The HuggingFace access token.

        Returns
        -------
        wrapper: EmbeddingModel
        """
        assert IS_PYANNOTE_AVAILABLE, "No pyannote.audio installation found"
        return EmbeddingModel(PipelineLoader(pipeline_name, "embedding", use_hf_token))

    @staticmethod
    def from_onnx(
        model_path: Union[str, Path],
        input_names: List[str] | None = None,
        output_name: str = "embedding",
    ) -> "EmbeddingModel":
        assert IS_ONNX_AVAILABLE, "No ONNX installation found"
        input_names = input_names or ["waveform", "weights"]
        loader = ONNXLoader(model_path, input_names, output_name)
        return EmbeddingModel(loader)

    @staticmethod
    def from_pretrained(
        model, use_hf_token: Union[Text, bool, None] = True
    ) -> "EmbeddingModel":
        if isinstance(model, str) or isinstance(model, Path):
            if Path(model).name.endswith(".onnx"):
                return EmbeddingModel.from_onnx(model)
        return EmbeddingModel.from_pyannote(model, use_hf_token)

    def __call__(
        self, waveform: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Call the forward pass of an embedding model with optional weights.
        Parameters
        ----------
        waveform: torch.Tensor, shape (batch, channels, samples)
        weights: Optional[torch.Tensor], shape (batch, frames)
            Temporal weights for each sample in the batch. Defaults to no weights.
        Returns
        -------
        speaker_embeddings: torch.Tensor, shape (batch, embedding_dim)
        """
        embeddings = super().__call__(waveform, weights)
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        return embeddings
