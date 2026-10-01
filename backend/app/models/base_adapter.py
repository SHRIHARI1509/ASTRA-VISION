from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from PIL import Image


class Prediction(BaseModel):
    label: str
    score: float


class ModelInferenceResult(BaseModel):
    model: str
    device: str
    predictions: List[Prediction]
    inference_time_ms: float
    status: str = "success"


class ModelAdapter(ABC):
    """Abstract model adapter layer decoupling inference service from model implementation."""

    @abstractmethod
    def load(self) -> None:
        """Load model and processor weights into memory and target device."""
        pass

    @abstractmethod
    def predict(
        self,
        image: Image.Image,
        candidate_labels: Optional[List[str]] = None,
    ) -> ModelInferenceResult:
        """Run zero-shot inference on an image given candidate labels."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if model and processor are loaded and cached in memory."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Return current execution device ('cuda' or 'cpu')."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return model identifier string."""
        pass

    @property
    @abstractmethod
    def initialization_time_s(self) -> Optional[float]:
        """Return time taken to initialize the model in seconds, if loaded."""
        pass
