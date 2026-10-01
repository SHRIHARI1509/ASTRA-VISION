import io
from typing import List, Optional, Dict, Any
from PIL import Image

from app.core.config import settings
from app.models.base_adapter import ModelAdapter, ModelInferenceResult
from app.models.siglip2_adapter import (
    SigLIP2Adapter,
    ModelInitializationError,
    ModelInferenceError,
)
from app.services.image_service import image_service, ImageValidationError
from app.utils.logger import logger


class ModelUnavailableError(Exception):
    """Raised when inference is requested but model is not available or failed to load."""
    def __init__(self, message: str = "Vision model is currently unavailable."):
        self.message = message
        self.code = "MODEL_UNAVAILABLE"
        self.status_code = 503
        super().__init__(message)


class InferenceExecutionError(Exception):
    """Raised when inference execution encounters a model error."""
    def __init__(self, message: str = "Inference execution failed."):
        self.message = message
        self.code = "INFERENCE_ERROR"
        self.status_code = 500
        super().__init__(message)


class InferenceService:
    """Orchestrates zero-shot vision inference decoupled from concrete model implementations."""

    def __init__(self, adapter: Optional[ModelAdapter] = None):
        self._adapter = adapter or SigLIP2Adapter()
        self._initialization_error: Optional[str] = None

    @property
    def adapter(self) -> ModelAdapter:
        return self._adapter

    @property
    def is_ready(self) -> bool:
        return self._adapter.is_loaded

    @property
    def device(self) -> str:
        return self._adapter.device

    @property
    def model_id(self) -> str:
        return self._adapter.model_id

    def initialize_model(self) -> bool:
        """Initialize and cache the model in memory at startup. Gracefully handles failures."""
        try:
            logger.info(f"Initializing inference adapter for '{self._adapter.model_id}'...")
            self._adapter.load()
            self._initialization_error = None
            logger.info("Inference adapter initialized successfully and cached in memory.")
            return True
        except Exception as e:
            self._initialization_error = str(e)
            logger.error(
                f"Gracefully captured model initialization failure: {e}. "
                "Backend will remain running with model marked as unavailable.",
                exc_info=True,
            )
            return False

    def run_inference(
        self,
        file_bytes: bytes,
        filename: str = "uploaded_image",
        content_type: Optional[str] = None,
        candidate_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Validate input image and perform zero-shot inference against candidate labels."""
        # 1. Validate image format, integrity, dimensions via image_service
        metadata = image_service.validate_and_inspect_image(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
        )

        # 2. Check if model is initialized and ready
        if not self._adapter.is_loaded:
            # Try once to load if not yet initialized
            try:
                self._adapter.load()
            except Exception as e:
                self._initialization_error = str(e)
                raise ModelUnavailableError(
                    f"Model '{self._adapter.model_id}' is unavailable: {str(e)}"
                )

        # 3. Decode raster into Pillow image
        try:
            pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception as e:
            raise ImageValidationError(
                code="CORRUPTED_IMAGE",
                message=f"Failed to decode image raster: {str(e)}",
            )

        # 4. Execute zero-shot inference via adapter
        labels = candidate_labels or settings.CANDIDATE_CATEGORIES
        try:
            result: ModelInferenceResult = self._adapter.predict(
                image=pil_image,
                candidate_labels=labels,
            )
        except ModelInferenceError as e:
            raise InferenceExecutionError(str(e))
        except Exception as e:
            raise InferenceExecutionError(f"Unexpected inference error: {str(e)}")

        return {
            "model": result.model,
            "device": result.device,
            "predictions": [
                {"label": p.label, "score": p.score}
                for p in result.predictions
            ],
            "inference_time_ms": result.inference_time_ms,
            "status": result.status,
        }


# Default singleton instance
inference_service = InferenceService()
