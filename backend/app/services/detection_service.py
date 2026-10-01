"""ASTRA VISION — Phase 10 Detection Service.

Orchestrates multi-object detection workflows using Grounding DINO.
Decoupled from the classification pipeline to ensure complete independence
and zero regressions.
"""

import io
from typing import Optional, Dict, Any
from PIL import Image

from app.core.config import settings
from app.models.grounding_dino_adapter import (
    GroundingDINOAdapter,
    DetectionModelInitializationError,
    DetectionModelInferenceError,
)
from app.services.image_service import image_service, ImageValidationError
from app.utils.logger import logger


class DetectionServiceError(Exception):
    """Base exception for detection service operations."""
    def __init__(self, message: str, code: str = "DETECTION_ERROR", status_code: int = 500):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(message)


class DetectionModelUnavailableError(DetectionServiceError):
    """Raised when Grounding DINO detector cannot be loaded or is unavailable."""
    def __init__(self, message: str = "Object detection model is currently unavailable."):
        super().__init__(message=message, code="DETECTOR_UNAVAILABLE", status_code=503)


class DetectionInferenceError(DetectionServiceError):
    """Raised when object detection inference execution fails."""
    def __init__(self, message: str = "Object detection inference failed."):
        super().__init__(message=message, code="DETECTION_INFERENCE_ERROR", status_code=500)


class DetectionService:
    """Orchestrates image validation, decoding, and Grounding DINO object detection."""

    def __init__(self, adapter: Optional[GroundingDINOAdapter] = None):
        self._adapter = adapter or GroundingDINOAdapter()
        self._initialization_error: Optional[str] = None

    @property
    def adapter(self) -> GroundingDINOAdapter:
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

    def initialize_detector(self) -> bool:
        """Initialize and cache detector in memory. Gracefully handles errors."""
        try:
            logger.info(f"Initializing detection adapter for '{self._adapter.model_id}'...")
            self._adapter.load()
            self._initialization_error = None
            return True
        except DetectionModelInitializationError as exc:
            self._initialization_error = exc.message
            logger.error(f"Detector initialization failed: {exc.message}")
            return False
        except Exception as exc:
            self._initialization_error = str(exc)
            logger.error(f"Unexpected detector initialization failure: {exc}", exc_info=True)
            return False

    def detect_objects(
        self,
        file_bytes: bytes,
        filename: str = "uploaded_image",
        content_type: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Validate input image and execute Grounding DINO object detection.
        
        Args:
            file_bytes: Raw binary image content
            filename: Original uploaded filename
            content_type: Optional MIME content type
            box_threshold: Detection box threshold override
            text_threshold: Text alignment threshold override
            
        Returns:
            Dictionary matching the DetectionResponse schema.
            
        Raises:
            ImageValidationError: If image fails verification boundaries
            DetectionModelUnavailableError: If detector is not available
            DetectionInferenceError: If inference execution fails
        """
        # 1. Server-side raster verification
        image_service.validate_and_inspect_image(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
        )

        # 2. Decode into standard PIL image (RGB)
        try:
            image = Image.open(io.BytesIO(file_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as exc:
            logger.error(f"Failed to decode image '{filename}': {exc}")
            raise ImageValidationError(f"Cannot decode image: {str(exc)}", code="DECODE_ERROR")

        # 3. Model lazy load and inference
        try:
            return self._adapter.detect(
                image=image,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        except DetectionModelInitializationError as exc:
            raise DetectionModelUnavailableError(exc.message)
        except DetectionModelInferenceError as exc:
            raise DetectionInferenceError(exc.message)


detection_service = DetectionService()
