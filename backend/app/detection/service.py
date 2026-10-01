"""ASTRA VISION — Phase 8E Detection Service Contract.

Defines the isolated abstract base detector interface for future object detection
extensions, ensuring zero coupling with the core classification pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from backend.app.detection.schemas import DetectionAnnotation, BoundingBox


class DetectionRequest(BaseModel):
    """Request payload for object detection inference."""
    candidate_labels: List[str] = Field(..., min_length=1, description="Candidate target labels to detect")
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Minimum detection score threshold")
    nms_iou_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Non-maximum suppression IoU threshold")


class DetectionResponse(BaseModel):
    """Response payload returned by an object detection model."""
    detector_name: str
    target_count: int
    detections: List[DetectionAnnotation]
    inference_time_ms: float
    model_metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseObjectDetector(ABC):
    """Abstract base class for object detection engines (e.g. OWLv2, Grounding DINO, YOLO)."""

    @abstractmethod
    def detect(
        self,
        image_bytes: bytes,
        request: DetectionRequest,
    ) -> DetectionResponse:
        """Run object detection on raw image bytes."""
        pass

    @abstractmethod
    def get_supported_labels(self) -> List[str]:
        """Return labels natively supported by this detector."""
        pass
