"""ASTRA VISION — Phase 10 Multi-Object Detection Schemas.

Defines Pydantic v2 validation contracts for the open-vocabulary multi-object
detection API endpoint, spatial bounding box coordinates, detected target schemas,
and error envelopes.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator


class BoxCoordinates(BaseModel):
    """Absolute pixel bounding box coordinates bounded to image dimensions."""
    x1: int = Field(..., ge=0, description="Left horizontal boundary pixel (0-indexed)")
    y1: int = Field(..., ge=0, description="Top vertical boundary pixel (0-indexed)")
    x2: int = Field(..., ge=0, description="Right horizontal boundary pixel (0-indexed)")
    y2: int = Field(..., ge=0, description="Bottom vertical boundary pixel (0-indexed)")

    @model_validator(mode="after")
    def validate_box_dimensions(self) -> "BoxCoordinates":
        if self.x2 <= self.x1:
            raise ValueError(f"Invalid bounding box: x2 ({self.x2}) must be strictly greater than x1 ({self.x1})")
        if self.y2 <= self.y1:
            raise ValueError(f"Invalid bounding box: y2 ({self.y2}) must be strictly greater than y1 ({self.y1})")
        return self


class DetectedObject(BaseModel):
    """A detected object with recognized class label, score, and bounding box."""
    class_name: str = Field(..., min_length=1, description="Normalized defence taxonomy class label")
    score: float = Field(..., ge=0.0, le=1.0, description="Empirical detection filtering score")
    box: BoxCoordinates = Field(..., description="Absolute pixel bounding box coordinates")


class DetectionResponse(BaseModel):
    """Response payload for multi-object detection requests."""
    detections: List[DetectedObject] = Field(default_factory=list, description="List of detected objects")
    image_width: int = Field(..., gt=0, description="Width of the analyzed image in pixels")
    image_height: int = Field(..., gt=0, description="Height of the analyzed image in pixels")
    count: int = Field(..., ge=0, description="Total number of detected objects passing operating threshold")
    detector: str = Field(default="IDEA-Research/grounding-dino-base", description="Detector model identifier")
    device: str = Field(default="cpu", description="Inference execution device (cpu or cuda)")
    inference_time_ms: float = Field(default=0.0, ge=0.0, description="Detector inference latency in milliseconds")


class DetectionErrorDetail(BaseModel):
    """Structured error object for detection endpoint failures."""
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error explanation")


class DetectionErrorResponse(BaseModel):
    """Error envelope for detection endpoint failures."""
    error: DetectionErrorDetail
