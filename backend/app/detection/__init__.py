"""ASTRA VISION — Phase 8E Object Detection Preparation & Infrastructure Package."""

from backend.app.detection.schemas import (
    BoundingBox,
    DetectionAnnotation,
    ImageDetectionRecord,
    DatasetDetectionReadinessReport,
)
from backend.app.detection.readiness import evaluate_detection_readiness

__all__ = [
    "BoundingBox",
    "DetectionAnnotation",
    "ImageDetectionRecord",
    "DatasetDetectionReadinessReport",
    "evaluate_detection_readiness",
]
