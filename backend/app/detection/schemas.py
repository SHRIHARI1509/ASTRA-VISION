"""ASTRA VISION — Phase 8E Detection Schemas and Contracts.

Provides canonical schemas for object detection representations, bounding box
math, export formats (YOLO, COCO, Pascal VOC), and dataset detection readiness.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates in [0.0, 1.0] range."""
    ymin: float = Field(..., ge=0.0, le=1.0, description="Top edge coordinate in range [0, 1]")
    xmin: float = Field(..., ge=0.0, le=1.0, description="Left edge coordinate in range [0, 1]")
    ymax: float = Field(..., ge=0.0, le=1.0, description="Bottom edge coordinate in range [0, 1]")
    xmax: float = Field(..., ge=0.0, le=1.0, description="Right edge coordinate in range [0, 1]")

    @model_validator(mode="after")
    def validate_box_dimensions(self) -> "BoundingBox":
        if self.ymax <= self.ymin:
            raise ValueError(f"Invalid bounding box: ymax ({self.ymax}) must be greater than ymin ({self.ymin})")
        if self.xmax <= self.xmin:
            raise ValueError(f"Invalid bounding box: xmax ({self.xmax}) must be greater than xmin ({self.xmin})")
        return self

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_pixel_coords(self, img_width: int, img_height: int) -> Dict[str, int]:
        """Convert normalized coordinates to absolute integer pixel bounds."""
        if img_width <= 0 or img_height <= 0:
            raise ValueError(f"Image dimensions must be positive integers, got ({img_width}, {img_height})")
        return {
            "xmin": int(round(self.xmin * img_width)),
            "ymin": int(round(self.ymin * img_height)),
            "xmax": int(round(self.xmax * img_width)),
            "ymax": int(round(self.ymax * img_height)),
            "width": int(round(self.width * img_width)),
            "height": int(round(self.height * img_height)),
        }

    def to_yolo_format(self, class_id: int) -> str:
        """Export box in standard normalized YOLO format: <class_id> <x_center> <y_center> <width> <height>."""
        x_center = self.xmin + (self.width / 2.0)
        y_center = self.ymin + (self.height / 2.0)
        return f"{class_id} {x_center:.6f} {y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def to_coco_bbox(self, img_width: int, img_height: int) -> List[float]:
        """Export box in standard COCO format: [x_top_left, y_top_left, width, height] in pixel coordinates."""
        px = self.to_pixel_coords(img_width, img_height)
        return [float(px["xmin"]), float(px["ymin"]), float(px["width"]), float(px["height"])]


class DetectionAnnotation(BaseModel):
    """An object detection annotation on an image."""
    box: BoundingBox
    category: str = Field(..., min_length=1, description="Category / class label name")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model confidence score if predicted")
    is_ground_truth: bool = Field(False, description="Flag indicating human-annotated ground truth vs inference")
    annotation_source: str = Field("unspecified", description="Source of annotation: human, ground_truth, synthetic, etc.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary auxiliary annotation metadata")


class ImageDetectionRecord(BaseModel):
    """Container for an image and all its spatial detection annotations."""
    file_name: str
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    annotations: List[DetectionAnnotation] = Field(default_factory=list)

    @property
    def has_annotations(self) -> bool:
        return len(self.annotations) > 0

    @property
    def box_count(self) -> int:
        return len(self.annotations)


class DatasetDetectionReadinessReport(BaseModel):
    """Audit report assessing whether a dataset contains spatial annotations for detection."""
    dataset_name: str
    dataset_path: str
    total_images: int
    images_with_annotations: int
    total_bounding_boxes: int
    detection_ready: bool
    label_level: str = Field("image_level", description="'image_level' or 'object_detection'")
    missing_annotation_reasons: List[str] = Field(default_factory=list)
    format_support: Dict[str, bool] = Field(
        default_factory=lambda: {
            "yolo_txt_found": False,
            "coco_json_found": False,
            "pascal_voc_xml_found": False,
        }
    )
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
