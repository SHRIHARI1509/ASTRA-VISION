"""Unit tests for Phase 8E Detection Schemas and Readiness Evaluation."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from backend.app.detection.schemas import (
    BoundingBox,
    DetectionAnnotation,
    ImageDetectionRecord,
    DatasetDetectionReadinessReport,
)
from backend.app.detection.readiness import evaluate_detection_readiness


def test_bounding_box_valid_coordinates():
    box = BoundingBox(ymin=0.1, xmin=0.2, ymax=0.8, xmax=0.9)
    assert box.ymin == 0.1
    assert box.xmin == 0.2
    assert box.ymax == 0.8
    assert box.xmax == 0.9
    assert pytest.approx(box.width) == 0.7
    assert pytest.approx(box.height) == 0.7
    assert pytest.approx(box.area) == 0.49


def test_bounding_box_invalid_inverted_y():
    with pytest.raises(ValidationError, match="ymax.*must be greater than ymin"):
        BoundingBox(ymin=0.8, xmin=0.2, ymax=0.1, xmax=0.9)


def test_bounding_box_invalid_inverted_x():
    with pytest.raises(ValidationError, match="xmax.*must be greater than xmin"):
        BoundingBox(ymin=0.1, xmin=0.9, ymax=0.8, xmax=0.2)


def test_bounding_box_out_of_bounds():
    with pytest.raises(ValidationError):
        BoundingBox(ymin=-0.1, xmin=0.2, ymax=0.8, xmax=0.9)
    with pytest.raises(ValidationError):
        BoundingBox(ymin=0.1, xmin=0.2, ymax=1.1, xmax=0.9)


def test_bounding_box_pixel_conversion():
    box = BoundingBox(ymin=0.1, xmin=0.2, ymax=0.5, xmax=0.6)
    px = box.to_pixel_coords(img_width=1000, img_height=500)
    assert px["xmin"] == 200
    assert px["ymin"] == 50
    assert px["xmax"] == 600
    assert px["ymax"] == 250
    assert px["width"] == 400
    assert px["height"] == 200


def test_bounding_box_yolo_and_coco_export():
    box = BoundingBox(ymin=0.2, xmin=0.1, ymax=0.6, xmax=0.5)
    # width = 0.4, height = 0.4, x_center = 0.3, y_center = 0.4
    yolo_str = box.to_yolo_format(class_id=2)
    assert yolo_str == "2 0.300000 0.400000 0.400000 0.400000"

    coco_bbox = box.to_coco_bbox(img_width=1000, img_height=1000)
    assert coco_bbox == [100.0, 200.0, 400.0, 400.0]


def test_detection_annotation_defaults_and_ground_truth_guard():
    box = BoundingBox(ymin=0.1, xmin=0.1, ymax=0.5, xmax=0.5)
    ann = DetectionAnnotation(box=box, category="Tank")
    assert ann.category == "Tank"
    assert ann.is_ground_truth is False  # Must NOT default to ground truth
    assert ann.annotation_source == "unspecified"


def test_supplied_dataset_detection_readiness_audit():
    dataset_path = Path("data/supplied_dataset")
    assert dataset_path.exists(), "data/supplied_dataset must exist"

    report = evaluate_detection_readiness(dataset_path)
    assert isinstance(report, DatasetDetectionReadinessReport)
    assert report.total_images == 150
    assert report.detection_ready is False
    assert report.label_level == "image_level"
    assert report.total_bounding_boxes == 0
    assert len(report.missing_annotation_reasons) >= 3
    assert any("YOLO" in r for r in report.missing_annotation_reasons)
    assert any("COCO" in r for r in report.missing_annotation_reasons)
    assert any("CSV" in r for r in report.missing_annotation_reasons)
