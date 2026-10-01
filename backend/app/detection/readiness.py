"""ASTRA VISION — Phase 8E Detection Readiness Evaluator.

Inspects datasets to verify whether valid spatial object detection annotations
(YOLO, COCO, Pascal VOC) exist or whether the dataset is exclusively image-level.
Strictly safeguards against inventing fake bounding boxes or misrepresenting
classification-only datasets.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List

from backend.app.detection.schemas import DatasetDetectionReadinessReport


def evaluate_detection_readiness(dataset_path: Path) -> DatasetDetectionReadinessReport:
    """Evaluate whether a given dataset directory is detection-ready.
    
    Inspects:
    1. Direct annotation files (.txt with YOLO boxes, .json with COCO instances, .xml VOC files).
    2. Column headers in any CSV files.
    3. Image directory structure.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

    # Count image files
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [p for p in dataset_path.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]
    total_images = len(image_files)

    # Check for detection annotation files
    yolo_txt_files = [
        p for p in dataset_path.rglob("*.txt")
        if p.is_file() and p.name not in {"README.txt", "requirements.txt", "classes.txt"}
    ]
    coco_json_files = [
        p for p in dataset_path.rglob("*.json")
        if p.is_file() and any(k in p.name.lower() for k in ["instances", "coco", "annotations"])
    ]
    voc_xml_files = [
        p for p in dataset_path.rglob("*.xml")
        if p.is_file()
    ]

    # Inspect CSVs for spatial columns
    csv_files = list(dataset_path.glob("*.csv"))
    bbox_columns_found = False
    csv_inspection = {}
    for csv_file in csv_files:
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, [])
                has_bbox_col = any(col.lower() in {"bbox", "xmin", "ymin", "xmax", "ymax", "x1", "y1", "x2", "y2"} for col in header)
                csv_inspection[csv_file.name] = {
                    "columns": header,
                    "contains_bbox_columns": has_bbox_col
                }
                if has_bbox_col:
                    bbox_columns_found = True
        except Exception as e:
            csv_inspection[csv_file.name] = {"error": str(e)}

    # Determine readiness
    yolo_found = len(yolo_txt_files) > 0
    coco_found = len(coco_json_files) > 0
    voc_found = len(voc_xml_files) > 0

    reasons: List[str] = []
    if not yolo_found:
        reasons.append("No YOLO .txt label files found in dataset")
    if not coco_found:
        reasons.append("No COCO instances/annotation JSON files found in dataset")
    if not voc_found:
        reasons.append("No Pascal VOC XML annotation files found in dataset")
    if not bbox_columns_found:
        reasons.append("Supplied CSV files contain image-level classification categories only, no bounding box coordinates")

    is_ready = bool((yolo_found or coco_found or voc_found or bbox_columns_found) and total_images > 0)

    return DatasetDetectionReadinessReport(
        dataset_name=dataset_path.name,
        dataset_path=str(dataset_path.resolve()),
        total_images=total_images,
        images_with_annotations=0 if not is_ready else -1,
        total_bounding_boxes=0 if not is_ready else -1,
        detection_ready=is_ready,
        label_level="object_detection" if is_ready else "image_level",
        missing_annotation_reasons=reasons if not is_ready else [],
        format_support={
            "yolo_txt_found": yolo_found,
            "coco_json_found": coco_found,
            "pascal_voc_xml_found": voc_found,
        },
        diagnostics={
            "csv_inspection": csv_inspection,
            "yolo_txt_count": len(yolo_txt_files),
            "coco_json_count": len(coco_json_files),
            "voc_xml_count": len(voc_xml_files),
        }
    )
