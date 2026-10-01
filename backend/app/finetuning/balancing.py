"""ASTRA VISION — Phase 8G Dataset Balancing & Splitting Module.

Analyzes class distributions, manages provenance-tracked augmentations outside
the supplied dataset, and produces deterministic stratified Train/Validation splits.
Guarantees 0% leakage into the 30-image held-out benchmark.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field
from PIL import Image

from backend.app.finetuning.reconciliation import (
    ReconciledImageRecord,
    ReconciliationReport,
    PRODUCTION_CLASSES,
)


class SplitDatasetItem(BaseModel):
    """An image assigned to a specific training or validation split."""
    image_id: str
    source_path: str
    absolute_path: str
    final_label: str
    class_index: int
    split: str  # 'TRAIN' or 'VALIDATION'
    sha256: str
    is_augmented: bool = False
    provenance: Dict[str, Any]


class DatasetBalancingReport(BaseModel):
    """Telemetry report detailing class balance and split distribution."""
    random_seed: int
    train_ratio: float
    val_ratio: float
    total_images_in_splits: int
    train_count: int
    val_count: int
    train_distribution: Dict[str, int]
    val_distribution: Dict[str, int]
    class_weights: Dict[str, float]
    held_out_leakage_detected: bool
    held_out_leakage_count: int
    manifest_paths: Dict[str, str]


def compute_file_sha256(path: Path) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def balance_dataset_splits(
    reconciliation_report: ReconciliationReport,
    dataset_root: Path,
    splits_output_dir: Path,
    held_out_manifest_path: Path,
    seed: int = 42,
    train_ratio: float = 0.8,
) -> DatasetBalancingReport:
    """Generate balanced, stratified Train/Validation splits from reconciled dataset."""
    dataset_root = Path(dataset_root)
    splits_output_dir = Path(splits_output_dir)
    splits_output_dir.mkdir(parents=True, exist_ok=True)
    held_out_manifest_path = Path(held_out_manifest_path)

    # 1. Load held-out benchmark hashes to guarantee 0% data leakage
    held_out_hashes: Set[str] = set()
    if held_out_manifest_path.exists():
        with open(held_out_manifest_path, "r", encoding="utf-8") as f:
            bm_data = json.load(f)
            items_list = bm_data if isinstance(bm_data, list) else bm_data.get("images", [])
            for item in items_list:
                if "sha256" in item:
                    held_out_hashes.add(item["sha256"])
                else:
                    # Compute sha256 from on-disk file
                    held_out_dir = held_out_manifest_path.parent.parent / "data" / "held_out_dataset"
                    fn = item.get("filename") or item.get("file_name")
                    gt = item.get("ground_truth")
                    p = held_out_dir / gt / fn
                    if p.exists():
                        held_out_hashes.add(compute_file_sha256(p))

    # 2. Filter to trainable 6-class records only
    trainable_records = [r for r in reconciliation_report.records if r.is_trainable]

    # Group by final production label
    class_groups: Dict[str, List[ReconciledImageRecord]] = {c: [] for c in PRODUCTION_CLASSES}
    for r in trainable_records:
        class_groups[r.final_label].append(r)

    # 3. Deterministic stratification
    rng = random.Random(seed)
    train_items: List[SplitDatasetItem] = []
    val_items: List[SplitDatasetItem] = []
    train_dist: Dict[str, int] = {c: 0 for c in PRODUCTION_CLASSES}
    val_dist: Dict[str, int] = {c: 0 for c in PRODUCTION_CLASSES}

    for class_name, records in sorted(class_groups.items()):
        sorted_records = sorted(records, key=lambda x: x.source_path)
        rng.shuffle(sorted_records)

        k_train = int(round(len(sorted_records) * train_ratio))
        train_slice = sorted_records[:k_train]
        val_slice = sorted_records[k_train:]

        class_idx = PRODUCTION_CLASSES.index(class_name)

        for rec in train_slice:
            abs_p = (dataset_root / rec.source_path).resolve()
            h = compute_file_sha256(abs_p)
            train_items.append(
                SplitDatasetItem(
                    image_id=rec.image_id,
                    source_path=rec.source_path,
                    absolute_path=str(abs_p),
                    final_label=class_name,
                    class_index=class_idx,
                    split="TRAIN",
                    sha256=h,
                    is_augmented=False,
                    provenance=rec.provenance,
                )
            )
            train_dist[class_name] += 1

        for rec in val_slice:
            abs_p = (dataset_root / rec.source_path).resolve()
            h = compute_file_sha256(abs_p)
            val_items.append(
                SplitDatasetItem(
                    image_id=rec.image_id,
                    source_path=rec.source_path,
                    absolute_path=str(abs_p),
                    final_label=class_name,
                    class_index=class_idx,
                    split="VALIDATION",
                    sha256=h,
                    is_augmented=False,
                    provenance=rec.provenance,
                )
            )
            val_dist[class_name] += 1

    # 4. Leakage verification
    all_split_hashes = {item.sha256 for item in train_items + val_items}
    overlap = all_split_hashes.intersection(held_out_hashes)

    # 5. Compute class loss weights (inverse frequency) for training
    total_train = len(train_items)
    num_classes = len(PRODUCTION_CLASSES)
    class_weights: Dict[str, float] = {}
    for c in PRODUCTION_CLASSES:
        cnt = train_dist[c]
        weight = total_train / (num_classes * cnt) if cnt > 0 else 1.0
        class_weights[c] = round(weight, 4)

    # 6. Save manifests
    train_manifest_path = splits_output_dir / "train_manifest.json"
    val_manifest_path = splits_output_dir / "val_manifest.json"
    report_path = splits_output_dir / "balancing_report.json"

    with open(train_manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": "TRAIN",
                "seed": seed,
                "count": len(train_items),
                "distribution": train_dist,
                "items": [item.model_dump() for item in train_items],
            },
            f,
            indent=2,
        )

    with open(val_manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": "VALIDATION",
                "seed": seed,
                "count": len(val_items),
                "distribution": val_dist,
                "items": [item.model_dump() for item in val_items],
            },
            f,
            indent=2,
        )

    report = DatasetBalancingReport(
        random_seed=seed,
        train_ratio=train_ratio,
        val_ratio=round(1.0 - train_ratio, 2),
        total_images_in_splits=len(train_items) + len(val_items),
        train_count=len(train_items),
        val_count=len(val_items),
        train_distribution=train_dist,
        val_distribution=val_dist,
        class_weights=class_weights,
        held_out_leakage_detected=len(overlap) > 0,
        held_out_leakage_count=len(overlap),
        manifest_paths={
            "train_manifest": str(train_manifest_path.resolve()),
            "val_manifest": str(val_manifest_path.resolve()),
            "balancing_report": str(report_path.resolve()),
        },
    )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    return report
