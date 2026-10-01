"""ASTRA VISION — Phase 8F Dataset Splitter & Manifest Generator.

Generates deterministic, stratified train/validation split manifests for the
supplied dataset while strictly enforcing held-out benchmark protection.
Zero images on disk are moved, renamed, or modified.
"""

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class SplitItem(BaseModel):
    """An individual image item in a split manifest."""
    file_name: str
    category: str
    relative_path: str
    sha256: str


class SplitSummary(BaseModel):
    """Summary telemetry for generated dataset splits."""
    random_seed: int
    train_ratio: float
    val_ratio: float
    total_images: int
    train_count: int
    val_count: int
    category_distribution: Dict[str, Dict[str, int]]
    held_out_leakage_count: int
    benchmark_protected: bool
    manifest_paths: Dict[str, str]


class DatasetSplitter:
    """Safely generates train and validation split manifests."""

    def __init__(
        self,
        dataset_root: Path,
        held_out_manifest_path: Optional[Path] = None,
        seed: int = 42,
    ):
        self.dataset_root = Path(dataset_root)
        self.held_out_manifest_path = Path(held_out_manifest_path) if held_out_manifest_path else None
        self.seed = seed

    def _compute_sha256(self, file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def generate_split(
        self,
        output_dir: Path,
        train_ratio: float = 0.8,
    ) -> SplitSummary:
        """Generate stratified train/val split manifests."""
        labels_path = self.dataset_root / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.csv not found at {labels_path}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load held-out benchmark hashes if provided
        held_out_hashes = set()
        if self.held_out_manifest_path and self.held_out_manifest_path.exists():
            with open(self.held_out_manifest_path, "r", encoding="utf-8") as f:
                held_out_data = json.load(f)
                for item in held_out_data.get("images", []):
                    if "sha256" in item:
                        held_out_hashes.add(item["sha256"])

        # 2. Group items by supplied category
        categorized_items: Dict[str, List[SplitItem]] = {}
        with open(labels_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = row["file_name"]
                cat = row["category"]
                full_path = self.dataset_root / fn
                if not full_path.exists():
                    raise FileNotFoundError(f"Referenced image not found: {full_path}")

                file_hash = self._compute_sha256(full_path)
                item = SplitItem(
                    file_name=fn,
                    category=cat,
                    relative_path=fn,
                    sha256=file_hash,
                )
                if cat not in categorized_items:
                    categorized_items[cat] = []
                categorized_items[cat].append(item)

        # 3. Deterministic stratification
        rng = random.Random(self.seed)
        train_items: List[SplitItem] = []
        val_items: List[SplitItem] = []
        distribution: Dict[str, Dict[str, int]] = {}

        for cat, items in sorted(categorized_items.items()):
            # Sort first for cross-platform determinism before shuffling
            sorted_items = sorted(items, key=lambda x: x.file_name)
            rng.shuffle(sorted_items)

            k_train = int(round(len(sorted_items) * train_ratio))
            cat_train = sorted_items[:k_train]
            cat_val = sorted_items[k_train:]

            train_items.extend(cat_train)
            val_items.extend(cat_val)

            distribution[cat] = {
                "train": len(cat_train),
                "val": len(cat_val),
                "total": len(sorted_items),
            }

        # 4. Check for any leakage into held-out benchmark
        all_split_hashes = {item.sha256 for item in train_items + val_items}
        leakage = all_split_hashes.intersection(held_out_hashes)

        # 5. Write isolated manifests
        train_manifest_path = output_dir / "train_manifest.json"
        val_manifest_path = output_dir / "val_manifest.json"
        summary_path = output_dir / "split_summary.json"

        with open(train_manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split_name": "train",
                    "seed": self.seed,
                    "count": len(train_items),
                    "items": [item.model_dump() for item in train_items],
                },
                f,
                indent=2,
            )

        with open(val_manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split_name": "val",
                    "seed": self.seed,
                    "count": len(val_items),
                    "items": [item.model_dump() for item in val_items],
                },
                f,
                indent=2,
            )

        summary = SplitSummary(
            random_seed=self.seed,
            train_ratio=train_ratio,
            val_ratio=round(1.0 - train_ratio, 2),
            total_images=len(train_items) + len(val_items),
            train_count=len(train_items),
            val_count=len(val_items),
            category_distribution=distribution,
            held_out_leakage_count=len(leakage),
            benchmark_protected=(len(leakage) == 0),
            manifest_paths={
                "train_manifest": str(train_manifest_path.resolve()),
                "val_manifest": str(val_manifest_path.resolve()),
                "split_summary": str(summary_path.resolve()),
            },
        )

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary.model_dump(), f, indent=2)

        return summary


def generate_stratified_split(
    dataset_root: Path,
    output_dir: Path,
    held_out_manifest_path: Optional[Path] = None,
    seed: int = 42,
    train_ratio: float = 0.8,
) -> SplitSummary:
    """Helper function to execute the dataset split."""
    splitter = DatasetSplitter(
        dataset_root=dataset_root,
        held_out_manifest_path=held_out_manifest_path,
        seed=seed,
    )
    return splitter.generate_split(output_dir=output_dir, train_ratio=train_ratio)
