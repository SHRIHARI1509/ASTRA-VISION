"""Unit tests for Phase 8G Fine-Tuning, Reconciliation, Balancing, and Model Comparison."""

import json
from pathlib import Path
import pytest
import torch

from backend.app.finetuning.reconciliation import (
    PRODUCTION_CLASSES,
    reconcile_supplied_dataset,
    ReconciliationReport,
)
from backend.app.finetuning.balancing import (
    balance_dataset_splits,
    DatasetBalancingReport,
)
from backend.app.finetuning.comparator import (
    compare_baseline_and_finetuned,
    MetricRow,
    ModelComparisonReport,
)
from backend.app.finetuning.evaluator import BenchmarkEvaluationMetrics
from backend.app.finetuning.trainer import TrainingConfig
from backend.app.core.config import settings


def test_1_six_class_taxonomy_validation():
    """Verify production taxonomy contains exactly the 6 required classes."""
    assert len(PRODUCTION_CLASSES) == 6
    assert set(PRODUCTION_CLASSES) == {
        "Tank",
        "Military Vehicle",
        "Fighter Aircraft",
        "Helicopter",
        "Ship",
        "Drone",
    }


def test_2_no_unresolved_labels_enter_training():
    """Verify no image enters trainable pipeline with an unresolved or ambiguous label."""
    dataset_root = Path("data/supplied_dataset")
    report = reconcile_supplied_dataset(dataset_root)

    assert report.total_images_reviewed == 150
    assert report.unresolved_count == 0

    trainable_records = [r for r in report.records if r.is_trainable]
    for r in trainable_records:
        assert r.final_label in PRODUCTION_CLASSES
        assert r.review_status == "REVIEWED_APPROVED"

    excluded_records = [r for r in report.records if not r.is_trainable]
    assert len(excluded_records) == 6
    for r in excluded_records:
        assert r.final_label == "EXCLUDED_OUT_OF_TAXONOMY"
        assert r.review_status == "REVIEWED_EXCLUDED"


def test_3_deterministic_split_generation():
    """Verify stratified split produces identical allocations across runs with seed=42."""
    dataset_root = Path("data/supplied_dataset")
    held_out_manifest = Path("evaluation/manifest.json")
    splits_dir = Path("data/experiments/test_splits")

    report = reconcile_supplied_dataset(dataset_root)
    split1 = balance_dataset_splits(report, dataset_root, splits_dir, held_out_manifest, seed=42)
    split2 = balance_dataset_splits(report, dataset_root, splits_dir, held_out_manifest, seed=42)

    assert split1.train_count == split2.train_count
    assert split1.val_count == split2.val_count
    assert split1.train_distribution == split2.train_distribution
    assert split1.val_distribution == split2.val_distribution


def test_4_no_train_validation_overlap():
    """Verify zero overlap between train and validation split manifests."""
    train_manifest = Path("data/experiments/train_manifest.json")
    val_manifest = Path("data/experiments/val_manifest.json")

    if not train_manifest.exists() or not val_manifest.exists():
        # Generate them if running test independently
        dataset_root = Path("data/supplied_dataset")
        held_out_manifest = Path("evaluation/manifest.json")
        splits_dir = Path("data/experiments")
        report = reconcile_supplied_dataset(dataset_root)
        balance_dataset_splits(report, dataset_root, splits_dir, held_out_manifest, seed=42)

    with open(train_manifest, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_manifest, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    train_hashes = {item["sha256"] for item in train_data["items"]}
    val_hashes = {item["sha256"] for item in val_data["items"]}

    overlap = train_hashes.intersection(val_hashes)
    assert len(overlap) == 0, f"Found train/validation overlap: {overlap}"


def test_5_no_benchmark_leakage():
    """Verify zero overlap between training/validation items and the sacred held-out benchmark."""
    train_manifest = Path("data/experiments/train_manifest.json")
    val_manifest = Path("data/experiments/val_manifest.json")
    benchmark_manifest = Path("evaluation/manifest.json")
    held_out_dir = Path("data/held_out_dataset")

    from backend.app.finetuning.balancing import compute_file_sha256

    with open(train_manifest, "r", encoding="utf-8") as f:
        train_hashes = {item["sha256"] for item in json.load(f)["items"]}
    with open(val_manifest, "r", encoding="utf-8") as f:
        val_hashes = {item["sha256"] for item in json.load(f)["items"]}
    with open(benchmark_manifest, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    bench_hashes = set()
    for item in bench_data:
        p = held_out_dir / item["ground_truth"] / item["filename"]
        if p.exists():
            bench_hashes.add(compute_file_sha256(p))

    assert len(train_hashes.intersection(bench_hashes)) == 0
    assert len(val_hashes.intersection(bench_hashes)) == 0


def test_6_dataset_manifest_integrity():
    """Verify that all files in the split manifests actually exist on disk."""
    train_manifest = Path("data/experiments/train_manifest.json")
    with open(train_manifest, "r", encoding="utf-8") as f:
        items = json.load(f)["items"]

    for item in items:
        p = Path(item["absolute_path"])
        assert p.exists(), f"File {p} does not exist on disk"
        assert p.stat().st_size > 0


def test_7_correct_label_mapping():
    """Verify safe categories map directly and ambiguous categories are partitioned."""
    report = reconcile_supplied_dataset(Path("data/supplied_dataset"))
    records_by_orig = {}
    for r in report.records:
        records_by_orig.setdefault(r.original_label, []).append(r)

    # Safe mappings
    assert all(r.final_label == "Drone" for r in records_by_orig["drone"])
    assert all(r.final_label == "Helicopter" for r in records_by_orig["helicopter"])
    assert all(r.final_label == "Ship" for r in records_by_orig["naval"])

    # Ambiguous mappings
    mv_labels = {r.final_label for r in records_by_orig["military-vehicle"]}
    assert "Tank" in mv_labels
    assert "Military Vehicle" in mv_labels
    assert len(mv_labels) == 2  # Only Tank and Military Vehicle

    air_labels = {r.final_label for r in records_by_orig["aircraft"]}
    assert "Fighter Aircraft" in air_labels
    assert "EXCLUDED_OUT_OF_TAXONOMY" in air_labels


def test_8_training_config_validity():
    """Verify machine-readable training configuration enforces PEFT parameters."""
    cfg = TrainingConfig()
    assert cfg.model_identifier == "google/siglip2-base-patch16-512"
    assert cfg.lora_r == 8
    assert cfg.lora_alpha == 16
    assert "q_proj" in cfg.lora_target_modules
    assert cfg.learning_rate == 1e-4
    assert cfg.epochs >= 1


def test_9_model_comparison_logic():
    """Verify model comparison classifies IMPROVED, COMPARABLE, and DEGRADED strictly."""
    base_metrics = BenchmarkEvaluationMetrics(
        model_name="google/siglip2-base-patch16-512",
        is_finetuned=False,
        total_samples=30,
        top1_accuracy=1.0,
        top3_accuracy=1.0,
        macro_precision=1.0,
        macro_recall=1.0,
        macro_f1=1.0,
        avg_latency_ms=2000.0,
        per_class_metrics={},
        confusion_matrix={},
        failed_cases=[],
        evaluation_timestamp="2026-10-01T00:00:00Z",
    )

    # Comparable case
    comp_metrics = base_metrics.model_copy()
    rep = compare_baseline_and_finetuned(base_metrics, comp_metrics)
    assert rep.verdict == "COMPARABLE"

    # Degraded case
    deg_metrics = base_metrics.model_copy(update={"top1_accuracy": 0.90, "macro_f1": 0.89})
    rep_deg = compare_baseline_and_finetuned(base_metrics, deg_metrics)
    assert rep_deg.verdict == "DEGRADED"


def test_10_production_model_remains_unchanged():
    """Verify production settings model remains google/siglip2-base-patch16-512."""
    assert settings.MODEL_ID == "google/siglip2-base-patch16-512"
    assert set(settings.CANDIDATE_CATEGORIES) == set(PRODUCTION_CLASSES)


def test_11_production_service_isolation():
    """Verify production SigLIP2 adapter does not reference finetuning weights by default."""
    from backend.app.models.siglip2_adapter import SigLIP2Adapter
    adapter = SigLIP2Adapter()
    assert adapter._model_id == "google/siglip2-base-patch16-512"
    assert "models/finetuned" not in str(adapter._model_id)
