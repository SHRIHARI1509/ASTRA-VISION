"""Unit tests for Phase 8F Dataset Splitting and Taxonomy Guardrails."""

import json
import pytest
from pathlib import Path

from backend.app.preparation.taxonomy_guardrails import (
    PRODUCTION_CLASSES,
    SUPPLIED_CLASSES,
    TAXONOMY_RULES,
    audit_taxonomy_alignment,
)
from backend.app.preparation.dataset_splitter import (
    DatasetSplitter,
    generate_stratified_split,
)


def test_production_and_supplied_taxonomy_definitions():
    assert len(PRODUCTION_CLASSES) == 6
    assert set(PRODUCTION_CLASSES) == {"Tank", "Military Vehicle", "Fighter Aircraft", "Helicopter", "Ship", "Drone"}

    assert len(SUPPLIED_CLASSES) == 5
    assert set(SUPPLIED_CLASSES) == {"aircraft", "drone", "helicopter", "military-vehicle", "naval"}


def test_taxonomy_guardrails_protect_ambiguous_categories():
    # Direct mappings allowed
    assert TAXONOMY_RULES["drone"].direct_mapping_allowed is True
    assert TAXONOMY_RULES["drone"].target_production_class == "Drone"

    assert TAXONOMY_RULES["helicopter"].direct_mapping_allowed is True
    assert TAXONOMY_RULES["helicopter"].target_production_class == "Helicopter"

    assert TAXONOMY_RULES["naval"].direct_mapping_allowed is True
    assert TAXONOMY_RULES["naval"].target_production_class == "Ship"

    # Ambiguous mappings strictly prohibited from automatic mapping
    assert TAXONOMY_RULES["military-vehicle"].direct_mapping_allowed is False
    assert TAXONOMY_RULES["military-vehicle"].is_ambiguous is True
    assert TAXONOMY_RULES["military-vehicle"].target_production_class is None

    assert TAXONOMY_RULES["aircraft"].direct_mapping_allowed is False
    assert TAXONOMY_RULES["aircraft"].is_ambiguous is True
    assert TAXONOMY_RULES["aircraft"].target_production_class is None


def test_taxonomy_audit_supplied_labels():
    labels_csv = Path("data/supplied_dataset/labels.csv")
    assert labels_csv.exists()

    result = audit_taxonomy_alignment(labels_csv)
    assert result.total_images == 150
    assert result.unambiguous_count == 90  # 30 drone + 30 helicopter + 30 naval
    assert result.ambiguous_count == 60    # 30 military-vehicle + 30 aircraft
    assert len(result.flagged_review_items) == 60
    assert result.guardrail_status == "ACTIVE_PROTECTION"


def test_dataset_splitter_80_20_stratification():
    dataset_root = Path("data/supplied_dataset")
    held_out_manifest = Path("data/held_out_dataset/manifest.json")
    output_dir = Path("data/splits")

    splitter = DatasetSplitter(
        dataset_root=dataset_root,
        held_out_manifest_path=held_out_manifest,
        seed=42,
    )
    summary = splitter.generate_split(output_dir=output_dir, train_ratio=0.8)

    assert summary.total_images == 150
    assert summary.train_count == 120
    assert summary.val_count == 30
    assert summary.held_out_leakage_count == 0
    assert summary.benchmark_protected is True

    # Check exact balance per category
    for cat in SUPPLIED_CLASSES:
        dist = summary.category_distribution[cat]
        assert dist["train"] == 24
        assert dist["val"] == 6
        assert dist["total"] == 30


def test_manifest_files_contain_valid_json():
    train_manifest = Path("data/splits/train_manifest.json")
    val_manifest = Path("data/splits/val_manifest.json")
    summary_file = Path("data/splits/split_summary.json")

    assert train_manifest.exists()
    assert val_manifest.exists()
    assert summary_file.exists()

    with open(train_manifest, "r", encoding="utf-8") as f:
        train_data = json.load(f)
        assert train_data["split_name"] == "train"
        assert train_data["count"] == 120
        assert len(train_data["items"]) == 120

    with open(val_manifest, "r", encoding="utf-8") as f:
        val_data = json.load(f)
        assert val_data["split_name"] == "val"
        assert val_data["count"] == 30
        assert len(val_data["items"]) == 30


def test_dataset_splitter_determinism():
    dataset_root = Path("data/supplied_dataset")
    output_dir = Path("data/splits")

    summary1 = generate_stratified_split(dataset_root, output_dir, seed=42)
    summary2 = generate_stratified_split(dataset_root, output_dir, seed=42)

    assert summary1.train_count == summary2.train_count
    assert summary1.val_count == summary2.val_count
    assert summary1.category_distribution == summary2.category_distribution
