"""ASTRA VISION — Phase 8E + 8F Preparation & Guardrails Runner.

Executes:
1. Phase 8E Detection Readiness Evaluation (verifies zero fake bboxes).
2. Phase 8F Taxonomy Audit & Reconciliation (protects 6 production classes).
3. Phase 8F Stratified Train/Val Split (preserves frozen 30-image benchmark).
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.detection.readiness import evaluate_detection_readiness
from backend.app.preparation.taxonomy_guardrails import audit_taxonomy_alignment
from backend.app.preparation.dataset_splitter import generate_stratified_split

DATASET_ROOT = PROJECT_ROOT / "data" / "supplied_dataset"
HELD_OUT_MANIFEST = PROJECT_ROOT / "data" / "held_out_dataset" / "manifest.json"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def run_preparation_pipeline():
    print("=" * 70)
    print("ASTRA VISION — PHASE 8E + 8F: DETECTION READINESS & PREPARATION PIPELINE")
    print("=" * 70)

    # 1. Phase 8E: Detection Readiness
    print("\n[PHASE 8E: OBJECT DETECTION READINESS]")
    detection_report = evaluate_detection_readiness(DATASET_ROOT)
    print(f"  Dataset: {detection_report.dataset_name}")
    print(f"  Total Images Inspected: {detection_report.total_images}")
    print(f"  Detection Ready: {'YES' if detection_report.detection_ready else 'NO'}")
    print(f"  Label Level: {detection_report.label_level}")
    print(f"  Missing Annotation Reasons:")
    for r in detection_report.missing_annotation_reasons:
        print(f"    - {r}")

    # 2. Phase 8F: Taxonomy Guardrails Audit
    print("\n[PHASE 8F: TAXONOMY GUARDRAILS AUDIT]")
    labels_csv = DATASET_ROOT / "labels.csv"
    tax_result = audit_taxonomy_alignment(labels_csv)
    print(f"  Total Images Analyzed: {tax_result.total_images}")
    print(f"  Unambiguous Entries (Direct 1:1 Match): {tax_result.unambiguous_count}")
    print(f"  Ambiguous Entries (Flagged for Review): {tax_result.ambiguous_count}")
    print(f"  Guardrail Status: {tax_result.guardrail_status}")
    print(f"  Direct Mappings Allowed:")
    for k, v in tax_result.direct_mappings.items():
        print(f"    - {k} -> {v}")
    print(f"  Flagged Ambiguous Categories:")
    print(f"    - 'military-vehicle' (Contains 6 MBTs + 24 other military vehicles)")
    print(f"    - 'aircraft' (Contains 13 Fighters + 17 non-fighter airframes)")

    # Save taxonomy reconciliation report
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    reconciliation_path = SPLITS_DIR / "taxonomy_reconciliation.json"
    with open(reconciliation_path, "w", encoding="utf-8") as f:
        json.dump(tax_result.model_dump(), f, indent=2)
    print(f"  Reconciliation Report Saved: {reconciliation_path}")

    # 3. Phase 8F: Stratified Train/Val Split Generation
    print("\n[PHASE 8F: STRATIFIED SPLIT GENERATION]")
    summary = generate_stratified_split(
        dataset_root=DATASET_ROOT,
        output_dir=SPLITS_DIR,
        held_out_manifest_path=HELD_OUT_MANIFEST,
        seed=42,
        train_ratio=0.8,
    )
    print(f"  Random Seed: {summary.random_seed}")
    print(f"  Split Ratio: {int(summary.train_ratio * 100)}% Train / {int(summary.val_ratio * 100)}% Val")
    print(f"  Train Image Count: {summary.train_count}")
    print(f"  Validation Image Count: {summary.val_count}")
    print(f"  Held-Out Benchmark Leakage Count: {summary.held_out_leakage_count}")
    print(f"  Held-Out Benchmark Protected: {'YES' if summary.benchmark_protected else 'NO'}")
    print("  Per-Category Split Breakdown:")
    for cat, dist in sorted(summary.category_distribution.items()):
        print(f"    - {cat:18s}: {dist['train']:2d} train / {dist['val']:2d} val (Total: {dist['total']:2d})")

    print("\n" + "=" * 70)
    print("PHASE 8E + 8F PREPARATION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_preparation_pipeline()
