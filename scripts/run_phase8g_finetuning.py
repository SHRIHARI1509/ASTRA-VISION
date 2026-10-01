"""ASTRA VISION — Phase 8G Fine-Tuning and Evaluation Pipeline Runner.

Executes the complete, end-to-end scientific workflow:
1. Human-reviewed taxonomy reconciliation (60 ambiguous images).
2. Deterministic Train/Val splitting with 0% held-out leakage.
3. Frozen zero-shot baseline evaluation on the 30-image benchmark.
4. Parameter-efficient LoRA fine-tuning with validation tracking.
5. Best-checkpoint selection based exclusively on validation metrics.
6. Final held-out evaluation on the sacred 30-image benchmark.
7. Factual baseline vs fine-tuned delta comparison and verdict.
"""

import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import psutil
from transformers import AutoProcessor, AutoModel
from peft import PeftModel

from backend.app.finetuning.reconciliation import reconcile_supplied_dataset
from backend.app.finetuning.balancing import balance_dataset_splits
from backend.app.finetuning.evaluator import evaluate_model_on_benchmark
from backend.app.finetuning.trainer import SigLIP2LoRATrainer, TrainingConfig
from backend.app.finetuning.comparator import compare_baseline_and_finetuned

DATASET_ROOT = PROJECT_ROOT / "data" / "supplied_dataset"
HELD_OUT_MANIFEST = PROJECT_ROOT / "evaluation" / "manifest.json"
HELD_OUT_DATASET_DIR = PROJECT_ROOT / "data" / "held_out_dataset"
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "finetuned"


def run_phase8g_pipeline():
    print("=" * 80)
    print("ASTRA VISION — PHASE 8G: PRODUCTION-ALIGNED SIGLIP 2 FINE-TUNING PIPELINE")
    print("=" * 80)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # 0. Hardware Telemetry
    print("\n[RESOURCE & HARDWARE DISCOVERY]")
    cuda_avail = torch.cuda.is_available()
    device = "cuda" if cuda_avail else "cpu"
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  Execution Device: {device.upper()}")
    print(f"  CPU Cores: {os.cpu_count()}")
    print(f"  Total RAM: {round(psutil.virtual_memory().total / (1024**3), 2)} GB")
    print(f"  Available RAM: {round(psutil.virtual_memory().available / (1024**3), 2)} GB")

    # Step 1: Taxonomy Reconciliation
    print("\n" + "-" * 70)
    print("[STEP 1: TAXONOMY RECONCILIATION]")
    reconciliation_manifest = EXPERIMENTS_DIR / "reconciliation_report.json"
    rec_report = reconcile_supplied_dataset(
        dataset_root=DATASET_ROOT,
        output_manifest_path=reconciliation_manifest,
    )
    print(f"  Total Images Reviewed: {rec_report.total_images_reviewed}")
    print(f"  Trainable 6-Class Images: {rec_report.trainable_images_count}")
    print(f"  Excluded Out-of-Taxonomy: {rec_report.excluded_images_count}")
    print(f"  Unresolved Labels: {rec_report.unresolved_count} (Must be 0)")
    print("  Reviewed 6-Class Distribution:")
    for cls_name, cnt in sorted(rec_report.class_distribution.items()):
        print(f"    - {cls_name:18s}: {cnt:2d} images")

    # Step 2 & 3: Balancing & Stratified Train/Val Split
    print("\n" + "-" * 70)
    print("[STEP 2 & 3: DATASET BALANCING & DETERMINISTIC SPLIT]")
    balancing_report = balance_dataset_splits(
        reconciliation_report=rec_report,
        dataset_root=DATASET_ROOT,
        splits_output_dir=EXPERIMENTS_DIR,
        held_out_manifest_path=HELD_OUT_MANIFEST,
        seed=42,
        train_ratio=0.8,
    )
    print(f"  Split Strategy: {int(balancing_report.train_ratio * 100)}% Train / {int(balancing_report.val_ratio * 100)}% Validation")
    print(f"  Train Set Size: {balancing_report.train_count} images")
    print(f"  Validation Set Size: {balancing_report.val_count} images")
    print(f"  Held-Out Leakage Count: {balancing_report.held_out_leakage_count} (Must be 0)")
    print(f"  Benchmark Integrity Protected: {'YES' if not balancing_report.held_out_leakage_detected else 'NO'}")
    print("  Per-Class Split Breakdown:")
    for cls_name in sorted(balancing_report.train_distribution.keys()):
        t_cnt = balancing_report.train_distribution[cls_name]
        v_cnt = balancing_report.val_distribution[cls_name]
        print(f"    - {cls_name:18s}: {t_cnt:2d} train / {v_cnt:2d} val (Weight: {balancing_report.class_weights[cls_name]:.2f})")

    train_manifest = Path(balancing_report.manifest_paths["train_manifest"])
    val_manifest = Path(balancing_report.manifest_paths["val_manifest"])

    # Step 4: Record Zero-Shot Baseline
    print("\n" + "-" * 70)
    print("[STEP 4: FROZEN ZERO-SHOT BASELINE EVALUATION]")
    baseline_artifact = EXPERIMENTS_DIR / "baseline_metrics.json"
    print("  Loading pre-trained base model 'google/siglip2-base-patch16-512'...")
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-512")
    base_model = AutoModel.from_pretrained("google/siglip2-base-patch16-512")

    print(f"  Evaluating zero-shot baseline on {HELD_OUT_MANIFEST}...")
    baseline_metrics = evaluate_model_on_benchmark(
        model=base_model,
        processor=processor,
        benchmark_manifest_path=HELD_OUT_MANIFEST,
        dataset_dir=HELD_OUT_DATASET_DIR,
        is_finetuned=False,
        device=device,
        output_artifact_path=baseline_artifact,
    )
    print(f"  Baseline Top-1 Accuracy: {baseline_metrics.top1_accuracy * 100:.1f}%")
    print(f"  Baseline Top-3 Accuracy: {baseline_metrics.top3_accuracy * 100:.1f}%")
    print(f"  Baseline Macro F1: {baseline_metrics.macro_f1:.4f}")
    print(f"  Baseline Latency: {baseline_metrics.avg_latency_ms:.1f} ms/image")
    print(f"  Saved baseline metrics to: {baseline_artifact}")

    # Free base model from RAM before training
    del base_model
    import gc
    gc.collect()

    # Step 5 & 6 & 7: Fine-Tuning Pipeline
    print("\n" + "-" * 70)
    print("[STEP 5, 6 & 7: LoRA FINE-TUNING PIPELINE]")
    train_config = TrainingConfig(
        model_identifier="google/siglip2-base-patch16-512",
        processor_identifier="google/siglip2-base-patch16-512",
        learning_rate=1e-4,
        batch_size=4,
        gradient_accumulation_steps=2,
        epochs=3,
        warmup_ratio=0.1,
        weight_decay=0.01,
        random_seed=42,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        device=device,
    )
    print(f"  LoRA Rank (r): {train_config.lora_r}, Alpha: {train_config.lora_alpha}")
    print(f"  Batch Size: {train_config.batch_size}, Grad Accum: {train_config.gradient_accumulation_steps}")
    print(f"  Epochs: {train_config.epochs}, Learning Rate: {train_config.learning_rate}")

    trainer = SigLIP2LoRATrainer(config=train_config, checkpoint_root=CHECKPOINTS_DIR)
    summary_file = CHECKPOINTS_DIR / "training_summary.json"
    checkpoint_file = CHECKPOINTS_DIR / "checkpoint-best" / "adapter_model.safetensors"

    if checkpoint_file.exists() and summary_file.exists():
        print(f"  Found verified existing fine-tuned checkpoint at {checkpoint_file}")
        with open(summary_file, "r", encoding="utf-8") as f:
            train_result_data = json.load(f)
        from backend.app.finetuning.trainer import TrainingResult
        train_result = TrainingResult(**train_result_data)
        print("  Reused existing validated training run metadata.")
    else:
        train_result = trainer.train(
            train_manifest_path=train_manifest,
            val_manifest_path=val_manifest,
            class_weights=balancing_report.class_weights,
        )

    print("\n  Training Completed:")
    print(f"  Total Duration: {train_result.total_training_duration_seconds:.1f} seconds")
    print(f"  Best Epoch: {train_result.best_epoch}")
    print(f"  Best Val Top-1 Accuracy: {train_result.best_val_top1_acc * 100:.1f}%")
    print(f"  Best Val Loss: {train_result.best_val_loss:.4f}")
    for ep in train_result.history:
        star = " [*] BEST" if ep.is_best_checkpoint else ""
        print(f"    Epoch {ep.epoch}: Train Loss={ep.train_loss:.4f} (Acc={ep.train_top1_acc*100:.1f}%) | Val Loss={ep.val_loss:.4f} (Acc={ep.val_top1_acc*100:.1f}%){star}")

    # Step 8: Final Held-Out Evaluation on Best Checkpoint
    print("\n" + "-" * 70)
    print("[STEP 8: FINAL HELD-OUT EVALUATION ON UNTOUCHED BENCHMARK]")
    best_checkpoint_path = Path(train_result.checkpoint_dir)
    print(f"  Loading best checkpoint from: {best_checkpoint_path}...")

    ft_base_model = AutoModel.from_pretrained("google/siglip2-base-patch16-512")
    ft_model = PeftModel.from_pretrained(ft_base_model, str(best_checkpoint_path))
    ft_processor = AutoProcessor.from_pretrained(str(best_checkpoint_path))

    finetuned_artifact = EXPERIMENTS_DIR / "finetuned_benchmark_metrics.json"
    print(f"  Evaluating fine-tuned model against frozen 30-image benchmark...")
    finetuned_metrics = evaluate_model_on_benchmark(
        model=ft_model,
        processor=ft_processor,
        benchmark_manifest_path=HELD_OUT_MANIFEST,
        dataset_dir=HELD_OUT_DATASET_DIR,
        is_finetuned=True,
        checkpoint_path=str(best_checkpoint_path),
        device=device,
        output_artifact_path=finetuned_artifact,
    )
    print(f"  Fine-Tuned Top-1 Accuracy: {finetuned_metrics.top1_accuracy * 100:.1f}%")
    print(f"  Fine-Tuned Top-3 Accuracy: {finetuned_metrics.top3_accuracy * 100:.1f}%")
    print(f"  Fine-Tuned Macro F1: {finetuned_metrics.macro_f1:.4f}")
    print(f"  Fine-Tuned Latency: {finetuned_metrics.avg_latency_ms:.1f} ms/image")
    print(f"  Saved fine-tuned benchmark metrics to: {finetuned_artifact}")

    # Step 9: Factual Comparison & Verdict
    print("\n" + "-" * 70)
    print("[STEP 9: FACTUAL MODEL COMPARISON & VERDICT]")
    comparison_artifact = EXPERIMENTS_DIR / "model_comparison.json"
    comparison = compare_baseline_and_finetuned(
        baseline_metrics=baseline_metrics,
        finetuned_metrics=finetuned_metrics,
        output_path=comparison_artifact,
    )

    print("\n  ┌─────────────────────┬──────────┬────────────┬──────────┬──────────────┐")
    print("  │ Metric              │ Baseline │ Fine-Tuned │ Delta    │ Outcome      │")
    print("  ├─────────────────────┼──────────┼────────────┼──────────┼──────────────┤")
    for row in comparison.metrics_table:
        print(f"  │ {row.metric_name:19s} │ {row.baseline_value:8.4f} │ {row.finetuned_value:10.4f} │ {row.delta:+8.4f} │ {row.interpretation:12s} │")
    print("  └─────────────────────┴──────────┴────────────┴──────────┴──────────────┘")

    print(f"\n  Final Verdict: {comparison.verdict}")
    print(f"  Rationale: {comparison.verdict_rationale}")
    print(f"  Production Recommendation: {comparison.production_recommendation}")
    print("\n" + "=" * 80)
    print("PHASE 8G PIPELINE EXECUTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_phase8g_pipeline()
