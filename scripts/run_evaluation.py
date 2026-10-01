"""ASTRA VISION — Offline Model Performance Evaluation Runner.

Executes reproducible evaluation of the production SigLIP 2 classification model
against the held-out evaluation dataset. Generates evaluation artifacts in evaluation/.
"""

import sys
from pathlib import Path

# Ensure root and backend directory in python path
repo_root = Path(__file__).resolve().parent.parent
backend_dir = repo_root / "backend"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.evaluation.evaluator import OfflineModelEvaluator


def main():
    dataset_path = repo_root / "data" / "held_out_dataset"
    output_path = repo_root / "evaluation"

    print("==================================================")
    print("ASTRA VISION — PHASE 8A: HELD-OUT MODEL EVALUATION")
    print("==================================================")
    print(f"Dataset: {dataset_path}")
    print(f"Output:  {output_path}\n")

    evaluator = OfflineModelEvaluator(
        dataset_dir=str(dataset_path),
        output_dir=str(output_path),
    )

    print("1. Scanning dataset and building manifest...")
    manifest = evaluator.build_manifest(split_name="held_out_test")
    print(f"   Collected {len(manifest)} valid images across classes.")

    print("\n2. Executing offline evaluation with SigLIP 2 adapter...")
    summary = evaluator.run_evaluation(manifest=manifest, save_artifacts=True)

    metrics = summary["metrics"]
    perf = summary["performance"]
    unc = summary["uncertainty_analysis"]

    print("\n3. Evaluation Complete!")
    print("--------------------------------------------------")
    print(f"Overall Top-1 Accuracy: {metrics['overall_accuracy'] * 100:.2f}% ({metrics['correct_samples']}/{metrics['total_samples']})")
    print(f"Top-3 Accuracy:         {metrics['top_3_accuracy'] * 100:.2f}%")
    print(f"Macro Precision:        {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:           {metrics['macro_recall']:.4f}")
    print(f"Macro F1 Score:         {metrics['macro_f1']:.4f}")
    print(f"Total Time:             {perf['total_evaluation_time_s']:.2f} s")
    print(f"Average Latency:        {perf['average_inference_time_ms']:.2f} ms/image")
    print(f"Throughput:             {perf['throughput_fps']:.2f} images/s")
    print("--------------------------------------------------")

    print("\nPer-Class Breakdown:")
    for cls_name, vals in metrics["per_class"].items():
        print(f"  {cls_name:<20}: Prec={vals['precision']:.4f} | Rec={vals['recall']:.4f} | F1={vals['f1_score']:.4f} | Support={vals['support']}")

    print("\nUncertainty Summary:")
    print(f"  Uncertain:     {unc['total_uncertain']} ({unc['percentage_uncertain']}%) — Acc: {unc['uncertain_accuracy']*100:.1f}%")
    print(f"  Non-Uncertain: {unc['total_non_uncertain']} — Acc: {unc['non_uncertain_accuracy']*100:.1f}%")

    print(f"\nEvaluation artifacts saved to: {output_path.resolve()}")
    print("==================================================")


if __name__ == "__main__":
    main()
