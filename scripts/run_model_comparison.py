"""ASTRA VISION — Phase 8B Model Comparison Benchmark Runner.

Executes an objective offline comparative benchmark between Model A (SigLIP 2)
and Model B (OpenAI CLIP) against the exact same 30-image held-out test set.
Generates:
  - backend/evaluation/results/phase8b_model_comparison.json
  - PHASE_8B_MODEL_COMPARISON.md
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
backend_dir = repo_root / "backend"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.evaluation.comparator import ModelComparator


def main():
    print("==================================================")
    print("ASTRA VISION — PHASE 8B: MODEL COMPARISON BENCHMARK")
    print("==================================================")

    comparator = ModelComparator(
        dataset_dir=str(repo_root / "data" / "held_out_dataset"),
        manifest_path=str(repo_root / "evaluation" / "manifest.json"),
        output_json_path=str(repo_root / "backend" / "evaluation" / "results" / "phase8b_model_comparison.json"),
        report_md_path=str(repo_root / "PHASE_8B_MODEL_COMPARISON.md"),
    )

    data = comparator.run_comparison()

    m_a = data["models"][0]
    m_b = data["models"][1]

    print("\n==================================================")
    print("PHASE 8B COMPARISON RESULTS")
    print("==================================================")
    print(f"{'Metric':<25} | {'Model A (SigLIP 2)':<20} | {'Model B (CLIP)':<20}")
    print("-" * 72)
    print(f"{'Top-1 Accuracy':<25} | {m_a['top1_accuracy']*100:<19.2f}% | {m_b['top1_accuracy']*100:<19.2f}%")
    print(f"{'Top-3 Accuracy':<25} | {m_a['top3_accuracy']*100:<19.2f}% | {m_b['top3_accuracy']*100:<19.2f}%")
    print(f"{'Macro Precision':<25} | {m_a['macro_precision']:<20.4f} | {m_b['macro_precision']:<20.4f}")
    print(f"{'Macro Recall':<25} | {m_a['macro_recall']:<20.4f} | {m_b['macro_recall']:<20.4f}")
    print(f"{'Macro F1':<25} | {m_a['macro_f1']:<20.4f} | {m_b['macro_f1']:<20.4f}")
    print(f"{'Mean Latency (ms)':<25} | {m_a['latency']['mean_ms']:<20.2f} | {m_b['latency']['mean_ms']:<20.2f}")
    print(f"{'Throughput (img/s)':<25} | {m_a['latency']['throughput_fps']:<20.2f} | {m_b['latency']['throughput_fps']:<20.2f}")
    print("==================================================")
    print(f"Machine-readable output: {comparator.output_json_path}")
    print(f"Human-readable report:   {comparator.report_md_path}")
    print("==================================================")


if __name__ == "__main__":
    main()
