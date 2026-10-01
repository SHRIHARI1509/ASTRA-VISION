"""ASTRA VISION — Phase 8G Baseline vs Fine-Tuned Model Comparison.

Computes exact mathematical deltas between zero-shot baseline and fine-tuned
evaluations on the untouched 30-image held-out benchmark.
Strictly categorizes verdict as IMPROVED, COMPARABLE, or DEGRADED.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from pydantic import BaseModel, Field

from backend.app.finetuning.evaluator import BenchmarkEvaluationMetrics
from backend.app.finetuning.reconciliation import PRODUCTION_CLASSES


class MetricRow(BaseModel):
    """An individual comparison metric row."""
    metric_name: str
    baseline_value: float
    finetuned_value: float
    delta: float
    interpretation: str


class ModelComparisonReport(BaseModel):
    """Factual comparison report comparing zero-shot baseline with fine-tuned model."""
    verdict: str  # 'IMPROVED', 'COMPARABLE', 'DEGRADED'
    verdict_rationale: str
    production_recommendation: str
    metrics_table: List[MetricRow]
    per_class_f1_comparison: Dict[str, Dict[str, float]]
    baseline_latency_ms: float
    finetuned_latency_ms: float
    latency_delta_ms: float


def compare_baseline_and_finetuned(
    baseline_metrics: BenchmarkEvaluationMetrics,
    finetuned_metrics: BenchmarkEvaluationMetrics,
    output_path: Optional[Path] = None,
) -> ModelComparisonReport:
    """Compare baseline and fine-tuned benchmark performance objectively."""
    metrics_table: List[MetricRow] = []

    def make_row(name: str, base_val: float, ft_val: float, higher_is_better: bool = True) -> MetricRow:
        d = round(ft_val - base_val, 4)
        if d > 0:
            interp = "Better" if higher_is_better else "Slower"
        elif d < 0:
            interp = "Lower" if higher_is_better else "Faster"
        else:
            interp = "Identical"
        return MetricRow(
            metric_name=name,
            baseline_value=base_val,
            finetuned_value=ft_val,
            delta=d,
            interpretation=interp,
        )

    metrics_table.append(make_row("Top-1 Accuracy", baseline_metrics.top1_accuracy, finetuned_metrics.top1_accuracy))
    metrics_table.append(make_row("Top-3 Accuracy", baseline_metrics.top3_accuracy, finetuned_metrics.top3_accuracy))
    metrics_table.append(make_row("Macro Precision", baseline_metrics.macro_precision, finetuned_metrics.macro_precision))
    metrics_table.append(make_row("Macro Recall", baseline_metrics.macro_recall, finetuned_metrics.macro_recall))
    metrics_table.append(make_row("Macro F1", baseline_metrics.macro_f1, finetuned_metrics.macro_f1))
    metrics_table.append(make_row("Latency (ms)", baseline_metrics.avg_latency_ms, finetuned_metrics.avg_latency_ms, higher_is_better=False))

    # Per-class F1
    per_class_f1: Dict[str, Dict[str, float]] = {}
    for c in PRODUCTION_CLASSES:
        b_f1 = baseline_metrics.per_class_metrics.get(c, {}).get("f1", 0.0)
        ft_f1 = finetuned_metrics.per_class_metrics.get(c, {}).get("f1", 0.0)
        per_class_f1[c] = {
            "baseline_f1": b_f1,
            "finetuned_f1": ft_f1,
            "delta": round(ft_f1 - b_f1, 4),
        }

    # Strict verdict evaluation
    top1_delta = finetuned_metrics.top1_accuracy - baseline_metrics.top1_accuracy
    f1_delta = finetuned_metrics.macro_f1 - baseline_metrics.macro_f1

    if top1_delta > 0 or (top1_delta == 0 and f1_delta > 0.005):
        verdict = "IMPROVED"
        rationale = f"Fine-tuned model demonstrated higher performance (Top-1 delta: {top1_delta:+.4f}, Macro F1 delta: {f1_delta:+.4f})."
        rec = "Archive fine-tuned weights in models/finetuned/ as a promotion candidate. Do NOT replace production model without human authorization."
    elif top1_delta == 0 and abs(f1_delta) <= 0.02:
        verdict = "COMPARABLE"
        rationale = f"Fine-tuned model matched the baseline performance (Top-1: {finetuned_metrics.top1_accuracy:.4f} vs {baseline_metrics.top1_accuracy:.4f})."
        rec = "Preserve existing pre-trained baseline as production classifier. Fine-tuning matched zero-shot generalization."
    else:
        verdict = "DEGRADED"
        rationale = f"Fine-tuned model degraded relative to pre-trained baseline (Top-1 delta: {top1_delta:+.4f}, Macro F1 delta: {f1_delta:+.4f})."
        rec = "Preserve existing pre-trained baseline as production classifier. Discard fine-tuned weights from production path."

    latency_delta = round(finetuned_metrics.avg_latency_ms - baseline_metrics.avg_latency_ms, 2)

    report = ModelComparisonReport(
        verdict=verdict,
        verdict_rationale=rationale,
        production_recommendation=rec,
        metrics_table=metrics_table,
        per_class_f1_comparison=per_class_f1,
        baseline_latency_ms=baseline_metrics.avg_latency_ms,
        finetuned_latency_ms=finetuned_metrics.avg_latency_ms,
        latency_delta_ms=latency_delta,
    )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)

    return report
