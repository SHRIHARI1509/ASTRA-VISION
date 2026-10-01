"""ASTRA VISION — Phase 8G Fine-Tuning and Model Adaptation Package."""

from backend.app.finetuning.reconciliation import (
    reconcile_supplied_dataset,
    ReconciledImageRecord,
    ReconciliationReport,
    PRODUCTION_CLASSES,
)
from backend.app.finetuning.balancing import (
    balance_dataset_splits,
    DatasetBalancingReport,
    SplitDatasetItem,
)
from backend.app.finetuning.evaluator import (
    evaluate_model_on_benchmark,
    BenchmarkEvaluationMetrics,
)
from backend.app.finetuning.trainer import (
    SigLIP2LoRATrainer,
    TrainingConfig,
    TrainingResult,
)
from backend.app.finetuning.comparator import (
    compare_baseline_and_finetuned,
    ModelComparisonReport,
    MetricRow,
)

__all__ = [
    "reconcile_supplied_dataset",
    "ReconciledImageRecord",
    "ReconciliationReport",
    "PRODUCTION_CLASSES",
    "balance_dataset_splits",
    "DatasetBalancingReport",
    "SplitDatasetItem",
    "evaluate_model_on_benchmark",
    "BenchmarkEvaluationMetrics",
    "SigLIP2LoRATrainer",
    "TrainingConfig",
    "TrainingResult",
    "compare_baseline_and_finetuned",
    "ModelComparisonReport",
    "MetricRow",
]
