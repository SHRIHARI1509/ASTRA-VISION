"""ASTRA VISION — Phase 8G Benchmark Evaluator.

Objectively evaluates vision models against the frozen 30-image held-out benchmark.
Computes Top-1, Top-3, per-class metrics, confusion matrices, and latency.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from PIL import Image
import torch

from backend.app.finetuning.reconciliation import PRODUCTION_CLASSES


class BenchmarkEvaluationMetrics(BaseModel):
    """Rigorous evaluation metrics on the held-out benchmark."""
    model_name: str
    is_finetuned: bool
    checkpoint_path: Optional[str] = None
    total_samples: int
    top1_accuracy: float
    top3_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    avg_latency_ms: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: Dict[str, Dict[str, int]]
    failed_cases: List[Dict[str, Any]]
    evaluation_timestamp: str


def evaluate_model_on_benchmark(
    model: Any,
    processor: Any,
    benchmark_manifest_path: Path,
    is_finetuned: bool = False,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    output_artifact_path: Optional[Path] = None,
    dataset_dir: Optional[Path] = None,
) -> BenchmarkEvaluationMetrics:
    """Run deterministic zero-shot or fine-tuned evaluation on the held-out benchmark."""
    benchmark_manifest_path = Path(benchmark_manifest_path)
    if not benchmark_manifest_path.exists():
        raise FileNotFoundError(f"Held-out benchmark manifest not found: {benchmark_manifest_path}")

    with open(benchmark_manifest_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    if isinstance(bench_data, list):
        images_data = bench_data
    else:
        images_data = bench_data.get("images", [])

    if dataset_dir:
        held_out_dir = Path(dataset_dir)
    else:
        held_out_dir = benchmark_manifest_path.parent / "held_out_dataset"
        if not held_out_dir.exists():
            held_out_dir = benchmark_manifest_path.parent.parent / "data" / "held_out_dataset"

    # Candidate text prompts exactly as configured in production
    candidate_labels = PRODUCTION_CLASSES
    prompt_templates = [f"a military {label.lower()} in a defence scenario" for label in candidate_labels]

    model.eval()
    model.to(device)

    # Pre-encode text queries with torch.no_grad()
    with torch.no_grad():
        text_inputs = processor(
            text=prompt_templates,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        ).to(device)

    latencies: List[float] = []
    y_true: List[str] = []
    y_pred: List[str] = []
    y_top3: List[List[str]] = []
    failed_cases: List[Dict[str, Any]] = []

    for item in images_data:
        file_name = item.get("filename") or item.get("file_name")
        true_label = item["ground_truth"]
        
        # Try resolving relative to held_out_dir or item path
        if "path" in item and (benchmark_manifest_path.parent / item["path"]).exists():
            img_path = (benchmark_manifest_path.parent / item["path"]).resolve()
        elif (held_out_dir / true_label / file_name).exists():
            img_path = (held_out_dir / true_label / file_name).resolve()
        elif (held_out_dir / file_name).exists():
            img_path = (held_out_dir / file_name).resolve()
        else:
            raise FileNotFoundError(f"Benchmark image '{file_name}' not found under {held_out_dir}")

        img = Image.open(img_path).convert("RGB")

        t0 = time.perf_counter()
        with torch.no_grad():
            img_inputs = processor(
                images=img,
                return_tensors="pt",
            ).to(device)

            # Combine inputs for forward pass
            combined_inputs = {**img_inputs, **text_inputs}
            outputs = model(**combined_inputs)

            # SigLIP similarity logits
            logits = outputs.logits_per_image[0]  # shape: [num_classes]
            # Sigmoid probabilities (SigLIP formulation)
            scores = torch.sigmoid(logits).cpu().tolist()

        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        # Ranked predictions
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top1_label = candidate_labels[ranked_indices[0]]
        top3_labels = [candidate_labels[i] for i in ranked_indices[:3]]

        y_true.append(true_label)
        y_pred.append(top1_label)
        y_top3.append(top3_labels)

        if top1_label != true_label:
            failed_cases.append({
                "file_name": file_name,
                "true_label": true_label,
                "predicted_label": top1_label,
                "top3": top3_labels,
                "scores": {lbl: round(scores[i], 4) for i, lbl in enumerate(candidate_labels)},
            })

    # Compute metrics
    total = len(y_true)
    top1_correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    top3_correct = sum(1 for yt, top3 in zip(y_true, y_top3) if yt in top3)

    top1_acc = round(top1_correct / total, 4) if total > 0 else 0.0
    top3_acc = round(top3_correct / total, 4) if total > 0 else 0.0
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    # Per-class precision, recall, F1, and confusion matrix
    cm: Dict[str, Dict[str, int]] = {c: {c2: 0 for c2 in candidate_labels} for c in candidate_labels}
    for yt, yp in zip(y_true, y_pred):
        if yt in cm and yp in cm[yt]:
            cm[yt][yp] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    precisions, recalls, f1s = [], [], []

    for c in candidate_labels:
        tp = cm[c][c]
        fp = sum(cm[other][c] for other in candidate_labels if other != c)
        fn = sum(cm[c][other] for other in candidate_labels if other != c)

        prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0

        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_prec = round(sum(precisions) / len(precisions), 4)
    macro_rec = round(sum(recalls) / len(recalls), 4)
    macro_f1 = round(sum(f1s) / len(f1s), 4)

    metrics = BenchmarkEvaluationMetrics(
        model_name="google/siglip2-base-patch16-512",
        is_finetuned=is_finetuned,
        checkpoint_path=checkpoint_path,
        total_samples=total,
        top1_accuracy=top1_acc,
        top3_accuracy=top3_acc,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        macro_f1=macro_f1,
        avg_latency_ms=avg_latency,
        per_class_metrics=per_class,
        confusion_matrix=cm,
        failed_cases=failed_cases,
        evaluation_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    if output_artifact_path:
        out_p = Path(output_artifact_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(metrics.model_dump(), f, indent=2)

    return metrics
