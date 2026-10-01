import os
import time
import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image

from app.core.config import settings
from app.services.classification_service import ClassificationService, classification_service
from app.evaluation.metrics import compute_classification_metrics


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class OfflineModelEvaluator:
    """Offline evaluation engine for benchmarking the production SigLIP 2 classification model

    against held-out datasets without modifying production pipeline or endpoints.
    """

    def __init__(
        self,
        dataset_dir: str,
        output_dir: str = "evaluation",
        service: Optional[ClassificationService] = None,
        candidate_labels: Optional[List[str]] = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.service = service or classification_service
        self.classes = candidate_labels or settings.CANDIDATE_CATEGORIES

    def build_manifest(self, split_name: str = "held_out_test") -> List[Dict[str, Any]]:
        """Scan dataset directory and generate a reproducible manifest.
        Expects class-labeled subdirectories.
        """
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        manifest: List[Dict[str, Any]] = []
        sample_idx = 1

        for class_dir in sorted(self.dataset_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in self.classes:
                continue

            for file_path in sorted(class_dir.iterdir()):
                if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        img_format = img.format or file_path.suffix.lstrip(".").upper()
                except Exception:
                    continue

                rel_path = file_path.relative_to(self.dataset_dir.parent).as_posix()
                manifest.append({
                    "id": f"sample_{sample_idx:04d}",
                    "path": rel_path,
                    "absolute_path": str(file_path.resolve()),
                    "filename": file_path.name,
                    "ground_truth": class_name,
                    "split": split_name,
                    "width": width,
                    "height": height,
                    "format": img_format,
                    "size_bytes": file_path.stat().st_size,
                })
                sample_idx += 1

        return manifest

    def run_evaluation(
        self,
        manifest: Optional[List[Dict[str, Any]]] = None,
        save_artifacts: bool = True,
    ) -> Dict[str, Any]:
        """Execute evaluation over all samples in manifest and return structured evaluation results."""
        samples = manifest if manifest is not None else self.build_manifest()
        total_samples = len(samples)

        results: List[Dict[str, Any]] = []
        misclassifications: List[Dict[str, Any]] = []

        y_true: List[str] = []
        y_pred: List[str] = []
        top_3_preds: List[List[str]] = []

        uncertain_samples: List[Dict[str, Any]] = []
        non_uncertain_samples: List[Dict[str, Any]] = []

        total_inference_time_ms = 0.0
        start_eval_time = time.perf_counter()

        for item in samples:
            image_bytes = Path(item["absolute_path"]).read_bytes()
            ground_truth = item["ground_truth"]

            t0 = time.perf_counter()
            response = self.service.classify_image(
                file_bytes=image_bytes,
                filename=item["filename"],
                candidate_labels=self.classes,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_inference_time_ms += elapsed_ms

            pred = response["prediction"]
            pred_label = pred["label"]
            pred_score = pred["score"]
            top_3 = response.get("top_3", [])
            top_3_labels = [c["label"] for c in top_3]
            uncertainty = response.get("uncertainty", {})

            y_true.append(ground_truth)
            y_pred.append(pred_label)
            top_3_preds.append(top_3_labels)

            is_correct = (pred_label == ground_truth)
            is_top3_correct = (ground_truth in top_3_labels)
            is_uncertain = bool(uncertainty.get("is_uncertain", False))

            sample_result = {
                "id": item["id"],
                "filename": item["filename"],
                "ground_truth": ground_truth,
                "predicted_label": pred_label,
                "model_score": pred_score,
                "is_correct": is_correct,
                "is_top3_correct": is_top3_correct,
                "top_3": top_3,
                "uncertainty": uncertainty,
                "inference_time_ms": round(elapsed_ms, 2),
            }
            results.append(sample_result)

            if is_uncertain:
                uncertain_samples.append(sample_result)
            else:
                non_uncertain_samples.append(sample_result)

            if not is_correct:
                candidates = response.get("candidates", [])
                rank_2_label = candidates[1]["label"] if len(candidates) > 1 else None
                rank_2_score = candidates[1]["score"] if len(candidates) > 1 else None
                score_margin = round(pred_score - rank_2_score, 4) if rank_2_score is not None else None

                misclassifications.append({
                    "id": item["id"],
                    "filename": item["filename"],
                    "ground_truth": ground_truth,
                    "predicted_label": pred_label,
                    "model_score": pred_score,
                    "rank_2_label": rank_2_label,
                    "rank_2_score": rank_2_score,
                    "score_margin": score_margin,
                    "uncertainty_reason": uncertainty.get("reason"),
                })

        total_eval_time_s = time.perf_counter() - start_eval_time

        # Compute standard classification metrics
        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            classes=self.classes,
            top_3_preds=top_3_preds,
        )

        # Descriptive Uncertainty Analysis (NO threshold tuning!)
        num_uncertain = len(uncertain_samples)
        num_non_uncertain = len(non_uncertain_samples)
        pct_uncertain = round((num_uncertain / total_samples) * 100.0, 2) if total_samples > 0 else 0.0

        uncertain_correct = sum(1 for s in uncertain_samples if s["is_correct"])
        non_uncertain_correct = sum(1 for s in non_uncertain_samples if s["is_correct"])

        uncertain_accuracy = (
            round(uncertain_correct / num_uncertain, 4) if num_uncertain > 0 else 0.0
        )
        non_uncertain_accuracy = (
            round(non_uncertain_correct / num_non_uncertain, 4) if num_non_uncertain > 0 else 0.0
        )

        uncertainty_analysis = {
            "total_uncertain": num_uncertain,
            "total_non_uncertain": num_non_uncertain,
            "percentage_uncertain": pct_uncertain,
            "uncertain_accuracy": uncertain_accuracy,
            "non_uncertain_accuracy": non_uncertain_accuracy,
            "score_threshold": settings.UNCERTAINTY_SCORE_THRESHOLD,
            "margin_threshold": settings.UNCERTAINTY_MARGIN_THRESHOLD,
            "note": "Descriptive heuristic evaluation only; no threshold tuning performed.",
        }

        # Performance measurements
        avg_inf_ms = round(total_inference_time_ms / total_samples, 2) if total_samples > 0 else 0.0
        fps = round(total_samples / total_eval_time_s, 2) if total_eval_time_s > 0 else 0.0

        performance_report = {
            "total_evaluation_time_s": round(total_eval_time_s, 2),
            "average_inference_time_ms": avg_inf_ms,
            "throughput_fps": fps,
            "device": settings.DEVICE,
            "model_id": settings.MODEL_ID,
        }

        eval_summary = {
            "metadata": {
                "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
                "model_id": settings.MODEL_ID,
                "dataset_directory": str(self.dataset_dir),
                "total_images": total_samples,
                "classes": self.classes,
                "reproducibility_seed": 42,
            },
            "metrics": metrics,
            "uncertainty_analysis": uncertainty_analysis,
            "performance": performance_report,
            "misclassifications": misclassifications,
            "results": results,
        }

        if save_artifacts:
            self._save_artifacts(samples, eval_summary)

        return eval_summary

    def _save_artifacts(self, manifest: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        """Serialize evaluation outputs into the dedicated evaluation output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. manifest.json
        manifest_clean = [
            {k: v for k, v in item.items() if k != "absolute_path"}
            for item in manifest
        ]
        with open(self.output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_clean, f, indent=2)

        # 2. results.json
        with open(self.output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(summary["results"], f, indent=2)

        # 3. metrics.json
        metrics_payload = {
            "metadata": summary["metadata"],
            "metrics": summary["metrics"],
            "uncertainty_analysis": summary["uncertainty_analysis"],
            "performance": summary["performance"],
        }
        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

        # 4. misclassifications.json
        with open(self.output_dir / "misclassifications.json", "w", encoding="utf-8") as f:
            json.dump(summary["misclassifications"], f, indent=2)

        # 5. confusion_matrix.csv
        cm = summary["metrics"]["confusion_matrix"]
        classes = cm["classes"]
        counts = cm["counts"]
        with open(self.output_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Ground Truth \\ Predicted"] + classes)
            for row_idx, cls_name in enumerate(classes):
                writer.writerow([cls_name] + counts[row_idx])

        # 6. README.md
        self._write_readme(summary)

    def _write_readme(self, summary: Dict[str, Any]) -> None:
        m = summary["metrics"]
        perf = summary["performance"]
        unc = summary["uncertainty_analysis"]
        meta = summary["metadata"]

        lines = [
            "# ASTRA VISION — Held-Out Model Performance Evaluation Report",
            "",
            "## 1. Overview",
            f"- **Evaluated Model**: `{meta['model_id']}`",
            f"- **Dataset Path**: `{meta['dataset_directory']}`",
            f"- **Timestamp**: `{meta['evaluation_timestamp']}`",
            f"- **Total Evaluated Images**: {meta['total_images']}",
            f"- **Device**: `{perf['device']}`",
            "",
            "## 2. Key Metrics Summary",
            f"- **Top-1 Overall Accuracy**: {m['overall_accuracy'] * 100.0:.2f}%",
            f"- **Top-3 Secondary Accuracy**: {m['top_3_accuracy'] * 100.0:.2f}%",
            f"- **Macro Precision**: {m['macro_precision']:.4f}",
            f"- **Macro Recall**: {m['macro_recall']:.4f}",
            f"- **Macro F1 Score**: {m['macro_f1']:.4f}",
            "",
            "## 3. Per-Class Performance",
            "| Class | Precision | Recall | F1 Score | Support |",
            "|---|---|---|---|---|",
        ]

        for cls_name, vals in m["per_class"].items():
            lines.append(
                f"| {cls_name} | {vals['precision']:.4f} | {vals['recall']:.4f} | "
                f"{vals['f1_score']:.4f} | {vals['support']} |"
            )

        lines.extend([
            "",
            "## 4. Confusion Matrix (Counts)",
            "```",
            f"{'Ground Truth \\ Predicted':<25} | " + " | ".join(f"{c[:8]:<8}" for c in m["confusion_matrix"]["classes"]),
            "-" * (28 + len(m["confusion_matrix"]["classes"]) * 11),
        ])

        for idx, cls_name in enumerate(m["confusion_matrix"]["classes"]):
            row_str = " | ".join(f"{val:<8}" for val in m["confusion_matrix"]["counts"][idx])
            lines.append(f"{cls_name:<25} | {row_str}")

        lines.extend([
            "```",
            "",
            "## 5. Uncertainty Analysis (Descriptive)",
            f"- **Uncertain Predictions**: {unc['total_uncertain']} ({unc['percentage_uncertain']}%)",
            f"- **Non-Uncertain Predictions**: {unc['total_non_uncertain']}",
            f"- **Accuracy among Uncertain**: {unc['uncertain_accuracy'] * 100.0:.2f}%",
            f"- **Accuracy among Non-Uncertain**: {unc['non_uncertain_accuracy'] * 100.0:.2f}%",
            "- *Note: Descriptive analysis only; operating thresholds were not tuned.*",
            "",
            "## 6. Inference Performance",
            f"- **Total Time**: {perf['total_evaluation_time_s']} s",
            f"- **Average Latency**: {perf['average_inference_time_ms']} ms/image",
            f"- **Throughput**: {perf['throughput_fps']} images/second",
            "",
            "## 7. Failure Analysis",
            f"- Total misclassified samples: {len(summary['misclassifications'])}",
            "See `misclassifications.json` for details on each misclassified instance.",
            "",
        ])

        with open(self.output_dir / "README.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
