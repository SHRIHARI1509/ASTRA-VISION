import time
import json
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from PIL import Image

from app.core.config import settings
from app.models.base_adapter import ModelAdapter
from app.models.siglip2_adapter import SigLIP2Adapter
from app.evaluation.clip_adapter import CLIPAdapter
from app.evaluation.metrics import compute_classification_metrics

# Strict 6-class order specified by Phase 8B Section 12
COMPARISON_CLASSES = [
    "Tank",
    "Drone",
    "Fighter Aircraft",
    "Military Vehicle",
    "Ship",
    "Helicopter",
]


class ModelComparator:
    """Offline comparative benchmarking engine for evaluating Model A (SigLIP 2)

    vs Model B (CLIP) against the exact same 30-image held-out benchmark.
    Strictly isolated from production inference.
    """

    def __init__(
        self,
        dataset_dir: str = "data/held_out_dataset",
        manifest_path: str = "evaluation/manifest.json",
        output_json_path: str = "backend/evaluation/results/phase8b_model_comparison.json",
        report_md_path: str = "PHASE_8B_MODEL_COMPARISON.md",
        model_a_adapter: Optional[ModelAdapter] = None,
        model_b_adapter: Optional[ModelAdapter] = None,
        classes: Optional[List[str]] = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.manifest_path = Path(manifest_path)
        self.output_json_path = Path(output_json_path)
        self.report_md_path = Path(report_md_path)
        self.classes = classes or COMPARISON_CLASSES
        self.model_a = model_a_adapter or SigLIP2Adapter()
        self.model_b = model_b_adapter or CLIPAdapter()

    def load_manifest(self) -> List[Dict[str, Any]]:
        """Load the exact 30-image held-out test manifest established in Phase 8A."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Evaluation manifest not found at {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if len(manifest) != 30:
            raise ValueError(f"Manifest must contain exactly 30 images, got {len(manifest)}")

        # Verify class taxonomy membership
        for item in manifest:
            if item["ground_truth"] not in self.classes:
                raise ValueError(f"Unknown ground truth '{item['ground_truth']}' in manifest")

        return manifest

    def evaluate_model(
        self,
        model: ModelAdapter,
        model_name: str,
        manifest: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate a single model deterministically across all images in the manifest."""
        y_true: List[str] = []
        y_pred: List[str] = []
        top_3_preds: List[List[str]] = []
        latencies: List[float] = []
        per_sample_results: List[Dict[str, Any]] = []

        start_wall_clock = time.perf_counter()

        for item in manifest:
            # Resolve image file path
            file_path = Path(item.get("absolute_path") or (self.dataset_dir.parent / item["path"]))
            if not file_path.exists():
                file_path = self.dataset_dir / item["ground_truth"] / item["filename"]

            with Image.open(file_path) as img:
                img_rgb = img.convert("RGB")

            t0 = time.perf_counter()
            inference_result = model.predict(img_rgb, candidate_labels=self.classes)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            preds = inference_result.predictions
            top_1 = preds[0]
            top_3 = preds[:3]

            pred_label = top_1.label
            pred_score = top_1.score
            top_3_labels = [p.label for p in top_3]

            y_true.append(item["ground_truth"])
            y_pred.append(pred_label)
            top_3_preds.append(top_3_labels)

            is_correct = (pred_label == item["ground_truth"])
            per_sample_results.append({
                "id": item["id"],
                "filename": item["filename"],
                "ground_truth": item["ground_truth"],
                "predicted": pred_label,
                "score": pred_score,
                "is_correct": is_correct,
                "top_3": [{"label": p.label, "score": p.score} for p in top_3],
                "latency_ms": round(latency_ms, 2),
            })

        total_wall_clock_s = time.perf_counter() - start_wall_clock

        # Calculate standard metrics
        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            classes=self.classes,
            top_3_preds=top_3_preds,
        )

        mean_latency = round(statistics.mean(latencies), 2)
        median_latency = round(statistics.median(latencies), 2)
        fps = round(len(manifest) / total_wall_clock_s, 2) if total_wall_clock_s > 0 else 0.0

        # Build per-class detail structure
        per_class_summary = {}
        for cls_name in self.classes:
            cls_m = metrics["per_class"][cls_name]
            tp = cls_m["true_positives"]
            fn = cls_m["false_negatives"]
            fp = cls_m["false_positives"]
            support = cls_m["support"]
            per_class_summary[cls_name] = {
                "precision": cls_m["precision"],
                "recall": cls_m["recall"],
                "f1": cls_m["f1_score"],
                "support": support,
                "correct": tp,
                "incorrect": fn,
                "false_positives": fp,
            }

        return {
            "name": model_name,
            "identifier": model.model_id,
            "device": model.device,
            "top1_accuracy": metrics["overall_accuracy"],
            "top3_accuracy": metrics["top_3_accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "per_class": per_class_summary,
            "confusion_matrix": metrics["confusion_matrix"],
            "latency": {
                "mean_ms": mean_latency,
                "median_ms": median_latency,
                "total_wall_clock_s": round(total_wall_clock_s, 2),
                "throughput_fps": fps,
            },
            "per_sample": per_sample_results,
        }

    def run_comparison(self) -> Dict[str, Any]:
        """Execute fair offline comparative benchmark across Model A and Model B."""
        manifest = self.load_manifest()

        print(f"Loaded {len(manifest)} held-out images for comparison.")
        print(f"Class taxonomy ({len(self.classes)} classes): {self.classes}\n")

        print("Evaluating Model A: Google SigLIP 2 Base (google/siglip2-base-patch16-512)...")
        result_a = self.evaluate_model(self.model_a, "Google SigLIP 2 Base", manifest)

        print("Evaluating Model B: OpenAI CLIP ViT-B/32 (openai/clip-vit-base-patch32)...")
        result_b = self.evaluate_model(self.model_b, "OpenAI CLIP ViT-B/32", manifest)

        comparison_data = {
            "phase": "8B",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "name": "ASTRA VISION Held-Out Benchmark",
                "path": str(self.dataset_dir),
                "images": len(manifest),
                "classes": len(self.classes),
                "images_per_class": len(manifest) // len(self.classes),
                "taxonomy": self.classes,
                "reproducibility_seed": 42,
            },
            "models": [result_a, result_b],
        }

        # Save machine-readable JSON artifacts
        self.output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(comparison_data, f, indent=2)

        # Mirror artifact to evaluation/phase8b_model_comparison.json
        alt_path = Path("evaluation/phase8b_model_comparison.json")
        alt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(alt_path, "w", encoding="utf-8") as f:
            json.dump(comparison_data, f, indent=2)

        # Generate human-readable Markdown report
        self.generate_report(comparison_data)

        return comparison_data

    def generate_report(self, data: Dict[str, Any]) -> None:
        """Write the PHASE_8B_MODEL_COMPARISON.md report."""
        m_a = data["models"][0]
        m_b = data["models"][1]
        ds = data["dataset"]

        lines = [
            "# PHASE 8B — ASTRA VISION MODEL COMPARISON REPORT",
            "",
            "## 1. Objective",
            "This report establishes an objective, empirical, offline benchmark comparing two compatible",
            "vision-language zero-shot classification foundation models on the exact same 30-image held-out",
            "defence object recognition dataset.",
            "",
            "## 2. Dataset & Integrity",
            f"- **Dataset**: {ds['name']}",
            f"- **Total Images**: {ds['images']} (strictly identical 30-image set)",
            f"- **Classes**: {ds['classes']} ({ds['images_per_class']} images per class)",
            "- **Evaluation Split**: `held_out_test` (100% held-out test split, seed=42)",
            "- **Zero Leakage**: Neither model was trained, fine-tuned, or adapted on any test images.",
            "",
            "## 3. Evaluated Models",
            "",
            "### Model A (Production Baseline)",
            f"- **Name**: {m_a['name']}",
            f"- **Identifier**: `{m_a['identifier']}`",
            "- **Architecture**: Vision Transformer ViT-B/16 (512×512 input resolution)",
            "- **Loss Formulation**: Pairwise Sigmoid cross-entropy loss (SigLIP)",
            f"- **Device**: `{m_a['device']}`",
            "",
            "### Model B (Comparative Candidate)",
            f"- **Name**: {m_b['name']}",
            f"- **Identifier**: `{m_b['identifier']}`",
            "- **Architecture**: Vision Transformer ViT-B/32 (224×224 input resolution)",
            "- **Loss Formulation**: InfoNCE / Softmax Contrastive loss (CLIP)",
            f"- **Device**: `{m_b['device']}`",
            "- **Selection Rationale**: Industry-standard zero-shot foundation model baseline against which SigLIP was originally benchmarked in literature. Evaluates identically across the same candidate text prompts with zero fine-tuning.",
            "",
            "## 4. Prompt & Evaluation Methodology",
            "Both models evaluated using identical prompt templates per class:",
            "- `Tank` -> `a photo of a tank`",
            "- `Drone` -> `a photo of a drone`",
            "- `Fighter Aircraft` -> `a photo of a fighter aircraft`",
            "- `Military Vehicle` -> `a photo of a military vehicle`",
            "- `Ship` -> `a photo of a ship`",
            "- `Helicopter` -> `a photo of a helicopter`",
            "",
            "## 5. Overall Metrics Comparison",
            "",
            "| Metric | Model A (SigLIP 2 Base) | Model B (CLIP ViT-B/32) |",
            "|---|---:|---:|",
            f"| **Top-1 Accuracy** | {m_a['top1_accuracy'] * 100:.2f}% | {m_b['top1_accuracy'] * 100:.2f}% |",
            f"| **Top-3 Accuracy** | {m_a['top3_accuracy'] * 100:.2f}% | {m_b['top3_accuracy'] * 100:.2f}% |",
            f"| **Macro Precision** | {m_a['macro_precision']:.4f} | {m_b['macro_precision']:.4f} |",
            f"| **Macro Recall** | {m_a['macro_recall']:.4f} | {m_b['macro_recall']:.4f} |",
            f"| **Macro F1** | {m_a['macro_f1']:.4f} | {m_b['macro_f1']:.4f} |",
            f"| **Mean Latency** | {m_a['latency']['mean_ms']:.2f} ms | {m_b['latency']['mean_ms']:.2f} ms |",
            f"| **Median Latency** | {m_a['latency']['median_ms']:.2f} ms | {m_b['latency']['median_ms']:.2f} ms |",
            f"| **Throughput** | {m_a['latency']['throughput_fps']:.2f} img/s | {m_b['latency']['throughput_fps']:.2f} img/s |",
            "",
            "## 6. Per-Class Performance Breakdown",
            "",
            "| Class | Support | Model A Correct | Model B Correct | Model A F1 | Model B F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]

        for cls_name in self.classes:
            ca = m_a["per_class"][cls_name]
            cb = m_b["per_class"][cls_name]
            lines.append(
                f"| **{cls_name}** | {ca['support']} | {ca['correct']}/{ca['support']} | {cb['correct']}/{cb['support']} | {ca['f1']:.4f} | {cb['f1']:.4f} |"
            )

        lines.extend([
            "",
            "## 7. Confusion Matrices",
            "",
            f"### Model A: {m_a['name']} (`{m_a['identifier']}`)",
            "```",
            f"{'Ground Truth \\ Predicted':<25} | " + " | ".join(f"{c[:8]:<8}" for c in self.classes),
            "-" * (28 + len(self.classes) * 11),
        ])

        for idx, cls_name in enumerate(self.classes):
            row_str = " | ".join(f"{val:<8}" for val in m_a["confusion_matrix"]["counts"][idx])
            lines.append(f"{cls_name:<25} | {row_str}")

        lines.extend([
            "```",
            "",
            f"### Model B: {m_b['name']} (`{m_b['identifier']}`)",
            "```",
            f"{'Ground Truth \\ Predicted':<25} | " + " | ".join(f"{c[:8]:<8}" for c in self.classes),
            "-" * (28 + len(self.classes) * 11),
        ])

        for idx, cls_name in enumerate(self.classes):
            row_str = " | ".join(f"{val:<8}" for val in m_b["confusion_matrix"]["counts"][idx])
            lines.append(f"{cls_name:<25} | {row_str}")

        lines.extend([
            "```",
            "",
            "## 8. Latency and Compute Observations",
            f"- **SigLIP 2 Base (512×512)**: Mean CPU latency ~{m_a['latency']['mean_ms']:.0f} ms. Operates at 4× higher pixel resolution (512×512 vs 224×224), requiring greater compute per forward pass.",
            f"- **CLIP ViT-B/32 (224×224)**: Mean CPU latency ~{m_b['latency']['mean_ms']:.0f} ms. Operates with larger patch size (patch32) and smaller resolution (224×224), delivering higher throughput on CPU at the cost of fine spatial detail.",
            "",
            "## 9. Limitations",
            "- **Sample Size**: Benchmark consists of $N=30$ curated reconnaissance images. Accuracy metrics are specific to this test partition.",
            "- **Uncertainty Metrics Omission**: SigLIP uses pairwise sigmoid activation scores, whereas CLIP uses softmax over contrastive cosine similarity logits. Because the mathematical ranges and calibration profiles differ, applying SigLIP heuristic thresholds to CLIP would be scientifically unsound. Uncertainty counts were excluded from the formal comparison table as specified.",
            "- **No Ensembling**: Per specification, models were evaluated strictly independently. Predictions were never averaged, voted, or combined.",
            "",
            "## 10. Reproducibility Instructions",
            "To re-run this exact comparison independently:",
            "```powershell",
            ".\\venv\\Scripts\\python scripts/run_model_comparison.py",
            "```",
            "Machine-readable outputs are stored in `backend/evaluation/results/phase8b_model_comparison.json`.",
            "",
            "## 11. Production-Isolation Confirmation",
            "- Production inference service `/api/classify` remains completely untouched.",
            "- Production model remains `google/siglip2-base-patch16-512`.",
            "- Frontend classification UI remains completely untouched.",
            "- Zero production endpoints or ensembling mechanisms were introduced.",
        ])

        with open(self.report_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
