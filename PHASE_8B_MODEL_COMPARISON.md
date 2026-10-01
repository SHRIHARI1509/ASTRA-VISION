# PHASE 8B — ASTRA VISION MODEL COMPARISON REPORT

## 1. Objective
This report establishes an objective, empirical, offline benchmark comparing two compatible
vision-language zero-shot classification foundation models on the exact same 30-image held-out
defence object recognition dataset.

## 2. Dataset & Integrity
- **Dataset**: ASTRA VISION Held-Out Benchmark
- **Total Images**: 30 (strictly identical 30-image set)
- **Classes**: 6 (5 images per class)
- **Evaluation Split**: `held_out_test` (100% held-out test split, seed=42)
- **Zero Leakage**: Neither model was trained, fine-tuned, or adapted on any test images.

## 3. Evaluated Models

### Model A (Production Baseline)
- **Name**: Google SigLIP 2 Base
- **Identifier**: `google/siglip2-base-patch16-512`
- **Architecture**: Vision Transformer ViT-B/16 (512×512 input resolution)
- **Loss Formulation**: Pairwise Sigmoid cross-entropy loss (SigLIP)
- **Device**: `cpu`

### Model B (Comparative Candidate)
- **Name**: OpenAI CLIP ViT-B/32
- **Identifier**: `openai/clip-vit-base-patch32`
- **Architecture**: Vision Transformer ViT-B/32 (224×224 input resolution)
- **Loss Formulation**: InfoNCE / Softmax Contrastive loss (CLIP)
- **Device**: `cpu`
- **Selection Rationale**: Industry-standard zero-shot foundation model baseline against which SigLIP was originally benchmarked in literature. Evaluates identically across the same candidate text prompts with zero fine-tuning.

## 4. Prompt & Evaluation Methodology
Both models evaluated using identical prompt templates per class:
- `Tank` -> `a photo of a tank`
- `Drone` -> `a photo of a drone`
- `Fighter Aircraft` -> `a photo of a fighter aircraft`
- `Military Vehicle` -> `a photo of a military vehicle`
- `Ship` -> `a photo of a ship`
- `Helicopter` -> `a photo of a helicopter`

## 5. Overall Metrics Comparison

| Metric | Model A (SigLIP 2 Base) | Model B (CLIP ViT-B/32) |
|---|---:|---:|
| **Top-1 Accuracy** | 100.00% | 93.33% |
| **Top-3 Accuracy** | 100.00% | 100.00% |
| **Macro Precision** | 1.0000 | 0.9524 |
| **Macro Recall** | 1.0000 | 0.9333 |
| **Macro F1** | 1.0000 | 0.9306 |
| **Mean Latency** | 1913.32 ms | 333.12 ms |
| **Median Latency** | 1600.33 ms | 120.17 ms |
| **Throughput** | 0.52 img/s | 2.91 img/s |

## 6. Per-Class Performance Breakdown

| Class | Support | Model A Correct | Model B Correct | Model A F1 | Model B F1 |
|---|---:|---:|---:|---:|---:|
| **Tank** | 5 | 5/5 | 3/5 | 1.0000 | 0.7500 |
| **Drone** | 5 | 5/5 | 5/5 | 1.0000 | 1.0000 |
| **Fighter Aircraft** | 5 | 5/5 | 5/5 | 1.0000 | 1.0000 |
| **Military Vehicle** | 5 | 5/5 | 5/5 | 1.0000 | 0.8333 |
| **Ship** | 5 | 5/5 | 5/5 | 1.0000 | 1.0000 |
| **Helicopter** | 5 | 5/5 | 5/5 | 1.0000 | 1.0000 |

## 7. Confusion Matrices

### Model A: Google SigLIP 2 Base (`google/siglip2-base-patch16-512`)
```
Ground Truth \ Predicted  | Tank     | Drone    | Fighter  | Military | Ship     | Helicopt
----------------------------------------------------------------------------------------------
Tank                      | 5        | 0        | 0        | 0        | 0        | 0       
Drone                     | 0        | 5        | 0        | 0        | 0        | 0       
Fighter Aircraft          | 0        | 0        | 5        | 0        | 0        | 0       
Military Vehicle          | 0        | 0        | 0        | 5        | 0        | 0       
Ship                      | 0        | 0        | 0        | 0        | 5        | 0       
Helicopter                | 0        | 0        | 0        | 0        | 0        | 5       
```

### Model B: OpenAI CLIP ViT-B/32 (`openai/clip-vit-base-patch32`)
```
Ground Truth \ Predicted  | Tank     | Drone    | Fighter  | Military | Ship     | Helicopt
----------------------------------------------------------------------------------------------
Tank                      | 3        | 0        | 0        | 2        | 0        | 0       
Drone                     | 0        | 5        | 0        | 0        | 0        | 0       
Fighter Aircraft          | 0        | 0        | 5        | 0        | 0        | 0       
Military Vehicle          | 0        | 0        | 0        | 5        | 0        | 0       
Ship                      | 0        | 0        | 0        | 0        | 5        | 0       
Helicopter                | 0        | 0        | 0        | 0        | 0        | 5       
```

## 8. Latency and Compute Observations
- **SigLIP 2 Base (512×512)**: Mean CPU latency ~1913 ms. Operates at 4× higher pixel resolution (512×512 vs 224×224), requiring greater compute per forward pass.
- **CLIP ViT-B/32 (224×224)**: Mean CPU latency ~333 ms. Operates with larger patch size (patch32) and smaller resolution (224×224), delivering higher throughput on CPU at the cost of fine spatial detail.

## 9. Limitations
- **Sample Size**: Benchmark consists of $N=30$ curated reconnaissance images. Accuracy metrics are specific to this test partition.
- **Uncertainty Metrics Omission**: SigLIP uses pairwise sigmoid activation scores, whereas CLIP uses softmax over contrastive cosine similarity logits. Because the mathematical ranges and calibration profiles differ, applying SigLIP heuristic thresholds to CLIP would be scientifically unsound. Uncertainty counts were excluded from the formal comparison table as specified.
- **No Ensembling**: Per specification, models were evaluated strictly independently. Predictions were never averaged, voted, or combined.

## 10. Reproducibility Instructions
To re-run this exact comparison independently:
```powershell
.\venv\Scripts\python scripts/run_model_comparison.py
```
Machine-readable outputs are stored in `backend/evaluation/results/phase8b_model_comparison.json`.

## 11. Production-Isolation Confirmation
- Production inference service `/api/classify` remains completely untouched.
- Production model remains `google/siglip2-base-patch16-512`.
- Frontend classification UI remains completely untouched.
- Zero production endpoints or ensembling mechanisms were introduced.