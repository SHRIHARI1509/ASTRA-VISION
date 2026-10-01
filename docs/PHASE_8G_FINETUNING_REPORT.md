# ASTRA VISION — PHASE 8G: PRODUCTION-ALIGNED SIGLIP 2 FINE-TUNING REPORT

**Document ID:** `AV-EXP-08G`  
**Date:** October 2026  
**Status:** COMPLETE (EXPERIMENT GATE COMPLETED)  
**Base Model:** `google/siglip2-base-patch16-512`  
**Fine-Tuned Checkpoint:** `models/finetuned/checkpoint-best/`  
**Held-Out Benchmark Set:** `evaluation/manifest.json` (30 images, 5 per class, 6 production classes)  
**Final Verdict:** **COMPARABLE**  
**Production Promotion Status:** **RETAIN BASELINE IN PRODUCTION** (No replacement authorized)

---

## 1. Objective

The primary objective of Phase 8G was to rigorously evaluate whether parameter-efficient fine-tuning (PEFT/LoRA) of `google/siglip2-base-patch16-512` on the supplied ASTRA starter dataset produces a measurable improvement over the existing frozen zero-shot baseline on the sacred 30-image held-out benchmark.

---

## 2. Hardware and Environment Telemetry

The training and evaluation were executed locally under standard hardware constraints:
* **Operating System:** Windows 11
* **Execution Device:** CPU (12 cores)
* **PyTorch Version:** `2.14.0+cpu`
* **PEFT Version:** `0.21.1`
* **Transformers Version:** `5.17.0`
* **RAM Profile:** 15.63 GB total (~5–6 GB allocated during execution)
* **GPU Availability:** None (explicit CPU fallback utilized; no unconstrained training jobs launched)

---

## 3. Dataset Composition & Integrity

* **Supplied Dataset Root:** `data/supplied_dataset/` (Strictly READ-ONLY; untouched)
* **Original Manifests:** `labels.csv` (150 rows), `credits.csv` (150 rows)
* **Integrity Audit:** 150/150 images decodable, 0 corrupted files, 0 exact duplicates, 0 perceptual duplicates.
* **Original Class Distribution:** `aircraft` (30), `drone` (30), `helicopter` (30), `military-vehicle` (30), `naval` (30).

---

## 4. Taxonomy Reconciliation (Human-in-the-Loop Review)

As demonstrated in the Dataset Audit, naive automatic mapping (`military-vehicle -> Tank` or `aircraft -> Fighter Aircraft`) would corrupt the model's decision boundaries. An exhaustive human review of all 60 ambiguous images was conducted:

### Safe Direct Mappings
* `drone` $\rightarrow$ `Drone` (30 images, 100% direct match)
* `helicopter` $\rightarrow$ `Helicopter` (30 images, 100% direct match)
* `naval` $\rightarrow$ `Ship` (30 images, 100% direct match)

### Ambiguous Mappings Partitioned
* **`military-vehicle` (30 images):**
  * **15 Main Battle Tanks / Tracked Armor $\rightarrow$ `Tank`**: Includes T-72B3, M3 Lee/Grant, M4 Sherman, Landsverk L-60, Tiger I, T-84 MBT components, Panzer 35(t), Renault FT-17, T-44.
  * **15 Support & Transport Vehicles $\rightarrow$ `Military Vehicle`**: Includes BMP-1 IFV, PzH 2000 SPG, BA-64 armored car, Soviet supply convoy, LPD resupply, BAHNA military trucks.
* **`aircraft` (30 images):**
  * **24 Combat Interceptors / Fighters $\rightarrow$ `Fighter Aircraft`**: Includes F-22 Raptor, F/A-18 Hornet, TEDBF, MiG-29K, MiG-21, Beaufighter night fighter, Su-47 Berkut, F2H Banshee, F3D-2 Skynight, F7F Tigercat.
  * **6 Out-of-Taxonomy Images $\rightarrow$ `EXCLUDED_OUT_OF_TAXONOMY`**: Excluded to protect the 6-class production taxonomy. Includes glider (`SGR-01`), wind-tunnel wooden mockup (`J50MOCKUP`), WW1 trainer biplane (`Avro 504`), world map infographic (`Most numerous fighter jet`), book cover (`Samoloty mysliwskie`), and sailplane (`Vindkast Mk.2E2`).

**Unresolved Labels Entering Training:** **0** (100% resolved).

---

## 5. Final Class Counts

| Production Class | Total Trainable Images | Percentage |
| :--- | :---: | :---: |
| `Drone` | 30 | 20.8% |
| `Helicopter` | 30 | 20.8% |
| `Ship` | 30 | 20.8% |
| `Fighter Aircraft` | 24 | 16.7% |
| `Military Vehicle` | 15 | 10.4% |
| `Tank` | 15 | 10.4% |
| **Total Trainable** | **144** | **100.0%** |
| *(Excluded Out-of-Taxonomy)* | *(6)* | — |

---

## 6. Deterministic Train/Validation Split

Using fixed seed `42`, stratified across all 6 reconciled classes:
* **Train Set:** 115 images (80%)
* **Validation Set:** 29 images (20%)
* **Test Set:** The frozen 30-image held-out benchmark (`evaluation/manifest.json`).

### Per-Class Split Allocation & Loss Weights

| Production Class | Train Count | Val Count | Inverse-Frequency Loss Weight |
| :--- | :---: | :---: | :---: |
| `Drone` | 24 | 6 | 0.80 |
| `Helicopter` | 24 | 6 | 0.80 |
| `Ship` | 24 | 6 | 0.80 |
| `Fighter Aircraft` | 19 | 5 | 1.01 |
| `Military Vehicle` | 12 | 3 | 1.60 |
| `Tank` | 12 | 3 | 1.60 |

* **Train/Validation Overlap:** **0** (0.0%)
* **Held-Out Benchmark Leakage:** **0** (0.0% cryptographic match; benchmark integrity 100% verified)

---

## 7. Zero-Shot Baseline Results (Frozen Pre-Trained Model)

Evaluated before training against the 30-image held-out benchmark:
* **Top-1 Accuracy:** **100.0%** (30/30 correct)
* **Top-3 Accuracy:** **100.0%** (30/30 correct)
* **Macro Precision:** **1.0000**
* **Macro Recall:** **1.0000**
* **Macro F1:** **1.0000**
* **Average Latency:** **1558.9 ms/image**
* **Misclassifications:** **0 / 30**

---

## 8. Training Configuration

* **Model Identifier:** `google/siglip2-base-patch16-512`
* **Processor Identifier:** `google/siglip2-base-patch16-512`
* **Learning Rate:** `1e-4`
* **Batch Size:** `4`
* **Gradient Accumulation Steps:** `2` (Effective batch size = 8)
* **Epochs:** `3`
* **Warmup Ratio:** `0.1`
* **Weight Decay:** `0.01`
* **Random Seed:** `42`
* **Optimizer:** AdamW
* **Scheduler:** Linear with Warmup
* **Loss Function:** Cross-Entropy with inverse-frequency class weights

---

## 9. LoRA / PEFT Configuration

* **PEFT Method:** LoRA (Low-Rank Adaptation)
* **LoRA Rank ($r$):** 8
* **LoRA Alpha ($\alpha$):** 16
* **Target Modules:** `["q_proj", "v_proj"]` (Vision Transformer attention projections)
* **LoRA Dropout:** 0.05
* **Trainable Parameters:** **589,824** out of 376,413,698 (**0.1567%**)
* **Base Model Weights:** 100% frozen

---

## 10 & 11. Training & Validation Progress

Evaluated strictly against the training and validation splits at each epoch:

| Epoch | Train Loss | Train Top-1 Acc | Val Loss | Val Top-1 Acc | Val Top-3 Acc | Duration (s) | Best Checkpoint? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.3997 | 86.1% | 0.1802 | 93.1% | 100.0% | 427.1s | Yes |
| **2** | 0.2848 | 92.2% | 0.1675 | 96.5% | 100.0% | 455.1s | Yes |
| **3** | **0.2491** | **93.0%** | **0.1663** | **96.5%** | **100.0%** | 453.5s | **Yes (BEST)** |

* **Total Training Wall Time:** 1345.4 seconds (~22.4 minutes on CPU)
* **Convergence Behavior:** Loss decreased steadily from 0.3997 to 0.2491. Validation Top-1 accuracy reached 96.55% (28/29 correct on validation set).

---

## 12. Best Checkpoint Selection

* **Selection Criterion:** Minimum validation loss among highest validation Top-1 accuracy checkpoints.
* **Selected Epoch:** **Epoch 3** (`val_loss = 0.1663`, `val_top1_acc = 0.9655`)
* **Saved Artifacts:** Saved to `models/finetuned/checkpoint-best/` containing `adapter_model.safetensors` (2.37 MB), `adapter_config.json`, and processor configurations.

---

## 13. Final Held-Out Evaluation (Sacred 30-Image Benchmark)

The best fine-tuned checkpoint was loaded and evaluated against the held-out benchmark:
* **Top-1 Accuracy:** **100.0%** (30/30 correct)
* **Top-3 Accuracy:** **100.0%** (30/30 correct)
* **Macro Precision:** **1.0000**
* **Macro Recall:** **1.0000**
* **Macro F1:** **1.0000**
* **Average Latency:** **1686.1 ms/image**
* **Misclassifications:** **0 / 30**

---

## 14. Baseline vs. Fine-Tuned Model Comparison Table

| Metric | Pre-Trained Baseline | LoRA Fine-Tuned Model | Delta | Outcome |
| :--- | :---: | :---: | :---: | :--- |
| **Top-1 Accuracy** | 100.0% (30/30) | 100.0% (30/30) | +0.0% | **Identical** |
| **Top-3 Accuracy** | 100.0% (30/30) | 100.0% (30/30) | +0.0% | **Identical** |
| **Macro Precision** | 1.0000 | 1.0000 | +0.0000 | **Identical** |
| **Macro Recall** | 1.0000 | 1.0000 | +0.0000 | **Identical** |
| **Macro F1** | 1.0000 | 1.0000 | +0.0000 | **Identical** |
| **Mean Inference Latency** | 1558.9 ms | 1686.1 ms | +127.2 ms | Slower (PEFT Adapter Overhead) |

---

## 15 & 16. Confusion Matrices & Per-Class Metrics

Both the Baseline and the Fine-Tuned Model achieved diagonal confusion matrices on the 30-image test set:

| Class | Ground Truth Count | Predicted as Target | Precision | Recall | F1-Score | Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tank** | 5 | 5 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| **Military Vehicle** | 5 | 5 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| **Fighter Aircraft** | 5 | 5 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| **Helicopter** | 5 | 5 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| **Ship** | 5 | 5 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| **Drone** | 5 | 5 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

---

## 17. Latency Comparison

* **Baseline Mean Latency:** 1558.92 ms/image
* **Fine-Tuned Mean Latency:** 1686.07 ms/image
* **Delta:** +127.15 ms/image (+8.15%)
* **Explanation:** In CPU execution mode, running through the PyTorch PEFT LoRA forward wrapper adds a minor computation overhead compared to the fused C++ native transformer layers of the stock model.

---

## 18. Data Leakage Verification

* **Cryptographic Hashing:** Every image in `data/experiments/train_manifest.json` and `data/experiments/val_manifest.json` was hashed using SHA-256 and compared with the 30 images in `evaluation/manifest.json`.
* **Overlap Count:** **0** (0.0%).
* **Contamination:** Zero. The benchmark was never seen during training or hyperparameter tuning.

---

## 19. Regression Results

* **Backend Unit & Integration Tests:** `pytest -v` $\rightarrow$ **190 / 190 PASSED** (0 failures)
* **Frontend Unit Tests:** `npm test` $\rightarrow$ **68 / 68 PASSED** (0 failures)
* **TypeScript Typecheck:** `npm run typecheck` $\rightarrow$ **0 ERRORS**
* **Frontend Production Build:** `npm run build` $\rightarrow$ **SUCCESS**

---

## 20. Browser Regression Verification

* Verified using automated browser inspection:
  1. Uploaded `tank.jpg`.
  2. Analyzed via live `/api/classify` endpoint.
  3. Verified Primary Prediction: `Tank` (score: 0.3132).
  4. Verified Top-3 Predictions: `Tank` (0.3132), `Military Vehicle` (0.0592), `Fighter Aircraft` (0.0001).
  5. Verified Uncertainty Badge and Model-Fit justification.
  6. Verified History Gallery retention.
  7. Verified bad input rejection.
* Zero regressions observed in user experience or system HUD.

---

## 21. Artifacts Created

1. `data/experiments/reconciliation_report.json`: Full 150-image reviewed taxonomy ledger.
2. `data/experiments/train_manifest.json`: 115-image stratified training manifest.
3. `data/experiments/val_manifest.json`: 29-image stratified validation manifest.
4. `data/experiments/balancing_report.json`: Stratification and class weight telemetry.
5. `data/experiments/baseline_metrics.json`: Pre-training zero-shot baseline metrics.
6. `models/finetuned/training_summary.json`: Epoch-by-epoch loss and accuracy telemetry.
7. `models/finetuned/checkpoint-best/`: Best LoRA adapter checkpoint weights (safetensors format).
8. `data/experiments/finetuned_benchmark_metrics.json`: Held-out benchmark metrics for fine-tuned checkpoint.
9. `data/experiments/model_comparison.json`: Exact mathematical comparison and promotion verdict.
10. `docs/PHASE_8G_FINETUNING_REPORT.md`: Comprehensive engineering report.

---

## 22. Limitations & Scientific Caveats

1. **Benchmark Size:** The held-out benchmark consists of 30 images (5 per class). While high accuracy is achieved on this benchmark, it cannot be claimed that general-world real-time battlefield classification is 100% solved.
2. **Dataset Scale:** 144 trainable images is a compact dataset. LoRA successfully prevented catastrophic forgetting and achieved 96.55% validation accuracy, matching the baseline.
3. **No Domain Shift Improvement:** Because the baseline pre-trained SigLIP 2 model already achieves 100% top-1 accuracy on the held-out benchmark, fine-tuning cannot mathematically exceed 100% on this benchmark.

---

## 23. Verdict: COMPARABLE

The fine-tuned model achieved identical top-1 (100.0%), top-3 (100.0%), and macro F1 (1.0000) scores on the held-out evaluation set, but incurs a minor latency overhead (+127.15 ms/image on CPU).

---

## 24. Production Promotion Status

* **Status:** **RETAIN BASELINE IN PRODUCTION**
* **Production Model:** `google/siglip2-base-patch16-512`
* **Fine-Tuned Checkpoint:** Preserved in `models/finetuned/checkpoint-best/` as an archived candidate.
* **Production `/api/classify`:** Continues using the pre-trained zero-shot SigLIP 2 pipeline without modification.
