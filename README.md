# ASTRA VISION — AI-Based Defence Object Recognition System

[![Backend Regression](https://img.shields.io/badge/Backend%20Tests-190%2F190%20PASSED-brightgreen.svg)]()
[![Frontend Tests](https://img.shields.io/badge/Frontend%20Tests-68%2F68%20PASSED-brightgreen.svg)]()
[![Typecheck](https://img.shields.io/badge/TypeScript-0%20Errors-brightgreen.svg)]()
[![Production Build](https://img.shields.io/badge/Vite%20Build-PASS-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

ASTRA VISION is an advanced reconnaissance and intelligence image analysis platform designed for automated categorization, uncertainty quantification, and multi-candidate ranking of defence assets across contested operational environments.

---

## 1. What ASTRA VISION Is
ASTRA VISION is a full-stack, automated defence asset recognition platform engineered for aerial, naval, and ground reconnaissance imagery. It provides operators with rapid zero-shot classification, deterministic Top-3 multi-candidate ranking, heuristic uncertainty warnings, and sequential batch inspection through a modern, responsive web console.

## 2. Problem Being Solved
Tactical image interpreters in modern defence environments face high volumes of heterogeneous reconnaissance imagery from electro-optical sensors, uncrewed aerial systems (UAS), and satellite passes. Manual verification is bottlenecked by cognitive fatigue, ambiguous viewing angles, and rapid operational tempos. ASTRA VISION solves this by delivering immediate, automated categorization against a standardized defence taxonomy, alerting operators whenever an asset's visual evidence is low-confidence or ambiguous.

## 3. Core Workflow
1. **Ingestion & Client Validation:** The operator uploads an image (drag-and-drop or file picker) or submits a batch of files. The client validates format, size boundaries, and integrity.
2. **Server-Side Raster Verification:** FastAPI backend validates file format, decodability via Pillow, dimensions, and converts to 3-channel RGB.
3. **Zero-Shot Foundation Inference:** The singleton `google/siglip2-base-patch16-512` model evaluates the image against centralized prompt templates using pairwise sigmoid similarity.
4. **Deterministic Multi-Candidate Ranking:** Candidates are ranked by score descending, with deterministic secondary tie-breaking by taxonomy index.
5. **Heuristic Uncertainty Evaluation:** Operating thresholds evaluate whether the primary score is weak (`< 0.0100`) or margin between ranks 1 and 2 is narrow (`< 0.0200`).
6. **Result & Justification Delivery:** The UI renders primary prediction badges, Top-3 candidate comparison bars, uncertainty diagnostics, and architectural model-fit justification.
7. **Session History:** The classified item is appended to an ephemeral, session-only history gallery for operator review.

## 4. Supported Classes
The production system operates against a standardized 6-class defence taxonomy:
1. **`Tank`** (Main battle tanks, heavy tracked armor)
2. **`Military Vehicle`** (Infantry fighting vehicles, self-propelled artillery, tactical utility/transport vehicles)
3. **`Fighter Aircraft`** (Combat interceptors, multi-role strike fighters)
4. **`Helicopter`** (Attack, transport, and naval rotary aircraft)
5. **`Ship`** (Naval warships, frigates, carriers, auxiliary vessels)
6. **`Drone`** (Uncrewed aerial systems, loitering munitions, reconnaissance UAS)

## 5. Production Model
* **Model Identifier:** `google/siglip2-base-patch16-512`
* **Architecture:** SigLIP 2 Vision Transformer (ViT-Base with Patch-16, 512×512 native resolution)
* **Pre-Training Objective:** Pairwise sigmoid vision-language alignment
* **Runtime Deployment:** Single thread-safe in-memory singleton instance loaded at application startup
* **Device Handling:** Automatic device resolution (`auto` $\rightarrow$ CUDA if available with graceful CPU fallback)

## 6. Top-3 Predictions
Rather than presenting an isolated, brittle classification, ASTRA VISION extracts and presents the top-3 ranked candidates for every input. This ensures operators can review the immediate runner-up categories, which is vital when distinguishing borderline assets (such as an armed IFV versus a tank).

## 7. Confidence & Uncertainty Behavior
* **Heuristic Scoring:** SigLIP 2 outputs raw pairwise sigmoid similarity scores, **not** statistically calibrated Bayesian probabilities.
* **Operating Thresholds:**
  * `UNCERTAINTY_SCORE_THRESHOLD = 0.0100`: Triggers `LOW_PRIMARY_SCORE` when overall visual evidence is weak.
  * `UNCERTAINTY_MARGIN_THRESHOLD = 0.0200`: Triggers `LOW_SCORE_MARGIN` when the score gap between rank 1 and rank 2 is slim.
  * Both conditions met $\rightarrow$ `BOTH`.
* **Visual Indication:** The UI displays an amber warning badge explaining the exact rationale, prompting human review.

## 8. Image Preprocessing
* **Accepted Formats:** JPEG, PNG, WEBP.
* **Channels:** Automatically standardizes RGB, RGBA, and single-channel Grayscale (L) inputs into canonical 3-channel RGB.
* **Dimension Bounds:** Enforces strict boundary checks (minimum $32\times32$, maximum $4096\times4096$, size $\le 10\text{ MB}$).
* **Normalization:** Image tensors are scaled and normalized to $[-1, 1]$ matching the native SigLIP 2 processor requirements (`padding="max_length"`).

## 9. Batch Processing
* **Endpoint:** `POST /api/classify/batch`
* **Capacity:** Supports up to 20 images per batch request.
* **Resilience:** Features partial failure tolerance—corrupted files or non-image assets fail individually with structured error objects while valid images are successfully classified.
* **UI Workflow:** Dedicated multi-file dropzone with real-time sequential progress tracking and a filterable batch results summary.

## 10. Session-Only History
* **Storage Modality:** Pure in-memory React state (`useState`).
* **Privacy & Isolation:** Zero data is transmitted to external databases, cloud services, or persistent browser caches (`localStorage`/`IndexedDB`).
* **Lifecycle:** Closing or refreshing the browser cleanly purges all historical session records. An explicit "Clear History" button is provided.

## 11. Held-Out Benchmark
* **Dataset:** 30 carefully curated defence images (5 balanced samples per production class).
* **Location:** [`data/held_out_dataset/`](file:///c:/FILES/astra-vision/data/held_out_dataset/) and manifest [`evaluation/manifest.json`](file:///c:/FILES/astra-vision/evaluation/manifest.json).
* **Sealed & Frozen:** Cryptographically verified via SHA-256; zero overlap with training or exploratory datasets.
* **Baseline Benchmark Result:** 100.0% Top-1 Accuracy, 100.0% Top-3 Accuracy, Macro F1 1.0000 on this 30-image set.

## 12. Model Comparison (Phase 8B)
A formal comparative benchmark was executed against `openai/clip-vit-base-patch16`:
* **SigLIP 2:** 100.0% Top-1 Accuracy, Macro F1 1.0000.
* **CLIP ViT-B/16:** 86.67% Top-1 Accuracy, Macro F1 0.8651 (struggled with Military Vehicle vs Tank separation).
* **Full Report:** See [`PHASE_8B_MODEL_COMPARISON.md`](file:///c:/FILES/astra-vision/PHASE_8B_MODEL_COMPARISON.md).

## 13. Fine-Tuning Experiment (Phase 8G)
* **Method:** LoRA PEFT ($r=8, \alpha=16$, 589,824 trainable parameters / 0.1567%) on 144 curated starter images over 3 epochs.
* **Outcome:** Reached 96.55% validation accuracy and matched the baseline 100.0% accuracy on the held-out benchmark.
* **Decision:** Incurred +127.15 ms/image (+8.15%) latency overhead on CPU. In accordance with strict deployment governance, the **frozen baseline was retained in production**. The fine-tuned weights remain archived in [`models/finetuned/checkpoint-best/`](file:///c:/FILES/astra-vision/models/finetuned/checkpoint-best/). See [`docs/PHASE_8G_FINETUNING_REPORT.md`](file:///c:/FILES/astra-vision/docs/PHASE_8G_FINETUNING_REPORT.md).

## 14. Detection Readiness & Infrastructure
* While the starter dataset contains only classification labels, Phase 8E established detection readiness scaffolding in [`backend/app/detection/`](file:///c:/FILES/astra-vision/backend/app/detection/).
* **Strict Guardrail:** Object detection is **not** active in production, and no synthetic or fake bounding boxes are generated.

## 15. Limitations & Caveats
* **Benchmark Size:** The 30-image benchmark is statistically compact; 100% accuracy does **not** imply universal real-world accuracy across all operational conditions.
* **No Calibrated Probabilities:** Similarity scores reflect empirical visual-text feature proximity, not Bayesian probabilities.
* **Starter Dataset Taxonomy:** The 150-image starter dataset contained ambiguous vehicle labels that required human-in-the-loop review.
* **CPU Latency:** Average CPU inference latency is ~1.5s per image; real-time video feeds require GPU hardware acceleration.
* **Full Limitations Document:** See [`docs/LIMITATIONS.md`](file:///c:/FILES/astra-vision/docs/LIMITATIONS.md).

## 16. Dataset Provenance
* **Source:** All 150 starter images were sourced from **Wikimedia Commons** under CC BY, CC BY-SA, and Public Domain licenses.
* **Attribution Ledger:** Complete attribution, artists, licenses, and Commons URLs are documented in [`data/supplied_dataset/credits.csv`](file:///c:/FILES/astra-vision/data/supplied_dataset/credits.csv).
* **Full Provenance Report:** See [`docs/DATA_PROVENANCE.md`](file:///c:/FILES/astra-vision/docs/DATA_PROVENANCE.md).

## 17. How to Run Locally

### Prerequisites
* Python 3.10+ (tested on Python 3.14)
* Node.js v18+ & npm

### Backend Setup
```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Run FastAPI API server (starts on port 8000)
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Swagger API documentation: `http://127.0.0.1:8000/docs`

### Frontend Setup
```powershell
# In a new terminal:
cd frontend
npm install
npm run dev
```
Web console runs at `http://localhost:5173` (proxies `/api` requests to backend).

## 18. How to Run Tests

### Run All Backend Regression Tests (190 Tests)
```powershell
.\venv\Scripts\pytest -v
```

### Run All Frontend Unit Tests (68 Tests)
```powershell
cd frontend
npm test
```

### Run TypeScript Typecheck & Production Build
```powershell
cd frontend
npm run typecheck   # Strict tsc -b
npm run build       # Vite production bundle
```

## 19. AI Usage Disclosure
Development of ASTRA VISION was accelerated using AI assistance (Google DeepMind agentic tools) in the following specific capacities:
* **Code Implementation:** Boilerplate service definitions, Pydantic v2 schemas, React 19 component layouts, and Vite proxy configuration.
* **Test Engineering:** Construction of comprehensive unit test suites covering edge cases, image corruptions, aspect ratio anomalies, and batch limits.
* **Taxonomy Review Assistance:** Cataloging and cross-referencing military equipment specifications for the human-in-the-loop audit.
* **Scientific Documentation:** Structuring engineering reports, metric tables, and architecture decision records.
* *Note: All training, benchmark evaluations, code execution, and test passes were executed deterministically on local hardware.*

## 20. Security & Secrets Policy
ASTRA VISION operates without hardcoded secrets, external API keys, or private tokens. All models are open-weights foundation models run locally or cached on-premise. Configuration is managed via environment templates without embedded credentials. See [`docs/phase-9-security-audit.md`](file:///c:/FILES/astra-vision/docs/phase-9-security-audit.md).
