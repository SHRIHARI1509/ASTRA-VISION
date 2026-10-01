# ASTRA VISION — System Architecture & Component Specification

**Document ID:** `AV-ARCH-09`  
**Phase:** 9.10 Architecture Documentation  
**Date:** October 2026  
**Auditor / Architect:** Final Release Engineer  
**Status:** COMPLETE & SYSTEM-ALIGNED  

---

## 1. System Overview & Core Workflow

ASTRA VISION is an AI-based defence reconnaissance image classification platform built with a high-performance decoupled architecture: a modern TypeScript/React 19 single-page interface backed by a Python/FastAPI service orchestrating zero-shot vision inference with Google's `SigLIP 2` foundation vision-language model.

### Primary Inference Pipeline

```mermaid
flowchart TD
    User([User / Reconnaissance Operator]) -->|Upload Image| Frontend[Frontend UI Shell (React 19 / Vite)]
    Frontend -->|Format / Size Pre-Checks| ClientValidator[Client-Side Image Validator]
    ClientValidator -->|POST multipart/form-data| API[FastAPI API Gateway]
    
    subgraph Backend [Backend Service Boundary]
        API -->|Route: /api/classify| ClassifyService[Classification Service]
        ClassifyService -->|Validate Raster Integrity| ImageService[Image Service / Pillow]
        ImageService -->|Decoded RGB Image| InferenceService[Inference Service (Singleton)]
        
        subgraph ModelExecution [Model Execution Layer]
            InferenceService -->|Cached Stock Weights| SigLIP2Adapter[SigLIP 2 Adapter]
            SigLIP2Adapter -->|Pre-Trained Weights| HFModel["AutoModel: google/siglip2-base-patch16-512"]
            SigLIP2Adapter -->|Text Prompts (6 Classes)| HFProc["AutoProcessor (Padding max_length)"]
            HFModel & HFProc -->|Pairwise Sigmoid Activation| RawScores[Raw Similarity Scores]
        end
        
        RawScores -->|Score & Secondary Tie-Breaking| DeterministicRanker[Deterministic Candidate Ranking]
        DeterministicRanker -->|Highest Scoring Candidate| PrimaryPred[Primary Prediction]
        DeterministicRanker -->|Top 3 Ranked Candidates| Top3Extract[Top-3 Secondary Candidates]
        
        PrimaryPred & Top3Extract -->|Score & Margin Evaluation| UncertaintyHeuristic[Uncertainty Heuristic Engine]
        UncertaintyHeuristic -->|Attach Model Fit Justification| ResponseBuilder[Response Assembler]
    end
    
    ResponseBuilder -->|JSON Response| Frontend
    Frontend -->|Display HUD & Top-3 Badges| ResultPanel[Classification Result Panel]
    Frontend -->|Record Session State| SessionHistory[Session-Only Gallery Hook]
```

---

## 2. Component Categorization & Operational Readiness

To maintain scientific integrity and operational transparency, all repository subsystems are strictly classified into one of three tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. IMPLEMENTED & PRODUCTION-ACTIVE                                     │
│    - FastAPI backend API (/api/classify, /api/classify/batch)          │
│    - Frozen SigLIP 2 zero-shot baseline adapter                        │
│    - Image validation & multi-format preprocessing (JPEG/PNG/WEBP/RGBA)│
│    - Deterministic Top-3 ranking with secondary taxonomy tie-break     │
│    - Operating threshold uncertainty heuristics (score < 0.01, margin) │
│    - Sequential batch classification (up to 20 images) with progress   │
│    - Session-only memory history gallery (ephemeral, zero persistence) │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. INFRASTRUCTURE & READINESS (NON-PRODUCTION)                         │
│    - Detection readiness evaluators (backend/app/detection/)           │
│    - Dataset preparation & stratification (backend/app/preparation/)   │
│    - Automated dataset audit & non-leakage verification suites         │
│    - Synthetic bounding-box guards (prohibiting fake annotations)      │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. EXPERIMENTAL & ARCHIVED (OFFLINE ONLY)                              │
│    - Phase 8G LoRA fine-tuned checkpoint (models/finetuned/checkpoint) │
│    - Model comparison framework (SigLIP 2 vs CLIP ViT-B/16)            │
│    - Offline fine-tuning trainer & comparison reports                  │
│    - 150-image human-in-the-loop taxonomy reconciliation manifests    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Subsystem Architectural Details

### 3.1 Production Model Adapter (`backend/app/models/siglip2_adapter.py`)
* **Underlying Model:** `google/siglip2-base-patch16-512`
* **Zero-Shot Alignment:** Uses pre-trained vision-language features to compute sigmoid similarity against 6 predefined target prompts:
  * `a photo of a fighter aircraft`
  * `a photo of a helicopter`
  * `a photo of a tank`
  * `a photo of a ship`
  * `a photo of a military vehicle`
  * `a photo of a drone`
* **Hardware Resilience:** Supports `auto`, `cuda`, and explicit `cpu` fallbacks.
* **Thread Safety:** Thread-locked singleton initialization prevents redundant weight loads or memory fragmentation.

### 3.2 Heuristic Uncertainty Engine (`backend/app/services/classification_service.py`)
* Evaluates raw similarity scores against centralized operating thresholds:
  * Primary score check: `primary_score < 0.0100` $\rightarrow$ `LOW_PRIMARY_SCORE`
  * Separation margin check: `(primary_score - runner_up_score) < 0.0200` $\rightarrow$ `LOW_SCORE_MARGIN`
  * Both conditions met $\rightarrow$ `BOTH`
* **Caveat:** Evaluates empirical score separation; does not represent statistically calibrated Bayesian probabilities.

### 3.3 Batch Processing Engine
* **Endpoint:** `POST /api/classify/batch`
* **Concurrency Model:** Sequential execution utilizing the cached singleton model instance to prevent CPU saturation or GPU out-of-memory errors.
* **Fault Tolerance:** Per-item try/catch handles image validation failures gracefully without aborting the batch.
* **Batch Bounds:** Strictly capped at `MAX_BATCH_SIZE = 20` images.

### 3.4 Session History Gallery (`frontend/src/hooks/useHistory.ts`)
* **Storage Modality:** Pure in-memory React state (`useState`).
* **Persistence:** Strictly session-only. Refreshing or closing the browser completely clears history.
* **Privacy & Isolation:** Zero data transmitted to cloud storage, databases, or local browser storage (`localStorage` / `IndexedDB`).

### 3.5 Fine-Tuning Experiment Isolation
* The fine-tuned LoRA weights (`models/finetuned/checkpoint-best/`) are completely unmounted from the production API.
* The API runtime never invokes PEFT or checks for local adapter weights, ensuring zero latency degradation or regression in production.
