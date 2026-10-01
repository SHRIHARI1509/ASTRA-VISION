# Phase 9: Production Model Integrity & Isolation Verification

**Document ID:** `AV-VER-09B`  
**Phase:** 9.2 Production Model Integrity Check  
**Date:** October 2026  
**Auditor:** Final Release Engineer  
**Status:** VERIFIED & PASS  

---

## 1. Executive Summary

A comprehensive architectural and runtime verification was performed to validate that ASTRA VISION operates with the pre-trained foundation baseline model and that experimental fine-tuned checkpoints remain strictly quarantined.

* **Production Model:** `google/siglip2-base-patch16-512`
* **Fine-Tuned Checkpoint:** `models/finetuned/checkpoint-best/`
* **Production Deployment Status:** **RETAIN BASELINE IN PRODUCTION**
* **LoRA Promotion Status:** **ARCHIVED CANDIDATE ONLY** (Not deployed to runtime)
* **Verification Result:** **PASS (Ceiling Baseline Fully Isolated & Active)**

---

## 2. Production Execution Path Trace

The production inference pipeline was traced end-to-end through every layer of the system:

### 2.1 Configuration Layer
* **Source:** [`backend/app/core/config.py`](file:///c:/FILES/astra-vision/backend/app/core/config.py)
* **Settings:**
  ```python
  MODEL_ID: str = "google/siglip2-base-patch16-512"
  DEVICE: str = "auto"
  PRELOAD_MODEL: bool = True
  CANDIDATE_CATEGORIES: List[str] = [
      "Fighter Aircraft", "Helicopter", "Tank", "Ship", "Military Vehicle", "Drone"
  ]
  ```
* **Finding:** The runtime configuration strictly points to the pre-trained Hugging Face identifier `google/siglip2-base-patch16-512`. No dynamic switch or override reads the local `models/finetuned/` folder.

### 2.2 Model Adapter Layer
* **Source:** [`backend/app/models/siglip2_adapter.py`](file:///c:/FILES/astra-vision/backend/app/models/siglip2_adapter.py)
* **Loading Implementation:**
  ```python
  processor = AutoProcessor.from_pretrained(self._model_id)
  model = AutoModel.from_pretrained(self._model_id)
  model.to(self._device)
  model.eval()
  ```
* **Finding:** `SigLIP2Adapter` directly loads `AutoModel` from the pre-trained foundation weights. There is no `PeftModel.from_pretrained` or adapter injection in the production adapter.

### 2.3 Inference Service Layer
* **Source:** [`backend/app/services/inference_service.py`](file:///c:/FILES/astra-vision/backend/app/services/inference_service.py)
* **Singleton:** `inference_service = InferenceService()` initializes an unadorned `SigLIP2Adapter` instance.
* **Finding:** Thread-safe singleton lifecycle enforces that only one foundation model instance resides in memory.

### 2.4 Domain Classification Service Layer
* **Source:** [`backend/app/services/classification_service.py`](file:///c:/FILES/astra-vision/backend/app/services/classification_service.py)
* **Flow:**
  * Validates image format and raster integrity via `image_service`.
  * Passes image to `inference_service.run_inference(...)`.
  * Computes deterministic ranking with secondary taxonomy tie-breaking.
  * Evaluates heuristic uncertainty against operating thresholds (`score < 0.0100` or `margin < 0.0200`).
  * Injects `MODEL_FIT_JUSTIFICATION` documenting `google/siglip2-base-patch16-512`.

### 2.5 API & Frontend Route Layer
* **API Endpoints:**
  * `POST /api/classify` $\rightarrow$ calls `classification_service.classify_image()`
  * `POST /api/classify/batch` $\rightarrow$ calls `classification_service.classify_batch()`
* **Frontend Service:** [`frontend/src/services/api.ts`](file:///c:/FILES/astra-vision/frontend/src/services/api.ts)
  * `classifyImage()` invokes `POST /api/classify`.
  * Result components display `inference.model` (`google/siglip2-base-patch16-512`).

---

## 3. Quarantined Fine-Tuning Artifacts Verification

An exhaustive grep across the entire production codebase (`backend/app/`) confirmed:
1. `models/finetuned/checkpoint-best/` is referenced solely within the offline experimental training and evaluation modules:
   * [`backend/app/finetuning/trainer.py`](file:///c:/FILES/astra-vision/backend/app/finetuning/trainer.py#L202)
   * [`scripts/run_phase8g_finetuning.py`](file:///c:/FILES/astra-vision/scripts/run_phase8g_finetuning.py)
2. No automatic promotion scripts, symlinks, or environment variable swaps exist to silently redirect `settings.MODEL_ID` to `models/finetuned/checkpoint-best/`.
3. The fine-tuning artifacts (`adapter_model.safetensors`, `adapter_config.json`) remain strictly archived on disk as scientific records.

---

## 4. Verification Verdict

| Verification Item | Specification | Observed Runtime Value | Status |
| :--- | :--- | :--- | :---: |
| Production Model ID | `google/siglip2-base-patch16-512` | `google/siglip2-base-patch16-512` | **PASS** |
| Processor Source | Foundation Pre-Trained | `google/siglip2-base-patch16-512` | **PASS** |
| Model Weights | Frozen Stock Weights | Frozen Stock Weights | **PASS** |
| PEFT LoRA Injection | None in Production | 0 adapter hooks active | **PASS** |
| Fine-Tuning Checkpoint | Archived in `models/finetuned/` | Isolated from API | **PASS** |
| Architecture Integrity | Decoupled Singleton | Verified | **PASS** |
