# ASTRA VISION — System Limitations & Scientific Caveats

**Document ID:** `AV-LIM-09`  
**Phase:** 9.12 Limitations Document  
**Date:** October 2026  
**Auditor / Author:** Final Release Engineer  
**Status:** COMPLETE & FACTUAL  

---

## 1. Benchmark & Accuracy Caveats

1. **Held-Out Benchmark Scale (30 Images):**
   * The frozen benchmark comprises 30 images (5 per class). While ideal for fast, deterministic regression gating, a 30-sample evaluation set is statistically compact.
   * **100% Benchmark Accuracy Does NOT Imply Universal Real-World Reliability:** Perfect accuracy on this controlled test set does not guarantee error-free performance in unconstrained operational deployments. Field conditions with extreme weather, heavy foliage, active camouflage, or non-standard angles will exhibit lower performance.
2. **Domain Generalization:**
   * Model inference reliability degrades when inputs diverge significantly from the pre-training distribution (e.g. synthetic IR, thermal imagery, low-resolution satellite apertures, or radar signatures).

---

## 2. Uncertainty & Scoring Interpretation

1. **Heuristic Confidence, Not Calibrated Probability:**
   * SigLIP 2 produces pairwise sigmoid similarity scores for text prompts, not a closed normalized probability distribution.
   * Uncertainty alerts are driven by empirical operating thresholds (`primary_score < 0.0100` or `margin < 0.0200`). These represent practical heuristic boundaries to flag low-confidence or ambiguous classifications, **not** Bayesian calibrated confidence intervals.
2. **Visually Ambiguous Military Assets:**
   * Assets sharing structural morphology (e.g., heavily armed Infantry Fighting Vehicles vs. Main Battle Tanks, or armed reconnaissance drones vs. small fighter jets) can yield tight score margins that trigger uncertainty warnings.

---

## 3. Dataset & Taxonomy Constraints

1. **Taxonomy Ambiguities in Starter Dataset:**
   * The original 150-image dataset grouped diverse vehicles under coarse classes (`military-vehicle` and `aircraft`). Rigorous human audit showed that `military-vehicle` contained both tracked tanks and wheeled support trucks, while `aircraft` contained trainers, gliders, and mockups.
   * The production system uses a 6-class taxonomy. Reconciling starter images required human curation rather than naive automated renaming.
2. **Absence of Bounding-Box Ground Truth:**
   * The dataset contains exclusively image-level classification labels. There are **zero** ground-truth bounding boxes, segmentation masks, or keypoints.
3. **Object Detection Is Non-Production:**
   * Object detection and spatial localization are **NOT** implemented in the production application. Spatial coordinates or multi-object bounding boxes are not returned by `/api/classify`.
   * Infrastructure modules in `backend/app/detection/` serve exclusively as architectural readiness scaffolding for future evaluation when annotated datasets become available.

---

## 4. Model Training & Deployment Governance

1. **LoRA Fine-Tuned Model Was Not Promoted:**
   * Phase 8G LoRA fine-tuning on 144 curated images achieved 100% Top-1 accuracy on the held-out benchmark—matching the pre-trained foundation baseline.
   * However, LoRA PEFT adapter forward calls introduced a +127.15 ms/image (+8.15%) latency penalty on CPU. Because accuracy was identical and latency was higher, the **pre-trained baseline was retained in production**. The fine-tuned checkpoint remains archived.

---

## 5. Architectural & Hardware Characteristics

1. **In-Memory Session History Only:**
   * The frontend history gallery stores analysis results strictly in transient React component state. No database, server-side store, or browser storage (`localStorage`) is used. Reloading or closing the browser resets session history.
2. **CPU Inference Latency:**
   * Under standard 12-core CPU execution without hardware acceleration (CUDA), SigLIP 2 512x512 inference requires approximately 1.5 to 1.7 seconds per image. Real-time video processing requires GPU hardware acceleration.
3. **Image-Centric Inference:**
   * The core `/api/classify` endpoint is designed for single-image inspection. Bulk workloads must use `/api/classify/batch` (capped at 20 images per request), which evaluates images sequentially to prevent hardware resource starvation.
