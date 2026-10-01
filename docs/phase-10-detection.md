# Phase 10: Multi-Object Detection with Bounding Boxes

## 1. Executive Summary & Objective

Phase 10 introduces the bonus operational capability: **Multi-Object Detection with Bounding Boxes** to ASTRA VISION.

This is an **additive bonus feature** implemented alongside the verified, production-hardened Phase 9 zero-shot classification system. The existing classification pipeline (`google/siglip2-base-patch16-512`) remains completely untouched, fully regression-tested, and retained as the primary target reconnaissance engine.

### Strict Governance Statement
> **"Phase 10 uses a pretrained open-vocabulary detector. The supplied ASTRA dataset contains classification labels but no ground-truth bounding boxes, so no custom detector was trained."**
>
> In accordance with strict AI engineering integrity guidelines:
> - **Zero fake bounding boxes** were invented or hallucinated.
> - **Zero pseudo-labels** were created for the dataset.
> - **No custom detector training** was performed or claimed on synthetic data.
> - The frozen 30-image held-out benchmark and original 150-image dataset were **not modified**.

---

## 2. Model Selection & Architecture

* **Model Identifier:** `IDEA-Research/grounding-dino-base`
* **Architecture:** Grounding DINO (Dual-encoder Vision-Language Transformer for Grounded Object Detection)
* **Pre-training Modality:** Pretrained open-vocabulary grounded object detection using cross-modality feature alignment between visual features and text queries.
* **Inference Pipeline:**
  ```
  User Image Upload
         ↓
  Image Validation & RGB Standardization (ImageService)
         ↓
  Open-Vocabulary Detection (Grounding DINO via Transformers)
         ↓
  Filtering by Threshold & Non-Maximum Suppression
         ↓
  Production Taxonomy Normalization (6 Defence Classes)
         ↓
  Bounded Absolute Pixel Coordinates [x1, y1, x2, y2]
         ↓
  Frontend Responsive SVG Overlay with Target Badges
  ```

---

## 3. Classification vs. Detection Separation

ASTRA VISION maintains strict architectural separation between classification and detection pipelines:

| Aspect | Classification Pipeline (Phase 1–9) | Multi-Object Detection Pipeline (Phase 10) |
| :--- | :--- | :--- |
| **Model** | `google/siglip2-base-patch16-512` | `IDEA-Research/grounding-dino-base` |
| **Primary Task** | Whole-image Zero-Shot Categorization | Spatial Multi-Object Localization & Bounding |
| **Output Type** | Primary Category + Top-3 Deterministic Ranking | Multiple Bounding Boxes + Confidence Scores |
| **API Endpoint** | `POST /api/classify` | `POST /api/detect` |
| **Dataset Training** | Frozen zero-shot foundation model | Pretrained zero-shot / open-vocabulary |
| **Uncertainty** | Operating heuristics (Score + Margin thresholds) | Detection filtering threshold (`box_threshold = 0.35`) |

---

## 4. Detection Prompts & Taxonomy Normalization

### A. Detection Vocabulary Prompts
Grounding DINO requires period-delimited lowercase text prompts:
```text
"a tank. a military vehicle. a fighter aircraft. a helicopter. a ship. a drone."
```

### B. Production Taxonomy Normalization Mapping
Grounding DINO produces open-vocabulary raw string tokens. ASTRA VISION normalizes these detected phrases strictly into its 6-class defence taxonomy:

| Raw Detector Output Phrasing | Normalized Production Taxonomy Class |
| :--- | :--- |
| `tank`, `a tank`, `a tank.` | **`Tank`** |
| `military vehicle`, `armored vehicle`, `apc`, `ifv`, `vehicle` | **`Military Vehicle`** |
| `fighter aircraft`, `fighter jet`, `warplane`, `aircraft`, `fighter` | **`Fighter Aircraft`** |
| `helicopter`, `a helicopter`, `rotary aircraft` | **`Helicopter`** |
| `ship`, `warship`, `destroyer`, `frigate`, `naval vessel` | **`Ship`** |
| `drone`, `a drone`, `uav`, `uas` | **`Drone`** |

### C. Strict Filter Guardrail
Any detection whose text alignment matches unrelated open-vocabulary concepts (e.g., `"person"`, `"building"`, `"tree"`, `"car"`, `"cloud"`) is **never force-mapped** into a defence class. It is discarded cleanly to ensure operational integrity.

---

## 5. API Specification (`POST /api/detect`)

### Request
* **Endpoint:** `POST /api/detect`
* **Content-Type:** `multipart/form-data`
* **Form Field:** `image` (`UploadFile`)
* **Optional Query Parameters:**
  * `box_threshold` (float, default `0.35`): Minimum detection filtering threshold.
  * `text_threshold` (float, default `0.25`): Minimum text alignment threshold.

### Response Envelope (`200 OK`)
```json
{
  "detections": [
    {
      "class_name": "Tank",
      "score": 0.9125,
      "box": {
        "x1": 120,
        "y1": 80,
        "x2": 540,
        "y2": 410
      }
    }
  ],
  "image_width": 1920,
  "image_height": 1080,
  "count": 1,
  "detector": "IDEA-Research/grounding-dino-base",
  "device": "cpu",
  "inference_time_ms": 13173.33
}
```

### Coordinate Invariants Enforced by Pydantic:
* $0 \le x_1 < x_2 \le \text{image\_width}$
* $0 \le y_1 < y_2 \le \text{image\_height}$
* Strict inequalities: $x_1 < x_2$ and $y_1 < y_2$ validated by model validator.

---

## 6. Thresholding Behavior

* **Threshold Configuration:** Centralized in [`backend/app/core/config.py`](file:///c:/FILES/astra-vision/backend/app/core/config.py) as `DETECTION_BOX_THRESHOLD = 0.35` and `DETECTION_TEXT_THRESHOLD = 0.25`.
* **Important Caveat:** Detector scores represent raw cross-attention similarity between image regions and text tokens. They are **empirical filtering thresholds**, **NOT** calibrated Bayesian probabilities.
* **Empty State:** If no detected object exceeds the filtering threshold, the API returns `count: 0` and `detections: []`. The UI displays: `"No matching objects detected."`

---

## 7. Hardware & Device Behavior

* **Device Resolution:** Evaluates `auto` $\rightarrow$ `cuda` if NVIDIA GPU is present; otherwise falls back gracefully to `cpu`.
* **In-Memory Caching:** Singleton `GroundingDINOAdapter` loads weights once upon first request and maintains model cache in memory across subsequent calls.
* **Failure Handling:** If weights cannot be loaded, returns structured `HTTP 503 Service Unavailable` with `DETECTOR_UNAVAILABLE`. If inference encounters an execution error, returns structured `HTTP 500 Internal Server Error`. The application process does not crash.

---

## 8. Frontend Responsive Visualization

* **Mode Switcher:** Prominent top-level navigation allows operators to toggle between `TARGET CLASSIFICATION (SigLIP 2)` and `OBJECT DETECTION (Grounding DINO)`.
* **SVG Coordinate Scaling:** Bounding boxes are rendered via an SVG overlay using `viewBox="0 0 {image_width} {image_height}"` and `preserveAspectRatio="none"`. Because SVG coordinate math is normalized to the native pixel dimensions, bounding boxes remain pixel-perfect regardless of display size, responsive browser scaling, or screen DPI.
* **Visual Elements:**
  * Color-coded category boxes (Cyan for Tank, Amber for Military Vehicle, Azure for Fighter Aircraft, Violet for Helicopter, Sky for Ship, Rose for Drone).
  * High-contrast tactical label badge displaying recognized class and percentage score.
  * Detected count banner and itemized bounding box coordinate ledger.
  * Interactive detection threshold slider for operational calibration.

---

## 9. Limitations & Caveats

1. **CPU Latency:** On CPU architectures, Grounding DINO requires ~12–19 seconds per 1200×896 image. Real-time tactical video feeds require GPU acceleration.
2. **Zero-Shot Boundary Precision:** Bounding boxes are predicted via zero-shot grounding queries without task-specific fine-tuning on defense assets, which may result in looser boxes around camouflage or partial occlusions.
3. **No Ground-Truth Annotations:** The project dataset does not include spatial bounding boxes, so quantitative Mean Average Precision (mAP) or Average Precision (AP) cannot be scientifically reported.
