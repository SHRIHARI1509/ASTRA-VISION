# ASTRA VISION — DATASET AUDIT & PREPARATION REPORT

**Document ID:** `AV-AUD-001`  
**Date:** October 2026  
**Status:** COMPLETE (READ-ONLY AUDIT)  
**Target Dataset:** `challenge-02-vision` (150 starter images)  
**Destination in Workspace:** `data/supplied_dataset/`  

---

## Executive Summary

| Audit Item | Status | Key Metric / Result |
| :--- | :---: | :--- |
| **Dataset Structure** | **PASS** | 150 images across 5 category subdirectories; 0 missing or extraneous files |
| **CSV Consistency** | **PASS** | `labels.csv` (150 rows) $\leftrightarrow$ `credits.csv` (150 rows) 1:1 match; 0 mismatches |
| **Image Integrity** | **PASS** | 150/150 images decodable by PIL; 0 corrupted, 0 zero-byte, 0 unreadable |
| **Duplicate Audit** | **PASS** | 150 unique SHA-256 hashes; 0 exact duplicates; 0 perceptual duplicates ($d \le 4$) |
| **Taxonomy Audit** | **FLAGGED / REVIEW REQUIRED** | 5 supplied classes vs. 6 production classes; critical overlap in Tanks and Fighters |
| **Held-Out Test Overlap** | **PASS (100% CLEAN)** | 0 exact SHA-256 matches; 0 perceptual matches against 30-image held-out benchmark |
| **Credits & Licenses** | **PASS** | 100% of images documented; 76 Public Domain, 38 CC BY-SA 4.0, 13 CC0, 23 other CC |
| **Detection Readiness** | **NO** | Zero bounding boxes, masks, or spatial annotations present |

---

## 1. Dataset Overview

The ASTRA challenge starter dataset consists of **150 labelled images** sourced from Wikimedia Commons representing aerospace and defence systems. The dataset was imported safely into the workspace at `data/supplied_dataset/` in a strictly read-only manner.

Accompanying metadata files:
1. `labels.csv`: Ground-truth mapping connecting `file_name` to `category`.
2. `credits.csv`: Attribution metadata containing `file_name`, `category`, `source_title`, `artist`, `license`, and `commons_page`.
3. `README.md`: General starter notes.

---

## 2. File Structure

The dataset adheres to a clean, well-organized hierarchy:

```
data/supplied_dataset/
├── credits.csv                # 150 rows (plus header)
├── labels.csv                 # 150 rows (plus header)
├── README.md                  # Dataset notes
└── images/
    ├── aircraft/              # 30 image files
    ├── drone/                 # 30 image files
    ├── helicopter/            # 30 image files
    ├── military-vehicle/      # 30 image files
    └── naval/                 # 30 image files
```

### Quantitative Metrics
* **Total Category Directories on Disk:** 5
* **Total Image Files on Disk:** 150
* **Non-Image / Extraneous Files in `images/`:** 0
* **Missing Files Referenced by CSVs:** 0
* **Unreferenced Files on Disk:** 0

---

## 3. Class Distribution

The dataset exhibits uniform class balance across all 5 provided categories:

| Category | Images on Disk | `labels.csv` Rows | `credits.csv` Rows | Percentage |
| :--- | :---: | :---: | :---: | :---: |
| `aircraft` | 30 | 30 | 30 | 20.0% |
| `drone` | 30 | 30 | 30 | 20.0% |
| `helicopter` | 30 | 30 | 30 | 20.0% |
| `military-vehicle` | 30 | 30 | 30 | 20.0% |
| `naval` | 30 | 30 | 30 | 20.0% |
| **Total** | **150** | **150** | **150** | **100.0%** |

---

## 4. CSV Consistency

Both CSV manifests were verified programmatically using strict cross-referencing:

1. **Row Count Verification:**
   * `labels.csv`: 150 data rows.
   * `credits.csv`: 150 data rows.
2. **Key Consistency (`file_name`):**
   * Filenames in `labels.csv` missing from `credits.csv`: **0**
   * Filenames in `credits.csv` missing from `labels.csv`: **0**
   * Duplicate filenames in `labels.csv`: **0**
   * Duplicate filenames in `credits.csv`: **0**
3. **Category Consistency:**
   * Category mismatches between `labels.csv` and `credits.csv`: **0**
4. **Filesystem Cross-Check:**
   * Every file path referenced (e.g. `images/aircraft/001-...jpg`) resolves to an existing file on disk.

---

## 5. Image Integrity & Telemetry

Every image was opened and decoded using Pillow (`PIL.Image`).

### Formats & Color Modes
* **Container Formats:**
  * JPEG: 135 files (90.0%)
  * PNG: 15 files (10.0%)
* **Color Modes:**
  * `RGB`: 116 files (77.3%)
  * `L` (Grayscale): 25 files (16.7%) — Historical military archives and monochromatic FLIR/UAV captures.
  * `RGBA`: 9 files (6.0%) — Rendered or masked PNGs containing alpha transparency channels.
  * *Note: The existing ASTRA VISION `ImagePreprocessor` pipeline automatically converts RGBA and Grayscale inputs to 3-channel RGB before tensor generation.*

### Resolution & File Size Metrics
* **Width:** Min = 300 px, Max = 960 px, Mean = 925.9 px
* **Height:** Min = 300 px, Max = 2059 px, Mean = 680.2 px
* **Shortest Dimension:** Min = 300 px (100% of images satisfy minimum resolution $\ge 300\times 300$)
* **File Size on Disk:** Min = 19.2 KB, Max = 859.0 KB, Mean = 185.2 KB (Total footprint = 27.78 MB)
* **Corrupted / Zero-byte / Unreadable Files:** **0**

---

## 6. Duplicate Detection

Two deterministic duplicate detection methodologies were conducted:

1. **Exact Duplicate Analysis (SHA-256):**
   * All 150 images yielded unique SHA-256 cryptographic hashes.
   * **Exact duplicates: 0**.
2. **Perceptual Duplicate Analysis (dHash 8x8 Difference Hash):**
   * Evaluated pairwise Hamming distances across all $\binom{150}{2} = 11,175$ image pairs.
   * Pairs with Hamming distance $d \le 4$: **0**.
   * **Visual / Perceptual duplicates: 0**.
   * **Cross-category duplicates: 0**.

---

## 7. Taxonomy Analysis (Supplied vs. Production)

### Background
The current ASTRA VISION production classifier uses **six target classes**:
1. `Tank`
2. `Military Vehicle`
3. `Fighter Aircraft`
4. `Helicopter`
5. `Ship`
6. `Drone`

The supplied dataset provides **five target classes**:
`aircraft`, `drone`, `helicopter`, `military-vehicle`, `naval`.

### Detailed Mapping & Ambiguity Matrix

| Supplied Category | Production Equivalent | Ambiguity Level | Evidence & Findings | Manual Review Required? |
| :--- | :--- | :---: | :--- | :---: |
| `drone` | `Drone` | **None** | High-altitude UAVs, multi-rotor quadcopters, reconnaissance drones. | **No** (Direct 1:1 match) |
| `helicopter` | `Helicopter` | **None** | Rotary-wing attack, transport, and reconnaissance helicopters. | **No** (Direct 1:1 match) |
| `naval` | `Ship` | **None** | Aircraft carriers, guided-missile destroyers, frigates, patrol crafts. | **No** (Direct 1:1 match) |
| `military-vehicle` | `Tank` vs. `Military Vehicle` | **CRITICAL** | Contains **6 Main Battle Tanks** (T-72B3, M3 Lee/Grant, M4 Sherman, Dutch Panzerhaubitze, Czołg T-72) and **24 other military vehicles** (APCs, MRAPs, IFVs, trucks). In production, `Tank` is separated from `Military Vehicle`. | **YES (Mandatory relabeling required)** |
| `aircraft` | `Fighter Aircraft` | **CRITICAL** | Contains **13 Fighter Jets** (F-22 Raptor, F/A-18 Hornet, TEDBF, MiG-29K, MiG-21) and **17 Non-Fighter Aircraft** (C-130 transports, B-52 bombers, trainer planes, civil craft). Production class is strictly `Fighter Aircraft`. | **YES (Mandatory filtering / relabeling required)** |

### Taxonomy Finding
**Blindly fine-tuning on the supplied 5-class dataset would corrupt the production 6-class model.** Specifically:
1. Merging tanks into `military-vehicle` would degrade the model's ability to differentiate `Tank` from `Military Vehicle`.
2. Including cargo/transport aircraft in `aircraft` would degrade the specificity of `Fighter Aircraft`.

---

## 8. Held-Out Benchmark Overlap Audit

The project maintains an official, frozen 30-image held-out benchmark in `data/held_out_dataset/` (5 images per class $\times$ 6 production classes), documented in `data/held_out_dataset/manifest.json`.

* **Exact SHA-256 Hash Overlap:** **0 / 30** (0.0%)
* **Perceptual Near-Duplicate Overlap ($d \le 2$):** **0 / 30** (0.0%)
* **Benchmark Integrity:** **100% CLEAN & PROTECTED**.

> [!IMPORTANT]
> The held-out evaluation benchmark remains completely uncontaminated by the supplied dataset. None of the 150 starter images appear in the 30-image benchmark.

---

## 9. Credits & License Audit

Attribution was verified against `credits.csv`:

* **Attribution Completeness:**
  * Missing Licenses: 0
  * Missing Artists / Creators: 0
  * Missing Wikimedia Commons URLs: 0
  * Missing Source Titles: 0

### License Breakdown
| License | Image Count | Percentage | Deployment Notes |
| :--- | :---: | :---: | :--- |
| **Public Domain** | 76 | 50.7% | Unrestricted use; US DoD/US Gov works |
| **CC BY-SA 4.0** | 38 | 25.3% | Attribution required; Share-Alike on adaptations |
| **CC0 (Public Domain Dedication)** | 13 | 8.7% | Universal waiver; unrestricted |
| **CC BY 4.0** | 7 | 4.7% | Attribution required |
| **CC BY 2.0** | 6 | 4.0% | Attribution required |
| **CC BY-SA 3.0** | 4 | 2.7% | Attribution required; Share-Alike |
| **CC BY-SA 2.0** | 3 | 2.0% | Attribution required; Share-Alike |
| **Attribution** | 1 | 0.7% | Attribution required |
| **CC BY-SA 3.0 de** | 1 | 0.7% | German jurisdiction CC BY-SA |
| **No restrictions** | 1 | 0.7% | Unrestricted |
| **Total** | **150** | **100.0%** | **Compliant** |

All images are freely usable for research, evaluation, and internal AI training under their respective open-source licenses.

---

## 10. Proposed Train/Validation/Test Split Strategy

To preserve rigorous ML evaluation practices without data leakage:

### Option A: Direct 5-Class Evaluation (Interim Analysis Only)
* **Training Set:** 120 images (80%, 24 per class)
* **Validation Set:** 30 images (20%, 6 per class)
* **Limitation:** Cannot be tested against the existing 6-class held-out benchmark.

### Option B: Production-Aligned 6-Class Split (RECOMMENDED)
1. **Taxonomy Alignment:**
   * Split `military-vehicle` $\rightarrow$ `Tank` (6 images) + `Military Vehicle` (24 images).
   * Split `aircraft` $\rightarrow$ `Fighter Aircraft` (13 images) + other aircraft archived or reassigned.
   * `drone` $\rightarrow$ `Drone` (30 images).
   * `helicopter` $\rightarrow$ `Helicopter` (30 images).
   * `naval` $\rightarrow$ `Ship` (30 images).
2. **Dataset Augmentation for Balanced Split:**
   * Supplement the under-represented `Tank` class (+15–20 images) and `Fighter Aircraft` class (+10–15 images).
3. **Split Proportions:**
   * **Train:** 80% stratified across 6 classes.
   * **Validation:** 20% stratified across 6 classes.
   * **Held-out Test Set:** The **existing frozen 30-image held-out benchmark** (`data/held_out_dataset/`) serves as the official test set!

---

## 11. Detection Readiness

* **Bounding Boxes:** None
* **Segmentation Masks:** None
* **YOLO Format Labels (`.txt`):** None
* **COCO Format Annotations (`.json`):** None
* **Pascal VOC Annotations (`.xml`):** None

**Detection Readiness: NO.**
The supplied dataset is exclusively an image-level classification dataset. It contains no bounding box coordinates or spatial localization information.

---

## 12. Risks & Limitations

1. **Small Sample Size:** 30 images per category (150 total) is susceptible to overfitting if full fine-tuning is attempted on modern vision transformers like SigLIP 2 (which has 200M+ parameters).
2. **Class Imbalance Post-Split:** Re-categorizing tanks creates an imbalance (6 tanks vs 30 drones/helicopters).
3. **Format Variations:** 25 grayscale images and 9 RGBA images require runtime format normalization (already handled by backend preprocessor).
4. **Single-Object Dominance:** Most images depict clear, centered objects under favorable lighting, unlike contested, cluttered tactical environments.

---

## 13. Recommendation for Phase 8G (Model Fine-Tuning)

1. **Do NOT fine-tune directly on the raw 5-class dataset.**
2. Prior to any training:
   * Perform manual relabeling of the 6 tanks and 13 fighter jets.
   * Supplement `Tank` with an additional 15–20 images from open military repositories to balance the distribution.
3. Use **Parameter-Efficient Fine-Tuning (PEFT / LoRA)** on the vision projection layer rather than updating the entire SigLIP 2 backbone. Full fine-tuning on 150 images would risk catastrophic forgetting of zero-shot general visual knowledge.
4. Evaluate any fine-tuned checkpoint strictly against the frozen 30-image held-out benchmark to verify improvement.

---

## 14. Recommendation for Phase 8E (Object Detection)

1. **Do NOT use this dataset for training traditional supervised detectors (YOLO, Faster R-CNN) without bounding boxes.**
2. Recommended paths forward:
   * **Path 1 (Zero-Shot Open-Vocabulary Detection):** Deploy a zero-shot detector such as **OWLv2** (`google/owlv2-base-patch16-ensemble`) or **Grounding DINO** which requires zero training bounding boxes and works out-of-the-box using text prompts.
   * **Path 2 (Annotated Dataset):** If custom detection training is strictly required, generate bounding box labels using a standard annotation tool before commencing training.

---

## Conclusion & Gate Status

```
======================================================================
ASTRA VISION DATASET AUDIT VERIFICATION GATE: PASS
======================================================================
  Dataset Audit:             PASS
  CSV Consistency:           PASS
  Image Integrity:           PASS
  Duplicate Audit:           PASS
  Taxonomy Audit:            PASS (Documented & Flagged for Relabeling)
  Held-Out Overlap Audit:    PASS (Zero contamination)
  Credits Audit:             PASS
  Detection Readiness:       NO (Image-level classification only)
======================================================================
```
