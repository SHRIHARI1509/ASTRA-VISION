# Phase 9: Supplied Starter Dataset Integrity & Guardrails Audit

**Document ID:** `AV-DAT-09E`  
**Phase:** 9.5 Dataset Integrity  
**Date:** October 2026  
**Auditor:** Final Release Engineer  
**Status:** PASS (PRISTINE & UNTOUCHED)  

---

## 1. Executive Summary

An exhaustive integrity and provenance audit was conducted on the supplied reconnaissance starter dataset located in [`data/supplied_dataset/`](file:///c:/FILES/astra-vision/data/supplied_dataset/). The audit confirmed that the original dataset remains strictly read-only, perfectly intact, and free from synthetic contamination or unauthorized alterations.

* **Dataset Path:** `data/supplied_dataset/`
* **Total Image Count:** Exactly **150 images** (all decodable, non-zero byte size)
* **Metadata Files:** [`labels.csv`](file:///c:/FILES/astra-vision/data/supplied_dataset/labels.csv) (150 rows), [`credits.csv`](file:///c:/FILES/astra-vision/data/supplied_dataset/credits.csv) (150 rows)
* **Annotation Modality:** Classification labels only (Zero bounding boxes, zero masks)
* **Integrity Status:** **100% UNMODIFIED & AUDITED (PASS)**

---

## 2. Directory Structure & File Inventory

The dataset adheres to the five-class source folder structure:

```
data/supplied_dataset/
├── README.md              (Original starter documentation & provenance rules)
├── labels.csv             (150 rows, mapping relative image path to source category)
├── credits.csv            (150 rows, detailing Wikimedia author, license, and URL)
└── images/
    ├── aircraft/          (30 images: 26 JPEG, 4 PNG)
    ├── drone/             (30 images: 28 JPEG, 2 PNG)
    ├── helicopter/        (30 images: 30 JPEG)
    ├── military-vehicle/  (30 images: 29 JPEG, 1 PNG)
    └── naval/             (30 images: 26 JPEG, 4 PNG)
```

* **Image Validation:** Every image is decodable by Pillow, has short dimension $\ge 300\text{ px}$, and contains no corrupted byte sequences.
* **Filename Integrity:** 0 duplicate filenames between or within categories.
* **1:1 Metadata Mapping:** Every row in `labels.csv` matches an exact corresponding entry in `credits.csv` with identical `file_name` and `category`.

---

## 3. Ground-Truth Bounding Boxes / Masks Guardrail

* **Ground-Truth Geometry:** **NONE.** The dataset contains **only image-level category classifications**.
* **Detection Infrastructure Boundary:** While Phase 8E and Phase 8F established data schemas and readiness evaluators (`backend/app/detection/`), no pseudo-bounding boxes or synthetic detection labels were injected into the starter dataset.
* **Original CSV Purity:** Neither `labels.csv` nor `credits.csv` has been modified to insert bounding coordinates or relabeled categories.

---

## 4. Taxonomy Discrepancy & Reconciliation History

The audit checked for consistency across historical reports:
* **Supplied 5 Categories:** `aircraft`, `drone`, `helicopter`, `military-vehicle`, `naval`
* **Production 6 Classes:** `Tank`, `Military Vehicle`, `Fighter Aircraft`, `Helicopter`, `Ship`, `Drone`
* **Historical Discrepancy Observation:**
  * In early exploratory notes (Phase 8A/8B), naive category mappings were discussed (`military-vehicle -> Tank` or `naval -> Ship`).
  * In Phase 8F/8G, an exhaustive human-in-the-loop audit revealed that `military-vehicle` is a mixture of 15 Main Battle Tanks and 15 Support/Transport vehicles, while `aircraft` contains 24 combat jets and 6 non-combat/out-of-taxonomy items.
  * **Integrity Guarantee:** The original `labels.csv` was **NOT modified** to resolve this. Instead, all re-mappings and exclusions were isolated strictly within external experiment manifests ([`data/experiments/reconciliation_report.json`](file:///c:/FILES/astra-vision/data/experiments/reconciliation_report.json)).

---

## 5. Summary Checklist

| Verification Item | Requirement | Measured Result | Status |
| :--- | :--- | :--- | :---: |
| Image Count | Exactly 150 files | 150 files verified | **PASS** |
| Category Count | Exactly 5 source folders | 5 folders verified | **PASS** |
| Per-Category Balance | Exactly 30 images each | 30 / 30 / 30 / 30 / 30 | **PASS** |
| `labels.csv` Count | Exactly 150 rows | 150 rows | **PASS** |
| `credits.csv` Count | Exactly 150 rows | 150 rows | **PASS** |
| Filename Parity | 1:1 match between CSVs | 100% matched | **PASS** |
| Corrupted Images | 0 files | 0 files | **PASS** |
| Synthetic Boxes/Masks | None | None | **PASS** |
| Original CSV Changes | None | None | **PASS** |
