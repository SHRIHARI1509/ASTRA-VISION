# Phase 9: Frozen Held-Out Benchmark Integrity Report

**Document ID:** `AV-BMK-09D`  
**Phase:** 9.4 Frozen Benchmark Integrity  
**Date:** October 2026  
**Auditor:** Final Release Engineer  
**Status:** FULLY VERIFIED & UNCOMPROMISED  

---

## 1. Executive Summary

The sacred 30-image held-out benchmark set established in Phase 8A and sealed in [`evaluation/manifest.json`](file:///c:/FILES/astra-vision/evaluation/manifest.json) was audited against all local files, training sets, and fine-tuning splits.

* **Benchmark Root:** [`data/held_out_dataset/`](file:///c:/FILES/astra-vision/data/held_out_dataset/)
* **Manifest Path:** [`evaluation/manifest.json`](file:///c:/FILES/astra-vision/evaluation/manifest.json)
* **Total Image Count:** Exactly **30 images**
* **Per-Class Distribution:** Exactly **5 images per production class** across all 6 classes
* **Hash Verification:** 30/30 cryptographic SHA-256 matches against sealed manifest
* **Train / Val / Fine-Tuning Overlap:** **0.0% (Zero Leakage)**
* **Benchmark Integrity Status:** **SEALED & UNTOUCHED (PASS)**

---

## 2. Benchmark Class Distribution

The benchmark set is strictly balanced across the 6 production categories:

| Class Taxonomy | Directory Path | Image Count | Formats Present |
| :--- | :--- | :---: | :--- |
| `Tank` | `data/held_out_dataset/Tank/` | 5 | 5 JPEG |
| `Military Vehicle` | `data/held_out_dataset/Military Vehicle/` | 5 | 5 JPEG |
| `Fighter Aircraft` | `data/held_out_dataset/Fighter Aircraft/` | 5 | 5 JPEG |
| `Helicopter` | `data/held_out_dataset/Helicopter/` | 5 | 5 JPEG |
| `Ship` | `data/held_out_dataset/Ship/` | 5 | 4 JPEG, 1 PNG |
| `Drone` | `data/held_out_dataset/Drone/` | 5 | 4 JPEG, 1 WEBP |
| **Total** | | **30** | **28 JPEG, 1 PNG, 1 WEBP** |

---

## 3. Cryptographic Hash & Overlap Verification

1. **Supplied Dataset Non-Overlap:**
   * An exhaustive SHA-256 cross-hash between `data/held_out_dataset/` and the 150 starter images in `data/supplied_dataset/images/` yielded **0 matches**.
2. **Phase 8G Fine-Tuning Non-Overlap:**
   * Cross-comparison against [`data/experiments/train_manifest.json`](file:///c:/FILES/astra-vision/data/experiments/train_manifest.json) (115 images) and [`data/experiments/val_manifest.json`](file:///c:/FILES/astra-vision/data/experiments/val_manifest.json) (29 images) verified that no benchmark image was ever included in training, validation, or hyperparameter selection.
3. **Immutability Assurance:**
   * File timestamps and file sizes match `evaluation/manifest.json` exactly.
   * No labels or annotations were adjusted or retuned.

---

## 4. Integrity Gate Verdict

| Audit Check | Requirement | Result | Status |
| :--- | :--- | :--- | :---: |
| Image Count | Exactly 30 images | 30 images verified | **PASS** |
| Distribution | 5 per class (6 classes) | 5/5/5/5/5/5 verified | **PASS** |
| File Integrity | Decodable by PIL without error | 30/30 decodable | **PASS** |
| Starter Dataset Leakage | 0 shared images | 0 hash matches | **PASS** |
| Fine-Tuning Train Leakage | 0 shared images | 0 hash matches | **PASS** |
| Fine-Tuning Val Leakage | 0 shared images | 0 hash matches | **PASS** |
| Manifest Synchronization | 1:1 match with manifest.json | 100% matched | **PASS** |
