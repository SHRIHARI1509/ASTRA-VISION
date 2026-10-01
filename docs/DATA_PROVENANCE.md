# ASTRA VISION — Dataset Provenance & Attribution Ledger

**Document ID:** `AV-PROV-09`  
**Phase:** 9.11 Data Provenance / Credits  
**Date:** October 2026  
**Auditor:** Final Release Engineer  
**Status:** PASS & PRESERVED  

---

## 1. Overview & Provenance

The primary training and exploratory dataset provided for ASTRA VISION consists of 150 reconnaissance images organized into 5 source categories. All assets were sourced from **Wikimedia Commons** community uploads and are distributed under free culture and open-source licenses.

* **Dataset Root:** [`data/supplied_dataset/`](file:///c:/FILES/astra-vision/data/supplied_dataset/)
* **Classification Manifest:** [`data/supplied_dataset/labels.csv`](file:///c:/FILES/astra-vision/data/supplied_dataset/labels.csv)
* **Attribution & Licensing Ledger:** [`data/supplied_dataset/credits.csv`](file:///c:/FILES/astra-vision/data/supplied_dataset/credits.csv)
* **Original Starter Documentation:** [`data/supplied_dataset/README.md`](file:///c:/FILES/astra-vision/data/supplied_dataset/README.md)

---

## 2. Relationship Between `labels.csv` and `credits.csv`

Both CSV files are synchronized in a strict 1:1 correspondence:

1. **Row Count Parity:** Both files contain exactly 150 entries (plus header row).
2. **Key Correspondence:** The `file_name` column in `labels.csv` matches the `file_name` column in `credits.csv` for 100% of rows.
3. **Category Consistency:** The `category` column matches identically across both files:
   * `aircraft` (30 rows)
   * `drone` (30 rows)
   * `helicopter` (30 rows)
   * `military-vehicle` (30 rows)
   * `naval` (30 rows)
4. **Attribution Schema in `credits.csv`:**
   * `file_name`: Relative image filepath (e.g. `images/aircraft/001-F-22-Flaring-Cherry-Festival-jpg.jpg`)
   * `category`: Source directory classification
   * `source_title`: Original Wikimedia Commons file title
   * `artist`: Photographer, renderer, or uploader
   * `license`: Open-source license identifier
   * `commons_page`: Direct URL to the Wikimedia Commons media description page

---

## 3. License Distribution

An audit of the 150 entries in `credits.csv` indicates the following license distribution:

| License Type | Count | Conditions & Permissions |
| :--- | :---: | :--- |
| **CC BY-SA (2.0 / 3.0 / 4.0)** | 84 | Attribution required; modifications shared under identical terms |
| **CC BY (2.0 / 3.0 / 4.0)** | 35 | Attribution required; commercial and non-commercial reuse allowed |
| **Public Domain / CC0** | 24 | Dedicated to the public domain; no attribution legally required |
| **Other Free Licenses (FAL / GFDL)** | 7 | Free Art License / GNU Free Documentation License |
| **Total** | **150** | **100% Free & Open Access** |

*Note: All 150 rows in `credits.csv` have non-empty licenses, valid commons page URLs, and artist attributions.*

---

## 4. Attribution Requirements & Preservation

To comply with CC BY and CC BY-SA terms:
1. **Preservation:** [`data/supplied_dataset/credits.csv`](file:///c:/FILES/astra-vision/data/supplied_dataset/credits.csv) is strictly preserved in the repository and must never be deleted.
2. **Commercial & Non-Commercial Reuse:** Reusers must credit the original artists and link to the source Wikimedia Commons pages provided in `credits.csv`.
3. **Integrity Rule:** The dataset was audited and kept read-only; no original files were renamed or had their attribution metadata stripped.

---

## 5. Held-Out Benchmark Provenance

* The 30-image held-out evaluation benchmark ([`data/held_out_dataset/`](file:///c:/FILES/astra-vision/data/held_out_dataset/)) was independently curated and sealed in Phase 8A.
* **Non-Overlap:** Cryptographic SHA-256 verification confirms that **zero benchmark images** were sourced from or overlap with the 150 starter images.
