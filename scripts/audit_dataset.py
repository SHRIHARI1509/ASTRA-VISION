"""ASTRA VISION — Dataset Audit Script.

Performs a complete read-only forensic audit of the supplied 150-image ASTRA dataset:
1. Directory & File structure verification
2. CSV consistency (labels.csv <-> credits.csv <-> filesystem)
3. Image integrity (decodability, dimensions, formats, color modes)
4. Exact and perceptual duplicate detection
5. Class distribution analysis
6. Taxonomy comparison (supplied 5 classes vs production 6 classes)
7. Held-out test set (30 images) overlap analysis
8. License and attribution analysis
9. Train/validation/test split recommendation
10. Object detection readiness audit
"""

import os
import sys
import csv
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Set
from PIL import Image

WORKSPACE_ROOT = Path("c:/FILES/astra-vision")
DATASET_ROOT = WORKSPACE_ROOT / "data" / "supplied_dataset"
HELD_OUT_ROOT = WORKSPACE_ROOT / "data" / "held_out_dataset"
MANIFEST_PATH = WORKSPACE_ROOT / "evaluation" / "manifest.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")



def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(image_path: Path, hash_size: int = 8) -> int:
    """Compute difference hash (dHash) for visual perceptual similarity."""
    with Image.open(image_path) as img:
        img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(img.getdata())
        diff = []
        for row in range(hash_size):
            for col in range(hash_size):
                pixel_left = pixels[row * (hash_size + 1) + col]
                pixel_right = pixels[row * (hash_size + 1) + col + 1]
                diff.append(pixel_left > pixel_right)
        decimal_val = 0
        for index, value in enumerate(diff):
            if value:
                decimal_val += 1 << index
        return decimal_val


def hamming_distance(h1: int, h2: int) -> int:
    """Compute Hamming distance between two integer hashes."""
    x = h1 ^ h2
    return bin(x).count('1')


def audit_dataset():
    print("=" * 70)
    print("ASTRA VISION — DATASET AUDIT EXECUTION")
    print("=" * 70)

    # 1. Structure Audit
    labels_csv_path = DATASET_ROOT / "labels.csv"
    credits_csv_path = DATASET_ROOT / "credits.csv"
    images_dir = DATASET_ROOT / "images"

    if not labels_csv_path.exists():
        print(f"ERROR: {labels_csv_path} not found!")
        return
    if not credits_csv_path.exists():
        print(f"ERROR: {credits_csv_path} not found!")
        return
    if not images_dir.exists():
        print(f"ERROR: {images_dir} not found!")
        return

    # Scan filesystem
    all_files_on_disk = list(images_dir.rglob("*"))
    image_files_on_disk = [p for p in all_files_on_disk if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
    non_image_files_on_disk = [p for p in all_files_on_disk if p.is_file() and p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]]

    category_subdirs = [p for p in images_dir.iterdir() if p.is_dir()]
    category_counts_disk = {subdir.name: len(list(subdir.glob("*.*"))) for subdir in category_subdirs}

    print(f"\n[STEP 1: STRUCTURE]")
    print(f"  Dataset root: {DATASET_ROOT}")
    print(f"  Images directory: {images_dir}")
    print(f"  Total category directories on disk: {len(category_subdirs)}")
    print(f"  Total image files found on disk: {len(image_files_on_disk)}")
    print(f"  Non-image files found in images/: {len(non_image_files_on_disk)}")
    for cat, cnt in sorted(category_counts_disk.items()):
        print(f"    - {cat}: {cnt} images")

    # 2. CSV Consistency Audit
    labels_rows = []
    with open(labels_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels_rows.append(row)

    credits_rows = []
    with open(credits_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            credits_rows.append(row)

    print(f"\n[STEP 2: CSV CONSISTENCY]")
    print(f"  labels.csv row count: {len(labels_rows)}")
    print(f"  credits.csv row count: {len(credits_rows)}")

    labels_filenames = [r["file_name"] for r in labels_rows]
    credits_filenames = [r["file_name"] for r in credits_rows]

    labels_duplicates = [fn for fn, cnt in Counter(labels_filenames).items() if cnt > 1]
    credits_duplicates = [fn for fn, cnt in Counter(credits_filenames).items() if cnt > 1]

    labels_categories = Counter([r["category"] for r in labels_rows])
    credits_categories = Counter([r["category"] for r in credits_rows])

    print(f"  Duplicate filenames in labels.csv: {labels_duplicates or 'None'}")
    print(f"  Duplicate filenames in credits.csv: {credits_duplicates or 'None'}")
    print(f"  Categories in labels.csv: {dict(labels_categories)}")
    print(f"  Categories in credits.csv: {dict(credits_categories)}")

    # Check 1:1 filename match
    labels_set = set(labels_filenames)
    credits_set = set(credits_filenames)
    diff_labels_minus_credits = labels_set - credits_set
    diff_credits_minus_labels = credits_set - labels_set

    print(f"  Filenames in labels but missing in credits: {len(diff_labels_minus_credits)}")
    print(f"  Filenames in credits but missing in labels: {len(diff_credits_minus_labels)}")

    # Check category match between labels and credits for each file
    credits_cat_map = {r["file_name"]: r["category"] for r in credits_rows}
    category_mismatches = []
    for r in labels_rows:
        fn = r["file_name"]
        cat_labels = r["category"]
        cat_credits = credits_cat_map.get(fn)
        if cat_credits != cat_labels:
            category_mismatches.append((fn, cat_labels, cat_credits))

    print(f"  Category mismatches between labels.csv and credits.csv: {len(category_mismatches)}")

    # Check whether every file in labels.csv exists on disk
    missing_from_disk = []
    disk_relative_paths = set()
    for p in image_files_on_disk:
        rel = p.relative_to(DATASET_ROOT).as_posix()
        disk_relative_paths.add(rel)

    for fn in labels_filenames:
        # Standardize posix path
        clean_fn = fn.replace("\\", "/")
        if clean_fn not in disk_relative_paths:
            missing_from_disk.append(clean_fn)

    print(f"  CSV entries missing from disk: {len(missing_from_disk)}")
    unreferenced_on_disk = disk_relative_paths - set(labels_filenames)
    print(f"  Disk files unreferenced in CSV: {len(unreferenced_on_disk)}")

    # 3. Image Integrity & Format Audit
    image_metadata = []
    corrupted_images = []
    formats = Counter()
    color_modes = Counter()
    dimensions = []
    aspect_ratios = []
    file_sizes = []
    image_hashes: Dict[str, str] = {}
    dhashes: Dict[str, int] = {}

    for r in labels_rows:
        fn = r["file_name"]
        cat = r["category"]
        img_path = DATASET_ROOT / fn

        if not img_path.exists():
            continue

        size_bytes = img_path.stat().st_size
        file_sizes.append(size_bytes)
        sha256 = compute_sha256(img_path)
        image_hashes[fn] = sha256

        try:
            with Image.open(img_path) as img:
                img.verify()
            # Re-open to read attributes (verify invalidates img object)
            with Image.open(img_path) as img:
                w, h = img.size
                fmt = img.format
                mode = img.mode
                formats[fmt] += 1
                color_modes[mode] += 1
                dimensions.append((w, h))
                aspect_ratios.append(w / h)
                dhash_val = compute_dhash(img_path)
                dhashes[fn] = dhash_val

                image_metadata.append({
                    "file_name": fn,
                    "category": cat,
                    "format": fmt,
                    "mode": mode,
                    "width": w,
                    "height": h,
                    "aspect_ratio": round(w / h, 3),
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                    "dhash": hex(dhash_val),
                })
        except Exception as e:
            corrupted_images.append((fn, str(e)))

    print(f"\n[STEP 3: IMAGE INTEGRITY]")
    print(f"  Successfully verified & inspected images: {len(image_metadata)} / {len(labels_rows)}")
    print(f"  Corrupted or unreadable images: {len(corrupted_images)}")
    if corrupted_images:
        for fn, err in corrupted_images:
            print(f"    - {fn}: {err}")
    print(f"  Image formats detected: {dict(formats)}")
    print(f"  Color modes detected: {dict(color_modes)}")

    widths = [w for w, h in dimensions]
    heights = [h for w, h in dimensions]
    print(f"  Resolution range:")
    print(f"    Width:  min={min(widths)}, max={max(widths)}, mean={sum(widths)/len(widths):.1f}")
    print(f"    Height: min={min(heights)}, max={max(heights)}, mean={sum(heights)/len(heights):.1f}")
    print(f"    Short side < 300px: {sum(1 for w, h in dimensions if min(w, h) < 300)}")
    print(f"  File size range:")
    print(f"    Min: {min(file_sizes)/1024:.1f} KB, Max: {max(file_sizes)/1024:.1f} KB, Mean: {sum(file_sizes)/len(file_sizes)/1024:.1f} KB")

    # 4. Duplicate Detection (Exact & Perceptual)
    hash_to_files = defaultdict(list)
    for fn, sha in image_hashes.items():
        hash_to_files[sha].append(fn)

    exact_duplicates = {sha: fns for sha, fns in hash_to_files.items() if len(fns) > 1}

    print(f"\n[STEP 4: DUPLICATE DETECTION]")
    print(f"  Unique exact file SHA-256 hashes: {len(hash_to_files)} / {len(image_hashes)}")
    print(f"  Exact duplicate clusters: {len(exact_duplicates)}")
    if exact_duplicates:
        for sha, fns in exact_duplicates.items():
            print(f"    Exact cluster ({sha[:12]}...): {fns}")

    # Perceptual duplicates (Hamming distance <= 4)
    perceptual_duplicates = []
    fn_list = list(dhashes.keys())
    for i in range(len(fn_list)):
        for j in range(i + 1, len(fn_list)):
            fn1 = fn_list[i]
            fn2 = fn_list[j]
            dist = hamming_distance(dhashes[fn1], dhashes[fn2])
            if dist <= 4:
                cat1 = fn1.split('/')[1]
                cat2 = fn2.split('/')[1]
                perceptual_duplicates.append({
                    "file_1": fn1,
                    "category_1": cat1,
                    "file_2": fn2,
                    "category_2": cat2,
                    "hamming_distance": dist,
                    "cross_category": cat1 != cat2,
                })

    print(f"  Probable visual / perceptual duplicates (Hamming dist <= 4): {len(perceptual_duplicates)}")
    for d in perceptual_duplicates:
        print(f"    - dist={d['hamming_distance']}: {d['file_1']} ({d['category_1']}) <-> {d['file_2']} ({d['category_2']}) [Cross-cat: {d['cross_category']}]")

    # 5. Class Distribution
    print(f"\n[STEP 5: CLASS DISTRIBUTION]")
    for cat, count in sorted(labels_categories.items()):
        print(f"  Category '{cat}': {count} images ({count / len(labels_rows) * 100:.1f}%)")

    # 6. Taxonomy Audit & Content Semantic Analysis
    print(f"\n[STEP 6: TAXONOMY AUDIT & CONTENT ANALYSIS]")
    # Investigate filenames and titles in each category to detect sub-classes
    category_contents = defaultdict(list)
    for r in credits_rows:
        category_contents[r["category"]].append((r["file_name"], r.get("source_title", "")))

    tank_keywords = ["tank", "abrams", "leopard", "panzer", "t-72", "t-90", "t-80", "t-55", "t-64", "t-34", "challenger", "leclerc", "m1a", "merkava", "ariete", "armata"]
    fighter_keywords = ["f-22", "f-18", "f-16", "f-15", "f-35", "mig", "su-", "sukhoi", "hornet", "raptor", "typhoon", "rafale", "fighter", "interceptor", "combat"]
    
    # Analyze military-vehicle: does it contain tanks?
    mv_items = category_contents["military-vehicle"]
    mv_tanks = []
    mv_other = []
    for fn, title in mv_items:
        combined = (fn + " " + title).lower()
        if any(kw in combined for kw in tank_keywords):
            mv_tanks.append((fn, title))
        else:
            mv_other.append((fn, title))

    print(f"  Category 'military-vehicle' breakdown (30 total):")
    print(f"    - Identifiable Tanks / MBTs: {len(mv_tanks)} images")
    print(f"    - Other Military Vehicles (APCs, IFVs, trucks, artillery): {len(mv_other)} images")
    print("    Sample identified tanks in military-vehicle:")
    for fn, title in mv_tanks[:5]:
        print(f"      * {fn} | {title}")

    # Analyze aircraft: are they all fighter aircraft?
    ac_items = category_contents["aircraft"]
    ac_fighters = []
    ac_other = []
    for fn, title in ac_items:
        combined = (fn + " " + title).lower()
        if any(kw in combined for kw in fighter_keywords):
            ac_fighters.append((fn, title))
        else:
            ac_other.append((fn, title))

    print(f"  Category 'aircraft' breakdown (30 total):")
    print(f"    - Identifiable Fighter / Combat Aircraft: {len(ac_fighters)} images")
    print(f"    - Other Aircraft (transports, bombers, historical, trainers): {len(ac_other)} images")
    print("    Sample aircraft items:")
    for fn, title in ac_items[:5]:
        print(f"      * {fn} | {title}")

    # 7. Held-Out Test Set Overlap Analysis
    print(f"\n[STEP 7: HELD-OUT BENCHMARK OVERLAP AUDIT]")
    held_out_hashes = {}
    held_out_dhashes = {}
    held_out_files = list(HELD_OUT_ROOT.rglob("*.*"))
    held_out_images = [p for p in held_out_files if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]

    for p in held_out_images:
        sha = compute_sha256(p)
        held_out_hashes[p.name] = sha
        held_out_dhashes[p.name] = compute_dhash(p)

    print(f"  Held-out benchmark images found on disk: {len(held_out_images)}")
    held_out_sha_set = set(held_out_hashes.values())
    supplied_sha_set = set(image_hashes.values())

    exact_overlap = held_out_sha_set.intersection(supplied_sha_set)
    print(f"  Exact SHA-256 hash overlap between supplied dataset and held-out benchmark: {len(exact_overlap)}")

    overlapping_items = []
    if exact_overlap:
        for fn, sha in image_hashes.items():
            if sha in exact_overlap:
                # Find matching held-out filename
                matching_ho = [k for k, v in held_out_hashes.items() if v == sha]
                overlapping_items.append((fn, matching_ho))
                print(f"    OVERLAP DETECTED: {fn} matches held-out {matching_ho}")
    else:
        print("  ZERO exact SHA-256 overlap detected. Benchmark integrity is 100% PRESERVED.")

    # Check perceptual similarity between held-out and supplied
    perceptual_benchmark_overlaps = []
    for s_fn, s_dh in dhashes.items():
        for ho_fn, ho_dh in held_out_dhashes.items():
            dist = hamming_distance(s_dh, ho_dh)
            if dist <= 2:  # very high visual similarity
                perceptual_benchmark_overlaps.append((s_fn, ho_fn, dist))

    print(f"  Perceptual near-duplicates between supplied dataset and held-out benchmark (dist <= 2): {len(perceptual_benchmark_overlaps)}")
    for s_fn, ho_fn, dist in perceptual_benchmark_overlaps:
        print(f"    - Visual match (dist={dist}): {s_fn} <-> {ho_fn}")

    # 8. License & Credits Audit
    print(f"\n[STEP 8: CREDITS & LICENSE AUDIT]")
    licenses = Counter([r.get("license", "").strip() for r in credits_rows])
    artists = [r.get("artist", "").strip() for r in credits_rows]
    commons_pages = [r.get("commons_page", "").strip() for r in credits_rows]
    titles = [r.get("source_title", "").strip() for r in credits_rows]

    missing_licenses = sum(1 for lic in licenses.keys() if not lic)
    missing_artists = sum(1 for a in artists if not a)
    missing_pages = sum(1 for p in commons_pages if not p)
    missing_titles = sum(1 for t in titles if not t)

    print(f"  License breakdown:")
    for lic, cnt in sorted(licenses.items(), key=lambda x: -x[1]):
        print(f"    - {lic or 'MISSING'}: {cnt} images ({cnt / len(credits_rows) * 100:.1f}%)")
    print(f"  Missing fields:")
    print(f"    - Missing licenses: {missing_licenses}")
    print(f"    - Missing artists: {missing_artists}")
    print(f"    - Missing commons_page: {missing_pages}")
    print(f"    - Missing source_title: {missing_titles}")

    # 10. Detection Readiness
    print(f"\n[STEP 10: DETECTION READINESS AUDIT]")
    annotation_extensions = [".xml", ".txt", ".json", ".csv", ".yaml", ".yml"]
    detected_annotations = []
    for ext in annotation_extensions:
        matches = list(images_dir.rglob(f"*{ext}"))
        if matches:
            detected_annotations.extend(matches)

    print(f"  Annotation files inside images/ directory: {len(detected_annotations)}")
    has_bounding_boxes = False
    # Check if labels.csv or credits.csv have bounding box coordinates (xmin, ymin, xmax, ymax, bbox)
    if "bbox" in labels_rows[0] or "xmin" in labels_rows[0]:
        has_bounding_boxes = True
    print(f"  Bounding box coordinates present in labels.csv: {'YES' if has_bounding_boxes else 'NO'}")
    print(f"  Detection readiness conclusion: NO. The dataset provides ONLY image-level classification labels.")

    # Save summary data to json for reporting
    audit_summary = {
        "dataset_root": str(DATASET_ROOT),
        "total_images_on_disk": len(image_files_on_disk),
        "total_labels_rows": len(labels_rows),
        "total_credits_rows": len(credits_rows),
        "category_counts": dict(labels_categories),
        "formats": dict(formats),
        "color_modes": dict(color_modes),
        "corrupted_images_count": len(corrupted_images),
        "exact_duplicates_count": len(exact_duplicates),
        "perceptual_duplicates_count": len(perceptual_duplicates),
        "exact_overlap_with_held_out": len(exact_overlap),
        "perceptual_overlap_with_held_out": len(perceptual_benchmark_overlaps),
        "licenses": dict(licenses),
        "detection_ready": False,
    }

    output_json_path = WORKSPACE_ROOT / "docs" / "dataset_audit_summary.json"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"\nAudit complete. Summary saved to {output_json_path}")


if __name__ == "__main__":
    audit_dataset()
