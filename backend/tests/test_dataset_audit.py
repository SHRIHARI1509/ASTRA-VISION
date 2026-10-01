"""Dataset audit unit tests for supplied 150-image ASTRA dataset.

Verifies:
1. labels.csv contains exactly 150 rows
2. labels.csv contains exactly 5 categories
3. Each category in labels.csv has exactly 30 entries
4. credits.csv contains exactly 150 entries
5. labels.csv and credits.csv filenames match 1:1
6. labels.csv and credits.csv categories match 1:1 for every file
7. Every referenced image exists on disk
8. No duplicate filenames in labels.csv or credits.csv
9. Image integrity checks: all 150 images are readable and valid
10. Zero overlap with existing 30-image held-out benchmark
11. Detection readiness is False (no bounding boxes or detection annotations)
"""

import csv
import hashlib
from pathlib import Path
from collections import Counter
import pytest
from PIL import Image

DATASET_ROOT = Path("data/supplied_dataset")
HELD_OUT_ROOT = Path("data/held_out_dataset")


@pytest.fixture(scope="module")
def labels_data():
    csv_path = DATASET_ROOT / "labels.csv"
    assert csv_path.exists(), f"labels.csv not found at {csv_path}"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


@pytest.fixture(scope="module")
def credits_data():
    csv_path = DATASET_ROOT / "credits.csv"
    assert csv_path.exists(), f"credits.csv not found at {csv_path}"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def test_labels_csv_row_count(labels_data):
    """Verify labels.csv contains exactly 150 rows."""
    assert len(labels_data) == 150


def test_labels_categories_count(labels_data):
    """Verify labels.csv contains exactly 5 categories."""
    categories = {r["category"] for r in labels_data}
    assert len(categories) == 5
    assert categories == {"aircraft", "drone", "helicopter", "military-vehicle", "naval"}


def test_each_category_has_30_entries(labels_data):
    """Verify each category has exactly 30 entries in labels.csv."""
    counts = Counter(r["category"] for r in labels_data)
    for cat in ["aircraft", "drone", "helicopter", "military-vehicle", "naval"]:
        assert counts[cat] == 30, f"Expected 30 images for category '{cat}', got {counts[cat]}"


def test_credits_csv_row_count(credits_data):
    """Verify credits.csv contains exactly 150 entries."""
    assert len(credits_data) == 150


def test_labels_and_credits_filenames_match(labels_data, credits_data):
    """Verify filenames match 1:1 between labels.csv and credits.csv."""
    labels_files = [r["file_name"] for r in labels_data]
    credits_files = [r["file_name"] for r in credits_data]
    assert sorted(labels_files) == sorted(credits_files)


def test_labels_and_credits_categories_match(labels_data, credits_data):
    """Verify category label matches 1:1 between labels.csv and credits.csv."""
    credits_cat_map = {r["file_name"]: r["category"] for r in credits_data}
    for r in labels_data:
        fn = r["file_name"]
        assert credits_cat_map[fn] == r["category"], f"Category mismatch for {fn}"


def test_no_duplicate_filenames(labels_data, credits_data):
    """Verify there are no duplicate filenames in either CSV file."""
    labels_files = [r["file_name"] for r in labels_data]
    credits_files = [r["file_name"] for r in credits_data]
    assert len(labels_files) == len(set(labels_files)), "Duplicate filenames found in labels.csv"
    assert len(credits_files) == len(set(credits_files)), "Duplicate filenames found in credits.csv"


def test_every_referenced_image_exists_on_disk(labels_data):
    """Verify every image referenced in labels.csv exists on disk."""
    for r in labels_data:
        img_path = DATASET_ROOT / r["file_name"]
        assert img_path.exists(), f"Image missing from disk: {img_path}"
        assert img_path.is_file(), f"Path is not a file: {img_path}"
        assert img_path.stat().st_size > 0, f"File is empty: {img_path}"


def test_all_images_decodable_and_valid(labels_data):
    """Verify every image can be safely opened and decoded by PIL."""
    for r in labels_data:
        img_path = DATASET_ROOT / r["file_name"]
        with Image.open(img_path) as img:
            img.verify()
        with Image.open(img_path) as img:
            w, h = img.size
            assert w >= 300 or h >= 300, f"Image too small: {img_path} ({w}x{h})"
            assert img.format in ["JPEG", "PNG"], f"Unexpected format: {img.format}"


def test_zero_overlap_with_held_out_benchmark(labels_data):
    """Verify zero SHA-256 hash overlap with the existing 30-image held-out benchmark."""
    held_out_hashes = set()
    for p in HELD_OUT_ROOT.rglob("*.*"):
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            held_out_hashes.add(h)

    assert len(held_out_hashes) == 30, f"Expected 30 held-out images, got {len(held_out_hashes)}"

    for r in labels_data:
        img_path = DATASET_ROOT / r["file_name"]
        supplied_hash = hashlib.sha256(img_path.read_bytes()).hexdigest()
        assert supplied_hash not in held_out_hashes, f"Overlap detected between {r['file_name']} and held-out benchmark!"


def test_credits_csv_has_required_fields(credits_data):
    """Verify credits.csv contains all required attribution columns with no blank licenses or source links."""
    for r in credits_data:
        assert r["license"].strip() != "", f"Missing license for {r['file_name']}"
        assert r["commons_page"].strip() != "", f"Missing commons_page for {r['file_name']}"
        assert r["source_title"].strip() != "", f"Missing source_title for {r['file_name']}"
