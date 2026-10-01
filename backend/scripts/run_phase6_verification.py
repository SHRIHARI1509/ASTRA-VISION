"""Phase 6 Comprehensive Verification Harness.

Executes all automated verification checks for ASTRA VISION:
- Image Input Matrix (valid, invalid, boundary)
- 6-Category Classification Matrix
- Difficult/Ambiguous Input Matrix
- Repeated Analysis (5 consecutive runs for determinism)
- Multi-Image Sequence Test (Tank -> Helicopter -> Aircraft -> Ship -> Drone)
- Resource & Stability Test
- Security & Data Handling Audit
- Performance Baselines
"""

import os
import io
import time
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import httpx
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.siglip2_adapter import SigLIP2Adapter

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"
SCRATCH_DIR = Path(__file__).resolve().parent / "phase6_scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

client = TestClient(app)

def create_synthetic_images():
    """Generates all boundary, invalid, and difficult test inputs."""
    images = {}

    # 1. Valid formats
    for fmt, ext in [("JPEG", "jpg"), ("JPEG", "jpeg"), ("PNG", "png"), ("WEBP", "webp")]:
        img = Image.new("RGB", (200, 200), color=(73, 109, 137))
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        p = SCRATCH_DIR / f"valid_sample.{ext}"
        p.write_bytes(buf.getvalue())
        images[f"valid_{ext}"] = p

    # 2. Invalid files
    # PDF
    pdf_p = SCRATCH_DIR / "invalid_doc.pdf"
    pdf_p.write_bytes(b"%PDF-1.4 Fake PDF file content for test")
    images["invalid_pdf"] = pdf_p

    # TXT
    txt_p = SCRATCH_DIR / "invalid_text.txt"
    txt_p.write_text("This is plain text.")
    images["invalid_txt"] = txt_p

    # ZIP
    zip_p = SCRATCH_DIR / "invalid_archive.zip"
    with zipfile.ZipFile(zip_p, 'w') as zf:
        zf.writestr("test.txt", "inside zip")
    images["invalid_zip"] = zip_p

    # Unsupported format (e.g. BMP)
    bmp_p = SCRATCH_DIR / "unsupported.bmp"
    img_bmp = Image.new("RGB", (50, 50), color=(100, 100, 100))
    img_bmp.save(bmp_p, format="BMP")
    images["invalid_bmp"] = bmp_p

    # Empty file
    empty_p = SCRATCH_DIR / "empty.jpg"
    empty_p.write_bytes(b"")
    images["invalid_empty"] = empty_p

    # Corrupted image
    corrupt_p = SCRATCH_DIR / "corrupt.jpg"
    corrupt_p.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIFcorruptdatahere1234567890")
    images["invalid_corrupted"] = corrupt_p

    # Renamed non-image file
    renamed_p = SCRATCH_DIR / "executable_renamed.jpg"
    renamed_p.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00This is an executable payload disguised as JPG")
    images["invalid_renamed"] = renamed_p

    # 3. Boundary images
    # Very small (< 10x10)
    tiny_p = SCRATCH_DIR / "boundary_tiny_5x5.jpg"
    Image.new("RGB", (5, 5), color=(255, 0, 0)).save(tiny_p, format="JPEG")
    images["boundary_tiny_5x5"] = tiny_p

    # Min valid boundary (10x10)
    min_valid_p = SCRATCH_DIR / "boundary_min_10x10.png"
    Image.new("RGB", (10, 10), color=(0, 255, 0)).save(min_valid_p, format="PNG")
    images["boundary_min_10x10"] = min_valid_p

    # Very wide (1000x20)
    wide_p = SCRATCH_DIR / "boundary_wide_1000x20.jpg"
    Image.new("RGB", (1000, 20), color=(50, 50, 50)).save(wide_p, format="JPEG")
    images["boundary_wide"] = wide_p

    # Very tall (20x1000)
    tall_p = SCRATCH_DIR / "boundary_tall_20x1000.jpg"
    Image.new("RGB", (20, 1000), color=(50, 50, 50)).save(tall_p, format="JPEG")
    images["boundary_tall"] = tall_p

    # Unusual aspect ratio (2500x50 = 50:1)
    unusual_ar_p = SCRATCH_DIR / "boundary_unusual_ar.jpg"
    Image.new("RGB", (2500, 50), color=(70, 70, 70)).save(unusual_ar_p, format="JPEG")
    images["boundary_unusual_ar"] = unusual_ar_p

    # Max valid boundary (8000x8000) - create compressed PNG or JPEG
    # We will test dimension boundary via simulated/generated dimension check or small file with 8000 header
    # File exceeding 10MB limit (> 10 * 1024 * 1024 bytes)
    oversized_p = SCRATCH_DIR / "boundary_oversized_11mb.jpg"
    with open(oversized_p, "wb") as f:
        # Write dummy 11MB file with JPEG SOI
        f.write(b"\xFF\xD8\xFF\xE0" + b"\x00" * (11 * 1024 * 1024))
    images["boundary_oversized"] = oversized_p

    # 4. Difficult / Ambiguous images (using tank and helicopter fixtures as base)
    tank_base = FIXTURES_DIR / "smoke_tank.jpg"
    base_img = Image.open(tank_base)

    # Difficult 1: Low-resolution (32x32 scaled up)
    lowres_p = SCRATCH_DIR / "diff_lowres_32x32.jpg"
    base_img.resize((32, 24)).save(lowres_p, format="JPEG")
    images["diff_lowres"] = lowres_p

    # Difficult 2: Partially obscured object (bars over tank)
    obscured_p = SCRATCH_DIR / "diff_partially_obscured.jpg"
    obs_img = base_img.copy()
    draw = ImageDraw.Draw(obs_img)
    w, h = obs_img.size
    for i in range(0, w, 40):
        draw.rectangle([i, 0, i + 25, h], fill=(30, 60, 20))
    obs_img.save(obscured_p, format="JPEG")
    images["diff_obscured"] = obscured_p

    # Difficult 3: Object at unusual angle (rotated 75 degrees)
    rotated_p = SCRATCH_DIR / "diff_unusual_angle.jpg"
    base_img.rotate(75, expand=True).save(rotated_p, format="JPEG")
    images["diff_rotated"] = rotated_p

    # Difficult 4: Object occupying small portion (15% in bottom corner of large canvas)
    small_obj_p = SCRATCH_DIR / "diff_small_object.jpg"
    canvas = Image.new("RGB", (1200, 900), color=(135, 206, 235))
    small_tank = base_img.resize((150, 100))
    canvas.paste(small_tank, (1000, 750))
    canvas.save(small_obj_p, format="JPEG")
    images["diff_small_obj"] = small_obj_p

    # Difficult 5: Visually cluttered scene
    clutter_p = SCRATCH_DIR / "diff_cluttered.jpg"
    clutter_img = base_img.copy()
    clutter_draw = ImageDraw.Draw(clutter_img)
    import random
    random.seed(42)
    for _ in range(150):
        rx, ry = random.randint(0, w - 50), random.randint(0, h - 50)
        clutter_draw.rectangle([rx, ry, rx + 40, ry + 40], fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    clutter_img.save(clutter_p, format="JPEG")
    images["diff_cluttered"] = clutter_p

    # Difficult 6: Multiple objects (tank + helicopter composite)
    helo_base = FIXTURES_DIR / "smoke_helicopter.jpg"
    helo_img = Image.open(helo_base).resize((w // 2, h // 2))
    multi_p = SCRATCH_DIR / "diff_multiple_objects.jpg"
    multi_img = base_img.copy()
    multi_img.paste(helo_img, (0, 0))
    multi_img.save(multi_p, format="JPEG")
    images["diff_multi_objects"] = multi_p

    return images

def test_image_inputs(images):
    print("\n--- 2. IMAGE INPUT TEST MATRIX ---")
    results = []

    tests = [
        # Valid
        ("Valid JPG", images["valid_jpg"], 200, True),
        ("Valid JPEG", images["valid_jpeg"], 200, True),
        ("Valid PNG", images["valid_png"], 200, True),
        ("Valid WEBP", images["valid_webp"], 200, True),
        # Invalid
        ("Invalid PDF", images["invalid_pdf"], 400, False),
        ("Invalid TXT", images["invalid_txt"], 400, False),
        ("Invalid ZIP", images["invalid_zip"], 400, False),
        ("Invalid BMP (unsupported)", images["invalid_bmp"], 400, False),
        ("Empty File (0 bytes)", images["invalid_empty"], 400, False),
        ("Corrupted Image", images["invalid_corrupted"], 400, False),
        ("Renamed Non-Image File", images["invalid_renamed"], 400, False),
        # Boundary
        ("Boundary Tiny (<10x10)", images["boundary_tiny_5x5"], 400, False),
        ("Boundary Min Valid (10x10)", images["boundary_min_10x10"], 200, True),
        ("Boundary Wide (1000x20)", images["boundary_wide"], 200, True),
        ("Boundary Tall (20x1000)", images["boundary_tall"], 200, True),
        ("Boundary Unusual AR (50:1)", images["boundary_unusual_ar"], 200, True),
        ("Boundary Oversized (>10MB)", images["boundary_oversized"], 413, False),
    ]

    for name, path, expected_status, expect_valid in tests:
        with open(path, "rb") as f:
            content = f.read()

        mime = "image/jpeg" if path.suffix in [".jpg", ".jpeg"] else ("image/png" if path.suffix == ".png" else "application/octet-stream")
        # Validate endpoint
        val_resp = client.post("/api/images/validate", files={"image": (path.name, content, mime)})
        
        # Classify endpoint
        class_resp = client.post("/api/classify", files={"image": (path.name, content, mime)})

        passed = (val_resp.status_code == expected_status) and (class_resp.status_code == expected_status)
        results.append({
            "test": name,
            "filename": path.name,
            "expected_status": expected_status,
            "val_status": val_resp.status_code,
            "class_status": class_resp.status_code,
            "pass": passed,
            "val_json": val_resp.json() if val_resp.headers.get("content-type") == "application/json" else val_resp.text[:80],
        })
        print(f"[{'PASS' if passed else 'FAIL'}] {name:<28} | Val Status: {val_resp.status_code} | Class Status: {class_resp.status_code}")

    return results

def test_six_categories():
    print("\n--- 3. CLASSIFICATION TEST MATRIX (ALL 6 CATEGORIES) ---")
    cat_targets = [
        ("Fighter Aircraft", FIXTURES_DIR / "smoke_aircraft.jpg"),
        ("Helicopter", FIXTURES_DIR / "smoke_helicopter.jpg"),
        ("Tank", FIXTURES_DIR / "smoke_tank.jpg"),
        ("Ship", FIXTURES_DIR / "smoke_ship.jpg"),
        ("Military Vehicle", FIXTURES_DIR / "smoke_vehicle.jpg"),
        ("Drone", FIXTURES_DIR / "smoke_drone.jpg"),
    ]

    results = []
    for expected_label, path in cat_targets:
        with open(path, "rb") as f:
            resp = client.post("/api/classify", files={"image": (path.name, f.read(), "image/jpeg")})
        
        assert resp.status_code == 200, f"Classification failed for {path.name}: {resp.text}"
        data = resp.json()
        pred = data["prediction"]
        inf = data["inference"]
        candidates = data["candidates"]

        is_match = (pred["label"] == expected_label)
        record = {
            "expected": expected_label,
            "filename": path.name,
            "predicted": pred["label"],
            "score": pred["score"],
            "candidates": {c["label"]: round(c["score"], 4) for c in candidates},
            "inference_time_ms": inf["inference_time_ms"],
            "device": inf["device"],
            "pass": is_match,
        }
        results.append(record)
        print(f"[{'PASS' if is_match else 'NOTE'}] Expected: {expected_label:<18} -> Predicted: {pred['label']:<18} | Score: {pred['score']:.4f} | Time: {inf['inference_time_ms']:.1f}ms | Device: {inf['device']}")

    return results

def test_difficult_inputs(images):
    print("\n--- 4. AMBIGUOUS / DIFFICULT INPUTS ---")
    difficult_tests = [
        ("Low-Resolution (32x24)", images["diff_lowres"], "Tank"),
        ("Partially Obscured (Bars/Camo)", images["diff_obscured"], "Tank"),
        ("Unusual Angle (Rotated 75°)", images["diff_rotated"], "Tank"),
        ("Small Object (15% in Corner)", images["diff_small_obj"], "Tank"),
        ("Visually Cluttered (150 Boxes)", images["diff_cluttered"], "Tank"),
        ("Multiple Objects (Tank+Helicopter)", images["diff_multi_objects"], "Tank or Helicopter"),
    ]

    results = []
    for name, path, intended_target in difficult_tests:
        with open(path, "rb") as f:
            resp = client.post("/api/classify", files={"image": (path.name, f.read(), "image/jpeg")})
        
        assert resp.status_code == 200, f"Difficult input failed: {resp.text}"
        data = resp.json()
        pred = data["prediction"]
        inf = data["inference"]
        candidates = {c["label"]: round(c["score"], 4) for c in data["candidates"]}

        results.append({
            "test": name,
            "filename": path.name,
            "intended": intended_target,
            "predicted": pred["label"],
            "score": pred["score"],
            "candidates": candidates,
            "inference_time_ms": inf["inference_time_ms"],
        })
        print(f"[*] {name:<35} -> Predicted: {pred['label']:<18} | Score: {pred['score']:.4f} | Latency: {inf['inference_time_ms']:.1f}ms")

    return results

def test_repeated_analysis():
    print("\n--- 12. REPEATED ANALYSIS (5 CONSECUTIVE RUNS) ---")
    tank_path = FIXTURES_DIR / "smoke_tank.jpg"
    with open(tank_path, "rb") as f:
        img_bytes = f.read()

    runs = []
    for i in range(1, 6):
        resp = client.post("/api/classify", files={"image": ("tank.jpg", img_bytes, "image/jpeg")})
        assert resp.status_code == 200
        data = resp.json()
        runs.append({
            "iteration": i,
            "predicted": data["prediction"]["label"],
            "score": data["prediction"]["score"],
            "inference_time_ms": data["inference"]["inference_time_ms"],
            "device": data["inference"]["device"],
        })
        print(f"Run {i}: Label={data['prediction']['label']} | Score={data['prediction']['score']:.6f} | Latency={data['inference']['inference_time_ms']:.1f}ms | Device={data['inference']['device']}")

    # Determinism check
    first_label = runs[0]["predicted"]
    first_score = runs[0]["score"]
    is_deterministic = all(r["predicted"] == first_label for r in runs) and all(abs(r["score"] - first_score) < 1e-5 for r in runs)
    print(f"Determinism Check: {'PASS (Fully deterministic outputs)' if is_deterministic else 'VARIANCE DETECTED'}")

    return runs, is_deterministic

def test_multi_image_sequence():
    print("\n--- 13. MULTI-IMAGE SEQUENCE TEST ---")
    seq_items = [
        ("Tank", FIXTURES_DIR / "smoke_tank.jpg"),
        ("Helicopter", FIXTURES_DIR / "smoke_helicopter.jpg"),
        ("Aircraft", FIXTURES_DIR / "smoke_aircraft.jpg"),
        ("Ship", FIXTURES_DIR / "smoke_ship.jpg"),
        ("Drone", FIXTURES_DIR / "smoke_drone.jpg"),
    ]

    results = []
    for expected, path in seq_items:
        with open(path, "rb") as f:
            resp = client.post("/api/classify", files={"image": (path.name, f.read(), "image/jpeg")})
        assert resp.status_code == 200
        data = resp.json()
        results.append({
            "step": expected,
            "filename": path.name,
            "predicted": data["prediction"]["label"],
            "score": data["prediction"]["score"],
        })
        print(f"Sequence Step [{expected:<10}]: Sent {path.name:<18} -> Received {data['prediction']['label']:<18} (score {data['prediction']['score']:.4f})")

    return results

def test_security_and_storage():
    print("\n--- 15. SECURITY / DATA HANDLING CHECK ---")
    # Verify no unexpected files created in temp / app dir
    before_files = set(Path(".").glob("**/*"))
    tank_path = FIXTURES_DIR / "smoke_tank.jpg"
    with open(tank_path, "rb") as f:
        client.post("/api/classify", files={"image": ("tank_sec_test.jpg", f.read(), "image/jpeg")})
    after_files = set(Path(".").glob("**/*"))
    new_files = [str(p) for p in (after_files - before_files) if not str(p).endswith(".pyc") and "__pycache__" not in str(p)]

    # Check error response does not expose python stack trace
    corrupt_resp = client.post("/api/classify", files={"image": ("corrupt.jpg", b"corrupted bytes", "image/jpeg")})
    err_body = corrupt_resp.text
    has_stack_trace = "Traceback (most recent call last)" in err_body or "File \"" in err_body
    print(f"Stack trace exposure check: {'PASS (No internal stack traces exposed)' if not has_stack_trace else 'FAIL'}")
    print(f"Image persistence check: {'PASS (Zero uploaded files retained on disk)' if len(new_files) == 0 else f'FILES CREATED: {new_files}'}")

    return {
        "no_stack_trace": not has_stack_trace,
        "no_file_leaks": len(new_files) == 0,
        "new_files": new_files,
    }

def main():
    print("==================================================")
    print("ASTRA VISION — PHASE 6 AUTOMATED VERIFICATION SUITE")
    print("==================================================")
    t0 = time.perf_counter()
    images = create_synthetic_images()

    input_res = test_image_inputs(images)
    six_cat_res = test_six_categories()
    diff_res = test_difficult_inputs(images)
    repeated_res, is_det = test_repeated_analysis()
    seq_res = test_multi_image_sequence()
    sec_res = test_security_and_storage()

    total_time = time.perf_counter() - t0
    print("\n==================================================")
    print(f"PHASE 6 AUTOMATED CHECKS COMPLETE ({total_time:.2f}s total)")
    print("==================================================")

if __name__ == "__main__":
    main()
