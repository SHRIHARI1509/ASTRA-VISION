import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def create_test_image(format: str = "JPEG", size=(100, 100), color="blue") -> bytes:
    """Helper to generate valid image bytes in memory."""
    buf = io.BytesIO()
    # JPEG does not support RGBA; use RGB
    mode = "RGB" if format.upper() in ["JPEG", "JPG"] else "RGBA"
    img = Image.new(mode, size, color=color)
    fmt = "JPEG" if format.upper() in ["JPEG", "JPG"] else format.upper()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ============================================================================
# VALID FORMAT TESTS
# ============================================================================

def test_validate_valid_jpeg():
    img_bytes = create_test_image("JPEG", size=(640, 480))
    response = client.post(
        "/api/images/validate",
        files={"image": ("recon_target.jpeg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["filename"] == "recon_target.jpeg"
    assert data["format"] == "JPEG"
    assert data["width"] == 640
    assert data["height"] == 480
    assert data["size_bytes"] == len(img_bytes)


def test_validate_valid_jpg_extension():
    img_bytes = create_test_image("JPEG", size=(800, 600))
    response = client.post(
        "/api/images/validate",
        files={"image": ("aircraft.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["filename"] == "aircraft.jpg"
    assert data["format"] == "JPEG"


def test_validate_valid_png():
    img_bytes = create_test_image("PNG", size=(320, 240))
    response = client.post(
        "/api/images/validate",
        files={"image": ("satellite.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["format"] == "PNG"
    assert data["width"] == 320
    assert data["height"] == 240


def test_validate_valid_webp():
    img_bytes = create_test_image("WEBP", size=(400, 300))
    response = client.post(
        "/api/images/validate",
        files={"image": ("radar_scan.webp", img_bytes, "image/webp")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["format"] == "WEBP"


# ============================================================================
# INVALID INPUT TESTS
# ============================================================================

def test_validate_missing_file():
    # Calling endpoint without the 'image' field
    response = client.post("/api/images/validate")
    assert response.status_code == 400
    data = response.json()
    assert data["valid"] is False
    assert data["error"]["code"] == "MISSING_FILE"


def test_validate_empty_file():
    response = client.post(
        "/api/images/validate",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["valid"] is False
    assert data["error"]["code"] == "EMPTY_FILE"


def test_validate_text_file():
    response = client.post(
        "/api/images/validate",
        files={"image": ("notes.txt", b"TACTICAL REPORT CONTENT", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["valid"] is False
    assert data["error"]["code"] in ["UNSUPPORTED_MIME_TYPE", "UNIDENTIFIED_IMAGE"]


def test_validate_pdf_file():
    pdf_bytes = b"%PDF-1.4\n%...\n%%EOF"
    response = client.post(
        "/api/images/validate",
        files={"image": ("briefing.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["valid"] is False
    assert data["error"]["code"] in ["UNSUPPORTED_MIME_TYPE", "UNIDENTIFIED_IMAGE"]


def test_validate_unsupported_format_gif():
    # Valid GIF image
    buf = io.BytesIO()
    Image.new("RGB", (50, 50), "red").save(buf, format="GIF")
    gif_bytes = buf.getvalue()

    response = client.post(
        "/api/images/validate",
        files={"image": ("animation.gif", gif_bytes, "image/gif")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["valid"] is False
    assert data["error"]["code"] in ["UNSUPPORTED_FORMAT", "UNSUPPORTED_MIME_TYPE"]


def test_validate_corrupted_image():
    # Valid JPEG header followed by junk
    corrupt_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00corrupted_payload_data_junk"
    response = client.post(
        "/api/images/validate",
        files={"image": ("corrupted.jpg", corrupt_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["valid"] is False
    assert data["error"]["code"] in ["CORRUPTED_IMAGE", "UNIDENTIFIED_IMAGE"]


def test_validate_renamed_non_image_file():
    # A text file renamed with .jpg extension and image/jpeg MIME
    fake_jpg = b"This is just plain text disguised as an image file."
    response = client.post(
        "/api/images/validate",
        files={"image": ("stealth.jpg", fake_jpg, "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["valid"] is False
    assert data["error"]["code"] in ["UNIDENTIFIED_IMAGE", "CORRUPTED_IMAGE"]


# ============================================================================
# BOUNDARY & ROBUSTNESS TESTS
# ============================================================================

def test_validate_oversized_file():
    # Exceeding 10MB limit
    oversized_bytes = b"0" * (10 * 1024 * 1024 + 1024)
    response = client.post(
        "/api/images/validate",
        files={"image": ("oversized.png", oversized_bytes, "image/png")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["valid"] is False
    assert data["error"]["code"] == "FILE_TOO_LARGE"


def test_validate_dimensions_too_small():
    # Smaller than 10x10
    tiny_bytes = create_test_image("PNG", size=(5, 5))
    response = client.post(
        "/api/images/validate",
        files={"image": ("tiny.png", tiny_bytes, "image/png")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["valid"] is False
    assert data["error"]["code"] == "DIMENSIONS_TOO_SMALL"


def test_validate_dimensions_too_large():
    # Exceeding 8000 max dimension
    img_bytes = create_test_image("PNG", size=(8500, 100))
    response = client.post(
        "/api/images/validate",
        files={"image": ("huge.png", img_bytes, "image/png")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["valid"] is False
    assert data["error"]["code"] == "DIMENSIONS_TOO_LARGE"


def test_validate_unusual_aspect_ratio():
    # Aspect ratio > 100 (e.g. 5000 x 20 = 250:1)
    extreme_bytes = create_test_image("PNG", size=(3000, 20))
    response = client.post(
        "/api/images/validate",
        files={"image": ("ribbon.png", extreme_bytes, "image/png")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["valid"] is False
    assert data["error"]["code"] == "UNUSUAL_ASPECT_RATIO"
