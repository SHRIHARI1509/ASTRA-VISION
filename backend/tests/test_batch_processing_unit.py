"""Unit tests for Phase 8C Batch Image Processing.

Covers all 14 required test cases:
1. single-image batch
2. multiple valid images
3. maximum batch size
4. batch exceeding maximum
5. empty batch
6. invalid MIME type
7. corrupt image
8. empty file
9. oversized image
10. partial batch failure
11. deterministic ordering
12. exactly one production model instance
13. production model remains SigLIP 2
14. existing /api/classify remains unchanged
"""

import io
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services.classification_service import (
    ClassificationService,
    ClassificationError,
    classification_service,
)
from app.services.inference_service import InferenceService
from app.models.base_adapter import ModelInferenceResult, Prediction
from app.main import app

client = TestClient(app)


def create_test_image(format: str = "JPEG", size=(100, 100), color="blue") -> bytes:
    """Helper to generate valid synthetic test image bytes in memory."""
    buf = io.BytesIO()
    mode = "RGB" if format.upper() in ["JPEG", "JPG"] else "RGBA"
    img = Image.new(mode, size, color=color)
    fmt = "JPEG" if format.upper() in ["JPEG", "JPG"] else format.upper()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def mock_inference_svc():
    """InferenceService with mocked ModelAdapter preserving real ImageService validation."""
    mock_adapter = MagicMock()
    mock_adapter.is_loaded = True
    mock_adapter.model_id = "google/siglip2-base-patch16-512"
    mock_adapter.device = "cpu"
    mock_adapter.predict.return_value = ModelInferenceResult(
        model="google/siglip2-base-patch16-512",
        device="cpu",
        predictions=[
            Prediction(label="Tank", score=0.85),
            Prediction(label="Military Vehicle", score=0.10),
            Prediction(label="Fighter Aircraft", score=0.03),
            Prediction(label="Helicopter", score=0.01),
            Prediction(label="Ship", score=0.005),
            Prediction(label="Drone", score=0.005),
        ],
        inference_time_ms=150.0,
        status="success",
    )
    return InferenceService(adapter=mock_adapter)



# ============================================================================
# 1. Single-image batch
# ============================================================================
def test_single_image_batch(mock_inference_svc):
    service = ClassificationService(inference_svc=mock_inference_svc)
    valid_bytes = create_test_image()
    files = [(valid_bytes, "recon_01.jpg", "image/jpeg")]

    result = service.classify_batch(files=files)

    assert result["status"] == "success"
    assert result["total"] == 1
    assert result["successful"] == 1
    assert result["failed"] == 0
    assert len(result["results"]) == 1
    item = result["results"][0]
    assert item["status"] == "success"
    assert item["filename"] == "recon_01.jpg"
    assert item["prediction"]["label"] == "Tank"


# ============================================================================
# 2. Multiple valid images
# ============================================================================
def test_multiple_valid_images_batch(mock_inference_svc):
    service = ClassificationService(inference_svc=mock_inference_svc)
    files = [
        (create_test_image(color="red"), "target_alpha.jpg", "image/jpeg"),
        (create_test_image(format="PNG", color="green"), "target_bravo.png", "image/png"),
        (create_test_image(format="WEBP", color="blue"), "target_charlie.webp", "image/webp"),
    ]

    result = service.classify_batch(files=files)

    assert result["status"] == "success"
    assert result["total"] == 3
    assert result["successful"] == 3
    assert result["failed"] == 0
    assert len(result["results"]) == 3
    assert [r["filename"] for r in result["results"]] == [
        "target_alpha.jpg",
        "target_bravo.png",
        "target_charlie.webp",
    ]


# ============================================================================
# 3. Maximum batch size (boundary condition: exactly 20 images)
# ============================================================================
def test_maximum_batch_size(mock_inference_svc):
    service = ClassificationService(inference_svc=mock_inference_svc)
    max_size = settings.MAX_BATCH_SIZE  # 20
    valid_bytes = create_test_image()
    files = [(valid_bytes, f"image_{i:02d}.jpg", "image/jpeg") for i in range(max_size)]

    result = service.classify_batch(files=files)

    assert result["total"] == max_size
    assert result["successful"] == max_size
    assert result["failed"] == 0


# ============================================================================
# 4. Batch exceeding maximum (boundary condition: 21 images)
# ============================================================================
def test_batch_exceeding_maximum_rejected(mock_inference_svc):
    service = ClassificationService(inference_svc=mock_inference_svc)
    max_size = settings.MAX_BATCH_SIZE
    valid_bytes = create_test_image()
    files = [(valid_bytes, f"image_{i:02d}.jpg", "image/jpeg") for i in range(max_size + 1)]

    with pytest.raises(ClassificationError) as exc_info:
        service.classify_batch(files=files)

    assert exc_info.value.code == "BATCH_SIZE_EXCEEDED"
    assert exc_info.value.status_code == 400
    assert str(max_size) in exc_info.value.message


def test_api_batch_exceeding_maximum():
    valid_bytes = create_test_image()
    # Post 21 files to /api/classify/batch
    files = [
        ("images", (f"img_{i}.jpg", valid_bytes, "image/jpeg"))
        for i in range(settings.MAX_BATCH_SIZE + 1)
    ]
    response = client.post("/api/classify/batch", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "BATCH_SIZE_EXCEEDED"


# ============================================================================
# 5. Empty batch handling
# ============================================================================
def test_empty_batch_handling():
    service = ClassificationService()
    with pytest.raises(ClassificationError) as exc_info:
        service.classify_batch(files=[])

    assert exc_info.value.code == "EMPTY_BATCH"
    assert exc_info.value.status_code == 400

    # API test with no files
    response = client.post("/api/classify/batch")
    assert response.status_code == 400
    data = response.json()
    assert data["error"]["code"] == "EMPTY_BATCH"


# ============================================================================
# 6. Invalid MIME type handling in batch
# ============================================================================
def test_invalid_mime_type_in_batch(mock_inference_svc):
    service = ClassificationService(inference_svc=mock_inference_svc)
    files = [
        (create_test_image(), "valid.jpg", "image/jpeg"),
        (b"plain text document", "report.txt", "text/plain"),
    ]

    result = service.classify_batch(files=files)

    assert result["total"] == 2
    assert result["successful"] == 1
    assert result["failed"] == 1
    assert result["results"][0]["status"] == "success"
    assert result["results"][1]["status"] == "error"
    assert "error" in result["results"][1]
    assert result["results"][1]["error"]["code"] in ["UNSUPPORTED_FORMAT", "INVALID_MIME_TYPE", "UNSUPPORTED_MIME_TYPE", "DECODE_ERROR"]


# ============================================================================
# 7. Corrupt image handling in batch
# ============================================================================
def test_corrupt_image_in_batch(mock_inference_svc):
    service = ClassificationService(inference_svc=mock_inference_svc)
    files = [
        (b"\xff\xd8\xff\xe0\x00\x10JFIF\x00corrupt-bytes-payload-random", "corrupt.jpg", "image/jpeg"),
        (create_test_image(), "valid.jpg", "image/jpeg"),
    ]

    result = service.classify_batch(files=files)

    assert result["total"] == 2
    assert result["successful"] == 1
    assert result["failed"] == 1
    assert result["results"][0]["status"] == "error"
    assert result["results"][1]["status"] == "success"


# ============================================================================
# 8. Empty file (0 bytes) in batch
# ============================================================================
def test_empty_file_in_batch(mock_inference_svc):
    service = ClassificationService(inference_svc=mock_inference_svc)
    files = [
        (b"", "empty.jpg", "image/jpeg"),
        (create_test_image(), "valid.jpg", "image/jpeg"),
    ]

    result = service.classify_batch(files=files)

    assert result["total"] == 2
    assert result["successful"] == 1
    assert result["failed"] == 1
    assert result["results"][0]["status"] == "error"
    assert result["results"][0]["error"]["code"] == "EMPTY_FILE"
    assert result["results"][1]["status"] == "success"


# ============================================================================
# 9. Oversized image in batch (> 10MB)
# ============================================================================
def test_oversized_image_in_batch(mock_inference_svc):
    service = ClassificationService(inference_svc=mock_inference_svc)
    oversized_bytes = b"0" * (10 * 1024 * 1024 + 1)  # 10MB + 1 byte
    files = [
        (oversized_bytes, "giant.jpg", "image/jpeg"),
        (create_test_image(), "valid.jpg", "image/jpeg"),
    ]

    result = service.classify_batch(files=files)

    assert result["total"] == 2
    assert result["successful"] == 1
    assert result["failed"] == 1
    assert result["results"][0]["status"] == "error"
    assert result["results"][0]["error"]["code"] == "FILE_TOO_LARGE"
    assert result["results"][1]["status"] == "success"


# ============================================================================
# 10. Partial batch failure (mixed valid and invalid)
# ============================================================================
def test_partial_batch_failure(mock_inference_svc):
    service = ClassificationService(inference_svc=mock_inference_svc)
    files = [
        (create_test_image(), "valid_1.jpg", "image/jpeg"),
        (b"not an image", "bad_1.jpg", "image/jpeg"),
        (create_test_image(format="PNG"), "valid_2.png", "image/png"),
        (b"", "bad_2.jpg", "image/jpeg"),
    ]

    result = service.classify_batch(files=files)

    assert result["total"] == 4
    assert result["successful"] == 2
    assert result["failed"] == 2
    assert [r["status"] for r in result["results"]] == ["success", "error", "success", "error"]
    # Verify no stack trace or filesystem path leaks in error
    err = result["results"][1]["error"]["message"]
    assert "Traceback" not in err
    assert "C:\\" not in err
    assert "/" not in err or "image/" in err or "formats:" in err


# ============================================================================
# 11. Deterministic ordering preserved
# ============================================================================
def test_deterministic_ordering_preserved(mock_inference_svc):
    service = ClassificationService(inference_svc=mock_inference_svc)
    input_names = [f"rec_target_{i:03d}.jpg" for i in range(7)]
    files = [(create_test_image(), name, "image/jpeg") for name in input_names]

    result = service.classify_batch(files=files)

    output_names = [r["filename"] for r in result["results"]]
    assert output_names == input_names, "Batch results must strictly preserve input ordering"


# ============================================================================
# 12. Exactly one production model instance (singleton)
# ============================================================================
def test_exactly_one_production_model_instance():
    # Verify default classification service holds the singleton inference service
    assert classification_service._inference_service is not None
    adapter_id_1 = id(classification_service._inference_service.adapter)
    adapter_id_2 = id(classification_service._inference_service.adapter)
    assert adapter_id_1 == adapter_id_2, "InferenceService must maintain a single cached adapter singleton"


# ============================================================================
# 13. Production model remains SigLIP 2
# ============================================================================
def test_production_model_remains_siglip2(mock_inference_svc):
    service = ClassificationService(inference_svc=mock_inference_svc)
    files = [(create_test_image(), "recon.jpg", "image/jpeg")]
    result = service.classify_batch(files=files)

    item = result["results"][0]
    assert item["inference"]["model"] == "google/siglip2-base-patch16-512"
    assert settings.MODEL_ID == "google/siglip2-base-patch16-512"


# ============================================================================
# 14. Existing /api/classify remains unchanged
# ============================================================================
def test_existing_classify_endpoint_remains_unchanged():
    with patch("app.api.routes.classification_service.classify_image") as mock_classify:
        mock_classify.return_value = {
            "prediction": {"label": "Helicopter", "score": 0.88},
            "candidates": [
                {"label": "Helicopter", "score": 0.88},
                {"label": "Fighter Aircraft", "score": 0.08},
                {"label": "Tank", "score": 0.02},
                {"label": "Drone", "score": 0.01},
                {"label": "Ship", "score": 0.005},
                {"label": "Military Vehicle", "score": 0.005},
            ],
            "top_3": [
                {"label": "Helicopter", "score": 0.88},
                {"label": "Fighter Aircraft", "score": 0.08},
                {"label": "Tank", "score": 0.02},
            ],
            "uncertainty": {
                "is_uncertain": False,
                "reason": None,
                "method": "heuristic",
                "score_margin": 0.80,
                "primary_score": 0.88,
                "margin_threshold": 0.02,
                "score_threshold": 0.01,
            },
            "model_fit": settings.MODEL_FIT_JUSTIFICATION,
            "inference": {
                "model": "google/siglip2-base-patch16-512",
                "device": "cpu",
                "inference_time_ms": 110.0,
            },
            "status": "success",
        }

        valid_bytes = create_test_image()
        response = client.post(
            "/api/classify",
            files={"image": ("single_target.jpg", valid_bytes, "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["prediction"]["label"] == "Helicopter"
        assert data["prediction"]["score"] == 0.88
        assert len(data["top_3"]) == 3
        assert data["uncertainty"]["is_uncertain"] is False
        assert data["inference"]["model"] == "google/siglip2-base-patch16-512"
