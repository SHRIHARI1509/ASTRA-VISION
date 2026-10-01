import io
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services.classification_service import ClassificationService, ClassificationError
from app.services.inference_service import InferenceService, ModelUnavailableError, InferenceExecutionError
from app.schemas.classification import ClassificationResponse
from app.main import app

client = TestClient(app)


def create_test_image(format: str = "JPEG", size=(100, 100), color="blue") -> bytes:
    buf = io.BytesIO()
    mode = "RGB" if format.upper() in ["JPEG", "JPG"] else "RGBA"
    img = Image.new(mode, size, color=color)
    fmt = "JPEG" if format.upper() in ["JPEG", "JPG"] else format.upper()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ============================================================================
# CLASSIFICATION SERVICE UNIT TESTS
# ============================================================================

def test_highest_score_becomes_primary_prediction():
    """Verify that the candidate with the highest score is selected as primary prediction."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    mock_inference_svc.run_inference.return_value = {
        "model": "google/siglip2-base-patch16-512",
        "device": "cpu",
        "predictions": [
            {"label": "Helicopter", "score": 0.15},
            {"label": "Fighter Aircraft", "score": 0.82},
            {"label": "Tank", "score": 0.02},
        ],
        "inference_time_ms": 120.5,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    result = service.classify_image(file_bytes=b"dummy_bytes", filename="test.jpg")

    assert result["prediction"]["label"] == "Fighter Aircraft"
    assert result["prediction"]["score"] == 0.82


def test_multiple_candidates_ranked_correctly():
    """Verify that all candidate categories are ranked in strict descending order of score."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    mock_inference_svc.run_inference.return_value = {
        "model": "google/siglip2-base-patch16-512",
        "device": "cpu",
        "predictions": [
            {"label": "Drone", "score": 0.05},
            {"label": "Ship", "score": 0.10},
            {"label": "Military Vehicle", "score": 0.35},
            {"label": "Fighter Aircraft", "score": 0.40},
            {"label": "Tank", "score": 0.08},
            {"label": "Helicopter", "score": 0.02},
        ],
        "inference_time_ms": 200.0,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    result = service.classify_image(file_bytes=b"dummy_bytes", filename="test.jpg")

    scores = [c["score"] for c in result["candidates"]]
    assert scores == sorted(scores, reverse=True)
    assert result["candidates"][0]["label"] == "Fighter Aircraft"
    assert result["candidates"][1]["label"] == "Military Vehicle"


def test_candidate_ordering_is_deterministic():
    """Verify that identical candidate inputs produce identical ranking across multiple runs."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    raw_data = {
        "model": "google/siglip2-base-patch16-512",
        "device": "cpu",
        "predictions": [
            {"label": "Tank", "score": 0.22},
            {"label": "Ship", "score": 0.65},
            {"label": "Drone", "score": 0.13},
        ],
        "inference_time_ms": 150.0,
        "status": "success",
    }
    mock_inference_svc.run_inference.return_value = raw_data

    service = ClassificationService(inference_svc=mock_inference_svc)

    result_1 = service.classify_image(file_bytes=b"dummy_bytes", filename="test.jpg")
    result_2 = service.classify_image(file_bytes=b"dummy_bytes", filename="test.jpg")

    assert result_1["candidates"] == result_2["candidates"]
    assert result_1["prediction"] == result_2["prediction"]


def test_tie_handling_is_deterministic():
    """Verify that tied scores are deterministically resolved by taxonomy order."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    # Fighter Aircraft comes before Helicopter in settings.CANDIDATE_CATEGORIES
    mock_inference_svc.run_inference.return_value = {
        "model": "google/siglip2-base-patch16-512",
        "device": "cpu",
        "predictions": [
            {"label": "Helicopter", "score": 0.50},
            {"label": "Fighter Aircraft", "score": 0.50},
        ],
        "inference_time_ms": 100.0,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    result = service.classify_image(file_bytes=b"dummy_bytes", filename="test.jpg")

    # Fighter Aircraft index (0) < Helicopter index (1) in settings.CANDIDATE_CATEGORIES
    assert result["candidates"][0]["label"] == "Fighter Aircraft"
    assert result["candidates"][1]["label"] == "Helicopter"
    assert result["prediction"]["label"] == "Fighter Aircraft"


def test_correct_label_score_pairing_preserved():
    """Verify each candidate preserves its exact original score without cross-contamination."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    orig_pairs = {
        "Fighter Aircraft": 0.7123,
        "Helicopter": 0.1456,
        "Tank": 0.0891,
        "Ship": 0.0312,
        "Military Vehicle": 0.0154,
        "Drone": 0.0064,
    }
    mock_inference_svc.run_inference.return_value = {
        "model": "google/siglip2-base-patch16-512",
        "device": "cpu",
        "predictions": [{"label": k, "score": v} for k, v in orig_pairs.items()],
        "inference_time_ms": 180.0,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    result = service.classify_image(file_bytes=b"dummy_bytes", filename="test.jpg")

    result_dict = {c["label"]: c["score"] for c in result["candidates"]}
    for label, score in orig_pairs.items():
        assert result_dict[label] == round(score, 4)


def test_classification_result_schema_correct():
    """Verify classification output conforms strictly to ClassificationResponse schema."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    mock_inference_svc.run_inference.return_value = {
        "model": "google/siglip2-base-patch16-512",
        "device": "cpu",
        "predictions": [
            {"label": "Fighter Aircraft", "score": 0.91},
            {"label": "Drone", "score": 0.09},
        ],
        "inference_time_ms": 384.0,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    result = service.classify_image(file_bytes=b"dummy_bytes", filename="test.jpg")

    # Pydantic schema validation
    validated = ClassificationResponse(**result)
    assert validated.prediction.label == "Fighter Aircraft"
    assert validated.prediction.score == 0.91
    assert validated.inference.model == "google/siglip2-base-patch16-512"
    assert validated.inference.device == "cpu"
    assert validated.inference.inference_time_ms == 384.0


def test_model_inference_metadata_preserved():
    """Verify model ID, device, and execution timing are correctly preserved."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    mock_inference_svc.run_inference.return_value = {
        "model": "google/siglip2-base-patch16-512",
        "device": "cuda",
        "predictions": [{"label": "Tank", "score": 0.95}],
        "inference_time_ms": 42.75,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    result = service.classify_image(file_bytes=b"dummy_bytes", filename="test.jpg")

    assert result["inference"]["model"] == "google/siglip2-base-patch16-512"
    assert result["inference"]["device"] == "cuda"
    assert result["inference"]["inference_time_ms"] == 42.75


def test_invalid_inference_result_handled_safely():
    """Verify malformed inference result raises ClassificationError."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    mock_inference_svc.run_inference.return_value = {"predictions": "not-a-list"}

    service = ClassificationService(inference_svc=mock_inference_svc)
    with pytest.raises(ClassificationError) as exc_info:
        service.classify_image(file_bytes=b"dummy_bytes", filename="test.jpg")

    assert exc_info.value.code == "INVALID_INFERENCE_RESULT"


def test_empty_candidate_results_handled_safely():
    """Verify empty candidate predictions raise ClassificationError."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    mock_inference_svc.run_inference.return_value = {"predictions": []}

    service = ClassificationService(inference_svc=mock_inference_svc)
    with pytest.raises(ClassificationError) as exc_info:
        service.classify_image(file_bytes=b"dummy_bytes", filename="test.jpg")

    assert exc_info.value.code == "INVALID_INFERENCE_RESULT"


def test_model_inference_failure_propagates_controlled_error():
    """Verify underlying model failures propagate without generating fake predictions."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    mock_inference_svc.run_inference.side_effect = ModelUnavailableError("Weights missing")

    service = ClassificationService(inference_svc=mock_inference_svc)
    with pytest.raises(ModelUnavailableError) as exc_info:
        service.classify_image(file_bytes=b"dummy_bytes", filename="test.jpg")

    assert exc_info.value.code == "MODEL_UNAVAILABLE"


# ============================================================================
# API ENDPOINT UNIT TESTS (POST /api/classify)
# ============================================================================

def test_api_classify_missing_image():
    """Verify POST /api/classify without image returns 400 MISSING_FILE."""
    response = client.post("/api/classify")
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "MISSING_FILE"


def test_api_classify_empty_image():
    """Verify POST /api/classify with empty image returns 400 EMPTY_FILE."""
    response = client.post(
        "/api/classify",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "EMPTY_FILE"


def test_api_classify_corrupted_image():
    """Verify POST /api/classify with corrupted image returns 400 error."""
    corrupt_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01junk_payload"
    response = client.post(
        "/api/classify",
        files={"image": ("corrupted.jpg", corrupt_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] in ["CORRUPTED_IMAGE", "UNIDENTIFIED_IMAGE"]


def test_api_classify_model_unavailable_returns_503():
    """Verify POST /api/classify when model is offline returns 503."""
    img_bytes = create_test_image("JPEG", size=(100, 100))
    with patch("app.api.routes.classification_service.classify_image", side_effect=ModelUnavailableError("Model offline")):
        response = client.post(
            "/api/classify",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "MODEL_UNAVAILABLE"


def test_api_classify_successful_contract():
    """Verify POST /api/classify returns 200 with required Phase 4 contract."""
    img_bytes = create_test_image("JPEG", size=(200, 200))
    mock_classification = {
        "prediction": {"label": "Fighter Aircraft", "score": 0.8841},
        "candidates": [
            {"label": "Fighter Aircraft", "score": 0.8841},
            {"label": "Helicopter", "score": 0.0812},
            {"label": "Drone", "score": 0.0210},
            {"label": "Military Vehicle", "score": 0.0085},
            {"label": "Tank", "score": 0.0032},
            {"label": "Ship", "score": 0.0020},
        ],
        "inference": {
            "model": "google/siglip2-base-patch16-512",
            "device": "cpu",
            "inference_time_ms": 310.5,
        },
        "status": "success",
    }

    with patch("app.api.routes.classification_service.classify_image", return_value=mock_classification):
        response = client.post(
            "/api/classify",
            files={"image": ("recon.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"]["label"] == "Fighter Aircraft"
        assert data["prediction"]["score"] == 0.8841
        assert len(data["candidates"]) == 6
        assert data["inference"]["model"] == "google/siglip2-base-patch16-512"
        assert data["inference"]["device"] == "cpu"
        assert data["status"] == "success"
